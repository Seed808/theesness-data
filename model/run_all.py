"""
Build every board, lock new projections, grade old ones, and write the site.

    python run_all.py --data ../data_raw --nfldata ../nflverse/nfldata/data --out site

Data folders expected (all public; see README "Data requirements"):
    data_raw/nfl/play_by_play_{2024,2025,2026}.parquet      nflverse
    data_raw/ncaaf/play_by_play_{2025,2026}.parquet          cfbfastR (sportsdataverse)
    data_raw/nba/play_by_play_{2025,2026}.parquet            ESPN via sportsdataverse
    data_raw/cbb/play_by_play_2026.parquet                   ESPN via sportsdataverse
    nfldata/games.csv, teams.csv                             nflverse schedule + current lines
MLB and NHL run on SYNTHETIC test data until Statcast / NHL API pulls are configured.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from edgemodels import site, alerts
from edgemodels.core.edges import EDGE_TIERS
from edgemodels.core.ledger import Ledger, record_summary
from edgemodels.core.margins import MarginDist, nfl_from_nflverse
from edgemodels.props.basketball_props import BasketballProps, first_basket_table, player_games
from edgemodels.props.football_props import FootballProps
from edgemodels.sports import basketball as bb
from edgemodels.sports import football as fb
from edgemodels.sports import nfl_injuries
from edgemodels.sports import cfb_elo
from edgemodels.sports.basketball_pipeline import BasketballModel
from run_mlb import mlb_section
from edgemodels.core.espn_feed import first_line_events, latest_events, latest_line_events
from edgemodels.core import pickbook, totals_tracker
from edgemodels.sports.nfl_starters import Starters
from edgemodels.sports.football_pipeline import FootballModel, nfl_slate, slate_from_plays


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def bt_summary(bt: pd.DataFrame, tiers):
    g = bt.dropna(subset=["market_line"]).copy()
    g = g[g["result"].isin(["WIN", "LOSS", "PUSH"])]
    return record_summary(g, tiers), g


def accuracy(bt):
    g = bt.dropna(subset=["market_line"])
    out = {"games": int(len(bt)), "model_mae": float((bt["margin"] - bt["proj_margin"]).abs().mean()),
           "model_sd": float((bt["margin"] - bt["proj_margin"]).std())}
    if len(g):
        out.update(market_mae=float((g["margin"] + g["market_line"]).abs().mean()),
                   market_sd=float((g["margin"] + g["market_line"]).std()), lined=int(len(g)))
    return out


def grade_cards(cards, finals: dict):
    """Attach final score and ATS result (vs the line on the card) to replayed cards."""
    for c in cards:
        fs = finals.get(str(c["game_id"]))
        if not fs or c.get("market_line") is None or any(pd.isna(v) for v in fs):
            c["result"] = None
            continue
        a, h = fs
        c["final_away"], c["final_home"] = float(a), float(h)
        m = (h - a) if c["pick"] == "home" else (a - h)
        cov = m + c["pick_line"]
        c["result"] = "WIN" if cov > 0 else ("LOSS" if cov < 0 else "PUSH")
    return cards


def cards_to_rows(cards, week):
    return [{"week": week, "away": c["away"], "home": c["home"], "pick": c.get("pick"), "pick_line": c.get("pick_line"),
             "edge": c.get("edge"), "proj_away": c["proj_away"], "proj_home": c["proj_home"],
             "away_score": c.get("final_away"), "home_score": c.get("final_home"), "result": c.get("result")}
            for c in cards if c.get("market_line") is not None]



def nfl_props(c, P, AV):
    """Prop card for one NFL game, using the injury report for that game's week."""
    wk = int(str(c["game_id"])[5:7])
    avail = AV.for_game(wk, (c["home"], c["away"]))
    pr = P.game_props(c, availability=avail)
    outs = {t: AV.out_list(wk, t) for t in (c["away"], c["home"])} if AV.published(wk) else {}
    pr["n_sims"] = 20000
    pr["note"] = ("No prop lines are loaded, so each prop shows the model's distribution and P(over) at alternate lines. "
                  "TD probabilities: team offensive TDs "
                  f"(projected points x {P.td_per_point:.3f} TDs per point) split by each player's touches and red-zone "
                  "opportunities, with injured players' shares given to available teammates. "
                  + (("Out: " + "; ".join(f"{t}: {', '.join(v) or 'none'}" for t, v in outs.items()) + ". Questionable "
                      f"players count at {100 * AV.q:.0f}%, the share of Questionable players who were active this season.")
                     if outs else f"The Week {wk} injury report is not out yet (Wednesday to Friday); IR, cuts and trades "
                     "are applied and players must have played in one of the last two games."))
    pr["players"] = [p for p in pr["players"] if p["anytime_td"]["p_yes"] > 0.12][:10]
    return pr


def nfl_actuals(raw: pd.DataFrame) -> dict:
    """Box-score results per game and player from nflverse play-by-play (two-point tries excluded)."""
    out = {}
    raw = raw[raw["two_point_attempt"].fillna(0) == 0]
    for gid, g in raw.groupby("game_id"):
        rc = g[g["complete_pass"] == 1]
        ru = g[g["rush_attempt"] == 1]
        td = g[g["td_player_id"].notna() & (g["touchdown"] == 1)].sort_values(["qtr", "game_seconds_remaining"],
                                                                                 ascending=[True, False])
        d = {"__first_td__": td["td_player_id"].iloc[0] if len(td) else None,
             "__first_td_name__": td["td_player_name"].iloc[0] if len(td) else None,
             "__final__": True}
        ids = set(rc["receiver_player_id"].dropna()) | set(ru["rusher_player_id"].dropna()) | set(td["td_player_id"])
        for pid in ids:
            d[pid] = {"receptions": int((rc["receiver_player_id"] == pid).sum()),
                      "receiving_yards": float(rc.loc[rc["receiver_player_id"] == pid, "receiving_yards"].sum()),
                      "rushing_yards": float(ru.loc[ru["rusher_player_id"] == pid, "rushing_yards"].sum()),
                      "tds": int((td["td_player_id"] == pid).sum())}
        out[str(gid)] = d
    return out


def attach_actuals(c, actuals):
    """Put each listed player's actual line on the prop card once the game is final."""
    a = actuals.get(str(c["game_id"]))
    pr = c.get("props")
    if not a or not pr:
        return
    pr["final"] = True
    pr["first_td_actual"] = a["__first_td_name__"]
    for p in pr["players"]:
        r = a.get(p["player_id"], {"receptions": 0, "receiving_yards": 0.0, "rushing_yards": 0.0, "tds": 0})
        p["actual"] = dict(r, first_td=p["player_id"] == a["__first_td__"])


ESPN_ABBR = {"LAR": "LA", "WSH": "WAS"}


def espn_first_lines(repo: Path, league: str, dates) -> dict:
    """{(away, home): (spread_home, total)} from the first ESPN snapshot saved for each date (theesness-data)."""
    out = {}
    for d in dates:
        folder = repo / "data" / "espn" / league / "scoreboard" / d.strftime("%Y%m%d")
        if not folder.exists():
            continue
        for ev, _seen in first_line_events(folder):
            c = ev["competitions"][0]
            teams = {t["homeAway"]: ESPN_ABBR.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in c["competitors"]}
            o = (c.get("odds") or [{}])[0]
            det, tot = o.get("details"), o.get("overUnder")
            if not det or det == "EVEN":
                sh = 0.0 if det == "EVEN" else None
            else:
                fav, num = det.rsplit(" ", 1)
                fav = ESPN_ABBR.get(fav, fav)
                sh = float(num) if fav == teams.get("home") else -float(num)
            out[(teams.get("away"), teams.get("home"))] = (sh, float(tot) if tot is not None else None)
    return out

def first_lock_date(*paths) -> str:
    """Hawaii date of the earliest lock across lock files (the live record's start), fixed once written."""
    ts = []
    for p in paths:
        if Path(p).exists():
            ts += list(pd.to_datetime(pd.read_csv(p)["locked_at"], utc=True))
    return (min(ts) - pd.Timedelta(hours=10)).strftime("%B %-d, %Y") if ts else "—"




NCAAF_MODEL = "v2 · live from Oct 16, 2026"
NO_EDGE = 0.5   # points: a model within half a point of the line takes no side (shown as NO EDGE, not graded)


def no_edge_map(cards):
    return {c["game_id"]: (f'model line {c["model_line"]:+.1f} vs market {c["market_line"]:+.1f} (home team), '
                           f'{c["edge"]:.1f} points apart, under {NO_EDGE}')
            for c in cards if c.get("market_line") is not None and c.get("edge") is not None and c["edge"] < NO_EDGE}


def espn_home_spread(c: dict, o: dict, abbr_map: dict | None = None):
    """Home team's spread from an ESPN competition + odds entry. ESPN's text ("BUF -6.5") sometimes uses a
    different abbreviation than the team record ("BUFF"; also "AF" vs "AFA"), which flipped two college lines on
    2026-10-16. The odds' own favourite flags decide first; the text abbreviation is only a fallback."""
    det = o.get("details")
    if not det:
        return None
    if det == "EVEN":
        return 0.0
    try:
        fav, num = det.rsplit(" ", 1)
        n = abs(float(num))
    except ValueError:
        return None
    hf = (o.get("homeTeamOdds") or {}).get("favorite")
    af = (o.get("awayTeamOdds") or {}).get("favorite")
    if hf is True and af is not True:
        return -n
    if af is True and hf is not True:
        return n
    m = abbr_map or {}
    ab = {x["homeAway"]: m.get(x["team"].get("abbreviation"), x["team"].get("abbreviation")) for x in c["competitors"]}
    f = m.get(fav, fav)
    if f == ab.get("home"):
        return -n
    if f == ab.get("away"):
        return n
    return None                      # cannot tell who is favoured: no line rather than a guess

def nfl_current_week(games: pd.DataFrame, now, season: int = 2026) -> int:
    """The live NFL week: the earliest week that still has a game not yet over (kickoff + 4.5 h in the future).
    Rolls to the next week on its own once the last game of the week (usually Monday night) is done."""
    g = games[games["season"] == season].copy()
    ko = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str), errors="coerce")
    ko = ko.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    pending = g[(ko + pd.Timedelta(hours=4.5)) > pd.Timestamp(now)]
    return int(pending["week"].min()) if len(pending) else int(g["week"].max())


def espn_ncaaf_events(repo: Path) -> dict:
    """{event id: {week, kickoff_utc, completed, final (away, home)}} from the newest snapshot of every ESPN
    college scoreboard folder."""
    out = {}
    base = repo / "data" / "espn" / "ncaaf" / "scoreboard"
    for folder in sorted(base.glob("2*")):
        if not folder.is_dir():
            continue
        for ev in latest_events(folder):
            c = ev["competitions"][0]
            done = bool(c.get("status", {}).get("type", {}).get("completed"))
            fin = None
            if done:
                sc = {t["homeAway"]: float(t["score"]) for t in c["competitors"]}
                fin = (sc["away"], sc["home"])
            prev = out.get(str(ev["id"]), {})
            out[str(ev["id"])] = {"week": (ev.get("week") or {}).get("number"), "kickoff_utc": ev.get("date"),
                                  "completed": done or prev.get("completed", False), "final": fin or prev.get("final")}
    return out


def ncaaf_current_week(events: dict, now, default: int) -> int:
    """The live college week: the earliest week with a game not yet over (kickoff + 5 h in the future).
    A postponed game whose time has passed does not hold the board back."""
    weeks = [v["week"] for v in events.values() if v["week"] and not v["completed"] and v["kickoff_utc"]
             and pd.Timestamp(v["kickoff_utc"]) + pd.Timedelta(hours=5) > pd.Timestamp(now)]
    return int(min(weeks)) if weeks else default


def espn_nba_events(repo: Path) -> dict:
    """{event id: {...}} for every NBA game in the saved ESPN scoreboards: teams, names, kickoff, finished score,
    and the newest pre-game line."""
    out = {}
    base = repo / "data" / "espn" / "nba" / "scoreboard"
    for folder in sorted(base.glob("2*")):
        if not folder.is_dir():
            continue
        for ev in latest_events(folder):
            c = ev["competitions"][0]
            t = {x["homeAway"]: x for x in c["competitors"]}
            done = bool(c.get("status", {}).get("type", {}).get("completed"))
            row = out.setdefault(str(ev["id"]), {})
            row.update(game_id=str(ev["id"]), kickoff_utc=ev["date"], away=t["away"]["team"]["abbreviation"],
                       home=t["home"]["team"]["abbreviation"], neutral_site=bool(c.get("neutralSite")),
                       venue=(c.get("venue") or {}).get("fullName", ""), season_type=(ev.get("season") or {}).get("type"),
                       names={t[s]["team"]["abbreviation"]: t[s]["team"].get("displayName") for s in ("away", "home")})
            if done:
                row["final"] = (float(t["away"]["score"]), float(t["home"]["score"]))
        for ev, seen in latest_line_events(folder):
            c = ev["competitions"][0]
            o = (c.get("odds") or [{}])[0]
            row = out.setdefault(str(ev["id"]), {})
            sh = espn_home_spread(c, o)
            if sh is not None:
                row.update(spread_home=sh, total=o.get("overUnder"), line_seen=seen)
    return out


def espn_ncaaf_slate(repo: Path, week: int):
    """(slate, finals) for one college week from the collected ESPN scoreboards: first line seen per game,
    finals from the newest snapshot."""
    rows, finals, seen = [], {}, set()
    base = repo / "data" / "espn" / "ncaaf" / "scoreboard"
    for folder in sorted(base.glob("2*")):
        if not folder.is_dir():
            continue
        for ev in latest_events(folder):
            c = ev["competitions"][0]
            if ev.get("week", {}).get("number") == week and c.get("status", {}).get("type", {}).get("completed"):
                sc = {t["homeAway"]: float(t["score"]) for t in c["competitors"]}
                finals[str(ev["id"])] = (sc["away"], sc["home"])
        for ev, _seen in latest_line_events(folder):      # current line while the game has not started
            if ev.get("week", {}).get("number") != week or ev["id"] in seen:
                continue
            seen.add(ev["id"])
            c = ev["competitions"][0]
            t = {x["homeAway"]: x["team"]["location"] for x in c["competitors"]}
            o = (c.get("odds") or [{}])[0]
            sh = espn_home_spread(c, o)
            ko = pd.Timestamp(ev["date"])
            rows.append({"game_id": str(ev["id"]), "date": (ko - pd.Timedelta(hours=10)).strftime("%A, %B %-d"),
                         "kickoff": (ko - pd.Timedelta(hours=10)).strftime("%-I:%M %p HST"), "kickoff_utc": str(ko),
                         "venue": (c.get("venue") or {}).get("fullName", ""),
                         "away": t["away"], "home": t["home"], "neutral_site": bool(c.get("neutralSite")),
                         "spread_home": sh, "total": o.get("overUnder")})
    sl = pd.DataFrame(rows)
    if not sl.empty:
        sl = sl.sort_values("kickoff_utc", kind="stable").reset_index(drop=True)
    return sl, finals


def live_starters(slate: pd.DataFrame, pbp: pd.DataFrame, injuries: pd.DataFrame, week: int) -> pd.DataFrame:
    """Starting QB for each team: the passer with the most dropbacks in the team's last game, unless this week's
    injury report lists him Out/Doubtful, in which case the schedule file's listed starter is used.
    (The schedule file's listed starter alone was wrong for SEA in Week 6: Lock listed, Darnold playing.)"""
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    inj = injuries[(injuries["week"] == week)]
    out_ids = set(inj.loc[inj["report_status"].isin(["Out", "Doubtful"]), "gsis_id"]) if "gsis_id" in inj else set()
    sl = slate.copy()
    for side in ("away", "home"):
        for i, r in sl.iterrows():
            t = r[side]
            x = d[d["posteam"] == t]
            if x.empty:
                continue
            last = x[x["week"] == x["week"].max()]
            top = last.groupby(["passer_player_id", "passer_player_name"]).size().sort_values().index[-1]
            if top[0] in out_ids:
                continue                                   # last starter ruled out: keep the listed starter
            full = top[1]
            names = pbp.loc[pbp["passer_player_id"] == top[0], "passer"].dropna() if "passer" in pbp else pd.Series([], dtype=str)
            sl.at[i, f"{side}_qb_id"] = top[0]
            sl.at[i, f"{side}_qb_name"] = names.iloc[-1] if len(names) else full
    return sl


ESPN_ABBR_REV = {"LA": "LAR", "WAS": "WSH"}


def espn_lines_nfl(repo: Path, dates) -> dict:
    """{(away, home): (spread_home, total, seen_at_HST)} from the newest ESPN snapshot that has a line."""
    out = {}
    for d in dates:
        folder = repo / "data" / "espn" / "nfl" / "scoreboard" / d.strftime("%Y%m%d")
        if not folder.exists():
            continue
        for ev, seen in latest_line_events(folder):
            c = ev["competitions"][0]
            teams = {t["homeAway"]: ESPN_ABBR.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in c["competitors"]}
            o = (c.get("odds") or [{}])[0]
            det, tot = o.get("details"), o.get("overUnder")
            if not det:
                continue
            sh = espn_home_spread(c, o, ESPN_ABBR)
            if sh is None:
                continue
            when = (pd.Timestamp(seen.replace("Z", ""), tz="UTC") - pd.Timedelta(hours=10)).strftime("%b %-d %-I:%M %p HST") if seen else ""
            out[(teams.get("away"), teams.get("home"))] = (sh, float(tot) if tot is not None else None, when)
    return out

def season_pages(out, sport, label, model, season, weeks, slate_fn, names, margin_sd, live_week_label,
                 game_pages=True, rec=None, locked=None, archive_dir=None, props_fn=None):
    """locked: lock-file rows. A game that was locked live shows its locked numbers (not a replay)
    and is left out of the replay record, so one game never carries two different picks."""
    locked = {} if locked is None or len(locked) == 0 else {str(r["game_id"]): r for r in locked.to_dict("records")}
    """Replay every past week (projected only from games before it), write week pages, return graded rows."""
    rows, tabs = [], []
    for w in weeks:
        sl = slate_fn(w)
        if sl.empty:
            continue
        finals = {str(r.game_id): (r.away_score, r.home_score) for r in sl.itertuples()}
        sl = sl.copy()
        # schedule order (undated games first, so a newest-first page puts them last)
        sl = sl.assign(_d=pd.to_datetime(sl["date"], errors="coerce")).sort_values("_d", na_position="first", kind="stable").drop(columns="_d")
        sl["date"] = pd.to_datetime(sl["date"], errors="coerce").dt.strftime("%A, %B %-d").fillna("Date not in feed")
        built = model.build(season, w)
        cards = grade_cards(model.project_slate(season, w, sl, margin_sd=margin_sd, built=built), finals)
        for i, c in enumerate(cards):
            L = locked.get(str(c["game_id"]))
            ap = Path(archive_dir) / f"{c['game_id']}.json" if archive_dir else None
            if L is not None and ap is not None and ap.exists():
                saved = json.loads(ap.read_text())
                saved["kickoff"] = f'{c.get("kickoff") or ""} · LOCKED LIVE {str(L["locked_at"])[:10]}'
                saved["date"], saved["locked"] = c["date"], True
                note = Path(archive_dir).parent.parent / "notes" / "nfl" / f"{c['game_id']}.txt"
                if note.exists():
                    saved["note"] = note.read_text().strip()
                cards[i] = grade_cards([saved], finals)[0]
                continue
            if L is not None:
                for k in ("proj_away", "proj_home", "model_line", "market_line", "pick", "pick_line", "edge",
                          "model_total", "market_total", "cover_prob", "push_prob", "ev"):
                    if k in L and pd.notna(L[k]):
                        c[k] = L[k]
                c["kickoff"] = f'{c.get("kickoff") or ""} · LOCKED LIVE {str(L["locked_at"])[:10]}'
                c["locked"] = True
                grade_cards([c], finals)
        if props_fn is not None:
            props_fn(w, cards, built)
        if sport == "ncaaf":
            cards = [c for c in cards if c.get("market_line") is not None]
        rows += cards_to_rows([c for c in cards if not c.get("locked")], w)
        g = [c for c in cards if c.get("result") in ("WIN", "LOSS", "PUSH")]
        wins = sum(c["result"] == "WIN" for c in g); losses = sum(c["result"] == "LOSS" for c in g)
        pushes = sum(c["result"] == "PUSH" for c in g)
        h3 = [c for c in g if c["edge"] >= 3]
        stats = [{"k": f"Week {w} against the spread", "v": f"{wins}-{losses}-{pushes}",
                  "s": f"{sum(c['result'] == 'WIN' for c in h3)}-{sum(c['result'] == 'LOSS' for c in h3)} on 3+ edges", "href": "../record.html"},
                 {"k": "Games", "v": str(len(cards)), "s": f"{len(cards) - len(g)} not graded" if len(cards) > len(g) else "all graded"}]
        nav = site.week_nav([(x, f"Wk {x}") for x in weeks] + [("live", live_week_label)], w,
                            lambda x: "../index.html" if x == "live" else f"{x}.html")
        site.write(out, f"{sport}/week/{w}.html", site.page(f"{label} Week {w}", site.board(
            sport, f"Week {w}, replayed", f"Projected using only games played before Week {w}, graded against the line in the "
            "data feed. These were generated today, after the fact; only picks in the lock file count as live.",
            cards, names, rec, stats, nav=nav, game_prefix="../game/", links=game_pages, newest_first=True), 2, sport))
        if game_pages:
            for c in cards:
                site.write(out, f"{sport}/game/{c['game_id']}.html", site.page(
                    f"{names.get(c['away'], c['away'])} at {names.get(c['home'], c['home'])}", site.game_page(c, names), 2, sport))
    return rows


def run(data: Path, nfldata: Path, out: Path, ledger_dir: Path):
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    report = {"built_at": now.isoformat(timespec="seconds")}
    home_tiles = []

    # ------------------------------------------------------------------ NFL
    plays = {y: fb.standardize(pd.read_parquet(data / f"nfl/play_by_play_{y}.parquet"), "nfl", fb.NFL_SAM) for y in (2024, 2025, 2026)}
    games = pd.read_csv(nfldata / "games.csv")
    # key-number margin weights, each fitted only on seasons before the one being priced
    nfl_dists = {2025: nfl_from_nflverse(games, range(2010, 2025)), 2026: nfl_from_nflverse(games, range(2010, 2026))}
    report["nfl_key_numbers"] = {"source": nfl_dists[2026].source, "top": nfl_dists[2026].table(8)}
    M = FootballModel("nfl", plays, fb.NFL_SAM, margin_dists=nfl_dists)   # Sam-matched settings + engine
    teams = pd.read_csv(nfldata / "teams.csv")
    names = dict(teams[teams["season"] == teams["season"].max()][["team", "full"]].values)
    g26 = games[(games["season"] == 2026) & games["result"].notna()]
    rec = {}
    for t in names:
        w = ((g26["home_team"] == t) & (g26["result"] > 0)).sum() + ((g26["away_team"] == t) & (g26["result"] < 0)).sum()
        l = ((g26["home_team"] == t) & (g26["result"] < 0)).sum() + ((g26["away_team"] == t) & (g26["result"] > 0)).sum()
        tie = (((g26["home_team"] == t) | (g26["away_team"] == t)) & (g26["result"] == 0)).sum()
        rec[t] = f"{w}-{l}" + (f"-{tie}" if tie else "")
    bt = pd.concat([M.backtest(2025, list(range(3, 19))).assign(date="2025"),
                    M.backtest(2026, [2, 3, 4, 5]).assign(date="2026")])
    acc = accuracy(bt)
    summ, graded = bt_summary(bt, EDGE_TIERS["nfl"])
    report["nfl_backtest"] = {"summary": summ, "accuracy": acc}
    NFL_W = nfl_current_week(games, now)
    report["nfl_live_week"] = NFL_W
    slate = nfl_slate(games, 2026, NFL_W)
    repo = data.parent / "theesness-data"
    pbp26 = pd.read_parquet(data / "nfl/play_by_play_2026.parquet")
    ST = Starters(repo, pbp26, pd.read_parquet(data / "nfl/injuries_2026.parquet"),
                  pd.read_parquet(data / "nfl/roster_weekly_2026.parquet"), NFL_W)
    held, starters = {}, {}
    for i, r in slate.iterrows():
        for side in ("away", "home"):
            x = ST.resolve(r[side], ESPN_ABBR_REV.get(r[side], r[side]), r[f"{side}_qb_name"])
            starters[(r["game_id"], side)] = x
            if x["status"] == "confirmed":
                slate.at[i, f"{side}_qb_id"], slate.at[i, f"{side}_qb_name"] = x["starter_id"], x["starter"]
            else:
                held[r["game_id"]] = f'{r[side]} starting quarterback not confirmed: {x["why"]}'
    # current line: the newest ESPN snapshot with a line (nflverse line only if ESPN has none)
    cur_lines = espn_lines_nfl(repo, pd.to_datetime(slate["date"]).dt.date.unique())
    line_seen = {}
    for i, r in slate.iterrows():
        fl = cur_lines.get((r["away"], r["home"]))
        if fl and fl[0] is not None:
            slate.at[i, "spread_home"] = fl[0]
            if fl[1] is not None:
                slate.at[i, "total"] = fl[1]
            line_seen[r["game_id"]] = f"ESPN/DraftKings, seen {fl[2]}"
        else:
            line_seen[r["game_id"]] = "nflverse schedule"
    g6 = games[(games["season"] == 2026) & (games["week"] == NFL_W)]
    kick = {r.game_id: str(pd.Timestamp(f"{r.gameday} {r.gametime}", tz="America/New_York").tz_convert("UTC"))
            for r in g6.itertuples()}
    slate["date"] = pd.to_datetime(slate["date"]).dt.strftime("%A, %B %-d")
    built = M.build(2026, NFL_W)
    cards = M.project_slate(2026, NFL_W, slate, margin_sd=acc["model_sd"], built=built)
    P = FootballProps(built[4], built[0], acc["model_sd"], 13.1)
    AV = nfl_injuries.Availability(*nfl_injuries.load(2026, data / "nfl"))
    report["nfl_questionable_active_rate"] = AV.q
    for c in cards:
        c["records"] = rec
        c["props"] = nfl_props(c, P, AV)
        sa, sh = starters[(c["game_id"], "away")], starters[(c["game_id"], "home")]
        c["starters"] = f'{sa.get("starter") or "?"} / {sh.get("starter") or "?"}'
        c["inputs_used"] = {"qb": [sa, sh], "line": line_seen.get(c["game_id"]),
                            "injury_report": f"nflverse Week {NFL_W} report + ESPN injury list"}
    # picks update until kickoff: newest roster + newest line; the last version before kickoff is graded
    led = Ledger(ledger_dir / "nfl_locks.csv")
    sync_stats = pickbook.sync(led, cards, kick, ledger_dir / "cards" / "nfl", now, _clean, "nfl", held=held,
                               no_edge=no_edge_map(cards))
    alerts.collect("nfl", cards, names, kick, now)
    report["nfl_pickbook"] = sync_stats
    n_new = sync_stats["locked"] + sync_stats["updated"]
    fin = games[games["season"] == 2026][["game_id", "away_score", "home_score"]]
    led.grade(fin)
    report["nfl_locked_now"] = n_new
    report["nfl_chain_ok"] = led.verify_chain()
    live_rows = led.load_graded()      # superseded (corrected before kickoff) rows are not shown
    live_rows = live_rows.assign(week=live_rows["game_id"].str[5:7].astype(int))
    actuals = nfl_actuals(pd.read_parquet(data / "nfl/play_by_play_2026.parquet"))
    for c in cards:
        attach_actuals(c, actuals)

    def replay_props(w, wcards, wbuilt):
        """Props for a replayed week, from data before that week, then the actual results beside them."""
        WP = FootballProps(wbuilt[4], wbuilt[0], acc["model_sd"], 13.1)
        for c in wcards:
            if not (c.get("locked") and c.get("props")):      # a locked card keeps the props it was locked with
                c["props"] = nfl_props(c, WP, AV)
            attach_actuals(c, actuals)
    live_graded = live_rows[live_rows["result"].isin(["WIN", "LOSS", "PUSH"])]
    live_summ = record_summary(live_graded, EDGE_TIERS["nfl"])
    replay_rows = season_pages(out, "nfl", "NFL", M, 2026, list(range(1, NFL_W)),
                               lambda w: nfl_slate(games, 2026, w), names, acc["model_sd"], f"Wk {NFL_W} · live", True, None,
                               locked=live_rows, archive_dir=ledger_dir / "cards" / "nfl", props_fn=replay_props)
    rp = pd.DataFrame(replay_rows)
    rp_summ = record_summary(rp, EDGE_TIERS["nfl"])
    report["nfl_2026_replay"] = rp_summ
    report["nfl_live"] = live_summ
    hot = [c for c in cards if c.get("edge") and c["edge"] >= 3]
    big = max(cards, key=lambda c: c.get("edge") or 0)
    stats = [{"k": "Live locked record", "v": f'{live_summ[0]["W"]}-{live_summ[0]["L"]}-{live_summ[0]["P"]}',
              "s": f'{len(live_rows)} picks locked, {len(live_graded)} graded', "href": "record.html"},
             {"k": f"2026 replay, Wk 1–{NFL_W - 1}", "v": f'{rp_summ[0]["W"]}-{rp_summ[0]["L"]}-{rp_summ[0]["P"]}',
              "s": f'{rp_summ[1]["W"]}-{rp_summ[1]["L"]} on 3+ edges · not live', "href": f"week/{NFL_W - 1}.html"},
             {"k": "Model disagrees 3+", "v": str(len(hot)), "s": "points off the market this week"},
             {"k": "Biggest gap", "v": f'{big["edge"]:.1f}', "s": f'{names[big["away"]]} at {names[big["home"]]}',
              "href": f'game/{big["game_id"]}.html'}]
    nfin = {str(r.game_id): (r.away_score, r.home_score) for r in games[games["season"] == 2026].itertuples()
            if pd.notna(r.home_score) and pd.notna(r.away_score)}
    totals_tracker.publish(out, ledger_dir, "nfl", cards, kick, now, nfin, names, stats)
    nav = site.week_nav([(x, f"Wk {x}") for x in range(1, NFL_W)] + [("live", f"Wk {NFL_W} · live")], "live",
                        lambda x: "index.html" if x == "live" else f"week/{x}.html")
    site.write(out, "nfl/index.html", site.page("NFL Model Board", site.board(
        "nfl", f"Week {NFL_W} projections", "Locked against the line seen at build time and graded after each game. "
        "Past weeks, including every locked pick and its result, are in the tabs above.",
        cards, names, rec, stats, nav=nav, tier_record=summ), 1, "nfl"))
    for c in cards:
        site.write(out, f"nfl/game/{c['game_id']}.html", site.page(f"{names[c['away']]} at {names[c['home']]}", site.game_page(c, names), 2, "nfl"))
    site.write(out, "nfl/record.html", site.page("NFL Model Record", site.record_page("nfl", "NFL record",
        "Three records, kept apart so the live one is never mixed with hindsight.", [
            {"title": "Live locked record", "note": f"Started {first_lock_date(ledger_dir / 'nfl_locks.csv')}. Every pick written to the hash-chained lock file before "
             "kickoff, against the line seen at lock time, never re-priced.", "summary": live_summ,
             "graded": live_rows.sort_values("week", ascending=False).to_dict("records")},
            {"title": f"2026 season replay, Weeks 1–{NFL_W - 1}", "note": "Each week projected using only games played before it, graded "
             "against the nflverse line. Generated today, not locked at the time, so treat it as a test.",
             "summary": rp_summ, "graded": rp.sort_values("week", ascending=False).to_dict("records")},
            {"title": "Backtest, 2025 Weeks 3–18 and 2026 Weeks 2–5", "note": "Same method over a larger sample.",
             "summary": summ, "graded": []}], names), 1, "nfl"))
    (out / "nfl" / "cards.json").write_text(json.dumps(_clean(cards), indent=1))
    home_tiles.append(("nfl", rp_summ, f"Week {NFL_W} board is live"))

    # ------------------------------------------------------------------ NCAAF
    cplays = {y: fb.standardize(pd.read_parquet(data / f"ncaaf/play_by_play_{y}.parquet"), "ncaaf", fb.NCAAF_SAM) for y in (2025, 2026)}
    cs = pd.concat([slate_from_plays(fb.team_games(cplays[2025]), w) for w in sorted(cplays[2025]["week"].unique())])
    cs = cs.dropna(subset=["spread_home", "home_score"])
    ncaaf_dist = MarginDist.fit(-cs["spread_home"], cs["home_score"] - cs["away_score"],
                                source=f"cfbfastR 2025, {len(cs)} games, closing spread as expectation")
    # Elo ("track record", as on Sam's college pages): CFBD pregame Elo from the play-by-play, carried forward
    # past the last play-by-play week with ESPN finals (see edgemodels/sports/cfb_elo.py)
    elo_games = {y: pd.read_csv(data / f"ncaaf/elo_games_{y}.csv") for y in (2025, 2026)}
    _elo_cache = {}

    elo_src = {}

    def cfb_elo_fn(season, week):
        if (season, week) not in _elo_cache:
            got = cfb_elo.cfbd_week(data.parent / "theesness-data", season, week)
            elo_src[(season, week)] = (f"CollegeFootballData Elo through Week {int(elo_games[season]['week'].max())}, "
                                       "carried forward with final scores")
            later = []
            if season == 2026:
                for w in range(int(elo_games[2026]["week"].max()) + 1, week):
                    wsl, wfin = espn_ncaaf_slate(data.parent / "theesness-data", w)
                    for r in wsl.itertuples():
                        if r.game_id in wfin:
                            a, h = wfin[r.game_id]
                            later.append({"week": w, "home": r.home, "away": r.away, "home_score": h,
                                          "away_score": a, "neutral_site": r.neutral_site})
            carried = cfb_elo.elo_before(elo_games[season], week, later)
            if got:
                # CFBD's own numbers win; a team CFBD names differently keeps the carried-forward value
                elo_src[(season, week)] = f"CollegeFootballData Elo going into Week {week}"
                report.setdefault("cfbd_elo_unmatched", sorted(set(carried) - set(got)))
                carried = {**carried, **got}
            _elo_cache[(season, week)] = carried
        return _elo_cache[(season, week)]

    C = FootballModel("ncaaf", cplays, cfg=fb.NCAAF_SAM, margin_dists={2026: ncaaf_dist}, elo_fn=cfb_elo_fn)
    cbt = pd.concat([C.backtest(2025, list(range(4, 15))).assign(date="2025"), C.backtest(2026, [3, 4, 5, 6]).assign(date="2026")])
    cacc = accuracy(cbt)
    csumm, cgraded = bt_summary(cbt, EDGE_TIERS["ncaaf"])
    report["ncaaf_backtest"] = {"summary": csumm, "accuracy": cacc}
    crepo = data.parent / "theesness-data"
    cevents = espn_ncaaf_events(crepo)
    PBP_LAST = int(C.tg[2026]["week"].max())

    def cslate(w):
        """Past-week slate: from the play-by-play while it covers the week, otherwise from the ESPN scoreboards."""
        sl = slate_from_plays(C.tg[2026], w)
        if not sl.empty or w <= PBP_LAST:
            return sl
        sl, _ = espn_ncaaf_slate(crepo, w)
        if sl.empty:
            return sl
        fin = {k: v["final"] for k, v in cevents.items() if v.get("final")}
        return sl.assign(date=pd.to_datetime(sl["kickoff_utc"]).dt.tz_convert("Pacific/Honolulu").dt.strftime("%Y-%m-%d"),
                         away_score=[fin.get(g, (np.nan, np.nan))[0] for g in sl["game_id"]],
                         home_score=[fin.get(g, (np.nan, np.nan))[1] for g in sl["game_id"]])
    cnames = {t: t for t in set(C.tg[2026]["off"])}
    # live week: slate and first-seen lines from the collected ESPN scoreboards (ESPN ids = cfbfastR game ids)
    LIVE_W = ncaaf_current_week(cevents, now, default=7)      # rolls on its own once the week's games are over
    report["ncaaf_live_week"] = LIVE_W
    cslate_live, _ = espn_ncaaf_slate(crepo, LIVE_W)
    cfinal = {k: v["final"] for k, v in cevents.items() if v.get("final")}     # every finished game, any week
    cled = Ledger(ledger_dir / "ncaaf_locks.csv")
    clocked = cled.load()
    replay_weeks = list(range(1, LIVE_W))
    crows = season_pages(out, "ncaaf", "NCAAF", C, 2026, replay_weeks, cslate, cnames, cacc["model_sd"],
                         f"Wk {LIVE_W} · live", game_pages=False, locked=clocked, archive_dir=ledger_dir / "cards" / "ncaaf")
    crp = pd.DataFrame(crows)
    crp_summ = record_summary(crp, EDGE_TIERS["ncaaf"])
    report["ncaaf_2026_replay"] = crp_summ
    cb = C.build(2026, LIVE_W)
    ccards = C.project_slate(2026, LIVE_W, cslate_live, margin_sd=cacc["model_sd"], built=cb)
    carch = ledger_dir / "cards" / "ncaaf"
    carch.mkdir(parents=True, exist_ok=True)
    started = {r.game_id for r in cslate_live.itertuples() if pd.Timestamp(r.kickoff_utc) <= pd.Timestamp(now)}
    # FBS vs FCS: too little play-by-play on FCS teams to rate them; shown, never locked (rule from 2026-10-15)
    fcs = {c["game_id"] for c in ccards if (c["away"] in cb[0].lower) != (c["home"] in cb[0].lower)}
    cvoids_all = cled.voids()
    voided_ids = set(cvoids_all.loc[cvoids_all["kind"] == "void", "game_id"].astype(str)) if len(cvoids_all) else set()
    ckick = {r.game_id: r.kickoff_utc for r in cslate_live.itertuples()}
    for c in ccards:
        el = c.get("elo") or {}
        c["inputs_used"] = {"line": "ESPN/DraftKings, newest snapshot with a line",
                            "elo": f'{c["away"]} {el.get("away") or 0:.0f} · {c["home"]} {el.get("home") or 0:.0f} '
                                   f'({elo_src.get((2026, LIVE_W), "")})'}
    syncable = [c for c in ccards if c["game_id"] not in fcs and c["game_id"] not in voided_ids]
    csync = pickbook.sync(cled, syncable, ckick, carch, now, _clean, "ncaaf", no_edge=no_edge_map(syncable))
    alerts.collect("ncaaf", syncable, cnames, ckick, now)        # FBS-vs-FCS and voided games are never picks
    n_new = csync["locked"] + csync["updated"]
    by_id = {c["game_id"]: c for c in syncable}
    carch.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(ccards):
        if c["game_id"] in by_id:
            ccards[i] = c = by_id[c["game_id"]]
        elif c["game_id"] in voided_ids and (carch / f'{c["game_id"]}.json').exists():
            saved = json.loads((carch / f'{c["game_id"]}.json').read_text())
            saved.update(date=c["date"], kickoff=c.get("kickoff"), locked=True)
            ccards[i] = c = saved
        if c["game_id"] in started and not c.get("locked"):
            c["kickoff"] = f'{c.get("kickoff") or ""} · started before it could be locked'
        elif c["game_id"] in fcs:
            c["kickoff"] = f'{c.get("kickoff") or ""} · FBS vs FCS, not locked'
    cvoids = {str(r["game_id"]): r for r in cled.voids().to_dict("records") if r.get("kind", "void") == "void"}
    for c in ccards:
        v = cvoids.get(str(c["game_id"]))
        if v:
            at = (pd.Timestamp(v["voided_at"]) - pd.Timedelta(hours=10)).strftime("%b %-d, %-I:%M %p HST")
            c["void"] = {"at": at, "reason": v["reason"]}
            c["note"] = f"VOID, voided {at} before kickoff. {v['reason']} The original locked pick is shown for the record."
            c["kickoff"] = f'{c.get("kickoff") or ""} · VOID'
    ccards = grade_cards(ccards, cfinal)
    for c in ccards:
        if c.get("void"):
            c["result"] = "VOID"
    cled.grade(pd.DataFrame([{"game_id": k, "away_score": v[0], "home_score": v[1]} for k, v in cfinal.items()],
                            columns=["game_id", "away_score", "home_score"]))
    clive = cled.load_graded()
    clive = clive.assign(week=[cevents.get(str(g), {}).get("week") or LIVE_W for g in clive["game_id"]])
    cvoid_rows = clive[clive["result"] == "VOID"]
    clive_summ = record_summary(clive[clive["result"].isin(["WIN", "LOSS", "PUSH"])], EDGE_TIERS["ncaaf"])
    report.update(ncaaf_locked_now=n_new, ncaaf_chain_ok=cled.verify_chain())
    with_line = [c for c in ccards if c.get("market_line") is not None and c["game_id"] not in fcs and not c.get("void")
                 and not c.get("no_edge")]
    big = max(with_line, key=lambda c: c.get("edge") or 0) if with_line else None
    stats = [{"k": "Live locked record", "v": f'{clive_summ[0]["W"]}-{clive_summ[0]["L"]}-{clive_summ[0]["P"]}',
              "s": f'Model {NCAAF_MODEL} · {len(clive)} picks locked', "href": "record.html"},
             {"k": f"2026 replay, Wk 1–{LIVE_W - 1}", "v": f'{crp_summ[0]["W"]}-{crp_summ[0]["L"]}-{crp_summ[0]["P"]}',
              "s": f'{crp_summ[1]["W"]}-{crp_summ[1]["L"]} on 3+ edges · not live', "href": "record.html"},
             {"k": "Model disagrees 3+", "v": str(sum((c.get("edge") or 0) >= 3 for c in ccards)), "s": "points off the market this week"}]
    if big:
        stats.append({"k": "Biggest gap", "v": f'{big["edge"]:.1f}', "s": f'{big["away"]} at {big["home"]}', "href": f'game/{big["game_id"]}.html'})
    totals_tracker.publish(out, ledger_dir, "ncaaf", syncable, ckick, now, cfinal, cnames, stats)
    nav = site.week_nav([(x, f"Wk {x}") for x in replay_weeks] + [("live", f"Wk {LIVE_W} · live")], "live",
                        lambda x: "index.html" if x == "live" else f"week/{x}.html")
    site.write(out, "ncaaf/index.html", site.page("NCAAF Model Board", site.board(
        "ncaaf", f"Week {LIVE_W} projections", f"FBS games. Opponent-adjusted ratings from every play of "
        f"Weeks 1–{PBP_LAST} (shrunk toward 2025) plus CollegeFootballData Elo. Picks update with the newest line and data until "
        "kickoff; the last version before kickoff is graded. NO EDGE = model within 0.5 points of the line (no side, not graded). "
        "Highlighted cards have a gap of 3+ points, darker 6+.", ccards, cnames, None, stats, nav=nav, tier_record=csumm), 1, "ncaaf"))
    for c in ccards:
        site.write(out, f"ncaaf/game/{c['game_id']}.html", site.page(f"{c['away']} at {c['home']}", site.game_page(c, cnames), 2, "ncaaf"))
    site.write(out, "ncaaf/record.html", site.page("NCAAF Model Record", site.record_page("ncaaf", "NCAAF record",
        "Live and replayed records are kept apart.", [
            {"title": f"Live locked record · model {NCAAF_MODEL}", "note": "Every pick is written to the hash-chained lock file "
             "and updates with the newest line and data until kickoff; the last version before kickoff is graded. Model v2 "
             "(efficiency plus CollegeFootballData Elo) replaced v1 on Oct 16, 2026 before public release; v1's "
             "picks (1-4) are archived in the ledger, not counted here.",
             "summary": clive_summ, "graded": clive[clive["result"] != "VOID"].to_dict("records")},
            {"title": "Voided", "note": "Locked picks voided before kickoff, with the reason. Each void is a separate "
             "hash-chained entry citing the original lock; the lock row is never edited. Voids are not in any record. "
             + " ".join(f"{r['away']} at {r['home']}: {cvoids[str(r['game_id'])]['reason']}" for r in cvoid_rows.to_dict("records")[:1]),
             "summary": [], "graded": cvoid_rows.to_dict("records")},
            {"title": f"2026 season replay, Weeks 1–{LIVE_W - 1}", "note": "Each week projected using only games before it "
             "(Week 1 from 2025 ratings alone), graded against the line in the cfbfastR feed. Generated after the fact, not locked.",
             "summary": crp_summ, "graded": crp.sort_values("week", ascending=False).to_dict("records")},
            {"title": "Backtest, 2025 Weeks 4–14 and 2026 Weeks 3–6", "note": "Same method over a larger sample.",
             "summary": csumm, "graded": []}], cnames), 1, "ncaaf"))
    (out / "ncaaf" / "cards.json").write_text(json.dumps(_clean(ccards), indent=1))
    home_tiles.append(("ncaaf", crp_summ, f"Week {LIVE_W} board is live"))

    # ------------------------------------------------------------------ NBA
    nba_raw = {y: pd.read_parquet(data / f"nba/play_by_play_{y}.parquet") for y in (2025, 2026, 2027)
               if (data / f"nba/play_by_play_{y}.parquet").exists()}       # 2027 = this season, once the feed has it
    tg = {y: bb.team_games(p, "nba") for y, p in nba_raw.items()}
    if 2027 in tg:
        tg[2027] = tg[2027][tg[2027]["season_type"] == 2]            # regular season only (no preseason)
    N = BasketballModel("nba", tg)
    nbt = N.backtest(2026, "2025-11-15")
    nbt = nbt[pd.to_datetime(nbt["date"]) < "2026-04-13"]       # regular season only
    nacc = accuracy(nbt)
    nsumm, ngraded = bt_summary(nbt, EDGE_TIERS["nba"])
    report["nba_backtest"] = {"summary": nsumm, "accuracy": nacc}
    p26 = nba_raw[2026]
    nnames = {r.home_team_abbrev: f"{r.home_team_name} {r.home_team_mascot}" for r in p26.drop_duplicates("home_team_abbrev").itertuples()}
    ids = dict(zip(p26["home_team_abbrev"], p26["home_team_id"]))
    day = "2026-04-10"
    br = N.build(2026, day)
    d = tg[2026][(tg[2026]["date"] == pd.Timestamp(day)) & (tg[2026]["is_home"] == 1)]
    sl = pd.DataFrame({"game_id": d["game_id"], "date": "Friday, April 10, 2026 (replay)", "away": d["away"], "home": d["home"],
                       "neutral_site": False, "home_b2b": d["b2b"], "away_b2b": d["opp_b2b"], "spread_home": d["market_home_line"]})
    ncards = N.project_slate(br, sl, nacc["model_sd"])
    pre = p26[pd.to_datetime(p26["game_date"]) < pd.Timestamp(day)]
    es, q, tip = first_basket_table(pre)
    BP = BasketballProps(player_games(pre), br, ids, es, q, tip)
    for c, (_, g) in zip(ncards, d.iterrows()):
        c["kickoff"] = f"Final {g['opp_pts']:.0f}–{g['pts']:.0f}"
        c["_total_sd"] = 18
        pr = BP.game_props(c, top_n=4)
        pr["note"] = (f"Per-game rates from each player's last 15 team games, scaled by projected pace and the opponent's adjusted defence. "
                      f"First basket = P(team scores the first field goal) x the player's share of his team's FGs in the first 6 minutes; "
                      f"the opening-tip winner scored first in {100 * q:.0f}% of games.")
        c["props"] = pr
    big = max(ncards, key=lambda c: c.get("edge") or 0)
    stats = [{"k": "Backtest ATS, 2025–26", "v": f'{nsumm[0]["W"]}-{nsumm[0]["L"]}-{nsumm[0]["P"]}',
              "s": f'{100 * nsumm[0]["pct"]:.1f}% · {nsumm[1]["W"]}-{nsumm[1]["L"]} on 2.5+', "href": "record.html"},
             {"k": "Margin error, model", "v": f'{nacc["model_mae"]:.1f}', "s": f'market {nacc["market_mae"]:.1f} (mean abs. error)'},
             {"k": "Games", "v": str(len(ncards)), "s": "replay of April 10"},
             {"k": "Biggest gap", "v": f'{big["edge"]:.1f}', "s": f'{nnames[big["away"]]} at {nnames[big["home"]]}', "href": f'game/{big["game_id"]}.html'}]
    site.write(out, "nba/index.html", site.page("NBA Model Board", site.board(
        "nba", "In season soon", "The 2026–27 season has not started. This board replays April 10, 2026 with ratings built only from "
        "games before that date, so you can see the full card format, four-factor breakdown and prop simulations.",
        ncards, nnames, None, stats, newest_first=True), 1, "nba"))
    for c in ncards:
        site.write(out, f"nba/game/{c['game_id']}.html", site.page(f"{nnames[c['away']]} at {nnames[c['home']]}", site.game_page(c, nnames), 2, "nba"))
    site.write(out, "nba/record.html", site.page("NBA Model Record", site.record_page(
        "nba", "NBA record", "Rolling out-of-sample backtest, November 15, 2025 to the end of the regular season: each day projected from "
        "games before it, graded against the ESPN line in the play-by-play feed.",
        [{"title": "Live locked record", "note": "Starts when the 2026-27 season tips off.",
          "summary": record_summary(ngraded.iloc[0:0], EDGE_TIERS["nba"]), "graded": []},
         {"title": "2025-26 replay", "note": "Generated today from games before each date, not locked at the time.",
          "summary": nsumm, "graded": ngraded.assign(week=pd.to_datetime(ngraded["date"]).dt.strftime("%m/%d"))
          .sort_values("date", ascending=False).head(400).to_dict("records")}], nnames), 1, "nba"))
    # ---- live board (2026-27): today and the next two days in Hawaii, from the ESPN scoreboards
    nrepo = data.parent / "theesness-data"
    nev = espn_nba_events(nrepo)
    today = (pd.Timestamp(now) - pd.Timedelta(hours=10)).date()
    hst_date = lambda v: (pd.Timestamp(v["kickoff_utc"]) - pd.Timedelta(hours=10)).date()
    window = [v for v in nev.values() if v.get("kickoff_utc") and v.get("season_type") == 2
              and today - pd.Timedelta(days=1) <= hst_date(v) <= today + pd.Timedelta(days=2)]
    nba_tile = (nsumm, "Tips off later this month")
    if window:
        # back-to-back: the team also played the day before (Hawaii dates), from the same scoreboards
        played = {}
        for v in nev.values():
            if v.get("kickoff_utc"):
                for t in (v.get("away"), v.get("home")):
                    played.setdefault(t, set()).add(hst_date(v))
        rows = []
        for v in sorted(window, key=lambda v: v["kickoff_utc"]):
            ko = pd.Timestamp(v["kickoff_utc"])
            d = hst_date(v)
            rows.append({"game_id": v["game_id"], "date": (ko - pd.Timedelta(hours=10)).strftime("%A, %B %-d"),
                         "kickoff": (ko - pd.Timedelta(hours=10)).strftime("%-I:%M %p HST"), "away": v["away"],
                         "home": v["home"], "neutral_site": v["neutral_site"], "venue": v["venue"],
                         "home_b2b": int((d - pd.Timedelta(days=1)) in played.get(v["home"], set())),
                         "away_b2b": int((d - pd.Timedelta(days=1)) in played.get(v["away"], set())),
                         "spread_home": v.get("spread_home"), "total": v.get("total")})
        # injuries: newest ESPN report x each player's impact (see edgemodels/sports/nba_injuries.py)
        from edgemodels.sports import nba_injuries as NI
        itab, irepl = NI.load_table(data)
        injs = NI.espn_injuries(nrepo)
        ab2id = dict(zip(nba_raw[2026]["home_team_abbrev"], nba_raw[2026]["home_team_id"].astype(int)))
        inj_rows = {}
        for r in rows:
            for side in ("home", "away"):
                tid = ab2id.get(r[side])
                d, lst = NI.team_adjustment(tid, injs, itab, irepl, now) if tid is not None else (0.0, [])
                r[f"inj_{side}"] = d
                inj_rows[(r["game_id"], side)] = lst
        nsl = pd.DataFrame(rows)
        nbr = N.build(2027, str(today))
        _miss = sorted({t for r in rows for t in (r["home"], r["away"])} - set(map(str, nbr.ortg.off.index)))
        if _miss:
            alerts.health("nba", "missing_ratings", f"No rating for {', '.join(_miss)}: those teams are projected as league average.")
        lcards = N.project_slate(nbr, nsl, nacc["model_sd"])
        lnames = dict(nnames)
        for v in window:
            lnames.update({k: n for k, n in v["names"].items() if n})
        for c, r in zip(lcards, rows):
            c.update(date=r["date"], kickoff=r["kickoff"], venue=r["venue"], rest={"away": r["away_b2b"], "home": r["home_b2b"]})
            txt = []
            for side in ("away", "home"):
                for x in inj_rows.get((c["game_id"], side), []):
                    if x["status"].startswith("day-to-day, not") or "old note" in x["status"]:
                        continue
                    txt.append(f'{c[side]} {x["name"]} ({x["status"]}, {x["pts100"]:+.1f} per 100; {x["note"]})')
            seen = nev[c["game_id"]].get("line_seen")
            try:
                seen = (pd.Timestamp(str(seen).replace("Z", ""), tz="UTC") - pd.Timedelta(hours=10)).strftime("%b %-d %-I:%M %p HST")
            except (ValueError, TypeError):
                pass
            c["inputs_used"] = {"line": f'ESPN/DraftKings, seen {seen or "—"}'
                                if c.get("market_line") is not None else "no line yet",
                                "injury_report": ("; ".join(txt) or "nobody listed out") + f' (ESPN injury list, {injs[0]["snapshot"] if injs else "none"})'}
        nkick = {v["game_id"]: v["kickoff_utc"] for v in window}
        nled = Ledger(ledger_dir / "nba_locks.csv")
        narch = ledger_dir / "cards" / "nba"
        nsync = pickbook.sync(nled, lcards, nkick, narch, now, _clean, "nba", no_edge=no_edge_map(lcards))
        alerts.collect("nba", lcards, lnames, nkick, now)
        nvoids = {str(r["game_id"]): r for r in nled.voids().to_dict("records") if r.get("kind", "void") == "void"}
        for c in lcards:
            v = nvoids.get(str(c["game_id"]))
            if v:
                at = (pd.Timestamp(v["voided_at"]) - pd.Timedelta(hours=10)).strftime("%b %-d, %-I:%M %p HST")
                when = "after the game" if pd.Timestamp(v["voided_at"]) > pd.Timestamp(v["kickoff_utc"]) else "before tip-off"
                c["void"] = {"at": at, "reason": v["reason"]}
                c["note"] = f"VOID, voided {at} ({when}). {v['reason']} The original locked pick is shown for the record."
                c["kickoff"] = f'{c.get("kickoff") or ""} · VOID'
        nfin = {k: v["final"] for k, v in nev.items() if v.get("final")}
        lcards = grade_cards(lcards, nfin)
        nled.grade(pd.DataFrame([{"game_id": k, "away_score": a, "home_score": h} for k, (a, h) in nfin.items()],
                                columns=["game_id", "away_score", "home_score"]))
        nlive = nled.load_graded()
        nlive = nlive.assign(week=[(pd.Timestamp(nev[str(g)]["kickoff_utc"]) - pd.Timedelta(hours=10)).strftime("%m/%d")
                                   if str(g) in nev else "" for g in nlive["game_id"]]) if len(nlive) else nlive
        nlive_summ = record_summary(nlive[nlive["result"].isin(["WIN", "LOSS", "PUSH"])] if len(nlive) else nlive, EDGE_TIERS["nba"])
        report.update(nba_locked_now=nsync["locked"] + nsync["updated"], nba_chain_ok=nled.verify_chain())
        withl = [c for c in lcards if c.get("market_line") is not None and not c.get("no_edge")]
        big = max(withl, key=lambda c: c.get("edge") or 0) if withl else None
        nstats = [{"k": "Live locked record", "v": f'{nlive_summ[0]["W"]}-{nlive_summ[0]["L"]}-{nlive_summ[0]["P"]}',
                   "s": f'{len(nlive)} picks locked', "href": "record.html"},
                  {"k": "Replay ATS, 2025–26", "v": f'{nsumm[0]["W"]}-{nsumm[0]["L"]}-{nsumm[0]["P"]}',
                   "s": f'{100 * nsumm[0]["pct"]:.1f}% · not live', "href": "record.html"},
                  {"k": "Model disagrees 2.5+", "v": str(sum((c.get("edge") or 0) >= 2.5 for c in withl)), "s": "points off the book, next 3 days"}]
        if big:
            nstats.append({"k": "Biggest gap", "v": f'{big["edge"]:.1f}', "s": f'{lnames[big["away"]]} at {lnames[big["home"]]}',
                           "href": f'game/{big["game_id"]}.html'})
        totals_tracker.publish(out, ledger_dir, "nba", lcards, nkick, now, nfin, lnames, nstats)
        site.write(out, "nba/index.html", site.page("NBA Model Board", site.board(
            "nba", "NBA projections", "Each team scores its points per 100 possessions against the opponent's defence, at the "
            "matchup's pace, plus home court. Early in the season the ratings are still half last season by design. Picks update "
            "with the newest line until tip-off; the last version before tip is graded. NO EDGE under 0.5 points. Highlighted "
            "cards have a gap of 2.5+ points, darker 5+. Injuries from the latest ESPN report are priced in by each missing "
            "player's impact; roster moves since last season are not yet.",
            lcards, lnames, None, nstats, tier_record=nsumm), 1, "nba"))
        for c in lcards:
            site.write(out, f"nba/game/{c['game_id']}.html", site.page(f"{lnames[c['away']]} at {lnames[c['home']]}",
                                                                      site.game_page(c, lnames), 2, "nba"))
        site.write(out, "nba/record.html", site.page("NBA Model Record", site.record_page(
            "nba", "NBA record", "Live picks and the replay are kept apart.",
            [{"title": "Live locked record, 2026-27", "note": "Every pick is written to the hash-chained lock file and updates "
              "until tip-off; the last version before tip is graded.", "summary": nlive_summ,
              "graded": nlive.sort_values("locked_at", ascending=False).to_dict("records") if len(nlive) else []},
             {"title": "2025-26 replay", "note": "Same formula, each day projected from games before it, graded against the ESPN "
              "line in the play-by-play feed. Generated after the fact, not locked.",
              "summary": nsumm, "graded": ngraded.assign(week=pd.to_datetime(ngraded["date"]).dt.strftime("%m/%d"))
              .sort_values("date", ascending=False).head(400).to_dict("records")}], lnames), 1, "nba"))
        (out / "nba" / "cards.json").write_text(json.dumps(_clean(lcards), indent=1))
        nba_tile = (nlive_summ if len(nlive) else nsumm, "Live: today's board")
    home_tiles.append(("nba", *nba_tile))

    # ------------------------------------------------------------------ CBB (no lines in feed -> accuracy only)
    cbb_raw = pd.read_parquet(data / "cbb/play_by_play_2026.parquet")
    ctg = {2026: bb.team_games(cbb_raw, "cbb")}
    ctg[2026] = ctg[2026][ctg[2026]["season_type"] == 2]
    K = BasketballModel("cbb", ctg)
    kbt = K.backtest(2026, "2026-01-01")
    kacc = accuracy(kbt)
    report["cbb_backtest"] = {"accuracy": kacc}
    knames = {r.home_team_abbrev: r.home_team_name for r in cbb_raw.drop_duplicates("home_team_abbrev").itertuples()}
    kday = "2026-03-07"
    kb = K.build(2026, kday)
    kd = ctg[2026][(ctg[2026]["date"] == pd.Timestamp(kday)) & (ctg[2026]["is_home"] == 1)].head(24)
    ksl = pd.DataFrame({"game_id": kd["game_id"], "date": "Saturday, March 7, 2026 (replay)", "away": kd["away"], "home": kd["home"],
                        "neutral_site": False, "home_b2b": 0, "away_b2b": 0, "spread_home": np.nan})
    kcards = K.project_slate(kb, ksl, kacc["model_sd"])
    for c, (_, g) in zip(kcards, kd.iterrows()):
        c["kickoff"] = f"Final {g['opp_pts']:.0f}–{g['pts']:.0f}"
    stats = [{"k": "Margin error, model", "v": f'{kacc["model_mae"]:.1f}', "s": f'{kacc["games"]} games, Jan 1 – season end'},
             {"k": "Market lines", "v": "none", "s": "the ESPN CBB feed carries no spreads"},
             {"k": "Games shown", "v": str(len(kcards)), "s": "replay of March 7"}]
    site.write(out, "cbb/index.html", site.page("CBB Model Board", site.board(
        "cbb", "In season soon", "College basketball returns in November. The free play-by-play feed has no betting lines, so the record "
        "here is projection accuracy only until an odds source is connected (see README).",
        kcards, knames, None, stats, newest_first=True), 1, "cbb"))
    for c in kcards:
        site.write(out, f"cbb/game/{c['game_id']}.html", site.page(f"{knames.get(c['away'], c['away'])} at {knames.get(c['home'], c['home'])}",
                                                                   site.game_page(c, knames), 2, "cbb"))
    home_tiles.append(("cbb", None, "Accuracy only until odds are connected"))

    # ------------------------------------------------------------------ MLB (real Statcast + collected lines)
    mlb_tile = mlb_section(out, data.parent / "theesness-data", ledger_dir, report, now)

    # ------------------------------------------------------------------ NHL (synthetic pipeline demo)
    from edgemodels.synthetic import nhl_shots
    from edgemodels.sports import hockey as hk
    sh = nhl_shots(40)
    sh = hk.add_xg(sh, hk.fit_xg(sh))
    hr = hk.fit_ratings(sh)
    hc = hk.project_game(hr, hk.NHL, {"game_id": "SYNTH_TOR_BOS", "date": "Synthetic test data", "away": "TOR", "home": "BOS",
                                      "away_goalie": 8000026, "home_goalie": 8000001}).card(None, 6.0, 2.3, 2.4)
    hc["kickoff"] = "SYNTHETIC"
    site.write(out, "nhl/index.html", site.page("NHL Model Board", site.board(
        "nhl", "Pipeline test", "Generated test data only. Connect the NHL web API loader to make it live.", [hc],
        {"TOR": "Toronto (synthetic)", "BOS": "Boston (synthetic)"}), 1, "nhl"))
    site.write(out, f"nhl/game/{hc['game_id']}.html", site.page("TOR at BOS synthetic", site.game_page(hc, {"TOR": "Toronto (synthetic)", "BOS": "Boston (synthetic)"}), 2, "nhl"))

    # ------------------------------------------------------------------ home
    start_txt = first_lock_date(ledger_dir / "nfl_locks.csv", ledger_dir / "ncaaf_locks.csv", ledger_dir / "mlb_locks.csv")
    mlb_live = Ledger(ledger_dir / "mlb_locks.csv", extra_cols=["market", "odds"]).load()
    ncaaf_live = Ledger(ledger_dir / "ncaaf_locks.csv").load_graded()
    both = pd.concat([live_rows[["result"]], ncaaf_live[["result"]], mlb_live[["result"]]])
    lw, ll, lp = ((both["result"] == r).sum() for r in ("WIN", "LOSS", "PUSH"))
    allw = sum(s[0]["W"] for _, s, _ in home_tiles if s)
    alll = sum(s[0]["L"] for _, s, _ in home_tiles if s)
    allp = sum(s[0]["P"] for _, s, _ in home_tiles if s)
    tiles = []
    for sp_, s, note in home_tiles:
        nm, desc = site.SPORT_NAMES[sp_]
        v = f'{s[0]["W"]}-{s[0]["L"]}-{s[0]["P"]}' if s else "—"
        tiles.append({"k": f"{nm}, {desc}", "v": v, "s": note, "href": f"{sp_}/index.html"})
    tiles += [mlb_tile,
              {"k": "NHL, hockey", "v": "test", "s": "Synthetic data until the NHL API is connected", "href": "nhl/index.html"}]
    big = 'style="font-family:var(--display);font-size:clamp(44px,13vw,64px);font-weight:800;line-height:1;margin:2px 0 6px"'
    body = (f'<h1>Our number vs the book, every game</h1><p class="lede">We build our own line for every game from '
            f'play-by-play efficiency, adjusted for the schedule each team actually played, and set it beside the sportsbook '
            f'line. Picks keep updating until kickoff, and every one is graded after the final.</p>'
            f'<div class="tiles"><div class="tile"><span class="k">Live record since {start_txt}</span><span class="v" {big}>{lw}-{ll}-{lp}</span>'
            f'<span class="s">{len(live_rows)} NFL spreads, {int((ncaaf_live["result"] != "VOID").sum())} college spreads and '
            f'{len(mlb_live)} MLB moneyline{"s" if len(mlb_live) != 1 else ""}</span></div>'
            f'<div class="tile"><span class="k">Replayed against the spread (not live)</span><span class="v" {big}>{allw}-{alll}-{allp}</span>'
            f'<span class="s">{100 * allw / (allw + alll):.1f}% on NFL 2026 Wk 1–{report.get("nfl_live_week", 6) - 1}, college 2026 Wk 1–{report.get("ncaaf_live_week", 7) - 1} and NBA 2025-26; '
            f'52.4% breaks even at -110</span></div></div>'
            '<h2>Pick a sport</h2><div class="tiles">' + "".join(
                f'<a class="tile" href="{t["href"]}"><span class="k">{site.e(t["k"])}</span><span class="v">{t["v"]}</span>'
                f'<span class="s">{site.e(t["s"])}</span></a>' for t in tiles) + "</div>"
            '<h2>How a pick gets made</h2><div class="tiles">'
            '<div class="tile"><span class="k">Start from the plays</span><span class="s">Football uses expected points and success rate on '
            'every snap, basketball points per possession, baseball quality of contact per plate appearance.</span></div>'
            '<div class="tile"><span class="k">Adjust for the schedule</span><span class="s">Every team is rated against the opponents it '
            'actually faced and blended with last season, so a hot start against weak teams does not fool it.</span></div>'
            '<div class="tile"><span class="k">Check it, then grade it</span><span class="s">Starters are confirmed from two sources before a '
            'pick is made. Each version is saved with a timestamp, and the last one before kickoff is graded.</span></div></div>')
    site.write(out, "index.html", site.page("TheEsness Predictions", body, 0))
    site.write(out, "home.html", site.page("TheEsness Predictions", body, 0, standalone=False))
    (out / "report.json").write_text(json.dumps(_clean(report), indent=1))
    (out / "alerts.json").write_text(json.dumps(_clean(alerts.payload(now)), indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data_raw")
    ap.add_argument("--nfldata", default="../nflverse/nfldata/data")
    ap.add_argument("--out", default="site")
    ap.add_argument("--ledger", default="ledger")
    a = ap.parse_args()
    r = run(Path(a.data), Path(a.nfldata), Path(a.out), Path(a.ledger))
    print(json.dumps(_clean(r), indent=1))
