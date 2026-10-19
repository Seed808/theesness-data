"""MLB section of the site build: real Statcast ratings, collected lines, locks, postseason replay, backtest."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from edgemodels import site
from edgemodels.core import pickbook
from edgemodels.core.ledger import Ledger, record_summary
from edgemodels.core.sim import binary_summary
from edgemodels.props.baseball_props import HIT, OUTCOMES, TB, BaseballProps
from edgemodels.sports import baseball as mb
from edgemodels.sports import mlb_pipeline as mp

MLB_NAMES = {
    "AZ": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals"}
LOCK_EXTRA = ["market", "odds"]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def _pname(raw: pd.DataFrame) -> dict:
    """pitcher id -> 'First Last' (Statcast player_name is the pitcher, 'Last, First')."""
    p = raw[["pitcher", "player_name"]].dropna().drop_duplicates("pitcher")
    return {int(r.pitcher): " ".join(reversed(str(r.player_name).split(", "))) for r in p.itertuples()}


def _id(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)


def _props(BP: BaseballProps, g: dict, lineups: dict, posted: bool = True,
           posted_txt: str | None = None) -> dict:
    """Props for both starters and the top four of each lineup. lineups: {side: [(id, name), ...]} in batting order."""
    g = dict(g, away_sp=_id(g.get("away_sp")), home_sp=_id(g.get("home_sp")))
    players = []
    for side, opp in (("away", "home"), ("home", "away")):
        sp = g.get(f"{side}_sp")
        if sp is not None:
            pr = BP.pitcher(int(sp), g[opp], name=g.get(f"{side}_sp_name") or str(sp))
            pr.update(team=g[side], player_id=int(sp))
            players.append(pr)
    for side, opp in (("away", "home"), ("home", "away")):
        for spot, (pid, nm) in enumerate(lineups.get(side, [])[:4], start=1):
            b = BP.batter(int(pid), spot, g.get(f"{opp}_sp"), g[opp], g["home"], name=nm)
            b.update(team=g[side], player_id=int(pid))
            b["home_run"] = binary_summary(float(b["home_run"]), label="Home run")
            players.append(b)
    posted_txt = posted_txt or ("posted lineup" if posted else "lineup")
    note = ("Starting pitchers: batters faced from his own starts, each one an outcome draw against the opposing lineup "
            "(odds-ratio combination of his rates and the team's). ")
    if any(lineups.values()):
        note += ("Hitters: the top four of the " + posted_txt + ", plate appearances by "
                 "lineup spot, against the starter then the bullpen. RBI counts runs scoring on his plate appearance.")
    else:
        note += "Hitter props appear once lineups are posted (usually a few hours before first pitch)."
    return {"players": players, "n_sims": 20000, "note": note}


def _lineups_or_last(repo: Path, day: str, g: dict, pa26: pd.DataFrame, names: dict) -> tuple[dict, bool]:
    """Posted lineup if MLB has it; otherwise each team's batting order from its most recent game."""
    lu = _lineups(repo, day, str(g["game_id"]))
    posted = True
    for side in ("away", "home"):
        if not lu.get(side):
            posted = False
            t = pa26[pa26["bat_team"] == g[side]]
            if t.empty:
                continue
            last = t[t["game_pk"] == t.sort_values("game_date")["game_pk"].iloc[-1]]
            x = last.drop_duplicates("batter").sort_values("spot")
            lu[side] = [(int(b), names.get(str(int(b)), str(int(b)))) for b in x["batter"]]
    return lu, posted


def _game_actuals(pa_game: pd.DataFrame) -> dict:
    """Box score per player from one game's plate appearances."""
    out = {}
    hit = dict(zip(OUTCOMES, HIT)); tb = dict(zip(OUTCOMES, TB))
    for pid, x in pa_game.groupby("batter"):
        out[int(pid)] = {"hits": int(x["outcome"].map(hit).sum()), "total_bases": int(x["outcome"].map(tb).sum()),
                         "home_run": int((x["outcome"] == "HR").sum()), "rbi": int(x["runs_on_pa"].sum())}
    for pid, x in pa_game.groupby("pitcher"):
        d = out.setdefault(int(pid), {})
        d.update(strikeouts=int((x["outcome"] == "K").sum()), hits_allowed=int(x["outcome"].map(hit).sum()),
                 batters_faced=int(len(x)))
    return out


def _attach_actuals(props: dict | None, act: dict, name_ids: dict | None = None):
    if not props:
        return
    props["final"] = True
    for p in props.get("players", []):
        pid = p.get("player_id")
        if pid is None and name_ids:                     # cards locked before ids were stored
            pid = name_ids.get(p.get("player"))
        p["actual"] = act.get(int(pid), {}) if pid is not None else {}


def _actual_lineups(pa_game: pd.DataFrame, g: dict, names: dict) -> dict:
    """Batting order from the game's own plate appearances (the lineup posted before first pitch)."""
    out = {}
    for side in ("away", "home"):
        x = pa_game[pa_game["bat_team"] == g[side]].drop_duplicates("batter").sort_values("spot")
        out[side] = [(int(b), names.get(str(int(b)), str(int(b)))) for b in x["batter"].head(4)]
    return out


def _lineups(repo: Path, day: str, gid: str) -> dict:
    f = repo / "data" / "mlb" / "schedule" / f"{day}.json"
    if not f.exists():
        return {}
    for d in json.loads(f.read_text()).get("dates", []):
        for g in d["games"]:
            if str(g["gamePk"]) == gid:
                lu = g.get("lineups") or {}
                return {s: [(p["id"], p.get("fullName", str(p["id"]))) for p in lu.get(f"{s}Players", [])]
                        for s in ("away", "home")}
    return {}


def _sched_finals(repo: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((repo / "data" / "mlb" / "schedule").glob("20*-*-[0-9][0-9].json")):
        for d in json.loads(f.read_text()).get("dates", []):
            for g in d["games"]:
                if g["status"].get("abstractGameState") == "Final":
                    rows.append({"game_id": str(g["gamePk"]), "away_score": g["teams"]["away"].get("score"),
                                 "home_score": g["teams"]["home"].get("score")})
    return pd.DataFrame(rows, columns=["game_id", "away_score", "home_score"])


def _acc_table(acc: dict, label: str) -> str:
    return (f'<tr><td>{site.e(label)}</td><td class=mono>{acc["games"]}</td><td class=mono>{100 * acc["winner_pct"]:.1f}%</td>'
            f'<td class=mono>{acc["log_loss"]:.4f}</td><td class=mono>{acc["log_loss_home_rate_only"]:.4f}</td>'
            f'<td class=mono>{acc["total_mae"]:.2f}</td><td class=mono>{acc["total_mae_league_avg"]:.2f}</td>'
            f'<td class=mono>{acc["avg_proj_total"]:.2f} / {acc["avg_actual_total"]:.2f}</td></tr>')


def mlb_section(out: Path, repo: Path, ledger_dir: Path, report: dict, now: datetime) -> dict:
    raw = {y: mp.load_statcast(repo, y) for y in (2025, 2026)}
    pa = {y: mp.plate_appearances(r) for y, r in raw.items()}
    fins = {y: mp.finals(r[r["game_type"].isin(mp.GAME_TYPES)]) for y, r in raw.items()}
    M = mp.MLBModel(pa)
    M.finals = fins
    pname = _pname(pd.concat(raw.values()))
    names = MLB_NAMES
    cfg = M.cfg

    # --- backtest (accuracy only: the feed has no historical odds)
    bt = mp.backtest(M, 2026, "2026-04-14", 7, end="2026-09-29")
    acc_all = mp.accuracy(bt)
    acc_h2 = mp.accuracy(bt[pd.to_datetime(bt["date"]) >= "2026-07-01"])
    report["mlb_backtest"] = {"apr14_sep28": acc_all, "jul_sep": acc_h2}

    # --- postseason replay: every 2026 playoff game, projected from games before its date, actual starters
    sp = pa[2026][pa[2026]["is_sp"] == 1].groupby(["game_pk", "fld_team"])["pitcher"].first()
    post = fins[2026][fins[2026]["game_type"].isin(mp.POST)].sort_values(["date", "game_id"])
    replay = []
    pnames = {}
    for y in (2025, 2026):
        f = repo / "data" / "mlb" / f"players_{y}.json"
        if f.exists():
            pnames.update(json.loads(f.read_text()))
    arch = ledger_dir / "cards" / "mlb"
    arch.mkdir(parents=True, exist_ok=True)
    led_rows = Ledger(ledger_dir / "mlb_locks.csv", extra_cols=LOCK_EXTRA).load_graded()
    locked_now = {str(r["game_id"]): r for r in led_rows.to_dict("records")}
    pa26 = pa[2026]
    name_ids = {v: int(k) for k, v in pnames.items()}
    name_ids.update({v: k for k, v in pname.items()})
    for day, games in post.groupby("date"):
        br = M.build(2026, day)
        BPd = BaseballProps(pa26[pa26["game_date"] < pd.Timestamp(day)], br.park)
        for g in games.to_dict("records"):
            gid = int(g["game_id"])
            pa_game = pa26[pa26["game_pk"] == gid]
            act = _game_actuals(pa_game)
            ap = arch / f"{gid}.json"
            if str(gid) in locked_now and ap.exists():
                c = json.loads(ap.read_text())          # locked live: shown exactly as locked, graded on the final
                c["final_away"], c["final_home"] = g["away_score"], g["home_score"]
                c["ml_result"] = locked_now[str(gid)].get("result") if isinstance(locked_now[str(gid)].get("result"), str) else None
                c["result_label"] = "Locked pick"
                c["fav_result"] = "WIN" if (g["home_score"] > g["away_score"]) == (c["p_home"] >= 0.5) else "LOSS"
                c["kickoff"] = f'{mp.POST[g["game_type"]]} · LOCKED LIVE'
                pr = c.get("props") or {"players": [], "n_sims": 20000, "note": ""}
                if not any("hits" in x for x in pr["players"]):
                    # locked before hitter props existed: add them from games before this date, actual lineup
                    gg0 = dict(g, away_sp=sp.get((gid, g["away"])), home_sp=sp.get((gid, g["home"])))
                    extra = _props(BPd, gg0, _actual_lineups(pa_game, g, pnames), posted_txt="lineup that started")
                    pr["players"] += [x for x in extra["players"] if "hits" in x]
                    pr["note"] = extra["note"]
                c["props"] = pr
                _attach_actuals(c["props"], act, name_ids)
                replay.append(c)
                continue
            a_sp, h_sp = sp.get((gid, g["away"])), sp.get((gid, g["home"]))
            gg = dict(g, away_sp=a_sp, home_sp=h_sp, away_sp_name=pname.get(int(a_sp)) if a_sp else None,
                      home_sp_name=pname.get(int(h_sp)) if h_sp else None, series=mp.POST[g["game_type"]],
                      date=pd.Timestamp(day).strftime("%A, %B %-d"), kickoff=mp.POST[g["game_type"]])
            gg.pop("total", None)
            ln = mp.espn_lines(repo, day).get((g["away"], g["home"]), {})       # first line seen, when the collector has one
            gg.update({k: ln.get(k) for k in ("home_ml", "away_ml", "total", "over_odds", "under_odds")})
            c = mp.project(br, cfg, gg)
            c["final_away"], c["final_home"] = g["away_score"], g["home_score"]
            home_won = g["home_score"] > g["away_score"]
            fav_home = c["p_home"] >= 0.5
            c["fav_result"] = "WIN" if home_won == fav_home else "LOSS"
            if c.get("ml_pick"):
                c["ml_result"] = "WIN" if home_won == (c["ml_pick"] == "home") else "LOSS"
                c["result_label"] = "Model pick (not locked)"
            else:
                c["ml_result"] = c["fav_result"]
                c["result_label"] = "Model favorite"
            c["pitchers"] = c.get("pitchers")
            c["props"] = _props(BPd, gg, _actual_lineups(pa_game, g, pnames), posted_txt="lineup that started")
            c["props"]["note"] = ("Projected before the game from earlier games only; actual results beside each prop. "
                                  + c["props"]["note"])
            _attach_actuals(c["props"], act)
            replay.append(c)
    fav_w = sum(c["fav_result"] == "WIN" for c in replay)

    # --- live slate: today and the next two days (Hawaii dates)
    today = (now - timedelta(hours=10)).date()
    abbr = mp.team_abbr(repo)
    br = M.build(2026, (today + timedelta(days=1)).isoformat())     # every game finished so far, today included
    BP = BaseballProps(pa[2026], br.park)
    led = Ledger(ledger_dir / "mlb_locks.csv", extra_cols=LOCK_EXTRA)
    cards, kick, held = [], {}, {}
    for k in range(3):
        day = (today + timedelta(days=k)).isoformat()
        sl = mp.schedule_slate(repo, day, abbr)
        if sl.empty:
            continue
        lines = mp.espn_lines(repo, day, which="latest")          # current line while the game has not started
        espn_sp = _espn_probables(repo, day)
        for g in sl.to_dict("records"):
            if g["status"] in ("Final", "Game Over", "Completed Early"):
                continue
            ln = lines.get((g["away"], g["home"]), {})
            gg = dict(g, **{k2: ln.get(k2) for k2 in ("home_ml", "away_ml", "total", "over_odds", "under_odds")},
                      date=pd.Timestamp(day).strftime("%A, %B %-d"))
            c = mp.project(br, cfg, gg)
            c["pitchers"] = c.get("pitchers")
            kick[str(g["game_id"])] = str(pd.Timestamp(_first_pitch(repo, day, g["game_id"]) or now.isoformat()))
            lu, posted = _lineups_or_last(repo, day, gg, pa[2026], pnames)
            c["props"] = _props(BP, gg, lu, posted_txt="posted lineup" if posted else "projected lineup (each team's order from its last game; replaced once the real lineup is posted)")
            # starting pitchers: MLB's probable must exist and match ESPN's probable
            checks = []
            for side in ("away", "home"):
                mlb_n, espn_n = g.get(f"{side}_sp_name"), espn_sp.get((g["away"], g["home"]), {}).get(side)
                mlb_n = mlb_n if isinstance(mlb_n, str) and mlb_n else None
                ok = bool(mlb_n) and (espn_n is None or _same(mlb_n, espn_n))
                checks.append({"team": g[side], "starter": mlb_n, "status": "confirmed" if ok else "hold",
                               "mlb": mlb_n, "espn": espn_n, "lineup": "posted" if posted else "projected from last game"})
                if not mlb_n:
                    held[str(g["game_id"])] = f"{g[side]} has no probable starting pitcher yet"
                elif not ok:
                    held[str(g["game_id"])] = f"{g[side]} starter differs: MLB lists {mlb_n}, ESPN lists {espn_n}"
            c["inputs_used"] = {"pitchers": checks, "line": f"ESPN/DraftKings, seen {ln.get('seen', '—')}" if ln.get("home_ml") else "no line yet"}
            c["starters"] = f'{g.get("away_sp_name") or "?"} / {g.get("home_sp_name") or "?"}'
            c["kickoff"] = (c.get("kickoff") or "") + (f' · {g["series"]}' if g.get("series") else "")
            cards.append(c)
    sync_stats = pickbook.sync(led, cards, kick, arch, now, _clean, "mlb",
                               rows_for=lambda c: [r for r in mp.lock_rows(c) if r["market"] == "moneyline"], held=held)
    from edgemodels import alerts
    alerts.collect("mlb", cards, names, kick, now)
    n_new = sync_stats["locked"] + sync_stats["updated"]
    fin_all = pd.concat([_sched_finals(repo), fins[2026][["game_id", "away_score", "home_score"]]]).drop_duplicates("game_id")
    led.grade(fin_all)
    live = led.load_graded()
    live_summ = record_summary(live, (3.0, 5.0, 8.0))
    report.update(mlb_locked_now=n_new, mlb_chain_ok=led.verify_chain(), mlb_live=live_summ)

    # --- pages
    nav = site.week_nav([("post", "Postseason replay"), ("live", "Live")], "live",
                        lambda x: "index.html" if x == "live" else "week/post.html")
    stats = [{"k": "Live locked record", "v": f'{live_summ[0]["W"]}-{live_summ[0]["L"]}-{live_summ[0]["P"]}',
              "s": f'{len(live)} moneylines locked · {live_summ[0]["units"]:+.1f} units', "href": "record.html"},
             {"k": "2026 backtest, Jul–Sep", "v": f'{100 * acc_h2["winner_pct"]:.1f}%',
              "s": f'winners picked, {acc_h2["games"]} games · log loss {acc_h2["log_loss"]:.3f}', "href": "record.html"},
             {"k": "Postseason replay", "v": f"{fav_w}-{len(replay) - fav_w}", "s": "model favorite won-lost", "href": "week/post.html"}]
    lede = ("Ratings from every 2025–26 plate appearance (Statcast), opponent-adjusted, with the probable starters. "
            "Each game is simulated 40,000 times and priced against the first DraftKings moneyline ESPN showed that day. "
            "The moneyline is the locked pick; totals are shown but not picked (no backtest skill yet).")
    site.write(out, "mlb/index.html", site.page("MLB Model Board", site.board(
        "mlb", "MLB postseason board", lede, cards, names, None, stats, nav=nav), 1, "mlb"))
    for c in cards + replay:
        site.write(out, f'mlb/game/{c["game_id"]}.html', site.page(
            f'{names.get(c["away"], c["away"])} at {names.get(c["home"], c["home"])}', site.game_page(c, names), 2, "mlb"))
    nav2 = site.week_nav([("post", "Postseason replay"), ("live", "Live")], "post",
                         lambda x: "../index.html" if x == "live" else "post.html")
    site.write(out, "mlb/week/post.html", site.page("MLB Postseason Replay", site.board(
        "mlb", "2026 postseason, replayed", "Each game projected using only games before its date, with the starters who "
        "actually started. Games before October 13 have no saved line (the collector started that day), so the result is "
        "whether the model's favorite won; games with a saved first line show the model's pick against it. None of these "
        "were locked.", replay, names,
        None, [{"k": "Model favorite", "v": f"{fav_w}-{len(replay) - fav_w}", "s": f"{len(replay)} playoff games"}],
        nav=nav2, game_prefix="../game/", newest_first=True), 2, "mlb"))
    rows = "".join(
        f'<tr><td>{site.e(str(r["date"]))}</td><td>{site.e(names.get(r["away"], r["away"]))} at {site.e(names.get(r["home"], r["home"]))}</td>'
        f'<td>{site.e(names.get(r[r["pick"]], r[r["pick"]]))} {site._ml(r["odds"])}</td><td class=mono>{r["edge"]:.1f}</td>'
        f'<td class=mono>{100 * r["cover_prob"]:.1f}%</td><td class=mono>{r["ev"]:+.3f}</td>'
        + (f'<td class=mono>{r["away_score"]:.0f}–{r["home_score"]:.0f}</td><td class="res-{r["result"]}">{r["result"]}</td>'
           if isinstance(r.get("result"), str) else "<td>—</td><td>pending</td>") + "</tr>"
        for r in live.iloc[::-1].to_dict("records"))
    tiles = "".join(f'<div class="tile"><span class="k">{site.e(s["label"])}</span><span class="v">{s["W"]}-{s["L"]}</span>'
                    f'<span class="s">{s["units"]:+.1f} units at locked odds</span></div>' for s in live_summ)
    body = (f'<p><a class="more" href="index.html">Back to the board</a></p><h1>MLB record</h1>'
            f'<h2>Live locked moneylines</h2><p class="note">Started {_start(live)}. Each pick is the side the model rates above '
            f'the no-vig market, written to the hash-chained lock file before first pitch at the first price seen, '
            f'and graded on the final score. Tiers are percentage points of win probability over the market.</p>'
            f'<div class="tiles">{tiles}</div><div class="tbl"><table><tr><th>Date</th><th>Game</th><th>Pick</th><th>Edge</th>'
            f'<th>Model</th><th>EV</th><th>Final</th><th>Result</th></tr>{rows}</table></div>'
            '<h2>Backtest accuracy, 2026 regular season</h2><p class="note">Ratings rebuilt every 7 days from games before '
            'that week only; each game projected with its actual starters. There are no saved odds for these games, so this '
            'measures accuracy, not profit. Log loss: lower is better; "home rate only" is the score from knowing nothing but '
            'how often the home team wins.</p><div class="tbl"><table><tr><th>Window</th><th>Games</th><th>Winners</th>'
            '<th>Log loss</th><th>Home rate only</th><th>Total error</th><th>League-average total error</th>'
            '<th>Avg total proj / actual</th></tr>'
            + _acc_table(acc_all, "Apr 14 – Sep 28") + _acc_table(acc_h2, "Jul 1 – Sep 28") + "</table></div>"
            '<p class="note">Totals: the model\'s total-runs error is no better than guessing the league average, so totals '
            'are displayed but never locked as picks.</p>')
    site.write(out, "mlb/record.html", site.page("MLB Model Record", body, 1, "mlb"))
    (out / "mlb" / "cards.json").write_text(json.dumps(_clean(cards), indent=1))
    return {"k": "MLB, baseball", "v": f'{live_summ[0]["W"]}-{live_summ[0]["L"]}',
            "s": f'{len(live)} moneylines locked · live', "href": "mlb/index.html"}


def _same(a: str, b: str) -> bool:
    import unicodedata
    n = lambda x: unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower().replace(".", "").replace(" jr", "").strip()
    return n(a) == n(b) or n(a).split()[-1] == n(b).split()[-1]


def _espn_probables(repo: Path, day: str) -> dict:
    """{(away, home): {"away": name, "home": name}} from the newest ESPN scoreboard snapshot."""
    from edgemodels.core.espn_feed import latest_events
    folder = repo / "data" / "espn" / "mlb" / "scoreboard" / day.replace("-", "")
    out = {}
    if not folder.exists():
        return out
    for ev in latest_events(folder):
        c = ev["competitions"][0]
        t, p = {}, {}
        for x in c["competitors"]:
            t[x["homeAway"]] = mp.ESPN_TO_SC.get(x["team"]["abbreviation"], x["team"]["abbreviation"])
            pr = (x.get("probables") or [{}])[0].get("athlete", {})
            p[x["homeAway"]] = pr.get("displayName") or pr.get("fullName")
        out[(t.get("away"), t.get("home"))] = p
    return out


def _start(live: pd.DataFrame) -> str:
    if live.empty:
        return "—"
    return (pd.to_datetime(live["locked_at"], utc=True).min() - pd.Timedelta(hours=10)).strftime("%B %-d, %Y")


def _first_pitch(repo: Path, day: str, gid: str):
    f = repo / "data" / "mlb" / "schedule" / f"{day}.json"
    for d in json.loads(f.read_text()).get("dates", []):
        for g in d["games"]:
            if str(g["gamePk"]) == gid:
                return g.get("gameDate")
    return None
