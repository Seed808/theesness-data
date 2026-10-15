"""
MLB end to end on real data from the theesness-data repo:

  data/mlb/statcast/<season>/<YYYY-MM>.parquet   pitch-level Statcast (regular season + postseason)
  data/mlb/schedule/<date>.json / .first.json    MLB Stats API schedule, probable pitchers, lineups, scores
  data/espn/mlb/scoreboard/<date>/first.json     first ESPN/DraftKings moneyline, run line and total seen that day

Ratings for any date use only plate appearances before that date (prior season carried at prior_keep).
Every game is priced by simulation: moneyline (model win probability vs the no-vig market) and total.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseball as mb
from ..core.edges import american_to_prob, ev_per_unit, no_vig, prob_to_american

GAME_TYPES = {"R", "F", "D", "L", "W"}           # regular season + wild card, division, LCS, World Series
POST = {"F": "Wild Card", "D": "Division Series", "L": "LCS", "W": "World Series"}
# ESPN abbreviations -> Statcast / MLB Stats API abbreviations
ESPN_TO_SC = {"ARI": "AZ", "CHW": "CWS", "WSH": "WSH", "OAK": "ATH", "LV": "ATH"}
# Playoff run environment: 2025 postseason, 47 games, actual runs / projected runs = 0.904 (better starters, shorter
# bullpen leashes, cold weather). Checked on the 2026 postseason (28 games): projected 7.88 -> 7.12 vs 7.04 actual.
POST_RUN_FACTOR = 0.904
ML_TIERS = (3.0, 5.0, 8.0)       # moneyline edge tiers, percentage points of win probability vs no-vig market
TOTAL_TIERS = (0.5, 1.0, 1.5)    # total edge tiers, runs


def load_statcast(repo: Path, season: int) -> pd.DataFrame:
    files = sorted((repo / "data" / "mlb" / "statcast" / str(season)).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def plate_appearances(sc: pd.DataFrame) -> pd.DataFrame:
    sc = sc[sc["game_type"].isin(GAME_TYPES)] if "game_type" in sc else sc
    sc = sc.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    return mb.plate_appearances(sc.assign(game_type="R"))   # baseball.plate_appearances keeps "R" only


def finals(sc: pd.DataFrame) -> pd.DataFrame:
    """Final score per game from the last pitch's post-pitch score."""
    last = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby("game_pk").tail(1)
    return pd.DataFrame({"game_id": last["game_pk"].astype(str), "date": last["game_date"].astype(str),
                         "game_type": last["game_type"], "away": last["away_team"], "home": last["home_team"],
                         "away_score": last["post_away_score"].astype(float),
                         "home_score": last["post_home_score"].astype(float)}).reset_index(drop=True)


class MLBModel:
    def __init__(self, pa_by_season: dict[int, pd.DataFrame], cfg: mb.BaseballConfig = mb.MLB):
        self.pa = pa_by_season
        self.cfg = cfg
        self._full: dict[int, mb.BaseballRatings] = {}

    def season_ratings(self, season: int):
        if season not in self.pa or self.pa[season].empty:
            return None
        if season not in self._full:
            self._full[season] = mb.fit_ratings(self.pa[season], self.cfg, prior=self.season_ratings(season - 1),
                                                through=f"{season} full")
        return self._full[season]

    def build(self, season: int, before: str) -> mb.BaseballRatings:
        pa = self.pa[season]
        now = pa[pa["game_date"] < pd.Timestamp(before)]
        prior = self.season_ratings(season - 1)
        br = mb.fit_ratings(now, self.cfg, prior=prior, through=f"games before {before}")
        # run environment, run slope and park factors from last season + this season so far
        hist = pd.concat([x for x in (self.pa.get(season - 1), now) if x is not None])
        tg = mb.team_games(hist)
        # runs per unit of quality, fitted on team-season totals: single games are too noisy in "quality"
        # (luck in contact quality) and flatten the slope, which squeezes every team toward average
        tg["season"] = pd.to_datetime(tg["date"]).dt.year
        ts = tg.groupby(["season", "bat_team"]).agg(runs=("runs", "sum"), PA=("PA", "sum"),
                                                    G=("game_pk", "nunique")).reset_index()
        qpa = hist.groupby([hist["game_date"].dt.year, "bat_team"])["quality"].mean()
        ts["quality"] = [qpa.get((r.season, r.bat_team), np.nan) for r in ts.itertuples()]
        ts = ts[ts["PA"] >= 1500]
        if len(ts) >= 20:
            # runs PER GAME on quality: a better lineup also gets more plate appearances per game,
            # which a per-PA slope with fixed PA/game misses. project_game uses slope * pa_per_game.
            X = np.column_stack([np.ones(len(ts)), ts["quality"], ts["season"] == season])
            beta, *_ = np.linalg.lstsq(X, (ts["runs"] / ts["G"]).to_numpy(), rcond=None)
            br.slope = float(beta[1]) / br.pa_per_game
        else:
            X = np.column_stack([np.ones(len(tg)), tg["quality"]])
            beta, *_ = np.linalg.lstsq(X, (tg["runs"] / tg["PA"]).to_numpy(), rcond=None)
            br.slope = float(beta[1])
        # league run rate weighted toward this season once it has a sample
        tg_now = mb.team_games(now)
        w = min(len(tg_now) / 1000, 1.0)
        rpa_prev = float((mb.team_games(self.pa[season - 1])["runs"].sum() / mb.team_games(self.pa[season - 1])["PA"].sum())) \
            if season - 1 in self.pa else br.runs_per_pa
        rpa_now = float(tg_now["runs"].sum() / tg_now["PA"].sum()) if len(tg_now) else rpa_prev
        br.runs_per_pa = w * rpa_now + (1 - w) * rpa_prev
        br.park = mb.park_factors(tg, self.cfg.park_regress)
        # runs scored between plate appearances (wild pitches, steals of home, balks) are not on any PA row:
        # scale the run environment so projected runs match final scores
        fin = getattr(self, "finals", {})
        ids = set(hist["game_pk"].astype(str))
        f = pd.concat([fin[s] for s in (season - 1, season) if s in fin])
        f = f[f["game_id"].isin(ids)]
        if len(f):
            scale = float((f["home_score"] + f["away_score"]).sum() / tg[tg["game_pk"].astype(str).isin(set(f["game_id"]))]["runs"].sum())
            br.runs_per_pa *= scale
            br.slope *= scale
            br.home_runs *= scale
            br.run_scale = scale
        return br


def project(br: mb.BaseballRatings, cfg: mb.BaseballConfig, g: dict) -> dict:
    """g: game_id, date, away, home, away_sp, home_sp (MLBAM ids or None), optional market fields."""
    g = dict(g)
    for k in ("away_sp", "home_sp"):
        v = g.get(k)
        g[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)
    for k in ("home_ml", "away_ml", "total", "over_odds", "under_odds"):
        v = g.get(k)
        g[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    gp = mb.project_game(br, cfg, g)
    if g.get("game_type") in POST:
        pa_, ph_ = gp.proj_away, gp.proj_home
        gp.away_parts["Postseason run environment"] = (POST_RUN_FACTOR - 1) * pa_
        gp.home_parts["Postseason run environment"] = (POST_RUN_FACTOR - 1) * ph_
    c = gp.card(None, g.get("total"), None, None)
    c["model_line"] = round(-(c["proj_home"] - c["proj_away"]), 2)
    sims = mb.simulate_runs(gp.proj_away, gp.proj_home, cfg)
    p_home = float((sims["home"] > sims["away"]).mean())
    tot = sims["home"] + sims["away"]
    c.update(p_home=p_home, fair_home_ml=prob_to_american(p_home), fair_away_ml=prob_to_american(1 - p_home),
             sp_names={"away": g.get("away_sp_name"), "home": g.get("home_sp_name")},
             kickoff=g.get("kickoff", ""), venue=g.get("venue", ""), series=g.get("series", ""))
    c["total_dist"] = {str(t): float((tot > t).mean()) for t in np.arange(5.5, 12.0, 1.0)}
    hm, am = g.get("home_ml"), g.get("away_ml")
    c["home_ml"], c["away_ml"] = hm, am
    if hm is not None and am is not None:
        mh, ma = no_vig(american_to_prob(hm), american_to_prob(am))
        side = "home" if p_home > mh else "away"
        p = p_home if side == "home" else 1 - p_home
        odds = hm if side == "home" else am
        c.update(market_p_home=mh, ml_pick=side, ml_odds=odds, ml_prob=p,
                 ml_edge=round(100 * abs(p_home - mh), 1), ml_ev=ev_per_unit(p, odds))
    if g.get("total") is not None:
        T = float(g["total"])
        po, pu = float((tot > T).mean()), float((tot < T).mean())
        side = "over" if po > pu else "under"
        oo = g.get("over_odds") or -110
        uo = g.get("under_odds") or -110
        p = po if side == "over" else pu
        odds = oo if side == "over" else uo
        # a push returns the stake: EV = p*win - P(loss)
        p_loss = pu if side == "over" else po
        win = odds / 100 if odds > 0 else 100 / -odds
        c.update(tot_pick=side, tot_odds=odds, tot_prob=p, tot_edge=round(abs(c["model_total"] - T), 1),
                 tot_ev=p * win - p_loss, p_over=po, p_under=pu)
    return c


def lock_rows(c: dict) -> list[dict]:
    """Ledger rows for one MLB card: the moneyline side and the total side, each at its locked price."""
    rows = []
    base = {k: c.get(k) for k in ("sport", "date", "away", "home", "proj_away", "proj_home", "model_line", "model_total")}
    if c.get("ml_pick"):
        rows.append(dict(base, game_id=str(c["game_id"]), market="moneyline", odds=c["ml_odds"],
                         market_line=c["home_ml"], pick=c["ml_pick"], pick_line=0.0, edge=c["ml_edge"],
                         tier=sum(c["ml_edge"] >= t for t in ML_TIERS), market_total=c.get("market_total"),
                         cover_prob=c["ml_prob"], push_prob=0.0, ev=c["ml_ev"]))
    if c.get("tot_pick"):
        rows.append(dict(base, game_id=f'{c["game_id"]}-T', market="total", odds=c["tot_odds"],
                         market_line=c.get("market_total"), pick=c["tot_pick"], pick_line=c["market_total"],
                         edge=c["tot_edge"], tier=sum(c["tot_edge"] >= t for t in TOTAL_TIERS),
                         market_total=c.get("market_total"), cover_prob=c["tot_prob"],
                         push_prob=1 - c["p_over"] - c["p_under"], ev=c["tot_ev"]))
    return rows


# ---------------------------------------------------------------- slates from the collected feeds
def _odds(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def espn_lines(repo: Path, day: str, which: str = "first") -> dict:
    """{(away, home): {home_ml, away_ml, total, over_odds, under_odds}} from the saved ESPN snapshot.
    which='first' is the first snapshot seen that day (what picks lock against)."""
    from ..core.espn_feed import first_line_events, latest_line_events
    folder = repo / "data" / "espn" / "mlb" / "scoreboard" / day.replace("-", "")
    if not folder.exists():
        return {}
    pairs = first_line_events(folder) if which == "first" else latest_line_events(folder)
    out = {}
    for ev, seen in pairs:
        c = ev["competitions"][0]
        t = {x["homeAway"]: ESPN_TO_SC.get(x["team"]["abbreviation"], x["team"]["abbreviation"]) for x in c["competitors"]}
        o = (c.get("odds") or [{}])[0]
        ml = o.get("moneyline") or {}
        tot = o.get("total") or {}

        def cur(d):
            d = d or {}
            return _odds((d.get("close") or {}).get("odds")) or _odds((d.get("open") or {}).get("odds"))
        out[(t.get("away"), t.get("home"))] = {
            "home_ml": cur(ml.get("home")), "away_ml": cur(ml.get("away")), "total": o.get("overUnder"),
            "over_odds": cur(tot.get("over")), "under_odds": cur(tot.get("under")), "espn_id": ev.get("id"),
            "seen": (pd.Timestamp(seen.replace("Z", ""), tz="UTC") - pd.Timedelta(hours=10)).strftime("%b %-d %-I:%M %p HST") if seen else None}
    return out


def schedule_slate(repo: Path, day: str, abbr: dict[int, str]) -> pd.DataFrame:
    """Games on `day` (Hawaii date of the ESPN/MLB feeds) with probable pitchers from the MLB Stats API.
    abbr: MLB team id -> Statcast abbreviation."""
    f = repo / "data" / "mlb" / "schedule" / f"{day}.json"
    if not f.exists():
        return pd.DataFrame()
    rows = []
    for d in json.loads(f.read_text()).get("dates", []):
        for g in d["games"]:
            if g.get("gameType") not in GAME_TYPES:
                continue
            ta, th = g["teams"]["away"], g["teams"]["home"]
            pa_, ph = ta.get("probablePitcher") or {}, th.get("probablePitcher") or {}
            utc = pd.Timestamp(g["gameDate"])
            rows.append({"game_id": str(g["gamePk"]), "game_type": g["gameType"],
                         "away": abbr.get(ta["team"]["id"], ta["team"].get("abbreviation")),
                         "home": abbr.get(th["team"]["id"], th["team"].get("abbreviation")),
                         "away_sp": pa_.get("id"), "home_sp": ph.get("id"),
                         "away_sp_name": pa_.get("fullName"), "home_sp_name": ph.get("fullName"),
                         "venue": g.get("venue", {}).get("name", ""),
                         "kickoff": (utc - pd.Timedelta(hours=10)).strftime("%-I:%M %p HST"),
                         "series": f'{POST.get(g["gameType"], "")} Game {g.get("seriesGameNumber", "")}'.strip()
                         if g["gameType"] in POST else "",
                         "status": g["status"]["detailedState"],
                         "away_score": ta.get("score"), "home_score": th.get("score")})
    return pd.DataFrame(rows)


def team_abbr(repo: Path) -> dict[int, str]:
    """MLB team id -> abbreviation, from the saved schedule files (hydrate=team)."""
    out = {}
    for f in sorted((repo / "data" / "mlb" / "schedule").glob("*.json")):
        for d in json.loads(f.read_text()).get("dates", []):
            for g in d["games"]:
                for s in ("away", "home"):
                    t = g["teams"][s]["team"]
                    if t.get("abbreviation"):
                        out[t["id"]] = t["abbreviation"]
    return out


def backtest(model: MLBModel, season: int, start: str, step_days: int = 7, end: str | None = None) -> pd.DataFrame:
    """Rebuild every `step_days`, project each game in the window with its actual starting pitchers.
    No historical odds exist in the feed, so this measures accuracy only (runs, totals, winners)."""
    pa = model.pa[season]
    sp = pa[pa["is_sp"] == 1].groupby(["game_pk", "fld_team"])["pitcher"].first()
    fin = model.finals[season]
    fin = fin[pd.to_datetime(fin["date"]) >= pd.Timestamp(start)]
    if end:
        fin = fin[pd.to_datetime(fin["date"]) < pd.Timestamp(end)]
    rows = []
    d0 = pd.Timestamp(start)
    while d0 <= pd.to_datetime(fin["date"]).max():
        d1 = d0 + pd.Timedelta(days=step_days)
        win = fin[(pd.to_datetime(fin["date"]) >= d0) & (pd.to_datetime(fin["date"]) < d1)]
        if len(win):
            br = model.build(season, d0.strftime("%Y-%m-%d"))
            for g in win.to_dict("records"):
                gid = int(g["game_id"])
                gp = mb.project_game(br, model.cfg, {**g, "away_sp": sp.get((gid, g["away"])), "home_sp": sp.get((gid, g["home"]))})
                sims = mb.simulate_runs(gp.proj_away, gp.proj_home, model.cfg, n=4000)
                rows.append({**g, "proj_away": gp.proj_away, "proj_home": gp.proj_home,
                             "p_home": float((sims["home"] > sims["away"]).mean())})
        d0 = d1
    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt["margin"] = bt["home_score"] - bt["away_score"]
        bt["total"] = bt["home_score"] + bt["away_score"]
    return bt


def accuracy(bt: pd.DataFrame) -> dict:
    y = (bt["margin"] > 0).astype(float)
    p = bt["p_home"].clip(0.01, 0.99)
    base = y.mean()
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    ll0 = float(-(y * np.log(base) + (1 - y) * np.log(1 - base)).mean())
    pt = bt["proj_home"] + bt["proj_away"]
    return {"games": int(len(bt)), "winner_pct": float(((p > 0.5) == (y == 1)).mean()),
            "log_loss": ll, "log_loss_home_rate_only": ll0, "brier": float(((p - y) ** 2).mean()),
            "total_mae": float((pt - bt["total"]).abs().mean()),
            "total_mae_league_avg": float((bt["total"] - bt["total"].mean()).abs().mean()),
            "margin_mae": float(((bt["proj_home"] - bt["proj_away"]) - bt["margin"]).abs().mean()),
            "avg_proj_total": float(pt.mean()), "avg_actual_total": float(bt["total"].mean())}
