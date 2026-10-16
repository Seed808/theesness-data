"""
College Elo ("track record") for the NCAAF model.

Source: CollegeFootballData (CFBD) pregame Elo, as carried in the cfbfastR play-by-play
(home_team_pregame_elo / away_team_pregame_elo). Those files stop at the last week cfbfastR
published, so Elo after that is carried forward with an update rule fitted to CFBD's own
week-to-week changes (2026 Weeks 1-6, 521 team-games):

    change = K x (actual margin - (Elo gap + H x home) / D)
    K = 2.45, H = 20.2, D = 28.53    median error vs CFBD 5 Elo points (about 0.15 pts of spread)

Check against Sam's Week 7 pages: Washington 1698 (Sam 1702), Purdue 1410 (Sam 1425).
When a CFBD API key is available, fetched Elo replaces the carried-forward numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

K, H, D = 2.45, 20.2, 28.53
FCS_ELO = 700.0      # teams CFBD does not rate (FCS). Fitted to Sam's Wk 7 FCS games (Clemson, Texas A&M, Stanford);
                     # display only, FBS-vs-FCS games are never locked


def games_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game with pregame Elo and final score (cfbfastR columns)."""
    d = pbp
    hs = np.where(d["pos_team"] == d["home"], d["pos_team_score"], d["def_pos_team_score"])
    as_ = np.where(d["pos_team"] == d["home"], d["def_pos_team_score"], d["pos_team_score"])
    d = d.assign(_hs=hs, _as=as_)
    return d.groupby("game_id").agg(week=("week", "first"), home=("home", "first"), away=("away", "first"),
                                    home_elo=("home_team_pregame_elo", "first"),
                                    away_elo=("away_team_pregame_elo", "first"),
                                    home_score=("_hs", "max"), away_score=("_as", "max"),
                                    neutral_site=("neutral_site", "first")).reset_index()


def change(home_elo, away_elo, home_score, away_score, neutral) -> float:
    loc = 0.0 if neutral else 1.0
    return K * ((home_score - away_score) - (home_elo + H * loc - away_elo) / D)


def elo_before(games: pd.DataFrame, week: int, later_finals: list[dict] | None = None) -> dict[str, float]:
    """Elo for every team going into `week`.
    games: games_from_pbp output (CFBD pregame Elo + finals).
    later_finals: games after the play-by-play ends, dicts with week, home, away, home_score, away_score,
    neutral_site (from the ESPN scoreboards); applied in week order."""
    g = games[games["week"] < week].sort_values("week")
    elo: dict[str, float] = {}
    for r in g.itertuples():
        # CFBD's pregame number is authoritative whenever we have it
        if not np.isnan(r.home_elo):
            elo[r.home] = float(r.home_elo)
        if not np.isnan(r.away_elo):
            elo[r.away] = float(r.away_elo)
        if r.home in elo and r.away in elo and not np.isnan(r.home_score):
            c = change(elo[r.home], elo[r.away], r.home_score, r.away_score, bool(r.neutral_site))
            # the next pregame value overwrites this when it exists; this carries the last week forward
            elo[r.home] += c
            elo[r.away] -= c
    last_pbp_week = int(games["week"].max()) if len(games) else 0
    for f in sorted(later_finals or [], key=lambda x: x["week"]):
        if f["week"] <= last_pbp_week or f["week"] >= week:
            continue
        h, a = f["home"], f["away"]
        if h in elo and a in elo:
            c = change(elo[h], elo[a], f["home_score"], f["away_score"], f.get("neutral_site", False))
            elo[h] += c
            elo[a] -= c
    # CFBD's pregame number for a game in `week` itself is known before kickoff: use it when present
    for r in games[games["week"] == week].itertuples():
        if not np.isnan(r.home_elo):
            elo[r.home] = float(r.home_elo)
        if not np.isnan(r.away_elo):
            elo[r.away] = float(r.away_elo)
    return elo


def save_games(pbp_path: Path, out_csv: Path) -> None:
    cols = ["game_id", "week", "home", "away", "pos_team", "pos_team_score", "def_pos_team_score",
            "home_team_pregame_elo", "away_team_pregame_elo", "neutral_site"]
    games_from_pbp(pd.read_parquet(pbp_path, columns=cols)).to_csv(out_csv, index=False)


def cfbd_week(repo: Path, year: int, week: int) -> dict[str, float] | None:
    """CFBD's own Elo going into `week` (data/cfbd/elo_{year}_wk{week}.json from scripts/fetch_cfbd.py)."""
    import json
    p = Path(repo) / "data" / "cfbd" / f"elo_{year}_wk{week}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    out = {r["team"]: float(r["elo"]) for r in rows if r.get("elo") is not None}
    return out or None
