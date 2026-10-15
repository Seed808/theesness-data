"""
Hockey (NHL): expected goals from shot events, opponent-adjusted, with a
goaltender adjustment (goals saved above expected) — the hockey analogue of
the football QB adjustment and the baseball starting pitcher.

  xG model     logistic regression on distance, angle, shot type, rebound, strength
  ratings      xGF per game: off = shooting team, def = opponent (home covariate)
               shots per game: same solver (drives props and the goalie term)
  goalie       GSAx per shot faced, shrunk toward 0 (prior 800 shots)
  projection   goals = xG matchup  -  opp goalie GSAx/shot * projected shots
  markets      Poisson simulation -> moneyline, puck line (+/-1.5), total
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.projection import GameProjection
from ..core.ratings import AdjustedRatings, solve_adjusted
from ..core.sim import RNG

NHL_PBP = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
SHOT_TYPES = {"shot-on-goal", "missed-shot", "goal"}


@dataclass
class HockeyConfig:
    prior_keep: float = 0.5
    prior_games: float = 8.0
    goalie_prior_shots: float = 800.0
    margin_sd: float = 2.3
    total_sd: float = 2.4

NHL = HockeyConfig()


def parse_nhl_pbp(js: dict) -> pd.DataFrame:
    """Flatten one game's JSON from the NHL web API into shot rows."""
    home, away = js["homeTeam"], js["awayTeam"]
    rows, prev = [], None
    for p in js.get("plays", []):
        if p.get("typeDescKey") not in SHOT_TYPES:
            prev = p
            continue
        d = p.get("details", {})
        team = home["abbrev"] if d.get("eventOwnerTeamId") == home["id"] else away["abbrev"]
        x, y = d.get("xCoord"), d.get("yCoord")
        if x is None or y is None:
            continue
        gx = 89 if x > 0 else -89
        dist = float(np.hypot(gx - x, y))
        ang = float(np.degrees(np.arctan2(abs(y), abs(gx - x))))
        reb = prev is not None and prev.get("typeDescKey") in SHOT_TYPES
        rows.append({"game_id": js["id"], "date": js.get("gameDate"), "home": home["abbrev"], "away": away["abbrev"],
                     "team": team, "opp": away["abbrev"] if team == home["abbrev"] else home["abbrev"],
                     "shooter": d.get("shootingPlayerId") or d.get("scoringPlayerId"),
                     "goalie": d.get("goalieInNetId"), "goal": int(p["typeDescKey"] == "goal"),
                     "on_goal": int(p["typeDescKey"] in ("shot-on-goal", "goal")),
                     "dist": dist, "angle": ang, "shot_type": d.get("shotType", "unknown"), "rebound": int(reb),
                     "strength": p.get("situationCode", "1551")})
        prev = p
    return pd.DataFrame(rows)


def fit_xg(shots: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    X = pd.get_dummies(shots[["dist", "angle", "rebound", "shot_type"]].assign(
        dist2=shots["dist"] ** 2, pp=(shots["strength"].astype(str) != "1551").astype(int)), columns=["shot_type"])
    m = LogisticRegression(max_iter=2000).fit(X, shots["goal"])
    m.feature_names = X.columns
    return m


def add_xg(shots: pd.DataFrame, model) -> pd.DataFrame:
    X = pd.get_dummies(shots[["dist", "angle", "rebound", "shot_type"]].assign(
        dist2=shots["dist"] ** 2, pp=(shots["strength"].astype(str) != "1551").astype(int)), columns=["shot_type"])
    X = X.reindex(columns=model.feature_names, fill_value=0)
    return shots.assign(xg=model.predict_proba(X)[:, 1])


def team_games(shots: pd.DataFrame) -> pd.DataFrame:
    g = shots.groupby(["game_id", "team", "opp"]).agg(
        date=("date", "first"), home=("home", "first"), away=("away", "first"),
        xg=("xg", "sum"), goals=("goal", "sum"), sog=("on_goal", "sum")).reset_index()
    g["is_home"] = np.where(g["team"] == g["home"], 1, -1)
    g["one"] = 1.0
    return g


@dataclass
class HockeyRatings:
    xg: AdjustedRatings
    sog: AdjustedRatings
    gsax: pd.Series            # goals saved above expected per shot on goal
    through: str


def fit_ratings(shots: pd.DataFrame, cfg: HockeyConfig = NHL, prior: HockeyRatings | None = None,
                through: str = "") -> HockeyRatings:
    tg = team_games(shots).rename(columns={"team": "o", "opp": "d"})
    k = cfg.prior_keep
    xg = solve_adjusted(tg, "xg", "o", "d", "one", ["is_home"],
                        prior.xg.off * k if prior else None, prior.xg.deff * k if prior else None, cfg.prior_games)
    sog = solve_adjusted(tg, "sog", "o", "d", "one", ["is_home"],
                         prior.sog.off * k if prior else None, prior.sog.deff * k if prior else None, cfg.prior_games)
    on = shots[shots["on_goal"] == 1].dropna(subset=["goalie"])
    gq = on.groupby("goalie").agg(n=("xg", "size"), xg=("xg", "sum"), ga=("goal", "sum"))
    gsax = (gq["xg"] - gq["ga"]) / (gq["n"] + cfg.goalie_prior_shots)
    return HockeyRatings(xg, sog, gsax, through)


def project_game(hr: HockeyRatings, cfg: HockeyConfig, game: dict) -> GameProjection:
    """game: game_id, date, away, home, away_goalie, home_goalie."""
    h, a = game["home"], game["away"]
    parts = {}
    for side, o, d, g_opp in (("home", h, a, game.get("away_goalie")), ("away", a, h, game.get("home_goalie"))):
        sog = hr.sog.mu + hr.sog.off.get(o, 0) + hr.sog.deff.get(d, 0)
        parts[side] = {"Offence (xG created)": hr.xg.off.get(o, 0.0),
                       "Defence (xG allowed)": hr.xg.deff.get(d, 0.0),
                       "Goaltending": -float(hr.gsax.get(g_opp, 0.0)) * sog}
    # note: for the margin, the home team's "Defence" bar is the away team's xG-allowed term, etc.
    situ = {"Home ice": 2 * hr.xg.covariates.get("is_home", 0.0)}
    return GameProjection("nhl", str(game["game_id"]), str(game.get("date", "")), a, h, hr.xg.mu,
                          parts["away"], parts["home"], situ)


def simulate_goals(mean_away: float, mean_home: float, n: int = 40000) -> dict:
    ga, gh = RNG.poisson(mean_away, n), RNG.poisson(mean_home, n)
    tie = ga == gh
    home_ot = RNG.random(n) < mean_home / (mean_home + mean_away)
    return {"away": ga, "home": gh, "home_win": (gh > ga) | (tie & home_ot)}
