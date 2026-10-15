"""
Baseball (MLB): run creation vs run prevention from Statcast plate appearances.

  1. One row per PA: batting team, pitcher, park, wOBA value, xwOBA (contact
     quality where a ball was put in play, actual wOBA value otherwise).
  2. Opponent adjustment: quality ~ batting team + pitcher + platoon + home,
     solved jointly (same ridge-to-prior solver as every other sport). Each
     hitter's team is credited for the pitchers it actually faced, and each
     pitcher for the lineups he actually faced.
  3. Starting pitcher is the heaviest input: the starter's own rating covers
     his projected share of batters faced; the bullpen (PA-weighted relievers,
     recent window) covers the rest.
  4. wOBA -> runs with a slope fitted on team-games, times park factor.
  5. Runs simulated (negative binomial) for moneyline, run line, total and F5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.projection import GameProjection
from ..core.ratings import AdjustedRatings, solve_adjusted
from ..core.sim import RNG

STATCAST_COLS = ["game_pk", "game_date", "home_team", "away_team", "inning", "inning_topbot", "at_bat_number",
                 "pitch_number", "batter", "pitcher", "player_name", "stand", "p_throws", "events",
                 "woba_value", "woba_denom", "estimated_woba_using_speedangle", "bat_score", "post_bat_score",
                 "home_score", "away_score", "post_home_score", "post_away_score", "game_type"]

EVENT_MAP = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
             "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "BB",
             "strikeout": "K", "strikeout_double_play": "K"}


@dataclass
class BaseballConfig:
    prior_keep: float = 0.5
    prior_pa_team: float = 600       # PA of evidence the team-offence prior is worth
    prior_pa_pitcher: float = 250    # batters faced the pitcher prior is worth
    pen_window_days: int = 30
    xwoba_weight: float = 0.5        # blend of actual wOBA and xwOBA used as "quality"
    park_regress: float = 0.5
    nb_dispersion: float = 5.0       # runs ~ NegBin; lower = fatter tails
    run_line: float = 1.5
    n_sims: int = 40000

MLB = BaseballConfig()


def load_statcast(start: str, end: str) -> pd.DataFrame:
    """Pull pitch-level Statcast with pybaseball (pip install pybaseball)."""
    from pybaseball import statcast
    return statcast(start_dt=start, end_dt=end)[STATCAST_COLS]


def plate_appearances(sc: pd.DataFrame) -> pd.DataFrame:
    s = sc[sc["events"].notna() & (sc.get("game_type", "R") == "R")].copy()
    s = s.sort_values(["game_pk", "at_bat_number"])
    top = s["inning_topbot"].eq("Top")
    s["bat_team"] = np.where(top, s["away_team"], s["home_team"])
    s["fld_team"] = np.where(top, s["home_team"], s["away_team"])
    s["is_home"] = np.where(top, -1, 1)
    s["same_hand"] = (s["stand"] == s["p_throws"]).astype(int)
    xw = s["estimated_woba_using_speedangle"]
    s["xwoba"] = xw.where(xw.notna(), s["woba_value"])
    s = s[s["woba_denom"].fillna(0) > 0]
    s["quality"] = MLB.xwoba_weight * s["xwoba"] + (1 - MLB.xwoba_weight) * s["woba_value"]
    first_p = s.groupby(["game_pk", "fld_team"])["pitcher"].transform("first")
    s["is_sp"] = (s["pitcher"] == first_p).astype(int)
    s["outcome"] = s["events"].map(EVENT_MAP).fillna("OUT")
    s["runs_on_pa"] = (s["post_bat_score"] - s["bat_score"]).clip(lower=0)
    # lineup spot: order of each batter's first PA within team-game
    first_pa = s.groupby(["game_pk", "bat_team", "batter"])["at_bat_number"].transform("min")
    s["_fp"] = first_pa
    spot = s.drop_duplicates(["game_pk", "bat_team", "batter"]).sort_values("_fp")
    spot["spot"] = spot.groupby(["game_pk", "bat_team"]).cumcount() + 1
    s = s.merge(spot[["game_pk", "bat_team", "batter", "spot"]], on=["game_pk", "bat_team", "batter"], how="left")
    s["game_date"] = pd.to_datetime(s["game_date"])
    s["w"] = 1.0
    return s.drop(columns="_fp")


def team_games(pa: pd.DataFrame) -> pd.DataFrame:
    g = pa.groupby(["game_pk", "bat_team"]).agg(
        date=("game_date", "first"), fld_team=("fld_team", "first"), home=("home_team", "first"),
        away=("away_team", "first"), PA=("w", "sum"), woba=("woba_value", "mean"), quality=("quality", "mean"),
        runs=("runs_on_pa", "sum"), sp=("pitcher", lambda s: s.iloc[0]),
        sp_bf=("is_sp", "sum")).reset_index()
    return g


def park_factors(tg: pd.DataFrame, regress: float = 0.5) -> pd.Series:
    """Runs per game at each park relative to the same teams' games elsewhere, regressed."""
    tot = tg.groupby(["game_pk", "home"])["runs"].sum().reset_index()
    at_park = tot.groupby("home")["runs"].mean()
    lg = tot["runs"].mean()
    raw = at_park / lg
    return (1 + (raw - 1) * (1 - regress)).rename("park_factor")


@dataclass
class BaseballRatings:
    quality: AdjustedRatings          # off = batting team, def = pitcher
    lg_quality: float
    runs_per_pa: float
    slope: float                      # runs per PA per unit of quality
    pa_per_game: float
    sp_bf: pd.Series                  # projected batters faced per start (shrunk)
    lg_sp_bf: float
    pen: pd.Series                    # team bullpen quality term (vs average)
    park: pd.Series
    home_runs: float                  # home-field, runs (margin)
    through: str


def fit_ratings(pa: pd.DataFrame, cfg: BaseballConfig = MLB, prior: BaseballRatings | None = None,
                through: str = "") -> BaseballRatings:
    d = pa.rename(columns={"bat_team": "o"}).assign(d=pa["pitcher"].astype(str))
    po = pdp = None
    if prior is not None:
        po = prior.quality.off * cfg.prior_keep
        pdp = prior.quality.deff * cfg.prior_keep
    q = solve_adjusted(d, "quality", "o", "d", "w", ["same_hand", "is_home"], po, pdp, cfg.prior_pa_pitcher)
    tg = team_games(pa)
    X = np.column_stack([np.ones(len(tg)), tg["quality"]])
    beta, *_ = np.linalg.lstsq(X, (tg["runs"] / tg["PA"]).to_numpy(), rcond=None)
    # starters: batters faced per start, shrunk toward league
    starts = pa[pa["is_sp"] == 1].groupby(["game_pk", "pitcher"]).size()
    per = starts.groupby(level=1).agg(["sum", "count"])
    lg_bf = float(starts.mean())
    sp_bf = (per["sum"] + 5 * lg_bf) / (per["count"] + 5)
    sp_bf.index = sp_bf.index.astype(str)
    # bullpen: relievers' adjusted terms, weighted by recent BF for each team
    recent = pa[(pa["is_sp"] == 0) & (pa["game_date"] >= pa["game_date"].max() - pd.Timedelta(days=cfg.pen_window_days))]
    rw = recent.groupby(["fld_team", "pitcher"]).size().rename("bf").reset_index()
    rw["term"] = rw["pitcher"].astype(str).map(q.deff).fillna(0)
    pen = rw.groupby("fld_team").apply(lambda x: np.average(x["term"], weights=x["bf"]), include_groups=False)
    pf = park_factors(tg, cfg.park_regress)
    home_runs = 2 * q.covariates.get("is_home", 0) * beta[1] * tg["PA"].mean()
    return BaseballRatings(q, q.mu, float((tg["runs"] / tg["PA"]).mean()), float(beta[1]), float(tg["PA"].mean()),
                           sp_bf, lg_bf, pen, pf, float(home_runs), through)


def project_game(br: BaseballRatings, cfg: BaseballConfig, game: dict) -> GameProjection:
    """game: game_id, date, away, home, away_sp, home_sp (Statcast pitcher ids)."""
    h, a = game["home"], game["away"]
    park = float(br.park.get(game.get("park", h), 1.0))
    q = br.quality
    parts, f5 = {}, {}
    for side, bat, fld, sp in (("home", h, a, game.get("away_sp")), ("away", a, h, game.get("home_sp"))):
        sp = str(sp) if sp is not None else None
        bf = float(br.sp_bf.get(sp, br.lg_sp_bf)) if sp else br.lg_sp_bf
        share = min(bf / br.pa_per_game, 1.0)
        sp_term = float(q.deff.get(sp, 0.0)) if sp else 0.0
        pen_term = float(br.pen.get(fld, 0.0))
        off_term = float(q.off.get(bat, 0.0))
        k = br.slope * br.pa_per_game          # runs per game per unit of quality
        parts[side] = {
            "Offence": k * off_term,
            "Starting pitcher": k * share * sp_term,
            "Bullpen": k * (1 - share) * pen_term,
        }
        # first five innings: starter covers min(1, 5 innings of BF / his BF) of it
        pa_f5 = br.pa_per_game * 5 / 9
        sp_f5 = min(bf / pa_f5, 1.0)
        f5[side] = (br.runs_per_pa + br.slope * (off_term + sp_f5 * sp_term + (1 - sp_f5) * pen_term)) * pa_f5 * park
        parts[side]["_meta"] = {"sp": sp, "sp_bf": bf, "sp_share": share, "sp_term": sp_term, "pen_term": pen_term}
    base = br.runs_per_pa * br.pa_per_game
    meta = {s: parts[s].pop("_meta") for s in parts}
    # park scales the whole run environment; express its effect as an additive term per team
    park_add = {s: (base + sum(parts[s].values())) * (park - 1) for s in parts}
    for s in parts:
        parts[s]["Park"] = park_add[s]
    gp = GameProjection("mlb", str(game["game_id"]), str(game.get("date", "")), a, h, base,
                        parts["away"], parts["home"], {"Home field": br.home_runs},
                        extra={"pitchers": meta, "park_factor": park, "f5": {"away": f5["away"], "home": f5["home"]}})
    return gp


def simulate_runs(mean_away: float, mean_home: float, cfg: BaseballConfig = MLB, n: int | None = None) -> dict:
    """NegBin runs; extra innings resolved by a coin weighted by run expectancy."""
    n = n or cfg.n_sims
    k = cfg.nb_dispersion
    ra = RNG.negative_binomial(k, k / (k + mean_away), n)
    rh = RNG.negative_binomial(k, k / (k + mean_home), n)
    tie = ra == rh
    p_home_xtra = mean_home / (mean_home + mean_away)
    xtra_home = RNG.random(n) < p_home_xtra
    rh = rh + (tie & xtra_home)
    ra = ra + (tie & ~xtra_home)
    return {"away": ra, "home": rh}


def market_view(sims: dict, cfg: BaseballConfig = MLB, home_ml=None, away_ml=None, total=None,
                f5_total=None, f5: dict | None = None) -> dict:
    from ..core.edges import american_to_prob, ev_per_unit, no_vig, prob_to_american
    m = sims["home"] - sims["away"]
    out = {"p_home": float((m > 0).mean()),
           "p_home_rl_minus": float((m > cfg.run_line).mean()),     # home -1.5
           "p_away_rl_plus": float((m < cfg.run_line).mean()),      # away +1.5
           "p_away_rl_minus": float((m < -cfg.run_line).mean()),
           "p_home_rl_plus": float((m > -cfg.run_line).mean())}
    out["fair_home_ml"] = prob_to_american(out["p_home"])
    out["fair_away_ml"] = prob_to_american(1 - out["p_home"])
    if home_ml is not None and away_ml is not None:
        ph, pa = no_vig(american_to_prob(home_ml), american_to_prob(away_ml))
        out["market_p_home"] = ph
        out["ml_edge_home"] = out["p_home"] - ph
        side = "home" if out["p_home"] > ph else "away"
        p = out["p_home"] if side == "home" else 1 - out["p_home"]
        out["ml_pick"] = side
        out["ml_ev"] = ev_per_unit(p, home_ml if side == "home" else away_ml)
    tot = sims["home"] + sims["away"]
    if total is not None:
        out["p_over"] = float((tot > total).mean())
        out["p_under"] = float((tot < total).mean())
    return out
