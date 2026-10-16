"""
Football (NFL + NCAAF): EPA-based ratings, opponent adjustment, QB adjustment,
score projection with a per-input margin breakdown.

Pipeline
  load_*  ->  standardize()  ->  team_games()  ->  fit_ratings()
          ->  fit_points_model() (historical)  ->  project_game()
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..core.projection import GameProjection
from ..core.ratings import AdjustedRatings, percentile, solve_adjusted

# ---------------------------------------------------------------- config
@dataclass
class FootballConfig:
    league: str = "nfl"
    prior_keep: float = 0.5          # how much of last season's rating carries into the prior
    prior_plays: dict = field(default_factory=lambda: {
        "pass_epa": 150, "rush_epa": 120, "pass_sr": 150, "rush_sr": 120,
        "pass_rate": 120, "db_rate": 120, "plays": 3, "ypt": 120, "ypc": 120})
    garbage_wp: tuple = (0.05, 0.95)   # drop 2nd-half plays outside this win-prob band
    qb_prior_dropbacks: float = 150
    home_field: float | None = None    # points; None = fit from data
    engine: str = "epa_points"         # "epa_points" (fitted on scores) or "sam" (linear weights decoded from Sam's pages)
    def_prior_mult: float = 1.0        # defence priors count this many times the offence prior (Sam-style: heavier)
    margin_sd: float = 13.2            # out-of-sample residual SD, NFL backtest 2025-26
    total_sd: float = 13.1

NFL = FootballConfig("nfl")
# NCAAF priors are lighter than the NFL's: with 130+ teams and lopsided schedules the
# backtest (2025 wk4-14, 2026 wk3-6) was best calibrated (slope ~1.0) at 40% of the NFL weights.
NCAAF = FootballConfig("ncaaf", prior_keep=0.6, margin_sd=16.9, total_sd=16.5,
                       prior_plays={"pass_epa": 48, "rush_epa": 48, "pass_sr": 48, "rush_sr": 48,
                                    "pass_rate": 40, "db_rate": 40, "plays": 0.8, "ypt": 40, "ypc": 40})

METRICS = {  # metric -> (value column, weight column)
    "pass_epa": ("pass_epa", "dropbacks"), "rush_epa": ("rush_epa", "rushes"),
    "pass_sr": ("pass_sr", "dropbacks"), "rush_sr": ("rush_sr", "rushes"),
    "pass_rate": ("pass_rate", "neutral_plays"),        # style: early-down, neutral-script pass rate
    "db_rate": ("db_rate", "plays"),                    # volume: share of all snaps that are dropbacks
    "plays": ("plays", "one"),
    "ypt": ("ypt", "targets"), "ypc": ("ypc", "rushes"),
}

# ---------------------------------------------------------------- loading
# Non-FBS opponents get a group-level term so they are not shrunk toward the FBS average.
FBS_CONFERENCES = {"ACC", "Big 12", "Big Ten", "SEC", "Pac-12", "American Athletic", "Mountain West",
                   "Sun Belt", "Mid-American", "Conference USA", "FBS Independents"}

NFLVERSE_PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
CFBFASTR_PBP = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfbfastR_cfb_pbp/play_by_play_{season}.parquet"
NFL_SCHEDULE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def load_pbp(league: str, season: int, path: str | None = None) -> pd.DataFrame:
    src = path or (NFLVERSE_PBP if league == "nfl" else CFBFASTR_PBP).format(season=season)
    return pd.read_parquet(src)


def standardize(pbp: pd.DataFrame, league: str, cfg: FootballConfig | None = None) -> pd.DataFrame:
    """One row per scrimmage play with a common schema for NFL and NCAAF."""
    cfg = cfg or (NFL if league == "nfl" else NCAAF)
    if league == "nfl":
        p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()
                & (pbp.get("season_type", "REG") != "PRE")].copy()
        p = p[(p["qb_kneel"] != 1) & (p["qb_spike"] != 1) & (p["two_point_attempt"] != 1)]
        out = pd.DataFrame({
            "game_id": p["game_id"].astype(str), "season": p["season"], "week": p["week"],
            "date": p["game_date"], "off": p["posteam"], "def": p["defteam"],
            "home": p["home_team"], "away": p["away_team"],
            "neutral_site": p["location"].eq("Neutral"),
            "dropback": p["qb_dropback"].fillna(0).astype(int),
            "epa": p["epa"], "success": p["success"],
            "down": p["down"], "wp": p["wp"], "half_secs": p["half_seconds_remaining"], "qtr": p["qtr"],
            "qb_id": p["id"].where(p["qb_dropback"] == 1), "qb": p["name"].where(p["qb_dropback"] == 1),
            "rusher": p["rusher_player_name"], "rusher_id": p["rusher_player_id"],
            "receiver": p["receiver_player_name"], "receiver_id": p["receiver_player_id"],
            "complete": p["complete_pass"], "yards": p["yards_gained"],
            "rush_yds": p["rushing_yards"], "rec_yds": p["receiving_yards"],
            "yardline_100": p["yardline_100"],
            "td_player": p["td_player_name"].where(p["return_touchdown"] != 1),
            "td_player_id": p["td_player_id"].where(p["return_touchdown"] != 1),
            "home_score": p["home_score"], "away_score": p["away_score"],
            "spread_home": -p["spread_line"],   # nflverse spread_line > 0 means home favoured
            "total": p["total_line"],
        })
    else:  # cfbfastR
        p = pbp[(pbp["rush"].eq(1) | pbp["pass"].eq(1)) & pbp["EPA"].notna()].copy()
        p = p[~p["play_type"].astype(str).str.contains("Kickoff|Punt|Field Goal|Timeout|End|Penalty", regex=True)]
        td_mask = p["play_type"].astype(str).str.contains("Rushing Touchdown|Passing Touchdown")
        td_name = np.where(p["rush"].eq(1), p["rusher_player_name"], p["receiver_player_name"])
        out = pd.DataFrame({
            "game_id": p["game_id"].astype(str), "season": p["season"], "week": p["week"],
            "date": p["start_date"].astype(str).str[:10] if "start_date" in p else p["week"].astype(str),
            "off": p["pos_team"], "def": p["def_pos_team"], "home": p["home"], "away": p["away"],
            "neutral_site": p["neutral_site"].fillna(False).astype(bool),
            "dropback": p["pass"].fillna(0).astype(int), "epa": p["EPA"], "success": p["success"],
            "down": p["down"], "wp": p["wp_before"], "half_secs": p["TimeSecsRem"], "qtr": p["period"],
            "qb_id": p["passer_player_name"].where(p["pass"].eq(1)),
            "qb": p["passer_player_name"].where(p["pass"].eq(1)),
            "rusher": p["rusher_player_name"], "rusher_id": p["rusher_player_name"],
            "receiver": p["receiver_player_name"], "receiver_id": p["receiver_player_name"],
            "complete": p["completion"], "yards": p["yards_gained"],
            "rush_yds": p["yds_rushed"].where(p["rush"].eq(1)), "rec_yds": p["yds_receiving"],
            "yardline_100": p["yards_to_goal"],
            "td_player": pd.Series(td_name, index=p.index).where(td_mask),
            "td_player_id": pd.Series(td_name, index=p.index).where(td_mask),
            "home_score": np.nan, "away_score": np.nan,
            "spread_home": p["spread"], "total": p["over_under"],
            "off_lower": (~p["offense_conference"].isin(FBS_CONFERENCES)).astype(int),
            "def_lower": (~p["defense_conference"].isin(FBS_CONFERENCES)).astype(int),
        })
        # final scores from the last row of each game
        # Final scores: in cfbfastR the score fields on kickoffs/returns are unreliable, so take each
        # team's best score seen while it had the ball on scrimmage plays (pre-play score or drive-end
        # score). Verified against published finals; a return TD after the last snap is the only miss.
        s = pbp[pbp["rush"].eq(1) | pbp["pass"].eq(1)]
        ho = s["pos_team"] == s["home"]
        best = np.fmax(s["drive_end_offense_score"], s["pos_team_score"])
        sc = pd.DataFrame({"game_id": s["game_id"].astype(str),
                           "hs": np.where(ho, best, np.nan), "as": np.where(~ho, best, np.nan)}
                          ).groupby("game_id").max().reset_index()
        out = out.merge(sc, on="game_id", how="left")
        out["home_score"], out["away_score"] = out.pop("hs"), out.pop("as")
    for c in ("off_lower", "def_lower"):
        if c not in out:
            out[c] = 0
    lo, hi = cfg.garbage_wp
    second_half = out["qtr"] >= 3
    out["garbage"] = second_half & ((out["wp"] < lo) | (out["wp"] > hi))
    out["neutral"] = out["down"].isin([1, 2]) & out["wp"].between(0.2, 0.8) & (out["qtr"] <= 3)
    out["rz"] = out["yardline_100"] <= 20
    out["target"] = (out["dropback"] == 1) & out["receiver"].notna()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- team-game table
def team_games(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per offence per game: efficiency rates + volumes + points scored."""
    p = plays[~plays["garbage"]]
    g = p.groupby(["game_id", "off", "def"], sort=False)
    db = p[p["dropback"] == 1].groupby(["game_id", "off", "def"])
    ru = p[p["dropback"] == 0].groupby(["game_id", "off", "def"])
    tg = p[p["target"]].groupby(["game_id", "off", "def"])
    nt = p[p["neutral"]].groupby(["game_id", "off", "def"])
    t = pd.DataFrame({
        "dropbacks": db.size(), "pass_epa": db["epa"].mean(), "pass_sr": db["success"].mean(),
        "rushes": ru.size(), "rush_epa": ru["epa"].mean(), "rush_sr": ru["success"].mean(),
        "targets": tg.size(), "ypt": tg["yards"].mean(), "ypc": ru["yards"].mean(),
        "neutral_plays": nt.size(), "pass_rate": nt["dropback"].mean(),
    }).fillna({"dropbacks": 0, "rushes": 0, "targets": 0, "neutral_plays": 0})
    allp = plays.groupby(["game_id", "off", "def"]).agg(
        season=("season", "first"), week=("week", "first"), date=("date", "first"),
        home=("home", "first"), away=("away", "first"), neutral_site=("neutral_site", "first"),
        home_score=("home_score", "max"), away_score=("away_score", "max"),
        spread_home=("spread_home", "first"), total=("total", "first"),
        off_lower=("off_lower", "max"), def_lower=("def_lower", "max"))
    t = t.join(allp, how="right").reset_index()
    t[["dropbacks", "rushes"]] = t[["dropbacks", "rushes"]].fillna(0)
    t["plays"] = t["dropbacks"] + t["rushes"]           # non-garbage scrimmage snaps
    t["db_rate"] = t["dropbacks"] / t["plays"].where(t["plays"] > 0)
    t["is_home"] = np.where(t["neutral_site"], 0, np.where(t["off"] == t["home"], 1, -1))
    t["points"] = np.where(t["off"] == t["home"], t["home_score"], t["away_score"])
    t["one"] = 1.0
    return t


# ---------------------------------------------------------------- ratings
@dataclass
class FootballRatings:
    league: str
    ratings: dict          # metric -> AdjustedRatings
    through: str           # label of the last data included
    lower: set = field(default_factory=set)   # non-FBS teams (group term applies)

    def mu(self, m):
        return self.ratings[m].mu

    def matchup(self, m, off_team, def_team):
        r = self.ratings[m]
        v = r.mu + r.off.get(off_team, 0.0) + r.deff.get(def_team, 0.0)
        if off_team in self.lower:
            v += r.covariates.get("off_lower", 0.0)
        if def_team in self.lower:
            v += r.covariates.get("def_lower", 0.0)
        return v

    def table(self) -> pd.DataFrame:
        cols = {}
        style = {"pass_rate", "plays"}   # style metrics: percentile is just rank, not good/bad
        for m, r in self.ratings.items():
            cols[(m, "off")] = r.off
            cols[(m, "def")] = r.deff
            cols[(m, "off_pct")] = percentile(r.off, True)
            cols[(m, "def_pct")] = percentile(r.deff, m in style)
        return pd.DataFrame(cols)


def fit_ratings(tg: pd.DataFrame, cfg: FootballConfig, prior: FootballRatings | None = None,
                through: str = "") -> FootballRatings:
    out = {}
    covs = ["is_home"] + (["off_lower", "def_lower"] if tg["off_lower"].any() else [])
    for m, (val, w) in METRICS.items():
        po = pdf = None
        if prior is not None and m in prior.ratings:
            po = prior.ratings[m].off * cfg.prior_keep
            pdf = prior.ratings[m].deff * cfg.prior_keep
        out[m] = solve_adjusted(tg.rename(columns={"off": "o", "def": "d"}), val, "o", "d",
                                weight=w, covariates=covs, prior_off=po, prior_def=pdf,
                                prior_weight=cfg.prior_plays[m], prior_weight_def=cfg.prior_plays[m] * cfg.def_prior_mult)
    lower = set(tg.loc[tg["off_lower"] == 1, "off"]) | set(tg.loc[tg["def_lower"] == 1, "def"])
    return FootballRatings(cfg.league, out, through, lower)


# ---------------------------------------------------------------- QB adjustment
def qb_ratings(plays: pd.DataFrame, fr: FootballRatings, cfg: FootballConfig,
               prior_plays: pd.DataFrame | None = None) -> pd.DataFrame:
    """Opponent-adjusted EPA/dropback per QB, shrunk toward backup-level QB play.
    Prior-season dropbacks enter at half weight."""
    d = plays[(plays["dropback"] == 1) & plays["qb_id"].notna() & ~plays["garbage"]].copy()
    d["opp_adj"] = d["def"].map(fr.ratings["pass_epa"].deff).fillna(0)
    d["epa_adj"] = d["epa"] - d["opp_adj"]
    d["w"] = 1.0
    if prior_plays is not None:
        pp = prior_plays[(prior_plays["dropback"] == 1) & prior_plays["qb_id"].notna() & ~prior_plays["garbage"]].copy()
        pp["epa_adj"], pp["w"] = pp["epa"], 0.5
        d = pd.concat([d, pp], ignore_index=True)
    d["wepa"] = d["epa_adj"] * d["w"]
    q = d.groupby("qb_id").agg(qb=("qb", "last"), team=("off", "last"), n=("w", "sum"), s=("wepa", "sum"))
    raw = q["s"] / q["n"]
    backups = q[q["n"] < 100]
    prior = float((backups["s"].sum() / backups["n"].sum())) if len(backups) else float(raw.mean())
    q["epa_db"] = (q["s"] + cfg.qb_prior_dropbacks * prior) / (q["n"] + cfg.qb_prior_dropbacks)
    q.attrs["replacement"] = prior
    return q


def qb_adjustment(team: str, starter_id: str | None, plays: pd.DataFrame, qbs: pd.DataFrame) -> dict:
    """Starter's rating minus the dropback-weighted rating of the QBs inside the team's numbers."""
    d = plays[(plays["off"] == team) & (plays["dropback"] == 1) & plays["qb_id"].notna() & ~plays["garbage"]]
    mix = d["qb_id"].value_counts()
    mix_rating = float((mix * qbs["epa_db"].reindex(mix.index).fillna(qbs.attrs["replacement"])).sum() / mix.sum()) if len(mix) else np.nan
    if starter_id is None and len(d):
        last_game = d[d["game_id"] == d["game_id"].iloc[-1]]
        starter_id = last_game["qb_id"].value_counts().index[0]
    s_rating = float(qbs["epa_db"].get(starter_id, qbs.attrs["replacement"]))
    name = qbs["qb"].get(starter_id, starter_id)
    adj = 0.0 if np.isnan(mix_rating) else s_rating - mix_rating
    return {"team": team, "starter_id": starter_id, "starter": name, "adj": adj,
            "starter_epa_db": s_rating, "in_numbers_epa_db": mix_rating}


# ---------------------------------------------------------------- points model
@dataclass
class PointsModel:
    """points = a + b_play*plays + b_pe*db*pass_epa + b_re*ru*rush_epa
                + b_ps*db*(pass_sr-mu_ps) + b_rs*ru*(rush_sr-mu_rs)"""
    coef: dict
    mu_ps: float
    mu_rs: float
    resid_sd: float
    n: int

    @classmethod
    def fit(cls, tg: pd.DataFrame) -> "PointsModel":
        t = tg.dropna(subset=["points", "pass_epa", "rush_epa", "pass_sr", "rush_sr"])
        mu_ps = np.average(t["pass_sr"], weights=t["dropbacks"])
        mu_rs = np.average(t["rush_sr"], weights=t["rushes"])
        X = np.column_stack([np.ones(len(t)), t["plays"],
                             t["dropbacks"] * t["pass_epa"], t["rushes"] * t["rush_epa"],
                             t["dropbacks"] * (t["pass_sr"] - mu_ps), t["rushes"] * (t["rush_sr"] - mu_rs)])
        beta, *_ = np.linalg.lstsq(X, t["points"].to_numpy(float), rcond=None)
        resid = t["points"] - X @ beta
        names = ["a", "b_play", "b_pe", "b_re", "b_ps", "b_rs"]
        return cls(dict(zip(names, beta)), mu_ps, mu_rs, float(resid.std()), len(t))

    def parts(self, plays, pass_rate, pe, re, ps, rs, lg: dict) -> tuple[float, dict]:
        """Return (league-average baseline, {input: points vs average})."""
        c = self.coef
        db, ru = plays * pass_rate, plays * (1 - pass_rate)
        ldb, lru = lg["plays"] * lg["pass_rate"], lg["plays"] * (1 - lg["pass_rate"])
        base = (c["a"] + c["b_play"] * lg["plays"] + c["b_pe"] * ldb * lg["pass_epa"]
                + c["b_re"] * lru * lg["rush_epa"])
        parts = {
            "Passing game": c["b_pe"] * db * (pe - lg["pass_epa"]),
            "Running game": c["b_re"] * ru * (re - lg["rush_epa"]),
            "Pass consistency": c["b_ps"] * db * (ps - self.mu_ps) - c["b_ps"] * ldb * (lg["pass_sr"] - self.mu_ps),
            "Rush consistency": c["b_rs"] * ru * (rs - self.mu_rs) - c["b_rs"] * lru * (lg["rush_sr"] - self.mu_rs),
            "Pace & play mix": (c["b_play"] * (plays - lg["plays"])
                                + c["b_pe"] * lg["pass_epa"] * (db - ldb) + c["b_re"] * lg["rush_epa"] * (ru - lru)),
        }
        return base, parts


def fit_home_field(tg: pd.DataFrame) -> tuple[float, int]:
    """Home-field in points: covariate on points scored, opponent adjusted. x2 for margin."""
    covs = ["is_home"] + (["off_lower", "def_lower"] if tg["off_lower"].any() else [])
    r = solve_adjusted(tg.rename(columns={"off": "o", "def": "d"}).assign(w=1.0), "points", "o", "d",
                       weight="w", covariates=covs, prior_weight=2.0)
    return 2 * r.covariates["is_home"], int((tg["is_home"] != 0).sum() // 2)


# ---------------------------------------------------------------- projection
def project_game(fr: FootballRatings, pm: PointsModel, cfg: FootballConfig, game: dict,
                 qb_adj: dict | None = None) -> GameProjection:
    """game: game_id, date, away, home, neutral_site(bool), optional rest_home/rest_away."""
    qb_adj = qb_adj or {}
    lg = {m: fr.mu(m) for m in ("plays", "pass_epa", "rush_epa", "pass_sr", "rush_sr")}
    lg["pass_rate"] = fr.mu("db_rate")
    proj, inputs = {}, {}
    for side, o, d in (("home", game["home"], game["away"]), ("away", game["away"], game["home"])):
        plays = fr.matchup("plays", o, d)
        pr = float(np.clip(fr.matchup("db_rate", o, d), 0.3, 0.8))
        pe = fr.matchup("pass_epa", o, d) + qb_adj.get(o, 0.0)
        re = fr.matchup("rush_epa", o, d)
        ps = fr.matchup("pass_sr", o, d)
        rs = fr.matchup("rush_sr", o, d)
        base, parts = pm.parts(plays, pr, pe, re, ps, rs, lg)
        proj[side] = parts
        inputs[side] = dict(plays=plays, pass_rate=pr, pass_epa=pe, rush_epa=re, pass_sr=ps, rush_sr=rs)
    situ = {}
    if not game.get("neutral_site"):
        situ["Home field"] = cfg.home_field
    gp = GameProjection(cfg.league, str(game["game_id"]), str(game.get("date", "")), game["away"], game["home"],
                        base, proj["away"], proj["home"], situ, extra={"inputs": inputs})
    return gp


# ---------------------------------------------------------------- Sam-style engine
# Linear weights recovered from Sam's published Week 5 2026 game pages (13 games, 26 team scores):
# each projected team score = const + home + sum(weight x matchup value), where matchup value =
# team's offence rating + opponent's defence rating (both "vs average"), QB adjustment added to pass EPA.
# Fit quality on his pages: team-score error 0.12 pts on average (max 0.43); margins within 0.2.
SAM_WEIGHTS = {"const": 22.31, "home": 1.80, "pass_epa": 27.5, "pass_sr": 6.8, "rush_epa": 7.7,
               "rush_sr": 18.9, "plays": 0.017, "pass_rate": 6.5, "rest_per_day": 0.27}
SAM_LABELS = {"pass_epa": "Passing game", "pass_sr": "Pass consistency", "rush_epa": "Running game",
              "rush_sr": "Rush consistency", "plays": "Pace", "pass_rate": "Pass rate"}


def project_game_sam(fr: "FootballRatings", cfg: FootballConfig, game: dict, qb_adj: dict | None = None,
                     weights: dict = SAM_WEIGHTS) -> GameProjection:
    qb_adj = qb_adj or {}
    parts = {}
    for side, o, d in (("home", game["home"], game["away"]), ("away", game["away"], game["home"])):
        p = {}
        for m, lab in SAM_LABELS.items():
            r = fr.ratings[m]
            v = r.off.get(o, 0.0) + r.deff.get(d, 0.0)
            if o in fr.lower:
                v += r.covariates.get("off_lower", 0.0)
            if d in fr.lower:
                v += r.covariates.get("def_lower", 0.0)
            if m == "pass_epa":
                v += qb_adj.get(o, 0.0)
            p[lab] = weights[m] * v
        parts[side] = p
    situ = {}
    if not game.get("neutral_site"):
        situ["Home field"] = weights["home"]
    rh, ra = game.get("home_rest"), game.get("away_rest")
    try:
        rest = float(rh) - float(ra)
        if rest and not np.isnan(rest):
            situ["Rest"] = weights["rest_per_day"] * float(np.clip(rest, -3, 3))
    except (TypeError, ValueError):
        pass
    return GameProjection(cfg.league, str(game["game_id"]), str(game.get("date", "")), game["away"], game["home"],
                          weights["const"], parts["away"], parts["home"], situ, extra={"inputs": _sam_inputs(fr, game, qb_adj)})


def _sam_inputs(fr, game, qb_adj):
    """Volumes for the prop models (same as the EPA engine provides)."""
    out = {}
    for side, o, d in (("home", game["home"], game["away"]), ("away", game["away"], game["home"])):
        out[side] = dict(plays=fr.matchup("plays", o, d), pass_rate=float(np.clip(fr.matchup("db_rate", o, d), 0.3, 0.8)),
                         pass_epa=fr.matchup("pass_epa", o, d) + qb_adj.get(o, 0.0), rush_epa=fr.matchup("rush_epa", o, d),
                         pass_sr=fr.matchup("pass_sr", o, d), rush_sr=fr.matchup("rush_sr", o, d))
    return out


# Sam-matching rating settings: no garbage-time filter, prior = 50% of last season, priors twice the
# default strength, defence priors twice the offence (his defensive ratings are much flatter).
# Chosen by matching his published ratings for 26 teams (avg error 0.19 pts per team-metric vs 0.34 before).
NFL_SAM = FootballConfig("nfl", prior_keep=0.5, garbage_wp=(0.0, 1.0), def_prior_mult=2.0, engine="sam",
                         home_field=SAM_WEIGHTS["home"],
                         prior_plays={k: v * 2 for k, v in {"pass_epa": 150, "rush_epa": 120, "pass_sr": 150, "rush_sr": 120,
                                                             "pass_rate": 120, "db_rate": 120, "plays": 3, "ypt": 120, "ypc": 120}.items()})


# ---------------------------------------------------------------- college, Sam-style with Elo
# Sam's college pages add an "Elo / track record" term (Washington @ Purdue, Wk 7 2026: 8.2 pts for a
# 277-point Elo gap) and shrink ratings toward last season. Weights below were fitted to his full Week 7
# 2026 board (59 FBS games, team scores) using our ratings at prior_keep=1.0 and Elo from cfb_elo:
# leave-one-game-out error 0.96 pts on the margin, 1.1 pts on the total.
CFB_SAM_WEIGHTS = {"const": 27.739, "home": 5.357, "elo": 0.0357, "pass_epa": 2.907, "pass_sr": 2.621,
                   "rush_epa": 5.579, "rush_sr": 15.160, "plays": -0.0032, "db_rate": 0.875}
CFB_LABELS = {"pass_epa": "Passing game", "pass_sr": "Pass consistency", "rush_epa": "Running game",
              "rush_sr": "Rush consistency", "plays": "Pace", "db_rate": "Pass rate"}
NCAAF_SAM = replace(NCAAF, prior_keep=1.0, engine="cfb_sam", home_field=CFB_SAM_WEIGHTS["home"])


def project_game_cfb(fr: "FootballRatings", cfg: FootballConfig, game: dict, qb_adj: dict | None = None,
                     weights: dict = CFB_SAM_WEIGHTS) -> GameProjection:
    qb_adj = qb_adj or {}
    he, ae = game.get("home_elo"), game.get("away_elo")
    ed = float(he) - float(ae) if he is not None and ae is not None and not (np.isnan(he) or np.isnan(ae)) else 0.0
    parts = {}
    for side, o, d, sgn in (("home", game["home"], game["away"], 1), ("away", game["away"], game["home"], -1)):
        p = {}
        for m, lab in CFB_LABELS.items():
            r = fr.ratings[m]
            v = r.off.get(o, 0.0) + r.deff.get(d, 0.0)
            if o in fr.lower:
                v += r.covariates.get("off_lower", 0.0)
            if d in fr.lower:
                v += r.covariates.get("def_lower", 0.0)
            if m == "pass_epa":
                v += qb_adj.get(o, 0.0)
            p[lab] = weights[m] * v
        p["Elo / track record"] = sgn * 0.5 * weights["elo"] * ed
        parts[side] = p
    situ = {} if game.get("neutral_site") else {"Home field": weights["home"]}
    ex = {"inputs": _sam_inputs(fr, game, qb_adj), "elo": {"away": ae, "home": he}}
    return GameProjection(cfg.league, str(game["game_id"]), str(game.get("date", "")), game["away"], game["home"],
                          weights["const"], parts["away"], parts["home"], situ, extra=ex)
