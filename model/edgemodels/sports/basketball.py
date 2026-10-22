"""
Basketball (NBA + CBB): possession-based efficiency from play-by-play,
opponent-adjusted, projected into scores with a four-factors margin breakdown.

  points  = ORtg_matchup / 100 * possessions_matchup
  ORtg_matchup = mu + off[team] + def[opp] + home + rest terms
  breakdown: the ORtg gap is split into shooting (eFG%), turnovers, offensive
  rebounding and free-throw rate using weights fitted from the data, plus
  pace, home court and rest. A residual "Other efficiency" row keeps the sum exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.projection import GameProjection
from ..core.ratings import AdjustedRatings, percentile, solve_adjusted

ESPN_PBP = {
    "nba": "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_pbp/play_by_play_{season}.parquet",
    "cbb": "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_pbp/play_by_play_{season}.parquet",
}


@dataclass
class BasketballConfig:
    league: str = "nba"
    prior_keep: float = 0.6
    prior_poss: float = 600          # possessions of evidence the prior is worth
    prior_games_pace: float = 6
    margin_sd: float = 12.5          # replaced by backtest residual SD
    total_sd: float = 18.0
    neutral_default: bool = False
    engine: str = "factors"          # "factors" (four-factor breakdown) or "sam" (Sam's NBA formula)
    home_court: float | None = None  # points to the home margin; None = fitted covariate

NBA = BasketballConfig("nba")
# Sam's NBA formula, read from his Oct 20 2026 game pages: points per 100 = league + team offence + opponent defence,
# times the matchup pace; home court a flat 1.8 points; ratings half of last season early on ("still part last
# year, by design"). Matches his NYK -5.0 (ours -5.1) and SAS -0.3 (ours -0.3) on opening night.
NBA_SAM = BasketballConfig("nba", prior_keep=0.5, engine="sam", home_court=1.8)
CBB = BasketballConfig("cbb", prior_keep=0.5, prior_poss=350, prior_games_pace=4, margin_sd=11.0, total_sd=16.0)

FACTORS = ["efg", "tov", "orb", "ftr"]
FACTOR_LABELS = {"efg": "Shooting (eFG%)", "tov": "Ball security (TOV%)",
                 "orb": "Offensive rebounding", "ftr": "Free throws (FT rate)"}


def load_pbp(league: str, season: int, path: str | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or ESPN_PBP[league].format(season=season))


def team_games(pbp: pd.DataFrame, league: str) -> pd.DataFrame:
    """One row per team per game with box-score style counts derived from events."""
    e = pbp.copy()
    tt = e["type_text"].astype(str)
    shot = e["shooting_play"].fillna(False).astype(bool)
    ft = shot & ((e["points_attempted"] == 1) | tt.str.contains("Free Throw|FreeThrow"))
    fga = shot & ~ft
    made = e["scoring_play"].fillna(False).astype(bool)
    e = e.assign(FGA=fga, FGM=fga & made, FG3A=fga & (e["points_attempted"] == 3),
                 FG3M=fga & made & (e["points_attempted"] == 3), FTA=ft, FTM=ft & made,
                 OREB=tt.eq("Offensive Rebound"), DREB=tt.eq("Defensive Rebound"),
                 TOV=tt.str.contains("Turnover") & ~tt.str.contains("No Turnover"))
    e["team_id"] = e["team_id"].astype("Int64")
    cnt = e.groupby(["game_id", "team_id"])[["FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "OREB", "DREB", "TOV"]].sum()
    meta = e.groupby("game_id").agg(
        date=("game_date", "first"), season=("season", "first"), season_type=("season_type", "first"),
        home_id=("home_team_id", "first"), away_id=("away_team_id", "first"),
        home=("home_team_abbrev", "first"), away=("away_team_abbrev", "first"),
        home_score=("home_score", "max"), away_score=("away_score", "max"),
        hts=("home_team_spread", "first"), spread_avail=("game_spread_available", "first"))
    rows = []
    for gid, m in meta.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            tid, oid = m[f"{side}_id"], m[f"{opp}_id"]
            if (gid, tid) not in cnt.index or (gid, oid) not in cnt.index:
                continue
            c, o = cnt.loc[(gid, tid)], cnt.loc[(gid, oid)]
            rows.append({"game_id": str(gid), "date": m["date"], "season": m["season"], "season_type": m["season_type"],
                         "team": m[side], "opp": m[opp], "home": m["home"], "away": m["away"],
                         "is_home": 1 if side == "home" else -1,
                         "pts": m[f"{side}_score"], "opp_pts": m[f"{opp}_score"],
                         **{k: float(c[k]) for k in cnt.columns}, "opp_DREB": float(o["DREB"]), "opp_OREB": float(o["OREB"]),
                         "market_home_line": (-m["hts"] if m["spread_avail"] and m["hts"] not in (0, None) and pd.notna(m["hts"]) else np.nan)})
    t = pd.DataFrame(rows)
    t["poss_raw"] = t["FGA"] - t["OREB"] + t["TOV"] + 0.44 * t["FTA"]
    gp = t.groupby("game_id")["poss_raw"].transform("mean")          # both teams share possessions
    t["poss"] = gp
    t["ortg"] = 100 * t["pts"] / t["poss"]
    t["efg"] = (t["FGM"] + 0.5 * t["FG3M"]) / t["FGA"]
    t["tov"] = t["TOV"] / t["poss"]
    t["orb"] = t["OREB"] / (t["OREB"] + t["opp_DREB"])
    t["ftr"] = t["FTM"] / t["FGA"]
    t["three_rate"] = t["FG3A"] / t["FGA"]
    t["one"] = 1.0
    # rest: back-to-back flags for this team and its opponent
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["team", "date"])
    t["days_rest"] = t.groupby("team")["date"].diff().dt.days
    t["b2b"] = (t["days_rest"] == 1).astype(int)
    opp_b2b = t.set_index(["game_id", "team"])["b2b"]
    t["opp_b2b"] = pd.Series(opp_b2b.reindex(list(zip(t["game_id"], t["opp"]))).to_numpy()).fillna(0).astype(int).to_numpy()
    t["rest_diff"] = t["b2b"].astype(float) - t["opp_b2b"].astype(float)
    return t.dropna(subset=["ortg"]).reset_index(drop=True)


@dataclass
class BasketballRatings:
    league: str
    ortg: AdjustedRatings
    pace: AdjustedRatings
    factors: dict
    factor_coef: dict          # ORtg points per unit of each factor
    through: str

    def ortg_matchup(self, o, d, home_sign, b2b_o=0, b2b_d=0):
        r = self.ortg
        return (r.mu + r.off.get(o, 0) + r.deff.get(d, 0) + home_sign * r.covariates.get("is_home", 0)
                + b2b_o * r.covariates.get("b2b", 0) + b2b_d * r.covariates.get("opp_b2b", 0))

    def factor_matchup(self, f, o, d):
        r = self.factors[f]
        return r.mu + r.off.get(o, 0) + r.deff.get(d, 0)

    def pace_matchup(self, a, b):
        r = self.pace
        return 0.5 * ((r.mu + r.off.get(a, 0) + r.deff.get(b, 0)) + (r.mu + r.off.get(b, 0) + r.deff.get(a, 0)))

    def table(self) -> pd.DataFrame:
        cols = {("ortg", "off"): self.ortg.off, ("ortg", "def"): self.ortg.deff,
                ("ortg", "off_pct"): percentile(self.ortg.off), ("ortg", "def_pct"): percentile(self.ortg.deff, False),
                ("pace", "off"): self.pace.off, ("pace", "off_pct"): percentile(self.pace.off)}
        for f, r in self.factors.items():
            good_high = f != "tov"
            cols[(f, "off")], cols[(f, "def")] = r.off, r.deff
            cols[(f, "off_pct")] = percentile(r.off, good_high)
            cols[(f, "def_pct")] = percentile(r.deff, not good_high)
        return pd.DataFrame(cols)


def fit_ratings(tg: pd.DataFrame, cfg: BasketballConfig, prior: BasketballRatings | None = None,
                through: str = "") -> BasketballRatings:
    d = tg.rename(columns={"team": "o", "opp": "d"})
    k = cfg.prior_keep
    pr = (lambda r: (r.off * k, r.deff * k)) if prior else (lambda r: (None, None))
    covs = ["is_home", "b2b", "opp_b2b"]
    po, pd_ = pr(prior.ortg) if prior else (None, None)
    ortg = solve_adjusted(d, "ortg", "o", "d", "poss", covs, po, pd_, cfg.prior_poss)
    po, pd_ = pr(prior.pace) if prior else (None, None)
    pace = solve_adjusted(d, "poss", "o", "d", "one", [], po, pd_, cfg.prior_games_pace)
    factors = {}
    for f in FACTORS:
        po, pd_ = pr(prior.factors[f]) if prior else (None, None)
        factors[f] = solve_adjusted(d, f, "o", "d", "poss", ["is_home"], po, pd_, cfg.prior_poss)
    if prior:
        # league scoring level and pace: early in a season a handful of games decide them, so blend with last season
        # (the prior counts as prior_poss possessions / prior_games_pace games per team, like the team ratings)
        from dataclasses import replace as _r
        nt = max(len(prior.ortg.off), 1)
        wp, gp = float(d["poss"].sum()), float(len(d))
        ortg = _r(ortg, mu=(wp * ortg.mu + k * cfg.prior_poss * nt * prior.ortg.mu + (1 - k) * cfg.prior_poss * nt * prior.ortg.mu)
                  / (wp + cfg.prior_poss * nt))
        pace = _r(pace, mu=(gp * pace.mu + cfg.prior_games_pace * nt * prior.pace.mu) / (gp + cfg.prior_games_pace * nt))
    # ORtg ~ four factors (game level) -> weights used only to *explain* the projection
    X = np.column_stack([np.ones(len(tg))] + [tg[f] for f in FACTORS])
    beta, *_ = np.linalg.lstsq(X, tg["ortg"].to_numpy(float), rcond=None)
    return BasketballRatings(cfg.league, ortg, pace, factors, dict(zip(FACTORS, beta[1:])), through)


def project_game(br: BasketballRatings, cfg: BasketballConfig, game: dict) -> GameProjection:
    """game: game_id, date, away, home, neutral_site, optional home_b2b / away_b2b (0/1)."""
    h, a = game["home"], game["away"]
    neutral = bool(game.get("neutral_site", cfg.neutral_default))
    hs = 0 if neutral else 1
    hb, ab = int(game.get("home_b2b", 0) or 0), int(game.get("away_b2b", 0) or 0)
    pace = br.pace_matchup(h, a)
    r = br.ortg
    # neutral-court, fully-rested matchup ratings; situational terms broken out separately
    o_h = br.ortg_matchup(h, a, 0)
    o_a = br.ortg_matchup(a, h, 0)
    lg = r.mu
    parts = {}
    for side, o, d, ortg in (("home", h, a, o_h), ("away", a, h, o_a)):
        p = {}
        explained = 0.0
        for f in FACTORS:
            dev = br.factor_matchup(f, o, d) - br.factors[f].mu
            pts = br.factor_coef[f] * dev * pace / 100
            p[FACTOR_LABELS[f]] = pts
            explained += pts
        p["Other efficiency"] = (ortg - lg) * pace / 100 - explained
        parts[side] = p
    base = lg * pace / 100
    situ = {}
    if hs:
        situ["Home court"] = 2 * r.covariates.get("is_home", 0) * pace / 100
    rest = (hb - ab) * (r.covariates.get("b2b", 0) - r.covariates.get("opp_b2b", 0)) * pace / 100
    if hb or ab:
        situ["Rest"] = rest
    gp = GameProjection(cfg.league, str(game["game_id"]), str(game.get("date", "")), a, h, base,
                        parts["away"], parts["home"], situ,
                        extra={"pace": pace, "inputs": {"home": {"ortg": o_h}, "away": {"ortg": o_a}}})
    return gp



def scaled_prior(prior: "BasketballRatings", k: float, through: str) -> "BasketballRatings":
    """Ratings before a season's first game: last season's, kept at k (the rest regresses to average)."""
    from dataclasses import replace as _r
    sc = lambda a: _r(a, off=a.off * k, deff=a.deff * k)
    return BasketballRatings(prior.league, sc(prior.ortg), sc(prior.pace), {f: sc(r) for f, r in prior.factors.items()},
                             prior.factor_coef, through)


def project_game_sam(br: BasketballRatings, cfg: BasketballConfig, game: dict) -> GameProjection:
    """Each team scores (league + its offence + the opponent's defence) per 100 possessions at the matchup pace;
    home court and back-to-backs are shown separately."""
    h, a = game["home"], game["away"]
    neutral = bool(game.get("neutral_site", cfg.neutral_default))
    hb, ab = int(game.get("home_b2b", 0) or 0), int(game.get("away_b2b", 0) or 0)
    pace = br.pace_matchup(h, a)
    r = br.ortg
    base = r.mu * pace / 100
    eff = lambda o, d: (r.off.get(o, 0) + r.deff.get(d, 0)) * pace / 100
    situ = {}
    if not neutral:
        situ["Home court"] = cfg.home_court if cfg.home_court is not None else 2 * r.covariates.get("is_home", 0) * pace / 100
    if hb or ab:
        situ["Back-to-back"] = (hb - ab) * (r.covariates.get("b2b", 0) - r.covariates.get("opp_b2b", 0)) * pace / 100
    ih, ia = float(game.get("inj_home") or 0), float(game.get("inj_away") or 0)     # points per 100, from nba_injuries
    if ih or ia:
        situ["Injuries"] = (ih - ia) * pace / 100
    return GameProjection(cfg.league, str(game["game_id"]), str(game.get("date", "")), a, h, base,
                          {"Efficiency": eff(a, h)}, {"Efficiency": eff(h, a)}, situ,
                          extra={"pace": pace, "inputs": {"home": {"ortg": r.mu + r.off.get(h, 0) + r.deff.get(a, 0)},
                                                          "away": {"ortg": r.mu + r.off.get(a, 0) + r.deff.get(h, 0)}}})
