"""
NBA player projections for prop markets (points, rebounds, assists, threes and their combinations).

Inputs: ESPN player box scores (sportsdataverse espn_nba_player_boxscores, ESPN athlete ids = the prop feed's ids).
For each player:
  minutes  : recent games weighted 0.8^k (newest first), blended with last season's average (worth 2 games)
  rates    : per-minute rate this season, shrunk toward last season's rate (worth K_PRIOR minutes; league rate if new)
  pace     : the matchup's projected possessions / league average (from the team ratings)
Simulation: minutes ~ Normal(projection, player's own spread); each stat ~ negative binomial around
rate x minutes x pace (dispersion per stat, fitted on 2025-26: see calibrate()). Combos are summed draw by draw,
so a short-minutes night lowers every stat together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
STATS = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "three_point_field_goals_made"}
COMBOS = {"pra": ("points", "rebounds", "assists"), "pr": ("points", "rebounds"), "pa": ("points", "assists"),
          "ar": ("assists", "rebounds")}
LABEL = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "threes": "Threes",
         "pra": "Pts + Reb + Ast", "pr": "Pts + Reb", "pa": "Pts + Ast", "ar": "Ast + Reb"}
# variance = mean * PHI (negative binomial); fitted by calibrate() on 2025-26 games (see player_impact_meta-style note)
PHI = {"points": 2.6, "rebounds": 1.3, "assists": 1.3, "threes": 1.01}
# level correction from the same check (projections ran ~3% low on points/rebounds/assists, ~5% high on threes)
LEVEL = {"points": 1.03, "rebounds": 1.03, "assists": 1.03, "threes": 0.95}
K_PRIOR = 240.0           # minutes of evidence last season's per-minute rate is worth
N_SIMS = 10000


def played(box: pd.DataFrame) -> pd.DataFrame:
    b = box[(box["minutes"].fillna(0) > 0) & (~box["did_not_play"].fillna(False).astype(bool))].copy()
    b["athlete_id"] = b["athlete_id"].astype(str)
    b["game_date"] = pd.to_datetime(b["game_date"])
    return b.sort_values("game_date")


class NBAPlayers:
    def __init__(self, last: pd.DataFrame, cur: pd.DataFrame, n_sims: int = N_SIMS):
        self.last, self.cur = played(last), played(cur)
        self.n = n_sims
        both = pd.concat([self.last, self.cur])
        self.lg_rate = {k: float(both[c].sum() / both["minutes"].sum()) for k, c in STATS.items()}
        g = self.last.groupby("athlete_id")
        self.prior_rate = {k: (g[c].sum() / g["minutes"].sum()) for k, c in STATS.items()}
        self.prior_min = g["minutes"].sum()
        self.prior_avg_min = g["minutes"].mean()

    def history(self, pid: str, before=None) -> pd.DataFrame:
        h = self.cur[self.cur["athlete_id"] == str(pid)]
        if before is not None:
            h = h[h["game_date"] < pd.Timestamp(before)]
        return h

    def project(self, pid: str, before=None, pace_factor: float = 1.0) -> dict | None:
        pid = str(pid)
        h = self.history(pid, before)
        last_min = float(self.prior_avg_min.get(pid, np.nan))
        if h.empty and np.isnan(last_min):
            return None
        # minutes
        mins = h["minutes"].to_numpy(float)[::-1][:10]
        w = 0.8 ** np.arange(len(mins))
        num, den = float((mins * w).sum()), float(w.sum())
        if not np.isnan(last_min):
            num, den = num + 2 * last_min, den + 2
        mu_min = num / den
        sd_min = float(np.clip(np.std(mins) if len(mins) >= 3 else 7.0, 4.5, 10.0))
        # rates
        rates = {}
        for k, c in STATS.items():
            pr = self.prior_rate[k].get(pid, np.nan)
            kp = K_PRIOR if not np.isnan(pr) else K_PRIOR / 2
            pr = self.lg_rate[k] if np.isnan(pr) else float(pr)
            rates[k] = (float(h[c].sum()) + kp * pr) / (float(h["minutes"].sum()) + kp)
        return {"min": mu_min, "sd_min": sd_min, "rates": rates, "pace": pace_factor, "games": int(len(h)),
                "last_season_min": None if np.isnan(last_min) else last_min}

    def simulate(self, proj: dict) -> dict:
        m = np.clip(RNG.normal(proj["min"], proj["sd_min"], self.n), 0, 48)
        out = {}
        for k in STATS:
            mean = np.maximum(proj["rates"][k] * m * proj["pace"] * LEVEL[k], 1e-6)
            phi = PHI[k]
            # negative binomial with variance = mean * phi  ->  n = mean / (phi - 1), p = 1 / phi
            n = mean / (phi - 1)
            out[k] = RNG.negative_binomial(n, 1 / phi)
        for k, parts in COMBOS.items():
            out[k] = sum(out[p] for p in parts)
        return out


def calibrate(last: pd.DataFrame, cur: pd.DataFrame, sample: int = 3000) -> dict:
    """Probability-integral check on 2025-26 games after each player's 10th: share of actuals under the
    projected median (ideal 50%) and inside the 10-90% band (ideal 80%), per stat."""
    P = NBAPlayers(last, cur, n_sims=4000)
    c = P.cur.groupby("athlete_id").cumcount()
    rows = P.cur[(c >= 10) & (P.cur["minutes"] > 0)].sample(min(sample, int(((c >= 10) & (P.cur["minutes"] > 0)).sum())),
                                                              random_state=1)
    res = {k: [] for k in list(STATS) + list(COMBOS)}
    for r in rows.itertuples():
        pr = P.project(r.athlete_id, before=r.game_date)
        if pr is None:
            continue
        s = P.simulate(pr)
        act = {k: getattr(r, c) for k, c in STATS.items()}
        for k, parts in COMBOS.items():
            act[k] = sum(act[p] for p in parts)
        for k in res:
            res[k].append(((s[k] < act[k]).mean() + 0.5 * (s[k] == act[k]).mean()))
    return {k: {"under_median": float(np.mean(np.array(v) < 0.5)), "in_10_90": float(np.mean((np.array(v) > 0.1) & (np.array(v) < 0.9))),
                "n": len(v)} for k, v in res.items()}
