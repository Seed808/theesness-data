"""
MLB player props: plate-appearance outcome simulation.

Batter (hits, total bases, HR, RBI):
  PA count by lineup spot (empirical), each PA drawn from a multinomial over
  1B/2B/3B/HR/BB/K/OUT. The batter's rates are shrunk to league, combined with
  the opposing starter's allowed rates by log5 (odds-ratio), then with the
  bullpen's rates for PAs after the starter's projected exit. HR is park-scaled.
  RBI per outcome is fitted from Statcast score changes (runs on the PA).

Pitcher (strikeouts, hits allowed, outs):
  Batters faced ~ his recent BF distribution (shrunk); each BF is a draw against
  the opposing lineup's combined rates (log5 again).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.sim import RNG, prop_summary

OUTCOMES = ["1B", "2B", "3B", "HR", "BB", "K", "OUT"]
TB = np.array([1, 2, 3, 4, 0, 0, 0])
HIT = np.array([1, 1, 1, 1, 0, 0, 0])
N_SIMS = 20000


def _rates(df: pd.DataFrame, by: str, prior: pd.Series, n0: float) -> pd.DataFrame:
    c = pd.crosstab(df[by], df["outcome"]).reindex(columns=OUTCOMES, fill_value=0)
    n = c.sum(axis=1)
    return (c.add(prior * n0, axis=1)).div(n + n0, axis=0)


def log5(b: np.ndarray, p: np.ndarray, lg: np.ndarray) -> np.ndarray:
    """Odds-ratio combination of batter and pitcher outcome rates (rows sum to 1)."""
    x = b * p / lg
    return x / x.sum(axis=-1, keepdims=True)


class BaseballProps:
    def __init__(self, pa: pd.DataFrame, park: pd.Series | None = None, n0_bat=150, n0_pit=200):
        self.pa = pa
        self.lg = pa["outcome"].value_counts(normalize=True).reindex(OUTCOMES).fillna(0)
        self.bat = _rates(pa, "batter", self.lg, n0_bat)
        self.pit = _rates(pa, "pitcher", self.lg, n0_pit)
        pen = pa[pa["is_sp"] == 0]
        self.pen = _rates(pen, "fld_team", self.lg, 400)
        self.team_bat = _rates(pa, "bat_team", self.lg, 400)
        self.park = park if park is not None else pd.Series(dtype=float)
        # PA per game by lineup spot
        pg = pa.groupby(["game_pk", "batter", "spot"]).size().reset_index(name="n")
        self.pa_by_spot = pg.groupby("spot")["n"].apply(lambda s: s.value_counts(normalize=True).sort_index())
        # runs credited per outcome (proxy for RBI)
        self.rbi_mean = pa.groupby("outcome")["runs_on_pa"].mean().reindex(OUTCOMES).fillna(0).to_numpy()
        starts = pa[pa["is_sp"] == 1].groupby(["game_pk", "pitcher"]).size()
        self.bf_starts = starts
        self.lg_bf = float(starts.mean())
        # share of each lineup spot's PAs that come against the starter (league, from data)
        self.sp_share_by_spot = pa.groupby("spot")["is_sp"].mean()

    def _pa_draw(self, spot: int, n: int) -> np.ndarray:
        dist = self.pa_by_spot.loc[spot] if spot in self.pa_by_spot.index.get_level_values(0) else self.pa_by_spot.loc[5]
        return RNG.choice(dist.index.to_numpy(), size=n, p=dist.to_numpy())

    def _bf_draw(self, pitcher, n: int) -> np.ndarray:
        own = self.bf_starts.xs(pitcher, level=1).to_numpy() if pitcher in self.bf_starts.index.get_level_values(1) else np.array([])
        pool = np.concatenate([own, RNG.choice(self.bf_starts.to_numpy(), size=5)])
        return RNG.choice(pool, size=n)

    def batter(self, batter, spot: int, opp_sp, opp_team: str, park_team: str, name: str = "") -> dict:
        lg = self.lg.to_numpy()
        b = self.bat.loc[batter].to_numpy() if batter in self.bat.index else lg
        r_sp = self.pit.loc[opp_sp].to_numpy() if opp_sp in self.pit.index else lg
        p_pen = self.pen.loc[opp_team].to_numpy() if opp_team in self.pen.index else lg
        vs_sp, vs_pen = log5(b, r_sp, lg), log5(b, p_pen, lg)
        pf = float(self.park.get(park_team, 1.0))
        for r in (vs_sp, vs_pen):                      # park scales HR, renormalise
            r[3] *= pf
            r /= r.sum()
        n_pa = self._pa_draw(spot, N_SIMS)
        # PAs against the starter: league share for this lineup spot, scaled by how deep this starter goes
        sp_bf = float(self._bf_draw(opp_sp, 2000).mean()) if opp_sp is not None else self.lg_bf
        p_sp = float(np.clip(self.sp_share_by_spot.get(spot, self.sp_share_by_spot.mean()) * sp_bf / self.lg_bf, 0, 1))
        sp_pa = RNG.binomial(n_pa, p_sp)
        cnt = RNG.multinomial(1, vs_sp, size=(N_SIMS, n_pa.max()))
        cnt2 = RNG.multinomial(1, vs_pen, size=(N_SIMS, n_pa.max()))
        idx = np.arange(n_pa.max())
        use_sp = (idx[None, :] < sp_pa[:, None])[..., None]
        live = (idx[None, :] < n_pa[:, None])[..., None]
        outc = np.where(use_sp, cnt, cnt2) * live
        tot = outc.sum(axis=1)                                          # sims x outcomes
        hits, tb, hr = tot @ HIT, tot @ TB, tot[:, 3]
        rbi = RNG.poisson(tot @ self.rbi_mean)
        return {"player": name or str(batter), "spot": spot,
                "hits": prop_summary(hits, bins=np.arange(-0.5, 6.5), label="Hits"),
                "total_bases": prop_summary(tb, bins=np.arange(-0.5, 12.5), label="Total bases"),
                "home_run": float((hr > 0).mean()), "rbi": prop_summary(rbi, bins=np.arange(-0.5, 8.5), label="RBI"),
                "hits_runs_rbis_note": "RBI = runs scoring on the PA (fitted per outcome); errors and runners' advancement are included"}

    def pitcher(self, pitcher, opp_team: str, name: str = "") -> dict:
        lg = self.lg.to_numpy()
        p = self.pit.loc[pitcher].to_numpy() if pitcher in self.pit.index else lg
        t = self.team_bat.loc[opp_team].to_numpy() if opp_team in self.team_bat.index else lg
        r = log5(t, p, lg)
        bf = self._bf_draw(pitcher, N_SIMS).astype(int)
        draws = RNG.multinomial(bf, r)        # vectorised: one multinomial per sim
        k = draws[:, 5]
        h = draws[:, :4].sum(axis=1)
        return {"player": name or str(pitcher), "proj_bf": float(bf.mean()),
                "strikeouts": prop_summary(k, bins=np.arange(-0.5, 15.5), label="Strikeouts"),
                "hits_allowed": prop_summary(h, bins=np.arange(-0.5, 13.5), label="Hits allowed"),
                "k_rate_vs_lineup": float(r[5])}
