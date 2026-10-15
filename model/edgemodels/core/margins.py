"""
Key-number-aware margin distribution.

A bell curve treats every point the same. Real final margins pile up on a few
numbers (football: 3, 7, 10, 6, 14, 4 ...). This module learns how much more
(or less) often each final margin happens than a bell curve centred on the
expected margin would predict, from real game history, and uses that to
price spreads:

    P(margin = k | expected m)  is proportional to  Normal_k(m, sd) x w_|k|

    w_|k| = (observed games ending at +-k + a) / (bell-curve-expected games at +-k + a)

The expected margin for each historical game is the closing market spread,
so the weights describe the *shape* of real results around an expectation,
not anyone's model. Pseudo-count `a` shrinks rare margins toward w = 1.

From the pmf every spread question has an exact answer: P(cover), P(push),
P(lose) for either side at any line, the expected value at a price, and the
"fair" line where the two sides are even.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class MarginDist:
    weights: np.ndarray        # w[|k|] for |k| = 0..kmax
    sd: float                  # spread of results around the expectation it was fitted with
    kmax: int
    n_games: int
    source: str = ""

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, expected, actual, sd: float | None = None, kmax: int = 70, pseudo: float = 3.0,
            source: str = "") -> "MarginDist":
        expected = np.asarray(expected, float)
        actual = np.rint(np.asarray(actual, float)).astype(int)
        ok = np.isfinite(expected) & (np.abs(actual) <= kmax)
        expected, actual = expected[ok], actual[ok]
        sd = float(np.std(actual - expected)) if sd is None else sd
        ks = np.arange(-kmax, kmax + 1)
        # bell-curve expected count of games at each integer margin
        cdf_hi = norm.cdf((ks[None, :] + 0.5 - expected[:, None]) / sd)
        cdf_lo = norm.cdf((ks[None, :] - 0.5 - expected[:, None]) / sd)
        E = (cdf_hi - cdf_lo).sum(axis=0)
        O = np.bincount(actual + kmax, minlength=2 * kmax + 1).astype(float)
        # pool +k and -k: key numbers are the same for either team
        Eabs = E[kmax:].copy()
        Oabs = O[kmax:].copy()
        Eabs[1:] += E[:kmax][::-1]
        Oabs[1:] += O[:kmax][::-1]
        w = (Oabs + pseudo) / (Eabs + pseudo)
        return cls(w, sd, kmax, int(len(actual)), source)

    # ---------------------------------------------------------------- use
    def pmf(self, mean: float, sd: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Integer margins k (home - away) and their probabilities."""
        sd = sd or self.sd
        ks = np.arange(-self.kmax, self.kmax + 1)
        base = norm.cdf((ks + 0.5 - mean) / sd) - norm.cdf((ks - 0.5 - mean) / sd)
        p = base * self.weights[np.abs(ks)]
        return ks, p / p.sum()

    def center_for(self, projected: float, sd: float | None = None) -> float:
        key = (round(float(projected), 3), round(float(sd or self.sd), 3))
        cache = self.__dict__.setdefault("_centers", {})
        if key not in cache:
            cache[key] = self._center_for(*key)
        return cache[key]

    def _center_for(self, projected: float, sd: float | None = None) -> float:
        """Bell-curve centre whose key-number distribution has its MEDIAN at the projected margin.

        A market line is a median: at a closing line of -3.5 the favourite covered 54.3% of 94 games
        (2020-25) and 50.8% of 260 (2010-25). Centring the bell at the projection instead puts too much
        weight on the 3 below it (a -3.5 favourite would cover only 47.3%), which flips picks the wrong way.
        So the projection is treated like a line: half the outcomes above it, half below."""
        sd = sd or self.sd
        lo, hi = projected - 6, projected + 6
        for _ in range(30):
            c = (lo + hi) / 2
            ks, p = self.pmf(c, sd)
            # continuous (interpolated) CDF at the projection: each integer k spreads over [k-0.5, k+0.5]
            k = int(np.floor(projected + 0.5))
            below = p[ks < k].sum() + p[ks == k].sum() * (projected - (k - 0.5))
            above = 1 - below
            if above < 0.5:
                lo = c
            else:
                hi = c
        return (lo + hi) / 2

    def spread_probs(self, mean: float, home_line: float, sd: float | None = None) -> dict:
        """mean: projected home margin (treated as the median, see center_for).
        home_line in home convention (-3 = home favoured by 3).
        Home covers when margin + home_line > 0; push when it equals 0."""
        ks, p = self.pmf(self.center_for(mean, sd), sd)
        x = ks + home_line
        return {"home": float(p[x > 0].sum()), "push": float(p[np.isclose(x, 0)].sum()),
                "away": float(p[x < 0].sum())}

    def side_ev(self, mean: float, home_line: float, sd: float | None = None,
                home_odds: float = -110, away_odds: float = -110) -> dict:
        """Expected profit per unit risked on each side (push returns the stake)."""
        pr = self.spread_probs(mean, home_line, sd)
        def pay(o):
            return o / 100 if o > 0 else 100 / -o
        ev_h = pr["home"] * pay(home_odds) - pr["away"]
        ev_a = pr["away"] * pay(away_odds) - pr["home"]
        return {**pr, "ev_home": ev_h, "ev_away": ev_a}

    def fair_line(self, mean: float, sd: float | None = None) -> float:
        """Home line (to the half point) where P(home covers) and P(away covers) are closest."""
        ks, p = self.pmf(self.center_for(mean, sd), sd)
        best, best_gap = 0.0, 9e9
        for L in np.arange(-40, 40.5, 0.5):
            x = ks + L
            gap = abs(p[x > 0].sum() - p[x < 0].sum())
            if gap < best_gap:
                best, best_gap = float(L), gap
        return best

    def mean_from_line(self, home_line: float, sd: float | None = None) -> float:
        """Expected home margin implied by a market line (inverse of fair_line, to 0.1)."""
        grid = np.arange(-40, 40.05, 0.1)
        diffs = [abs(self.spread_probs(m, home_line, sd)["home"] - self.spread_probs(m, home_line, sd)["away"]) for m in grid]
        return float(grid[int(np.argmin(diffs))])

    def table(self, top: int = 12) -> list[dict]:
        """Margins that happen most often relative to a bell curve."""
        idx = np.argsort(-self.weights)[:top]
        return [{"margin": int(i), "weight": float(self.weights[i])} for i in idx]


def nfl_from_nflverse(games_csv, seasons: range, sd: float | None = None) -> MarginDist:
    """Fit on nflverse games.csv: expected home margin = spread_line (home favoured if > 0)."""
    g = games_csv[games_csv["season"].isin(list(seasons)) & games_csv["result"].notna() & games_csv["spread_line"].notna()]
    return MarginDist.fit(g["spread_line"], g["result"], sd=sd,
                          source=f"nflverse games {min(seasons)}-{max(seasons)}, {len(g)} games, closing spread as expectation")
