"""
Monte Carlo helpers shared by every prop model.

Each prop model returns a vector of simulated outcomes; `prop_summary`
turns that into the card the user sees: model mean/median, P(over),
P(under), fair odds, edge vs the posted line and price, and a histogram.
"""
from __future__ import annotations

import numpy as np

from .edges import american_to_prob, ev_per_unit, prob_to_american

RNG = np.random.default_rng(20261012)


def negbin(mean: np.ndarray | float, dispersion: float, size: int, rng=RNG) -> np.ndarray:
    """Counts with variance = mean + mean^2/dispersion (dispersion -> inf gives Poisson)."""
    mean = np.broadcast_to(np.asarray(mean, float), (size,))
    if not np.isfinite(dispersion):
        return rng.poisson(mean)
    p = dispersion / (dispersion + mean)
    return rng.negative_binomial(dispersion, p)


def compound_sum(counts: np.ndarray, pool: np.ndarray, scale: float = 1.0, rng=RNG) -> np.ndarray:
    """Sum of `counts[i]` draws from an empirical per-event pool (bootstrap), scaled.
    Used for yards: carries x yards-per-carry draws, catches x yards-per-catch draws."""
    total = counts.sum()
    if total == 0 or len(pool) == 0:
        return np.zeros_like(counts, dtype=float)
    draws = rng.choice(pool, size=int(total), replace=True) * scale
    idx = np.repeat(np.arange(len(counts)), counts)
    return np.bincount(idx, weights=draws, minlength=len(counts))


def prop_summary(sims: np.ndarray, line: float | None = None, over_odds: float = -110,
                 under_odds: float = -110, bins: int | list | None = None, label: str = "") -> dict:
    sims = np.asarray(sims, float)
    out = {"label": label, "n_sims": int(sims.size), "mean": float(sims.mean()),
           "median": float(np.median(sims)), "p10": float(np.percentile(sims, 10)),
           "p90": float(np.percentile(sims, 90))}
    if line is not None:
        p_over = float((sims > line).mean())
        p_push = float((sims == line).mean())
        p_under = 1 - p_over - p_push
        # judge each side conditional on no push
        po, pu = (p_over / (1 - p_push), p_under / (1 - p_push)) if p_push < 1 else (0, 0)
        ev_o, ev_u = ev_per_unit(po, over_odds), ev_per_unit(pu, under_odds)
        side = "over" if ev_o > ev_u else "under"
        out.update({
            "line": line, "p_over": p_over, "p_under": p_under, "p_push": p_push,
            "fair_over": prob_to_american(po), "fair_under": prob_to_american(pu),
            "market_over_prob": american_to_prob(over_odds), "market_under_prob": american_to_prob(under_odds),
            "side": side, "ev": max(ev_o, ev_u),
            "edge_units": out["mean"] - line,
        })
    # alt-line ladder around the median, so a card is useful even before a market line is posted
    sd = float(sims.std()) or 1.0
    step = max(1, round(sd / 2))          # whole-number steps keep every line on a .5
    mid = np.floor(np.median(sims)) + 0.5
    lines = [mid + k * step for k in range(-2, 3)]
    lines = [L for L in lines if L > 0]
    out["ladder"] = [{"line": float(L), "p_over": float((sims > L).mean())} for L in lines]
    if bins is None:
        lo, hi = np.percentile(sims, [1, 99])
        bins = np.linspace(np.floor(lo), np.ceil(hi) + 1, 16) if hi > lo else [lo - .5, lo + .5]
    h, e = np.histogram(sims, bins=bins)
    out["hist"] = {"edges": [float(x) for x in e], "share": (h / sims.size).round(4).tolist()}
    return out


def binary_summary(p_yes: float, odds: float | None = None, label: str = "") -> dict:
    """For yes/no props (anytime TD, first TD, first basket, HR)."""
    out = {"label": label, "p_yes": p_yes, "fair_odds": prob_to_american(p_yes)}
    if odds is not None:
        out.update({"market_odds": odds, "market_prob": american_to_prob(odds),
                    "ev": ev_per_unit(p_yes, odds)})
    return out
