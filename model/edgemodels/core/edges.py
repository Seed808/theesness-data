"""
Market comparison. Lines use the home-team convention:
    home_line = -3.5  ->  home team favoured by 3.5
    model_line = -(proj_home - proj_away)
Edge = |model_line - market_line|, pointing to the side the model prefers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Edge flags per sport (points / goals / runs). Football follows the 3+ convention;
# the others are starting points to be re-tuned from your own graded record.
EDGE_TIERS = {
    "nfl": (3.0, 6.0, 10.0),
    "ncaaf": (3.0, 6.0, 10.0),
    "nba": (2.5, 5.0, 8.0),
    "cbb": (3.0, 6.0, 9.0),
    "mlb": (3.0, 5.0, 8.0),      # moneyline: percentage points of win probability over the no-vig market
    "nhl": (0.35, 0.7, 1.0),     # goals
}


def fmt_line(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if abs(x) < 1e-9:
        return "PK"
    s = f"{x:+.2f}".rstrip("0").rstrip(".")
    return s


@dataclass
class SpreadEdge:
    model_line: float          # home convention
    market_line: float         # home convention
    edge: float                # points, >= 0
    pick_side: str             # "home" | "away"
    pick_line: float           # line on the picked team, e.g. +9.5
    tier: int                  # 0 none, 1 = tier1+, 2, 3
    cover_prob: float | None   # P(pick covers) under the model's margin distribution
    push_prob: float | None = None
    lose_prob: float | None = None
    ev: float | None = None     # expected profit per unit at -110 (push returns stake)

    def as_dict(self):
        return self.__dict__.copy()


def spread_edge(proj_home: float, proj_away: float, market_home_line: float,
                sport: str, margin_sd: float | None = None, dist=None) -> SpreadEdge:
    """With `dist` (core.margins.MarginDist) the cover / push / lose probabilities come from
    real margin history, so a move across 3 or 7 counts for more than a move across 9 or 12,
    and the side is chosen by expected value instead of by which side of the line the model sits."""
    model_line = -(proj_home - proj_away)
    if dist is not None:
        m = proj_home - proj_away
        r = dist.side_ev(m, market_home_line, margin_sd)
        side = "home" if r["ev_home"] >= r["ev_away"] else "away"
        pick_line = market_home_line if side == "home" else -market_home_line
        edge = abs(market_home_line - model_line)
        tiers = EDGE_TIERS.get(sport, (3, 6, 10))
        return SpreadEdge(round(model_line, 1), market_home_line, round(edge, 1), side, pick_line,
                          sum(edge >= t for t in tiers), r[side], r["push"], r["away" if side == "home" else "home"],
                          r[f"ev_{side}"])
    diff = market_home_line - model_line          # >0 -> home getting more than model thinks
    side = "home" if diff > 0 else "away"
    pick_line = market_home_line if side == "home" else -market_home_line
    edge = abs(diff)
    tiers = EDGE_TIERS.get(sport, (3, 6, 10))
    tier = sum(edge >= t for t in tiers)
    cp = None
    if margin_sd:
        m = proj_home - proj_away if side == "home" else proj_away - proj_home
        cp = float(1 - norm.cdf(-pick_line, loc=m, scale=margin_sd))
    return SpreadEdge(round(model_line, 1), market_home_line, round(edge, 1), side, pick_line, tier, cp)


def total_edge(proj_total: float, market_total: float, total_sd: float | None = None) -> dict:
    side = "over" if proj_total > market_total else "under"
    out = {"model_total": round(proj_total, 1), "market_total": market_total,
           "edge": round(abs(proj_total - market_total), 1), "side": side}
    if total_sd:
        p_over = 1 - norm.cdf(market_total, proj_total, total_sd)
        out["prob"] = float(p_over if side == "over" else 1 - p_over)
    return out


# ---------- odds helpers (moneylines, props) ----------
def american_to_prob(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def no_vig(p_a: float, p_b: float) -> tuple[float, float]:
    s = p_a + p_b
    return p_a / s, p_b / s


def ev_per_unit(p_win: float, odds: float) -> float:
    """Expected profit per 1 unit risked at American odds."""
    payout = odds / 100 if odds > 0 else 100 / -odds
    return p_win * payout - (1 - p_win)


def margin_to_winprob(margin: float, sd: float) -> float:
    return float(norm.cdf(margin / sd))
