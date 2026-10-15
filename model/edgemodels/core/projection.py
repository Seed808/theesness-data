"""
Projection container shared by every sport.

A sport module produces, for each team in a game, a projected score built
as   baseline + sum(component points)   where each component is that
input's effect in points versus a league-average team. The margin
breakdown is then simply  home component - away component  for each input,
plus the situational terms (home field, rest, travel), so the bars on the
card always sum exactly to the projected margin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .edges import spread_edge, total_edge


@dataclass
class GameProjection:
    sport: str
    game_id: str
    date: str
    away: str
    home: str
    baseline: float                              # league-average team score in this game environment
    away_parts: dict = field(default_factory=dict)  # input -> points vs average
    home_parts: dict = field(default_factory=dict)
    situational: dict = field(default_factory=dict)  # input -> points to HOME margin (split half/half in score)
    extra: dict = field(default_factory=dict)        # ratings tables, QB rows, props, notes

    @property
    def proj_home(self) -> float:
        return self.baseline + sum(self.home_parts.values()) + sum(self.situational.values()) / 2

    @property
    def proj_away(self) -> float:
        return self.baseline + sum(self.away_parts.values()) - sum(self.situational.values()) / 2

    def margin_breakdown(self) -> dict:
        """Points toward the HOME team per input (negative = toward away). Sums to the margin."""
        keys = list(dict.fromkeys(list(self.home_parts) + list(self.away_parts)))
        out = {k: self.home_parts.get(k, 0) - self.away_parts.get(k, 0) for k in keys}
        out.update(self.situational)
        return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))

    def card(self, market_line: float | None, market_total: float | None,
             margin_sd: float | None = None, total_sd: float | None = None, margin_dist=None) -> dict:
        c = {"sport": self.sport, "game_id": self.game_id, "date": self.date,
             "away": self.away, "home": self.home,
             "proj_away": round(self.proj_away, 1), "proj_home": round(self.proj_home, 1),
             "model_total": round(self.proj_home + self.proj_away, 1),
             "market_total": market_total, "market_line": market_line,
             "breakdown": {k: round(v, 2) for k, v in self.margin_breakdown().items()}}
        c["model_line"] = round(-(self.proj_home - self.proj_away), 1)
        if market_line is not None:
            e = spread_edge(self.proj_home, self.proj_away, market_line, self.sport, margin_sd, margin_dist)
            c.update(model_line=e.model_line, edge=e.edge, pick=e.pick_side,
                     pick_line=e.pick_line, tier=e.tier, cover_prob=e.cover_prob,
                     push_prob=e.push_prob, lose_prob=e.lose_prob, ev=e.ev)
        if margin_dist is not None:
            c["fair_line"] = margin_dist.fair_line(self.proj_home - self.proj_away, margin_sd)
        if market_total is not None:
            c["total"] = total_edge(self.proj_home + self.proj_away, market_total, total_sd)
        c.update(self.extra)
        return c
