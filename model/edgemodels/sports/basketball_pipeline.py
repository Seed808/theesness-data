"""Basketball runs: daily slate projection and a rolling out-of-sample backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import basketball as bb


class BasketballModel:
    def __init__(self, league: str, tg_by_season: dict[int, pd.DataFrame], cfg: bb.BasketballConfig | None = None):
        self.league = league
        self.cfg = cfg or (bb.NBA_SAM if league == "nba" else bb.CBB)
        self.tg = tg_by_season
        self._full: dict[int, bb.BasketballRatings] = {}

    def season_ratings(self, season):
        if season not in self.tg:
            return None
        if season not in self._full:
            t = self.tg[season]
            self._full[season] = bb.fit_ratings(t[t["season_type"] == 2], self.cfg,
                                                self.season_ratings(season - 1), f"{season} full")
        return self._full[season]

    def build(self, season, before_date):
        t = self.tg.get(season, pd.DataFrame())
        through = f"through {pd.Timestamp(before_date).date() - pd.Timedelta(days=1)}"
        if len(t):
            t = t[t["date"] < pd.Timestamp(before_date)]
        if len(t) == 0:                       # no games yet this season: last season, regressed
            return bb.scaled_prior(self.season_ratings(season - 1), self.cfg.prior_keep,
                                   f"{season - 1} season, {int(100 * self.cfg.prior_keep)}% kept (no {season} games yet)")
        return bb.fit_ratings(t, self.cfg, self.season_ratings(season - 1), through)

    def project_slate(self, br, slate: pd.DataFrame, margin_sd=None, total_sd=None) -> list[dict]:
        cards = []
        for g in slate.to_dict("records"):
            gp = bb.project_game_sam(br, self.cfg, g) if self.cfg.engine == "sam" else bb.project_game(br, self.cfg, g)
            ml = g.get("spread_home")
            ml = None if ml is None or pd.isna(ml) else float(ml)
            tot = g.get("total")
            tot = None if tot is None or pd.isna(tot) else float(tot)
            c = gp.card(ml, tot, margin_sd or self.cfg.margin_sd, total_sd or self.cfg.total_sd)
            c["ratings"] = rating_rows(br, g["away"], g["home"])
            cards.append(c)
        return cards

    def backtest(self, season, start_date, min_games=8) -> pd.DataFrame:
        t = self.tg[season]
        games = t[t["is_home"] == 1].sort_values("date")
        games = games[games["date"] >= pd.Timestamp(start_date)]
        rows = []
        for d, day in games.groupby("date"):
            br = self.build(season, d)
            slate = pd.DataFrame({"game_id": day["game_id"], "date": d, "away": day["away"], "home": day["home"],
                                  "neutral_site": False, "home_b2b": day["b2b"], "away_b2b": day["opp_b2b"],
                                  "spread_home": day["market_home_line"]})
            for c, (_, g) in zip(self.project_slate(br, slate), day.iterrows()):
                rows.append({"game_id": c["game_id"], "date": d, "away": c["away"], "home": c["home"],
                             "proj_home": c["proj_home"], "proj_away": c["proj_away"],
                             "model_line": c["model_line"], "market_line": c.get("market_line"),
                             "edge": c.get("edge"), "pick": c.get("pick"), "pick_line": c.get("pick_line"),
                             "model_total": c["model_total"],
                             "home_score": g["pts"], "away_score": g["opp_pts"]})
        bt = pd.DataFrame(rows)
        bt["margin"] = bt["home_score"] - bt["away_score"]
        bt["proj_margin"] = bt["proj_home"] - bt["proj_away"]
        cover = np.where(bt["pick"] == "home", bt["margin"], -bt["margin"]) + bt["pick_line"].astype(float)
        bt["result"] = np.where(bt["market_line"].isna(), None, np.select([cover > 0, cover < 0], ["WIN", "LOSS"], "PUSH"))
        return bt


LABELS = [("ortg", "Points / 100 poss", "Efficiency"), ("efg", "eFG%", "Shooting"), ("tov", "Turnover rate", "Ball security"),
          ("orb", "Off. rebound rate", "Off. boards"), ("ftr", "FT rate (FTM/FGA)", "Free throws")]


def rating_rows(br, away, home):
    t = br.table()
    rows = []
    for m, label, short in LABELS:
        def v(team, k):
            try:
                x = t.at[team, (m, k)]
                return None if pd.isna(x) else (int(x) if k.endswith("pct") else float(x))
            except KeyError:
                return None
        rows.append({"metric": m, "label": label, "short": short,
                     **{f"{s}_{k}": v(team, k) for s, team in (("away", away), ("home", home))
                        for k in ("off", "def", "off_pct", "def_pct")}})
    rows.append({"metric": "pace", "label": "Possessions / game", "short": "Pace",
                 "away_off": float(br.pace.off.get(away, np.nan)), "home_off": float(br.pace.off.get(home, np.nan)),
                 "away_def": None, "home_def": None,
                 "away_off_pct": _p(t, away), "home_off_pct": _p(t, home), "away_def_pct": None, "home_def_pct": None})
    return {"rows": rows, "through": br.through}


def _p(t, team):
    try:
        return int(t.at[team, ("pace", "off_pct")])
    except (KeyError, ValueError, TypeError):
        return None
