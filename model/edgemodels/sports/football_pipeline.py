"""
End-to-end football runs: weekly board (projections + edges) and an
out-of-sample backtest that only ever uses data available before kickoff.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import football as fb


def _week_key(df):
    return df["season"] * 100 + df["week"]


class FootballModel:
    """Holds a league's standardized plays across seasons and builds any week."""

    def __init__(self, league: str, plays_by_season: dict[int, pd.DataFrame], cfg: fb.FootballConfig | None = None,
                 margin_dists: dict | None = None):
        """margin_dists: {season: MarginDist} fitted only on games before that season
        (key-number pricing); missing season -> bell curve."""
        self.margin_dists = margin_dists or {}
        self.league = league
        self.cfg = cfg or (fb.NFL if league == "nfl" else fb.NCAAF)
        self.plays = {s: p for s, p in plays_by_season.items()}
        self.tg = {s: fb.team_games(p) for s, p in self.plays.items()}
        self._season_ratings: dict[int, fb.FootballRatings] = {}

    # full-season ratings of a past season, used as the next season's prior
    def season_ratings(self, season: int) -> fb.FootballRatings | None:
        if season not in self.tg:
            return None
        if season not in self._season_ratings:
            self._season_ratings[season] = fb.fit_ratings(self.tg[season], self.cfg,
                                                          prior=self.season_ratings(season - 1)
                                                          if season - 1 in self.tg else None,
                                                          through=f"{season} full")
        return self._season_ratings[season]

    def build(self, season: int, before_week: int):
        """Ratings, points model, home field and QB table using only data before `before_week`."""
        tg = self.tg[season]
        tg_now = tg[tg["week"] < before_week]
        prior = self.season_ratings(season - 1)
        if tg_now.empty:
            # Week 1: no games yet this season -> last season's ratings at prior strength
            if prior is None:
                raise ValueError(f"{season} week {before_week}: no current games and no prior season to start from")
            fr = fb.FootballRatings(prior.league, {
                m: replace(r, off=r.off * self.cfg.prior_keep, deff=r.deff * self.cfg.prior_keep)
                for m, r in prior.ratings.items()}, f"{season - 1} carried into {season}", prior.lower)
        else:
            fr = fb.fit_ratings(tg_now, self.cfg, prior=prior, through=f"{season} wk{before_week - 1}")
        hist = pd.concat([self.tg[s] for s in self.tg if s < season] + [tg_now])
        pm = fb.PointsModel.fit(hist)
        cfg = self.cfg
        if cfg.home_field is None:
            hfa, n = fb.fit_home_field(pd.concat([self.tg[s] for s in self.tg if s < season] + [tg_now]))
            cfg = replace(cfg, home_field=float(hfa))
        plays_now = self.plays[season][self.plays[season]["week"] < before_week]
        qbs = fb.qb_ratings(plays_now, fr, cfg, prior_plays=self.plays.get(season - 1))
        if plays_now.empty and season - 1 in self.plays:
            plays_now = self.plays[season - 1]      # week 1: team QB mix and usage come from last season
        return fr, pm, cfg, qbs, plays_now

    def project_slate(self, season: int, week: int, slate: pd.DataFrame, margin_sd=None, total_sd=None,
                      built=None) -> list[dict]:
        """slate columns: game_id, date, away, home, neutral_site, spread_home, total,
        optional away_qb_id / home_qb_id (starter ids)."""
        fr, pm, cfg, qbs, plays_now = built or self.build(season, week)
        cards = []
        for g in slate.to_dict("records"):
            qa = {}
            qb_rows = []
            for side in ("away", "home"):
                team = g[side]
                sid = g.get(f"{side}_qb_id")
                sid = sid if isinstance(sid, str) and sid in qbs.index else None
                a = fb.qb_adjustment(team, sid, plays_now, qbs)
                if isinstance(g.get(f"{side}_qb_name"), str) and sid is None and g.get(f"{side}_qb_id") is not None:
                    a["starter"] = g[f"{side}_qb_name"] + " (no snaps in data)"
                    a["adj"] = qbs.attrs["replacement"] - a["in_numbers_epa_db"]
                qa[team] = a["adj"]
                qb_rows.append(a)
            gp = (fb.project_game_sam(fr, cfg, g, qa) if cfg.engine == "sam" else fb.project_game(fr, pm, cfg, g, qa))
            card = gp.card(_num(g.get("spread_home")), _num(g.get("total")),
                           margin_sd or cfg.margin_sd, total_sd or cfg.total_sd, self.margin_dists.get(season))
            card["qb"] = qb_rows
            card["week"] = week
            card["rest"] = {"away": g.get("away_rest"), "home": g.get("home_rest")}
            card["kickoff"] = g.get("kickoff", "")
            card["venue"] = g.get("venue", "")
            card["records"] = g.get("records", {})
            card["ratings"] = _rating_rows(fr, g["away"], g["home"])
            cards.append(card)
        return cards

    def backtest(self, season: int, weeks: list[int]) -> pd.DataFrame:
        rows = []
        for w in weeks:
            slate = slate_from_plays(self.tg[season], w)
            if slate.empty:
                continue
            built = self.build(season, w)
            for c in self.project_slate(season, w, slate, built=built):
                g = slate.set_index("game_id").loc[c["game_id"]]
                rows.append({"game_id": c["game_id"], "week": w, "away": c["away"], "home": c["home"],
                             "proj_away": c["proj_away"], "proj_home": c["proj_home"],
                             "model_line": c["model_line"], "market_line": c.get("market_line"),
                             "edge": c.get("edge"), "pick": c.get("pick"), "pick_line": c.get("pick_line"),
                             "cover_prob": c.get("cover_prob"), "push_prob": c.get("push_prob"), "ev": c.get("ev"),
                             "model_total": c["model_total"], "market_total": c.get("market_total"),
                             "away_score": g["away_score"], "home_score": g["home_score"]})
        bt = pd.DataFrame(rows)
        if bt.empty:
            return bt
        bt["margin"] = bt["home_score"] - bt["away_score"]
        bt["proj_margin"] = bt["proj_home"] - bt["proj_away"]
        cover = np.where(bt["pick"] == "home", bt["margin"], -bt["margin"]) + bt["pick_line"]
        bt["result"] = np.select([cover > 0, cover < 0], ["WIN", "LOSS"], "PUSH")
        bt.loc[bt["market_line"].isna(), "result"] = None
        return bt


def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def slate_from_plays(tg: pd.DataFrame, week: int) -> pd.DataFrame:
    g = tg[(tg["week"] == week) & (tg["off"] == tg["home"])]
    return pd.DataFrame({"game_id": g["game_id"], "date": g["date"].astype(str), "away": g["away"],
                         "home": g["home"], "neutral_site": g["neutral_site"],
                         "spread_home": g["spread_home"], "total": g["total"],
                         "away_score": g["away_score"], "home_score": g["home_score"]}).reset_index(drop=True)


def nfl_slate(games_csv: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    g = games_csv[(games_csv["season"] == season) & (games_csv["week"] == week)].copy()
    return pd.DataFrame({
        "game_id": g["game_id"], "date": g["gameday"], "kickoff": g["gametime"], "venue": g["stadium"],
        "away": g["away_team"], "home": g["home_team"], "neutral_site": g["location"].eq("Neutral"),
        "spread_home": -g["spread_line"], "total": g["total_line"],
        "away_qb_id": g["away_qb_id"], "home_qb_id": g["home_qb_id"],
        "away_qb_name": g["away_qb_name"], "home_qb_name": g["home_qb_name"],
        "away_rest": g["away_rest"], "home_rest": g["home_rest"],
        "away_score": g["away_score"], "home_score": g["home_score"],
    }).reset_index(drop=True)


RATING_LABELS = [("pass_epa", "Pass EPA / dropback", "Passing"), ("rush_epa", "Rush EPA / carry", "Rushing"),
                 ("pass_sr", "Pass success rate", "Pass cons."), ("rush_sr", "Rush success rate", "Rush cons."),
                 ("pass_rate", "Pass rate (neutral)", "Pass rate"), ("plays", "Plays per game", "Pace")]


def _rating_rows(fr: fb.FootballRatings, away: str, home: str) -> dict:
    t = fr.table()
    rows = []
    for m, label, short in RATING_LABELS:
        rows.append({"metric": m, "label": label, "short": short,
                     "away_off": float(t.at[away, (m, "off")]) if away in t.index else None,
                     "home_off": float(t.at[home, (m, "off")]) if home in t.index else None,
                     "away_def": float(t.at[away, (m, "def")]) if away in t.index else None,
                     "home_def": float(t.at[home, (m, "def")]) if home in t.index else None,
                     "away_off_pct": _i(t, away, (m, "off_pct")), "home_off_pct": _i(t, home, (m, "off_pct")),
                     "away_def_pct": _i(t, away, (m, "def_pct")), "home_def_pct": _i(t, home, (m, "def_pct"))})
    return {"rows": rows, "through": fr.through}


def _i(t, team, col):
    try:
        v = t.at[team, col]
        return None if pd.isna(v) else int(v)
    except KeyError:
        return None
