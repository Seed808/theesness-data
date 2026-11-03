"""
NHL model on real play-by-play (scripts/fetch_nhl.py -> data/nhl).

  xG          logistic regression on unblocked shots (distance, angle, rebound, shot type, power play / short-handed),
              fitted on last season, empty-net shots left out
  ratings     each team's xG for and against per game, opponent-adjusted with a home-ice term, solved on this season
              with last season (kept at PRIOR_KEEP) as the prior worth PRIOR_GAMES games
  goalies     goals saved above expected per shot on goal, last season counted at half, shrunk with 800 shots of zero
  starter     each team's expected goalie = share of its last 10 starts (a back-to-back shifts weight to the backup)
  goals       (xG matchup x league goals-per-xG) - opponent goalie GSAx x projected shots, plus the league's average
              empty-net and shootout goals
  markets     Poisson simulation with overtime -> win chance, puck line (+/-1.5), total
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.edges import american_to_prob, ev_per_unit, no_vig, prob_to_american
from ..core.projection import GameProjection
from ..core.ratings import solve_adjusted

PRIOR_KEEP = 0.6
PRIOR_GAMES = 10.0
GOALIE_PRIOR_SHOTS = 800.0
N_SIMS = 40000
RNG = np.random.default_rng(11)


def _X(s: pd.DataFrame, cols=None) -> pd.DataFrame:
    X = pd.DataFrame({"dist": s["dist"], "dist2": s["dist"] ** 2, "angle": s["angle"], "rebound": s["rebound"],
                      "pp": (s["skaters"] > s["opp_skaters"]).astype(int), "sh": (s["skaters"] < s["opp_skaters"]).astype(int)})
    X = pd.concat([X, pd.get_dummies(s["shot_type"].fillna("unknown"), prefix="st").astype(int)], axis=1)
    return X.reindex(columns=cols, fill_value=0) if cols is not None else X


class NHLModel:
    def __init__(self, shots_last: pd.DataFrame, games_last: pd.DataFrame, shots_cur: pd.DataFrame, games_cur: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression
        fit = shots_last[shots_last["empty_net"] == 0]
        X = _X(fit)
        self.cols = list(X.columns)
        self.xgm = LogisticRegression(max_iter=3000).fit(X, fit["goal"])
        self.last = self._prep(shots_last, games_last)
        self.cur = self._prep(shots_cur, games_cur)
        self.games_last, self.games_cur = games_last, games_cur
        # league conversion and the goals the xG model never sees (empty nets, shootout winners), this season
        # weighted fully and last season counted as LAST_GAMES_EQUIV games, so a higher-scoring season shows up
        LAST_GAMES_EQUIV = 300.0
        def tallies(sh, gm):
            ne = sh[sh["empty_net"] == 0]
            return float(ne["goal"].sum()), float(ne["xg"].sum()), float((gm["home_score"] + gm["away_score"]).sum()), len(gm)
        gl_, xl_, fl_, nl_ = tallies(self.last, games_last)
        gc_, xc_, fc_, nc_ = tallies(self.cur, games_cur) if len(self.cur) else (0.0, 0.0, 0.0, 0)
        wl = LAST_GAMES_EQUIV / max(nl_, 1)
        self.goals_per_xg = (gc_ + wl * gl_) / (xc_ + wl * xl_)
        self.extra_goals = ((fc_ - gc_) + wl * (fl_ - gl_)) / (nc_ + wl * nl_) / 2

    def _prep(self, s: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        if s is None or s.empty:
            return pd.DataFrame(columns=["game_id", "team", "opp", "is_home", "xg", "goal", "on_goal", "goalie", "empty_net", "date"])
        s = s.copy()
        s["xg"] = self.xgm.predict_proba(_X(s, self.cols))[:, 1]
        s.loc[s["empty_net"] == 1, "xg"] = 0.0
        d = g.set_index("game_id")["date"]
        s["date"] = pd.to_datetime(s["game_id"].map(d))
        return s

    @staticmethod
    def _team_games(s: pd.DataFrame) -> pd.DataFrame:
        ne = s[s["empty_net"] == 0]
        tg = ne.groupby(["game_id", "team", "opp"]).agg(xg=("xg", "sum"), sog=("on_goal", "sum"), date=("date", "first"),
                                                        home=("is_home", "max")).reset_index()
        tg["is_home"] = np.where(tg["home"] == 1, 1, -1)
        tg["one"] = 1.0
        return tg.rename(columns={"team": "o", "opp": "d"})

    def ratings(self, before=None) -> dict:
        cur = self.cur if before is None else self.cur[self.cur["date"] < pd.Timestamp(before)]
        tl = self._team_games(self.last)
        prior = {m: solve_adjusted(tl, m, "o", "d", "one", ["is_home"], None, None, 4.0) for m in ("xg", "sog")}
        if len(cur):
            tc = self._team_games(cur)
            r = {m: solve_adjusted(tc, m, "o", "d", "one", ["is_home"], prior[m].off * PRIOR_KEEP, prior[m].deff * PRIOR_KEEP,
                                   PRIOR_GAMES) for m in ("xg", "sog")}
            n_cur = tc.groupby("o").size()
        else:
            from dataclasses import replace
            r = {m: replace(prior[m], off=prior[m].off * PRIOR_KEEP, deff=prior[m].deff * PRIOR_KEEP) for m in ("xg", "sog")}
            n_cur = pd.Series(dtype=float)
        # goalies: GSAx per shot on goal, last season at half weight
        on = pd.concat([self.last.assign(w=0.5), cur.assign(w=1.0)])
        on = on[(on["on_goal"] == 1) & (on["empty_net"] == 0)].dropna(subset=["goalie"])
        on["goalie"] = on["goalie"].astype(int)
        gq = on.assign(wx=on["xg"] * on["w"], wg=on["goal"] * on["w"]).groupby("goalie").agg(n=("w", "sum"), xg=("wx", "sum"), ga=("wg", "sum"))
        # expected goals on shots that reached the net, scaled so the league's goalies sum to zero GSAx
        k_on = float(gq["ga"].sum() / gq["xg"].sum())
        gsax = (gq["xg"] * k_on - gq["ga"]) / (gq["n"] + GOALIE_PRIOR_SHOTS)
        return {"xg": r["xg"], "sog": r["sog"], "gsax": gsax, "n_games": n_cur, "shots_faced": gq["n"]}

    def expected_goalie(self, team: str, before, b2b: bool) -> list[tuple[int, float]]:
        """[(goalie id, weight)] from the team's last 10 starts before this game."""
        g = pd.concat([self.games_last, self.games_cur])
        b = pd.Timestamp(before)
        b = b.tz_localize("UTC") if b.tzinfo is None else b
        g = g[pd.to_datetime(g["start_utc"], utc=True) < b].sort_values("start_utc")
        st = []
        for r in g.itertuples():
            if r.home == team and not pd.isna(r.home_goalie):
                st.append(int(r.home_goalie))
            elif r.away == team and not pd.isna(r.away_goalie):
                st.append(int(r.away_goalie))
        last = st[-10:]
        if not last:
            return []
        cnt = pd.Series(last).value_counts(normalize=True)
        if b2b and len(cnt) > 1:                         # second night of a back-to-back: the backup usually starts
            top = cnt.index[0]
            w = {gid: (0.35 if gid == top else 0.65 * v / max(1 - cnt[top], 1e-9)) for gid, v in cnt.items()}
            return list(w.items())
        return list(cnt.items())

    def project(self, R: dict, g: dict) -> dict:
        """g: game_id, date, away, home (NHL abbreviations), date_utc, b2b flags, optional market fields."""
        h, a = g["home"], g["away"]
        xg, sog, gsax = R["xg"], R["sog"], R["gsax"]
        ha = xg.covariates.get("is_home", 0.0)
        parts, goalies = {}, {}
        for side, o, d in (("home", h, a), ("away", a, h)):
            sgn = 1 if side == "home" else -1
            xg_for = xg.mu + xg.off.get(o, 0.0) + xg.deff.get(d, 0.0) + sgn * ha
            shots = sog.mu + sog.off.get(o, 0.0) + sog.deff.get(d, 0.0) + sgn * sog.covariates.get("is_home", 0.0)
            opp_side = "away" if side == "home" else "home"
            gw = self.expected_goalie(d, g["date_utc"], bool(g.get(f"{opp_side}_b2b")))
            goalies[opp_side] = gw
            g_term = -sum(w * float(gsax.get(gid, 0.0)) for gid, w in gw) * shots
            k = self.goals_per_xg
            parts[side] = {"Offence (xG created)": k * xg.off.get(o, 0.0), "Defence (xG allowed)": k * xg.deff.get(d, 0.0),
                           "Goaltending": g_term}
        base = self.goals_per_xg * xg.mu + self.extra_goals
        gp = GameProjection("nhl", str(g["game_id"]), str(g.get("date", "")), a, h, base, parts["away"], parts["home"],
                            {"Home ice": 2 * self.goals_per_xg * ha})
        mh_, ma_ = gp.proj_home, gp.proj_away
        c = gp.card(None, g.get("total"), None, None)
        ga, gh = RNG.poisson(max(ma_, 0.2), N_SIMS), RNG.poisson(max(mh_, 0.2), N_SIMS)
        tie = ga == gh
        home_ot = RNG.random(N_SIMS) < mh_ / (mh_ + ma_)
        home_win = (gh > ga) | (tie & home_ot)
        gh_f = gh + (tie & home_ot)
        ga_f = ga + (tie & ~home_ot)
        tot = gh_f + ga_f
        p_home = float(home_win.mean())
        c.update(p_home=p_home, fair_home_ml=prob_to_american(p_home), fair_away_ml=prob_to_american(1 - p_home),
                 p_home_by2=float((gh_f - ga_f >= 2).mean()), p_away_by2=float((ga_f - gh_f >= 2).mean()),
                 model_line=round(-(mh_ - ma_), 2), goalies={k: [(int(i), float(w)) for i, w in v] for k, v in goalies.items()})
        c["total_dist"] = {str(t): float((tot > t).mean()) for t in np.arange(4.5, 8.0, 1.0)}
        hm, am = g.get("home_ml"), g.get("away_ml")
        c["home_ml"], c["away_ml"] = hm, am
        if hm is not None and am is not None:
            mh, ma = no_vig(american_to_prob(hm), american_to_prob(am))
            side = "home" if p_home > mh else "away"
            p = p_home if side == "home" else 1 - p_home
            odds = hm if side == "home" else am
            c.update(market_p_home=mh, ml_pick=side, ml_odds=odds, ml_prob=p, ml_edge=round(100 * abs(p_home - mh), 1),
                     ml_ev=ev_per_unit(p, odds))
        if g.get("total") is not None:
            T = float(g["total"])
            po, pu = float((tot > T).mean()), float((tot < T).mean())
            side = "over" if po > pu else "under"
            oo, uo = g.get("over_odds") or -110, g.get("under_odds") or -110
            p, odds = (po, oo) if side == "over" else (pu, uo)
            p_loss = pu if side == "over" else po
            win = odds / 100 if odds > 0 else 100 / -odds
            c.update(tot_pick=side, tot_odds=odds, tot_prob=p, tot_edge=round(abs(c["model_total"] - T), 2),
                     tot_ev=p * win - p_loss, p_over=po, p_under=pu)
        return c
