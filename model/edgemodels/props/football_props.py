"""
Football player props from play-by-play.

Opportunity  x  efficiency  x  matchup, then Monte Carlo:
  receptions     targets ~ NegBin(team dropbacks x target share), catches ~ Binomial(targets, catch rate)
  receiving yds  sum of per-catch yard draws (player pool shrunk toward league pool) x defence factor
  rushing yds    carries ~ NegBin(team rushes x carry share), sum of per-carry draws x defence factor
  anytime TD     Poisson: lambda = team offensive TDs x player TD share
  first TD       P(team scores first TD) x player share of that team's TDs
  team / game totals from the projected scores and the backtest residual SD
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.sim import RNG, binary_summary, compound_sum, negbin, prop_summary

N_SIMS = 20000
SHARE_DECAY = 0.85        # weight on each older game when averaging shares
PRIOR_EVENTS = 40         # league-pool events mixed into each player's yardage pool
TD_SHARE_TEMP = 0.7       # share^T, renormalised. Best first-TD log-loss (3.017) on 2025 wk4-18 + 2026 wk2-5, 284 games, with injuries
TD_PRIOR_GAMES = 100.0    # games of opportunity-based expected TDs blended with actual TDs; high = touches and red-zone usage decide, TD luck barely counts


def _ewm(series_by_game: pd.Series, game_order: list) -> float:
    s = series_by_game.reindex(game_order).dropna().to_numpy()[::-1]   # newest first, games he played only
    w = SHARE_DECAY ** np.arange(len(s))
    return float((s * w).sum() / w.sum()) if len(s) else 0.0


class TeamUsage:
    """Usage per player from the offence's recent games.

    Shares are averaged over the games each player actually played, so a player returning
    from injury is not dragged down by the games he missed, and the teammate who filled in
    is not credited with a role he no longer has. Who plays this week is decided later
    (availability), and the shares of available players are renormalised to the whole team.
    """

    def __init__(self, plays: pd.DataFrame, team: str, last_n: int = 8):
        p = plays[(plays["off"] == team) & ~plays["garbage"]]
        self.games = list(dict.fromkeys(p.sort_values(["season", "week"])["game_id"]))[-last_n:]
        p = p[p["game_id"].isin(self.games)]
        self.p = p
        tgt = p[p["target"]]
        ru = p[p["dropback"] == 0]
        rz = p[p["rz"]]
        team_tg = tgt.groupby("game_id").size()
        team_ru = ru.groupby("game_id").size()
        rows = []
        for pid in set(tgt["receiver_id"].dropna()) | set(ru["rusher_id"].dropna()):
            pt, pr = tgt[tgt["receiver_id"] == pid], ru[ru["rusher_id"] == pid]
            name = (pt["receiver"].dropna().tolist() or pr["rusher"].dropna().tolist())[-1]
            played = [g for g in self.games if g in set(pt["game_id"]) | set(pr["game_id"])]
            n = len(played)
            g_t = pt.groupby("game_id").size().reindex(played).fillna(0)
            g_r = pr.groupby("game_id").size().reindex(played).fillna(0)
            rows.append({
                "player_id": pid, "player": name, "games": n,
                "tgt_share": _ewm(g_t / team_tg.reindex(played), self.games),
                "car_share": _ewm(g_r / team_ru.reindex(played), self.games),
                "catch_rate_n": len(pt), "catches": float(pt["complete"].sum()),
                "rz_tgt": int((rz["target"] & (rz["receiver_id"] == pid)).sum()),
                "rz_car": int(((rz["rusher_id"] == pid) & (rz["dropback"] == 0)).sum()),
                "tgt": int(len(pt)), "car": int(len(pr)),
                "tds": int((p["td_player_id"] == pid).sum()),
                "active_last2": bool(set(played) & set(self.games[-2:])),
            })
        self.players = pd.DataFrame(rows).set_index("player_id") if rows else pd.DataFrame()
        self.team_tds = int(p["td_player_id"].notna().sum())
        self.n_games = len(self.games)


class FootballProps:
    def __init__(self, plays: pd.DataFrame, ratings, margin_sd: float, total_sd: float):
        self.plays = plays[~plays["garbage"]]
        self.fr = ratings
        self.margin_sd, self.total_sd = margin_sd, total_sd
        t = self.plays
        self.lg_catch = float(t.loc[t["target"], "complete"].mean())
        self.pool_rec = t.loc[t["target"] & (t["complete"] == 1), "yards"].dropna().to_numpy()
        self.pool_rush = t.loc[t["dropback"] == 0, "yards"].dropna().to_numpy()
        # projections are built on non-garbage snaps; props settle on every snap
        a = plays
        self.k_ru = float((a["dropback"] == 0).sum() / max((t["dropback"] == 0).sum(), 1))
        self.k_db = float((a["dropback"] == 1).sum() / max((t["dropback"] == 1).sum(), 1))
        # offensive TDs per point scored (league), used to turn projected points into TD expectation
        tt = plays.assign(pts=np.where(plays["off"] == plays["home"], plays["home_score"], plays["away_score"]),
                          td=plays["td_player_id"].notna())
        g = tt.groupby(["game_id", "off"]).agg(td=("td", "sum"), pts=("pts", "max")).dropna()
        self.td_per_point = float(g["td"].sum() / g["pts"].sum())
        # league TD rate per opportunity type: red-zone / other, carry / target
        td = t["td_player_id"].notna()
        car, tgt = t["dropback"] == 0, t["target"]
        self.td_rate = {k: float(td[m].mean()) for k, m in {
            "rz_car": car & t["rz"], "rz_tgt": tgt & t["rz"], "car": car & ~t["rz"], "tgt": tgt & ~t["rz"]}.items()}

    def _pool(self, own: np.ndarray, league: np.ndarray) -> np.ndarray:
        extra = RNG.choice(league, size=PRIOR_EVENTS, replace=True) if len(league) else np.array([])
        return np.concatenate([own, extra])

    def _def_factor(self, metric: str, defense: str) -> float:
        r = self.fr.ratings[metric]
        return float((r.mu + r.deff.get(defense, 0.0)) / r.mu)

    def game_props(self, card: dict, top_n: int = 4, availability: dict | None = None) -> dict:
        """Props for both teams of one projected game card.

        availability: {team: {player_id: P(active)}} from the injury report / inactives (sports.nfl_injuries).
        Players at 0 are removed and their targets, carries and TD share go to available teammates
        in proportion to what those teammates already get; Questionable players count at P(active).
        Without it, a player must have played in one of the team's last two games."""
        out = {"team_totals": {}, "players": [], "first_td": []}
        lam = {}
        for side, opp in (("home", "away"), ("away", "home")):
            team, defense = card[side], card[opp]
            inp = card["inputs"][side]
            proj_pts = card[f"proj_{side}"]
            team_sd = self.total_sd / np.sqrt(2)
            out["team_totals"][team] = prop_summary(RNG.normal(proj_pts, team_sd, N_SIMS), label=f"{team} team total")
            lam[team] = proj_pts * self.td_per_point
            u = TeamUsage(self.plays, team)
            if u.players.empty:
                continue
            pl = u.players.copy()
            av = (availability or {}).get(team)
            if av is not None:
                # report published: anyone not listed is healthy. Not yet published: keep the
                # played-in-the-last-two-games rule, but still apply IR / cuts / trades from the roster.
                published = av.get("__published__", True)
                pl["p_active"] = [av.get(pid, 1.0 if (published or r.active_last2) else 0.0)
                                  for pid, r in pl.iterrows()]
            else:
                pl["p_active"] = pl["active_last2"].astype(float)
            pl = pl[pl["p_active"] > 0].copy()
            if pl.empty:
                continue
            # redistribute: expected shares of available players scaled up to cover the whole team
            for col in ("tgt_share", "car_share"):
                exp_sum = float((pl[col] * pl["p_active"]).sum())
                if exp_sum > 0:
                    pl[col] = pl[col] / exp_sum
            db = inp["plays"] * inp["pass_rate"] * self.k_db
            ru = inp["plays"] * (1 - inp["pass_rate"]) * self.k_ru
            team_targets = db * float(u.p["target"].sum() / max((u.p["dropback"] == 1).sum(), 1))
            # TD share: expected TDs from each player's opportunities (red-zone and other carries/targets,
            # valued at league TD rates), nudged toward his actual share of the team's TDs
            r = self.td_rate
            xtd = (pl["rz_car"] * r["rz_car"] + pl["rz_tgt"] * r["rz_tgt"]
                   + (pl["car"] - pl["rz_car"]) * r["car"] + (pl["tgt"] - pl["rz_tgt"]) * r["tgt"])
            g = pl["games"].clip(lower=1)
            # per-game TD expectation: actual TDs per game shrunk toward opportunity-based expected TDs per game
            rate = (pl["tds"] + TD_PRIOR_GAMES * xtd / g) / (g + TD_PRIOR_GAMES)
            flat = rate.clip(lower=1e-6) ** TD_SHARE_TEMP
            pl["td_share"] = flat / float((flat * pl["p_active"]).sum())     # available players cover the team
            fac_rec = self._def_factor("ypt", defense)
            fac_rush = self._def_factor("ypc", defense)
            for pid, r in pl.iterrows():
                rec_mean = team_targets * r["tgt_share"]
                car_mean = ru * r["car_share"]
                row = {"player": r["player"], "player_id": pid, "team": team, "opp": defense, "p_active": r["p_active"],
                       "proj_targets": rec_mean, "proj_carries": car_mean, "td_share": r["td_share"]}
                if rec_mean >= 2:
                    targets = negbin(rec_mean, 6.0, N_SIMS)
                    cr = (r["catches"] + 20 * self.lg_catch) / (r["catch_rate_n"] + 20)
                    catches = RNG.binomial(targets, cr)
                    own = u.p.loc[(u.p["receiver_id"] == pid) & (u.p["complete"] == 1), "yards"].dropna().to_numpy()
                    yds = compound_sum(catches, self._pool(own, self.pool_rec), fac_rec)
                    row["receptions"] = prop_summary(catches, bins=np.arange(-0.5, 15.5), label="Receptions")
                    row["receiving_yards"] = prop_summary(yds, label="Receiving yards")
                if car_mean >= 4:
                    carries = negbin(car_mean, 8.0, N_SIMS)
                    own = u.p.loc[(u.p["rusher_id"] == pid) & (u.p["dropback"] == 0), "yards"].dropna().to_numpy()
                    yds = compound_sum(carries, self._pool(own, self.pool_rush), fac_rush)
                    row["rushing_yards"] = prop_summary(yds, label="Rushing yards")
                pa = r["p_active"]                 # questionable: probability-weighted
                row["anytime_td"] = binary_summary(float(pa * (1 - np.exp(-lam[team] * r["td_share"]))), label="Anytime TD")
                row["_first_td_share"] = pa * r["td_share"]
                out["players"].append(row)
        # first touchdown: whichever team's Poisson TD clock fires first
        a, h = card["away"], card["home"]
        tot = lam[a] + lam[h]
        p_any = 1 - np.exp(-tot)
        for row in out["players"]:
            p_team_first = lam[row["team"]] / tot * p_any
            row["first_td"] = binary_summary(float(p_team_first * row.pop("_first_td_share")), label="First TD")
        out["players"].sort(key=lambda r: -r["anytime_td"]["p_yes"])
        out["game_total"] = prop_summary(RNG.normal(card["model_total"], self.total_sd, N_SIMS),
                                         line=card.get("market_total"), label="Game total")
        out["p_no_td"] = float(np.exp(-tot))
        return out


def ladder(summary_sims: np.ndarray, lines: list[float]) -> list[dict]:
    """P(over) at several thresholds — useful when no market line is supplied."""
    return [{"line": L, "p_over": float((summary_sims > L).mean())} for L in lines]
