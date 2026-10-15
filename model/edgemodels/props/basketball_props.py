"""
Basketball player props from play-by-play (NBA / CBB).

Per-game rates (exponentially weighted, recent games count more), scaled by
projected game pace and the opponent's adjusted defence, then simulated:

  points   = 2*Bin(2PA, p2) + 3*Bin(3PA, p3) + Bin(FTA, pft), attempts ~ NegBin
  rebounds, assists, threes ~ NegBin
  PRA, P+R are summed per simulation, and every stat in a simulation shares a
  common Gamma "role" multiplier (minutes / usage swings) so they correlate.
  first basket = P(team scores first FG) x player's share of early-game FGM

Without minutes data the rates are per game. Pass `minutes_scale` (projected /
recent minutes) for injuries or rotation changes you know about.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core.sim import RNG, binary_summary, negbin, prop_summary

N_SIMS = 20000
DECAY = 0.9
ROLE_SHAPE = 14.0          # Gamma shape of the shared role multiplier (CV ~ 27%)
SHOT_PRIOR = 30            # attempts of league-average shooting mixed into each player's %


def player_games(pbp: pd.DataFrame) -> pd.DataFrame:
    e = pbp.copy()
    tt = e["type_text"].astype(str)
    shot = e["shooting_play"].fillna(False).astype(bool)
    ft = shot & ((e["points_attempted"] == 1) | tt.str.contains("Free Throw|FreeThrow"))
    fga = shot & ~ft
    made = e["scoring_play"].fillna(False).astype(bool)
    three = e["points_attempted"] == 3
    a1 = e["athlete_id_1"]
    rows = pd.DataFrame({
        "game_id": e["game_id"].astype(str), "date": pd.to_datetime(e["game_date"]), "team_id": e["team_id"],
        "pid": a1, "name": e["athlete_name_1"] if "athlete_name_1" in e else a1.astype(str),
        "PTS": np.where(made & shot, e["score_value"], 0),
        "FG2A": (fga & ~three).astype(int), "FG2M": (fga & ~three & made).astype(int),
        "FG3A": (fga & three).astype(int), "FG3M": (fga & three & made).astype(int),
        "FTA": ft.astype(int), "FTM": (ft & made).astype(int),
        "REB": tt.isin(["Offensive Rebound", "Defensive Rebound"]).astype(int),
        "AST": 0,
    })
    ast = pd.DataFrame({"game_id": e["game_id"].astype(str), "date": pd.to_datetime(e["game_date"]),
                        "team_id": e["team_id"], "pid": e["athlete_id_2"],
                        "name": e["athlete_name_2"] if "athlete_name_2" in e else e["athlete_id_2"].astype(str),
                        "AST": (fga & made).astype(int)})
    ast = ast[(ast["AST"] == 1) & ast["pid"].notna()]
    allr = pd.concat([rows.dropna(subset=["pid"]), ast.dropna(subset=["pid"])], ignore_index=True).fillna(0)
    g = allr.groupby(["game_id", "date", "pid"]).agg(
        name=("name", lambda s: next((x for x in s if isinstance(x, str) and x), "")),
        team_id=("team_id", lambda s: s.mode().iloc[0]),
        **{k: (k, "sum") for k in ["PTS", "FG2A", "FG2M", "FG3A", "FG3M", "FTA", "FTM", "REB", "AST"]}).reset_index()
    return g


def first_basket_table(pbp: pd.DataFrame, early_secs: float = 360) -> tuple[pd.DataFrame, float, pd.Series]:
    """Player share of team FGM in the first `early_secs` of the game, P(tip winner scores first FG),
    and each team's opening-tip win rate."""
    e = pbp[(pbp["period_number"] == 1)].copy()
    tt = e["type_text"].astype(str)
    fgm = e["shooting_play"].fillna(False) & e["scoring_play"].fillna(False) & ~((e["points_attempted"] == 1) | tt.str.contains("Free Throw|FreeThrow"))
    t_el = 720 - e["start_quarter_seconds_remaining"] if "start_quarter_seconds_remaining" in e else None
    early = e[fgm & (t_el <= early_secs)] if t_el is not None else e[fgm]
    share = early.groupby(["team_id", "athlete_id_1"]).size()
    share = (share / share.groupby(level=0).transform("sum")).rename("early_fgm_share")
    # opening tip: the jump ball text names who gained possession; map that player to a team
    first_fg = e[fgm].sort_values("sequence_number").groupby("game_id").head(1).set_index("game_id")["team_id"]
    tips = e[tt.eq("Jumpball")].sort_values("sequence_number").groupby("game_id").head(1)
    name_team = e.dropna(subset=["athlete_name_1"]).groupby("athlete_name_1")["team_id"].agg(lambda s: s.mode().iloc[0])
    def winner(txt):
        m = re.search(r"\((.+?) gains possession\)", str(txt))
        return name_team.get(m.group(1)) if m else np.nan
    tips = tips.assign(tip_team=tips["text"].map(winner)).dropna(subset=["tip_team"]).set_index("game_id")
    both = tips.join(first_fg.rename("first_fg_team"), how="inner")
    q = float((both["tip_team"] == both["first_fg_team"]).mean()) if len(both) else 0.5
    tip_rate = tips.assign(home=tips["home_team_id"]).apply(lambda r: r["tip_team"], axis=1).value_counts()
    games_per_team = pd.concat([tips["home_team_id"], tips["away_team_id"]]).value_counts()
    tip_rate = ((tip_rate.reindex(games_per_team.index).fillna(0) + 5) / (games_per_team + 10)).rename("tip_win_rate")
    return share.reset_index(), q, tip_rate


class BasketballProps:
    def __init__(self, player_games_df: pd.DataFrame, ratings, team_ids: dict, early_share=None, q_first=0.5,
                 tip_rate=None, last_n: int = 15):
        self.pg = player_games_df
        self.br = ratings
        self.team_ids = team_ids           # abbrev -> ESPN team_id
        self.early = early_share
        self.q = q_first
        self.tip = tip_rate if tip_rate is not None else pd.Series(dtype=float)
        self.last_n = last_n
        tot = self.pg[["FG2A", "FG2M", "FG3A", "FG3M", "FTA", "FTM"]].sum()
        self.lg = {"p2": tot["FG2M"] / tot["FG2A"], "p3": tot["FG3M"] / tot["FG3A"], "pft": tot["FTM"] / tot["FTA"]}

    def _player_rates(self, team: str, include: list | None = None) -> pd.DataFrame:
        tid = self.team_ids[team]
        g = self.pg[self.pg["team_id"] == tid]
        dates = sorted(g["date"].unique())[-self.last_n:]
        g = g[g["date"].isin(dates)]
        last = dates[-1] if dates else None
        out = []
        for pid, d in g.groupby("pid"):
            nm = self.pg.loc[self.pg["pid"] == pid, "name"].iloc[-1]
            if include is not None:
                if nm not in include:
                    continue                              # caller supplied the expected rotation
            elif last is None or last not in set(d["date"]):
                continue                                  # did not appear in the team's last game
            d = d.set_index("date").reindex(dates).fillna(0)
            w = DECAY ** np.arange(len(dates))[::-1]
            appeared = (d[["PTS", "FG2A", "FG3A", "FTA", "REB", "AST"]].sum(axis=1) > 0).to_numpy()
            w = w * appeared                              # average over games he played
            if w.sum() == 0:
                continue
            m = {k: float((d[k] * w).sum() / w.sum()) for k in ["PTS", "FG2A", "FG2M", "FG3A", "FG3M", "FTA", "FTM", "REB", "AST"]}
            m["p2"] = (d["FG2M"].sum() + SHOT_PRIOR * self.lg["p2"]) / (d["FG2A"].sum() + SHOT_PRIOR)
            m["p3"] = (d["FG3M"].sum() + SHOT_PRIOR * self.lg["p3"]) / (d["FG3A"].sum() + SHOT_PRIOR)
            m["pft"] = (d["FTM"].sum() + SHOT_PRIOR * self.lg["pft"]) / (d["FTA"].sum() + SHOT_PRIOR)
            names = self.pg.loc[self.pg["pid"] == pid, "name"]
            m.update(pid=pid, name=names.iloc[-1] if len(names) else str(pid), games=int(appeared.sum()))
            out.append(m)
        return pd.DataFrame(out)

    def game_props(self, card: dict, top_n: int = 5, minutes_scale: dict | None = None,
                   rotations: dict | None = None) -> dict:
        """rotations: optional {team: [player names expected to play]} to override the
        'appeared in the team's last game' rule (rest days, injuries, trades)."""
        minutes_scale = minutes_scale or {}
        br = self.br
        pace = card["pace"]
        out = {"players": [], "team_totals": {}}
        p_first = {}
        tip_h = float(self.tip.get(self.team_ids.get(card["home"]), 0.5))
        tip_a = float(self.tip.get(self.team_ids.get(card["away"]), 0.5))
        p_tip_home = tip_h / (tip_h + tip_a)
        p_first[card["home"]] = p_tip_home * self.q + (1 - p_tip_home) * (1 - self.q)
        p_first[card["away"]] = 1 - p_first[card["home"]]
        for side, opp in (("home", "away"), ("away", "home")):
            team, o = card[side], card[opp]
            sd = card.get("_total_sd", 18) / np.sqrt(2)
            out["team_totals"][team] = prop_summary(RNG.normal(card[f"proj_{side}"], sd, N_SIMS), label=f"{team} team total")
            rates = self._player_rates(team, (rotations or {}).get(team))
            if rates.empty:
                continue
            team_pace = br.pace.mu + br.pace.off.get(team, 0)
            pace_f = pace / team_pace
            d_eff = (br.ortg.mu + br.ortg.deff.get(o, 0)) / br.ortg.mu      # >1: soft defence
            d_efg = br.factors["efg"].mu + br.factors["efg"].deff.get(o, 0)
            efg_f = d_efg / br.factors["efg"].mu
            rates = rates.sort_values("PTS", ascending=False).head(top_n)
            es = self.early
            tid = self.team_ids.get(team)
            for _, r in rates.iterrows():
                ms = minutes_scale.get(r["name"], 1.0)
                role = RNG.gamma(ROLE_SHAPE, 1 / ROLE_SHAPE, N_SIMS) * ms * pace_f
                a2 = negbin(r["FG2A"] * role, 25, N_SIMS)
                a3 = negbin(r["FG3A"] * role, 25, N_SIMS)
                af = negbin(r["FTA"] * role * d_eff, 10, N_SIMS)
                m2 = RNG.binomial(a2, min(r["p2"] * efg_f, 0.95))
                m3 = RNG.binomial(a3, min(r["p3"] * efg_f, 0.95))
                mf = RNG.binomial(af, r["pft"])
                pts = 2 * m2 + 3 * m3 + mf
                reb = negbin(r["REB"] * role, 12, N_SIMS)
                ast = negbin(r["AST"] * role, 12, N_SIMS)
                row = {"player": r["name"], "team": team, "opp": o, "games": r["games"],
                       "points": prop_summary(pts, label="Points"),
                       "rebounds": prop_summary(reb, bins=np.arange(-0.5, 21.5), label="Rebounds"),
                       "assists": prop_summary(ast, bins=np.arange(-0.5, 16.5), label="Assists"),
                       "threes": prop_summary(m3, bins=np.arange(-0.5, 10.5), label="Threes made"),
                       "pra": prop_summary(pts + reb + ast, label="Pts + Reb + Ast"),
                       "pr": prop_summary(pts + reb, label="Pts + Reb"),
                       "points_10plus": float((pts >= 10).mean()), "points_20plus": float((pts >= 20).mean()),
                       "points_30plus": float((pts >= 30).mean())}
                if es is not None and tid is not None:
                    s = es[(es["team_id"] == tid) & (es["athlete_id_1"] == r["pid"])]["early_fgm_share"]
                    share = float(s.iloc[0]) if len(s) else 0.0
                    row["first_basket"] = binary_summary(p_first[team] * share, label="First basket (FG)")
                out["players"].append(row)
        out["p_team_first"] = p_first
        return out
