"""
NBA lineups and player impact from ESPN play-by-play.

1. stints(pbp): rebuild the 10 players on the floor from substitution events. Starters of each period are
   the players who record an event or are subbed out before they are subbed in. A stint is the stretch
   between substitutions; for each one we count points and possessions for both teams.
2. player_impact(stints): regularised adjusted plus-minus (ridge regression). Every stint is a row:
   +1 for each home player on the floor, -1 for each away player, target = home margin per 100
   possessions, weight = possessions. Ridge shrinks low-minute players toward 0 (league average).
3. availability_adjustment(): how much a team's projected margin changes when its usual players are
   missing (or back), relative to how often they played in the games its ratings were built from.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


def _player_team(g: pd.DataFrame) -> dict:
    """athlete -> team_id for one game (substitutions are exact; other events by majority)."""
    votes: dict = {}
    sub = g["type_text"].eq("Substitution")
    for a1, a2, t in zip(g.loc[sub, "athlete_id_1"], g.loc[sub, "athlete_id_2"], g.loc[sub, "team_id"]):
        for a in (a1, a2):
            if pd.notna(a):
                votes.setdefault(a, Counter())[t] += 10
    ev = g[~sub & g["athlete_id_1"].notna() & g["team_id"].notna()]
    for a, t in zip(ev["athlete_id_1"], ev["team_id"]):
        votes.setdefault(a, Counter())[t] += 1
    return {a: c.most_common(1)[0][0] for a, c in votes.items()}


def stints(pbp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tt_all = pbp["type_text"].astype(str)
    shot = pbp["shooting_play"].fillna(False).astype(bool)
    is_ft = shot & ((pbp["points_attempted"] == 1) | tt_all.str.contains("Free Throw|FreeThrow"))
    pbp = pbp.assign(c_fga=(shot & ~is_ft).astype(int), c_fta=is_ft.astype(int),
                     c_oreb=tt_all.eq("Offensive Rebound").astype(int),
                     c_tov=(tt_all.str.contains("Turnover") & ~tt_all.str.contains("No Turnover")).astype(int))
    pbp = pbp.sort_values(["game_id", "sequence_number"])
    # a few rows carry a stale or zero score; the running maximum is the true score at that moment
    pbp["home_score"] = pbp.groupby("game_id")["home_score"].cummax()
    pbp["away_score"] = pbp.groupby("game_id")["away_score"].cummax()
    for gid, g in pbp.groupby("game_id", sort=False):
        home, away = g["home_team_id"].iloc[0], g["away_team_id"].iloc[0]
        pt = _player_team(g)
        for per, p in g.groupby("period_number", sort=True):
            on = {home: set(), away: set()}
            entered = set()
            for typ, a1, a2, t in zip(p["type_text"], p["athlete_id_1"], p["athlete_id_2"], p["team_id"]):
                if typ == "Substitution":
                    if pd.notna(a2) and a2 not in entered:
                        on.setdefault(pt.get(a2, t), set()).add(a2)
                    if pd.notna(a1):
                        entered.add(a1)
                else:
                    for a in (a1, a2):
                        if pd.notna(a) and a not in entered and a in pt:
                            on.setdefault(pt[a], set()).add(a)
            if len(on.get(home, ())) != 5 or len(on.get(away, ())) != 5:
                continue                      # could not reconstruct this period cleanly; skip it
            cur = {home: set(on[home]), away: set(on[away])}
            acc = None

            def flush():
                if acc and (acc["poss"] > 0):
                    rows.append(acc)

            def new_acc(sh, sa):
                return {"game_id": str(gid), "home_id": home, "away_id": away, "period": per,
                        "home_players": tuple(sorted(cur[home])), "away_players": tuple(sorted(cur[away])),
                        "h0": sh, "a0": sa, "fga_h": 0, "fga_a": 0, "fta_h": 0, "fta_a": 0, "oreb_h": 0,
                        "oreb_a": 0, "tov_h": 0, "tov_a": 0, "poss": 0.0}
            first = p.iloc[0]
            acc = new_acc(first["home_score"] - (first["score_value"] if first["scoring_play"] and first["team_id"] == home else 0),
                          first["away_score"] - (first["score_value"] if first["scoring_play"] and first["team_id"] == away else 0))
            last_h, last_a = acc["h0"], acc["a0"]
            for r in p.itertuples(index=False):
                if r.type_text == "Substitution":
                    team = pt.get(r.athlete_id_2, r.team_id)
                    if pd.notna(r.athlete_id_2):
                        cur.setdefault(team, set()).discard(r.athlete_id_2)
                    if pd.notna(r.athlete_id_1):
                        cur.setdefault(team, set()).add(r.athlete_id_1)
                    acc.update(pts_h=last_h - acc["h0"], pts_a=last_a - acc["a0"])
                    acc["poss"] = 0.5 * ((acc["fga_h"] - acc["oreb_h"] + acc["tov_h"] + 0.44 * acc["fta_h"])
                                         + (acc["fga_a"] - acc["oreb_a"] + acc["tov_a"] + 0.44 * acc["fta_a"]))
                    if len(cur[home]) == 5 and len(cur[away]) == 5:
                        flush()
                    acc = new_acc(last_h, last_a)
                    continue
                side = "h" if r.team_id == home else "a"
                acc[f"fga_{side}"] += r.c_fga
                acc[f"fta_{side}"] += r.c_fta
                acc[f"oreb_{side}"] += r.c_oreb
                acc[f"tov_{side}"] += r.c_tov
                last_h, last_a = r.home_score, r.away_score
            acc.update(pts_h=last_h - acc["h0"], pts_a=last_a - acc["a0"])
            acc["poss"] = 0.5 * ((acc["fga_h"] - acc["oreb_h"] + acc["tov_h"] + 0.44 * acc["fta_h"])
                                 + (acc["fga_a"] - acc["oreb_a"] + acc["tov_a"] + 0.44 * acc["fta_a"]))
            flush()
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s = s[s["poss"] > 0]
    s["date"] = s["game_id"].map(pbp.drop_duplicates("game_id").set_index(pbp.drop_duplicates("game_id")["game_id"].astype(str))["game_date"])
    return s.reset_index(drop=True)


def player_impact(st: pd.DataFrame, alpha: float = 3000.0, prior: pd.Series | None = None) -> pd.Series:
    """Ridge adjusted plus-minus, points per 100 possessions vs average. Prior (e.g. last season) is the
    value each player is shrunk toward instead of 0."""
    players = sorted({p for t in st["home_players"] for p in t} | {p for t in st["away_players"] for p in t})
    idx = {p: i for i, p in enumerate(players)}
    rows, cols, vals = [], [], []
    for k, (hp, ap) in enumerate(zip(st["home_players"], st["away_players"])):
        for p in hp:
            rows.append(k); cols.append(idx[p]); vals.append(1.0)
        for p in ap:
            rows.append(k); cols.append(idx[p]); vals.append(-1.0)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(st), len(players)))
    y = 100 * (st["pts_h"] - st["pts_a"]).to_numpy(float) / st["poss"].to_numpy(float)
    w = st["poss"].to_numpy(float)
    pr = np.zeros(len(players)) if prior is None else prior.reindex(players).fillna(0).to_numpy()
    hca = np.average(y, weights=w)                                  # home edge absorbed as a constant
    y = y - hca - X @ pr
    sw = np.sqrt(w)
    A = sparse.vstack([sparse.diags(sw) @ X, np.sqrt(alpha) * sparse.identity(len(players))])
    b = np.concatenate([sw * y, np.zeros(len(players))])
    beta = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=5000)[0]
    return pd.Series(pr + beta, index=players, name="impact")


def usage(st: pd.DataFrame) -> pd.DataFrame:
    """Per player per game: team, possessions on the floor and share of the team's possessions."""
    recs = []
    for r in st.itertuples(index=False):
        for side, team in (("home_players", r.home_id), ("away_players", r.away_id)):
            for p in getattr(r, side):
                recs.append((r.game_id, r.date, team, p, r.poss))
    u = pd.DataFrame(recs, columns=["game_id", "date", "team_id", "player", "poss"])
    u = u.groupby(["game_id", "date", "team_id", "player"], as_index=False)["poss"].sum()
    team_poss = st.groupby("game_id")["poss"].sum()
    u["share"] = u["poss"] / u["game_id"].map(team_poss)
    return u


def availability_adjustment(team_id, played: set, before, use: pd.DataFrame, impact: pd.Series,
                            replacement: float, window_games: int = 20) -> float:
    """Change in the team's net rating (pts / 100 poss) for this game's actual or expected players,
    versus the mix of players in its recent games (which is what its team rating already reflects).

    Each regular gets weight = his average floor share when he plays. Delta = sum over regulars of
    share x (impact - replacement) x (available now - fraction of recent games he played)."""
    u = use[(use["team_id"] == team_id) & (use["date"] < before)]
    games = sorted(u["game_id"].unique(), key=lambda g: u.loc[u["game_id"] == g, "date"].iloc[0])[-window_games:]
    u = u[u["game_id"].isin(games)]
    if not len(games):
        return 0.0
    g = u.groupby("player").agg(share=("share", "mean"), n=("game_id", "nunique"))
    g["presence"] = g["n"] / len(games)
    g = g[g["share"] >= 0.15]                                      # rotation players
    imp = impact.reindex(g.index).fillna(replacement)
    avail = g.index.to_series().isin(played).astype(float)
    return float((g["share"] * (imp - replacement) * (avail - g["presence"])).sum())
