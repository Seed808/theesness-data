"""
SYNTHETIC data generators — for unit-testing the MLB and NHL modules only.
Nothing produced here is real; real runs use load_statcast() / the NHL loader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA",
         "MIL", "MIN", "NYM", "NYY", "ATH", "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH"]
BASE = {"1B": .142, "2B": .044, "3B": .004, "HR": .031, "BB": .092, "K": .224, "OUT": .463}
WOBA = {"1B": .88, "2B": 1.25, "3B": 1.58, "HR": 2.03, "BB": .70, "K": 0, "OUT": 0}
EV = {"1B": "single", "2B": "double", "3B": "triple", "HR": "home_run", "BB": "walk", "K": "strikeout", "OUT": "field_out"}


def statcast_like(n_days=150, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    off = dict(zip(TEAMS, rng.normal(0, 0.12, 30)))           # log-odds hit/HR boost
    park = dict(zip(TEAMS, rng.normal(0, 0.06, 30)))
    sps = {t: [f"{t}_SP{i}" for i in range(5)] for t in TEAMS}
    rps = {t: [f"{t}_RP{i}" for i in range(7)] for t in TEAMS}
    pq = {p: rng.normal(0, 0.15) for t in TEAMS for p in sps[t] + rps[t]}
    pid = {p: 100000 + i for i, p in enumerate(pq)}
    hand = {p: rng.choice(["R", "L"], p=[.7, .3]) for p in pq}
    batters = {t: [(9000 + 20 * k + j) for j in range(9)] for k, t in enumerate(TEAMS)}
    bq = {b: rng.normal(0, 0.12) for t in TEAMS for b in batters[t]}
    bh = {b: rng.choice(["R", "L"], p=[.6, .4]) for b in bq}
    keys = list(BASE)
    base = np.array([BASE[k] for k in keys])
    rows, gpk = [], 700000
    dates = pd.date_range("2026-04-01", periods=n_days)
    for d in dates:
        order = rng.permutation(TEAMS)
        for i in range(0, 30, 2):
            a, h = order[i], order[i + 1]
            gpk += 1
            score = {a: 0, h: 0}
            for half, (bat, fld) in enumerate([(a, h), (h, a)] * 9):
                pass
            ab = 0
            lineup_pos = {a: 0, h: 0}
            sp = {t: sps[t][(d.dayofyear) % 5] for t in (a, h)}
            pitcher_bf = {a: 0, h: 0}
            for inning in range(1, 10):
                for bat, fld, tb in ((a, h, "Top"), (h, a, "Bot")):
                    outs, bases_runners = 0, 0
                    while outs < 3:
                        b = batters[bat][lineup_pos[bat] % 9]
                        lineup_pos[bat] += 1
                        p = sp[fld] if pitcher_bf[fld] < 24 else rps[fld][rng.integers(7)]
                        pitcher_bf[fld] += 1
                        lo = np.log(base / (1 - base))
                        boost = off[bat] + bq[b] - pq[p] + park[h] + (0.05 if bat == h else 0)
                        lo[:4] += boost
                        lo[5] -= boost
                        pr = 1 / (1 + np.exp(-lo)); pr /= pr.sum()
                        o = keys[rng.choice(7, p=pr)]
                        before = score[bat]
                        if o in ("K", "OUT"):
                            outs += 1
                        elif o == "HR":
                            score[bat] += 1 + bases_runners; bases_runners = 0
                        else:
                            bases_runners += 1
                            if bases_runners > 3:
                                score[bat] += 1; bases_runners = 3
                            if o in ("2B", "3B") and bases_runners > 1:
                                score[bat] += 1; bases_runners -= 1
                        ab += 1
                        xw = WOBA[o] + rng.normal(0, .15) if o in ("1B", "2B", "3B", "HR", "OUT") else np.nan
                        rows.append((gpk, d, h, a, inning, tb, ab, b, pid[p], f"P {pid[p]}", bh[b], hand[p], EV[o],
                                     WOBA[o], 1, xw, before, score[bat], "R"))
    cols = ["game_pk", "game_date", "home_team", "away_team", "inning", "inning_topbot", "at_bat_number", "batter",
            "pitcher", "player_name", "stand", "p_throws", "events", "woba_value", "woba_denom",
            "estimated_woba_using_speedangle", "bat_score", "post_bat_score", "game_type"]
    return pd.DataFrame(rows, columns=cols)


NHL_TEAMS = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
             "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"]


def nhl_shots(n_days=80, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    o = dict(zip(NHL_TEAMS, rng.normal(0, .12, 32)))
    d = dict(zip(NHL_TEAMS, rng.normal(0, .12, 32)))
    goalie = {t: 8000000 + i for i, t in enumerate(NHL_TEAMS)}
    gq = {g: rng.normal(0, .006) for g in goalie.values()}
    rows, gid = [], 2026020000
    for day in pd.date_range("2026-10-07", periods=n_days):
        order = rng.permutation(NHL_TEAMS)
        for i in range(0, 32, 2):
            a, h = order[i], order[i + 1]
            gid += 1
            for team, opp in ((a, h), (h, a)):
                n = rng.poisson(np.exp(np.log(55) + o[team] + d[opp] + (0.03 if team == h else 0)))
                dist = rng.gamma(3, 12, n)
                ang = rng.uniform(0, 80, n)
                reb = (rng.random(n) < .08).astype(int)
                p = 1 / (1 + np.exp(-(-1.0 - 0.055 * dist - 0.01 * ang + 0.8 * reb)))
                on = rng.random(n) < .55
                g = (rng.random(n) < np.clip(p - gq[goalie[opp]], 0, 1)) & on
                for k in range(n):
                    rows.append((gid, str(day.date()), h, a, team, opp, 100 + k % 12, goalie[opp], int(g[k]), int(on[k]),
                                 dist[k], ang[k], "wrist", reb[k], "1551"))
    return pd.DataFrame(rows, columns=["game_id", "date", "home", "away", "team", "opp", "shooter", "goalie", "goal",
                                       "on_goal", "dist", "angle", "shot_type", "rebound", "strength"])
