"""
DraftKings player props (from ESPN's public feed, saved by scripts/fetch_props.py) set against our player simulations.

For every prop: our projection, our chance of over/under, DraftKings' price and its no-vig chance, a blended chance
(our model counted at MODEL_WEIGHT, the market at the rest, because these player models are new and untested against
closing prices), and the value of the better side at DraftKings' price. "Favorites" are the best of those, a few per
game. Nothing here is part of any record; the props tracker grades them separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.edges import american_to_prob, ev_per_unit

MODEL_WEIGHT = 0.35          # share of our own number in the blended chance
FAV_MIN_EV = 0.03            # favourites need at least +3% expected value at the posted price
N_FAV = 3

NBA_MARKETS = {"Total Points": "points", "Total Rebounds": "rebounds", "Total Assists": "assists",
               "Total 3-Point Field Goals": "threes", "Total Points, Rebounds, and Assists": "pra",
               "Total Points and Rebounds": "pr", "Total Points and Assists": "pa", "Total Assists and Rebounds": "ar"}
NFL_MARKETS = {"Total Rushing Yards (incl. overtime)": "rushing_yards", "Total Receiving Yards (incl. overtime)": "receiving_yards",
               "Total Receptions (incl. overtime)": "receptions", "Anytime Touchdown Scorer": "anytime_td",
               "First Touchdown Scorer": "first_td"}
LABEL = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "threes": "Threes", "pra": "Pts + Reb + Ast",
         "pr": "Pts + Reb", "pa": "Pts + Ast", "ar": "Ast + Reb", "rushing_yards": "Rushing yards",
         "receiving_yards": "Receiving yards", "receptions": "Receptions", "anytime_td": "Anytime TD", "first_td": "First TD"}
ORDER = ["points", "rebounds", "assists", "threes", "pra", "pr", "pa", "ar",
         "receptions", "receiving_yards", "rushing_yards", "anytime_td", "first_td"]


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v                          # NaN -> None


def load_market(repo: Path, sport: str, event_id: str) -> tuple[pd.DataFrame, str | None]:
    """Two-way props as one row per (player, market, line) with over/under prices; one-way props (TD scorer)
    with a single yes price. Returns (rows, time seen)."""
    d = Path(repo) / "data" / "espn" / sport / "props" / str(event_id)
    f = d / "latest.csv"
    if not f.exists():
        return pd.DataFrame(), None
    seen = (d / "seen_at.txt").read_text().strip() if (d / "seen_at.txt").exists() else None
    m = NBA_MARKETS if sport == "nba" else NFL_MARKETS
    raw = pd.read_csv(f, dtype={"athlete_id": str, "team_id": str})
    raw = raw[raw["market"].isin(m) & raw["athlete_id"].notna()]
    rows = []
    for (aid, mk, line), g in raw.groupby(["athlete_id", "market", "line"], dropna=False):
        key = m[mk]
        g = g.sort_values("n")
        if key in ("anytime_td", "first_td"):
            rows.append({"athlete_id": aid, "market": key, "line": None, "yes": _num(g["price"].iloc[0])})
        elif len(g) >= 2:
            o, u = _num(g["price"].iloc[0]), _num(g["price"].iloc[1])
            assumed = o is None and u is None             # ESPN's NFL feed carries DraftKings' lines without prices
            rows.append({"athlete_id": aid, "market": key, "line": float(line), "over": -110.0 if assumed else o,
                         "under": -110.0 if assumed else u, "assumed": assumed, "open_line": _num(g["open_line"].iloc[0])})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(["athlete_id", "market", "line"])
    return out, seen


FULL_WEIGHT_GAMES = {"nba": 15, "nfl": 6}   # games this season before our number gets its full MODEL_WEIGHT
BIG_GAP = 0.30                              # our chance vs the book's: beyond this the book likely knows something


def weight_for(sport: str, games: int) -> float:
    return MODEL_WEIGHT * min(1.0, max(games, 0) / FULL_WEIGHT_GAMES.get(sport, 10))


def evaluate_two_way(sims: np.ndarray, line: float, over: float | None, under: float | None, w: float = MODEL_WEIGHT) -> dict:
    sims = np.asarray(sims, float)
    p_over = float((sims > line).mean())
    p_push = float((sims == line).mean())
    p_under = 1 - p_over - p_push
    out = {"mean": float(sims.mean()), "p10": float(np.percentile(sims, 10)), "p50": float(np.median(sims)),
           "p90": float(np.percentile(sims, 90)), "p_over": p_over, "p_under": p_under, "p_push": p_push}
    if over is None or under is None:
        return out
    qo, qu = american_to_prob(over), american_to_prob(under)
    mo = qo / (qo + qu)                                   # DraftKings' no-vig chance of the over
    mo_model = p_over / max(p_over + p_under, 1e-9)       # ours, pushes set aside
    bo = w * mo_model + (1 - w) * mo
    ev_o, ev_u = ev_per_unit(bo, over), ev_per_unit(1 - bo, under)
    side = "over" if ev_o >= ev_u else "under"
    out.update(market_over=mo, model_over=mo_model, blend_over=bo, side=side, ev=max(ev_o, ev_u), weight=w,
               price=over if side == "over" else under, big_gap=abs(mo_model - mo) > BIG_GAP)
    return out


def evaluate_yes(p_model: float, yes: float | None, w: float = MODEL_WEIGHT) -> dict:
    out = {"p_yes": p_model}
    if yes is None:
        return out
    q = american_to_prob(yes)                              # includes DraftKings' margin (no other side to remove it)
    b = w * p_model + (1 - w) * q
    out.update(market_yes=q, blend_yes=b, ev=ev_per_unit(b, yes), price=yes, side="yes", weight=w)
    return out


def histogram(sims: np.ndarray, bins: int = 24) -> dict:
    sims = np.asarray(sims, float)
    lo, hi = float(np.percentile(sims, 0.5)), float(np.percentile(sims, 99.5))
    if hi <= lo:
        hi = lo + 1
    if np.allclose(sims, np.round(sims)) and hi - lo <= 30:
        edges = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
    else:
        edges = np.linspace(lo, hi, bins + 1)
    h, e = np.histogram(np.clip(sims, lo, hi), bins=edges)
    return {"edges": [round(float(x), 2) for x in e], "counts": [int(x) for x in h]}


def build_game(sport: str, market: pd.DataFrame, players: dict, seen: str | None) -> dict:
    wt = {aid: weight_for(sport, int(p.get("games") or 0)) for aid, p in players.items()}
    """players: {athlete_id: {"name", "team", "pos", "sims": {market: array}, "td": {"anytime_td": p, "first_td": p},
    "status", "proj": {...}, "log": [...]}}. Returns the props block for one game (JSON-safe)."""
    rows = []
    for r in market.to_dict("records"):
        p = players.get(r["athlete_id"])
        if not p:
            continue
        k = r["market"]
        if k in ("anytime_td", "first_td"):
            pm = (p.get("td") or {}).get(k)
            if pm is None:
                continue
            ev = evaluate_yes(pm, r.get("yes"), wt[r["athlete_id"]])
            rows.append({"athlete_id": r["athlete_id"], "market": k, "line": None, **ev})
            continue
        s = (p.get("sims") or {}).get(k)
        if s is None:
            continue
        ev = evaluate_two_way(s, r["line"], r.get("over"), r.get("under"), wt[r["athlete_id"]])
        rows.append({"athlete_id": r["athlete_id"], "market": k, "line": r["line"], "over": r.get("over"),
                     "under": r.get("under"), "assumed": bool(r.get("assumed")), "open_line": r.get("open_line"), **ev})
    # favourites: best expected value at the posted price, two-way markets, one per player, model has real data
    cand = [x for x in rows if x.get("ev") is not None and x["market"] not in ("anytime_td", "first_td")
            and x["ev"] >= FAV_MIN_EV and not x.get("big_gap") and players[x["athlete_id"]].get("reliable", True)]
    cand.sort(key=lambda x: -x["ev"])
    fav, used = [], set()
    for x in cand:
        if x["athlete_id"] in used:
            continue
        fav.append(x)
        used.add(x["athlete_id"])
        if len(fav) == N_FAV:
            break
    for i, x in enumerate(fav):
        x["favorite"] = i + 1
    pl = {}
    for aid, p in players.items():
        if any(x["athlete_id"] == aid for x in rows):
            pl[aid] = {k: v for k, v in p.items() if k not in ("sims",)}
            pl[aid]["hist"] = {k: histogram(v) for k, v in (p.get("sims") or {}).items()
                               if any(x["athlete_id"] == aid and x["market"] == k for x in rows)}
    rows.sort(key=lambda x: (ORDER.index(x["market"]) if x["market"] in ORDER else 99, -(x.get("ev") or -9)))
    return {"source": "DraftKings via ESPN", "seen": seen, "model_weight": MODEL_WEIGHT, "rows": rows,
            "favorites": [{"athlete_id": x["athlete_id"], "market": x["market"], "line": x["line"]} for x in fav],
            "players": pl}


def save(path: Path, gp: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gp, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if Path(path).exists() else None


# ----------------------------------------------------------------------------------------------- NBA
def nba_players(card: dict, P, market: pd.DataFrame, athletes: dict, injuries: list[dict], pace: float, now) -> dict:
    """Simulate every player DraftKings lists for this game (NBAPlayers P), with his game log for the player page."""
    from ..sports.nba_injuries import missing_prob
    inj = {str(e["player"]): e for e in injuries if e.get("player") is not None}
    cur = P.cur
    out = {}
    for aid in sorted(set(market["athlete_id"])) if len(market) else []:
        a = athletes.get(aid) or {}
        h = cur[cur["athlete_id"] == aid]
        team = h["team_abbreviation"].iloc[-1] if len(h) else None
        if team not in (card["home"], card["away"]):
            team = None
        pr = P.project(aid, pace_factor=pace)
        if pr is None:
            continue
        e = inj.get(aid)
        p_out, why = missing_prob(e, now) if e else (0.0, "")
        sims = P.simulate(pr)
        log = [{"date": str(r.game_date.date()), "opp": r.opponent_team_abbreviation, "home": r.home_away == "home",
                "min": float(r.minutes), "pts": float(r.points), "reb": float(r.rebounds), "ast": float(r.assists),
                "threes": float(r.three_point_field_goals_made)} for r in h.iloc[::-1].head(10).itertuples()]
        last = P.last[P.last["athlete_id"] == aid]
        avg = lambda df: None if df.empty else {"g": int(len(df)), "min": float(df["minutes"].mean()), "pts": float(df["points"].mean()),
                                                 "reb": float(df["rebounds"].mean()), "ast": float(df["assists"].mean()),
                                                 "threes": float(df["three_point_field_goals_made"].mean())}
        out[aid] = {"name": a.get("name") or (h["athlete_display_name"].iloc[-1] if len(h) else aid), "pos": a.get("pos"),
                    "team": team, "sims": sims, "status": why if p_out > 0 else "", "p_out": p_out, "games": pr["games"],
                    "reliable": pr["games"] >= 3 or pr["last_season_min"] is not None,
                    "proj": {"min": pr["min"], "games": pr["games"], "pace": pace,
                             "per36": {k: 36 * v for k, v in pr["rates"].items()}},
                    "log": log, "season": avg(h), "last5": avg(h.iloc[-5:]) if len(h) else None, "last_season": avg(last)}
    return out


# ----------------------------------------------------------------------------------------------- NFL
def espn_ids_nfl(repo: Path, dates) -> dict:
    """(away, home) -> ESPN event id, from the saved NFL scoreboards (nflverse team codes)."""
    from glob import glob
    ab = {"LAR": "LA", "WSH": "WAS"}
    out = {}
    for d in dates:
        snaps = sorted(glob(str(Path(repo) / "data" / "espn" / "nfl" / "scoreboard" / pd.Timestamp(d).strftime("%Y%m%d") / "2*.json")))
        if not snaps:
            continue
        for ev in json.loads(Path(snaps[-1]).read_text()).get("events", []):
            t = {x["homeAway"]: x["team"]["abbreviation"] for x in ev["competitions"][0]["competitors"]}
            out[(ab.get(t["away"], t["away"]), ab.get(t["home"], t["home"]))] = str(ev["id"])
    return out


def nfl_players(card: dict, all_rows: list[dict], market: pd.DataFrame, espn2gsis: dict, athletes: dict, logs: dict) -> dict:
    """Our NFL props model rows (FootballProps, with simulations) keyed by ESPN athlete id."""
    by_gsis = {r["player_id"]: r for r in all_rows}
    out = {}
    for aid in sorted(set(market["athlete_id"])) if len(market) else []:
        g = espn2gsis.get(aid)
        r = by_gsis.get(g)
        if r is None:
            continue
        a = athletes.get(aid) or {}
        out[aid] = {"name": a.get("name") or r["player"], "pos": a.get("pos"), "team": r["team"], "gsis": g,
                    "sims": dict(r.get("_sims") or {}),
                    "td": {"anytime_td": r["anytime_td"]["p_yes"], "first_td": (r.get("first_td") or {}).get("p_yes")},
                    "status": "questionable" if r.get("p_active", 1) < 1 else "", "p_out": 1 - float(r.get("p_active", 1)),
                    "games": len(logs.get(g, [])),
                    "reliable": True,
                    "proj": {"targets": r.get("proj_targets"), "carries": r.get("proj_carries"), "td_share": r.get("td_share")},
                    "log": logs.get(g, [])}
    return out


def grade(gp: dict, actual: dict) -> dict:
    """actual: {athlete_id: {market: value}} for a finished game. Adds the actual number and HIT / MISS / PUSH
    (judged on the side we leaned) to each prop, and a favourites tally. Shown only; never part of any record."""
    n_fav = hits = 0
    favkeys = {(f["athlete_id"], f["market"], f["line"]) for f in gp["favorites"]}
    for r in gp["rows"]:
        v = (actual.get(r["athlete_id"]) or {}).get(r["market"])
        if v is None:
            continue
        r["actual"] = float(v)
        if r["market"] in ("anytime_td", "first_td"):
            r["result"] = "HIT" if v else "MISS"
        elif r.get("side"):
            d = (v - r["line"]) * (1 if r["side"] == "over" else -1)
            r["result"] = "HIT" if d > 0 else "MISS" if d < 0 else "PUSH"
        if (r["athlete_id"], r["market"], r["line"]) in favkeys and r.get("result") in ("HIT", "MISS"):
            n_fav += 1
            hits += r["result"] == "HIT"
    gp["final"] = True
    gp["fav_tally"] = (hits, n_fav)
    return gp


def nba_actual(box: pd.DataFrame, game_id: str) -> dict:
    g = box[(box["game_id"].astype(str) == str(game_id)) & (box["minutes"].fillna(0) > 0)]
    out = {}
    for r in g.itertuples():
        p, rb, a, t = float(r.points), float(r.rebounds), float(r.assists), float(r.three_point_field_goals_made)
        out[str(r.athlete_id)] = {"points": p, "rebounds": rb, "assists": a, "threes": t, "pra": p + rb + a, "pr": p + rb,
                                  "pa": p + a, "ar": a + rb}
    return out


def nfl_actual(gp: dict, a: dict) -> dict:
    """a: one game of run_all.nfl_actuals (keyed by nflverse id); players inactive or without a touch count as 0."""
    out = {}
    for aid, p in gp["players"].items():
        x = a.get(p.get("gsis")) or {"receptions": 0, "receiving_yards": 0.0, "rushing_yards": 0.0, "tds": 0}
        out[aid] = {"receptions": x["receptions"], "receiving_yards": x["receiving_yards"], "rushing_yards": x["rushing_yards"],
                    "anytime_td": int(x["tds"] > 0), "first_td": int(a.get("__first_td__") == p.get("gsis"))}
    return out
