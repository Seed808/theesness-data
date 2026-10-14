"""
MLB data for the model:
  * Statcast pitch-level data (pybaseball) -> data/mlb/statcast/<season>/<YYYY-MM>.parquet, incremental by day
  * MLB Stats API schedule with probable pitchers and posted lineups -> data/mlb/schedule/<date>.json

    python scripts/fetch_mlb.py                        # yesterday's Statcast + schedule for yesterday..+2 days
    python scripts/fetch_mlb.py --statcast-from 2026-03-26   # backfill a season (run once)
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1] / "data" / "mlb"
COLS = ["game_pk", "game_date", "game_type", "home_team", "away_team", "inning", "inning_topbot", "at_bat_number",
        "pitch_number", "batter", "pitcher", "player_name", "stand", "p_throws", "events", "description",
        "woba_value", "woba_denom", "estimated_woba_using_speedangle", "launch_speed", "launch_angle",
        "bat_score", "post_bat_score", "home_score", "away_score", "post_home_score", "post_away_score",
        "outs_when_up", "on_1b", "on_2b", "on_3b", "pitch_type", "release_speed"]
API = "https://statsapi.mlb.com/api/v1/schedule"


def final_games(d0: date, d1: date) -> dict[int, str]:
    """{gamePk: gameType} the MLB Stats API marks Final between d0 and d1 (a game in progress is never saved)."""
    r = requests.get(API, params={"sportId": 1, "startDate": d0.isoformat(), "endDate": d1.isoformat()}, timeout=60)
    r.raise_for_status()
    return {g["gamePk"]: g["gameType"] for d in r.json().get("dates", []) for g in d["games"]
            if g["status"].get("abstractGameState") == "Final"}


# wOBA weights used for fallback rows (FanGraphs-scale, close to Savant's woba_value)
WOBA = {"walk": 0.69, "hit_by_pitch": 0.72, "single": 0.89, "double": 1.27, "triple": 1.62, "home_run": 2.10}
NO_DENOM = {"intent_walk", "sac_bunt", "catcher_interf", "sac_bunt_double_play"}
# plays that end without the batter's plate appearance finishing (kept for the score, not counted as a PA)
NOT_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "balk", "other_advance",
          "runner_double_play", "other_out", "game_advisory")


def gumbo_rows(pk: int) -> pd.DataFrame:
    """One row per plate appearance from the MLB Stats API live feed, in Statcast's columns, for a final game
    Baseball Savant has not posted yet. No xwOBA (Savant computes it), so the model uses actual wOBA for these."""
    f = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", timeout=60).json()
    gd, plays = f["gameData"], f["liveData"]["plays"]["allPlays"]
    away, home = gd["teams"]["away"]["abbreviation"], gd["teams"]["home"]["abbreviation"]
    players = gd.get("players", {})
    rows, a_sc, h_sc = [], 0, 0
    for pl in plays:
        res, ab, m = pl["result"], pl["about"], pl["matchup"]
        ev = res.get("eventType")
        if res.get("type") != "atBat" or not ev:
            a_sc, h_sc = res.get("awayScore", a_sc), res.get("homeScore", h_sc)
            continue
        top = ab["halfInning"] == "top"
        if ev.startswith(NOT_PA):
            ev = None
        pid = m["pitcher"]["id"]
        pinfo = players.get(f"ID{pid}", {})
        hit = next((e.get("hitData") for e in reversed(pl.get("playEvents", [])) if e.get("hitData")), {}) or {}
        na, nh = res.get("awayScore", a_sc), res.get("homeScore", h_sc)
        rows.append({"game_pk": pk, "game_date": gd["datetime"]["officialDate"], "game_type": gd["game"]["type"],
                     "home_team": home, "away_team": away, "inning": ab["inning"], "inning_topbot": "Top" if top else "Bot",
                     "at_bat_number": pl["atBatIndex"] + 1, "pitch_number": len([e for e in pl.get("playEvents", []) if e.get("isPitch")]),
                     "batter": m["batter"]["id"], "pitcher": pid,
                     "player_name": f'{pinfo.get("lastName", "")}, {pinfo.get("useName") or pinfo.get("firstName", "")}',
                     "stand": m.get("batSide", {}).get("code"), "p_throws": m.get("pitchHand", {}).get("code"),
                     "events": ev, "description": None, "woba_value": WOBA.get(ev, 0.0) if ev else None,
                     "woba_denom": (0 if ev in NO_DENOM else 1) if ev else None, "estimated_woba_using_speedangle": None,
                     "launch_speed": hit.get("launchSpeed"), "launch_angle": hit.get("launchAngle"),
                     "bat_score": a_sc if top else h_sc, "post_bat_score": na if top else nh,
                     "home_score": h_sc, "away_score": a_sc, "post_home_score": nh, "post_away_score": na,
                     "outs_when_up": None, "on_1b": None, "on_2b": None, "on_3b": None, "pitch_type": None,
                     "release_speed": None, "source": "mlb_statsapi"})
        a_sc, h_sc = na, nh
    return pd.DataFrame(rows)


def save(df: pd.DataFrame):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    for month, part in df.groupby(df["game_date"].str[:7]):
        path = ROOT / "statcast" / month[:4] / f"{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_parquet(path)
            part = pd.concat([old[~old["game_pk"].isin(part["game_pk"])], part])
        part.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).to_parquet(path, index=False)


def fill_missing(d0: date, d1: date):
    """Final games (regular season / postseason) with no Savant rows yet: add rows from the MLB live feed."""
    have = set()
    for f in (ROOT / "statcast").glob("*/*.parquet"):
        x = pd.read_parquet(f, columns=["game_pk", "game_date"])
        have |= set(x.loc[x["game_date"] >= d0.isoformat(), "game_pk"])
    for pk, gt in final_games(d0, d1).items():
        if gt in ("R", "F", "D", "L", "W") and pk not in have:
            rows = gumbo_rows(pk)
            if len(rows):
                save(rows)
                print(f"fallback {pk}: {len(rows)} plate appearances from the MLB live feed (Savant not posted yet)")


def statcast_days(d0: date, d1: date):
    from pybaseball import statcast
    d = d0
    while d <= d1:
        chunk_end = min(d + timedelta(days=6), d1)
        df = statcast(start_dt=d.isoformat(), end_dt=chunk_end.isoformat(), verbose=False)
        n_raw = 0 if df is None else len(df)
        if df is not None and len(df):
            fg = final_games(d - timedelta(days=1), chunk_end + timedelta(days=1))
            print(f"statcast {d}..{chunk_end}: {n_raw} pitches from Savant, games {sorted(df['game_pk'].unique().tolist())}, final per MLB API {sorted(fg)}")
            df = df[df["game_pk"].isin(set(fg))]
        else:
            print(f"statcast {d}..{chunk_end}: Savant returned nothing")
        if df is not None and len(df):
            df = df[[c for c in COLS if c in df.columns]].copy()
            df["source"] = "savant"
            save(df)
            print(f"statcast {d}..{chunk_end}: {len(df)} pitches")
        d = chunk_end + timedelta(days=1)


def schedule(days):
    for d in days:
        r = requests.get(API, params={"sportId": 1, "date": d.isoformat(),
                                      "hydrate": "probablePitcher,lineups,linescore,team,venue"}, timeout=30)
        r.raise_for_status()
        p = ROOT / "schedule" / f"{d.isoformat()}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r.json(), separators=(",", ":")))
        if not (ROOT / "schedule" / f"{d.isoformat()}.first.json").exists():
            (ROOT / "schedule" / f"{d.isoformat()}.first.json").write_text(p.read_text())


def main(statcast_from: str | None):
    today = (datetime.now(timezone.utc) - timedelta(hours=10)).date()
    # through today (Hawaii date): finished games only, so a same-day rerun picks up tonight's finals
    if statcast_from:
        statcast_days(date.fromisoformat(statcast_from), today)
    else:
        statcast_days(today - timedelta(days=2), today)
    fill_missing(today - timedelta(days=3), today)
    schedule([today + timedelta(days=k) for k in (-1, 0, 1, 2)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--statcast-from")
    main(ap.parse_args().statcast_from)
