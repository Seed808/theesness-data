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


def final_games(d0: date, d1: date) -> set[int]:
    """gamePks the MLB Stats API marks Final between d0 and d1 (a game in progress is never saved)."""
    r = requests.get(API, params={"sportId": 1, "startDate": d0.isoformat(), "endDate": d1.isoformat()}, timeout=60)
    r.raise_for_status()
    return {g["gamePk"] for d in r.json().get("dates", []) for g in d["games"]
            if g["status"].get("abstractGameState") == "Final"}


def statcast_days(d0: date, d1: date):
    from pybaseball import statcast
    d = d0
    while d <= d1:
        chunk_end = min(d + timedelta(days=6), d1)
        df = statcast(start_dt=d.isoformat(), end_dt=chunk_end.isoformat(), verbose=False)
        if df is not None and len(df):
            df = df[df["game_pk"].isin(final_games(d - timedelta(days=1), chunk_end + timedelta(days=1)))]
        if df is not None and len(df):
            df = df[[c for c in COLS if c in df.columns]].copy()
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
            for month, part in df.groupby(df["game_date"].str[:7]):
                path = ROOT / "statcast" / month[:4] / f"{month}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    old = pd.read_parquet(path)
                    part = pd.concat([old[~old["game_pk"].isin(part["game_pk"])], part])
                part.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).to_parquet(path, index=False)
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
    schedule([today + timedelta(days=k) for k in (-1, 0, 1, 2)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--statcast-from")
    main(ap.parse_args().statcast_from)
