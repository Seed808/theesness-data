"""MLB schedule, probable pitchers and posted lineups for yesterday..+2 days (Hawaii dates), every collector run,
so lineups posted a few hours before first pitch are captured. Same files as fetch_mlb.py (requests only)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "data" / "mlb" / "schedule"
API = "https://statsapi.mlb.com/api/v1/schedule"

if __name__ == "__main__":
    today = (datetime.now(timezone.utc) - timedelta(hours=10)).date()
    ROOT.mkdir(parents=True, exist_ok=True)
    for k in (-1, 0, 1, 2):
        d = (today + timedelta(days=k)).isoformat()
        r = requests.get(API, params={"sportId": 1, "date": d, "hydrate": "probablePitcher,lineups,linescore,team,venue"},
                         timeout=30)
        r.raise_for_status()
        p = ROOT / f"{d}.json"
        p.write_text(json.dumps(r.json(), separators=(",", ":")))
        if not (ROOT / f"{d}.first.json").exists():
            (ROOT / f"{d}.first.json").write_text(p.read_text())
