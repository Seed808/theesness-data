"""
Snapshot ESPN scoreboards (schedule, scores, betting lines) and injury lists.

Raw JSON is saved, not parsed, so nothing is lost if ESPN changes a field; the model parses it.
A new snapshot is written only when the content that matters (odds, status, scores, injuries)
changed, and the first snapshot of every game is kept forever: that is the "first line seen"
the model locks against.

    python scripts/fetch_espn.py              # today + next 7 days, all leagues, plus injuries
    python scripts/fetch_espn.py --days 0     # today only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports"
LEAGUES = {
    "nfl": "football/nfl",
    "ncaaf": "football/college-football",
    "nba": "basketball/nba",
    "cbb": "basketball/mens-college-basketball",
    "mlb": "baseball/mlb",
    "nhl": "hockey/nhl",
}
EXTRA = {"ncaaf": {"groups": "80", "limit": "400"}, "cbb": {"groups": "50", "limit": "400"}}
INJURIES = {"nba": "basketball/nba", "nfl": "football/nfl"}
ROOT = Path(__file__).resolve().parents[1] / "data" / "espn"
UA = {"User-Agent": "Mozilla/5.0 (data collection for a personal sports model)"}


def get(url, params=None):
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def fingerprint(obj) -> str:
    """Hash of the parts that matter, so unchanged polls don't create files."""
    keep = []
    for ev in obj.get("events", []):
        c = (ev.get("competitions") or [{}])[0]
        keep.append([ev.get("id"), ev.get("status", {}).get("type", {}).get("name"), c.get("odds"),
                     [(t.get("homeAway"), t.get("score")) for t in c.get("competitors", [])]])
    if "injuries" in obj:
        keep = obj["injuries"]
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save(dirpath: Path, obj, stamp: str) -> bool:
    dirpath.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(obj)
    last = dirpath / "latest.fingerprint"
    if last.exists() and last.read_text() == fp:
        return False
    (dirpath / f"{stamp}.json").write_text(json.dumps(obj, separators=(",", ":")))
    last.write_text(fp)
    first = dirpath / "first.json"
    if not first.exists():
        first.write_text(json.dumps(obj, separators=(",", ":")))
    return True


def main(days: int, leagues: list[str]):
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%MZ")
    written = 0
    for lg in leagues:
        for d in range(-1, days + 1):          # yesterday too: late finals
            day = (now - timedelta(hours=10) + timedelta(days=d)).strftime("%Y%m%d")   # HST calendar day
            try:
                obj = get(f"{BASE}/{LEAGUES[lg]}/scoreboard", {"dates": day, **EXTRA.get(lg, {})})
            except Exception as e:  # keep going: one league failing must not stop the rest
                print(f"{lg} {day}: {e}")
                continue
            if obj.get("events"):
                written += save(ROOT / lg / "scoreboard" / day, obj, stamp)
    for lg, path in INJURIES.items():
        if lg not in leagues:
            continue
        try:
            written += save(ROOT / lg / "injuries", get(f"{BASE}/{path}/injuries"), stamp)
        except Exception as e:
            print(f"{lg} injuries: {e}")
    print(f"snapshots written: {written}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--leagues", default=",".join(LEAGUES))
    a = ap.parse_args()
    main(a.days, a.leagues.split(","))
