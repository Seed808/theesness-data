"""
Game-finished watcher (runs every 10 minutes from .github/workflows/watch.yml).

Starts a site build as soon as a game with a live pick is final on ESPN but not yet graded in the ledger,
instead of waiting for the next :19/:49 build. It never edits picks or results itself.

  python scripts/watch_results.py            # prints what it found; exits 0 always
  Writes "trigger=true|false" to $GITHUB_OUTPUT for the workflow.

A pick counts as waiting when its newest ledger row has no result and is not voided / on hold / no edge.
Matching: NBA and college by ESPN event id (the ledger uses ESPN ids); NFL and MLB by team abbreviations.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports"
LEAGUES = {"nfl": "football/nfl", "ncaaf": "football/college-football", "nba": "basketball/nba",
           "mlb": "baseball/mlb", "cbb": "basketball/mens-college-basketball"}
EXTRA = {"ncaaf": {"groups": "80", "limit": "400"}, "cbb": {"groups": "50", "limit": "400"}}
ABBR = {"LAR": "LA", "WSH": "WAS"}                    # ESPN -> nflverse (NFL)
LEDGER = Path(__file__).resolve().parents[1] / "model" / "ledger"
UA = {"User-Agent": "Mozilla/5.0 (personal sports model)"}


def waiting_picks(sport: str) -> list[dict]:
    p, v = LEDGER / f"{sport}_locks.csv", LEDGER / f"{sport}_locks_voids.csv"
    if not p.exists():
        return []
    with p.open() as f:
        rows = list(csv.DictReader(f))
    voided = set()
    if v.exists():
        with v.open() as f:
            voided = {r["lock_hash"] for r in csv.DictReader(f)}
    last = {}
    for r in rows:
        last[r["game_id"]] = r
    return [r for r in last.values() if not (r.get("result") or "").strip() and r["hash"] not in voided]


def finals(sport: str, now: datetime) -> tuple[set, set]:
    """(completed ESPN event ids, completed (away, home) pairs) for yesterday and today, Hawaii dates."""
    ids, pairs = set(), set()
    for k in (1, 0):
        day = (now - timedelta(hours=10) - timedelta(days=k)).strftime("%Y%m%d")
        try:
            r = requests.get(f"{BASE}/{LEAGUES[sport]}/scoreboard", params={"dates": day, **EXTRA.get(sport, {})},
                             headers=UA, timeout=20)
            r.raise_for_status()
            evs = r.json().get("events", [])
        except Exception as ex:
            print(f"{sport} {day}: {type(ex).__name__}")
            continue
        for ev in evs:
            c = ev["competitions"][0]
            if not ((c.get("status") or ev.get("status") or {}).get("type") or {}).get("completed"):
                continue
            ids.add(str(ev["id"]))
            t = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
            m = ABBR if sport == "nfl" else {}
            pairs.add((m.get(t.get("away"), t.get("away")), m.get(t.get("home"), t.get("home"))))
    return ids, pairs


def main() -> bool:
    now = datetime.now(timezone.utc)
    hits = []
    for sport in LEAGUES:
        wait = waiting_picks(sport)
        if not wait:
            continue
        ids, pairs = finals(sport, now)
        for r in wait:
            if str(r["game_id"]) in ids or (r.get("away"), r.get("home")) in pairs:
                hits.append(f'{sport} {r.get("away")} @ {r.get("home")}')
    for h in hits:
        print(f"final, not graded yet: {h}")
        print(f"::notice title=watch::final, not graded yet: {h}")
    if not hits:
        print("nothing waiting")
    return bool(hits)


if __name__ == "__main__":
    try:
        trig = main()
    except Exception as ex:                                  # never fail the workflow
        print(f"watcher error: {type(ex).__name__}: {ex}")
        trig = False
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"trigger={'true' if trig else 'false'}\n")
