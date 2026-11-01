"""
Collect DraftKings player props (lines + prices) from ESPN's public feed for NFL and NBA games that have not started.

Saved per game (only the newest snapshot and the first one seen are kept, as compact CSV):
  data/espn/{sport}/props/{event_id}/latest.csv   first.csv
Columns: event_id, athlete_id, team_id, type_id, market, line, price, open_line, open_price, n, updated
  n = position of the row within (athlete, market, line): ESPN lists a two-way prop as two rows, over first, then under.
Player names/teams/positions are cached in data/espn/{sport}/athletes.json.

  python scripts/fetch_props.py              # NFL this week + NBA today/tomorrow (Hawaii)
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0 (personal sports model)"}
CORE = "https://sports.core.api.espn.com/v2/sports"
SITE = "https://site.api.espn.com/apis/site/v2/sports"
PATH = {"nfl": ("football", "nfl"), "nba": ("basketball", "nba")}
ROOT = Path(__file__).resolve().parents[1] / "data" / "espn"
COLS = ["event_id", "athlete_id", "team_id", "type_id", "market", "line", "price", "open_line", "open_price", "n", "updated"]


def get(u, **p):
    r = requests.get(u.replace("http://", "https://"), params=p, headers=H, timeout=25)
    r.raise_for_status()
    return r.json()


def rid(ref: str | None, word: str) -> str:
    m = re.search(rf"/{word}/(\d+)", ref or "")
    return m.group(1) if m else ""


def events(sport: str, now: datetime) -> list[dict]:
    sp, lg = PATH[sport]
    out = []
    days = [None] if sport == "nfl" else [(now - timedelta(hours=10) + timedelta(days=k)).strftime("%Y%m%d") for k in (0, 1)]
    for d in days:
        sb = get(f"{SITE}/{sp}/{lg}/scoreboard", **({"dates": d} if d else {}))
        for ev in sb.get("events", []):
            st = ((ev.get("status") or {}).get("type") or {}).get("state")
            if st == "pre":
                out.append(ev)
    return out


def props(sport: str, eid: str) -> list[dict]:
    sp, lg = PATH[sport]
    rows, page = [], 1
    while True:
        pb = get(f"{CORE}/{sp}/leagues/{lg}/events/{eid}/competitions/{eid}/odds/100/propBets", limit=1000, page=page)
        rows += pb.get("items", [])
        if page >= int(pb.get("pageCount") or 1):
            return rows
        page += 1


def main():
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    for sport in PATH:
        try:
            evs = events(sport, now)
        except Exception as ex:
            print(f"{sport}: scoreboard error {type(ex).__name__}")
            continue
        ap = ROOT / sport / "athletes.json"
        ath = json.loads(ap.read_text()) if ap.exists() else {}
        saved = 0
        for ev in evs:
            eid = str(ev["id"])
            try:
                items = props(sport, eid)
            except Exception as ex:
                print(f"{sport} {eid}: props error {type(ex).__name__}")
                continue
            out, count = [], {}
            for x in items:
                a = rid((x.get("athlete") or {}).get("$ref"), "athletes")
                t = rid((x.get("team") or {}).get("$ref"), "teams")
                ty = x.get("type") or {}
                cur = ((x.get("current") or {}).get("target") or {}).get("value")
                opn = ((x.get("open") or {}).get("target") or {}).get("value")
                od = (x.get("odds") or {}).get("american") or {}
                key = (a, t, ty.get("id"), cur)
                count[key] = count.get(key, 0) + 1
                out.append([eid, a, t, ty.get("id"), ty.get("name"), cur, od.get("value"), opn, od.get("open"), count[key],
                            x.get("lastUpdated")])
                if a and a not in ath:
                    ath[a] = None                                   # filled below
            if not out:
                continue
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(COLS)
            w.writerows(out)
            d = ROOT / sport / "props" / eid
            d.mkdir(parents=True, exist_ok=True)
            txt = buf.getvalue()
            prev = (d / "latest.csv").read_text() if (d / "latest.csv").exists() else None
            if prev != txt:
                (d / "latest.csv").write_text(txt)
                (d / "seen_at.txt").write_text(stamp)
                saved += 1
            if not (d / "first.csv").exists():
                (d / "first.csv").write_text(txt)
        # names for new athletes
        sp, lg = PATH[sport]
        for a, v in list(ath.items()):
            if v and v.get("name"):
                continue
            try:
                j = get(f"{CORE}/{sp}/leagues/{lg}/athletes/{a}")
                ath[a] = {"name": j.get("displayName"), "short": j.get("shortName"),
                          "pos": (j.get("position") or {}).get("abbreviation"),
                          "team_id": rid((j.get("team") or {}).get("$ref"), "teams")}
            except Exception:
                ath[a] = {"name": None}
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(json.dumps(ath, indent=0, sort_keys=True))
        print(f"{sport}: {len(evs)} games before start, {saved} prop files updated")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:                       # never fail the collector
        print(f"props: {type(ex).__name__}: {ex}")
