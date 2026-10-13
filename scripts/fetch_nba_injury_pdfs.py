"""
Download the official NBA injury report PDFs (published by the league on game days).

URL pattern: https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf
Older seasons use times like 05PM; newer ones may use 05_00PM / 05_15PM. Both are tried.
PDFs are saved untouched; the model parses them.

    python scripts/fetch_nba_injury_pdfs.py                       # recent slots for today (ET)
    python scripts/fetch_nba_injury_pdfs.py --start 2025-10-21 --end 2026-04-12 --slots 17:00,18:30
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
ROOT = Path(__file__).resolve().parents[1] / "data" / "nba_injury_pdfs"
UA = {"User-Agent": "Mozilla/5.0 (data collection for a personal sports model)"}


def time_codes(h: int, m: int) -> list[str]:
    ap = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    codes = [f"{h12:02d}_{m:02d}{ap}"]
    if m == 0:
        codes.append(f"{h12:02d}{ap}")
    return codes


def fetch(d: date, h: int, m: int) -> str | None:
    for t in time_codes(h, m):
        name = f"Injury-Report_{d.isoformat()}_{t}.pdf"
        dest = ROOT / str(d.year) / name
        if dest.exists():
            return name
        r = requests.get(BASE.format(d=d.isoformat(), t=t), headers=UA, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return name
    return None


def main(start: str | None, end: str | None, slots: str | None):
    if start:
        d0, d1 = date.fromisoformat(start), date.fromisoformat(end or start)
        times = [tuple(map(int, s.split(":"))) for s in (slots or "17:00").split(",")]
        n = 0
        d = d0
        while d <= d1:
            for h, m in times:
                n += fetch(d, h, m) is not None
            d += timedelta(days=1)
        print(f"backfill saved/found: {n}")
        return
    # live: the last 3 hours of 15-minute slots, US Eastern (UTC-4 in season start, UTC-5 after DST ends)
    now = datetime.now(timezone.utc) - timedelta(hours=4 if datetime.now().month in (4, 5, 6, 7, 8, 9, 10) else 5)
    got = []
    for k in range(12):
        t = now - timedelta(minutes=15 * k)
        t = t.replace(minute=(t.minute // 15) * 15)
        name = fetch(t.date(), t.hour, t.minute)
        if name:
            got.append(name)
            break   # newest report found; older ones of the day are already stored
    print("latest:", got or "none found in the last 3 hours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--slots")
    a = ap.parse_args()
    main(a.start, a.end, a.slots)
