"""One-off snapshot of Sam's public model pages (board + game pages) for calibration.
Usage: python scripts/fetch_sam.py ncaaf 7   -> data/sam/ncaaf_wk7/*.html"""
import re, sys, time
from pathlib import Path
import requests
BASE = "https://samthemodelman.smmodel.workers.dev"
league, week = sys.argv[1], sys.argv[2]
out = Path(f"data/sam/{league}_wk{week}"); out.mkdir(parents=True, exist_ok=True)
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (theesness-data calibration snapshot)"
board = s.get(f"{BASE}/{league}/", timeout=30); board.raise_for_status()
(out / "board.html").write_text(board.text)
ids = sorted(set(re.findall(rf'/?{league}/game/(\d+)\.html|game/(\d+)\.html', board.text)))
ids = sorted({a or b for a, b in ids})
print(len(ids), "game pages")
for gid in ids:
    r = s.get(f"{BASE}/{league}/game/{gid}.html", timeout=30)
    print(gid, r.status_code)
    if r.ok:
        (out / f"{gid}.html").write_text(r.text)
    time.sleep(0.5)
