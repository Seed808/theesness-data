"""CollegeFootballData (CFBD) pulls: Elo ratings and games (scores + pre/postgame Elo).
Needs the CFBD_API_KEY secret. Free tier allows 1,000 calls a month; this uses 2-3 per run.
Writes data/cfbd/games_{year}.json and data/cfbd/elo_{year}_wk{week}.json (Elo going into that week)."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import requests

KEY = os.environ.get("CFBD_API_KEY", "").strip()
BASE = "https://api.collegefootballdata.com"
OUT = Path("data/cfbd")


def get(path, **params):
    r = requests.get(BASE + path, params=params, headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"}, timeout=60)
    print(path, params, r.status_code, len(r.content))
    r.raise_for_status()
    return r.json()


def main():
    if not KEY:
        print("CFBD_API_KEY not set"); sys.exit(1)
    year = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now(timezone.utc).year
    OUT.mkdir(parents=True, exist_ok=True)
    games = get("/games", year=year, seasonType="regular", division="fbs")
    (OUT / f"games_{year}.json").write_text(json.dumps(games))
    # the next week to be played = earliest week with an unfinished game
    now = datetime.now(timezone.utc).isoformat()
    upcoming = sorted({g["week"] for g in games if not g.get("completed") and (g.get("startDate") or "") >= now[:10]})
    weeks = upcoming[:1]
    for w in weeks:
        elo = get("/ratings/elo", year=year, week=w)      # CFBD: ratings as of the start of that week
        (OUT / f"elo_{year}_wk{w}.json").write_text(json.dumps(elo))
    (OUT / "last_run.json").write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "year": year,
                                                    "games": len(games), "elo_weeks": weeks}))


if __name__ == "__main__":
    main()
