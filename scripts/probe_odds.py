"""One-off probe: print raw structure of ESPN/DraftKings propBets items (a few per market type) for one NFL and one NBA game."""
import requests, json
H = {"User-Agent": "Mozilla/5.0"}
CORE = "https://sports.core.api.espn.com/v2/sports"
def j(u, **p):
    r = requests.get(u, params=p, headers=H, timeout=20); r.raise_for_status(); return r.json()
for sport, path in (("nfl", "football/leagues/nfl"), ("nba", "basketball/leagues/nba")):
    sb = j(f"https://site.api.espn.com/apis/site/v2/sports/{path.replace('leagues/','')}/scoreboard")
    ev = sb["events"][0]; eid = ev["id"]
    pb = j(f"{CORE}/{path}/events/{eid}/competitions/{eid}/odds/100/propBets", limit=1000)
    print(f"::notice title={sport} paging::count={pb.get('count')} pageCount={pb.get('pageCount')} pageSize={pb.get('pageSize')}")
    seen = {}
    for x in pb["items"]:
        k = (x.get("type") or {}).get("name")
        seen.setdefault(k, []).append(x)
    for k in ["Total Rushing Yards (incl. overtime)", "Total Receptions (incl. overtime)", "Anytime Touchdown Scorer", "First Touchdown Scorer",
              "Rushing Yards Milestones", "Total Points", "Total Rebounds", "Total 3-Point Field Goals", "Points Milestones"]:
        if k in seen:
            for x in seen[k][:2]:
                s = json.dumps(x)[:900].replace("\n", " ")
                print(f"::notice title={sport} {k}::{s}")
    a = pb["items"][0].get("athlete", {}).get("$ref")
    if a:
        print(f"::notice title={sport} athlete::{json.dumps(j(a.replace('http://','https://')))[:600]}")
    print(f"::notice title={sport} types::" + " | ".join(sorted(seen)))
