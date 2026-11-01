"""One-off probe: look for prices in ESPN/DraftKings propBets, and check NBA."""
import requests, json, collections, traceback
H = {"User-Agent": "Mozilla/5.0"}
CORE = "https://sports.core.api.espn.com/v2/sports"
def j(u, **p):
    r = requests.get(u, params=p, headers=H, timeout=20); r.raise_for_status(); return r.json()
def walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, pre + "." + k)
    elif isinstance(o, list):
        for v in o[:3]: yield from walk(v, pre + "[]")
    else: yield pre
for sport, path in (("nba", "basketball/leagues/nba"), ("nfl", "football/leagues/nfl")):
    try:
        sb = j(f"https://site.api.espn.com/apis/site/v2/sports/{path.replace('leagues/','')}/scoreboard")
        ev = sb["events"][0]; eid = ev["id"]
        pb = j(f"{CORE}/{path}/events/{eid}/competitions/{eid}/odds/100/propBets", limit=1000)
        keys = collections.Counter(k for x in pb["items"] for k in set(walk(x)))
        print(f"::notice title={sport} keys::" + " | ".join(f"{k} {v}" for k, v in keys.most_common(40)))
        tot = [x for x in pb["items"] if (x.get("type") or {}).get("name", "").startswith("Total Points")][:2]
        print(f"::notice title={sport} sample::{json.dumps(tot)[:1500]}")
        # summary endpoint pickcenter / odds
        s = j(f"https://site.api.espn.com/apis/site/v2/sports/{path.replace('leagues/','')}/summary", event=eid)
        print(f"::notice title={sport} summary keys::{sorted(s.keys())}")
        od = j(f"{CORE}/{path}/events/{eid}/competitions/{eid}/odds/100")
        print(f"::notice title={sport} odds100 keys::{sorted(od.keys())}")
    except Exception as ex:
        print(f"::notice title={sport} error::{traceback.format_exc()[-400:]}")
