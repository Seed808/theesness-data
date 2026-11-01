"""One-off probe: which sportsbooks and player-prop markets ESPN's free feed carries for today's NFL and NBA games."""
import requests, collections
H = {"User-Agent": "Mozilla/5.0"}
CORE = "https://sports.core.api.espn.com/v2/sports"
def j(u, **p):
    r = requests.get(u, params=p, headers=H, timeout=20); r.raise_for_status(); return r.json()
for sport, path in (("nfl", "football/leagues/nfl"), ("nba", "basketball/leagues/nba")):
    sb = j(f"https://site.api.espn.com/apis/site/v2/sports/{path.replace('leagues/','')}/scoreboard")
    evs = sb.get("events", [])[:3]
    for ev in evs:
        eid = ev["id"]
        try:
            od = j(f"{CORE}/{path}/events/{eid}/competitions/{eid}/odds")
        except Exception as ex:
            print(f"::notice title={sport} {ev['shortName']}::odds error {ex}"); continue
        provs = []
        for it in od.get("items", []):
            p = it.get("provider", {}); provs.append(f'{p.get("name")}({p.get("id")})')
        print(f"::notice title={sport} {ev['shortName']} game-line books::{', '.join(provs)}")
        for it in od.get("items", []):
            pid = it.get("provider", {}).get("id")
            try:
                pb = j(f"{CORE}/{path}/events/{eid}/competitions/{eid}/odds/{pid}/propBets", limit=1000)
            except Exception:
                continue
            items = pb.get("items", [])
            if not items:
                continue
            kinds = collections.Counter((x.get("type") or {}).get("name", "?") for x in items)
            print(f"::notice title={sport} {ev['shortName']} props from {it['provider'].get('name')}::{len(items)} props; markets: "
                  + "; ".join(f"{k} {v}" for k, v in kinds.most_common(14)))
