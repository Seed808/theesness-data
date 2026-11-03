"""One-off probe: NHL public API (schedule, play-by-play with shot coordinates) and ESPN NHL lines."""
import requests, json
H = {"User-Agent": "Mozilla/5.0"}
def j(u):
    r = requests.get(u, headers=H, timeout=20); r.raise_for_status(); return r.json()
try:
    s = j("https://api-web.nhle.com/v1/schedule/now")
    wk = s.get("gameWeek", [])
    games = [g for d in wk for g in d.get("games", [])]
    print(f"::notice title=nhl schedule::{len(games)} games this week; first: {games[0]['awayTeam']['abbrev']} @ {games[0]['homeTeam']['abbrev']} id {games[0]['id']}")
    done = None
    sc = j("https://api-web.nhle.com/v1/score/now")
    for g in sc.get("games", []):
        if g.get("gameState") in ("OFF", "FINAL"):
            done = g["id"]; break
    gid = done or games[0]["id"]
    pbp = j(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
    plays = pbp.get("plays", [])
    shots = [p for p in plays if p.get("typeDescKey") in ("shot-on-goal", "goal", "missed-shot", "blocked-shot")]
    print(f"::notice title=nhl pbp::game {gid}: {len(plays)} plays, {len(shots)} shot attempts; sample {json.dumps(shots[0]['details'])[:300] if shots else 'none'}")
    print(f"::notice title=nhl season pbp::previous season game id example check {j('https://api-web.nhle.com/v1/gamecenter/2025020001/play-by-play').get('id')}")
except Exception as ex:
    print(f"::notice title=nhl api error::{type(ex).__name__}: {ex}")
try:
    sb = j("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard")
    ev = sb.get("events", [])
    print(f"::notice title=espn nhl::{len(ev)} games today; lines: " + "; ".join(f"{e['shortName']} {(e['competitions'][0].get('odds') or [{}])[0].get('details')} o/u {(e['competitions'][0].get('odds') or [{}])[0].get('overUnder')}" for e in ev[:6]))
except Exception as ex:
    print(f"::notice title=espn error::{ex}")
