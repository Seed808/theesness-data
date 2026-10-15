"""ESPN NFL depth charts (QB1/QB2... per team), saved when they change. Used to cross-check the starting quarterback:
the model's starter must agree with the last game's play-by-play and the injury report, or the game is held.

    python scripts/fetch_espn_depth.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "data" / "espn" / "nfl" / "depth"
UA = {"User-Agent": "Mozilla/5.0 (data collection for a personal sports model)"}
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"


def get(url, params=None):
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def athlete_name(ref_or_obj, cache):
    if isinstance(ref_or_obj, dict) and "$ref" in ref_or_obj:
        u = ref_or_obj["$ref"]
        if u not in cache:
            try:
                a = get(u)
                cache[u] = {"id": a.get("id"), "name": a.get("displayName") or a.get("fullName")}
            except Exception:
                cache[u] = {"id": u.rstrip("/").split("/")[-1].split("?")[0], "name": None}
        return cache[u]
    if isinstance(ref_or_obj, dict):
        return {"id": ref_or_obj.get("id"), "name": ref_or_obj.get("displayName") or ref_or_obj.get("fullName")}
    return {"id": None, "name": None}


def team_qbs(team_id: str, season: int, cache) -> tuple[list[dict], str]:
    """Ordered QB depth for one team, and which endpoint answered."""
    try:   # site API (athletes inline)
        d = get(f"{SITE}/teams/{team_id}/depthcharts")
        for chart in d.get("depthchart", d.get("items", [])):
            pos = chart.get("positions", {})
            q = pos.get("qb") or pos.get("QB")
            if q:
                return [athlete_name(a, cache) for a in q.get("athletes", [])], "site"
    except Exception:
        pass
    d = get(f"{CORE}/seasons/{season}/teams/{team_id}/depthcharts")   # core API ($ref links)
    for chart in d.get("items", []):
        pos = chart.get("positions", {})
        q = pos.get("qb") or pos.get("QB")
        if q:
            ath = sorted(q.get("athletes", []), key=lambda a: a.get("slot", 99))
            return [athlete_name(a.get("athlete", a), cache) for a in ath], "core"
    return [], "none"


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    teams = get(f"{SITE}/teams")["sports"][0]["leagues"][0]["teams"]
    cache, out, used = {}, {}, {}
    for t in teams:
        tm = t["team"]
        try:
            qbs, src = team_qbs(tm["id"], now.year, cache)
        except Exception as e:
            print(f"{tm['abbreviation']}: {e}")
            continue
        out[tm["abbreviation"]] = qbs
        used[src] = used.get(src, 0) + 1
    print("depth charts:", len(out), "teams; endpoints used:", used, "; sample:", {k: v[:2] for k, v in list(out.items())[:3]})
    ROOT.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(out, sort_keys=True)
    fp = hashlib.sha256(blob.encode()).hexdigest()[:16]
    last = ROOT / "latest.fingerprint"
    if not (last.exists() and last.read_text() == fp):
        (ROOT / f"{now.strftime('%Y%m%dT%H%MZ')}.json").write_text(json.dumps({"fetched_at": now.isoformat(), "qb": out}))
        (ROOT / "latest.json").write_text(json.dumps({"fetched_at": now.isoformat(), "qb": out}))
        last.write_text(fp)
