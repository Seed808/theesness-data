"""
NHL play-by-play from the league's public API (api-web.nhle.com), saved compact for the hockey model.

  data/nhl/shots_{season}.parquet   one row per unblocked shot attempt (shot, miss, goal) with location
  data/nhl/games_{season}.csv       one row per finished game: teams, names, start time, final score,
                                     how it ended (REG / OT / SO), each team's starting goalie

Regular season only. Incremental: games already saved are not fetched again.
  python scripts/fetch_nhl.py                 # 2025-26 (if missing) + 2026-27 to date
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0 (personal sports model)"}
URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
OUT = Path(__file__).resolve().parents[1] / "data" / "nhl"
SHOTS = {"shot-on-goal", "missed-shot", "goal"}


def parse(js: dict) -> tuple[list[dict], dict]:
    home, away = js["homeTeam"], js["awayTeam"]
    hid = home["id"]
    rows, prev, start_goalie = [], None, {}
    for p in js.get("plays", []):
        t = p.get("typeDescKey")
        d = p.get("details") or {}
        if t not in SHOTS:
            if t not in ("stoppage", "period-start", "period-end"):
                prev = p
            continue
        x, y = d.get("xCoord"), d.get("yCoord")
        own = d.get("eventOwnerTeamId")
        if x is None or y is None or own is None:
            prev = p
            continue
        is_home = own == hid
        # which net: the shooting team attacks the end its opponent defends
        hds = p.get("homeTeamDefendingSide")
        if hds in ("left", "right"):
            attack_right = (hds == "left") if is_home else (hds == "right")
            gx = 89.0 if attack_right else -89.0
        else:
            gx = 89.0 if x > 0 else -89.0
        dist = float(np.hypot(gx - x, y))
        ang = float(np.degrees(np.arctan2(abs(y), max(abs(gx - x), 0.1))))
        code = str(p.get("situationCode") or "1551").zfill(4)     # away goalie, away skaters, home skaters, home goalie
        ag, ask, hsk, hg = (int(c) for c in code)
        shooter_sk, opp_sk = (hsk, ask) if is_home else (ask, hsk)
        empty_net = (ag == 0) if is_home else (hg == 0)
        sec = None
        try:
            mm, ss = p.get("timeInPeriod", "0:0").split(":")
            sec = (int(p.get("periodDescriptor", {}).get("number", 1)) - 1) * 1200 + int(mm) * 60 + int(ss)
        except (ValueError, AttributeError):
            pass
        reb = prev is not None and prev.get("typeDescKey") in SHOTS and sec is not None and prev.get("_sec") is not None \
            and 0 <= sec - prev["_sec"] <= 3
        goalie = d.get("goalieInNetId")
        dteam = "away" if is_home else "home"
        if goalie and dteam not in start_goalie:
            start_goalie[dteam] = goalie
        rows.append({"game_id": js["id"], "team": home["abbrev"] if is_home else away["abbrev"],
                     "opp": away["abbrev"] if is_home else home["abbrev"], "is_home": int(is_home),
                     "period": int(p.get("periodDescriptor", {}).get("number", 1)),
                     "period_type": p.get("periodDescriptor", {}).get("periodType", "REG"), "sec": sec,
                     "type": t, "goal": int(t == "goal"), "on_goal": int(t in ("shot-on-goal", "goal")),
                     "x": x, "y": y, "dist": dist, "angle": ang, "shot_type": d.get("shotType") or "unknown",
                     "rebound": int(bool(reb)), "skaters": shooter_sk, "opp_skaters": opp_sk, "empty_net": int(bool(empty_net)),
                     "shooter": d.get("shootingPlayerId") or d.get("scoringPlayerId"), "goalie": goalie})
        p["_sec"] = sec
        prev = p
    nm = lambda t: f'{(t.get("placeName") or {}).get("default", "")} {(t.get("commonName") or {}).get("default", "")}'.strip()
    last = (js.get("periodDescriptor") or {}).get("periodType") or ""
    game = {"game_id": js["id"], "date": js.get("gameDate"), "start_utc": js.get("startTimeUTC"),
            "home": home["abbrev"], "away": away["abbrev"], "home_name": nm(home), "away_name": nm(away),
            "home_score": home.get("score"), "away_score": away.get("score"), "ended": last,
            "home_goalie": start_goalie.get("home"), "away_goalie": start_goalie.get("away")}
    # SO winner: the shootout goal is not a play-level goal row in the score; the API's team score includes it
    return [r for r in rows if r["period_type"] != "SO"], game


def season(year: int, s: requests.Session) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gp, sp = OUT / f"games_{year}.csv", OUT / f"shots_{year}.parquet"
    games = pd.read_csv(gp) if gp.exists() else pd.DataFrame()
    shots = pd.read_parquet(sp) if sp.exists() else pd.DataFrame()
    have = set(games["game_id"].astype(int)) if len(games) else set()
    new_g, new_s, misses, n = [], [], 0, 0
    for k in range(1, 1400):
        gid = int(f"{year}02{k:04d}")
        if gid in have:
            continue
        try:
            r = s.get(URL.format(gid=gid), timeout=25)
        except requests.RequestException:
            time.sleep(2)
            continue
        if r.status_code == 404:
            misses += 1
            if misses > 15:
                break
            continue
        if not r.ok:
            continue
        js = r.json()
        if js.get("gameState") not in ("OFF", "FINAL"):
            misses += 1
            if misses > 40:                      # well past the last finished game
                break
            continue
        misses = 0
        rows, g = parse(js)
        new_s += rows
        new_g.append(g)
        n += 1
        time.sleep(0.12)
    if new_g:
        games = pd.concat([games, pd.DataFrame(new_g)], ignore_index=True).drop_duplicates("game_id", keep="last")
        shots = pd.concat([shots, pd.DataFrame(new_s)], ignore_index=True)
        games.sort_values("game_id").to_csv(gp, index=False)
        shots.to_parquet(sp, index=False)
    print(f"{year}: {n} new games, {len(games)} total, {len(shots)} shot rows")


def goalie_names(s: requests.Session) -> None:
    """data/nhl/goalies.json: goalie id -> name, for every goalie who has started a game we saved."""
    p = OUT / "goalies.json"
    known = json.loads(p.read_text()) if p.exists() else {}
    ids = set()
    for f in OUT.glob("games_*.csv"):
        g = pd.read_csv(f)
        for c in ("home_goalie", "away_goalie"):
            ids |= {str(int(x)) for x in g[c].dropna()}
    for gid in sorted(ids - set(known)):
        try:
            j = s.get(f"https://api-web.nhle.com/v1/player/{gid}/landing", timeout=20).json()
            known[gid] = f'{(j.get("firstName") or {}).get("default", "")} {(j.get("lastName") or {}).get("default", "")}'.strip()
        except Exception:
            continue
        time.sleep(0.1)
    p.write_text(json.dumps(known, indent=0, sort_keys=True))
    print(f"goalies: {len(known)} named")


if __name__ == "__main__":
    s = requests.Session()
    s.headers.update(H)
    years = [int(a) for a in sys.argv[1:]] or [2025, 2026]
    for y in years:
        try:
            season(y, s)
        except Exception as ex:                  # never fail the collector
            print(f"{y}: {type(ex).__name__}: {ex}")
    try:
        goalie_names(s)
    except Exception as ex:
        print(f"goalie names: {type(ex).__name__}: {ex}")
