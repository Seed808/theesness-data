"""
Picks update until kickoff.

Every rebuild re-projects each game that has not started, with the newest roster and the newest line. If the pick,
the line or the projection changed, the previous lock row is superseded (a dated, hash-chained entry in the voids
file) and the new version is locked. When the game starts, the last version stands and is graded.
A game whose starter cannot be confirmed is put on HOLD: its current version is withdrawn and no pick is shown
until the sources agree. Every version is kept in <game_id>.history.json and shown on the game page.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ledger import SUPERSEDED, Ledger


def _key(row: dict) -> tuple:
    f = lambda v: None if v is None or (isinstance(v, float) and v != v) else round(float(v), 1)
    return (row.get("pick"), f(row.get("pick_line")), f(row.get("market_line")), f(row.get("model_line")),
            )


def current_rows(led: Ledger) -> dict:
    """game_id -> its live (not superseded) lock row."""
    rows, v = led.load(), led.voids()
    dead = set(v.loc[v["kind"].isin(SUPERSEDED), "lock_hash"]) if len(v) else set()
    rows = rows[~rows["hash"].isin(dead)]
    return {str(r["game_id"]): r for r in rows.to_dict("records")}


def sync(led: Ledger, cards: list[dict], kickoff_utc: dict, arch: Path, now: datetime, clean, sport: str,
         rows_for=None, held: dict | None = None, no_edge: dict | None = None) -> dict:
    """cards: freshly projected cards (latest roster + line). kickoff_utc: game_id -> UTC kickoff.
    rows_for(card) -> lock rows (default: the card itself). held: game_id -> reason (starter not confirmed).
    Returns {"locked": n, "updated": n, "held": n}; updates cards in place (current version, history attached)."""
    held = held or {}
    no_edge = no_edge or {}     # game_id -> reason: model within NO_EDGE points of the line, no side taken
    arch.mkdir(parents=True, exist_ok=True)
    rows_for = rows_for or (lambda c: [dict(c, sport=sport)])
    stats = {"locked": 0, "updated": 0, "held": 0}
    cur = current_rows(led)
    for i, c in enumerate(cards):
        gid = str(c["game_id"])
        ko = kickoff_utc.get(gid)
        started = ko is not None and pd.Timestamp(ko) <= pd.Timestamp(now)
        hist_p = arch / f"{gid}.history.json"
        hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
        cur_p = arch / f"{gid}.json"
        if started:
            if cur_p.exists() and gid in cur:            # the last version before kickoff stands
                saved = json.loads(cur_p.read_text())
                saved.update({k: c[k] for k in ("records", "date", "kickoff", "props") if k in c})
                saved["locked"], saved["history"] = True, hist
                cards[i] = saved
            else:
                c["history"] = hist
            continue
        stamp = now.isoformat(timespec="seconds")
        if gid in held:
            if gid in cur:
                led.void(gid, f"Hold: {held[gid]}", ko, kind="hold")
                hist.append({"at": stamp, "event": "hold", "why": held[gid]})
                hist_p.write_text(json.dumps(hist))
                if cur_p.exists():
                    cur_p.unlink()
                stats["held"] += 1
            c["hold"], c["history"] = held[gid], hist
            continue
        if gid in no_edge:
            if gid in cur:
                led.void(gid, f"No edge: {no_edge[gid]}", ko, kind="hold")
                hist.append({"at": stamp, "event": "no edge", "why": no_edge[gid],
                             "model_line": c.get("model_line"), "market_line": c.get("market_line")})
                hist_p.write_text(json.dumps(hist))
                if cur_p.exists():
                    cur_p.unlink()
            c["no_edge"], c["history"] = no_edge[gid], hist
            continue
        new_rows = [r for r in rows_for(c) if r.get("market_line") is not None]
        if not new_rows:
            c["history"] = hist
            continue
        old = cur.get(gid)
        if old is not None and _key(old) == _key(new_rows[0]):
            c["locked"], c["history"] = True, hist
            continue
        if old is not None:
            led.void(gid, "Update before kickoff: newer roster or line", ko, kind="update")
            stats["updated"] += 1
        else:
            stats["locked"] += 1
        for r in new_rows:
            r["locked_at"] = stamp
        led.lock(new_rows)
        hist.append({"at": stamp, "event": "update" if old is not None else "locked",
                     "pick": c.get(c["pick"]) if c.get("pick") in ("home", "away") else c.get("pick"),
                     "pick_line": new_rows[0].get("pick_line"), "market_line": new_rows[0].get("market_line"),
                     "model_line": c.get("model_line"), "proj": [c.get("proj_away"), c.get("proj_home")],
                     "edge": new_rows[0].get("edge"), "starters": c.get("starters", "")})
        hist_p.write_text(json.dumps(hist))
        cur_p.write_text(json.dumps(clean(c)))
        c["locked"], c["history"] = True, hist
    return stats
