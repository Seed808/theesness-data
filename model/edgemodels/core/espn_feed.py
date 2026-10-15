"""Reading the collected ESPN scoreboard snapshots (theesness-data/data/espn/<league>/scoreboard/<date>/).

The collector writes a timestamped snapshot every time a day's board changes. A game's "first line seen" is the
earliest snapshot in which that game has odds, which can be days after the folder's first snapshot (MLB playoff
lines post only once the previous game ends). Picks lock against that line.
"""
from __future__ import annotations

import json
from pathlib import Path


def _snapshots(folder: Path) -> list[Path]:
    snaps = sorted(folder.glob("2*.json"))            # 20261014T0329Z.json ... in time order
    if not snaps and (folder / "first.json").exists():
        snaps = [folder / "first.json"]
    return snaps


def _has_line(ev: dict) -> bool:
    o = (ev["competitions"][0].get("odds") or [{}])[0]
    return bool(o.get("details") or o.get("moneyline") or o.get("overUnder"))


def first_line_events(folder: Path) -> list[tuple[dict, str | None]]:
    """[(event, seen_at)] for every game in the folder: the event as it looked in the earliest snapshot that had
    a line for it (seen_at = that snapshot's stamp), or its latest version with seen_at None if no line ever posted."""
    first, latest, order = {}, {}, []
    for f in _snapshots(folder):
        for ev in json.loads(f.read_text()).get("events", []):
            i = str(ev["id"])
            if i not in latest:
                order.append(i)
            latest[i] = ev
            if i not in first and _has_line(ev):
                first[i] = (ev, f.stem)
    return [first.get(i, (latest[i], None)) for i in order]


def latest_events(folder: Path) -> list[dict]:
    snaps = _snapshots(folder)
    return json.loads(snaps[-1].read_text()).get("events", []) if snaps else []


def latest_line_events(folder: Path) -> list[tuple[dict, str | None]]:
    """[(event, seen_at)] using, for each game, the most recent snapshot that had a line (the current line
    while the game has not started)."""
    last, latest, order = {}, {}, []
    for f in _snapshots(folder):
        for ev in json.loads(f.read_text()).get("events", []):
            i = str(ev["id"])
            if i not in latest:
                order.append(i)
            latest[i] = ev
            if _has_line(ev):
                last[i] = (ev, f.stem)
    return [last.get(i, (latest[i], None)) for i in order]
