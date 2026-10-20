"""
NBA injury adjustment for the live board.

Who is out: the newest ESPN injury snapshot the collector saved (data/espn/nba/injuries). ESPN marks players
"Out" or "Day-To-Day"; for Day-To-Day the note decides ("will miss", "ruled out" -> out; "doubtful" -> 85% out;
"questionable" -> 50%; anything else, or a note older than 3 days -> not counted).

What it costs: each player's impact (ridge adjusted plus-minus, points per 100 possessions, 2025-26 with
2024-25 as the prior) times his usual share of his team's floor time, versus a replacement player, and only
for the part of last season he actually played (the team rating already reflects the games he missed).
Only players on the same team last season count: newcomers are not in the team rating either way.
Source table: data_static/nba/player_impact_2026.csv (see player_impact_meta.json).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

OUT_WORDS = ("will miss", "won't play", "will not play", "ruled out", "out for", "sidelined", "is out", "won't suit up",
             "not expected to play", "will be held out")


def load_table(data_static: Path):
    t = pd.read_csv(data_static / "nba" / "player_impact_2026.csv")
    meta = json.loads((data_static / "nba" / "player_impact_meta.json").read_text())
    return t, float(meta["replacement"])


def espn_injuries(repo: Path) -> list[dict]:
    snaps = sorted((repo / "data" / "espn" / "nba" / "injuries").glob("2*.json"))
    if not snaps:
        return []
    d = json.loads(snaps[-1].read_text())
    out = []
    for team in d.get("injuries", []):
        for inj in team.get("injuries", []):
            a = inj.get("athlete") or {}
            aid = None
            for ln in a.get("links") or []:
                m = re.search(r"/id/(\d+)", ln.get("href", ""))
                if m:
                    aid = int(m.group(1))
                    break
            out.append({"team_id": int(team["id"]), "player": aid, "name": a.get("displayName", "?"),
                        "status": inj.get("status") or "", "note": inj.get("shortComment") or "",
                        "date": inj.get("date") or "", "snapshot": snaps[-1].stem})
    return out


def missing_prob(e: dict, now) -> tuple[float, str]:
    """Chance the player sits, and a short reason shown on the game page."""
    st, note = e["status"].lower(), e["note"].lower()
    if st == "out":
        return 1.0, "out"
    try:
        age = (pd.Timestamp(now) - pd.Timestamp(e["date"])).total_seconds() / 86400
    except (ValueError, TypeError):
        age = 99
    if age > 3:
        return 0.0, "day-to-day (old note, not counted)"
    if any(w in note for w in OUT_WORDS):
        return 1.0, "day-to-day, note says out"
    if "doubtful" in note:
        return 0.85, "doubtful"
    if "questionable" in note:
        return 0.5, "questionable"
    return 0.0, "day-to-day, not counted"


def team_adjustment(team_id: int, injuries: list[dict], table: pd.DataFrame, repl: float, now) -> tuple[float, list[dict]]:
    """Change to the team's rating in points per 100 possessions, and the players behind it."""
    reg = table[table["team_id"] == team_id].set_index("player")
    total, rows = 0.0, []
    for e in injuries:
        if e["team_id"] != team_id or e["player"] is None:
            continue
        p, why = missing_prob(e, now)
        if e["player"] not in reg.index:
            rows.append({"name": e["name"], "status": why, "pts100": 0.0, "note": "not in last season's rotation for this team"})
            continue
        r = reg.loc[e["player"]]
        d = -p * float(r["share"]) * (float(r["impact"]) - repl) * float(r["presence"])
        total += d
        rows.append({"name": e["name"], "status": why, "pts100": round(d, 2),
                     "note": f'impact {r["impact"]:+.1f}, {100 * r["share"]:.0f}% of floor time'})
    return total, rows
