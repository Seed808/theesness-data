"""
Starting quarterback, cross-checked. Three sources must line up before a pick is published:

  1. play-by-play: the passer with the most dropbacks in the team's last game
  2. ESPN depth chart: QB1 (collected every 30 minutes)
  3. injury reports (nflverse weekly report + ESPN injury list): Out / Doubtful rules a player out

Decision
  last-game QB == ESPN QB1, not ruled out                 -> confirmed
  one of them ruled out, the other healthy                -> the healthy one, confirmed by the injury report
  they disagree and neither is ruled out                  -> HOLD: no pick until the sources agree
The nflverse schedule's listed starter is shown for reference only; it is not a deciding source.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

OUT = {"Out", "Doubtful", "Injured Reserve", "IR", "Suspension", "Reserve/PUP"}


def _espn_injury_status(repo: Path) -> dict:
    """espn athlete id -> status, from the newest ESPN injury snapshot."""
    folder = repo / "data" / "espn" / "nfl" / "injuries"
    snaps = sorted(folder.glob("2*.json"))
    if not snaps:
        return {}
    d = json.loads(snaps[-1].read_text())
    out = {}
    for team in d.get("injuries", []):
        for inj in team.get("injuries", []):
            a = inj.get("athlete") or {}
            aid = a.get("id")
            if not aid:      # ESPN's injuries feed has no athlete id field; the id is in the player-card link
                for ln in a.get("links") or []:
                    m = re.search(r"/id/(\d+)", ln.get("href", ""))
                    if m:
                        aid = m.group(1)
                        break
            if aid:
                out[str(aid)] = inj.get("status") or inj.get("type", {}).get("description")
                NOTES[str(aid)] = (inj.get("shortComment") or "", inj.get("date") or "")
    return out


NOTES: dict = {}   # espn athlete id -> (ESPN injury note, date); filled by _espn_injury_status
START_WORDS = ("expected to start", "will start", "is starting", "set to start", "named the starter", "will make the start")


class Starters:
    def __init__(self, repo: Path, pbp: pd.DataFrame, injuries: pd.DataFrame, rosters: pd.DataFrame, week: int):
        self.week = week
        d = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
        self.drop = d
        r = rosters.sort_values("week").drop_duplicates("gsis_id", keep="last")
        self.espn_to_gsis = {str(e): g for e, g in zip(r["espn_id"], r["gsis_id"]) if pd.notna(e)}
        self.name = dict(zip(r["gsis_id"], r["full_name"]))
        inj = injuries[injuries["week"] == week]
        self.nfl_status = dict(zip(inj["gsis_id"], inj["report_status"]))
        self.nfl_practice = dict(zip(inj["gsis_id"], inj["practice_status"]))
        espn = _espn_injury_status(repo)
        self.espn_status = {self.espn_to_gsis.get(k): v for k, v in espn.items() if self.espn_to_gsis.get(k)}
        self.espn_note = {self.espn_to_gsis.get(k): v for k, v in NOTES.items() if self.espn_to_gsis.get(k)}
        f = repo / "data" / "espn" / "nfl" / "depth" / "latest.json"
        self.depth = json.loads(f.read_text()) if f.exists() else {"qb": {}, "fetched_at": None}

    def _ruled_out(self, gid) -> str | None:
        for src, st in (("NFL injury report", self.nfl_status.get(gid)), ("ESPN injuries", self.espn_status.get(gid))):
            if isinstance(st, str) and st in OUT:
                return f"{st} ({src})"
        return None

    def resolve(self, team: str, espn_abbr: str | None = None, listed: str | None = None) -> dict:
        x = self.drop[self.drop["posteam"] == team]
        last_id = last_wk = None
        if not x.empty:
            last_wk = int(x["week"].max())
            last_id = x[x["week"] == last_wk].groupby("passer_player_id").size().idxmax()
        qbs = self.depth.get("qb", {}).get(espn_abbr or team, [])
        espn1 = self.espn_to_gsis.get(str(qbs[0]["id"])) if qbs else None
        espn1_name = qbs[0]["name"] if qbs else None
        info = {"team": team, "last_game": self.name.get(last_id, last_id), "last_game_week": last_wk,
                "espn_qb1": espn1_name, "listed": listed, "depth_fetched": self.depth.get("fetched_at")}
        out_last, out_espn = (self._ruled_out(last_id) if last_id else None), (self._ruled_out(espn1) if espn1 else None)
        def st(g):
            for v in (self.nfl_status.get(g), self.nfl_practice.get(g)):
                if isinstance(v, str) and v:
                    return v
            return "not on report"
        info["injury"] = {self.name.get(g, g): st(g) for g in {last_id, espn1} if g}
        if last_id and espn1 and last_id == espn1 and not out_last:
            return dict(info, status="confirmed", starter_id=last_id, starter=self.name.get(last_id),
                        why="last game's quarterback and ESPN QB1 agree")
        if last_id and out_last and espn1 and espn1 != last_id and not out_espn:
            return dict(info, status="confirmed", starter_id=espn1, starter=self.name.get(espn1),
                        why=f"{self.name.get(last_id)} is {out_last}; ESPN QB1 starts")
        if espn1 and out_espn and last_id and last_id != espn1 and not out_last:
            return dict(info, status="confirmed", starter_id=last_id, starter=self.name.get(last_id),
                        why=f"ESPN QB1 {espn1_name} is {out_espn}; last game's quarterback starts")
        if last_id and not espn1 and not out_last:
            return dict(info, status="confirmed", starter_id=last_id, starter=self.name.get(last_id),
                        why="last game's quarterback (no ESPN depth chart)")
        # ESPN QB1 returning from injury: his own ESPN injury note says he starts (e.g. "Williams (hamstring) is
        # expected to start Sunday's game ... Adam Schefter of ESPN reports") and no report has him Out/Doubtful.
        note, when = self.espn_note.get(espn1, ("", "")) if espn1 else ("", "")
        if espn1 and not out_espn and any(w in note.lower() for w in START_WORDS):
            return dict(info, status="confirmed", starter_id=espn1, starter=self.name.get(espn1),
                        why=f'ESPN QB1, and ESPN injury note ({when[:10]}): "{note}"')
        return dict(info, status="hold", starter_id=None, starter=None,
                    why=f"sources disagree: last game {self.name.get(last_id, last_id)}, ESPN QB1 {espn1_name}; "
                        "no pick until they agree or the injury report settles it")
