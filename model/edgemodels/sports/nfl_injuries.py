"""
NFL availability from the official injury report and weekly rosters (nflverse).

    injuries_{season}.parquet       team, week, gsis_id, report_status (Out / Doubtful / Questionable)
    roster_weekly_{season}.parquet  team, week, gsis_id, status (ACT, INA = game-day inactive, RES = IR/reserve, CUT ...)

Two modes, matching when a projection is locked:
  "report"     Friday injury report: Out and Doubtful -> 0, Questionable -> P(plays), IR / cut / traded -> 0.
  "inactives"  after the game-day inactives list (about 90 minutes before kickoff): roster status decides, ACT -> 1.

P(plays | Questionable) is measured, not assumed: in 2025, 67.0% of Questionable players were active
(Doubtful 0.8%, Out 0.1%). `questionable_rate()` recomputes it from whatever seasons are loaded.
"""
from __future__ import annotations

import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/{kind}/{name}_{season}.parquet"
GONE = {"RES", "CUT", "RET", "TRD", "TRC", "EXE", "DEV"}


def load(season: int, data_dir=None):
    if data_dir:
        inj = pd.read_parquet(f"{data_dir}/injuries_{season}.parquet")
        ros = pd.read_parquet(f"{data_dir}/roster_weekly_{season}.parquet")
    else:
        inj = pd.read_parquet(URL.format(kind="injuries", name="injuries", season=season))
        ros = pd.read_parquet(URL.format(kind="weekly_rosters", name="roster_weekly", season=season))
    return inj, ros


def questionable_rate(inj: pd.DataFrame, ros: pd.DataFrame) -> float:
    m = inj[inj["report_status"] == "Questionable"].merge(ros[["gsis_id", "week", "status"]], on=["gsis_id", "week"])
    return float((m["status"] == "ACT").mean()) if len(m) else 0.67


class Availability:
    def __init__(self, inj: pd.DataFrame, ros: pd.DataFrame, q_rate: float | None = None):
        self.inj, self.ros = inj, ros
        self.q = q_rate if q_rate is not None else questionable_rate(inj, ros)

    def for_week(self, week: int, mode: str = "report") -> dict:
        """{gsis_id: probability the player is active} for players with a known status.
        Players not listed are treated as available. Returns {} if nothing is published for that week."""
        out = {}
        r = self.ros[self.ros["week"] == week]
        if len(r):
            if mode == "inactives":
                for pid, st in zip(r["gsis_id"], r["status"]):
                    out[pid] = 1.0 if st == "ACT" else 0.0
                return out
            for pid, st in zip(r["gsis_id"], r["status"]):
                if st in GONE:
                    out[pid] = 0.0
        i = self.inj[self.inj["week"] == week]
        for pid, st in zip(i["gsis_id"], i["report_status"]):
            if st in ("Out", "Doubtful"):
                out[pid] = 0.0
            elif st == "Questionable":
                out[pid] = min(out.get(pid, 1.0), self.q)
        return out

    def for_team(self, week: int, team: str, mode: str = "report") -> dict:
        """for_week, plus 0 for anyone on another team's roster that week (trades, releases)."""
        av = dict(self.for_week(week, mode))
        r = self.ros[self.ros["week"] == week]
        for pid, t in zip(r["gsis_id"], r["team"]):
            if t != team:
                av[pid] = 0.0
        return av

    def for_game(self, week: int, teams, mode: str = "report") -> dict:
        """{team: availability} for game_props. If this week's report is not out yet, carry IR, cuts and
        trades from the latest roster and mark it unpublished (props then require recent playing time)."""
        if self.published(week):
            return {t: self.for_team(week, t, mode) for t in teams}
        latest = int(self.ros["week"].max()) if len(self.ros) else None
        out = {}
        for t in teams:
            av = {"__published__": False}
            if latest is not None:
                r = self.ros[self.ros["week"] == latest]
                for pid, st, tm in zip(r["gsis_id"], r["status"], r["team"]):
                    if st in GONE or tm != t:
                        av[pid] = 0.0
            out[t] = av
        return out

    def published(self, week: int) -> bool:
        return bool((self.inj["week"] == week).any() or (self.ros["week"] == week).any())

    def out_list(self, week: int, team: str, mode: str = "report") -> list[str]:
        """Names the card shows as OUT (probability 0) for a team."""
        av = self.for_week(week, mode)
        i = self.inj[(self.inj["week"] == week) & (self.inj["team"] == team)]
        names = dict(zip(i["gsis_id"], i["full_name"]))
        r = self.ros[(self.ros["week"] == week) & (self.ros["team"] == team)]
        names.update({k: v for k, v in zip(r["gsis_id"], r["full_name"]) if k not in names})
        skill = set(self.ros.loc[self.ros["position"].isin(["QB", "RB", "WR", "TE", "FB"]), "gsis_id"])
        return sorted(names[p] for p, v in av.items() if v == 0.0 and p in names and p in skill and
                      (p in set(i["gsis_id"]) or mode == "inactives"))
