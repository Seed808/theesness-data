"""
Telegram alerts for highlighted cards.

During the build, each LIVE board calls collect(); run_all writes everything collected to <site>/alerts.json.
After the build, send_alerts.py compares that file with model/ledger/telegram_alerts.csv and sends only what is new.
The highlight rule itself lives in site.highlight_level (the same function the board uses).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import site
from .core.edges import fmt_line

BASE_URL = "https://seed808.github.io/theesness-data"
SPORT_LABEL = {"nfl": "NFL", "ncaaf": "College football", "nba": "NBA", "cbb": "College basketball", "mlb": "MLB"}
COLLECTED: list[dict] = []
LIVE: set = set()


def _hst(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert("Pacific/Honolulu")
    return t.strftime("%a %b %-d, %-I:%M %p HST")


def _ml(x):
    return "—" if x is None else (f"+{x:.0f}" if x > 0 else f"{x:.0f}")


def collect(sport: str, cards: list[dict], names: dict, kickoff_utc: dict, now: datetime) -> None:
    """Record every card on a live board that has not started. Only locked picks can alert."""
    LIVE.add(sport)
    for c in cards:
        gid = str(c["game_id"])
        ko = kickoff_utc.get(gid) or kickoff_utc.get(c["game_id"])
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue                                   # started (or no kickoff known): never alert
        lvl = site.highlight_level(c, sport)
        status = "void" if c.get("void") else "hold" if c.get("hold") else "no_edge" if c.get("no_edge") else "ok"
        a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
        row = {"sport": sport, "game_id": gid, "level": lvl, "status": status, "locked": bool(c.get("locked")),
               "away": a, "home": h, "kickoff_utc": str(pd.Timestamp(ko)), "kickoff_hst": _hst(ko),
               "url": f"{BASE_URL}/{sport}/game/{gid}.html"}
        v = c.get("void")
        row["reason"] = c.get("hold") or (v.get("reason") if isinstance(v, dict) else None) or c.get("no_edge")
        if sport == "mlb":
            side = c.get("ml_pick")
            if side in ("home", "away"):
                mph = c.get("market_p_home")
                row.update(pick=names.get(c[side], c[side]), pick_line=_ml(c.get("ml_odds")),
                           book_p=None if mph is None else round(100 * (mph if side == "home" else 1 - mph), 1),
                           model_p=round(100 * c["ml_prob"], 1) if c.get("ml_prob") is not None else None,
                           gap=round(c.get("ml_edge") or 0, 1))
        else:
            side = c.get("pick")
            if side in ("home", "away") and c.get("market_line") is not None:
                row.update(pick=names.get(c[side], c[side]), pick_line=fmt_line(c["pick_line"]),
                           book=site._fav_text(c.get("market_line"), c, names),
                           ours=site._fav_text(c.get("model_line"), c, names), gap=c.get("edge"))
        COLLECTED.append(row)


def payload(now: datetime) -> dict:
    return {"built_at": now.isoformat(timespec="seconds"), "live_sports": sorted(LIVE), "cards": COLLECTED}
