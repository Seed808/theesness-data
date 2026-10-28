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
HEALTH: list[dict] = []      # model sanity warnings, sent once per day each by send_alerts.py


def health(sport: str, key: str, msg: str) -> None:
    """Flag something that looks broken in a board (sent to Telegram as a MODEL CHECK, once per day per key)."""
    if not any(h["sport"] == sport and h["key"] == key for h in HEALTH):
        HEALTH.append({"sport": sport, "key": key, "msg": msg})
        print(f"[health] {sport}: {msg}")


def check_board(sport: str, cards: list[dict]) -> None:
    """Cheap checks that catch a broken model: many games with the identical projected total or margin."""
    proj = [c for c in cards if c.get("model_total") is not None]
    if len(proj) >= 4:
        from collections import Counter
        tot, n = Counter(round(float(c["model_total"]), 1) for c in proj).most_common(1)[0]
        if n >= max(3, len(proj) // 2):
            health(sport, "flat_totals", f"{n} of {len(proj)} games project the exact same total ({tot:g}). "
                                         "The ratings look flat; check before trusting any pick.")
        lines = [round(float(c["model_line"]), 1) for c in proj if c.get("model_line") is not None]
        if lines and max(abs(x) for x in lines) < 1.0:
            health(sport, "flat_lines", f"All {len(lines)} projected lines are within 1 point of a pick'em. "
                                        "The ratings look flat; check before trusting any pick.")


def check_dates(sport: str, cards: list[dict], kickoff_utc: dict, now) -> None:
    """Date sanity: each card's date label must be its kickoff's Hawaii date, and no card may carry a final score
    or a result before its own kickoff. Any miss sends one MODEL CHECK warning a day."""
    bad_label, early = [], []
    for c in cards:
        ko = kickoff_utc.get(str(c["game_id"])) or kickoff_utc.get(c["game_id"])
        if ko is None:
            continue
        t = pd.Timestamp(ko)
        t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("Pacific/Honolulu")
        want = f"{t:%A}, {t:%B} {t.day}"
        lab = c.get("date")
        if lab and lab != "Date not in feed" and lab != want:
            bad_label.append(f'{c["away"]} @ {c["home"]} (card says {lab}, kickoff is {want})')
        if t > pd.Timestamp(now).tz_convert("Pacific/Honolulu") and (c.get("final_home") is not None or c.get("result")):
            early.append(f'{c["away"]} @ {c["home"]} ({want})')
    if bad_label:
        health(sport, "date_label", f"{len(bad_label)} card(s) show the wrong date: " + "; ".join(bad_label[:4]))
    if early:
        health(sport, "final_early", f"{len(early)} card(s) show a final or result before kickoff: " + "; ".join(early[:4]))


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
    try:
        check_board(sport, [c for c in cards if not c.get("void")])
        check_dates(sport, cards, kickoff_utc, now)
    except Exception as ex:                        # a check never breaks the board
        print(f"[health] {sport}: check failed ({type(ex).__name__})")
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
        ti = site.total_info(c, sport)
        if ti:
            row.update(tot_book=ti["book"], tot_ours=ti["ours"], tot_gap=ti["gap"], tot_lean=ti["lean"], tot_level=ti["level"])
        COLLECTED.append(row)


def payload(now: datetime) -> dict:
    return {"built_at": now.isoformat(timespec="seconds"), "live_sports": sorted(LIVE), "cards": COLLECTED, "health": HEALTH}
