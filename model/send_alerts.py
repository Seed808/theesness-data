"""
Send Telegram alerts for highlighted cards (run after run_all.py in the site workflow).

  python send_alerts.py --alerts <site>/alerts.json --ledger ledger/telegram_alerts.csv
  python send_alerts.py --alerts <site>/alerts.json --test        # one TEST sample per live sport, not recorded

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment. Never prints them.
Always exits 0: an alert problem must never fail the site build.

Rules (the highlight level itself comes from site.highlight_level via alerts.json):
  new        first time a locked pick on a live board is highlighted (level >= 1) before kickoff
  upgrade    an alerted pick goes from highlighted to strong
  flip       an alerted pick switches sides
  withdrawn  an alerted pick becomes VOID, HOLD or NO EDGE (no pick any more)
  back       a withdrawn pick is highlighted again (at most once)
Each game gets at most one alert of each kind; every send is appended to the ledger CSV.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

FIELDS = ["sent_at", "sport", "game_id", "kind", "level", "pick", "pick_line", "gap", "telegram_ok"]
LABEL = {"nfl": "NFL", "ncaaf": "College football", "nba": "NBA", "cbb": "College basketball", "mlb": "MLB"}


def log(msg):
    print(f"[alerts] {msg}", flush=True)


DRY = False


def tg_send(text: str) -> bool:
    if DRY:
        print("----\n" + text)
        return True
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(), os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        log("Telegram secrets not set; skipping send")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(4):
        try:
            r = requests.post(url, json=body, timeout=20)
        except requests.RequestException as ex:
            log(f"network error ({type(ex).__name__}); retrying")
            time.sleep(3 * (attempt + 1))
            continue
        if r.ok:
            return True
        try:
            j = r.json()
        except ValueError:
            j = {}
        if r.status_code == 429:
            wait = int((j.get("parameters") or {}).get("retry_after", 5)) + 1
            log(f"rate limited; waiting {wait}s")
            time.sleep(min(wait, 60))
            continue
        if r.status_code == 400 and "parse_mode" in body:
            log(f"formatting rejected ({j.get('description', '')[:80]}); resending as plain text")
            body = {"chat_id": chat, "text": html.unescape(re.sub(r"<[^>]+>", "", text)), "disable_web_page_preview": True}
            continue
        log(f"Telegram error {r.status_code}: {str(j.get('description', ''))[:120]}")
        return False
    return False


def message(c: dict, kind: str, prev: dict | None = None, test: bool = False) -> str:
    sport = LABEL.get(c["sport"], c["sport"].upper())
    level = "STRONG" if c.get("level", 0) >= 2 else "HIGHLIGHTED"
    head = {"new": level, "back": f"BACK ON · {level}", "upgrade": "UPGRADED TO STRONG", "flip": "PICK FLIPPED",
            "withdrawn": "NO LONGER A PLAY"}[kind]
    e = html.escape
    lines = [f"<b>{'TEST · ' if test else ''}SPORTS · {e(sport)} · {head}</b>",
             f"<b>{e(c['away'])} @ {e(c['home'])}</b>", e(c["kickoff_hst"])]
    if kind == "withdrawn":
        what = {"void": "VOID", "hold": "HOLD (starter not confirmed)", "no_edge": "NO EDGE"}.get(c["status"], c["status"])
        lines.append(f"Earlier play {e(prev.get('pick', ''))} {e(prev.get('pick_line', ''))} is now <b>{e(what)}</b>.")
        if c.get("reason"):
            lines.append(e(str(c["reason"]))[:300])
    else:
        if c["sport"] == "mlb":
            lines.append(f"Book win chance: {c.get('book_p', '—')}% · Model: {c.get('model_p', '—')}% · Edge: {c.get('gap')} pts")
        else:
            lines.append(f"Book: {html.unescape(c.get('book', '—'))} · Ours: {html.unescape(c.get('ours', '—'))} · Gap: {c.get('gap')} pts")
        lines.append(f"<b>The Play: {e(str(c.get('pick')))} {e(str(c.get('pick_line')))}</b>")
        if kind == "upgrade":
            lines.append(f"Gap grew to {c.get('gap')} (was highlighted).")
        if kind in ("flip", "back") and prev:
            lines.append(f"Was {e(prev.get('pick', ''))} {e(prev.get('pick_line', ''))}.")
    lines.append(f'<a href="{e(c["url"])}">Open the game</a>')
    return "\n".join(lines)


def load_ledger(p: Path) -> dict:
    """game key -> list of sent rows (oldest first)."""
    out: dict = {}
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                out.setdefault((r["sport"], r["game_id"]), []).append(r)
    return out


def decide(c: dict, sent: list[dict]) -> tuple[str | None, dict | None]:
    """Which alert (if any) this card needs now, given what was already sent for the game."""
    ok_pick = c["status"] == "ok" and c.get("locked") and c.get("pick")
    if not sent:
        return ("new", None) if ok_pick and c.get("level", 0) >= 1 else (None, None)
    kinds = [r["kind"] for r in sent]
    picks = [r for r in sent if r["kind"] in ("new", "back", "upgrade", "flip")]
    last = picks[-1] if picks else None
    if kinds[-1] == "withdrawn":
        # the pick came back (e.g. the starter was confirmed): one "back on" alert, ever
        if ok_pick and c.get("level", 0) >= 1 and "back" not in kinds:
            return "back", last
        return None, None
    if c["status"] in ("void", "hold", "no_edge"):
        return ("withdrawn", last) if "withdrawn" not in kinds else (None, None)
    if not ok_pick or last is None:
        return None, None
    if c["pick"] != last["pick"] and "flip" not in kinds:
        return "flip", last
    if c.get("level", 0) >= 2 and int(float(last["level"] or 0)) < 2 and "upgrade" not in kinds and c["pick"] == last["pick"]:
        return "upgrade", last
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--ledger", default="ledger/telegram_alerts.csv")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print messages instead of sending; ledger not written")
    a = ap.parse_args()
    global DRY
    DRY = a.dry_run
    try:
        data = json.loads(Path(a.alerts).read_text())
    except Exception as ex:                                     # no file: nothing to do
        log(f"no alerts file ({type(ex).__name__})")
        return
    cards, live = data.get("cards", []), data.get("live_sports", [])
    if a.test:
        for sport in live:
            pool = [c for c in cards if c["sport"] == sport and c.get("pick") and c.get("level", 0) >= 1]
            pool.sort(key=lambda c: -(c.get("gap") or 0))
            c = pool[0] if pool else {"sport": sport, "away": "Sample Away", "home": "Sample Home", "level": 1,
                                      "kickoff_hst": "(sample)", "url": f"https://seed808.github.io/theesness-data/{sport}/",
                                      "pick": "Sample Home", "pick_line": "+3.5" if sport != "mlb" else "+120",
                                      "book": "Sample Away -3.5", "ours": "Pick'em", "gap": 3.5, "book_p": 45.5, "model_p": 49.0}
            ok = tg_send(message(dict(c, level=max(1, c.get("level", 1))), "new", test=True))
            log(f"test {sport}: {'sent' if ok else 'not sent'}")
        if not live:
            log("no live boards in this build; nothing to test")
        return
    led_p = Path(a.ledger)
    ledger = load_ledger(led_p)
    new_rows = []
    for c in cards:
        kind, prev = decide(c, ledger.get((c["sport"], c["game_id"]), []))
        if not kind:
            continue
        ok = tg_send(message(c, kind, prev))
        log(f"{kind} {c['sport']} {c['game_id']}: {'sent' if ok else 'NOT sent'}")
        if not ok and kind == "new":
            continue                                            # try again next build
        new_rows.append({"sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sport": c["sport"],
                         "game_id": c["game_id"], "kind": kind, "level": c.get("level", 0),
                         "pick": c.get("pick") or "", "pick_line": c.get("pick_line") or "", "gap": c.get("gap") or "",
                         "telegram_ok": ok})
        time.sleep(1.1)                                         # stay well under Telegram's per-chat limit
    if new_rows and not DRY:
        led_p.parent.mkdir(parents=True, exist_ok=True)
        exists = led_p.exists()
        with led_p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    log(f"{len(new_rows)} alert(s) recorded; {len(cards)} live cards checked")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:                                     # never fail the build
        log(f"alert step error: {type(ex).__name__}: {str(ex)[:200]}")
    sys.exit(0)
