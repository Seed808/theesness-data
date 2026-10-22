"""
Tell Telegram when the site build breaks, keeps failing, or recovers (run by the site workflow after the build step).

  BUILD_OUTCOME=success|failure|cancelled  RUN_URL=<link to the run>  python build_status.py

State lives in ledger/build_status.json (committed with the ledger), so there is no spam:
  first failure after a good build        -> "SITE BUILD FAILED" with the last error line
  still failing                           -> a reminder at most every 3 hours
  first good build after failures         -> "site is building again", with how long it was down
Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the environment, never prints them, always exits 0.
"""
from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "ledger" / "build_status.json"
REMIND = timedelta(hours=3)


def hst(t: datetime) -> str:
    return t.astimezone(timezone(timedelta(hours=-10))).strftime("%b %-d, %-I:%M %p HST")


def last_error() -> str:
    p = HERE / "build_log.txt"
    if not p.exists():
        return "no build log"
    lines = [ln.strip() for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Error" in ln or "Exception" in ln or "Traceback" in ln:
            return ln[:300]
    return (lines[-1] if lines else "no output")[:300]


def main():
    sys.path.insert(0, str(HERE))
    from send_alerts import tg_send          # same sender: HTML, 429 retry, plain-text fallback
    outcome = os.environ.get("BUILD_OUTCOME", "")
    run = os.environ.get("RUN_URL", "")
    now = datetime.now(timezone.utc)
    st = json.loads(STATE.read_text()) if STATE.exists() else {"failing_since": None, "last_alert": None}
    if outcome == "success":
        if st.get("failing_since"):
            since = datetime.fromisoformat(st["failing_since"])
            h = (now - since).total_seconds() / 3600
            tg_send(f"<b>SPORTS · ✅ SITE BUILDING AGAIN</b>\nBuilds failed from {hst(since)} ({h:.1f} h). "
                    "Picks, grading and alerts are running again.")
        st = {"failing_since": None, "last_alert": None}
    elif outcome == "failure":
        first = not st.get("failing_since")
        if first:
            st["failing_since"] = now.isoformat(timespec="seconds")
        last = datetime.fromisoformat(st["last_alert"]) if st.get("last_alert") else None
        if first or last is None or now - last >= REMIND:
            since = datetime.fromisoformat(st["failing_since"])
            head = "⚠️ SITE BUILD FAILED" if first else "⚠️ SITE BUILD STILL FAILING"
            ok = tg_send(f"<b>SPORTS · {head}</b>\nFailing since {hst(since)}. While this lasts, picks do not update, "
                         f"finished games are not graded and no play alerts go out.\nError: {html.escape(last_error())}\n"
                         + (f'<a href="{html.escape(run)}">Open the run</a>' if run else ""))
            if ok:
                st["last_alert"] = now.isoformat(timespec="seconds")
    else:
        return                                   # cancelled / skipped: say nothing, keep state
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[build_status] {type(ex).__name__}: {str(ex)[:200]}")
    sys.exit(0)
