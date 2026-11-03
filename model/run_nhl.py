"""
NHL live board: real play-by-play model (edgemodels/sports/nhl_model.py) against ESPN / DraftKings moneylines.

PREVIEW until NHL_RECORD_FROM: picks are locked and graded on a separate preview tracker (never the record) and send
no Telegram alerts. Cards use ESPN event ids and team codes; the model uses the NHL feed's codes (matched by name).
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from edgemodels import alerts, site
from edgemodels.core import firstline, pickbook, totals_tracker
from edgemodels.core.edges import EDGE_TIERS
from edgemodels.core.ledger import Ledger, record_summary
from edgemodels.sports import mlb_pipeline as mp
from edgemodels.sports.nhl_model import NHLModel

NHL_RECORD_FROM = pd.Timestamp("2026-11-10").date()     # one preview week, then the record starts at 0-0
LOCK_EXTRA = ["market", "odds"]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()).strip()


def _odds(d):
    d = d or {}
    for k in ("close", "open"):
        v = (d.get(k) or {}).get("odds")
        try:
            return float(str(v).replace("EVEN", "100").replace("+", ""))
        except (TypeError, ValueError):
            continue
    return None


def espn_events(repo: Path) -> dict:
    """event id -> teams, names, kickoff, final score, newest pre-game moneyline / total (DraftKings via ESPN)."""
    out = {}
    base = repo / "data" / "espn" / "nhl" / "scoreboard"
    for folder in sorted(base.glob("2*")):
        snaps = sorted(folder.glob("2*.json"))
        for f in snaps:
            for ev in json.loads(f.read_text()).get("events", []):
                c = ev["competitions"][0]
                t = {x["homeAway"]: x for x in c["competitors"]}
                st = ((c.get("status") or ev.get("status") or {}).get("type") or {})
                row = out.setdefault(str(ev["id"]), {})
                row.update(game_id=str(ev["id"]), kickoff_utc=ev["date"], away=t["away"]["team"]["abbreviation"],
                           home=t["home"]["team"]["abbreviation"], venue=(c.get("venue") or {}).get("fullName", ""),
                           season_type=(ev.get("season") or {}).get("type"),
                           names={t[s]["team"]["abbreviation"]: t[s]["team"].get("displayName") for s in ("away", "home")})
                if st.get("completed"):
                    row["final"] = (float(t["away"]["score"]), float(t["home"]["score"]))
                if st.get("state") == "pre":
                    o = (c.get("odds") or [{}])[0]
                    ml, tot = o.get("moneyline") or {}, o.get("total") or {}
                    hm, am = _odds(ml.get("home")), _odds(ml.get("away"))
                    if hm is not None and am is not None:
                        row.update(home_ml=hm, away_ml=am, total=o.get("overUnder"), over_odds=_odds(tot.get("over")),
                                   under_odds=_odds(tot.get("under")), line_seen=f.stem)
    return out


def nhl_section(out: Path, repo: Path, ledger_dir: Path, report: dict, now: datetime) -> tuple:
    D = repo / "data" / "nhl"
    if not (D / "games_2025.csv").exists():
        return None, "Hockey data not collected yet"
    gl, gc = pd.read_csv(D / "games_2025.csv"), pd.read_csv(D / "games_2026.csv") if (D / "games_2026.csv").exists() else None
    sc = pd.read_parquet(D / "shots_2026.parquet") if (D / "shots_2026.parquet").exists() else pd.DataFrame()
    M = NHLModel(pd.read_parquet(D / "shots_2025.parquet"), gl, sc, gc if gc is not None else gl.iloc[0:0])
    gnames = json.loads((D / "goalies.json").read_text()) if (D / "goalies.json").exists() else {}
    allg = pd.concat([gl, gc]) if gc is not None else gl
    byname = {_norm(n): a for a, n in list(zip(allg["home"], allg["home_name"])) + list(zip(allg["away"], allg["away_name"]))}
    ev = espn_events(repo)
    today = (pd.Timestamp(now) - pd.Timedelta(hours=10)).date()
    live = today >= NHL_RECORD_FROM
    hst = lambda v: (pd.Timestamp(v["kickoff_utc"]) - pd.Timedelta(hours=10)).date()
    played = {}
    for v in ev.values():
        if v.get("kickoff_utc"):
            for t in (v["away"], v["home"]):
                played.setdefault(t, set()).add(hst(v))
    window = sorted([v for v in ev.values() if v.get("kickoff_utc") and v.get("season_type") == 2
                     and today - pd.Timedelta(days=1) <= hst(v) <= today + pd.Timedelta(days=1)], key=lambda v: v["kickoff_utc"])
    R = M.ratings(before=str(today))
    names, cards = {}, []
    for v in window:
        names.update({k: n for k, n in v["names"].items() if n})
        ma, mh = byname.get(_norm(v["names"].get(v["away"], ""))), byname.get(_norm(v["names"].get(v["home"], "")))
        if not ma or not mh:
            continue
        d = hst(v)
        ko = pd.Timestamp(v["kickoff_utc"])
        g = {"game_id": v["game_id"], "date": (ko - pd.Timedelta(hours=10)).strftime("%A, %B %-d"), "away": ma, "home": mh,
             "date_utc": v["kickoff_utc"], "home_b2b": (d - pd.Timedelta(days=1)) in played.get(v["home"], set()),
             "away_b2b": (d - pd.Timedelta(days=1)) in played.get(v["away"], set()),
             **{k: v.get(k) for k in ("home_ml", "away_ml", "total", "over_odds", "under_odds")}}
        c = M.project(R, g)
        c.update(game_id=v["game_id"], away=v["away"], home=v["home"], date=g["date"], venue=v.get("venue", ""),
                 kickoff=(ko - pd.Timedelta(hours=10)).strftime("%-I:%M %p HST"))
        sp = {}
        for side in ("away", "home"):
            gw = sorted(c["goalies"].get(side, []), key=lambda x: -x[1])
            sp[side] = (f'{gnames.get(str(gw[0][0]), gw[0][0])} ({100 * gw[0][1]:.0f}% likely)' if gw else None)
        c["sp_names"] = sp
        c["inputs_used"] = {"line": "ESPN/DraftKings" + (f', seen {v["line_seen"]}' if v.get("line_seen") else "") if c.get("home_ml") else "no line yet",
                            "goalies": "expected starters from each team's last 10 starts; a back-to-back shifts weight to the backup. "
                                       "Starters are confirmed only at puck drop."}
        if g["home_b2b"] or g["away_b2b"]:
            c["note"] = "Back-to-back: " + ", ".join(names.get(t, t) for t, b in ((v["away"], g["away_b2b"]), (v["home"], g["home_b2b"])) if b) + \
                        " played yesterday; the backup goalie is weighted more."
        cards.append(c)
    kick = {v["game_id"]: v["kickoff_utc"] for v in window}
    lname = "nhl_locks.csv" if live else "nhl_preview_locks.csv"
    led = Ledger(ledger_dir / lname, extra_cols=LOCK_EXTRA)
    sync = pickbook.sync(led, cards, kick, ledger_dir / "cards" / ("nhl" if live else "nhl_preview"), now, _clean, "nhl",
                         rows_for=lambda c: [r for r in mp.lock_rows(c) if r["market"] == "moneyline"])
    if live:
        alerts.collect("nhl", cards, names, kick, now)
    fin = {k: v["final"] for k, v in ev.items() if v.get("final")}
    for c in cards:
        f = fin.get(str(c["game_id"]))
        if f and c.get("ml_pick"):
            c["final_away"], c["final_home"] = f
            won = (f[1] > f[0]) == (c["ml_pick"] == "home")
            c["ml_result"] = "WIN" if won else "LOSS"
            c["result_label"] = "ML"
    led.grade(pd.DataFrame([{"game_id": k, "away_score": a, "home_score": h} for k, (a, h) in fin.items()],
                           columns=["game_id", "away_score", "home_score"]))
    rows = led.load_graded()
    graded = rows[rows["result"].isin(["WIN", "LOSS", "PUSH"])] if len(rows) else rows
    summ = record_summary(graded, EDGE_TIERS["nhl"])
    report.update(nhl_locked_now=sync["locked"] + sync["updated"], nhl_chain_ok=led.verify_chain(), nhl_live=live)
    lab = "Live locked record" if live else "Preview tracker (not the record)"
    withl = [c for c in cards if c.get("ml_edge") is not None]
    big = max(withl, key=lambda c: c["ml_edge"]) if withl else None
    stats = [{"k": lab, "v": f'{summ[0]["W"]}-{summ[0]["L"]}-{summ[0]["P"]}',
              "s": f"{len(rows)} picks locked" + ("" if live else f" · record starts {NHL_RECORD_FROM:%b %-d}"), "href": "record.html"},
             {"k": "Win-chance gap 3+", "v": str(sum(c["ml_edge"] >= 3 for c in withl)), "s": "points over the no-vig market"}]
    if big:
        stats.append({"k": "Biggest gap", "v": f'{big["ml_edge"]:.1f}', "s": f'{names.get(big["away"], big["away"])} at '
                      f'{names.get(big["home"], big["home"])}', "href": f'game/{big["game_id"]}.html'})
    totals_tracker.publish(out, ledger_dir, "nhl", cards, kick, now, fin, names, stats)
    firstline.publish(out, ledger_dir, "nhl", cards, kick, now, fin, names, stats, ledger_dir / lname)
    lede = ("Expected goals from every unblocked shot (distance, angle, rebound, shot type, man advantage), opponent-adjusted "
            "team ratings for and against, and each expected starting goalie's goals saved above expected. Picks are on the "
            "moneyline: the side where our win chance beats DraftKings' no-vig chance; highlighted at 3+ points with value at the "
            "price. Picks update until puck drop; the last version is graded. ")
    lede += "" if live else (f"PREVIEW: until {NHL_RECORD_FROM:%B %-d} picks go on a separate preview tracker, not the record, and "
                             "send no alerts.")
    site.write(out, "nhl/index.html", site.page("NHL Model Board", site.board(
        "nhl", "NHL projections" + ("" if live else " (preview)"), lede, cards, names, None, stats), 1, "nhl"))
    for c in cards:
        site.write(out, f"nhl/game/{c['game_id']}.html", site.page(f"{names.get(c['away'], c['away'])} at "
                   f"{names.get(c['home'], c['home'])}", site.game_page(c, names), 2, "nhl"))
    site.write(out, "nhl/record.html", site.page("NHL Model Record", site.record_page(
        "nhl", "NHL record" if live else "NHL preview tracker",
        "Moneyline picks, locked until puck drop and graded after the final." + ("" if live else
         f" Preview picks are kept apart; the record starts at 0-0 on {NHL_RECORD_FROM:%B %-d, %Y}."),
        [{"title": lab, "note": "Hash-chained lock file; the last version before puck drop is graded.", "summary": summ,
          "graded": rows.assign(pick_line=rows["odds"]).sort_values("locked_at", ascending=False).to_dict("records")
          if len(rows) else []}], names), 1, "nhl"))
    (out / "nhl" / "cards.json").write_text(json.dumps(_clean(cards), indent=1, default=str))
    return (summ if (live and len(graded)) else None), ("Live record" if live else f"Preview: record starts {NHL_RECORD_FROM:%b %-d}")
