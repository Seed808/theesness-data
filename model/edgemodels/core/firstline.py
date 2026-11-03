"""
First-line picks: a side on every game against the FIRST line we saw, saved once and never changed, graded after the
final. NO EDGE games get a side too. This is a comparison tracker (Sam-style grading), never part of the record:
the official record stays the last version before kickoff, with NO EDGE / HOLD rules.

File: ledger/firstline_<sport>.csv. Page: <sport>/firstline.html. Each card gets c["firstline"] for display.
Backfill: games locked before this tracker existed take the first locked version from the lock file (source=backfill).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import site
from .edges import fmt_line

COLS = ["game_id", "sport", "date", "away", "home", "kickoff_utc", "first_line", "model_line", "pick", "pick_line", "edge",
        "no_edge", "hold", "saved_at", "source", "away_score", "home_score", "result"]


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    d = pd.read_csv(p, dtype={"game_id": str})
    return {str(r["game_id"]): {c: r.get(c) for c in COLS} for r in d.to_dict("records")}


def _save(p: Path, rows: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values()), columns=COLS).to_csv(p, index=False)


def backfill(rows: dict, sport: str, lock_file: Path, kick: dict) -> None:
    if not Path(lock_file).exists():
        return
    led = pd.read_csv(lock_file, dtype={"game_id": str}).sort_values("locked_at", kind="stable")
    vf = Path(str(lock_file).replace(".csv", "_voids.csv"))
    voided = set()
    if vf.exists():                                   # games voided for a proven error stay out here too
        v = pd.read_csv(vf, dtype={"game_id": str})
        voided = set(v.loc[v.get("kind", pd.Series("void", index=v.index)).fillna("void") == "void", "game_id"])
    for r in led.drop_duplicates("game_id", keep="first").to_dict("records"):
        gid = str(r["game_id"])
        if gid in rows or gid in voided or r.get("pick") not in ("home", "away"):
            continue
        rows[gid] = {"game_id": gid, "sport": sport, "date": r.get("date"), "away": r["away"], "home": r["home"],
                     "kickoff_utc": kick.get(gid), "first_line": r["market_line"], "model_line": r["model_line"],
                     "pick": r["pick"], "pick_line": r["pick_line"], "edge": r["edge"], "no_edge": False, "hold": False,
                     "saved_at": r["locked_at"], "source": "backfill (first locked version)",
                     "away_score": None, "home_score": None, "result": None}


def publish(out, ledger_dir, sport, cards, kick, now, finals, names, stats, lock_file=None):
    """Save first-line picks for games not started, grade finished ones, attach to cards, write the page, add a tile."""
    try:
        p = Path(ledger_dir) / f"firstline_{sport}.csv"
        rows = _load(p)
        if not rows and lock_file is not None:
            backfill(rows, sport, lock_file, kick)
        stamp = pd.Timestamp(now).isoformat(timespec="seconds")
        ml = sport in site.ML_SPORTS
        for c in cards:
            gid = str(c["game_id"])
            pick = c.get("ml_pick") if ml else c.get("pick")
            if gid in rows or pick not in ("home", "away") or (not ml and c.get("market_line") is None):
                continue
            ko = kick.get(gid) or kick.get(c["game_id"])
            if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
                continue                                  # only before kickoff
            rows[gid] = {"game_id": gid, "sport": sport, "date": c.get("date"), "away": c["away"], "home": c["home"],
                         "kickoff_utc": str(pd.Timestamp(ko)), "first_line": c.get("ml_odds") if ml else c["market_line"],
                         "model_line": c["model_line"], "pick": pick, "pick_line": 0.0 if ml else c["pick_line"],
                         "edge": c.get("ml_edge") if ml else c.get("edge"),
                         "no_edge": bool(c.get("no_edge")), "hold": bool(c.get("hold")), "saved_at": stamp, "source": "live",
                         "away_score": None, "home_score": None, "result": None}
        for gid, r in rows.items():
            f = finals.get(gid)
            if not f or any(pd.isna(v) for v in f):
                continue
            a, h = float(f[0]), float(f[1])
            m = (h - a) if r["pick"] == "home" else (a - h)
            cov = m + float(r["pick_line"])
            r.update(away_score=a, home_score=h, result="WIN" if cov > 0 else ("LOSS" if cov < 0 else "PUSH"))
        _save(p, rows)
        for c in cards:
            r = rows.get(str(c["game_id"]))
            if r:
                team = names.get(c[r["pick"]], c[r["pick"]]) if r["pick"] in c else r["pick"]
                lab = (f'{team} {float(r["first_line"]):+.0f}' if sport in site.ML_SPORTS and not pd.isna(r["first_line"])
                       else f'{team} {fmt_line(float(r["pick_line"]))}')
                c["firstline"] = {"pick": lab,
                                  "result": r["result"] if isinstance(r["result"], str) else None}
        d = pd.DataFrame(list(rows.values()), columns=COLS)
        g = d[d["result"].isin(["WIN", "LOSS", "PUSH"])]
        w, l, pu = ((g["result"] == x).sum() for x in ("WIN", "LOSS", "PUSH"))
        site.write(out, f"{sport}/firstline.html", site.page(f"{sport.upper()} first-line picks", page(sport, d, names), 1, sport))
        stats.append({"k": "First-line picks", "v": f"{w}-{l}-{pu}", "s": "every game vs the first line · not the record",
                      "href": "firstline.html"})
    except Exception as ex:                               # never breaks the board
        print(f"[firstline] {sport}: {type(ex).__name__}: {ex}")


def page(sport: str, d: pd.DataFrame, names: dict) -> str:
    e = site.e
    g = d[d["result"].isin(["WIN", "LOSS", "PUSH"])]
    tiles = []
    for lab, sub in [("Every game", g), ("Games we also graded (not NO EDGE / HOLD)", g[~g["no_edge"].astype(bool) & ~g["hold"].astype(bool)]),
                     ("NO EDGE games only", g[g["no_edge"].astype(bool)])]:
        w, l, pu = ((sub["result"] == x).sum() for x in ("WIN", "LOSS", "PUSH"))
        pct = f"{100 * w / (w + l):.1f}%" if w + l else "—"
        tiles.append(f'<div class="tile"><span class="k">{e(lab)}</span><span class="v">{w}-{l}-{pu}</span><span class="s">{pct}</span></div>')
    dd = d.assign(_k=pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)
                  .fillna(pd.to_datetime(d["saved_at"], errors="coerce", utc=True))).sort_values("_k", ascending=False)
    rows = "".join(
        f"<tr><td>{e(str(r.date))}</td><td>{e(names.get(r.away, r.away))} at {e(names.get(r.home, r.home))}</td>"
        f"<td>{e(names.get(getattr(r, r.pick), getattr(r, r.pick)))} {fmt_line(float(r.pick_line))}"
        f"{' <small>(NO EDGE)</small>' if bool(r.no_edge) else ''}{' <small>(HOLD)</small>' if bool(r.hold) else ''}</td>"
        f"<td class=mono>{'' if pd.isna(r.away_score) else format(r.away_score, '.0f') + '–' + format(r.home_score, '.0f')}</td>"
        f"<td class='res-{r.result}'>{r.result if isinstance(r.result, str) else 'pending'}</td></tr>" for r in dd.itertuples())
    return ('<p><a class="more" href="index.html">Back to the board</a></p><h1>First-line picks</h1>'
            '<p class="lede">A side on every game against the first line we saw, saved once and never changed, graded after the '
            'final. This is how sites that grade every game against the opening number keep score. It is a comparison only: '
            'our official record is the last pick before kickoff, and NO EDGE / HOLD games are not in it.</p>'
            f'<div class="tiles">{"".join(tiles)}</div>'
            '<p class="note">Games locked before this tracker started (Nov 3, 2026) use the first version we locked, so games that '
            'were NO EDGE from the very start are only tracked from Nov 3 on.</p>'
            '<div class="tbl"><table><tr><th>Date</th><th>Game</th><th>First-line pick</th><th>Final</th><th>Result</th></tr>'
            f'{rows}</table></div>')
