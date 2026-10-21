"""
Over/under tracker: never part of the record.

For every game on a live board, the newest pre-game total lean is saved (overwritten until kickoff, then frozen),
and graded after the final: WIN if the game went the way we leaned, LOSS if not, PUSH on the number.
File: ledger/totals_<sport>.csv. Page: <sport>/totals.html.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import site

COLS = ["game_id", "sport", "date", "away", "home", "kickoff_utc", "book_total", "model_total", "gap", "lean", "level",
        "updated_at", "final_total", "result"]


def _load(p: Path) -> pd.DataFrame:
    if p.exists():
        d = pd.read_csv(p, dtype={"game_id": str})
        for c in COLS:
            if c not in d:
                d[c] = None
        return d[COLS]
    return pd.DataFrame(columns=COLS)


def sync(ledger_dir: Path, sport: str, cards: list[dict], kickoff_utc: dict, now, finals: dict) -> pd.DataFrame:
    """Save each unstarted game's current lean, then grade every saved game that has a final.
    finals: game_id -> (away_score, home_score)."""
    p = Path(ledger_dir) / f"totals_{sport}.csv"
    d = _load(p).set_index("game_id", drop=False)
    stamp = pd.Timestamp(now).isoformat(timespec="seconds")
    for c in cards:
        gid = str(c["game_id"])
        ko = kickoff_utc.get(gid) or kickoff_utc.get(c["game_id"])
        if ko is None or pd.Timestamp(ko) <= pd.Timestamp(now):
            continue                                       # started: the last saved version stands
        ti = site.total_info(c, sport)
        if ti is None:
            continue
        d.loc[gid, COLS] = [gid, sport, c.get("date", ""), c["away"], c["home"], str(pd.Timestamp(ko)), ti["book"],
                            ti["ours"], ti["gap"], ti["lean"], ti["level"], stamp, None, None]
    for gid, r in d.iterrows():
        f = finals.get(str(gid))
        if not f or pd.isna(r["book_total"]) or r["lean"] == "EVEN":
            continue
        tot = float(f[0]) + float(f[1])
        book = float(r["book_total"])
        d.at[gid, "final_total"] = tot
        d.at[gid, "result"] = "PUSH" if tot == book else ("WIN" if (tot > book) == (r["lean"] == "OVER") else "LOSS")
    d = d.reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False)
    return d


def summary(d: pd.DataFrame, sport: str) -> list[dict]:
    g = d[d["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    g["absgap"] = g["gap"].astype(float).abs()
    t = site.TOTAL_TIERS.get(sport, (3, 6))
    out = []
    for label, sub in [("All leans", g), (f"Gap {t[0]:g}+", g[g["absgap"] >= t[0]]), (f"Gap {t[1]:g}+", g[g["absgap"] >= t[1]])]:
        w, l, pu = ((sub["result"] == r).sum() for r in ("WIN", "LOSS", "PUSH"))
        out.append({"label": label, "W": int(w), "L": int(l), "P": int(pu), "pct": w / (w + l) if (w + l) else float("nan"),
                    "units": round(w * 100 / 110 - l, 1)})
    return out


def page(sport: str, d: pd.DataFrame, names: dict) -> str:
    e = site.e
    summ = summary(d, sport)
    tiles = "".join(f'<div class="tile"><span class="k">{e(s["label"])}</span><span class="v">{s["W"]}-{s["L"]}-{s["P"]}</span>'
                    f'<span class="s">{site._f(100 * s["pct"]) if s["W"] + s["L"] else "—"}% · {s["units"]:+.1f} units at -110</span></div>'
                    for s in summ)
    dd = d.sort_values("kickoff_utc", ascending=False)
    rows = "".join(
        f"<tr><td>{e(str(r.date))}</td><td>{e(names.get(r.away, r.away))} at {e(names.get(r.home, r.home))}</td>"
        f"<td class=mono>{float(r.book_total):g}</td><td class=mono>{float(r.model_total):g}</td><td><b>{e(str(r.lean))}</b> "
        f"{float(r.gap):+.1f}</td><td class=mono>{'' if pd.isna(r.final_total) else format(float(r.final_total), 'g')}</td>"
        f"<td class='res-{r.result}'>{r.result if isinstance(r.result, str) else 'pending'}</td></tr>" for r in dd.itertuples())
    t = site.TOTAL_TIERS.get(sport, (3, 6))
    return ('<p><a class="more" href="index.html">Back to the board</a></p><h1>Over/under tracker</h1>'
            '<p class="lede">Our total against the book\'s on every game, saved until kickoff and graded after the final. '
            'This is a test of the totals model and is never part of the record.</p>'
            f'<div class="tiles">{tiles}</div>'
            f'<p class="note">Highlighted at a gap of {t[0]:g}+, strong at {t[1]:g}+. Break-even at -110 is 52.4%.</p>'
            '<div class="tbl"><table><tr><th>Date</th><th>Game</th><th>Book</th><th>Ours</th><th>Lean</th><th>Final total</th>'
            f'<th>Result</th></tr>{rows}</table></div>')


def publish(out, ledger_dir, sport, cards, kick, now, finals, names, stats):
    """Save/grade the over/under leans, write <sport>/totals.html and add a tile to the board's stats."""
    try:
        tt = sync(ledger_dir, sport, cards, kick, now, finals)
        site.write(out, f"{sport}/totals.html", site.page(f"{sport.upper()} over/under tracker", page(sport, tt, names), 1, sport))
        s0 = summary(tt, sport)[0]
        stats.append({"k": "Over/under tracker", "v": f'{s0["W"]}-{s0["L"]}-{s0["P"]}', "s": "not part of the record",
                      "href": "totals.html"})
    except Exception as ex:                      # a totals problem never breaks the board
        print(f"[totals] {sport}: {type(ex).__name__}: {ex}")
