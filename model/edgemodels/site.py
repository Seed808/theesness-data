"""
Static site generator: home, per-sport board, per-game breakdown, record.

Every number on a page is read from the card dicts the models produce.
Nothing is typed in by hand, so the page is exactly as good as the model.
"""
from __future__ import annotations

import html
from .core.edges import prob_to_american
import math
from pathlib import Path

from .core.edges import EDGE_TIERS, fmt_line

SPORT_NAMES = {"nfl": ("NFL", "pro football"), "ncaaf": ("NCAAF", "college football"),
               "nba": ("NBA", "pro basketball"), "cbb": ("CBB", "college basketball"),
               "mlb": ("MLB", "baseball"), "nhl": ("NHL", "hockey")}
UNIT = {"nfl": "points", "ncaaf": "points", "nba": "points", "cbb": "points", "mlb": "runs", "nhl": "goals"}

CSS = """
:root{
  /* TheEsness: night-game program. Crimson masthead, gold for our numbers, grey for the book. */
  --bg:#f1eeec; --panel:#ffffff; --panel2:#f7f3f1; --ink:#26181b; --muted:#6f6061; --rule:#e2d8d5;
  --crimson:#a3122a; --mast:#7d0d1f; --gold:#9c7426; --goldsoft:#e9d9b4; --book:#8a7f7d;
  --model:#a3122a; --market:#8a7f7d; --edge:#9c7426; --win:#1e7a45; --loss:#b3261e;
  --display:"Big Shoulders Display","Arial Narrow",Impact,sans-serif;
  --body:"Figtree",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"Figtree",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#1c1315; --panel:#271a1d; --panel2:#2f2023; --ink:#f3ebe3; --muted:#b5a59f; --rule:#43302f;
  --crimson:#e0364d; --mast:#5e0a17; --gold:#d8b46a; --goldsoft:#4a3a1f; --book:#9a8d88;
  --model:#e9c77f; --market:#9a8d88; --edge:#d8b46a; --win:#62c890; --loss:#ff7a6b; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#1c1315; --panel:#271a1d; --panel2:#2f2023; --ink:#f3ebe3; --muted:#b5a59f; --rule:#43302f;
  --crimson:#e0364d; --mast:#5e0a17; --gold:#d8b46a; --goldsoft:#4a3a1f; --book:#9a8d88;
  --model:#e9c77f; --market:#9a8d88; --edge:#d8b46a; --win:#62c890; --loss:#ff7a6b; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15.5px;line-height:1.55;margin:0;font-variant-numeric:tabular-nums}
.mast{background:var(--mast);border-bottom:4px solid #c9a45c}
.mast .in{max-width:1120px;margin:0 auto;padding:14px 16px 0;display:flex;flex-direction:column;gap:12px}
.brand{font-family:var(--display);font-weight:800;font-size:34px;line-height:.9;letter-spacing:.01em;color:#f1d79c;text-decoration:none;text-transform:uppercase}
.brand small{display:block;font-family:var(--body);font-weight:500;font-size:12.5px;letter-spacing:.02em;color:#f3c9cf;text-transform:none;margin-top:3px}
nav.sports{display:flex;flex-wrap:wrap;gap:2px}
.mast nav.sports a{font-family:var(--display);font-weight:700;font-size:18px;letter-spacing:.04em;text-decoration:none;color:#f6dfe2;padding:6px 12px 7px;border-radius:8px 8px 0 0}
.mast nav.sports a.on{background:var(--bg);color:var(--ink)}
.mast nav.sports a:hover{color:#fff}
.wrap{max-width:1120px;margin:0 auto;padding-inline:16px;padding-block:0 48px}
a{color:var(--crimson)} a:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
nav.weeks{margin-top:18px;gap:6px}
nav.weeks a{font-family:var(--body);font-size:13px;font-weight:600;text-decoration:none;color:var(--muted);padding:4px 10px;border:1px solid var(--rule);border-radius:999px;background:var(--panel)}
nav.weeks a.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
h1{font-family:var(--display);font-weight:800;font-size:clamp(40px,7.5vw,68px);line-height:.92;margin:26px 0 10px;text-wrap:balance;letter-spacing:.005em}
h2{font-family:var(--display);font-weight:700;font-size:26px;letter-spacing:.01em;margin:36px 0 12px;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px}
h2::before{content:"";width:14px;height:4px;background:var(--crimson);align-self:center;flex:none}
.lede{max-width:64ch;color:var(--muted)}
.eyebrow{font-size:12.5px;color:var(--muted);font-weight:500}
h2 .eyebrow{font-family:var(--body)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:0;margin:20px 0;background:var(--panel);border:1px solid var(--rule);border-radius:14px;overflow:hidden}
.tile{padding:14px 16px;text-decoration:none;color:var(--ink);display:flex;flex-direction:column;gap:1px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule);margin:0 -1px -1px 0}
a.tile:hover{background:var(--panel2)}
.tile .k{font-size:12.5px;font-weight:600;color:var(--muted)}
.tile .v{font-family:var(--display);font-size:34px;font-weight:800;line-height:1.05}
.tile .s{font-size:13px;color:var(--muted)}
.day{margin-top:30px}
.day h2{font-size:30px}
.board{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.game{background:var(--panel);border:1px solid var(--rule);border-radius:16px;padding:0 0 12px;display:flex;flex-direction:column;min-width:0;overflow:hidden}
.game > *{margin-inline:16px}
.ghead{margin:0 0 10px!important;padding:9px 16px;background:var(--panel2);border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:12.5px;color:var(--muted)}
.game.hot{border-color:var(--gold);box-shadow:0 0 0 1px var(--gold)}
.game.hot .ghead{background:color-mix(in srgb,var(--gold) 14%,var(--panel))}
.game.hot2{box-shadow:0 0 0 2px var(--gold)}
.game.hot2 .ghead{background:var(--gold);color:#2a1d07;border-bottom-color:var(--gold)}
.badge{flex:none;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;border:1px solid var(--rule);color:var(--muted);background:var(--panel)}
.badge.gap{border-color:var(--gold);color:var(--gold)}
.badge.gap2{background:#2a1d07;border-color:#2a1d07;color:#f1d79c}
.badge.none{border-style:dashed}
.meta{font-size:12.5px;color:var(--muted)}
.teamrow{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:2px 0}
.teamrow .t{font-weight:700;font-size:16px}
.teamrow .t small{color:var(--muted);font-weight:500;margin-left:6px;font-size:12.5px}
.teamrow .sc{font-family:var(--display);font-weight:800;font-size:34px;line-height:1}
.meter{margin-top:10px!important;margin-bottom:4px!important}
.mtrack{position:relative;height:22px}
.mtrack::before{content:"";position:absolute;left:0;right:0;top:10px;height:2px;background:var(--rule);border-radius:2px}
.mtrack .mid{position:absolute;left:50%;top:5px;width:1px;height:12px;background:var(--muted);opacity:.5}
.mgap{position:absolute;top:8px;height:6px;background:var(--goldsoft);border-radius:4px}
.game.hot .mgap{background:var(--gold);opacity:.55}
.mk{position:absolute;top:4px;width:14px;height:14px;margin-left:-7px;border-radius:50%}
.mk.book{border:2.5px solid var(--book);background:var(--panel)}
.mk.ours{background:var(--crimson);border:2.5px solid var(--panel);box-shadow:0 0 0 1px var(--crimson)}
.mends{display:flex;justify-content:space-between;gap:10px;font-size:11.5px;color:var(--muted)}
.mends span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:48%}
.mends span:last-child{text-align:right}
.mkey{font-size:12.5px;color:var(--muted);margin:-4px 0 8px}
.mkey i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:0}
.mkey i.b{border:2px solid var(--book)} .mkey i.o{background:var(--crimson)}
.mends .key i{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 3px 0 8px;vertical-align:0}
.mends .key i.b{border:2px solid var(--book)} .mends .key i.o{background:var(--crimson)}
dl.lines{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;margin:8px 16px 0;font-size:13.5px;padding-top:8px;border-top:1px dashed var(--rule)}
dl.lines dt{color:var(--muted)} dl.lines dd{margin:0;font-weight:500}
.pick{color:var(--crimson);font-weight:800}
.more{font-size:13px;font-weight:700;text-decoration:none;margin-top:10px}
.upd{font-size:12px;color:var(--muted);margin-top:6px} .upd b{color:var(--ink);font-weight:700}
.gaps{background:var(--panel);border:1px solid var(--gold);border-radius:14px;padding:14px 16px;margin:16px 0}
.gaps ol{margin:8px 0 6px;padding-left:24px;font-size:14px} .gaps li{margin:4px 0}
.numbers{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;background:var(--panel);border:1px solid var(--rule);border-radius:14px;overflow:hidden}
.numbers .tile{border-bottom:0}
.numbers .tile .v{font-size:28px;line-height:1.1}
@media (max-width:560px){.numbers{grid-template-columns:1fr}.numbers .tile{border-right:0;border-bottom:1px solid var(--rule)}}
.scorebig{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:end;background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:18px;border-top:5px solid var(--crimson)}
.scorebig .n{font-family:var(--display);font-size:clamp(52px,12vw,84px);font-weight:800;line-height:.9}
.scorebig .r{text-align:right}
.bars{display:flex;flex-direction:column;gap:8px;background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:16px}
.bar{display:grid;grid-template-columns:minmax(110px,170px) 1fr 52px;gap:10px;align-items:center;font-size:13.5px}
.track{position:relative;height:16px;background:linear-gradient(var(--muted),var(--muted)) center/1px 100% no-repeat}
.fill{position:absolute;top:2px;height:12px;background:var(--crimson);border-radius:3px}
.fill.away{background:var(--gold)}
.bar .val{text-align:right;font-weight:700}
.axis{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:var(--muted);margin-top:4px;padding-top:8px;border-top:1px dashed var(--rule)}
.note{font-size:13.5px;color:var(--muted);max-width:70ch}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:14px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 12px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
tr:last-child td{border-bottom:0}
th{font-size:12.5px;color:var(--muted);font-weight:600;background:var(--panel2)}
th:first-child,td:first-child{text-align:left}
td.mono{font-weight:500}
.pct{display:inline-grid;grid-template-columns:64px 28px;gap:6px;align-items:center}
.pct i{display:block;height:7px;background:var(--rule);position:relative;border-radius:4px;overflow:hidden}
.pct i b{position:absolute;inset:0 auto 0 0;background:var(--gold)}
.res-VOID{color:var(--muted);font-weight:700}.res-WIN{color:var(--win);font-weight:800}.res-LOSS{color:var(--loss);font-weight:800}.res-PUSH{color:var(--muted);font-weight:700}
.hist{display:inline-flex;gap:1px;align-items:flex-end;height:22px;vertical-align:middle}
.hist span{width:5px;background:var(--crimson);opacity:.8;border-radius:1px}
.flag{font-size:12px;padding:1px 8px;border:1px solid var(--gold);color:var(--gold);border-radius:999px}
footer{margin-top:48px;padding-top:16px;border-top:3px solid var(--crimson);font-size:12.5px;color:var(--muted)}
@media (max-width:560px){.brand{font-size:28px}.mast nav.sports a{font-size:16px;padding:5px 9px 6px}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800'
         '&family=Figtree:wght@400;500;600;700;800&display=swap">')

e = html.escape


def _f(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def page(title: str, body: str, depth: int, active: str = "", sports=("ncaaf", "nfl", "nba", "cbb", "mlb", "nhl"),
         brand: str = "TheEsness Predictions", standalone: bool = True) -> str:
    up = "../" * depth
    nav = "".join(f'<a href="{up}{s}/index.html" class="{"on" if s == active else ""}">{SPORT_NAMES[s][0]}</a>' for s in sports)
    first, *rest = brand.split(" ", 1)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1,viewport-fit=cover">') if standalone else ""
    tail = "</body></html>" if standalone else ""
    return (f"{head}<title>{e(title)}</title>{FONTS}<style>{CSS}</style>{'</head><body>' if standalone else ''}"
            f'<header class="mast"><div class="in"><a class="brand" href="{up}index.html">{e(first)}<small>{e(rest[0] if rest else "")}'
            '</small></a>'
            f'<nav class="sports" aria-label="Sports">{nav}</nav></div></header><div class="wrap">{body}'
            "<footer>Model projections for entertainment and research, not betting advice. Every number is produced by "
            f"the model code and graded after the final; nothing is typed in by hand.</footer></div>{tail}")


def _pick_text(c, names):
    if c.get("hold"):
        return '<span class="res-PUSH"><b>HOLD</b></span> <span class="eyebrow">starter not confirmed</span>'
    if c.get("market_line") is None:
        return "—"
    if c.get("void"):
        team = names.get(c[c["pick"]], c[c["pick"]])
        return f'<s>{e(team)} {fmt_line(c["pick_line"])}</s> <span class="res-PUSH"><b>VOID</b></span>'
    if c.get("no_edge"):
        return f'<span class="res-PUSH"><b>NO EDGE</b></span> <span class="eyebrow">({_f(c["edge"])})</span>'
    team = names.get(c[c["pick"]], c[c["pick"]])
    return f"{e(team)} {fmt_line(c['pick_line'])} <span class=\"eyebrow\">({_f(c['edge'])})</span>"


def _fav_text(line, c, names):
    if line is None:
        return "—"
    if abs(line) < 1e-9:
        return "Pick'em"
    team = c["home"] if line < 0 else c["away"]
    return f"{e(names.get(team, team))} {fmt_line(-abs(line))}"


def week_nav(weeks: list, current, href) -> str:
    """Week tabs. href(w) -> link for week w (relative to the page being written)."""
    items = "".join(f'<a href="{href(w)}" class="{"on" if w == current else ""}">{e(str(lab))}</a>' for w, lab in weeks)
    return f'<nav class="sports weeks" aria-label="Weeks">{items}</nav>'


def _result_line(c, names):
    if c.get("final_home") is not None and c.get("final_away") is not None and (
            c.get("result") is None or c.get("no_edge") or c.get("hold") or c.get("void")):
        if c.get("void"):
            return f'<dt>Final</dt><dd>{c["final_away"]:.0f}–{c["final_home"]:.0f} <span class="res-VOID">VOID</span></dd>'

        return f'<dt>Final</dt><dd>{c["final_away"]:.0f}–{c["final_home"]:.0f} <span class="eyebrow">no pick graded</span></dd>'
    if c.get("final_home") is None or c.get("result") is None:
        return ""
    return (f'<dt>Final</dt><dd>{c["final_away"]:.0f}–{c["final_home"]:.0f} '
            f'<span class="res-{c["result"]}">{c["result"]}</span></dd>')


def _ml(x):
    return "—" if x is None else (f"{x:+.0f}" if x > 0 else f"{x:.0f}")


def _mlb_lines(c, names):
    """Moneyline + total rows for a baseball card."""
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    out = (f'<dt>Starters</dt><dd>{e(str(c.get("sp_names", {}).get("away") or "TBD"))} vs '
           f'{e(str(c.get("sp_names", {}).get("home") or "TBD"))}</dd>'
           f'<dt>Book</dt><dd>{e(h)} {_ml(c.get("home_ml"))} / {e(a)} {_ml(c.get("away_ml"))}</dd>'
           f'<dt>Ours</dt><dd>{e(h)} {100 * c["p_home"]:.1f}% · fair {_ml(c["fair_home_ml"])} / {_ml(c["fair_away_ml"])}</dd>')
    if c.get("ml_pick"):
        hot = c["ml_edge"] >= EDGE_TIERS["mlb"][0] and c["ml_ev"] > 0
        team = names.get(c[c["ml_pick"]], c[c["ml_pick"]])
        out += (f'<dt>Play</dt><dd class="{"pick" if hot else ""}">{e(team)} {_ml(c["ml_odds"])} '
                f'<span class="eyebrow">({c["ml_edge"]:.1f} pts · EV {c["ml_ev"]:+.3f})</span></dd>')
    out += _total_row(c, "mlb")
    if c.get("final_home") is not None:
        res = " ".join(f'<span class="res-{r}">{m} {r}</span>' for m, r in
                       ((c.get("result_label", "ML"), c.get("ml_result")), ("Total", c.get("tot_result"))) if r)
        out += f'<dt>Final</dt><dd>{c["final_away"]:.0f}–{c["final_home"]:.0f} {res}</dd>'
    return out



def highlight_level(c: dict, sport: str) -> int:
    """The board's highlight rule, in one place (the board and the Telegram alerts both use it).
    0 = not highlighted, 1 = highlighted (gold outline), 2 = strong (solid gold; not used for MLB).
    Football/basketball: gap in spread points vs EDGE_TIERS[sport][0] / [1]; never for NO EDGE, HOLD or VOID.
    MLB: moneyline edge in percentage points of win chance >= EDGE_TIERS['mlb'][0] and positive value at the price."""
    if c.get("void") or c.get("hold"):
        return 0
    t = EDGE_TIERS[sport]
    if sport == "mlb":
        return int(c.get("ml_edge") is not None and c["ml_edge"] >= t[0] and c.get("ml_ev", 0) > 0)
    if c.get("no_edge") or c.get("edge") is None:
        return 0
    return 2 if c["edge"] >= t[1] else (1 if c["edge"] >= t[0] else 0)


# ---------------------------------------------------------------- over/under (shown and tracked, never in the record)
# Gap thresholds (highlighted, strong) in points (runs for MLB). Backtests found no reliable totals edge yet:
# NFL 2025-26 gaps of 3+ went 44-49, college 4+ went 377-373, MLB totals no better than the league average;
# the badge marks a big disagreement, not a proven play. The tracker on each sport's Totals page is the test.
TOTAL_TIERS = {"nfl": (3.0, 6.0), "ncaaf": (4.0, 8.0), "nba": (5.0, 8.0), "cbb": (5.0, 8.0), "mlb": (1.0, 1.5)}
TOTAL_SD = {"nfl": 13.1, "ncaaf": 16.5, "nba": 18.0, "cbb": 16.0}


def total_info(c: dict, sport: str) -> dict | None:
    """Book total, our total, gap, lean and over/under/push chances for one card (None if either total is missing)."""
    ours, book = c.get("model_total"), c.get("market_total")
    try:
        ours, book = float(ours), float(book)
    except (TypeError, ValueError):
        return None
    if math.isnan(ours) or math.isnan(book):
        return None
    gap = ours - book
    if sport == "mlb" and c.get("p_over") is not None:
        po, pu = float(c["p_over"]), float(c["p_under"])
        pp = max(0.0, 1 - po - pu)
    else:
        from scipy.stats import norm
        sd = TOTAL_SD.get(sport, 15.0)
        if float(book).is_integer():
            pp = float(norm.cdf(book + 0.5, ours, sd) - norm.cdf(book - 0.5, ours, sd))
            pu = float(norm.cdf(book - 0.5, ours, sd))
        else:
            pp, pu = 0.0, float(norm.cdf(book, ours, sd))
        po = 1 - pu - pp
    t = TOTAL_TIERS.get(sport, (99, 99))
    lvl = 2 if abs(gap) >= t[1] else (1 if abs(gap) >= t[0] else 0)
    return {"book": book, "ours": round(ours, 1), "gap": round(gap, 1), "lean": "OVER" if gap > 0 else "UNDER" if gap < 0 else "EVEN",
            "level": lvl, "p_over": po, "p_under": pu, "p_push": pp}


def _total_row(c, sport):
    ti = total_info(c, sport)
    if ti is None:
        return f'<dt>O/U</dt><dd>ours {_f(c.get("model_total"))}, book {_f(c.get("market_total"))}</dd>'
    badge = f' <span class="badge {"gap2" if ti["level"] == 2 else "gap"}">O/U</span>' if ti["level"] else ""
    res = ""
    if c.get("final_home") is not None and c.get("final_away") is not None and ti["lean"] != "EVEN":
        tot = float(c["final_home"]) + float(c["final_away"])
        r = "PUSH" if tot == ti["book"] else ("WIN" if (tot > ti["book"]) == (ti["lean"] == "OVER") else "LOSS")
        res = f' <span class="res-{r}">{tot:g} {r}</span>'
    return (f'<dt>O/U</dt><dd>book {ti["book"]:g}, ours {ti["ours"]:g} · <b>{ti["lean"]} {ti["gap"]:+.1f}</b>{badge}{res}</dd>')


def _totals_section(c, names):
    ti = total_info(c, c["sport"])
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    unit = UNIT[c["sport"]]
    if ti is None:
        return (f'<h2>Over/under</h2><p class="note">Our total: {_f(c.get("model_total"))} {unit} '
                f'({e(a)} {_f(c["proj_away"])} + {e(h)} {_f(c["proj_home"])}). No book total yet.</p>')
    built = f'{e(a)} {_f(c["proj_away"])} + {e(h)} {_f(c["proj_home"])} = {ti["ours"]:g}'
    if c.get("pace"):
        built += f', at {float(c["pace"]):.1f} possessions'
    t = TOTAL_TIERS.get(c["sport"], (0, 0))
    lvl = {0: "not highlighted", 1: f"highlighted ({t[0]:g}+)", 2: f"strong ({t[1]:g}+)"}[ti["level"]]
    return (f'<h2>Over/under <span class="eyebrow">tracked, not part of the record</span></h2>'
            f'<div class="tbl"><table><tr><th>Book total</th><th>Ours</th><th>Gap</th><th>Lean</th><th>Over</th><th>Under</th><th>Push</th></tr>'
            f'<tr><td class=mono>{ti["book"]:g}</td><td class=mono>{ti["ours"]:g}</td><td class=mono>{ti["gap"]:+.1f}</td>'
            f'<td><b>{ti["lean"]}</b></td><td class=mono>{100 * ti["p_over"]:.1f}%</td><td class=mono>{100 * ti["p_under"]:.1f}%</td>'
            f'<td class=mono>{100 * ti["p_push"]:.1f}%</td></tr></table></div>'
            f'<p class="note">How our total is built: {built}. Gap {lvl}. Chances assume a normal spread of outcomes around our '
            f'total{" (simulated games for baseball)" if c["sport"] == "mlb" else ""}. Totals backtests have not shown a reliable '
            'edge yet, so this is tracked on its own page and never counts in the record.</p>')


def _gap_level(c, sport):
    """0 none (<0.5), 1 small, 2 = tier 1 (3+ in football), 3 = tier 2 (6+), 4 = tier 3 (10+)."""
    if sport == "mlb" or c.get("edge") is None or c.get("market_line") is None:
        return None
    if c.get("no_edge"):
        return 0
    t = EDGE_TIERS[sport]
    ed = c["edge"]
    return 1 + sum(ed >= x for x in t)


def _gap_badge(c, sport):
    if c.get("void") or c.get("hold"):
        return ""
    lv = _gap_level(c, sport)
    if lv is None:
        return ""
    t = EDGE_TIERS[sport]
    if lv == 0:
        return '<span class="badge none">NO EDGE</span>'
    if lv == 1:
        return f'<span class="badge">GAP {_f(c["edge"])}</span>'
    cls = "gap2" if lv >= 3 else "gap"
    return f'<span class="badge {cls}">GAP {_f(c["edge"])} · {t[lv - 2]:g}+</span>'


def _updated_line(c, names):
    """'Updated 8:44 AM · was Tennessee +7.5' when the pick changed since it was first locked."""
    hist = [h for h in (c.get("history") or []) if h.get("event") in ("locked", "update", "no edge", "hold")]
    if len(hist) < 2:
        return ""
    last = hist[-1]
    first = next((h for h in hist if h.get("event") in ("locked", "update") and h.get("pick")), None)
    if first is None:
        return ""
    now_pick = None if last.get("event") in ("no edge", "hold") else last.get("pick")
    fl = first.get("pick_line")
    first_txt = f'{e(names.get(first["pick"], first["pick"]))} {fmt_line(fl) if isinstance(fl, (int, float)) else ""}'.strip()
    if now_pick == first.get("pick") and last.get("pick_line") == fl:
        what = "same pick as first posted"
    elif now_pick == first.get("pick"):
        what = f"first posted as {first_txt} (line moved)"
    else:
        what = f"first posted as {first_txt}"
    t = pd_ts(last["at"]).split(" ", 2)[-1]
    return f'<div class="upd"><b>Updated</b> {e(t)} HST · {what} · {len(hist)} versions</div>'


def _gaps_box(cards, sport, names, tier_record):
    """Top-of-board list of the largest model-vs-market gaps (tier 2+, e.g. 6+ points in football) for games
    still to be played and eligible to lock, with the backtest record of each tier beside it."""
    if sport == "mlb":
        return ""
    t1, t2 = EDGE_TIERS[sport][0], EDGE_TIERS[sport][1]
    live = [c for c in cards if c.get("edge") is not None and not c.get("void") and not c.get("hold")
            and not c.get("no_edge") and c.get("result") is None
            and not any(k in str(c.get("kickoff") or "") for k in ("FCS", "started", "VOID"))]
    n1 = sum(1 for c in live if c["edge"] >= t1)
    big = sorted([c for c in live if c["edge"] >= t2], key=lambda c: -c["edge"])
    if not n1:
        return ""
    items = "".join(f'<li><b>{e(names.get(c[c["pick"]], c[c["pick"]]))} {fmt_line(c["pick_line"])}</b> '
                    f'<span class="eyebrow">gap {_f(c["edge"])} · {e(names.get(c["away"], c["away"]))} at '
                    f'{e(names.get(c["home"], c["home"]))} · {e(str(c.get("kickoff") or ""))}</span></li>' for c in big)
    recs = []
    for r in tier_record or []:
        for t in (t1, t2):
            if r["label"] in (f"Edge {t}+", f"Edge {t:g}+") and (r["W"] + r["L"]):
                recs.append(f'{t:g}+ points went {r["W"]}-{r["L"]}-{r["P"]} ({r["pct"]:.1%})')
    rec = (" In the backtest, gaps of " + " and ".join(recs) + "; 52.4% is break-even at -110.") if recs else ""
    return (f'<div class="gaps"><b>Biggest gaps vs the book, {t2:g}+ points ({len(big)})</b>'
            + (f'<ol>{items}</ol>' if items else '<p class="note">None this slate.</p>')
            + f'<p class="note">{n1} games have a gap of {t1:g}+ points and are highlighted below (darker at {t2:g}+).{rec} '
            'A big gap is a question worth asking (injury, weather, a team the ratings misread), not a signal to follow.</p></div>')


def board(sport: str, title: str, lede: str, cards: list[dict], names: dict, records: dict | None = None,
          stats: list[dict] | None = None, nav: str = "", game_prefix: str = "game/", links: bool = True,
          newest_first: bool = False, tier_record: list | None = None) -> str:
    """newest_first: for boards of games already played (replays, past weeks) the latest day and latest
    game come first; live boards of upcoming games stay in schedule order."""
    tier1 = EDGE_TIERS[sport][0]
    tiles = "".join(f'<a class="tile" href="{s.get("href", "#board")}"><span class="k">{e(s["k"])}</span>'
                    f'<span class="v">{e(s["v"])}</span><span class="s">{e(s.get("s", ""))}</span></a>' for s in (stats or []))
    days: dict[str, list] = {}
    for c in cards:
        days.setdefault(c.get("date", ""), []).append(c)
    out = [f'{nav}<h1>{e(title)}</h1><p class="lede">{e(lede)}</p><div class="tiles">{tiles}</div>'
           + ("" if newest_first else _gaps_box(cards, sport, names, tier_record))
           + ('' if sport == "mlb" else '<p class="mkey">On each card the line shows where the spread lands, from the away team on the '
              'left to the home team on the right:<i class="b"></i>the book<i class="o"></i>our number. The shaded stretch between them is the gap.</p>')
           + '<div id="board"></div>']
    items = list(days.items())                 # cards arrive in schedule order
    if newest_first:
        items = [(d, cs[::-1]) for d, cs in items[::-1]]
    for d, cs in items:
        out.append(f'<section class="day"><h2>{e(d)} <span class="eyebrow">{len(cs)} game{"s" if len(cs) != 1 else ""}</span></h2><div class="board">')
        for c in cs:
            hl = highlight_level(c, sport)
            hot, strong = hl >= 1, hl >= 2
            rec = (records or {})
            ab = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
            out.append(
                f'<article class="game{" hot" if hot else ""}{" hot2" if strong else ""}">'
                f'<div class="ghead"><span>{e(str(c.get("kickoff") or ""))}{", " + e(c["venue"]) if c.get("venue") else ""}</span>'
                f'{_gap_badge(c, sport)}</div>'
                f'<div class="teamrow"><span class="t">{e(ab[0])}<small>{e(rec.get(c["away"], ""))}</small></span>'
                f'<span class="sc">{_f(c["proj_away"])}</span></div>'
                f'<div class="teamrow"><span class="t">{e(ab[1])}<small>{e(rec.get(c["home"], ""))}</small></span>'
                f'<span class="sc">{_f(c["proj_home"])}</span></div>'
                + (f'<dl class="lines">{_mlb_lines(c, names)}</dl>' if sport == "mlb" else
                   _meter(c, names) +
                   f'<dl class="lines"><dt>Book</dt><dd>{_fav_text(c.get("market_line"), c, names)}</dd>'
                   f'<dt>Ours</dt><dd>{_fav_text(c.get("model_line"), c, names)}</dd>'
                   f'<dt>Play</dt><dd class="{"pick" if hot else ""}">{_pick_text(c, names)}</dd>'
                   + _total_row(c, sport) +
                   f'{_result_line(c, names)}</dl>{_updated_line(c, names)}')
                + (f'<a class="more" href="{game_prefix}{e(str(c["game_id"]))}.html">Full breakdown</a>' if links else "")
                + '</article>')
        out.append("</div></section>")
    return "".join(out)



def _meter(c, names):
    """Spread meter: where the book and our model put the margin, on one line from the away side to the home side.
    The shaded stretch between the two dots is the gap."""
    ml, bk = c.get("model_line"), c.get("market_line")
    if ml is None:
        return ""
    ours = -float(ml)                       # projected home margin
    book = -float(bk) if bk is not None else None
    span = max(10.0, abs(ours) + 3, abs(book) + 3 if book is not None else 0)
    span = min(span, 45.0)
    pos = lambda v: 50 + max(-span, min(span, v)) / span * 50
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    gap = ""
    marks = f'<span class="mk ours" style="left:{pos(ours):.1f}%"></span>'
    label = f"Ours {_fav_text(ml, c, names)}"
    if book is not None:
        lo, hi = sorted((pos(ours), pos(book)))
        gap = f'<span class="mgap" style="left:{lo:.1f}%;width:{hi - lo:.1f}%"></span>'
        marks = f'<span class="mk book" style="left:{pos(book):.1f}%"></span>' + marks
        label = f"Book {_fav_text(bk, c, names)}, " + label
    return (f'<div class="meter" role="img" aria-label="{e(label)}"><div class="mtrack"><span class="mid"></span>{gap}{marks}</div>'
            f'<div class="mends"><span>{e(a)}</span><span>{e(h)}</span></div></div>')

def _bars(c, names):
    bd = c["breakdown"]
    m = max([abs(v) for v in bd.values()] + [0.5])
    rows = []
    for k, v in bd.items():
        w = abs(v) / m * 50
        left = 50 if v >= 0 else 50 - w
        cls = "" if v >= 0 else "away"
        rows.append(f'<div class="bar"><span>{e(k)}</span><span class="track"><span class="fill {cls}" '
                    f'style="left:{left:.1f}%;width:{w:.1f}%"></span></span><span class="val">{abs(v):.1f}</span></div>')
    total = sum(bd.values())
    fav = c["home"] if total >= 0 else c["away"]
    return (f'<div class="bars">{"".join(rows)}<div class="axis"><span>helps {e(names.get(c["away"], c["away"]))}</span>'
            f'<span>helps {e(names.get(c["home"], c["home"]))}</span></div></div>'
            f'<p class="note">Each bar is how many {UNIT[c["sport"]]} that piece adds to the spread, on the side of the team it helps. '
            f'Together they come to {abs(total):.1f} for {e(names.get(fav, fav))}.</p>')


def _pct(v):
    if v is None:
        return "—"
    return f'<span class="pct"><i><b style="width:{v}%"></b></i><span>{v}</span></span>'


def _ratings(c, names):
    r = c.get("ratings")
    if not r:
        return ""
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    head = f"<tr><th>Percentile</th><th>{e(a)} off</th><th>{e(h)} off</th><th>{e(a)} def</th><th>{e(h)} def</th></tr>"
    rows = "".join(f"<tr><td>{e(x['short'])}</td><td>{_pct(x['away_off_pct'])}</td><td>{_pct(x['home_off_pct'])}</td>"
                   f"<td>{_pct(x['away_def_pct'])}</td><td>{_pct(x['home_def_pct'])}</td></tr>" for x in r["rows"])
    nd = 3 if c["sport"] in ("nfl", "ncaaf", "mlb") or True else 1
    def fv(x):
        return "—" if x is None else (f"{x:+.3f}" if abs(x) < 2 else f"{x:+.1f}")
    head2 = f"<tr><th>vs league average</th><th>{e(a)} off</th><th>{e(h)} off</th><th>{e(a)} def</th><th>{e(h)} def</th></tr>"
    rows2 = "".join(f"<tr><td>{e(x['label'])}</td><td class=mono>{fv(x['away_off'])}</td><td class=mono>{fv(x['home_off'])}</td>"
                    f"<td class=mono>{fv(x['away_def'])}</td><td class=mono>{fv(x['home_def'])}</td></tr>" for x in r["rows"])
    return (f'<h2>League percentiles <span class="eyebrow">100 is the best in the league</span></h2><div class="tbl"><table>{head}{rows}</table></div>'
            f'<h2>Adjusted efficiency</h2><div class="tbl"><table>{head2}{rows2}</table></div>'
            f'<p class="note">Adjusted for who each team played, and blended with last season ({e(r.get("through", ""))}). '
            f'For defense a negative number is good; percentiles already flip it so higher is better. '
            f'{"Pass rate and pace are" if c["sport"] in ("nfl", "ncaaf") else "Pace is"} style, ranked from low to high.</p>')


def _hist(h):
    if not h:
        return ""
    mx = max(h["share"]) or 1
    return '<span class="hist">' + "".join(f'<span style="height:{max(1, 22 * s / mx):.0f}px"></span>' for s in h["share"]) + "</span>"


def _ladder(s):
    return " ".join(f'<span class="mono">{l["line"]:g}: {100 * l["p_over"]:.0f}%</span>' for l in s.get("ladder", []))


def _actual_cell(v, s):
    if v is None:
        return "<td class=mono>—</td>"
    side = "over" if v > s["median"] else ("under" if v < s["median"] else "at")
    return f'<td class=mono><b>{v:.0f}</b> <span class=eyebrow>{side} median</span></td>'


def _td_cell(a):
    if not a:
        return "<td class=mono>—</td>"
    if "home_run" in a and "tds" not in a:
        t = a["home_run"]
        txt = (f'<span class="res-WIN">Homered{" ×" + str(t) if t > 1 else ""}</span>' if t else '<span class="res-LOSS">No HR</span>')
        return f"<td class=mono>{txt}</td>"
    t = a.get("tds", 0)
    txt = (f'<span class="res-WIN">Scored{" ×" + str(t) if t > 1 else ""}</span>' if t else '<span class="res-LOSS">No TD</span>')
    return f"<td class=mono>{txt}{' · <b>first TD</b>' if a.get('first_td') else ''}</td>"


def _props(c):
    pr = c.get("props")
    if not pr:
        return ""
    rows = []
    fin = bool(pr.get("final"))
    stat_keys = [("receptions", "Rec"), ("receiving_yards", "Rec yds"), ("rushing_yards", "Rush yds"),
                 ("points", "Pts"), ("rebounds", "Reb"), ("assists", "Ast"), ("threes", "3PM"), ("pra", "PRA"),
                 ("hits", "Hits"), ("total_bases", "TB"), ("rbi", "RBI"), ("strikeouts", "K"), ("hits_allowed", "H allowed")]
    for p in pr.get("players", []):
        for k, lab in stat_keys:
            if k in p:
                s = p[k]
                mk = ""
                if "line" in s:
                    mk = f'{s["line"]:g} · {s["side"]} {100 * max(s["p_over"], s["p_under"]):.0f}%'
                rows.append(f"<tr><td>{e(p['player'])} <span class=eyebrow>{e(p['team'])}</span></td><td>{lab}</td>"
                            f"<td class=mono>{s['mean']:.1f}</td><td class=mono>{s['median']:.0f}</td>"
                            f"<td class=mono>{s['p10']:.0f}–{s['p90']:.0f}</td><td>{_hist(s.get('hist'))}</td>"
                            f"<td>{_ladder(s)}</td><td class=mono>{mk or '—'}</td>"
                            + (_actual_cell(p.get("actual", {}).get(k), s) if fin else "") + "</tr>")
    tbl = ""
    if rows:
        tbl = ('<div class="tbl"><table><tr><th>Player</th><th>Prop</th><th>Mean</th><th>Median</th><th>10–90%</th>'
               '<th>Distribution</th><th>P(over) ladder</th><th>vs line</th>' + ("<th>Actual</th>" if fin else "") + '</tr>' + "".join(rows) + "</table></div>")
    yn = []
    for p in pr.get("players", []):
        cells = []
        for k in ("anytime_td", "first_td", "first_basket", "home_run"):
            if isinstance(p.get(k), dict):
                cells.append((p[k]["label"], p[k]["p_yes"], p[k]["fair_odds"]))
        if cells:
            yn.append(f"<tr><td>{e(p['player'])} <span class=eyebrow>{e(p['team'])}</span></td>" +
                      "".join(f"<td class=mono>{100 * pv:.1f}% <span class=eyebrow>{fo:+d}</span></td>" for _, pv, fo in cells)
                      + (_td_cell(p.get("actual")) if fin else "") + "</tr>")
    ytbl = ""
    if yn:
        labs = [lab for lab in ("Anytime TD", "First TD", "First basket (FG)", "Home run")
                if any(isinstance(p.get(k), dict) and lab == p[k].get("label")
                       for p in pr["players"] for k in ("anytime_td", "first_td", "first_basket", "home_run"))]
        ytbl = ('<div class="tbl"><table><tr><th>Player</th>' + "".join(f"<th>{l} · fair odds</th>" for l in labs) + ("<th>Actual</th>" if fin else "") + "</tr>" + "".join(yn) + "</table></div>")
    note = pr.get("note", "")
    if fin:
        note = ((f"Final. First touchdown: {pr.get('first_td_actual') or 'none'}. Actual results are from nflverse "
                 "play-by-play; the projections were made before kickoff. ") if c["sport"] == "nfl" else
                "Final. Actual results are from Statcast for this game; the projections were made before first pitch. ") + note
    return (f'<h2>Player props <span class="eyebrow">{pr.get("n_sims", 20000):,} simulated games</span></h2>{tbl}'
            + (f'<h2>{"Home runs" if c["sport"] == "mlb" else "Scorer markets"}</h2>{ytbl}' if ytbl else "")
            + f'<p class="note">{e(note)}</p>')


def _mlb_number(c, names):
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    fav = h if c["p_home"] >= 0.5 else a
    pf = max(c["p_home"], 1 - c["p_home"])
    num = f'<p class="lede">The model makes {e(fav)} a {100 * pf:.1f}% favorite (fair price {_ml(prob_to_american(pf))}).'
    if c.get("market_p_home") is not None:
        mp = c["market_p_home"] if fav == h else 1 - c["market_p_home"]
        team = names.get(c[c["ml_pick"]], c[c["ml_pick"]])
        num += (f' The market, with the bookmaker margin removed, has {e(fav)} at {100 * mp:.1f}%. '
                f'The side the model likes better than the market is {e(team)} at {_ml(c["ml_odds"])}: '
                f'model {100 * c["ml_prob"]:.1f}%, {c["ml_edge"]:.1f} points above the market, '
                f'expected value {c["ml_ev"]:+.3f} per unit at that price.')
        if c["ml_ev"] <= 0:
            num += " That is below zero after the price, so it is not a bet."
    else:
        num += " No moneyline was available when this was written."
    num += "</p>"
    rows = ""
    if c.get("tot_pick"):
        rows = (f'<div class="tbl" style="margin-top:10px"><table><tr><th>Total {_f(c["market_total"])}</th><th>Over</th>'
                f'<th>Under</th><th>Push</th><th>Model lean (not a pick)</th><th>EV at {_ml(c["tot_odds"])}</th></tr><tr><td>Simulated</td>'
                f'<td class=mono>{100 * c["p_over"]:.1f}%</td><td class=mono>{100 * c["p_under"]:.1f}%</td>'
                f'<td class=mono>{100 * (1 - c["p_over"] - c["p_under"]):.1f}%</td><td>{e(c["tot_pick"])}</td>'
                f'<td class=mono>{c["tot_ev"]:+.3f}</td></tr></table></div>')
    tiles = (f'<div class="numbers"><div class="tile"><span class="k">Moneyline</span><span class="v">{_ml(c.get("home_ml"))}</span>'
             f'<span class="s">{e(h)} · {e(a)} {_ml(c.get("away_ml"))}</span></div>'
             f'<div class="tile"><span class="k">Model fair price</span><span class="v">{_ml(c["fair_home_ml"])}</span>'
             f'<span class="s">{e(h)} {100 * c["p_home"]:.1f}% · {e(a)} {_ml(c["fair_away_ml"])}</span></div>'
             f'<div class="tile"><span class="k">Run line (model)</span><span class="v">{_fav_text(c["model_line"], c, names)}</span>'
             f'<span class="s">projected margin</span></div></div>{rows}'
             '<p class="note">Totals are shown, never locked: in the 2026 backtest the model\'s total-runs error was no '
             'better than guessing the league average. Win and total probabilities come from 40,000 simulated games: each team\'s runs drawn from a '
             'negative binomial around its projected runs, extra innings split by run expectancy. Market probabilities have the '
             'bookmaker margin removed. Lines are the first DraftKings numbers ESPN showed that day.</p>')
    return num, tiles


def _inputs(c, names):
    """What the projection used: starters and how they were confirmed, the line and when it was seen, history."""
    iu = c.get("inputs_used")
    out = ""
    if iu:
        rows = ""
        for q in iu.get("qb", []):
            st = q.get("status", "")
            inj = "; ".join(f"{k}: {v}" for k, v in (q.get("injury") or {}).items())
            rows += (f"<tr><td>{e(names.get(q['team'], q['team']))}</td><td>{e(str(q.get('starter') or '—'))}</td>"
                     f"<td class='{'res-WIN' if st == 'confirmed' else 'res-LOSS'}'>{e(st)}</td>"
                     f"<td>{e(str(q.get('last_game') or '—'))} (wk {e(str(q.get('last_game_week') or '—'))})</td>"
                     f"<td>{e(str(q.get('espn_qb1') or '—'))}</td><td>{e(inj or '—')}</td></tr>")
        for p in iu.get("pitchers", []):
            rows += (f"<tr><td>{e(names.get(p['team'], p['team']))}</td><td>{e(str(p.get('starter') or '—'))}</td>"
                     f"<td class='{'res-WIN' if p.get('status') == 'confirmed' else 'res-LOSS'}'>{e(p.get('status', ''))}</td>"
                     f"<td>MLB: {e(str(p.get('mlb') or '—'))}</td><td>ESPN: {e(str(p.get('espn') or '—'))}</td><td>{e(p.get('lineup', ''))}</td></tr>")
        head = ("<tr><th>Team</th><th>Starter used</th><th>Check</th><th>Last game</th><th>ESPN depth QB1</th><th>Injury report</th></tr>"
                if iu.get("qb") else "<tr><th>Team</th><th>Starter used</th><th>Check</th><th>Source 1</th><th>Source 2</th><th>Lineup</th></tr>")
        out += (f'<h2>What went in</h2>' + (f'<div class="tbl"><table>{head}{rows}</table></div>' if rows else "")
                + f'<p class="note">Line: {e(str(iu.get("line") or "—"))}.'
                + (f' Elo: {e(iu["elo"])}.' if iu.get("elo") else "")
                + (f' Injuries: {e(iu["injury_report"])}.' if iu.get("injury_report") else "")
                + ' A starter is used only when the sources agree (or the injury report settles it); otherwise the game is on HOLD and has no pick.</p>')
    hist = c.get("history") or []
    if hist:
        rows = "".join(
            f"<tr><td class=mono>{e((pd_ts(h['at'])))}</td><td>{e(h.get('event', ''))}</td>"
            f"<td>{e(str(h.get('pick') or '—'))} {fmt_line(h['pick_line']) if isinstance(h.get('pick_line'), (int, float)) and h.get('event') != 'hold' else ''}</td>"
            f"<td class=mono>{_f(h.get('model_line')) if h.get('model_line') is not None else '—'}</td>"
            f"<td class=mono>{_f(h.get('market_line')) if isinstance(h.get('market_line'), (int, float)) else '—'}</td>"
            f"<td>{e(str(h.get('starters') or h.get('why') or ''))}</td></tr>" for h in reversed(hist))
        out += ('<h2>How the pick moved</h2><div class="tbl"><table><tr><th>Time (HST)</th><th>Event</th><th>Pick</th><th>Ours</th>'
                f'<th>Book</th><th>Starters / reason</th></tr>{rows}</table></div><p class="note">Picks update until kickoff '
                'with the newest roster and line; the last version before kickoff is the one graded.</p>')
    return out


def pd_ts(iso):
    from datetime import datetime, timedelta
    try:
        return (datetime.fromisoformat(iso) - timedelta(hours=10)).strftime("%b %-d %-I:%M %p")
    except Exception:
        return str(iso)


def game_page(c: dict, names: dict) -> str:
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    unit = UNIT[c["sport"]]
    if c.get("market_line") is not None:
        side = names.get(c[c["pick"]], c[c["pick"]])
        # which team the model rates higher than the market does, on points alone
        pts_team = c["home"] if c["model_line"] < c["market_line"] else c["away"]
        pts_side = names.get(pts_team, pts_team)
        num = (f'<p class="lede">We have {_fav_text(c["model_line"], c, names)}; the book has '
               f'{_fav_text(c["market_line"], c, names)}. That is {_f(c["edge"])} {unit} more on {e(pts_side)} '
               f'than the book gives.')
        if c[c["pick"]] != pts_team and c.get("cover_prob") is not None and c.get("lose_prob") is not None:
            num += (f' The pick is still {e(side)} {fmt_line(c["pick_line"])}: final margins land on key numbers '
                    f'(3, 7) far more often than others, so at this line {e(side)} covers more often '
                    f'({100 * c["cover_prob"]:.1f}% vs {100 * c["lose_prob"]:.1f}%).')
        if c.get("ev") is not None and c["ev"] <= 0:
            num += ' At -110 neither side is worth a bet (expected value below zero).'
        num += '</p>'
    else:
        num = f'<p class="lede">We have {_fav_text(c["model_line"], c, names)}. No book line was posted yet.</p>'
    if c["sport"] == "mlb":
        num, tiles = _mlb_number(c, names)
    cp = c.get("cover_prob")
    tiles = tiles if c["sport"] == "mlb" else (f'<div class="numbers"><div class="tile"><span class="k">Book</span><span class="v">{_fav_text(c.get("market_line"), c, names)}</span></div>'
             f'<div class="tile"><span class="k">Ours</span><span class="v">{_fav_text(c["model_line"], c, names)}</span></div>'
             f'<div class="tile"><span class="k">Gap</span><span class="v">{_f(c.get("edge"))}</span>'
             f'<span class="s">{"pick covers " + format(cp, ".0%") + " of the time" if cp else ""}</span></div></div>')
    if c.get("push_prob") is not None and c.get("lose_prob") is not None and c.get("cover_prob") is not None and c.get("ev") is not None:
        side = names.get(c[c["pick"]], c[c["pick"]])
        tiles += (f'<div class="tbl" style="margin-top:10px"><table><tr><th>At {e(side)} {fmt_line(c["pick_line"])}</th>'
                  f'<th>Covers</th><th>Push</th><th>Loses</th><th>Value per unit at -110</th><th>Fair line</th></tr>'
                  f'<tr><td>Using real final margins</td><td class=mono>{100 * c["cover_prob"]:.1f}%</td>'
                  f'<td class=mono>{100 * c["push_prob"]:.1f}%</td><td class=mono>{100 * c["lose_prob"]:.1f}%</td>'
                  f'<td class=mono>{c["ev"]:+.3f}</td><td class=mono>{_fav_text(c.get("fair_line"), c, names)}</td></tr></table></div>'
                  '<p class="note">These chances come from how often real games end on each margin (3 and 7 come up far '
                  'more than anything else), centred on our projected margin. Value above zero means the side is worth more '
                  'than the -110 price. In backtests these model-only chances run high, because the book\'s line knows things '
                  'the model does not.</p>')
    total_txt = ""
    qb = ""
    if c.get("qb") and c["sport"] == "nfl":     # college passer names in the free feed are unreliable
        rows = "".join(f"<tr><td>{e(str(q['starter']))}</td><td>{e(names.get(q['team'], q['team']))}</td>"
                       f"<td class=mono>{q['adj']:+.3f}</td></tr>" for q in c["qb"])
        qb = ('<h2>Quarterbacks</h2><div class="tbl"><table><tr><th>Starter</th><th>Team</th><th>Adjustment, EPA / dropback</th></tr>'
              f'{rows}</table></div><p class="note">Each listed starter measured against whoever actually played in the games '
              'these ratings were built from. Zero means the starter is the quarterback already in the numbers.</p>')
    extra = ""
    if c["sport"] == "mlb" and c.get("pitchers"):
        nm = c.get("sp_names") or {}
        opp = {"home": "away", "away": "home"}
        rows = "".join(f"<tr><td>{e(str(nm.get(opp[s]) or v['sp']))}</td><td>vs {e(names.get(c[s], c[s]))} bats</td><td class=mono>{v['sp_bf']:.1f}</td>"
                       f"<td class=mono>{v['sp_term']:+.3f}</td><td class=mono>{v['pen_term']:+.3f}</td></tr>"
                       for s, v in c["pitchers"].items())
        extra = ('<h2>Pitching</h2><div class="tbl"><table><tr><th>Starter faced</th><th></th><th>Proj. batters faced</th>'
                 f'<th>SP quality allowed</th><th>Bullpen quality allowed</th></tr>{rows}</table></div>')
    return (f'<p><a class="more" href="../index.html">Back to the board</a></p><h1>{e(a)} at {e(h)}</h1>'
            f'<p class="eyebrow">{e(str(c.get("date", "")))} {e(str(c.get("kickoff") or ""))} {("· " + e(c["venue"])) if c.get("venue") else ""}</p>'
            + (f'<p class="note" style="border-left:3px solid var(--edge);padding-left:10px">{e(c["note"])}</p>' if c.get("note") else "")
            + (f'<p class="note" style="border-left:3px solid var(--loss);padding-left:10px"><b>HOLD.</b> {e(c["hold"])}</p>' if c.get("hold") else "")
            + (f'<p class="note" style="border-left:3px solid var(--edge);padding-left:10px"><b>NO EDGE.</b> {e(c["no_edge"])}</p>' if c.get("no_edge") else "")
            + f'<h2>Line check</h2>{num}{tiles}'
            f'<h2>Projected final</h2><div class="scorebig"><div><div class="eyebrow">{e(a)}</div><div class="n">{_f(c["proj_away"])}</div></div>'
            f'<div class="eyebrow">at</div><div class="r"><div class="eyebrow">{e(h)}</div><div class="n">{_f(c["proj_home"])}</div></div></div>{total_txt}'
            f'{_totals_section(c, names)}<h2>What builds the spread</h2>{_bars(c, names)}{qb}{extra}{_inputs(c, names)}{_ratings(c, names)}{_props(c)}')


def record_page(sport: str, title: str, lede: str, sections: list[dict], names: dict) -> str:
    """sections: [{"title", "note", "summary", "graded"}] shown in order, each with its own tiles and list."""
    out = [f'<p><a class="more" href="index.html">Back to the board</a></p><h1>{e(title)}</h1><p class="lede">{e(lede)}</p>']
    for sec in sections:
        tiles = "".join(f'<div class="tile"><span class="k">{e(s["label"])}</span><span class="v">{s["W"]}-{s["L"]}-{s["P"]}</span>'
                        f'<span class="s">{_f(100 * s["pct"]) if s["W"] + s["L"] else "—"}% · {s["units"]:+.1f} units</span></div>'
                        for s in sec["summary"])
        rows = "".join(
            f"<tr><td class=mono>{e(str(g.get('week', '')))}</td>"
            f"<td>{e(names.get(g['away'], g['away']))} at {e(names.get(g['home'], g['home']))}</td>"
            f"<td>{e(names.get(g[g['pick']], g[g['pick']]))} {fmt_line(g['pick_line'])}</td><td class=mono>{_f(g['edge'])}</td>"
            f"<td class=mono>{_f(g['proj_away'])}–{_f(g['proj_home'])}</td>"
            + ("<td class=mono>—</td><td class='res-VOID'>VOID" + (f"<br><small>{e(str(g['void_reason']))}</small>"
               if isinstance(g.get('void_reason'), str) else "") + "</td>" if g.get('result') == "VOID" else
               f"<td class=mono>{g['away_score']:.0f}–{g['home_score']:.0f}</td><td class='res-{g['result']}'>{g['result']}</td>"
               if isinstance(g.get('result'), str) and g['result'] else "<td class=mono>—</td><td>pending</td>")
            + "</tr>" for g in sec["graded"])
        out.append(f'<h2>{e(sec["title"])}</h2><p class="note">{e(sec["note"])}</p><div class="tiles">{tiles}</div>'
                   + (f'<div class="tbl"><table><tr><th>Wk</th><th>Game</th><th>Pick</th><th>Edge</th><th>Projected</th>'
                      f'<th>Final</th><th>Result</th></tr>{rows}</table></div>' if rows else '<p class="note">No picks yet.</p>'))
    out.append('<p class="note">Break-even at -110 is 52.4%. Samples of a few hundred games are still mostly noise.</p>')
    return "".join(out)


def write(root: str | Path, rel: str, content: str):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p
