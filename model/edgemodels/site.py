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
  /* Layout: a scoreboard ledger. Narrow reading column, game rows as ruled tiles, numbers in mono. */
  --bg:#eef1f4; --panel:#ffffff; --ink:#121820; --muted:#5b6672; --rule:#d5dbe2;
  --model:#1d4fd1; --market:#6b7480; --edge:#b86a00; --win:#1f7a4a; --loss:#b3352b;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f1318; --panel:#171c23; --ink:#e7ebf0; --muted:#94a0ad; --rule:#2a323c;
  --model:#7ea2ff; --market:#9aa4b0; --edge:#f0a53a; --win:#4fc283; --loss:#f07a6f; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f1318; --panel:#171c23; --ink:#e7ebf0; --muted:#94a0ad; --rule:#2a323c;
  --model:#7ea2ff; --market:#9aa4b0; --edge:#f0a53a; --win:#4fc283; --loss:#f07a6f; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.5;margin:0}
.wrap{max-width:1080px;margin:0 auto;padding-inline:16px;padding-block:0 48px}
a{color:var(--model)} a:focus-visible{outline:2px solid var(--model);outline-offset:2px}
header.top{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:space-between;padding-block:18px;border-bottom:2px solid var(--ink)}
.brand{font-family:var(--display);font-weight:700;font-size:26px;letter-spacing:.02em;text-transform:uppercase;color:var(--ink);text-decoration:none}
.brand b{color:var(--model)}
nav.sports{display:flex;flex-wrap:wrap;gap:6px}
nav.sports a{font-family:var(--display);font-weight:600;font-size:16px;letter-spacing:.06em;text-decoration:none;color:var(--ink);padding:4px 10px;border:1px solid var(--rule);border-radius:3px}
nav.sports a.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
nav.weeks{margin-top:18px}
nav.weeks a{font-family:var(--mono);font-size:12px;font-weight:500;letter-spacing:.04em}
h1{font-family:var(--display);font-weight:700;font-size:clamp(34px,6vw,54px);line-height:1;margin:28px 0 10px;text-wrap:balance;text-transform:uppercase}
h2{font-family:var(--display);font-weight:600;font-size:22px;letter-spacing:.04em;text-transform:uppercase;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--rule)}
.lede{max-width:66ch;color:var(--muted)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:18px 0}
.tile{background:var(--panel);border:1px solid var(--rule);padding:12px 14px;text-decoration:none;color:var(--ink);display:flex;flex-direction:column;gap:2px}
.tile .k{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-family:var(--display);font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}
.tile .s{font-size:13px;color:var(--muted)}
.day{margin-top:26px}
.board{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:10px}
.game{background:var(--panel);border:1px solid var(--rule);padding:12px 14px;display:flex;flex-direction:column;gap:8px;min-width:0}
.game.hot{border-color:var(--edge);box-shadow:inset 3px 0 0 var(--edge)}
.meta{font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.teamrow{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:baseline}
.teamrow .t{font-weight:600}
.teamrow .t small{color:var(--muted);font-weight:400;margin-left:6px;font-family:var(--mono);font-size:12px}
.teamrow .sc{font-family:var(--display);font-weight:700;font-size:26px;font-variant-numeric:tabular-nums}
dl.lines{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:4px 0 0;font-size:13px}
dl.lines dt{color:var(--muted)} dl.lines dd{margin:0;font-family:var(--mono);font-variant-numeric:tabular-nums}
.pick{color:var(--edge);font-weight:600}
.more{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;text-decoration:none}
.numbers{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
@media (max-width:560px){.numbers{grid-template-columns:1fr}}
.scorebig{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;background:var(--panel);border:1px solid var(--rule);padding:16px}
.scorebig .n{font-family:var(--display);font-size:clamp(40px,9vw,64px);font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.scorebig .r{text-align:right}
.bars{display:flex;flex-direction:column;gap:6px;background:var(--panel);border:1px solid var(--rule);padding:14px}
.bar{display:grid;grid-template-columns:minmax(110px,170px) 1fr 52px;gap:10px;align-items:center;font-size:13px}
.track{position:relative;height:14px;background:linear-gradient(var(--rule),var(--rule)) center/1px 100% no-repeat}
.fill{position:absolute;top:2px;height:10px;background:var(--model)}
.fill.away{background:var(--market)}
.bar .val{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--muted);padding-inline:calc(min(170px,30%) + 10px) 62px}
.note{font-size:13px;color:var(--muted);max-width:70ch}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}
th:first-child,td:first-child{text-align:left}
td.mono{font-family:var(--mono)}
.pct{display:inline-grid;grid-template-columns:60px 28px;gap:6px;align-items:center}
.pct i{display:block;height:6px;background:var(--rule);position:relative}
.pct i b{position:absolute;inset:0 auto 0 0;background:var(--model)}
.res-VOID{color:var(--muted);font-weight:600}.res-WIN{color:var(--win);font-weight:600}.res-LOSS{color:var(--loss);font-weight:600}.res-PUSH{color:var(--muted)}
.hist{display:inline-flex;gap:1px;align-items:flex-end;height:22px;vertical-align:middle}
.hist span{width:5px;background:var(--model);opacity:.75}
.flag{font-family:var(--mono);font-size:11px;padding:1px 6px;border:1px solid var(--edge);color:var(--edge);border-radius:2px}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--rule);font-size:12px;color:var(--muted)}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700'
         '&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap">')

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
    return (f"{head}<title>{e(title)}</title>{FONTS}<style>{CSS}</style>{'</head><body>' if standalone else ''}<div class=\"wrap\">"
            f"<header class=\"top\"><a class=\"brand\" href=\"{up}index.html\"><b>{e(first)}</b> {e(rest[0] if rest else '')}</a>"
            f"<nav class=\"sports\">{nav}</nav></header>{body}"
            "<footer>Projections are model output, not betting advice. Every number on these pages is written by the "
            f"same code that produced the score; nothing is entered by hand.</footer></div>{tail}")


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
           f'<dt>Moneyline</dt><dd>{e(h)} {_ml(c.get("home_ml"))} / {e(a)} {_ml(c.get("away_ml"))}</dd>'
           f'<dt>Model</dt><dd>{e(h)} {100 * c["p_home"]:.1f}% · fair {_ml(c["fair_home_ml"])} / {_ml(c["fair_away_ml"])}</dd>')
    if c.get("ml_pick"):
        hot = c["ml_edge"] >= EDGE_TIERS["mlb"][0] and c["ml_ev"] > 0
        team = names.get(c[c["ml_pick"]], c[c["ml_pick"]])
        out += (f'<dt>Model pick</dt><dd class="{"pick" if hot else ""}">{e(team)} {_ml(c["ml_odds"])} '
                f'<span class="eyebrow">({c["ml_edge"]:.1f} pts · EV {c["ml_ev"]:+.3f})</span></dd>')
    out += f'<dt>Total</dt><dd>{_f(c.get("model_total"))} model · {_f(c.get("market_total"))} market · not picked</dd>'
    if c.get("final_home") is not None:
        res = " ".join(f'<span class="res-{r}">{m} {r}</span>' for m, r in
                       ((c.get("result_label", "ML"), c.get("ml_result")), ("Total", c.get("tot_result"))) if r)
        out += f'<dt>Final</dt><dd>{c["final_away"]:.0f}–{c["final_home"]:.0f} {res}</dd>'
    return out


def board(sport: str, title: str, lede: str, cards: list[dict], names: dict, records: dict | None = None,
          stats: list[dict] | None = None, nav: str = "", game_prefix: str = "game/", links: bool = True,
          newest_first: bool = False) -> str:
    """newest_first: for boards of games already played (replays, past weeks) the latest day and latest
    game come first; live boards of upcoming games stay in schedule order."""
    tier1 = EDGE_TIERS[sport][0]
    tiles = "".join(f'<a class="tile" href="{s.get("href", "#board")}"><span class="k">{e(s["k"])}</span>'
                    f'<span class="v">{e(s["v"])}</span><span class="s">{e(s.get("s", ""))}</span></a>' for s in (stats or []))
    days: dict[str, list] = {}
    for c in cards:
        days.setdefault(c.get("date", ""), []).append(c)
    out = [f'{nav}<h1>{e(title)}</h1><p class="lede">{e(lede)}</p><div class="tiles">{tiles}</div><div id="board"></div>']
    items = list(days.items())                 # cards arrive in schedule order
    if newest_first:
        items = [(d, cs[::-1]) for d, cs in items[::-1]]
    for d, cs in items:
        out.append(f'<section class="day"><h2>{e(d)} <span class="eyebrow">{len(cs)} game{"s" if len(cs) != 1 else ""}</span></h2><div class="board">')
        for c in cs:
            hot = (c.get("edge") is not None and c["edge"] >= tier1 and not c.get("void") and not c.get("hold")) if sport != "mlb" else \
                (c.get("ml_edge") is not None and c["ml_edge"] >= tier1 and c.get("ml_ev", 0) > 0)
            rec = (records or {})
            out.append(
                f'<article class="game{" hot" if hot else ""}"><div class="meta">{e(str(c.get("kickoff") or ""))}'
                f'{" · " + e(c["venue"]) if c.get("venue") else ""}</div>'
                f'<div class="teamrow"><span class="t">{e(names.get(c["away"], c["away"]))}<small>{e(rec.get(c["away"], ""))}</small></span>'
                f'<span class="sc">{_f(c["proj_away"])}</span></div>'
                f'<div class="teamrow"><span class="t">{e(names.get(c["home"], c["home"]))}<small>{e(rec.get(c["home"], ""))}</small></span>'
                f'<span class="sc">{_f(c["proj_home"])}</span></div>'
                + (f'<dl class="lines">{_mlb_lines(c, names)}</dl>' if sport == "mlb" else
                   f'<dl class="lines"><dt>Vegas line</dt><dd>{_fav_text(c.get("market_line"), c, names)}</dd>'
                   f'<dt>Model line</dt><dd>{_fav_text(c.get("model_line"), c, names)}</dd>'
                   f'<dt>Model pick</dt><dd class="{"pick" if hot else ""}">{_pick_text(c, names)}</dd>'
                   f'<dt>Total</dt><dd>{_f(c.get("model_total"))} model · {_f(c.get("market_total"))} market</dd>'
                   f'{_result_line(c, names)}</dl>')
                + (f'<a class="more" href="{game_prefix}{e(str(c["game_id"]))}.html">Game breakdown →</a>' if links else "")
                + '</article>')
        out.append("</div></section>")
    return "".join(out)


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
    return (f'<div class="bars">{"".join(rows)}<div class="axis"><span>← {e(names.get(c["away"], c["away"]))}</span>'
            f'<span>{e(names.get(c["home"], c["home"]))} →</span></div></div>'
            f'<p class="note">Each bar is that input\'s contribution to the projected margin, in {UNIT[c["sport"]]}, pointing toward the '
            f'team it favours. They sum to {abs(total):.1f} toward {e(names.get(fav, fav))}.</p>')


def _pct(v):
    if v is None:
        return "—"
    return f'<span class="pct"><i><b style="width:{v}%"></b></i><span>{v}</span></span>'


def _ratings(c, names):
    r = c.get("ratings")
    if not r:
        return ""
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    head = f"<tr><th>Percentile, 0–100</th><th>{e(a)} off</th><th>{e(h)} off</th><th>{e(a)} def</th><th>{e(h)} def</th></tr>"
    rows = "".join(f"<tr><td>{e(x['short'])}</td><td>{_pct(x['away_off_pct'])}</td><td>{_pct(x['home_off_pct'])}</td>"
                   f"<td>{_pct(x['away_def_pct'])}</td><td>{_pct(x['home_def_pct'])}</td></tr>" for x in r["rows"])
    nd = 3 if c["sport"] in ("nfl", "ncaaf", "mlb") or True else 1
    def fv(x):
        return "—" if x is None else (f"{x:+.3f}" if abs(x) < 2 else f"{x:+.1f}")
    head2 = f"<tr><th>Per play / possession, vs average</th><th>{e(a)} off</th><th>{e(h)} off</th><th>{e(a)} def</th><th>{e(h)} def</th></tr>"
    rows2 = "".join(f"<tr><td>{e(x['label'])}</td><td class=mono>{fv(x['away_off'])}</td><td class=mono>{fv(x['home_off'])}</td>"
                    f"<td class=mono>{fv(x['away_def'])}</td><td class=mono>{fv(x['home_def'])}</td></tr>" for x in r["rows"])
    return (f'<h2>How they rate <span class="eyebrow">percentile vs the league</span></h2><div class="tbl"><table>{head}{rows}</table></div>'
            f'<h2>Team ratings</h2><div class="tbl"><table>{head2}{rows2}</table></div>'
            f'<p class="note">Opponent adjusted and shrunk toward last season ({e(r.get("through", ""))}). On defence, a negative number is good '
            f'for efficiency metrics; the percentile already accounts for direction. '
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
    return (f'<h2>Player props <span class="eyebrow">Monte Carlo, {pr.get("n_sims", 20000):,} sims</span></h2>{tbl}'
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
        out += (f'<h2>Inputs used</h2>' + (f'<div class="tbl"><table>{head}{rows}</table></div>' if rows else "")
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
        out += ('<h2>Pick history</h2><div class="tbl"><table><tr><th>Time (HST)</th><th>Event</th><th>Pick</th><th>Model line</th>'
                f'<th>Market line</th><th>Starters / reason</th></tr>{rows}</table></div><p class="note">Picks update until kickoff '
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
        num = (f'<p class="lede">The model makes this {_fav_text(c["model_line"], c, names)}. The market has it '
               f'{_fav_text(c["market_line"], c, names)}. The model rates {e(pts_side)} {_f(c["edge"])} {unit} '
               f'better than the market does.')
        if c[c["pick"]] != pts_team and c.get("cover_prob") is not None:
            num += (f' The pick is still {e(side)} {fmt_line(c["pick_line"])}: final margins land on key numbers '
                    f'(3, 7) far more often than others, so at this line {e(side)} covers more often '
                    f'({100 * c["cover_prob"]:.1f}% vs {100 * c["lose_prob"]:.1f}%).')
        if c.get("ev") is not None and c["ev"] <= 0:
            num += ' At -110 neither side is worth a bet (expected value below zero).'
        num += '</p>'
    else:
        num = f'<p class="lede">The model makes this {_fav_text(c["model_line"], c, names)}. No market line was available when this was written.</p>'
    if c["sport"] == "mlb":
        num, tiles = _mlb_number(c, names)
    cp = c.get("cover_prob")
    tiles = tiles if c["sport"] == "mlb" else (f'<div class="numbers"><div class="tile"><span class="k">Vegas line</span><span class="v">{_fav_text(c.get("market_line"), c, names)}</span></div>'
             f'<div class="tile"><span class="k">Model</span><span class="v">{_fav_text(c["model_line"], c, names)}</span></div>'
             f'<div class="tile"><span class="k">Difference</span><span class="v">{_f(c.get("edge"))}</span>'
             f'<span class="s">{"P(pick covers) " + format(cp, ".0%") if cp else ""}</span></div></div>')
    if c.get("push_prob") is not None:
        side = names.get(c[c["pick"]], c[c["pick"]])
        tiles += (f'<div class="tbl" style="margin-top:10px"><table><tr><th>At {e(side)} {fmt_line(c["pick_line"])}</th>'
                  f'<th>Covers</th><th>Push</th><th>Loses</th><th>EV per unit at -110</th><th>Fair line</th></tr>'
                  f'<tr><td>Key-number pricing</td><td class=mono>{100 * c["cover_prob"]:.1f}%</td>'
                  f'<td class=mono>{100 * c["push_prob"]:.1f}%</td><td class=mono>{100 * c["lose_prob"]:.1f}%</td>'
                  f'<td class=mono>{c["ev"]:+.3f}</td><td class=mono>{_fav_text(c.get("fair_line"), c, names)}</td></tr></table></div>'
                  '<p class="note">Probabilities come from how often real games finish on each margin (3 and 7 far more than a '
                  'bell curve allows), centred on the model\'s projected margin. A positive EV means the side is worth more than '
                  'the -110 price; a negative EV on both sides means no play. Backtests show these model-only probabilities run '
                  'high: the market line carries information the model does not.</p>')
    tot = c.get("total") or {}
    if c["sport"] == "mlb":
        tot = {}
    total_txt = (f'<p class="note">Total: model {_f(c["model_total"])}, market {_f(c.get("market_total"))}'
                 f'{", lean " + tot["side"] + (" (" + format(tot["prob"], ".0%") + ")" if "prob" in tot else "") if tot else ""}.</p>')
    qb = ""
    if c.get("qb"):
        rows = "".join(f"<tr><td>{e(str(q['starter']))}</td><td>{e(names.get(q['team'], q['team']))}</td>"
                       f"<td class=mono>{q['adj']:+.3f}</td></tr>" for q in c["qb"])
        qb = ('<h2>Quarterback</h2><div class="tbl"><table><tr><th>Starter</th><th>Team</th><th>Adjustment, EPA / dropback</th></tr>'
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
    return (f'<p><a class="more" href="../index.html">← Back to the board</a></p><h1>{e(a)} at {e(h)}</h1>'
            f'<p class="eyebrow">{e(str(c.get("date", "")))} {e(str(c.get("kickoff") or ""))} {("· " + e(c["venue"])) if c.get("venue") else ""}</p>'
            + (f'<p class="note" style="border-left:3px solid var(--edge);padding-left:10px">{e(c["note"])}</p>' if c.get("note") else "")
            + (f'<p class="note" style="border-left:3px solid var(--loss);padding-left:10px"><b>HOLD.</b> {e(c["hold"])}</p>' if c.get("hold") else "")
            + (f'<p class="note" style="border-left:3px solid var(--edge);padding-left:10px"><b>NO EDGE.</b> {e(c["no_edge"])}</p>' if c.get("no_edge") else "")
            + f'<h2>The number</h2>{num}{tiles}'
            f'<h2>Projected score</h2><div class="scorebig"><div><div class="eyebrow">{e(a)}</div><div class="n">{_f(c["proj_away"])}</div></div>'
            f'<div class="eyebrow">at</div><div class="r"><div class="eyebrow">{e(h)}</div><div class="n">{_f(c["proj_home"])}</div></div></div>{total_txt}'
            f'<h2>Where the margin comes from</h2>{_bars(c, names)}{qb}{extra}{_inputs(c, names)}{_ratings(c, names)}{_props(c)}')


def record_page(sport: str, title: str, lede: str, sections: list[dict], names: dict) -> str:
    """sections: [{"title", "note", "summary", "graded"}] shown in order, each with its own tiles and list."""
    out = [f'<p><a class="more" href="index.html">← Back to the board</a></p><h1>{e(title)}</h1><p class="lede">{e(lede)}</p>']
    for sec in sections:
        tiles = "".join(f'<div class="tile"><span class="k">{e(s["label"])}</span><span class="v">{s["W"]}-{s["L"]}-{s["P"]}</span>'
                        f'<span class="s">{_f(100 * s["pct"]) if s["W"] + s["L"] else "—"}% · {s["units"]:+.1f} units</span></div>'
                        for s in sec["summary"])
        rows = "".join(
            f"<tr><td class=mono>{e(str(g.get('week', '')))}</td>"
            f"<td>{e(names.get(g['away'], g['away']))} at {e(names.get(g['home'], g['home']))}</td>"
            f"<td>{e(names.get(g[g['pick']], g[g['pick']]))} {fmt_line(g['pick_line'])}</td><td class=mono>{_f(g['edge'])}</td>"
            f"<td class=mono>{_f(g['proj_away'])}–{_f(g['proj_home'])}</td>"
            + ("<td class=mono>—</td><td class='res-VOID'>VOID</td>" if g.get('result') == "VOID" else
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
