"""
Player-prop pages: one page per game (favourites + every DraftKings prop we can price) and one page per player
(his numbers this game, the simulated spread for each market, and his recent games).
"""
from __future__ import annotations

from html import escape as e

from .core.edges import fmt_line
from .props.board_props import LABEL, MODEL_WEIGHT, ORDER

CSS = """
.favs{border:2px solid var(--gold,#c9a227);border-radius:12px;padding:12px 14px;margin:14px 0;background:rgba(201,162,39,.07)}
.favs h2{margin-top:0}.fav{padding:8px 0;border-top:1px solid rgba(127,127,127,.2)}.fav:first-of-type{border-top:0}
.fav b.n{display:inline-block;width:1.4em;color:var(--gold,#c9a227)}
tr.favrow td{background:rgba(201,162,39,.12)}td.lean{font-weight:700}
.hist{width:100%;max-width:560px;height:auto;display:block;margin:6px 0 2px}
.pl-h{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
"""


def _hst(stamp):
    try:
        import pandas as pd
        return (pd.Timestamp(stamp.replace("Z", ""), tz="UTC") - pd.Timedelta(hours=10)).strftime("%a %-I:%M %p HST")
    except Exception:
        return str(stamp)


def _odds(x):
    if x is None:
        return "—"
    x = float(x)
    return f"+{x:.0f}" if x > 0 else f"{x:.0f}"


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def _f1(x):
    return "—" if x is None else f"{x:.1f}"


def _pick_txt(r):
    if r["market"] in ("anytime_td", "first_td"):
        return f'YES {_odds(r.get("price"))}'
    return f'{r["side"].upper()} {r["line"]:g} {_odds(r.get("price"))}'


def _res(r):
    if r.get("actual") is None:
        return ""
    v = r["actual"]
    val = ("yes" if v else "no") if r["market"] in ("anytime_td", "first_td") else f"{v:g}"
    res = r.get("result")
    cls = {"HIT": "WIN", "MISS": "LOSS", "PUSH": "PUSH"}.get(res, "")
    return f'{e(val)}' + (f' <span class="res-{cls}">{res}</span>' if res else "")


def _why(r, p):
    if r["market"] in ("anytime_td", "first_td"):
        return (f'Our chance {_pct(r["p_yes"])} against the {_pct(r.get("market_yes"))} DraftKings\' price implies '
                f'(margin included).')
    side_p = r["model_over"] if r["side"] == "over" else 1 - r["model_over"]
    mk = r["market_over"] if r["side"] == "over" else 1 - r["market_over"]
    return (f'Our projection is {r["mean"]:.1f} ({"above" if r["mean"] > r["line"] else "below"} the {r["line"]:g} line); '
            f'we have the {r["side"]} at {_pct(side_p)}, DraftKings\' price says {_pct(mk)} after removing its margin.')


def hist_svg(h: dict, line=None, mean=None) -> str:
    if not h or not h.get("counts"):
        return ""
    W, H, pad = 560, 120, 18
    ed, ct = h["edges"], h["counts"]
    lo, hi = ed[0], ed[-1]
    mx = max(ct) or 1
    x = lambda v: pad + (v - lo) / (hi - lo) * (W - 2 * pad)
    bars = "".join(
        f'<rect x="{x(ed[i]) + 0.5:.1f}" y="{H - 20 - (c / mx) * (H - 34):.1f}" width="{max(x(ed[i + 1]) - x(ed[i]) - 1, 1):.1f}" '
        f'height="{(c / mx) * (H - 34):.1f}" fill="{"var(--loss,#b3261e)" if line is not None and ed[i + 1] <= line else "var(--win,#1a7f4b)"}" opacity=".55"/>'
        for i, c in enumerate(ct))
    marks = ""
    if line is not None and lo <= line <= hi:
        marks += (f'<line x1="{x(line):.1f}" x2="{x(line):.1f}" y1="8" y2="{H - 20}" stroke="currentColor" stroke-width="2"/>'
                  f'<text x="{x(line) + 4:.1f}" y="16" font-size="11" fill="currentColor">line {line:g}</text>')
    if mean is not None and lo <= mean <= hi:
        marks += f'<line x1="{x(mean):.1f}" x2="{x(mean):.1f}" y1="8" y2="{H - 20}" stroke="currentColor" stroke-dasharray="4 3"/>'
    ticks = "".join(f'<text x="{x(v):.1f}" y="{H - 5}" font-size="10" text-anchor="middle" fill="currentColor">{v:g}</text>'
                    for v in [lo, (lo + hi) / 2, hi])
    return (f'<svg class="hist" viewBox="0 0 {W} {H}" role="img" aria-label="Simulated results">{bars}{marks}{ticks}</svg>'
            '<p class="note">Bars: our simulated results (red at or under the line, green over). Solid line: DraftKings\' line. '
            'Dashed: our average.</p>')


def props_page(sport: str, c: dict, gp: dict, names: dict, final: dict | None = None) -> str:
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    pl = gp["players"]
    rows = gp["rows"]
    favkeys = {(f["athlete_id"], f["market"], f["line"]) for f in gp["favorites"]}
    link = lambda aid: f'<a href="../player/{e(str(c["game_id"]))}_{e(aid)}.html">{e(pl[aid]["name"])}</a>'
    out = [f'<style>{CSS}</style><p><a class="more" href="../game/{e(str(c["game_id"]))}.html">Back to the game</a> · '
           f'<a class="more" href="../index.html">Board</a></p>',
           f'<h1>{e(a)} at {e(h)}: player props</h1>',
           f'<p class="eyebrow">{e(str(c.get("date", "")))} {e(str(c.get("kickoff") or ""))} · {len(pl)} players · {len(rows)} props · '
           f'lines and prices: {e(gp["source"])}{", updated " + e(_hst(gp["seen"])) if gp.get("seen") else ""}</p>']
    fin = bool(gp.get("final"))
    if fin:
        h_, n_ = gp.get("fav_tally", (0, 0))
        out.append(f'<p class="note" style="border-left:3px solid var(--edge);padding-left:10px"><b>Final.</b> Lines and prices are '
                   f'frozen at the last update before kickoff. Favorites: <b>{h_} of {n_} hit</b>. Results are shown for interest only '
                   'and are not part of any record.</p>')
    if any(r.get("assumed") for r in rows):
        out.append('<p class="note"><b>Prices:</b> ESPN\'s feed carries DraftKings\' lines for these props but not the prices, so '
                   'over and under are both treated as -110 (shown as "-110*"). The real price can differ.</p>')
    favs = [r for r in rows if (r["athlete_id"], r["market"], r["line"]) in favkeys]
    favs.sort(key=lambda r: -r["ev"])
    if favs:
        items = "".join(
            f'<div class="fav"><b class="n">{i}.</b> {link(r["athlete_id"])} <span class="eyebrow">{e(str(pl[r["athlete_id"]].get("team") or ""))}</span> '
            f'· <b>{e(_pick_txt(r))}</b> {e(LABEL[r["market"]].lower())} · value {r["ev"]:+.3f} per unit'
            + (f' · actual {_res(r)}' if r.get("actual") is not None else "") +
            f'<div class="note">{e(_why(r, pl[r["athlete_id"]]))}</div></div>' for i, r in enumerate(favs, 1))
        out.append(f'<div class="favs"><h2>Our favorite props</h2>{items}</div>')
    else:
        out.append('<div class="favs"><h2>Our favorite props</h2><p class="note">No prop clears +3% value at DraftKings\' price '
                   'right now.</p></div>')
    out.append(f'<p class="note">How to read this: "ours" is our simulation of the player\'s night. The chance we act on blends '
               f'ours with DraftKings\' own no-vig chance. Ours counts up to {100 * MODEL_WEIGHT:.0f}%, less early in the season '
               '(full weight after 15 NBA or 6 NFL games), because these player models are new and untested against closing prices. '
               'Value is the expected profit per unit at DraftKings\' price using that blended chance. When ours and the book\'s '
               'differ by more than 30 points, the book usually knows something (a role change, a teammate out): those are marked '
               '"check the news" and are never favorites. Props are never part of the record.</p>')
    for mk in [m for m in ORDER if any(r["market"] == m for r in rows)]:
        rs = [r for r in rows if r["market"] == mk]
        if mk in ("anytime_td", "first_td"):
            body = "".join(
                f'<tr class="{"favrow" if (r["athlete_id"], r["market"], r["line"]) in favkeys else ""}"><td>{link(r["athlete_id"])}</td>'
                f'<td>{e(str(pl[r["athlete_id"]].get("team") or ""))}</td><td class=mono>{_odds(r.get("price"))}</td>'
                f'<td class=mono>{_pct(r["p_yes"])}</td><td class=mono>{_pct(r.get("market_yes"))}</td>'
                f'<td class=mono>{"—" if r.get("ev") is None else format(r["ev"], "+.3f")}</td>'
                + (f'<td class=mono>{_res(r)}</td>' if fin else "") + '</tr>'
                for r in sorted(rs, key=lambda r: -r["p_yes"]))
            out.append(f'<h2>{e(LABEL[mk])}</h2><div class="tbl"><table><tr><th>Player</th><th>Team</th><th>DK price</th>'
                       f'<th>Ours</th><th>DK implies</th><th>Value</th>{"<th>Actual</th>" if fin else ""}</tr>{body}</table></div>')
            continue
        body = "".join(
            f'<tr class="{"favrow" if (r["athlete_id"], r["market"], r["line"]) in favkeys else ""}"><td>{link(r["athlete_id"])}</td>'
            f'<td>{e(str(pl[r["athlete_id"]].get("team") or ""))}</td><td class=mono>{r["line"]:g}</td>'
            f'<td class=mono>{_odds(r.get("over"))} / {_odds(r.get("under"))}{"*" if r.get("assumed") else ""}</td><td class=mono>{r["mean"]:.1f}</td>'
            f'<td class=mono>{_f1(r["p10"])}–{_f1(r["p90"])}</td><td class=mono>{_pct(r.get("model_over"))}</td>'
            f'<td class=mono>{_pct(r.get("market_over"))}</td>'
            f'<td class="lean">{e((r.get("side") or "").upper())}{"<br><small>check the news</small>" if r.get("big_gap") else ""}</td>'
            f'<td class=mono>{"—" if r.get("ev") is None else format(r["ev"], "+.3f")}</td>'
            + (f'<td class=mono>{_res(r)}</td>' if fin else "") + '</tr>' for r in rs)
        out.append(f'<h2>{e(LABEL[mk])}</h2><div class="tbl"><table><tr><th>Player</th><th>Team</th><th>Line</th>'
                   '<th>Over / under</th><th>Ours</th><th>80% range</th><th>Over: ours</th><th>Over: DK</th><th>Lean</th>'
                   f'<th>Value</th>{"<th>Actual</th>" if fin else ""}</tr>{body}</table></div>')
    return "".join(out)


def player_page(sport: str, c: dict, gp: dict, aid: str, names: dict) -> str:
    p = gp["players"][aid]
    a, h = names.get(c["away"], c["away"]), names.get(c["home"], c["home"])
    rows = [r for r in gp["rows"] if r["athlete_id"] == aid]
    out = [f'<style>{CSS}</style><p><a class="more" href="../props/{e(str(c["game_id"]))}.html">Back to the props</a></p>',
           f'<div class="pl-h"><h1>{e(p["name"])}</h1><span class="eyebrow">{e(str(p.get("pos") or ""))} · {e(str(p.get("team") or ""))}'
           f' · {e(a)} at {e(h)}, {e(str(c.get("date", "")))}</span></div>']
    if p.get("status"):
        out.append(f'<p class="note" style="border-left:3px solid var(--loss);padding-left:10px">Injury report: {e(p["status"])}.</p>')
    pj = p.get("proj") or {}
    if sport == "nba":
        r36 = pj.get("per36") or {}
        out.append('<div class="tiles">'
                   f'<div class="tile"><span class="k">Projected minutes</span><span class="v">{_f1(pj.get("min"))}</span>'
                   f'<span class="s">{pj.get("games", 0)} games this season</span></div>'
                   f'<div class="tile"><span class="k">Per 36: pts / reb / ast</span><span class="v">{_f1(r36.get("points"))} / '
                   f'{_f1(r36.get("rebounds"))} / {_f1(r36.get("assists"))}</span><span class="s">this season, shrunk toward last</span></div>'
                   f'<div class="tile"><span class="k">Game pace</span><span class="v">{100 * (pj.get("pace") or 1):.0f}</span>'
                   '<span class="s">vs league 100</span></div></div>')
    else:
        out.append('<div class="tiles">'
                   f'<div class="tile"><span class="k">Projected targets</span><span class="v">{_f1(pj.get("targets"))}</span></div>'
                   f'<div class="tile"><span class="k">Projected carries</span><span class="v">{_f1(pj.get("carries"))}</span></div>'
                   f'<div class="tile"><span class="k">Share of team TDs</span><span class="v">{_pct(pj.get("td_share"))}</span></div></div>')
    for r in sorted(rows, key=lambda r: ORDER.index(r["market"]) if r["market"] in ORDER else 99):
        mk = r["market"]
        if mk in ("anytime_td", "first_td"):
            out.append(f'<h2>{e(LABEL[mk])}</h2><p>Ours {_pct(r["p_yes"])} · DraftKings {_odds(r.get("price"))} '
                       f'(implies {_pct(r.get("market_yes"))}) · value {"—" if r.get("ev") is None else format(r["ev"], "+.3f")}'
                       + (f' · actual {_res(r)}' if r.get("actual") is not None else "") + '</p>')
            continue
        out.append(f'<h2>{e(LABEL[mk])}: line {r["line"]:g}</h2>'
                   f'<p>Ours {r["mean"]:.1f} (middle 80%: {r["p10"]:.0f}–{r["p90"]:.0f}) · over {_pct(r.get("model_over"))} by us, '
                   f'{_pct(r.get("market_over"))} by DraftKings ({_odds(r.get("over"))} / {_odds(r.get("under"))})'
                   + (f' · lean <b>{e(r["side"].upper())}</b>, value {r["ev"]:+.3f}' if r.get("side") else "")
                   + (f' · actual {_res(r)}' if r.get("actual") is not None else "") + '</p>'
                   + hist_svg((p.get("hist") or {}).get(mk), r["line"], r["mean"]))
    log = p.get("log") or []
    if sport == "nba":
        def avg_row(lab, s):
            return "" if not s else (f'<tr><td><b>{lab}</b></td><td class=mono>{s["g"]} g</td><td class=mono>{s["min"]:.1f}</td>'
                                     f'<td class=mono>{s["pts"]:.1f}</td><td class=mono>{s["reb"]:.1f}</td><td class=mono>{s["ast"]:.1f}</td>'
                                     f'<td class=mono>{s["threes"]:.1f}</td></tr>')
        body = "".join(f'<tr><td>{e(g["date"])}</td><td>{"vs" if g["home"] else "at"} {e(str(g["opp"]))}</td><td class=mono>{g["min"]:.0f}</td>'
                       f'<td class=mono>{g["pts"]:.0f}</td><td class=mono>{g["reb"]:.0f}</td><td class=mono>{g["ast"]:.0f}</td>'
                       f'<td class=mono>{g["threes"]:.0f}</td></tr>' for g in log)
        out.append('<h2>Recent games</h2><div class="tbl"><table><tr><th>Date</th><th>Opp</th><th>Min</th><th>Pts</th><th>Reb</th>'
                   f'<th>Ast</th><th>3PM</th></tr>{avg_row("Last 5", p.get("last5"))}{avg_row("This season", p.get("season"))}'
                   f'{avg_row("Last season", p.get("last_season"))}{body}</table></div>'
                   '<p class="note">ESPN box scores (via sportsdataverse).</p>')
    else:
        body = "".join(f'<tr><td class=mono>{g["week"]}</td><td>{e(str(g["opp"]))}</td><td class=mono>{g["tgt"]}</td><td class=mono>{g["rec"]}</td>'
                       f'<td class=mono>{g["rec_yds"]:.0f}</td><td class=mono>{g["car"]}</td><td class=mono>{g["rush_yds"]:.0f}</td>'
                       f'<td class=mono>{g["td"]}</td></tr>' for g in log)
        out.append('<h2>Games this season</h2><div class="tbl"><table><tr><th>Wk</th><th>Opp</th><th>Tgt</th><th>Rec</th><th>Rec yds</th>'
                   f'<th>Car</th><th>Rush yds</th><th>TD</th></tr>{body or "<tr><td>No games yet</td></tr>"}</table></div>'
                   '<p class="note">nflverse play-by-play.</p>')
    return "".join(out)
