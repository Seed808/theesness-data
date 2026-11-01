"""Save a full copy of Sam's public site (every same-site page/file linked from the home page), for offline study.
Usage: python scripts/crawl_sam.py OUTDIR [max_pages]"""
import re, sys, time, json
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
BASE = "https://samthemodelman.smmodel.workers.dev/"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
host = urlparse(BASE).netloc
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0 (personal research snapshot)"
seen, q, log = set(), deque([BASE]), []
# seed every sport section explicitly
for lg in ("ncaaf", "nfl", "nba", "cbb", "mlb", "nhl", "members", "props", "record"):
    q.append(urljoin(BASE, lg + "/"))
t0 = time.time()
while q and len(seen) < cap and time.time() - t0 < 50 * 60:
    u = q.popleft().split("#")[0]
    if u in seen or urlparse(u).netloc != host:
        continue
    seen.add(u)
    try:
        r = s.get(u, timeout=30)
    except Exception as ex:
        log.append([u, "ERR", str(ex)[:80]]); continue
    log.append([u, r.status_code, len(r.content)])
    if not r.ok:
        continue
    p = urlparse(u).path
    if p.endswith("/") or p == "":
        p += "index.html"
    f = out / p.lstrip("/")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(r.content)
    ct = r.headers.get("content-type", "")
    if "html" in ct or "javascript" in ct or "json" in ct or p.endswith((".html", ".js", ".json")):
        txt = r.text
        for m in re.findall(r'(?:href|src|action)=["\']([^"\'#]+)', txt) + re.findall(r'["\'](/[A-Za-z0-9_\-/\.]+\.(?:html|json|js|csv))["\']', txt) \
                + re.findall(r'fetch\(\s*["\']([^"\']+)', txt):
            v = urljoin(u, m)
            if urlparse(v).netloc == host and v not in seen:
                q.append(v)
    time.sleep(0.15)
(out / "_crawl_log.json").write_text(json.dumps(log, indent=0))
print(len(seen), "urls,", sum(1 for x in log if x[1] == 200), "saved,", int(time.time() - t0), "s")
