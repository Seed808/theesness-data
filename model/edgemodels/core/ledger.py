"""
Lock-then-grade ledger.

A projection is written once per game, against the first market line the
model sees, and is never re-priced. Each lock row carries a SHA-256 of its
own contents plus the previous row's hash, so any later edit to the file is
detectable (verify_chain). Grading only fills in the result columns.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOCK_COLS = ["sport", "game_id", "date", "away", "home", "proj_away", "proj_home",
             "model_line", "market_line", "pick", "pick_line", "edge", "tier",
             "model_total", "market_total", "cover_prob", "push_prob", "ev", "locked_at", "prev_hash", "hash"]
GRADE_COLS = ["away_score", "home_score", "result", "graded_at"]
# kinds of void entry that replace a lock with a newer version before kickoff (not shown as VOID)
SUPERSEDED = {"correction", "update", "hold"}


def _norm(v) -> str:
    """Canonical text for hashing so a CSV round-trip hashes identically."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    try:
        f = float(v)
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


def _hash(row: dict, prev: str, cols=None) -> str:
    payload = json.dumps({k: _norm(row.get(k)) for k in (cols or LOCK_COLS) if k not in ("hash", "prev_hash")},
                         sort_keys=True)
    return hashlib.sha256((prev + payload).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | Path, extra_cols: list[str] | None = None):
        """extra_cols: more locked (hashed) fields for a sport, e.g. MLB market and odds.
        The NFL file uses none, so its existing hashes are unchanged."""
        self.extra = list(extra_cols or [])
        self.cols = LOCK_COLS[:-3] + self.extra + LOCK_COLS[-3:]
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_csv(self.path, dtype={"game_id": str})
        return pd.DataFrame(columns=self.cols + GRADE_COLS)

    def lock(self, cards: list[dict]) -> int:
        """Append cards whose game_id is not locked yet. Returns number newly locked."""
        led = self.load()
        v = self.voids()
        corrected = set(v.loc[v["kind"].isin(SUPERSEDED), "lock_hash"]) if len(v) else set()
        locked = set(led.loc[~led["hash"].isin(corrected), "game_id"].astype(str))
        prev = led["hash"].iloc[-1] if len(led) else ""
        new = []
        for c in cards:
            if str(c["game_id"]) in locked or c.get("market_line") is None:
                continue
            row = {k: c.get(k) for k in self.cols}
            row["locked_at"] = c.get("locked_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
            row["prev_hash"] = prev
            row["hash"] = _hash(row, prev, self.cols)
            prev = row["hash"]
            new.append(row)
            locked.add(str(c["game_id"]))
        if new:
            out = pd.concat([led, pd.DataFrame(new)], ignore_index=True)
            out.to_csv(self.path, index=False)
        return len(new)

    # ---- voids: a separate, hash-chained list. The lock row itself is never edited.
    @property
    def void_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_voids.csv")

    def voids(self) -> pd.DataFrame:
        if self.void_path.exists():
            v = pd.read_csv(self.void_path, dtype={"game_id": str})
            if "kind" not in v:
                v["kind"] = "void"
            v["kind"] = v["kind"].fillna("void")
            return v
        return pd.DataFrame(columns=["game_id", "lock_hash", "voided_at", "kickoff_utc", "reason", "kind", "prev_hash", "hash"])

    def void(self, game_id: str, reason: str, kickoff_utc: str, kind: str = "void") -> bool:
        """Void a locked pick. Allowed only before kickoff; the void cites the lock row's hash."""
        now = datetime.now(timezone.utc)
        if pd.Timestamp(kickoff_utc) <= pd.Timestamp(now):
            raise ValueError(f"{game_id}: kickoff {kickoff_utc} has passed; a void must be recorded before kickoff")
        led = self.load()
        row = led[led["game_id"].astype(str) == str(game_id)]
        if row.empty:
            raise ValueError(f"{game_id} is not locked")
        v = self.voids()
        live = row[~row["hash"].isin(set(v["lock_hash"]))]
        if live.empty:
            return False
        prev = v["hash"].iloc[-1] if len(v) else ""
        rec = {"game_id": str(game_id), "lock_hash": live["hash"].iloc[-1], "voided_at": now.isoformat(timespec="seconds"),
               "kickoff_utc": str(kickoff_utc), "reason": reason, "kind": kind, "prev_hash": prev}
        rec["hash"] = hashlib.sha256((prev + json.dumps({k: str(rec[k]) for k in sorted(rec) if k != "prev_hash"},
                                                        sort_keys=True)).encode()).hexdigest()
        pd.concat([v, pd.DataFrame([rec])], ignore_index=True).to_csv(self.void_path, index=False)
        return True

    def void_after_game(self, game_id: str, reason: str, kickoff_utc: str) -> bool:
        """Void a pick AFTER kickoff, only for a proven error in the model or data on that pick (never because it lost).
        The entry is kept in the same hash-chained void list, its time shows it was made after the game, and the
        reason is printed on the card and the record page. The lock row and its grade are never edited."""
        now = datetime.now(timezone.utc)
        led = self.load()
        row = led[led["game_id"].astype(str) == str(game_id)]
        if row.empty:
            raise ValueError(f"{game_id} is not locked")
        v = self.voids()
        live = row[~row["hash"].isin(set(v["lock_hash"]))]
        if live.empty:
            return False
        prev = v["hash"].iloc[-1] if len(v) else ""
        rec = {"game_id": str(game_id), "lock_hash": live["hash"].iloc[-1], "voided_at": now.isoformat(timespec="seconds"),
               "kickoff_utc": str(kickoff_utc), "reason": f"Voided after the game: {reason}", "kind": "void", "prev_hash": prev}
        rec["hash"] = hashlib.sha256((prev + json.dumps({k: str(rec[k]) for k in sorted(rec) if k != "prev_hash"},
                                                        sort_keys=True)).encode()).hexdigest()
        pd.concat([v, pd.DataFrame([rec])], ignore_index=True).to_csv(self.void_path, index=False)
        return True

    def load_graded(self) -> pd.DataFrame:
        """Lock rows with voided picks marked result VOID (excluded from every record)."""
        led = self.load()
        v = self.voids()
        if len(v):
            led["result"] = led["result"].astype(object)
            led = led[~led["hash"].isin(set(v.loc[v["kind"].isin(SUPERSEDED), "lock_hash"]))].copy()
            vv = v[v["kind"] == "void"]
            led.loc[led["hash"].isin(set(vv["lock_hash"])), "result"] = "VOID"
            led["void_reason"] = led["hash"].map(dict(zip(vv["lock_hash"], vv["reason"])))
        return led

    def grade(self, finals: pd.DataFrame) -> pd.DataFrame:
        """finals: game_id, away_score, home_score. Grades ATS vs the LOCKED line only."""
        led = self.load()
        voided = set(self.voids()["lock_hash"])
        for col in ("result", "graded_at"):
            led[col] = led[col].astype(object) if col in led else None
        f = finals.dropna(subset=["home_score", "away_score"])
        f = f.set_index(f["game_id"].astype(str))
        for i, r in led.iterrows():
            gid = str(r["game_id"])
            if (isinstance(r.get("result"), str) and r["result"]) or r["hash"] in voided:
                continue
            gid_f = gid.split("-")[0]
            if gid_f not in f.index:
                continue
            hs, as_ = float(f.at[gid_f, "home_score"]), float(f.at[gid_f, "away_score"])
            if r["pick"] in ("over", "under"):
                cover = (hs + as_ - float(r["pick_line"])) * (1 if r["pick"] == "over" else -1)
            else:
                margin = (hs - as_) if r["pick"] == "home" else (as_ - hs)
                cover = margin + float(r["pick_line"])
            led.at[i, "home_score"], led.at[i, "away_score"] = hs, as_
            led.at[i, "result"] = "WIN" if cover > 0 else ("LOSS" if cover < 0 else "PUSH")
            led.at[i, "graded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        led.to_csv(self.path, index=False)
        return led

    def verify_chain(self) -> bool:
        led = self.load()
        prev = ""
        for _, r in led.iterrows():
            row = r.to_dict()
            if row["prev_hash"] != prev and not (pd.isna(row["prev_hash"]) and prev == ""):
                return False
            if _hash(row, prev, self.cols) != row["hash"]:
                return False
            prev = row["hash"]
        return True


def record_summary(graded: pd.DataFrame, tiers=(3, 6, 10), juice: float = -110) -> list[dict]:
    """Win-loss-push and units at -110 for all picks and each edge tier."""
    g = graded[graded["result"].isin(["WIN", "LOSS", "PUSH"])]
    win_pay = 100 / -juice if juice < 0 else juice / 100
    out = []
    for label, sub in [("All picks", g)] + [(f"Edge {t}+", g[g["edge"] >= t]) for t in tiers]:
        w, l, p = (sub["result"] == "WIN").sum(), (sub["result"] == "LOSS").sum(), (sub["result"] == "PUSH").sum()
        pct = w / (w + l) if (w + l) else float("nan")
        if "odds" in sub and sub["odds"].notna().any():
            # priced markets (MLB moneyline / totals): pay each win at the odds it was locked at
            o = sub["odds"].astype(float).fillna(juice)
            pay = o.where(o > 0, 0) / 100 + (100 / -o).where(o < 0, 0)
            units = float((pay * (sub["result"] == "WIN")).sum() - (sub["result"] == "LOSS").sum())
        else:
            units = w * win_pay - l
        out.append({"label": label, "W": int(w), "L": int(l), "P": int(p),
                    "pct": pct, "units": round(units, 1)})
    return out
