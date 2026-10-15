"""
Shared rating engine: opponent adjustment solved against the full slate,
with shrinkage toward a prior (last season or league average).

Every sport reduces its event-level data to rows of the form

    offense_entity | defense_entity | value | weight | covariates...

e.g. one row per team-game for "pass EPA per dropback" (weight = dropbacks),
or one row per plate appearance for wOBA (weight = 1).

The model solved for each metric is

    value_k = mu + off[o_k] + def[d_k] + sum_c beta_c * x_ck + e_k

by weighted ridge regression. Ridge pulls each team's off/def term toward
its *prior* (not toward zero), so a team with few plays sits near last
season's number and moves away from it only as evidence accumulates.
Because every team's def term is estimated jointly with every opponent's
off term, a strong number against a hard schedule is credited more than the
same number against a soft one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class AdjustedRatings:
    metric: str
    mu: float                                  # league mean of the metric
    off: pd.Series                             # offence term per entity (vs average)
    deff: pd.Series                            # defence term per entity (vs average; + = allows more)
    covariates: dict = field(default_factory=dict)
    raw_off: pd.Series | None = None           # unadjusted weighted means, for transparency
    raw_def: pd.Series | None = None
    n_off: pd.Series | None = None             # total weight (plays) behind each rating
    n_def: pd.Series | None = None

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame({
            "off_adj": self.off, "def_adj": self.deff,
            "off_raw": self.raw_off, "def_raw": self.raw_def,
            "n_off": self.n_off, "n_def": self.n_def,
        })
        t["metric"] = self.metric
        return t


def solve_adjusted(
    df: pd.DataFrame,
    value: str,
    off: str = "off",
    deff: str = "def",
    weight: str | None = "w",
    covariates: list[str] | None = None,
    prior_off: pd.Series | dict | None = None,
    prior_def: pd.Series | dict | None = None,
    prior_weight: float = 200.0,
    def_entities_separate: bool = True,
    prior_weight_def: float | None = None,
) -> AdjustedRatings:
    """
    Weighted ridge-to-prior opponent adjustment.

    prior_weight is in the same units as `weight` (e.g. dropbacks). A value
    of 200 means "the prior counts as much as 200 plays of evidence".
    Priors are expressed vs league average (0 = average) and default to 0.
    """
    covariates = covariates or []
    d = df.dropna(subset=[value, off, deff]).copy()
    w = d[weight].to_numpy(float) if weight else np.ones(len(d))
    keep = w > 0
    d, w = d[keep], w[keep]
    y = d[value].to_numpy(float)

    off_ids = pd.Index(sorted(d[off].astype(str).unique()))
    def_ids = pd.Index(sorted(d[deff].astype(str).unique()))
    oi = off_ids.get_indexer(d[off].astype(str))
    di = def_ids.get_indexer(d[deff].astype(str))
    n, no, nd, nc = len(d), len(off_ids), len(def_ids), len(covariates)

    mu0 = float(np.average(y, weights=w))
    po = pd.Series(prior_off if prior_off is not None else {}, dtype=float).reindex(off_ids).fillna(0).to_numpy()
    pdv = pd.Series(prior_def if prior_def is not None else {}, dtype=float).reindex(def_ids).fillna(0).to_numpy()

    # Re-centre on priors: solve for deviations from prior, penalised toward 0.
    y_res = y - mu0 - po[oi] - pdv[di]

    rows = np.arange(n)
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (rows, oi)), shape=(n, no)),
        sparse.csr_matrix((np.ones(n), (rows, di)), shape=(n, nd)),
        sparse.csr_matrix(d[covariates].to_numpy(float)) if nc else sparse.csr_matrix((n, 0)),
        sparse.csr_matrix(np.ones((n, 1))),   # intercept correction
    ]).tocsr()
    W = sparse.diags(w)
    pwd = prior_weight if prior_weight_def is None else prior_weight_def
    pen = np.r_[np.full(no, prior_weight), np.full(nd, pwd), np.full(nc, 1e-6), [1e-6]]
    A = (X.T @ W @ X + sparse.diags(pen)).tocsc()
    b = X.T @ (w * y_res)
    beta = spsolve(A, b)

    o_hat = po + beta[:no]
    d_hat = pdv + beta[no:no + nd]
    cov = dict(zip(covariates, beta[no + nd:no + nd + nc]))
    mu = mu0 + beta[-1]

    # Identify: weighted-centre off and def at 0 so terms read "vs average".
    wo = np.bincount(oi, weights=w, minlength=no)
    wd = np.bincount(di, weights=w, minlength=nd)
    co = np.average(o_hat, weights=np.maximum(wo, 1e-9))
    cd = np.average(d_hat, weights=np.maximum(wd, 1e-9))
    o_hat, d_hat, mu = o_hat - co, d_hat - cd, mu + co + cd

    raw_o = pd.Series(np.bincount(oi, weights=w * y, minlength=no) / np.maximum(wo, 1e-9) - mu0, index=off_ids)
    raw_d = pd.Series(np.bincount(di, weights=w * y, minlength=nd) / np.maximum(wd, 1e-9) - mu0, index=def_ids)
    return AdjustedRatings(
        metric=value, mu=mu,
        off=pd.Series(o_hat, index=off_ids), deff=pd.Series(d_hat, index=def_ids),
        covariates=cov, raw_off=raw_o, raw_def=raw_d,
        n_off=pd.Series(wo, index=off_ids), n_def=pd.Series(wd, index=def_ids),
    )


def regress_prior(prev: pd.Series | None, keep: float) -> pd.Series | None:
    """Carry last season forward at `keep` strength (0 = league average, 1 = full)."""
    return None if prev is None else prev * keep


def percentile(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0-100 percentile vs the league. For defence pass higher_is_better=False."""
    r = s.rank(pct=True, ascending=higher_is_better)
    return (100 * (r - r.min()) / (r.max() - r.min())).round().astype("Int64")


def rating_book(ratings: dict[str, AdjustedRatings], lower_is_better_def: set[str] | None = None) -> pd.DataFrame:
    """One wide table: <metric>_off, <metric>_def, and their league percentiles."""
    lower_is_better_def = lower_is_better_def or set(ratings)
    out = {}
    for m, r in ratings.items():
        out[f"{m}_off"] = r.off
        out[f"{m}_def"] = r.deff
        out[f"{m}_off_pct"] = percentile(r.off, True)
        # for defence, allowing less of a good-for-offence metric is better
        out[f"{m}_def_pct"] = percentile(r.deff, m not in lower_is_better_def)
    return pd.DataFrame(out)
