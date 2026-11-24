"""Sensitivity to two constants of the quote: the fitting-set size and the number of rollouts. Each setting
runs on its own set of assignments with AgentQuote and B7 only; margins over B7 are paired within each set."""
from __future__ import annotations

import numpy as np

from ..config import Settings
from .common import Ctx, interval
from .evaluate import QSpec, eval_quotes, paired_reduction
from .harness import prepare

SPECS = [QSpec("pq", "pq"), QSpec("b7", "b7")]


def margins(ctx, p, sessions, st, a, tag, **kw):
    df = prepare(p, sessions, st, 100 + ctx.a.seed + a, ctx.cache, variants=(), selection=False, tag=tag,
                 log=ctx.log, **kw)
    rng = ctx.rng(a, 9)
    res = eval_quotes(df, SPECS, st, rng)
    est, _, _ = paired_reduction(res["Q"]["pq"], res["Q"]["b7"], df.loc[res["test"], "unit"].values, rng, B=50)
    return res["metrics"]["pq"]["cov_j"], est


def run(ctx: Ctx, n_assign: int | None = None):
    p = "earthengine"
    if p not in ctx.platforms((p,)):
        return
    n = n_assign or max(2, ctx.a.assign // 2)
    _, sessions = ctx.data(p)
    out = {"fit_regions": {}, "rollouts": {}, "assignments": n}
    n_fit = len({s.unit for s in sessions}) * 0.2
    for k in (30, 60, 90, None):
        if k is not None and k >= n_fit:
            continue
        covs, reds = [], []
        for a in range(n):
            c, r = margins(ctx, p, sessions, ctx.st, a, f"_fit{k or 'all'}", fit_regions=k)
            covs.append(c), reds.append(r)
        reds = np.array(reds)
        out["fit_regions"][str(k or "all")] = {"cov": interval(covs),
                                               **{d: interval(reds[:, i]) for i, d in enumerate(("c", "t", "w"))}}
    for m in (100, 300, ctx.st.n_roll):
        st = Settings(**{**ctx.st.__dict__, "n_roll": m})
        covs, reds = [], []
        for a in range(n):
            c, r = margins(ctx, p, sessions, st, a, f"_roll{m}")
            covs.append(c), reds.append(r)
        reds = np.array(reds)
        out["rollouts"][str(m)] = {"cov": interval(covs),
                                   **{d: interval(reds[:, i]) for i, d in enumerate(("c", "t", "w"))}}
    ctx.save(f"{p}/sensitivity.json", out)
