"""Time and cost of quoting: latency per stage, scaling in candidates and rollouts, fitting time, and the cost of
the drafts of options not chosen."""
from __future__ import annotations

import time

import numpy as np

from ..adapters import get_adapter
from ..choice import ErrorRiskScore
from ..config import BACKBONE_NAMES, DIMS
from ..worldmodel import SessionWorldModel
from .common import Ctx
from .evaluate import M


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def run(ctx: Ctx):
    p = "earthengine"
    if p not in ctx.platforms((p,)):
        return
    st = ctx.st
    _, sessions = ctx.data(p)
    df = ctx.table(p, 0)
    by_id = {s.session_id: s for s in sessions}
    fit = [by_id[s] for s in df.loc[df["split"] == "fit", "session_id"]]
    t0 = time.perf_counter()
    wm = SessionWorldModel(get_adapter(p), st, "full", list(BACKBONE_NAMES)).fit(fit)
    fit_s = time.perf_counter() - t0
    test = df[df["split"] == "test"]
    rng = np.random.default_rng(0)
    reqs = test["request_id"].unique()[:40]
    roll_ms, score_ms, draft_s, lookup_s, quote_s = [], [], [], [], []
    score = ErrorRiskScore(st).fit(df[df["split"] == "fit"])
    for r in reqs:
        rows = test[test["request_id"] == r]
        ss = [by_id[sid] for sid in rows["session_id"]]
        t = time.perf_counter()
        for s in ss:
            wm.estimate(wm.pre_run(s), rng)
        roll = (time.perf_counter() - t)
        roll_ms.append(1000 * roll / len(ss))
        t = time.perf_counter()
        score.score(rows)
        score_ms.append(1000 * (time.perf_counter() - t))
        # draft and lookups: the logged planning turn of each data option (drafts run in parallel)
        drafts = {}
        for s in ss:
            drafts[s.d] = s.draft
        dr = max(d.turn.latency for d in drafts.values())
        lk = max(sum(c.latency for c in d.calls) for d in drafts.values())
        draft_s.append(dr)
        lookup_s.append(lk)
        quote_s.append(dr + lk + roll + score_ms[-1] / 1000)
    s0 = by_id[test["session_id"].iloc[0]]
    pr = wm.pre_run(s0)
    ks = [1, 2, 4, 9, 18, 36, 72]
    tk = []
    for k in ks:
        t = time.perf_counter()
        for _ in range(k):
            wm.estimate(pr, rng)
        tk.append(time.perf_counter() - t)
    ms = [100, 300, 1000, 3000]
    tm = []
    for m in ms:
        t = time.perf_counter()
        for _ in range(5):
            wm.estimate(pr, rng, M=m)
        tm.append((time.perf_counter() - t) / 5)
    # drafts of the options not chosen, as a share of the median session's cost per meter
    per = df.groupby("request_id").first()
    extra = M(per, "Dall") - M(per, "D")
    extra[:, 2] = 0.0
    med = np.median(M(df, "S"), 0)
    ovh = {k: 100 * float(np.median(extra[:, i]) / med[i]) for i, k in enumerate(("c", "t", "w"))}
    ctx.save(f"{p}/efficiency.json", {
        "quote_s": float(np.median(quote_s)), "stage": {"draft_s": float(np.median(draft_s)),
                                                        "lookup_s": float(np.median(lookup_s)),
                                                        "rollout_ms": float(np.median(roll_ms)),
                                                        "score_ms": float(np.median(score_ms))},
        "candidates": {"k": ks, "seconds": tk, "slope": slope(ks, tk)},
        "rollouts": {"m": ms, "seconds": tm, "slope": slope(ms, tm)},
        "fit_s": fit_s, "draft_overhead": ovh, "n_roll": st.n_roll})
