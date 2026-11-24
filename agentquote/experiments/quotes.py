"""Quote quality: joint and per-meter coverage, width and miss of every quoting method, all calibrated with the
same joint score on the same U_Q; paired width reductions; coverage across nominal levels."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..conformal import coverage_cap
from .common import Ctx, interval, mean_of
from .evaluate import QSpec, eval_quotes, paired_reduction

SPECS = [
    QSpec("b1raw", "b1", "raw"), QSpec("b5raw", "b5", "raw"),
    QSpec("b1", "b1"), QSpec("b2", "b2"), QSpec("b3", "b3"), QSpec("b4", "b4"), QSpec("b5", "b5"),
    QSpec("b6", "b6"), QSpec("b7", "b7"), QSpec("b7s", "b7s"),
    QSpec("pq", "pq"), QSpec("bonf", "pq", "bonferroni"), QSpec("sel", "sel"),
]
SQL_EXTRA = [QSpec("pq_perq", "pq", per_question=True)]
VS = ("b2", "b4", "b5", "b7", "b7s")
LEVELS = (0.80, 0.85, 0.90, 0.95)


def assignment(ctx: Ctx, p: str, a: int, sub=None):
    df = ctx.table(p, a, sub=sub, **({"variants": (), "selection": False} if sub else {}))
    rng = ctx.rng(a, 1)
    specs = SPECS + (SQL_EXTRA if p == "bigquery" else [])
    res = eval_quotes(df, specs, ctx.st, rng)
    T = df.loc[res["test"]]
    ratios = {}
    for b in VS:
        if "pq" in res["Q"] and b in res["Q"]:
            est, lo, hi = paired_reduction(res["Q"]["pq"], res["Q"][b], T["unit"].values, rng)
            ratios[b] = {"c": est[0], "t": est[1], "w": est[2], "lo": lo.tolist(), "hi": hi.tolist()}
    # which meter sets the joint score on U_Q
    q = df[df["split"] == "quote"]
    R = q[[f"S_{k}" for k in ("comp", "tok", "time")]].values / np.maximum(
        q[[f"pq_{k}" for k in ("comp", "tok", "time")]].values, df.attrs["floors"])
    binding = {k: 100 * float((R.argmax(1) == i).mean()) for i, k in enumerate(("c", "t", "w"))}
    return res["metrics"], ratios, binding, res["metrics"]["pq"]["n_cal"]


def alpha_sweep(ctx: Ctx, p: str, a: int):
    df = ctx.table(p, a)
    rng = ctx.rng(a, 2)
    out = {}
    for lv in LEVELS:
        r = eval_quotes(df, [QSpec("pq", "pq"), QSpec("b7", "b7"), QSpec("succ_att", "v_success_only", target="Alast")]
                        if "v_success_only_comp" in df else [QSpec("pq", "pq"), QSpec("b7", "b7")],
                        ctx.st, rng, alpha=1 - lv)["metrics"]
        out[f"{lv:.2f}"] = {k: v["cov_j"] for k, v in r.items()}
    return out


def run(ctx: Ctx):
    for p in ctx.platforms():
        per, rat, bind, ncal = [], [], [], []
        sweeps = []
        for a in range(ctx.a.assign):
            m, r, b, n = assignment(ctx, p, a)
            per.append(m), rat.append(r), bind.append(b), ncal.append(n)
            if p == "earthengine":
                sweeps.append(alpha_sweep(ctx, p, a))
        summary = {name: mean_of([x[name] for x in per if name in x]) for name in per[0]}
        cov = interval([x["pq"]["cov_j"] for x in per])
        ratios = {}
        for b in VS:
            vals = [x[b] for x in rat if b in x]
            if vals:
                ratios[b] = {k: interval([v[k] for v in vals]) for k in ("c", "t", "w")}
        out = {"methods": summary, "pq_joint_coverage": cov, "ratios": ratios, "binding": mean_of(bind),
               "coverage_cap": coverage_cap(int(np.mean(ncal)), ctx.st.alpha), "n_cal": float(np.mean(ncal)),
               "assignments": ctx.a.assign}
        if sweeps:
            out["alpha_sweep"] = {lv: {k: interval([s[lv][k] for s in sweeps if k in s[lv]])
                                       for k in sweeps[0][lv]} for lv in sweeps[0]}
        if (Path(ctx.a.data) / "pool2" / p / "sessions.jsonl").exists():
            r2, cov2 = [], []
            for a in range(ctx.a.assign):
                m, r, _, _ = assignment(ctx, p, a, sub="pool2")
                r2.append(r), cov2.append(m["pq"]["cov_j"])
            out["pool2"] = {"ratios": {b: {k: interval([v[b][k] for v in r2 if b in v]) for k in ("c", "t", "w")}
                                       for b in ("b7", "b7s") if any(b in v for v in r2)},
                            "pq_joint_coverage": interval(cov2)}
        ctx.save(f"{p}/quotes.json", out)
