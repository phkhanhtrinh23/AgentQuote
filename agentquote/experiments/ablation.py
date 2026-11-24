"""Component ablations: each variant replaces one component; quotes are calibrated and evaluated as the full method.
Also the weaker-adapter arm, with costs the attempt estimator cannot fit added to every logged call."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import Ctx, interval, mean_of
from .evaluate import ChoiceEval, QSpec, eval_quotes, paired_reduction

ABL = [
    QSpec("full", "pq"), QSpec("reqonly", "v_reqonly"), QSpec("noflags", "v_noflags"), QSpec("onecoef", "v_onecoef"),
    QSpec("succ_att", "v_success_only", target="Alast"), QSpec("succ_ses", "v_success_only"), QSpec("mult", "b2"),
    QSpec("noprev", "v_noprev"), QSpec("nofrail", "v_nofrail"), QSpec("pooled_resid", "v_pooled_resid"),
    QSpec("raw", "pq", "raw"), QSpec("copula", "pq", "copula"), QSpec("stdscore", "pq", "stdscore"),
    QSpec("money", "pq", "money"),
]


def assignment(ctx: Ctx, p: str, a: int):
    df = ctx.table(p, a)
    rng = ctx.rng(a, 4)
    specs = [s for s in ABL if f"{s.est}_comp" in df]
    res = eval_quotes(df, specs, ctx.st, rng)
    m = res["metrics"]
    out = {}
    units = df.loc[res["test"], "unit"].values
    for s in specs:
        r = m.get(s.name)
        if r is None:
            continue
        row = {"cov": r["cov_j"]}
        if "wid_c" in r:
            row["wid"] = float(np.exp(np.mean(np.log([r["wid_c"], r["wid_t"], r["wid_w"]]))))
            if s.name != "full" and s.name in res["Q"]:
                est, _, _ = paired_reduction(res["Q"][s.name], res["Q"]["full"], units, rng, B=50)
                row["dwid"] = float(-np.mean(est))
            row["cov_flagged"] = r.get("cov_flagged")
            row["flag_share"] = r.get("flag_share")
        else:
            row["wid"] = r.get("wid_money")
        out[s.name] = row
    if p == "earthengine":
        ce = ChoiceEval(df, ctx.st, ctx.rng(a, 3))
        for s in specs:
            if s.name in ("money",):
                continue
            c = ce.run(s)
            out[s.name].update({"risk": c["risk"], "cents": c["cents"], "adm": c["adm"]})
    return out


def weaker_adapter(ctx: Ctx, p: str):
    if not (Path(ctx.a.data) / "hard" / p / "sessions.jsonl").exists():
        return None
    cov_pq, cov_b7, red = [], [], []
    for a in range(ctx.a.assign):
        df = ctx.table(p, a, sub="hard", variants=(), selection=False)
        rng = ctx.rng(a, 5)
        res = eval_quotes(df, [QSpec("pq", "pq"), QSpec("b7", "b7")], ctx.st, rng)
        cov_pq.append(res["metrics"]["pq"]["cov_j"])
        cov_b7.append(res["metrics"]["b7"]["cov_j"])
        est, _, _ = paired_reduction(res["Q"]["pq"], res["Q"]["b7"], df.loc[res["test"], "unit"].values, rng, B=50)
        red.append(est)
    red = np.array(red)
    return {"pq_cov": interval(cov_pq), "b7_cov": interval(cov_b7),
            "r_b7": {k: interval(red[:, i]) for i, k in enumerate(("c", "t", "w"))}}


def run(ctx: Ctx):
    for p in ctx.platforms():
        per = [assignment(ctx, p, a) for a in range(ctx.a.assign)]
        out = mean_of(per)
        for name in out:
            vals = [x[name].get("dwid") for x in per if name in x and "dwid" in x[name]]
            if vals:
                out[name]["dwid_ci"] = interval(vals)
        hard = weaker_adapter(ctx, p)
        if hard:
            out["weaker_adapter"] = hard
        ctx.save(f"{p}/ablation.json", out)
