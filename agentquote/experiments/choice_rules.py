"""Configuration choice: every choice rule on the same requests and budgets, binding budgets and risk levels."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..types import load_jsonl
from .common import Ctx, interval, mean_of
from .evaluate import ChoiceEval, QSpec

PQ = QSpec("pq", "pq")
DELTAS = (0.10, 0.15, 0.20, 0.25, 0.30)
PCTS = (90.0, 95.0, 99.0)


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def assignment(ctx: Ctx, p: str, a: int, templates: dict):
    df = ctx.table(p, a)
    st = ctx.st
    ce = ChoiceEval(df, st, ctx.rng(a, 3))
    out = {}
    pq = ce.run(PQ)
    out["agentquote"] = strip(pq)
    out["best_avg"] = strip(ce.run(PQ, score="train_err", cost="train_err"))
    out["most_acc"] = strip(ce.run(PQ, cost="r"))
    out["plugin"] = strip(ce.run(PQ, rule="plugin"))
    out["cost_feasible"] = strip(ce.run(QSpec("b4raw", "b4", "raw"), score="train_err", cost="train_err",
                                        lam_override=math.inf))
    for b in ("b2", "b4", "b7"):
        out[f"on_{b}"] = strip(ce.run(QSpec(b, b)))
    if templates:
        out["template"] = ce.template(templates)
    out["oracle"] = ce.oracle()
    out["hyper_certified"] = strip(ce.run(PQ, mode="hyper"))["certified"]
    out["cond"] = {f"{d:.2f}": strip(ce.run(PQ, mode="cond", delta=d)) for d in (st.delta, 0.35)}
    out["delta_sweep"] = {f"{d:.2f}": strip(ce.run(PQ, delta=d, aq=d / 2)) for d in DELTAS}
    # ablations of the choice
    if "v_success_only_comp" in df:
        out["abl_succ_ses"] = strip(ce.run(QSpec("succ_ses", "v_success_only")))
    out["abl_meanerr"] = strip(ce.run(PQ, score="r_mean"))
    out["abl_nothreshold"] = strip(ce.run(PQ, lam_override=math.inf))
    # overruns of the chosen quote against a random candidate's
    pick = pq["_pick"]
    ct = ce.df.loc[ce.test_idx]
    Qt = ce.quotes(PQ, st.delta / 2, ce.test_idx)
    rnd = ct.groupby("request_id").sample(1, random_state=a).index
    over = np.any([ct.loc[rnd, f"S_{k}"].values > Qt.loc[rnd, f"Q_{k}"].values for k in ("comp", "tok", "time")], 0)
    out["agentquote"]["ov_random"] = 100 * float(over.mean())
    auc_d = ct[["r", "err"]].assign(y=lambda d: d["err"].astype(float) > st.tau)
    out["auc"] = auc(auc_d["y"].values, auc_d["r"].values)
    out["test_risk_above_delta"] = float(out["agentquote"]["risk"] > 100 * st.delta)
    # alpha_Q read on the test units
    out["aq_test"] = {f"{s:.3f}": strip(ce.run(PQ, aq=s)) for s in (st.delta / 4, 3 * st.delta / 4)}
    # binding budgets
    bind = {}
    for pct in PCTS:
        cb = ChoiceEval(df, st, ctx.rng(a, 3), budget_pct=pct)
        aq = cb.select_alpha_q(PQ)
        r_pq = cb.run(PQ, aq=aq)
        row = {"agentquote": strip(r_pq), "aq": aq,
               "money_time": cb.money_rule(QSpec("mt", "mt", "money_time"), aq),
               "money": cb.money_rule(QSpec("money", "pq", "money"), aq),
               "best_avg": strip(cb.run(PQ, aq=aq, score="train_err", cost="train_err")),
               "hard_caps": cb.hard_caps(r_pq["lam"])}
        bind[f"{pct:.0f}"] = row
    out["bind"] = bind
    return out


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().values
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(ctx: Ctx):
    for p in ctx.platforms(("earthengine",)):
        tp = Path(ctx.a.data) / p / "templates.jsonl"
        templates = {t["request_id"]: t for t in load_jsonl(tp)} if tp.exists() else {}
        per = [assignment(ctx, p, a, templates) for a in range(ctx.a.assign)]
        out = mean_of(per)
        out["agentquote_risk"] = interval([x["agentquote"]["risk"] for x in per])
        out["certified_share"] = 100 * float(np.mean([bool(x["agentquote"]["certified"]) for x in per]))
        out["exceed_count"] = int(sum(x["test_risk_above_delta"] for x in per))
        out["assignments"] = ctx.a.assign
        ctx.save(f"{p}/choice.json", out)
