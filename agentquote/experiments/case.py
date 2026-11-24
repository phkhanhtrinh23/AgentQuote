"""Follow one request through the quote and the choice. The request is picked by a fixed rule: among the flood
test requests of the first assignment that AgentQuote answers and that have candidates over the budget and
above the threshold, the one with the most candidates over the budget."""
from __future__ import annotations

import numpy as np

from ..config import DIMS
from .common import Ctx
from .evaluate import ChoiceEval, QSpec, candidates


def run(ctx: Ctx, rtype: str = "flood"):
    p = "earthengine"
    if p not in ctx.platforms((p,)):
        return
    df = ctx.table(p, 0)
    ce = ChoiceEval(df, ctx.st, ctx.rng(0, 3))
    res = ce.run(QSpec("pq", "pq"))
    lam = res["lam"]
    pick = res["_pick"].set_index("request_id")
    ct = candidates(df, ce.test_idx, ce.quotes(QSpec("pq", "pq"), ctx.st.delta / 2, ce.test_idx), "r", ce.B)
    over = np.any([ct[f"Q_{k}"].values > ct[f"NB_{k}"].values for k in DIMS], axis=0)
    ct = ct.assign(over_budget=over, above=ct["r"].values > lam)
    best, best_n = None, -1
    for r, g in ct.groupby("request_id"):
        if not g["rtype"].iloc[0].startswith(rtype) or not pick.loc[r, "answered"]:
            continue
        if g["over_budget"].any() and g["above"].any() and g["over_budget"].sum() > best_n:
            best, best_n = r, int(g["over_budget"].sum())
    if best is None:
        ctx.save(f"{p}/case.json", {"request_id": None})
        return
    g = ct[ct["request_id"] == best]
    chosen = pick.loc[best, "config"]
    rows = []
    for _, x in g.iterrows():
        rows.append({"config": x["config"], "r": float(x["r"]), "over_budget": bool(x["over_budget"]),
                     "above_threshold": bool(x["above"]), "chosen": x["config"] == chosen,
                     "Q": [float(x[f"Q_{k}"]) for k in DIMS], "B": [float(x[f"NB_{k}"]) for k in DIMS],
                     "S": [float(x[f"S_{k}"]) for k in DIMS], "err": float(x["err"])})
    c = next(r for r in rows if r["chosen"])
    ratio = [q / b for q, b in zip(c["Q"], c["B"])]
    ctx.save(f"{p}/case.json", {"request_id": best, "lam": lam, "n_candidates": len(rows),
                                "n_over_budget": best_n, "n_above_threshold": int(g["above"].sum()),
                                "chosen": chosen, "tightest_meter": DIMS[int(np.argmax(ratio))],
                                "tightest_ratio": float(max(ratio)), "covered": all(s <= q for s, q in zip(c["S"], c["Q"])),
                                "accurate": c["err"] <= ctx.st.tau, "candidates": rows})
