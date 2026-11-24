"""Where a session spends beyond the clean run: cost by part and meter for every session outside U_fit."""
from __future__ import annotations

import numpy as np

from ..conformal import ConformalQuoter
from ..config import DIMS
from .common import Ctx
from .evaluate import M
from .harness import one_per_unit


def run(ctx: Ctx):
    for p in ctx.platforms():
        _, sessions = ctx.data(p)
        df = ctx.table(p, 0)
        outside = set(df.loc[df["split"] != "fit", "session_id"])
        by_id = {s.session_id: s for s in sessions}
        parts = {}
        for sid in outside:
            s = by_id[sid]
            for name, v in s.parts().items():
                parts.setdefault((s.b, name), []).append(v)
        share = {}
        for b in sorted({k[0] for k in parts}):
            tot = sum(np.sum(parts[(b, n)], 0) for n in ("draft_explore", "failed", "final"))
            share[b] = {n: {k: 100 * float(np.sum(parts[(b, n)], 0)[i] / max(tot[i], 1e-12))
                            for i, k in enumerate(DIMS)} for n in ("draft_explore", "failed", "final")}
        avg = {n: {k: float(np.mean([share[b][n][k] for b in share])) for k in DIMS}
               for n in ("draft_explore", "failed", "final")}
        d = df[df["split"] != "fit"]
        excess = 100 * float((M(d, "S") > 2 * M(d, "b1")).any(1).mean())
        # deficit: draft + last attempt covered by a quote calibrated on last attempts, whole session not covered
        rng = ctx.rng(0, 7)
        cal = one_per_unit(df[df["split"] == "quote"], rng)
        q = ConformalQuoter("joint", ctx.st.alpha, df.attrs["floors"]).fit(M(df.loc[cal], "Alast"),
                                                                           M(df.loc[cal], "v_success_only")) \
            if "v_success_only_comp" in df else None
        deficit = float("nan")
        if q is not None:
            Q = q.quote(M(d, "v_success_only"))
            a_ok = (M(d, "Alast") <= Q).all(1)
            s_ok = (M(d, "S") <= Q).all(1)
            deficit = 100 * float((a_ok & ~s_ok).mean())
        ctx.save(f"{p}/breakdown.json", {"share_by_backbone": share, "share": avg, "excess_2x_b1": excess,
                                         "deficit": deficit, "n_sessions": len(d)})
