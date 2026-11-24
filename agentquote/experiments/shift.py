"""Quotes under shift and re-quoting mid-session (one assignment).

  continent held out of calibration; later re-executions; a backbone held out of fitting (borrowed heads,
  then its factor recomputed on a few of its own sessions); own-draft sessions; re-quotes after a failed first
  attempt and after exploration. Each coverage is compared with the band of test sampling alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import binom

from ..adapters import get_adapter
from ..conformal import ConformalQuoter
from ..config import BACKBONE_NAMES, DIMS
from ..types import load_sessions
from ..worldmodel import SessionWorldModel
from .common import Ctx
from .evaluate import M
from .harness import one_per_unit


def band(n, alpha):
    if n == 0:
        return [float("nan")] * 2
    return [100 * binom.ppf(0.025, n, 1 - alpha) / n, 100 * binom.ppf(0.975, n, 1 - alpha) / n]


def joint_cov(S, Q):
    return 100 * float((S <= Q).all(1).mean()) if len(S) else float("nan")


def run(ctx: Ctx):
    p = "earthengine"
    if p not in ctx.platforms((p,)):
        return
    st, alpha = ctx.st, ctx.st.alpha
    _, sessions = ctx.data(p)
    df = ctx.table(p, 0)
    floors = df.attrs["floors"]
    rng = ctx.rng(0, 6)
    by_id = {s.session_id: s for s in sessions}
    quote_rows, test_rows = df[df["split"] == "quote"], df[df["split"] == "test"]
    out = {}

    def fitq(rows, est="pq", target="S"):
        return ConformalQuoter("joint", alpha, floors).fit(M(rows, target), M(rows, est))

    # continent held out of calibration
    cont = {}
    for c in sorted(df["continent"].unique()):
        cal = one_per_unit(quote_rows[quote_rows["continent"] != c], rng)
        te = one_per_unit(test_rows[test_rows["continent"] == c], rng)
        if len(te) == 0:
            continue
        q = fitq(df.loc[cal])
        T = df.loc[te]
        cont[c] = {"cov": joint_cov(M(T, "S"), q.quote(M(T, "pq"))), "n": len(te), "band": band(len(te), alpha)}
    out["continent"] = cont

    # fitted world model of this assignment
    fit = [by_id[s] for s in df.loc[df["split"] == "fit", "session_id"]]
    wm = SessionWorldModel(get_adapter(p), st, "full", list(BACKBONE_NAMES)).fit(fit)
    q0 = fitq(df.loc[one_per_unit(quote_rows, rng)])

    # later re-executions of the same requests
    rp = Path(ctx.a.data) / p / "rerun_sessions.jsonl"
    if rp.exists():
        rer = [s for s in load_sessions(rp) if s.unit in set(test_rows["unit"])]
        key = df.set_index(["request_id", "config"])
        pick = {}
        for s in rer:
            pick.setdefault(s.unit, []).append(s)
        chosen = [g[rng.integers(len(g))] for g in pick.values()]
        S = np.array([s.S() for s in chosen])
        Sh = np.array([key.loc[(s.request_id, s.config), [f"pq_{k}" for k in DIMS]].values.astype(float)
                       for s in chosen])
        out["rerun"] = {"cov": joint_cov(S, q0.quote(Sh)), "n": len(chosen), "band": band(len(chosen), alpha)}

    # a backbone held out of fitting
    bb = {}
    for b in BACKBONE_NAMES:
        wb = SessionWorldModel(get_adapter(p), st, "full", list(BACKBONE_NAMES)).fit([s for s in fit if s.b != b])
        wb.borrow(b)
        qr = quote_rows[quote_rows["b"] != b]
        cal = one_per_unit(qr, rng)
        te = one_per_unit(test_rows[test_rows["b"] == b], rng)
        own = one_per_unit(quote_rows[quote_rows["b"] == b], rng)[:30]

        def est(idx):
            return np.array([wb.estimate(wb.pre_run(by_id[sid]), rng)["S_hat"] for sid in df.loc[idx, "session_id"]])
        Ec, Et, Eo = est(cal), est(te), est(own)
        q = ConformalQuoter("joint", alpha, floors).fit(M(df.loc[cal], "S"), Ec)
        qrc = ConformalQuoter("joint", alpha, floors).fit(M(df.loc[own], "S"), Eo)
        St = M(df.loc[te], "S")
        bb[b] = {"borrowed": joint_cov(St, q.quote(Et)), "recal": joint_cov(St, qrc.quote(Et)), "n": len(te),
                 "band": band(len(te), alpha), "recal_sessions": len(own)}
    out["held_out_backbone"] = bb

    # own-draft sessions
    op = Path(ctx.a.data) / p / "owndraft_sessions.jsonl"
    if op.exists():
        test_units = set(test_rows["unit"])
        own = [s for s in load_sessions(op) if s.unit in test_units]
        key = df.set_index(["request_id", "config"])
        So = np.array([s.S() for s in own])
        Ss = np.array([key.loc[(s.request_id, s.config), [f"S_{k}" for k in DIMS]].values.astype(float) for s in own])
        Sh = np.array([key.loc[(s.request_id, s.config), [f"pq_{k}" for k in DIMS]].values.astype(float) for s in own])
        rho = df.attrs["rho"]
        mo, ms = rho * So[:, 0] + So[:, 1], rho * Ss[:, 0] + Ss[:, 1]
        Q = q0.quote(Sh)
        out["own_draft"] = {"money_more": 100 * float(np.median(mo / np.maximum(ms, 1e-12)) - 1),
                            "cov_own": joint_cov(So, Q), "cov_shared": joint_cov(Ss, Q), "n": len(own),
                            "regions": len({s.unit for s in own}), "band": band(len(own), alpha)}

    # re-quoting mid-session
    def states(rows, phase):
        res = []
        for sid in rows["session_id"]:
            s = by_id[sid]
            if phase == "failed":
                if not s.attempts or s.attempts[0].outcome == "success":
                    continue
                a1 = s.attempts[0]
                spent = s.X() + a1.cost(False)
                v, _ = wm.ad.quantities(a1.plan)
                start = {"phase": "failed", "v": v, "j": 1, "y": a1.outcome,
                         "ctx": a1.step.turn.tok_in + a1.step.turn.tok_out + a1.step.turn.obs_tokens,
                         "iota": a1.inspected, "nu_probs": wm.posterior(s, 1)}
            else:
                if not s.explore:
                    continue
                spent = s.X()
                start = {"phase": "explored", "iota": any(c.reads_data for x in s.explore for c in x.calls),
                         "ctx": s.attempts[0].step.turn.tok_in if s.attempts else 0.0,
                         "nu_probs": wm.posterior(s, 0)}
            res.append((sid, s, spent, start))
        return res

    rq = {}
    for phase in ("failed", "explored"):
        cal = states(df.loc[one_per_unit(quote_rows, rng)], phase)
        te = states(df.loc[one_per_unit(test_rows, rng)], phase)
        if len(cal) < 5 or not te:
            continue

        def remaining(items):
            R, E = [], []
            for sid, s, spent, start in items:
                R.append(np.maximum(s.S() - spent, 0.0))
                E.append(wm.estimate(wm.pre_run(s), rng, start=start)["S_hat"])
            return np.array(R), np.array(E)
        Rc, Ec = remaining(cal)
        Rt, Et = remaining(te)
        q = ConformalQuoter("joint", alpha, floors).fit(Rc, Ec)
        Qr = q.quote(Et)
        St = np.array([s.S() for _, s, _, _ in te])
        Q0 = q0.quote(df.set_index("session_id").loc[[sid for sid, *_ in te], [f"pq_{k}" for k in DIMS]].values)
        share = 100 * len(te) / max(len(one_per_unit(test_rows, rng)), 1)
        rq[phase] = {"cov_requote": joint_cov(Rt, Qr), "cov_start": joint_cov(St, Q0), "n": len(te),
                     "share": share, "band": band(len(te), alpha),
                     "wid_time_requote": float(np.median(Qr[:, 2] / np.maximum(Rt[:, 2], 1e-9))),
                     "wid_time_start": float(np.median(np.maximum(Q0[:, 2] - (St[:, 2] - Rt[:, 2]), 1e-9)
                                                       / np.maximum(Rt[:, 2], 1e-9)))}
    out["requote"] = rq
    ctx.save(f"{p}/shift.json", out)
