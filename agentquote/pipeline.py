"""AgentQuote end to end.

  fit(U_fit)                     adapter estimator, session world model, error-risk score
  estimate(sessions)             per configuration: S_hat (rollout quantile), failure probability, features
  calibrate(U_Q, U_lambda)       joint conformal factors at alpha and alpha_Q, Learn-then-Test threshold
  quote_and_choose(candidates)   quotes at alpha, the admissible candidate with the lowest remaining money quote
                                 (quoted at alpha_Q), or a decline

Estimates read only pre-run fields of a logged session (the compiled draft, the configuration and the draft's
cost), so every logged candidate can be re-quoted offline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .adapters.base import DomainAdapter
from .choice import ErrorRiskScore, add_net_budget, calibrate_threshold, choose, threshold_grid
from .conformal import ConformalQuoter
from .config import BACKBONE_NAMES, DIMS, Settings
from .types import Session
from .worldmodel import SessionWorldModel


def session_rows(sessions: Sequence[Session], ad: DomainAdapter) -> pd.DataFrame:
    """Realised costs, answer errors and pre-run metadata of logged sessions, one row per session."""
    rows = []
    for s in sessions:
        S, A, D = s.S(), s.last_attempt_part(), s.delta()
        r = {"session_id": s.session_id, "request_id": s.request_id, "unit": s.unit, "rtype": s.rtype,
             "config": s.config, "b": s.b, "d": s.d, "m": s.m, "has_ref": bool(s.has_ref), "ref_kind": s.ref_kind,
             "err": math.inf if (s.has_ref and s.error is None) else (s.error if s.error is not None else np.nan),
             "success": s.success, "N": s.N, "continent": s.extras.get("continent", "")}
        for i, k in enumerate(DIMS):
            r[f"S_{k}"], r[f"Alast_{k}"], r[f"D_{k}"] = S[i], A[i], D[i]
        for k, v in ad.quality(s.plan).items():
            r[f"q_{k}"] = v
        for f, v in zip(ad.flag_names, ad.flags(s.plan)):
            r[f"flag_{f}"] = bool(v)
        rows.append(r)
    df = pd.DataFrame(rows)
    return add_draft_totals(df)


def add_draft_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Delta_all(q): every data option's draft once; in wall time the drafts run in parallel (admission time)."""
    per = df.groupby(["request_id", "d"])[[f"D_{k}" for k in DIMS]].first().reset_index()
    agg = per.groupby("request_id").agg(Dall_comp=("D_comp", "sum"), Dall_tok=("D_tok", "sum"),
                                        Dall_time=("D_time", "max")).reset_index()
    return df.drop(columns=[c for c in agg.columns if c != "request_id" and c in df]).merge(agg, on="request_id")


def mat(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_{k}" for k in DIMS]].values.astype(float)


@dataclass
class Calibration:
    quoter: ConformalQuoter
    quoter_q: ConformalQuoter
    lam: float
    info: dict = field(default_factory=dict)


class AgentQuote:
    def __init__(self, adapter: DomainAdapter, settings: Settings | None = None, variant: str = "full",
                 backbones: Sequence[str] = BACKBONE_NAMES):
        self.ad, self.st = adapter, settings or Settings()
        self.wm = SessionWorldModel(adapter, self.st, variant, list(backbones))

    # ---------------------------------------------------------------------------------------- offline
    def fit(self, fit_sessions: Sequence[Session], rng: np.random.Generator | None = None) -> "AgentQuote":
        rng = rng or np.random.default_rng(self.st.seed)
        self.wm.fit(fit_sessions)
        df = self.estimate(fit_sessions, rng)
        self.score = ErrorRiskScore(self.st).fit(df)
        self.grid = threshold_grid(self.score.score(df[df["has_ref"]]) if df["has_ref"].any() else [0, 1], self.st.H)
        self.fit_table = df
        return self

    def estimate(self, sessions: Sequence[Session], rng: np.random.Generator, M: int | None = None) -> pd.DataFrame:
        df = session_rows(sessions, self.ad)
        est = [self.wm.estimate(self.wm.pre_run(s), rng, M) for s in sessions]
        for i, k in enumerate(DIMS):
            df[f"Shat_{k}"] = [e["S_hat"][i] for e in est]
        df["Shat_money"] = [e["money_hat"] for e in est]
        df["p_fail"] = [e["p_fail"] for e in est]
        df["mean_N"] = [e["mean_N"] for e in est]
        return df

    def calibrate(self, df_Q: pd.DataFrame, df_lam: pd.DataFrame | None, alpha_q: float | None = None,
                  budget: dict | None = None) -> Calibration:
        st = self.st
        aq = st.aq if alpha_q is None else alpha_q
        qa = ConformalQuoter("joint", st.alpha, self.ad.floors).fit(mat(df_Q, "S"), mat(df_Q, "Shat"))
        ref = df_Q[df_Q["has_ref"]]
        qq = ConformalQuoter("joint", aq, self.ad.floors).fit(mat(ref, "S"), mat(ref, "Shat"))
        lam, info = math.inf, {}
        if df_lam is not None and len(df_lam):
            c = self.prepare_candidates(df_lam, qq, budget)
            res = calibrate_threshold(c, self.grid, st, self.ad.rho_comp)
            lam, info = res["lam"], res
        return Calibration(qa, qq, lam, info)

    def prepare_candidates(self, df: pd.DataFrame, quoter: ConformalQuoter, budget: dict | None) -> pd.DataFrame:
        d = df.copy()
        Q = quoter.quote(mat(d, "Shat"))
        for i, k in enumerate(DIMS):
            d[f"Q_{k}"] = Q[:, i]
        d["r"] = self.score.score(d)
        if budget is not None:
            for k in DIMS:
                d[f"B_{k}"] = [budget[k].get(t, math.inf) if isinstance(budget[k], dict) else budget[k] for t in d["rtype"]]
        else:
            for k in DIMS:
                d[f"B_{k}"] = math.inf
        return add_net_budget(d)

    # ---------------------------------------------------------------------------------------- online
    def quote_and_choose(self, cands: pd.DataFrame, cal: Calibration, budget: dict | None = None):
        """cands: estimate rows of one or more requests. Returns (quotes at alpha, choice per request)."""
        quotes = cands[["request_id", "config"]].copy()
        Q = cal.quoter.quote(mat(cands, "Shat"))
        for i, k in enumerate(DIMS):
            quotes[f"Q_{k}"] = Q[:, i]
        c = self.prepare_candidates(cands, cal.quoter_q, budget)
        pick = choose(c, cal.lam, self.ad.rho_comp)
        return quotes, pick
