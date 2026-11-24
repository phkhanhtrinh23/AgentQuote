"""Earth Engine adapter.

A raster operator u over region G has volume v_u = n_img * n_band * |G ∩ F_u| / w_u^2 (images left after the
plan's filters, bands read, footprint, scale of the materialising call). An operator of class xi gets

    log c_u = beta0_xi + beta_xi * log v_u + gamma_xi * z_u,

fitted by least squares on profiled attempts of U_fit (z_u marks vector inputs). Failed work is billed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import USD_PER_EECU_S
from ..types import CompiledPlan, Request, Session
from .base import DomainAdapter
from .ee_compiler import EECompiler

CLASSES = ("load", "filter", "mask", "composite", "inference", "reduce", "export")


class EarthEngineAdapter(DomainAdapter):
    name = "earthengine"
    comp_unit = "EECU-s"
    rho_comp = USD_PER_EECU_S
    classes = CLASSES
    outcomes = ("pixel_limit", "missing_band", "empty_collection", "client_transfer", "memory", "other")
    flag_names = ("pixel_limit", "missing_band", "empty_collection", "client_transfer", "unparsed")
    flag_outcome = {"pixel_limit": "pixel_limit", "missing_band": "missing_band",
                    "empty_collection": "empty_collection", "client_transfer": "client_transfer"}
    upto_class = {"missing_band": "reduce", "empty_collection": "load", "client_transfer": "filter"}
    timeout_outcomes = frozenset({"memory"})
    floors = (1e-3, 1e-6, 1.0)

    def __init__(self, resolver=None, single_coef: bool = False, min_rows: int = 5):
        self.compiler = EECompiler(resolver)
        self.single_coef, self.min_rows = single_coef, min_rows
        self.theta = np.zeros(len(CLASSES))
        self.beta = np.ones(len(CLASSES))
        self.gamma = np.zeros(len(CLASSES))

    # ------------------------------------------------------------------------------------- live
    def compile(self, req: Request, code: str, method: str = "") -> CompiledPlan:
        plan = self.compiler.compile(req, code, method)
        plan.meta.setdefault("quality", dict(req.meta.get("quality", {})))
        return plan

    def classify(self, feedback: str) -> str:
        f = (feedback or "").lower()
        rules = [("too many pixels", "pixel_limit"), ("maxpixels", "pixel_limit"),
                 ("pattern", "missing_band"), ("band", "missing_band"),
                 ("empty", "empty_collection"), ("no bands", "empty_collection"),
                 ("memory", "memory"), ("computation timed out", "memory"), ("deadline", "memory"),
                 ("collection query aborted after accumulating over", "client_transfer")]
        for k, y in rules:
            if k in f:
                return y
        return "other"

    # ------------------------------------------------------------------------------------- estimator
    def fit(self, sessions: Sequence[Session]) -> "EarthEngineAdapter":
        rows = []
        for s in sessions:
            for a in s.attempts:
                if a.outcome != "success" or not a.plan:
                    continue
                v, z = self.quantities(a.plan)
                e = self.metered(a.plan)
                for u in range(len(CLASSES)):
                    if v[u] > 0 and e[u] > 0:
                        rows.append((u, np.log(max(v[u], 1.0)), z[u], np.log(e[u])))
        if not rows:
            return self
        R = np.array(rows)
        if self.single_coef:
            A = np.c_[np.ones(len(R)), R[:, 1], R[:, 2]]
            sol = np.linalg.lstsq(A, R[:, 3], rcond=None)[0]
            self.theta[:], self.beta[:], self.gamma[:] = sol
            return self
        for u in range(len(CLASSES)):
            m = R[:, 0] == u
            if m.sum() < self.min_rows:
                continue
            A = np.c_[np.ones(m.sum()), R[m, 1], R[m, 2]]
            self.theta[u], self.beta[u], self.gamma[u] = np.linalg.lstsq(A, R[m, 3], rcond=1e-6)[0]
        return self

    def price(self, V, z, upto=None):
        V = np.asarray(V, float)
        mu = self.theta + self.beta * np.log(np.maximum(V, 1.0)) + self.gamma * np.asarray(z, float)
        c = np.where(V > 0, np.exp(mu), 0.0)
        if upto is not None:
            c = c * (np.arange(len(CLASSES)) <= np.asarray(upto)[..., None])
        return c.sum(-1)
