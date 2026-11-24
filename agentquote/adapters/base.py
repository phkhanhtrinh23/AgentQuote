"""Domain adapter: everything the quoting pipeline knows about one metered platform.

The adapter supplies
  * a compiler from a drafted plan to typed operators and risky-step flags,
  * the attempt-cost estimator, one cost per operator class (fitted on U_fit or fixed),
  * the outcome classifier, the billing rule for failed work and the prefix of a plan a failure pays for,
  * floors per meter and the price of the compute meter.

The world model, the conformal step and the choice read a platform only through this interface. The
estimator uses pre-run information only and returns a cost per operator class, so a failed attempt can be
charged the part of the plan it ran.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..types import CompiledPlan, Request, Session


class DomainAdapter(ABC):
    name: str = "base"
    comp_unit: str = "units"
    rho_comp: float = 0.0                       # USD per unit of the compute meter
    classes: tuple[str, ...] = ()               # operator classes, in plan order
    outcomes: tuple[str, ...] = ()              # failure outcomes Y (success and give-up excluded)
    flag_names: tuple[str, ...] = ()
    flag_outcome: dict[str, str] = {}           # flag -> outcome it predicts
    upto_class: dict[str, str] = {}             # outcome -> class of the first operator it fails at
    unbilled: frozenset = frozenset()           # outcomes the platform does not bill
    timeout_outcomes: frozenset = frozenset()   # outcomes whose latency is the platform's time limit
    floors: tuple[float, float, float] = (1e-3, 1e-6, 1.0)

    # ------------------------------------------------------------------------------------- compile (live)
    def compile(self, req: Request, code: str, method: str = "") -> CompiledPlan:
        raise NotImplementedError

    def classify(self, feedback: str) -> str:
        return self.outcomes[-1] if self.outcomes else "other"

    # ------------------------------------------------------------------------------------- quantities
    def quantities(self, plan: CompiledPlan | dict | None) -> tuple[np.ndarray, np.ndarray]:
        """Compiled quantity per operator class and a per-class feature (e.g. vector input)."""
        plan = CompiledPlan.from_dict(plan) if isinstance(plan, dict) or plan is None else plan
        v, z = np.zeros(len(self.classes)), np.zeros(len(self.classes))
        for o in plan.ops:
            if o.cls in self.classes:
                i = self.classes.index(o.cls)
                v[i] += float(o.feats.get("volume", 0.0))
                z[i] = max(z[i], float(o.feats.get("vector", 0.0)))
        return v, z

    def metered(self, plan: CompiledPlan | dict | None) -> np.ndarray:
        """Profiled cost per operator class of an executed attempt."""
        plan = CompiledPlan.from_dict(plan) if isinstance(plan, dict) or plan is None else plan
        e = np.zeros(len(self.classes))
        for o in plan.ops:
            if o.cls in self.classes:
                e[self.classes.index(o.cls)] += float(o.feats.get("metered", 0.0))
        return e

    def flags(self, plan: CompiledPlan | dict | None) -> np.ndarray:
        plan = CompiledPlan.from_dict(plan) if isinstance(plan, dict) or plan is None else plan
        return np.array([float(plan.flags.get(f, 0)) for f in self.flag_names])

    def upto_index(self, outcome: str) -> int:
        """Index of the last class a failure with this outcome is charged for (all classes by default)."""
        c = self.upto_class.get(outcome)
        return self.classes.index(c) if c in self.classes else len(self.classes) - 1

    def quality(self, plan: CompiledPlan | dict | None) -> dict:
        plan = CompiledPlan.from_dict(plan) if isinstance(plan, dict) or plan is None else plan
        q = dict(plan.meta.get("quality", {}))
        q["n_obs"] = float(plan.meta.get("n_obs", 0.0))
        return q

    # ------------------------------------------------------------------------------------- estimator
    def fit(self, sessions: Sequence[Session]) -> "DomainAdapter":
        return self

    @abstractmethod
    def price(self, V: np.ndarray, z: np.ndarray, upto: np.ndarray | int | None = None) -> np.ndarray:
        """Attempt cost of compiled quantities V (... x classes), summed over classes up to ``upto``."""

    def money(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, float)
        return self.rho_comp * S[..., 0] + S[..., 1]
