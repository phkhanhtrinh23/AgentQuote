"""Requests, compiled plans and logged sessions.

A session is stored as one JSON line. Everything after collection (fitting, calibration, evaluation) reads
these logs only. The order of a session is: the planning turn (draft) with its catalogue lookups, the
exploration steps, then the attempts. Each attempt keeps the inspections the agent made after it, so the
sequence of turns and calls is preserved.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import DIMS


@dataclass
class Request:
    request_id: str
    unit: str                     # region or database; requests of one unit are dependent
    rtype: str                    # request type
    text: str
    platform: str                 # "earthengine" | "bigquery"
    candidates: list[str] = field(default_factory=list)   # configuration keys "b|d|m"
    reference: Any = None
    ref_kind: str = ""            # "gold" | "layer" | "" (no reference)
    answer_kind: str = "numeric"  # numeric | binary | categorical | table
    meta: dict = field(default_factory=dict)


def config_key(b: str, d: str, m: str = "") -> str:
    return f"{b}|{d}|{m}"


def split_key(k: str) -> tuple[str, str, str]:
    b, d, m = (k.split("|") + ["", ""])[:3]
    return b, d, m


@dataclass
class Operator:
    uid: int
    cls: str
    call_id: int = 0
    feats: dict = field(default_factory=dict)   # volume, scale, ...; "metered" on executed attempts


@dataclass
class CompiledPlan:
    ops: list[Operator] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)    # n_obs, quality metadata, catalogue lookups

    def to_dict(self) -> dict:
        return {"ops": [asdict(o) for o in self.ops], "flags": dict(self.flags), "meta": dict(self.meta)}

    @staticmethod
    def from_dict(d: dict | None) -> "CompiledPlan":
        d = d or {}
        return CompiledPlan([Operator(**o) for o in d.get("ops", [])], dict(d.get("flags", {})),
                            dict(d.get("meta", {})))


@dataclass
class Turn:
    kind: str                     # draft | explore | attempt | inspect | final | giveup
    tok_in: float
    tok_out: float
    obs_tokens: float = 0.0       # size of the result or error shown after the turn
    latency: float = 0.0
    usd: float = 0.0


@dataclass
class Call:
    kind: str                     # lookup | metadata | inspect | run
    comp: float = 0.0             # metered platform cost
    latency: float = 0.0
    reads_data: bool = False      # an inspection that reads data (not only metadata)
    meta: dict = field(default_factory=dict)


@dataclass
class Step:
    turn: Turn
    calls: list[Call] = field(default_factory=list)

    def cost(self) -> np.ndarray:
        return np.array([sum(c.comp for c in self.calls), self.turn.usd,
                         self.turn.latency + sum(c.latency for c in self.calls)])


@dataclass
class Attempt:
    j: int
    outcome: str                  # "success" or a failure outcome of the platform
    step: Step
    plan: dict | None = None      # compiled executed code; ops carry the profiled cost ("metered")
    inspected: bool = False       # the agent had read data before this attempt
    final: Turn | None = None     # answer turn after a success
    after: list[Step] = field(default_factory=list)   # inspections before the next attempt

    def cost(self, with_after: bool = True) -> np.ndarray:
        c = self.step.cost()
        if self.final is not None:
            c = c + np.array([0.0, self.final.usd, self.final.latency])
        if with_after:
            for s in self.after:
                c = c + s.cost()
        return c


@dataclass
class Session:
    session_id: str
    request_id: str
    unit: str
    rtype: str
    platform: str
    config: str                   # "b|d|m"
    draft: Step | None = None
    plan: dict = field(default_factory=dict)          # compiled draft of this configuration
    explore: list[Step] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    giveup: Turn | None = None
    answer: Any = None
    error: float | None = None    # answer error; inf for a failed session; None without a reference
    has_ref: bool = False
    ref_kind: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def b(self) -> str:
        return split_key(self.config)[0]

    @property
    def d(self) -> str:
        return split_key(self.config)[1]

    @property
    def m(self) -> str:
        return split_key(self.config)[2]

    @property
    def success(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].outcome == "success"

    @property
    def N(self) -> int:
        return len(self.attempts)

    def delta(self) -> np.ndarray:
        """Cost of the draft and its catalogue lookups (known at quote time)."""
        return self.draft.cost() if self.draft else np.zeros(len(DIMS))

    def X(self) -> np.ndarray:
        """Planning turn and exploration."""
        return self.delta() + sum((s.cost() for s in self.explore), np.zeros(len(DIMS)))

    def S(self) -> np.ndarray:
        """Session cost: X plus every attempt, the inspections between them and a give-up turn."""
        s = self.X() + sum((a.cost() for a in self.attempts), np.zeros(len(DIMS)))
        if self.giveup is not None:
            s = s + np.array([0.0, self.giveup.usd, self.giveup.latency])
        return s

    def parts(self) -> dict[str, np.ndarray]:
        """Draft and exploration (with inspections), failed or rejected attempts, final attempt."""
        x = self.X() + sum((st.cost() for a in self.attempts for st in a.after), np.zeros(len(DIMS)))
        if self.giveup is not None:
            x = x + np.array([0.0, self.giveup.usd, self.giveup.latency])
        failed = sum((a.cost(False) for a in self.attempts if a.outcome != "success"), np.zeros(len(DIMS)))
        final = self.attempts[-1].cost(False) if self.success else np.zeros(len(DIMS))
        return {"draft_explore": x, "failed": failed, "final": final}

    def last_attempt_part(self) -> np.ndarray:
        """Draft plus last attempt: what a clean-run estimator prices."""
        return self.delta() + (self.attempts[-1].cost(False) if self.attempts else np.zeros(len(DIMS)))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Session":
        d = dict(d)

        def step(x):
            return Step(Turn(**x["turn"]), [Call(**c) for c in x.get("calls", [])]) if x else None

        d["draft"] = step(d.get("draft"))
        d["explore"] = [step(x) for x in d.get("explore", [])]
        atts = []
        for a in d.get("attempts", []):
            a = dict(a)
            a["step"] = step(a["step"])
            a["final"] = Turn(**a["final"]) if a.get("final") else None
            a["after"] = [step(x) for x in a.get("after", [])]
            atts.append(Attempt(**a))
        d["attempts"] = atts
        d["giveup"] = Turn(**d["giveup"]) if d.get("giveup") else None
        if d.get("error") == "inf":
            d["error"] = math.inf
        return Session(**d)


# ------------------------------------------------------------------------------------------------ JSON lines
def _default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _enc(d):
    if isinstance(d, dict):
        return {k: _enc(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_enc(v) for v in d]
    if isinstance(d, float) and math.isinf(d):
        return "inf"
    return d


def save_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for r in rows:
            f.write(json.dumps(_enc(r), default=_default) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_sessions(path, sessions: Iterable[Session], append: bool = False) -> None:
    save_jsonl(path, (s.to_dict() for s in sessions), append)


def load_sessions(path) -> list[Session]:
    return [Session.from_dict(d) for d in load_jsonl(path)]


def save_requests(path, reqs: Iterable[Request]) -> None:
    save_jsonl(path, (asdict(r) for r in reqs))


def load_requests(path) -> list[Request]:
    return [Request(**d) for d in load_jsonl(path)]
