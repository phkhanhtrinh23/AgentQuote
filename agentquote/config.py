"""Defaults, meters and list prices shared by every module."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # optional: credentials for live collection
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

# Metered dimensions, in the order used by every cost vector: platform compute, token spend (USD), wall time (s).
DIMS: tuple[str, ...] = ("comp", "tok", "time")
COMP, TOK, TIME = 0, 1, 2

# Backbones: short name -> (OpenRouter id, USD per 1M input tokens, USD per 1M output tokens).
# Open-weight models are billed at the list price of a hosted endpoint serving the same weights.
BACKBONES: dict[str, tuple[str, float, float]] = {
    "qwen3-32b": ("qwen/qwen3-32b", 0.10, 0.30),
    "llama-3.3-70b": ("meta-llama/llama-3.3-70b-instruct", 0.60, 0.80),
    "gpt-5-mini": ("openai/gpt-5-mini", 0.25, 2.00),
}
BACKBONE_NAMES = tuple(BACKBONES)
DRAFTER = "qwen3-32b"

# Platform list prices of the compute meter (USD per unit).
USD_PER_EECU_S = 0.40 / 3600.0          # Earth Engine, commercial tier
USD_PER_BYTE = 6.25 / 2.0 ** 40          # BigQuery on-demand
BQ_MIN_BYTES = 10 * 2 ** 20              # BigQuery minimum bill per query


def price_in(b: str) -> float:
    return BACKBONES[b][1] / 1e6 if b in BACKBONES else 0.0


def price_out(b: str) -> float:
    return BACKBONES[b][2] / 1e6 if b in BACKBONES else 0.0


def turn_usd(b: str, tok_in: float, tok_out: float) -> float:
    return price_in(b) * tok_in + price_out(b) * tok_out


@dataclass
class Settings:
    alpha: float = 0.10          # joint miscoverage of a quote
    alpha0: float = 0.10         # quantile level of the rollout estimate
    delta: float = 0.20          # risk level of the certified choice
    eta: float = 0.10            # Learn-then-Test failure probability
    tau: float = 0.10            # answer-error tolerance
    alpha_q: float | None = None  # quote level read by the choice (default: chosen on pseudo-splits of U_fit)
    n_roll: int = 1000           # imagined rollouts per configuration
    n_max: int = 5               # attempt cap
    max_turns: int = 50
    n_nb: int = 60               # nearest successful attempts for cost residuals
    ridge: float = 1.0           # ridge penalty of the hazard head
    H: int = 100                 # threshold grid size
    budget_pct: float = 99.0     # budget per meter: percentile of the type's U_fit session cost
    gbm_trees: int = 200
    gbm_leaves: int = 15
    gbm_lr: float = 0.05
    gbm_min_leaf: int = 20
    b7_trials: int = 8
    temperature: float = 0.7
    max_out_tokens: int = 4096
    obs_chars: int = 2000
    seed: int = 0
    floors: dict = field(default_factory=lambda: {"comp": 1e-3, "tok": 1e-6, "time": 1.0})

    @property
    def aq(self) -> float:
        return self.alpha_q if self.alpha_q is not None else self.delta / 2


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)
