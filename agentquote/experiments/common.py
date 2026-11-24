"""Command-line context shared by the analyses."""
from __future__ import annotations

import argparse
from functools import cached_property
from pathlib import Path

import numpy as np

from ..config import Settings
from .harness import load, platform_of, prepare, write_json


def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--data", default="data", help="directory with <platform>/sessions.jsonl")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--platform", default="all", help="earthengine | bigquery | all")
    ap.add_argument("--assign", type=int, default=10, help="random unit assignments")
    ap.add_argument("--n-roll", type=int, default=1000, help="imagined rollouts per configuration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


class Ctx:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out)
        self.cache = self.out / "cache"
        self.st = Settings(n_roll=a.n_roll, seed=a.seed)
        self._data = {}

    def platforms(self, allowed=("earthengine", "bigquery")):
        p = self.a.platform
        ps = list(allowed) if p == "all" else [platform_of(p)]
        return [x for x in ps if x in allowed and (Path(self.a.data) / x / "sessions.jsonl").exists()]

    def data(self, platform, sub=None):
        key = (platform, sub)
        if key not in self._data:
            d = Path(self.a.data) if sub is None else Path(self.a.data) / sub
            self._data[key] = load(d, platform)
        return self._data[key]

    def table(self, platform, a, **kw):
        sub = kw.pop("sub", None)
        _, sess = self.data(platform, sub)
        tag = kw.pop("tag", "" if sub is None else f"_{sub}")
        return prepare(platform, sess, self.st, self.a.seed + a, self.cache, tag=tag, log=self.log, **kw)

    def rng(self, a, salt=0):
        return np.random.default_rng(1_000 * (self.a.seed + a) + salt)

    def log(self, *x):
        if not self.a.quiet:
            print(*x, flush=True)

    def save(self, name, obj):
        write_json(self.out / name, obj)
        self.log(f"  wrote {self.out / name}")


def interval(x):
    """Mean and 95% interval of the mean over assignments."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = float(x.mean())
    h = 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"mean": m, "lo": m - h, "hi": m + h, "n": len(x)}


def mean_of(dicts, skip=("_pick",)):
    """Mean over assignments of nested metric dicts (numbers only)."""
    out = {}
    keys = {k for d in dicts for k in d} - set(skip)
    for k in keys:
        vals = [d.get(k) for d in dicts]
        if all(v is None for v in vals):
            out[k] = None
        elif all(isinstance(v, dict) for v in vals if v is not None):
            out[k] = mean_of([v for v in vals if v is not None])
        else:
            nums = [float(v) for v in vals if isinstance(v, (int, float, np.floating, np.integer)) and v is not None]
            nums = [v for v in nums if not np.isnan(v)]
            out[k] = float(np.mean(nums)) if nums else float("nan")
    return out
