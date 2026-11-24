"""Shared evaluation harness: unit splits and the per-assignment estimate table.

For one assignment (a random partition of the pool's units into U_fit, U_Q, U_lambda and the test set) the
harness fits AgentQuote and every baseline on U_fit and writes one row per logged session with its realised
cost, answer error, the world model's estimate S_hat, every baseline's estimate, the ablation variants and the
error-risk score. Analyses read these tables; they are cached under <out>/cache.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..adapters import get_adapter
from ..baselines import Baselines, per_dim_selection
from ..choice import ErrorRiskScore
from ..config import BACKBONE_NAMES, DIMS, Settings
from ..pipeline import session_rows
from ..types import Session, load_requests, load_sessions
from ..worldmodel import SessionWorldModel

SHARES = {"earthengine": (0.20, 0.20, 0.35, 0.25), "bigquery": (0.40, 0.30, 0.0, 0.30)}
VARIANTS = {"earthengine": ("reqonly", "noflags", "onecoef", "success_only", "noprev", "nofrail", "pooled_resid"),
            "bigquery": ("noflags", "success_only", "noprev")}


def platform_of(name: str) -> str:
    return {"ee": "earthengine", "bq": "bigquery", "sql": "bigquery"}.get(name, name)


def load(data_dir: str | Path, platform: str):
    d = Path(data_dir) / platform
    return load_requests(d / "requests.jsonl"), load_sessions(d / "sessions.jsonl")


# ------------------------------------------------------------------------------------------------ splits
def split_units(sessions: Sequence[Session], platform: str, rng: np.random.Generator) -> dict[str, str]:
    units = sorted({s.unit for s in sessions})
    ref = sorted({s.unit for s in sessions if s.has_ref})
    f, q, c, t = SHARES[platform]
    n = len(units)
    out = {}
    nc = int(round(c * n))
    if nc:
        for u in rng.choice(ref, size=min(nc, len(ref)), replace=False):
            out[u] = "choice"
    rest = [u for u in rng.permutation(units) if u not in out]
    nf, nq = int(round(f * n)), int(round(q * n))
    for i, u in enumerate(rest):
        out[u] = "fit" if i < nf else ("quote" if i < nf + nq else "test")
    return out


def one_per_unit(df: pd.DataFrame, rng: np.random.Generator, ref_only: bool = False,
                 all_candidates: bool = False, per_request: bool = False) -> pd.Index:
    """One request per unit (or every request) and one random candidate (or all of them)."""
    d = df[df["has_ref"]] if ref_only else df
    idx = []
    for _, g in d.groupby("unit", sort=True):
        reqs = g["request_id"].unique()
        for r in (reqs if per_request else [rng.choice(reqs)]):
            gg = g[g["request_id"] == r]
            idx.extend(gg.index if all_candidates else [rng.choice(gg.index)])
    return pd.Index(idx)


def budgets(df: pd.DataFrame, pct: float) -> dict:
    """Budget per meter and request type: a percentile of the type's U_fit session cost."""
    f = df[df["split"] == "fit"]
    out = {}
    for k in DIMS:
        g = f.groupby("rtype")[f"S_{k}"].quantile(pct / 100).to_dict()
        allq = float(f[f"S_{k}"].quantile(pct / 100))
        out[k] = {t: g.get(t, allq) for t in df["rtype"].unique()}
    return out


# ------------------------------------------------------------------------------------------------ table
def _put(df, name, M, rows=None):
    for i, k in enumerate(DIMS):
        col = np.full(len(df), np.nan)
        if rows is None:
            col[:] = M[:, i]
        else:
            col[rows] = M[:, i]
        df[f"{name}_{k}"] = col


def rollouts(wm: SessionWorldModel, sessions, rng, M=None):
    est = [wm.estimate(wm.pre_run(s), rng, M) for s in sessions]
    return (np.array([e["S_hat"] for e in est]), np.array([e["money_hat"] for e in est]),
            np.array([e["p_fail"] for e in est]))


def prepare(platform: str, sessions: Sequence[Session], st: Settings, seed: int, cache: Path | None = None,
            variants: Sequence[str] | None = None, selection: bool = True, fit_regions: int | None = None,
            tag: str = "", log=print) -> pd.DataFrame:
    platform = platform_of(platform)
    path = cache / f"{platform}{tag}_a{seed:02d}.pkl" if cache else None
    if path is not None and path.exists():
        return pd.read_pickle(path)
    t0 = time.time()
    rng = np.random.default_rng(10_000 + seed)
    split = split_units(sessions, platform, rng)
    fit_units = sorted(u for u, s in split.items() if s == "fit")
    if fit_regions is not None:
        fit_units = fit_units[:fit_regions]
        for u, s in list(split.items()):
            if s == "fit" and u not in fit_units:
                split[u] = "unused"
    fit = [s for s in sessions if split[s.unit] == "fit"]
    df = session_rows(sessions, get_adapter(platform))
    df["split"] = df["unit"].map(split)
    is_fit = (df["split"] == "fit").values

    ad = get_adapter(platform)
    wm = SessionWorldModel(ad, st, "full", list(BACKBONE_NAMES)).fit(fit)
    Sh, mh, pf = rollouts(wm, sessions, rng)
    _put(df, "pq", Sh)
    df["pq_money"], df["p_fail"] = mh, pf
    df["mt_money"] = mh
    df["mt_time"] = Sh[:, 2]
    t_wm = time.time() - t0

    # error-risk score: fitted on U_fit; out-of-fold scores for the fit rows set the grid and the alpha_Q choice
    score = ErrorRiskScore(st).fit(df[is_fit])
    df["r"] = score.score(df)
    df["r_oof"] = df["r"]
    fu = np.array(fit_units)
    fold = dict(zip(fu, rng.integers(2, size=len(fu))))
    fv = df["unit"].map(fold)
    for f in (0, 1):
        tr, te = is_fit & (fv != f).values, is_fit & (fv == f).values
        if te.any() and tr.any():
            df.loc[te, "r_oof"] = ErrorRiskScore(st).fit(df[tr]).score(df[te])
    df["r_mean"] = ErrorRiskScore(st, kind="mean").fit(df[is_fit]).score(df)
    acc = df[is_fit & df["has_ref"]].assign(w=lambda d: (d["err"].astype(float) > st.tau).astype(float))
    tab = acc.groupby(["rtype", "config"])["w"].mean().to_dict()
    df["train_err"] = [tab.get((t, c), 1.0) for t, c in zip(df["rtype"], df["config"])]

    # baselines
    base = Baselines(wm, st).fit(fit, Sh[is_fit], [s.unit for s in fit])
    for name, M in base.estimate(sessions, Sh).items():
        _put(df, name, M)

    # estimate chosen per meter by out-of-fold pinball loss on U_fit
    if selection:
        oof_wm, oof_b7 = np.full((len(df), 3), np.nan), np.full((len(df), 3), np.nan)
        for f in (0, 1):
            tr = [s for s in fit if fold[s.unit] != f]
            te_idx = np.flatnonzero(is_fit & (fv == f).values)
            if not tr or not len(te_idx):
                continue
            te = [sessions[i] for i in te_idx]
            w2 = SessionWorldModel(get_adapter(platform), st, "full", list(BACKBONE_NAMES)).fit(tr)
            oof_wm[te_idx] = rollouts(w2, te, rng, min(st.n_roll, 300))[0]
            oof_b7[te_idx] = Baselines(w2, st).fit(tr).estimate(te)["b7"]
        ok = np.isfinite(oof_wm).all(1) & np.isfinite(oof_b7).all(1)
        S_fit = df[[f"S_{k}" for k in DIMS]].values[ok]
        use_wm = per_dim_selection(S_fit, oof_wm[ok], oof_b7[ok], 1 - st.alpha0) if ok.any() else np.ones(3, bool)
        sel = np.where(use_wm, Sh, df[[f"b7_{k}" for k in DIMS]].values)
        _put(df, "sel", sel)
        df.attrs["sel_use_wm"] = use_wm.tolist()

    # ablation variants of the world model (rows outside U_fit)
    rows = np.flatnonzero(~is_fit)
    others = [sessions[i] for i in rows]
    for v in (VARIANTS[platform] if variants is None else variants):
        wv = SessionWorldModel(get_adapter(platform), st, v, list(BACKBONE_NAMES)).fit(fit)
        Sv, _, pv = rollouts(wv, others, rng)
        _put(df, f"v_{v}", Sv, rows)
        col = np.full(len(df), np.nan)
        col[rows] = pv
        df[f"pf_{v}"] = col
    df.attrs.update({"platform": platform, "seed": seed, "fit_seconds": t_wm, "seconds": time.time() - t0,
                     "rho": ad.rho_comp, "floors": list(ad.floors)})
    log(f"  [{platform}{tag} a{seed:02d}] {len(df)} sessions, {int(is_fit.sum())} fit, {time.time() - t0:.0f}s")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    return df


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else (None if math.isnan(o) else ("inf" if o > 0 else "-inf"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
