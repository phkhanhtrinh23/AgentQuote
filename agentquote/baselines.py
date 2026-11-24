"""Quoting baselines. Each one reads only its pre-run inputs, is fitted on U_fit and is calibrated on U_Q with the
same joint score as AgentQuote.

  B1  clean-run estimate: a template plan priced by volume, plus the draft and one final turn
  B2  B1 scaled per backbone (median session-to-B1 ratio per meter)
  B3  sample-and-extrapolate: the draft run on a small tile, cost scaled by area (Earth Engine)
  B4  per-group quantile: past session cost per request type and backbone
  B5  agent self-estimate: the backbone's stated costs
  B6  token forecaster: quantile regression of session tokens on request features, B1 for compute and time
  B7  learned quantile: gradient-boosted quantile regression on the compiled plan, the adapter's estimate,
      the configuration, summaries of the world model's fits and B1; tuned on U_fit by grouped pinball loss
  B7 stacked on S_hat: the same learner fitted to log S - log S_hat
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .choice import gbm_quantile
from .config import DIMS, Settings, price_in, price_out
from .types import Session
from .worldmodel import SessionWorldModel


def clean_run(wm: SessionWorldModel, s: Session) -> np.ndarray:
    """B1: template plan priced by volume, draft cost, one attempt turn and one final turn at the median sizes."""
    ad, b = wm.ad, s.b
    tmpl = s.extras.get("template_plan")
    v, z = ad.quantities(tmpl) if tmpl else ad.quantities(s.plan)
    e = float(ad.price(v, z))
    ctx0 = wm.sys_tok.get(b, 3000.0) + (s.draft.turn.tok_out if s.draft else 0.0)
    out = float(np.median(wm.tok_out[b]))
    fo = float(np.median(wm.fout[b])) if wm.has_final else 0.0
    obs = float(np.median(wm.obs_s))
    tok = ctx0 * price_in(b) + out * price_out(b) + (ctx0 + out + obs) * price_in(b) * float(wm.has_final) \
        + fo * price_out(b)
    (sl, ic), _ = wm.turn[b]
    lat = math.exp(wm.lat[0] + wm.lat[1] * math.log(max(e, 1e-3)))
    t = out * sl + ic + lat + (float(np.median(wm.fturn[b])) if wm.has_final else 0.0)
    return s.delta() + np.array([e, tok, t])


def plan_features(wm: SessionWorldModel, s: Session) -> np.ndarray:
    ad = wm.ad
    v, z = ad.quantities(s.plan)
    return np.r_[np.log1p(v), z, ad.flags(s.plan), math.log1p(float(ad.price(v, z))),
                 [float(x) for x in ad.quality(s.plan).values()]]


class Baselines:
    def __init__(self, wm: SessionWorldModel, st: Settings):
        self.wm, self.st = wm, st
        self.q = 1 - st.alpha0

    def _cats(self, s: Session):
        wm = self.wm
        return [wm.backbones.index(s.b) if s.b in wm.backbones else -1,
                self.d_codes.get(s.d, -1), self.m_codes.get(s.m, -1),
                wm.rtypes.index(s.rtype) if s.rtype in wm.rtypes else -1]

    def _summary(self, s: Session):
        g = self.summ_b.get(s.b, self.summ_all)
        t = self.summ_t.get(s.rtype, self.summ_all)
        return list(g) + list(t)

    def features(self, s: Session, b1: np.ndarray) -> np.ndarray:
        return np.r_[plan_features(self.wm, s), self._cats(s), self._summary(s), np.log(np.maximum(b1, 1e-9)),
                     np.log(np.maximum(s.delta(), 1e-9))]

    def fit(self, fit: Sequence[Session], S_hat_fit: np.ndarray | None = None, groups: Sequence[str] | None = None):
        wm, st, q = self.wm, self.st, self.q
        rng = np.random.default_rng(st.seed)
        self.d_codes = {d: i for i, d in enumerate(sorted({s.d for s in fit}))}
        self.m_codes = {m: i for i, m in enumerate(sorted({s.m for s in fit}))}
        S = np.array([s.S() for s in fit])
        B1 = np.array([clean_run(wm, s) for s in fit])
        b = np.array([s.b for s in fit])
        self.mult = {x: np.median(S[b == x] / np.maximum(B1[b == x], 1e-12), 0) for x in set(b)}
        self.mult_all = np.median(S / np.maximum(B1, 1e-12), 0)
        # B4: (1 - alpha0) quantile per (type, backbone)
        grp = {}
        for s, v in zip(fit, S):
            grp.setdefault((s.rtype, s.b), []).append(v)
            grp.setdefault(("*", s.b), []).append(v)
        self.b4 = {k: np.quantile(np.array(v), q, axis=0) for k, v in grp.items()}
        # fit summaries read by B7: attempts, failure share and exploration per backbone and per type
        def summ(ss):
            return [np.mean([x.N for x in ss]), np.mean([not x.success for x in ss]), np.mean([len(x.explore) for x in ss])]
        self.summ_all = summ(fit)
        self.summ_b = {x: summ([s for s in fit if s.b == x]) for x in set(b)}
        self.summ_t = {t: summ([s for s in fit if s.rtype == t]) for t in {s.rtype for s in fit}}
        # B6: token forecaster on request features
        Xr = np.array([self.req_features(s) for s in fit])
        self.b6 = gbm_quantile(st, q, st.seed).fit(Xr, np.log(np.maximum(S[:, 1], 1e-9)))
        # B7: tuned quantile GBM per meter
        X = np.array([self.features(s, b1) for s, b1 in zip(fit, B1)])
        groups = np.array(groups if groups is not None else [s.unit for s in fit])
        self.b7 = [self._tuned(X, np.log(np.maximum(S[:, k], 1e-12)), groups, rng) for k in range(len(DIMS))]
        self.b7s = None
        if S_hat_fit is not None:
            Xs = np.c_[X, np.log(np.maximum(S_hat_fit, 1e-12))]
            self.b7s = [self._tuned(Xs, np.log(np.maximum(S[:, k], 1e-12)) - np.log(np.maximum(S_hat_fit[:, k], 1e-12)),
                                    groups, rng) for k in range(len(DIMS))]
        return self

    def req_features(self, s: Session) -> np.ndarray:
        txt = s.extras.get("request_tokens", 0.0)
        return np.r_[self._cats(s), txt, s.draft.turn.tok_out if s.draft else 0.0, len(s.plan.get("ops", []))]

    def _tuned(self, X, y, groups, rng):
        st, q = self.st, self.q
        space = [dict(num_leaves=l, learning_rate=r, min_child_samples=m, n_estimators=n)
                 for l in (7, 15, 31) for r in (0.03, 0.05, 0.1) for m in (10, 20, 40) for n in (100, 200, 400)]
        picks = [space[i] for i in rng.choice(len(space), min(st.b7_trials, len(space)), replace=False)]
        picks.append(dict(num_leaves=st.gbm_leaves, learning_rate=st.gbm_lr, min_child_samples=st.gbm_min_leaf,
                          n_estimators=st.gbm_trees))
        ug = np.unique(groups)
        fold = dict(zip(ug, rng.integers(2, size=len(ug))))
        fv = np.array([fold[g] for g in groups])
        best, bl = picks[-1], math.inf
        if len(np.unique(fv)) == 2:
            for cfg in picks:
                loss = 0.0
                for f in (0, 1):
                    m = gbm_quantile(st, q, 0, **cfg).fit(X[fv != f], y[fv != f])
                    r = y[fv == f] - m.predict(X[fv == f])
                    loss += np.mean(np.maximum(q * r, (q - 1) * r))
                if loss < bl:
                    best, bl = cfg, loss
        return gbm_quantile(st, q, 0, **best).fit(X, y)

    # ------------------------------------------------------------------------------------------ estimates
    def estimate(self, sessions: Sequence[Session], S_hat: np.ndarray | None = None) -> dict[str, np.ndarray]:
        wm = self.wm
        B1 = np.array([clean_run(wm, s) for s in sessions])
        out = {"b1": B1}
        out["b2"] = np.array([b1 * self.mult.get(s.b, self.mult_all) for s, b1 in zip(sessions, B1)])
        b3 = []
        for s, b1 in zip(sessions, B1):
            p = s.extras.get("probe")
            b3.append(s.delta() + np.asarray(p, float) if p is not None else np.full(3, np.nan))
        out["b3"] = np.array(b3)
        out["b4"] = np.array([self.b4.get((s.rtype, s.b), self.b4.get(("*", s.b), b1)) for s, b1 in zip(sessions, B1)])
        b5 = []
        for s, b1 in zip(sessions, B1):
            se = s.extras.get("self_estimate")
            v = np.array([float(se.get(k, np.nan)) for k in DIMS]) if se else np.full(3, np.nan)
            b5.append(np.where(np.isfinite(v) & (v > 0), s.delta() + np.nan_to_num(v), b1))
        out["b5"] = np.array(b5)
        Xr = np.array([self.req_features(s) for s in sessions])
        b6 = B1.copy()
        b6[:, 1] = np.exp(self.b6.predict(Xr))
        out["b6"] = b6
        X = np.array([self.features(s, b1) for s, b1 in zip(sessions, B1)])
        out["b7"] = np.exp(np.c_[[m.predict(X) for m in self.b7]].T)
        if self.b7s is not None and S_hat is not None:
            Xs = np.c_[X, np.log(np.maximum(S_hat, 1e-12))]
            out["b7s"] = S_hat * np.exp(np.c_[[m.predict(Xs) for m in self.b7s]].T)
        return out


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    r = y - pred
    return float(np.mean(np.maximum(q * r, (q - 1) * r)))


def per_dim_selection(S_fit: np.ndarray, oof_wm: np.ndarray, oof_b7: np.ndarray, q: float) -> np.ndarray:
    """True where the world model's out-of-fold pinball loss on log cost is lower than B7's."""
    L = lambda a: np.log(np.maximum(a, 1e-12))  # noqa: E731
    return np.array([pinball(L(S_fit[:, k]), L(oof_wm[:, k]), q) <= pinball(L(S_fit[:, k]), L(oof_b7[:, k]), q)
                     for k in range(S_fit.shape[1])])


def to_frame(est: dict[str, np.ndarray]) -> pd.DataFrame:
    cols = {}
    for name, M in est.items():
        for i, k in enumerate(DIMS):
            cols[f"{name}_{k}"] = M[:, i]
    return pd.DataFrame(cols)
