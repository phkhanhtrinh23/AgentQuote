"""Certified choice: error-risk score, choice rule and Learn-then-Test threshold.

For a threshold lambda the rule picks, among candidates with score r(q, o) <= lambda and quote within the net
budget, the one with the lowest remaining money quote, or declines. The loss of a request is
1[answered and (error > tau or some meter overruns its bound)]. For each threshold of a grid fixed on U_fit,
the null R(pi_lambda) > delta has the binomial p-value P[Bin(n, delta) <= sum of losses]; fixed-sequence
testing in grid order returns the largest rejected threshold. A hypergeometric p-value certifies the share of
a finite held-out pool under any dependence among its units; the conditional variant tests the risk among
answered requests, on the thresholds expected to admit at least n_min units.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom

from .config import DIMS, Settings

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


def gbm_classifier(st: Settings, seed: int = 0):
    if lgb is not None:
        return lgb.LGBMClassifier(n_estimators=st.gbm_trees, num_leaves=st.gbm_leaves, learning_rate=st.gbm_lr,
                                  min_child_samples=st.gbm_min_leaf, random_state=seed, verbose=-1,
                                  n_jobs=4)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=st.gbm_trees, max_leaf_nodes=st.gbm_leaves,
                                          learning_rate=st.gbm_lr, min_samples_leaf=st.gbm_min_leaf,
                                          random_state=seed)


def gbm_quantile(st: Settings, alpha: float, seed: int = 0, **over):
    p = dict(n_estimators=st.gbm_trees, num_leaves=st.gbm_leaves, learning_rate=st.gbm_lr,
             min_child_samples=st.gbm_min_leaf)
    p.update(over)
    if lgb is not None:
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=seed, verbose=-1, n_jobs=4, **p)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=p["n_estimators"],
                                         max_leaf_nodes=p["num_leaves"], learning_rate=p["learning_rate"],
                                         min_samples_leaf=p["min_child_samples"], random_state=seed)


# ------------------------------------------------------------------------------------------- error-risk score
class ErrorRiskScore:
    """Estimates P[error > tau] from pre-run features: configuration, quality metadata of the data read,
    usable observations and the world model's session-failure probability."""

    def __init__(self, st: Settings, tau: float | None = None, kind: str = "gbm"):
        self.st, self.tau, self.kind = st, st.tau if tau is None else tau, kind

    def features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in df.columns if c.startswith("q_")] + ["p_fail"]
        X = df[cols].astype(float).fillna(0.0).values
        cats = np.c_[[self.cat["b"].get(x, -1) for x in df["b"]], [self.cat["d"].get(x, -1) for x in df["d"]],
                     [self.cat["m"].get(x, -1) for x in df["m"]], [self.cat["rtype"].get(x, -1) for x in df["rtype"]]]
        return np.c_[cats, X]

    def fit(self, df: pd.DataFrame) -> "ErrorRiskScore":
        d = df[df["has_ref"]]
        y = (d["err"].astype(float) > self.tau).values.astype(int)
        if self.kind == "mean":           # mean error rate of the configuration's data option, per type
            self.table = d.assign(y=y).groupby(["rtype", "d", "m", "b"])["y"].mean().to_dict()
            self.prior = float(y.mean()) if len(y) else 0.5
            return self
        self.cat = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in ("b", "d", "m", "rtype")}
        self.cols = [c for c in df.columns if c.startswith("q_")] + ["p_fail"]
        self.model = gbm_classifier(self.st, self.st.seed)
        if len(np.unique(y)) < 2:
            self.const = float(y.mean()) if len(y) else 0.5
            return self
        self.const = None
        self.model.fit(self.features(d), y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "mean":
            return np.array([self.table.get((r, d, m, b), self.prior)
                             for r, d, m, b in zip(df["rtype"], df["d"], df["m"], df["b"])])
        if self.const is not None:
            return np.full(len(df), self.const)
        return self.model.predict_proba(self.features(df))[:, 1]


# ------------------------------------------------------------------------------------------- rule
def money(df: pd.DataFrame, prefix: str, rho: float) -> np.ndarray:
    return rho * df[f"{prefix}_comp"].values + df[f"{prefix}_tok"].values


def add_net_budget(df: pd.DataFrame) -> pd.DataFrame:
    """B(q, o) = B - Delta_all(q) + Delta(q, o); in wall time Delta_all is the admission time."""
    df = df.copy()
    for k in DIMS:
        df[f"NB_{k}"] = df[f"B_{k}"] - df[f"Dall_{k}"] + df[f"D_{k}"]
    return df


def admissible(df: pd.DataFrame, lam: float, Q: str = "Q", score: str = "r", rho: float = 0.0) -> np.ndarray:
    ok = df[score].values <= lam
    if f"{Q}_comp" not in df:         # money quote (and deadline): checked against the money budget
        ok &= df[f"{Q}_money"].values <= rho * df["NB_comp"].values + df["NB_tok"].values
        if f"{Q}_time" in df:
            ok &= df[f"{Q}_time"].values <= df["NB_time"].values
        return ok
    for k in DIMS:
        ok &= df[f"{Q}_{k}"].values <= df[f"NB_{k}"].values
    return ok


def choose(df: pd.DataFrame, lam: float, rho: float, Q: str = "Q", score: str = "r", cost: str | None = None,
           **kw) -> pd.DataFrame:
    """One row per request: the chosen candidate (lowest remaining money quote among the admissible) or none."""
    d = df.assign(_adm=admissible(df, lam, Q, score, rho=rho, **kw))
    if cost is None:
        qm = d[f"{Q}_money"].values if f"{Q}_money" in d else money(d, Q, rho)
        cost_v = qm - money(d, "D", rho)
    else:
        cost_v = d[cost].values
    d = d.assign(_cost=np.where(d["_adm"], cost_v, np.inf))
    idx = d.groupby("request_id")["_cost"].idxmin()
    pick = d.loc[idx]
    pick = pick.assign(answered=np.isfinite(pick["_cost"].values))
    return pick


def overrun(df: pd.DataFrame, Q: str = "Q", rho: float = 0.0) -> np.ndarray:
    if f"{Q}_comp" not in df and f"{Q}_money" in df:
        o = money(df, "S", rho) > df[f"{Q}_money"].values
        if f"{Q}_time" in df:
            o |= df["S_time"].values > df[f"{Q}_time"].values
        return o
    return np.any([df[f"S_{k}"].values > df[f"{Q}_{k}"].values for k in DIMS], axis=0)


def losses(pick: pd.DataFrame, tau: float, Q: str = "Q", rho: float = 0.0) -> np.ndarray:
    wrong = pick["err"].astype(float).values > tau
    return pick["answered"].values & (wrong | overrun(pick, Q, rho))


# ------------------------------------------------------------------------------------------- Learn-then-Test
def pvalue_binom(k: int, n: int, delta: float) -> float:
    return float(binom.cdf(k, n, delta)) if n > 0 else 1.0


def pvalue_hyper(k: int, n: int, M: int, delta: float) -> float:
    """P[Hyp(M, floor(delta M) + 1, n) <= k]: the null that at least floor(delta M)+1 of M pool units are losses."""
    K = min(int(math.floor(delta * M)) + 1, M)
    return float(hypergeom.cdf(k, M, K, n))


def n_min_floor(delta: float, eta: float, rc: float = 0.10, power: float = 0.80, nmax: int = 2000) -> int:
    """Smallest number of answered units from which a true conditional risk rc is certified with the power."""
    for n in range(1, nmax):
        kstar = int(binom.ppf(eta, n, delta))
        while kstar >= 0 and binom.cdf(kstar, n, delta) > eta:
            kstar -= 1
        if kstar >= 0 and binom.cdf(kstar, n, rc) >= power:
            return n
    return nmax


def threshold_grid(r_fit: np.ndarray, H: int) -> np.ndarray:
    return np.unique(np.quantile(np.asarray(r_fit, float), np.linspace(0, 1, H)))


def ltt(loss: np.ndarray, delta: float, eta: float, mode: str = "binom", answered: np.ndarray | None = None,
        M: int | None = None, keep: np.ndarray | None = None) -> tuple[int, list]:
    """loss: n x H loss matrix in grid order. Returns (index of the largest rejected threshold or -1, p-values)."""
    n, H = loss.shape
    best, pv = -1, []
    for h in range(H):
        if keep is not None and not keep[h]:
            pv.append(np.nan)
            continue
        k = int(loss[:, h].sum())
        if mode == "binom":
            p = pvalue_binom(k, n, delta)
        elif mode == "hyper":
            p = pvalue_hyper(k, n, M or n, delta)
        elif mode == "cond":
            na = int(answered[:, h].sum())
            p = pvalue_binom(k, na, delta) if na > 0 else 1.0
        else:
            raise ValueError(mode)
        pv.append(p)
        if p > eta:
            break
        best = h
    return best, pv


def calibrate_threshold(cands: pd.DataFrame, grid: np.ndarray, st: Settings, rho: float, Q: str = "Q",
                        mode: str = "binom", M: int | None = None, keep=None, delta: float | None = None,
                        **kw) -> dict:
    """cands: every candidate of one reference-bearing request per unit of U_lambda, quoted at alpha_Q."""
    delta = st.delta if delta is None else delta
    reqs = cands["request_id"].unique()
    L = np.zeros((len(reqs), len(grid)), bool)
    A = np.zeros((len(reqs), len(grid)), bool)
    for h, lam in enumerate(grid):
        pick = choose(cands, lam, rho, Q, **kw).set_index("request_id").loc[reqs]
        L[:, h] = losses(pick, st.tau, Q, rho)
        A[:, h] = pick["answered"].values
    h, pv = ltt(L, delta, st.eta, mode, A, M, keep)
    return {"lam": float(grid[h]) if h >= 0 else -math.inf, "h": h, "pvals": pv,
            "cal_risk": L.mean(0), "cal_adm": A.mean(0)}


def plugin_threshold(cands: pd.DataFrame, grid: np.ndarray, st: Settings, rho: float, Q: str = "Q", **kw) -> float:
    """Largest threshold whose calibration risk is at most delta, without correction."""
    reqs = cands["request_id"].unique()
    lam = -math.inf
    for g in grid:
        pick = choose(cands, g, rho, Q, **kw).set_index("request_id").loc[reqs]
        if losses(pick, st.tau, Q, rho).mean() <= st.delta:
            lam = float(g)
    return lam
