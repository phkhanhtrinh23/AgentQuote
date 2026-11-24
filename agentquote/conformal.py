"""Conformal quotes with joint coverage.

The score of a calibration session is its largest normalised overshoot over the meters,

    z_i = max_k S_k / max(S_hat_k, floor_k),

and one quantile of the scores, z_hat = z_(ceil((n + 1)(1 - alpha))), scales every meter:
Q_k = z_hat * max(S_hat_k, floor_k). For exchangeable calibration and test units, all bounds hold jointly with
probability at least 1 - alpha (and at most 1 - alpha + 1/(n + 1) for distinct scores).

Other quoters share the interface: per-meter Bonferroni factors, an empirical-copula joint quote, a score
standardised per meter, the raw estimate, and money-only / money-and-deadline quotes.
"""
from __future__ import annotations

import math

import numpy as np

METHODS = ("joint", "bonferroni", "copula", "stdscore", "raw", "money", "money_time")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    j = math.ceil((n + 1) * (1 - alpha))
    return float(s[j - 1]) if 1 <= j <= n else math.inf


def coverage_cap(n: int, alpha: float) -> float:
    """Upper bound of joint coverage for n calibration units and distinct scores."""
    return 1 - alpha + 1 / (n + 1)


class ConformalQuoter:
    def __init__(self, method: str = "joint", alpha: float = 0.1, floors=(1e-3, 1e-6, 1.0), rho_comp: float = 0.0):
        assert method in METHODS, method
        self.method, self.alpha = method, alpha
        self.floors = np.asarray(floors, float)
        self.rho = rho_comp
        self.z = 1.0

    # money helpers
    def _money(self, X):
        X = np.asarray(X, float)
        return self.rho * X[..., 0] + X[..., 1]

    def _base(self, S_hat, money_hat=None):
        S_hat = np.atleast_2d(np.asarray(S_hat, float))
        B = np.maximum(S_hat, self.floors)
        if self.method in ("money", "money_time"):
            m = self._money(S_hat) if money_hat is None else np.asarray(money_hat, float)
            m = np.maximum(m, self.rho * self.floors[0] + self.floors[1])
            return B, m
        return B, None

    def scores(self, S, S_hat, money_hat=None):
        S = np.atleast_2d(np.asarray(S, float))
        B, m = self._base(S_hat, money_hat)
        R = S / B
        if self.method == "money":
            return self._money(S) / m
        if self.method == "money_time":
            return np.maximum(self._money(S) / m, R[:, 2])
        if self.method == "stdscore":
            return ((np.log(np.maximum(R, 1e-12)) - self.mu) / self.sd).max(1)
        return R.max(1)

    def fit(self, S, S_hat, money_hat=None) -> "ConformalQuoter":
        S = np.atleast_2d(np.asarray(S, float))
        n, K = S.shape
        if self.method == "raw":
            self.z = 1.0
            return self
        B, _ = self._base(S_hat, money_hat)
        R = S / B
        if self.method == "bonferroni":
            self.zk = np.array([conformal_quantile(R[:, k], self.alpha / K) for k in range(K)])
            return self
        if self.method == "copula":
            self.zk = self._copula(R)
            return self
        if self.method == "stdscore":
            L = np.log(np.maximum(R, 1e-12))
            self.mu, self.sd = np.median(L, 0), np.maximum(L.std(0), 1e-6)
        self.z = conformal_quantile(self.scores(S, S_hat, money_hat), self.alpha)
        return self

    def _copula(self, R):
        """Empirical copula: the smallest common marginal level whose box covers the joint target."""
        n, K = R.shape
        target = math.ceil((n + 1) * (1 - self.alpha)) / n
        if target > 1:
            return np.full(K, math.inf)
        srt = np.sort(R, 0)
        for r in range(1, n + 1):
            t = srt[r - 1]
            if np.mean((R <= t).all(1)) >= target:
                return t
        return srt[-1]

    def quote(self, S_hat, money_hat=None):
        """Per-meter bounds (n x K). Money quoters return their money bound in column 'money' via quote_money."""
        B, _ = self._base(S_hat, money_hat)
        if self.method in ("bonferroni", "copula"):
            return B * self.zk
        if self.method == "stdscore":
            return B * np.exp(self.mu + self.sd * self.z) if np.isfinite(self.z) else np.full_like(B, np.inf)
        return B * self.z

    def quote_money(self, S_hat, money_hat=None):
        _, m = self._base(S_hat, money_hat)
        return m * self.z

    def covered(self, S, S_hat, money_hat=None) -> np.ndarray:
        S = np.atleast_2d(np.asarray(S, float))
        if self.method == "money":
            return self._money(S) <= self.quote_money(S_hat, money_hat)
        if self.method == "money_time":
            Q = self.quote(S_hat, money_hat)
            return (self._money(S) <= self.quote_money(S_hat, money_hat)) & (S[:, 2] <= Q[:, 2])
        return (S <= self.quote(S_hat, money_hat)).all(1)
