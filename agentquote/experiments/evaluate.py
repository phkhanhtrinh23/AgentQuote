"""Quote quality and choice evaluation on one assignment's estimate table."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..choice import (add_net_budget, calibrate_threshold, choose, losses, money, n_min_floor, overrun,
                      plugin_threshold, threshold_grid)
from ..conformal import ConformalQuoter
from ..config import DIMS, Settings
from .harness import budgets, one_per_unit

D = ("c", "t", "w")


@dataclass
class QSpec:
    name: str
    est: str                 # column prefix of the estimate
    method: str = "joint"    # conformal method
    target: str = "S"        # calibration target: S (whole session) or Alast (draft + last attempt)
    per_question: bool = False


def M(df, prefix):
    return df[[f"{prefix}_{k}" for k in DIMS]].values.astype(float)


def make_quoter(df, spec: QSpec, alpha: float, floors, rho, cal_idx):
    c = df.loc[cal_idx]
    q = ConformalQuoter(spec.method, alpha, floors, rho)
    mh = c[f"{spec.est}_money"].values if spec.method in ("money", "money_time") else None
    return q.fit(M(c, spec.target), M(c, spec.est) if f"{spec.est}_comp" in c else _mt(c, spec), mh)


def _mt(c, spec):
    """Money-type estimates: only money (and time) quantiles exist; fill the per-meter slots for the time check."""
    X = np.zeros((len(c), 3))
    if f"{spec.est}_time" in c:
        X[:, 2] = c[f"{spec.est}_time"].values
    return X


def quote_frame(df, spec: QSpec, quoter: ConformalQuoter, idx, out="Q"):
    d = df.loc[idx]
    est = M(d, spec.est) if f"{spec.est}_comp" in d else _mt(d, spec)
    mh = d[f"{spec.est}_money"].values if spec.method in ("money", "money_time") else None
    res = {}
    if spec.method == "money":
        res[f"{out}_money"] = quoter.quote_money(est, mh)
    elif spec.method == "money_time":
        res[f"{out}_money"] = quoter.quote_money(est, mh)
        res[f"{out}_time"] = quoter.quote(est, mh)[:, 2]
    else:
        Q = quoter.quote(est, mh)
        for i, k in enumerate(DIMS):
            res[f"{out}_{k}"] = Q[:, i]
    return pd.DataFrame(res, index=idx)


# ------------------------------------------------------------------------------------------------ quotes
def eval_quotes(df: pd.DataFrame, specs, st: Settings, rng, alpha=None, test_idx=None, cal_idx=None) -> dict:
    alpha = st.alpha if alpha is None else alpha
    floors, rho = df.attrs["floors"], df.attrs["rho"]
    quote_rows = df[df["split"] == "quote"]
    test_rows = df[df["split"] == "test"]
    cal = cal_idx if cal_idx is not None else one_per_unit(quote_rows, rng)
    cal_q = one_per_unit(quote_rows, rng, per_request=True)
    test = test_idx if test_idx is not None else one_per_unit(test_rows, rng)
    T = df.loc[test]
    S = M(T, "S")
    out, Qs = {}, {}
    for sp in specs:
        if not np.isfinite(M(df.loc[cal], sp.est)).all() and f"{sp.est}_comp" in df:
            continue
        q = make_quoter(df, sp, alpha, floors, rho, cal_q if sp.per_question else cal)
        Qf = quote_frame(df, sp, q, test)
        cov = q.covered(S, M(T, sp.est) if f"{sp.est}_comp" in T else _mt(T, sp),
                        T[f"{sp.est}_money"].values if sp.method in ("money", "money_time") else None)
        r = {"cov_j": 100 * cov.mean(), "n_cal": len(cal_q if sp.per_question else cal), "n_test": len(test),
             "factor": float(getattr(q, "z", np.nan))}
        if sp.method in ("money", "money_time"):
            sm = rho * S[:, 0] + S[:, 1]
            r["wid_money"] = float(np.median(Qf["Q_money"].values / np.maximum(sm, 1e-12)))
        else:
            Q = M(Qf, "Q")
            for i, k in enumerate(D):
                r[f"cov_{k}"] = 100 * float((S[:, i] <= Q[:, i]).mean())
                r[f"wid_{k}"] = float(np.median(Q[:, i] / np.maximum(S[:, i], 1e-12)))
            over = (S / np.maximum(Q, 1e-300)).max(1)
            r["miss"] = float(np.median(over[~cov])) if (~cov).any() else 1.0
            r["covtype"] = {t: 100 * float(cov[(T["rtype"] == t).values].mean()) for t in T["rtype"].unique()}
            fl = _flagged(T)
            r["cov_flagged"] = 100 * float(cov[fl].mean()) if fl.any() else float("nan")
            r["flag_share"] = 100 * float(fl.mean())
            Qs[sp.name] = Q
        out[sp.name] = r
    return {"metrics": out, "Q": Qs, "test": test, "S": S}


def _flagged(T):
    cols = [c for c in T.columns if c.startswith("flag_")]
    return T[cols].any(axis=1).values if cols else np.zeros(len(T), bool)


def paired_reduction(Qa, Qb, units, rng, B=500):
    """Width reduction of a against b in %: 1 - exp(mean log(Qa / Qb)), with a unit bootstrap interval."""
    with np.errstate(all="ignore"):
        L = np.log(np.maximum(Qa, 1e-300) / np.maximum(Qb, 1e-300))
    ok = np.isfinite(L).all(1)
    L, units = L[ok], np.asarray(units)[ok]
    if not len(L):
        nan = np.full(Qa.shape[1], np.nan)
        return nan, nan, nan
    est = 100 * (1 - np.exp(L.mean(0)))
    uu = np.unique(units)
    pos = {u: np.flatnonzero(units == u) for u in uu}
    bs = []
    for _ in range(B):
        idx = np.concatenate([pos[u] for u in rng.choice(uu, len(uu))])
        bs.append(100 * (1 - np.exp(L[idx].mean(0))))
    bs = np.array(bs)
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


# ------------------------------------------------------------------------------------------------ choice
def candidates(df, idx, Qframe, score_col, B):
    d = df.loc[idx].copy()
    for c in Qframe.columns:
        d[c] = Qframe[c].values
    d["r"] = d[score_col].values
    for k in DIMS:
        d[f"B_{k}"] = [B[k].get(t, math.inf) for t in d["rtype"]]
    return add_net_budget(d)


def choice_metrics(pick: pd.DataFrame, cands: pd.DataFrame, lam: float, st: Settings, rho: float, Q="Q") -> dict:
    n = len(pick)
    ans = pick["answered"].values
    err = pick["err"].astype(float).values
    wrong = ans & (err > st.tau)
    ov = ans & overrun(pick, Q, rho) & ~wrong
    loss = losses(pick, st.tau, Q, rho)
    passes = cands.assign(_p=cands["r"].values <= lam).groupby("request_id")["_p"].any()
    passes = passes.loc[pick["request_id"]].values
    sm = money(pick, "S", rho)
    out = {"risk": 100 * loss.mean(), "wrong": 100 * wrong.mean(), "overrun": 100 * ov.mean(),
           "cond": 100 * loss.sum() / max(ans.sum(), 1), "adm": 100 * ans.mean(),
           "decl_r": 100 * float((~passes).mean()), "decl_b": 100 * float((passes & ~ans).mean()),
           "cents": 100 * float(sm[ans].mean()) if ans.any() else float("nan"), "n": n, "lam": lam}
    for kind, key in (("gold", "gold"), ("layer", "layer")):
        m = (pick["ref_kind"] == kind).values
        out[key] = 100 * float(loss[m].mean()) if m.any() else float("nan")
    out["ov_chosen"] = 100 * float((ans & overrun(pick, Q, rho)).sum() / max(ans.sum(), 1))
    return out


def breaks(pick, rho):
    ans = pick["answered"].values
    br = {k: ans & (pick[f"S_{k}"].values > pick[f"NB_{k}"].values) for k in DIMS}
    anyb = np.any([br[k] for k in DIMS], axis=0)
    inb = ans & ~anyb
    out = {"within": 100 * inb.mean(), "brk_any": 100 * anyb.sum() / max(ans.sum(), 1)}
    for k, c in zip(DIMS, D):
        out[f"brk_{c}"] = 100 * br[k].sum() / max(ans.sum(), 1)
    sm = money(pick, "S", rho)
    out["cents"] = 100 * float(sm[ans].mean()) if ans.any() else float("nan")
    return out


class ChoiceEval:
    """Quotes at alpha_Q on U_Q (reference-bearing), the threshold on U_lambda, evaluation on the test units."""

    def __init__(self, df: pd.DataFrame, st: Settings, rng, budget_pct: float | None = None):
        self.df, self.st, self.rng = df, st, rng
        self.rho, self.floors = df.attrs["rho"], df.attrs["floors"]
        self.B = budgets(df, st.budget_pct if budget_pct is None else budget_pct)
        q = df[df["split"] == "quote"]
        self.cal = one_per_unit(q, rng, ref_only=True)
        self.lam_idx = one_per_unit(df[df["split"] == "choice"], rng, ref_only=True, all_candidates=True)
        self.test_idx = one_per_unit(df[df["split"] == "test"], rng, ref_only=True, all_candidates=True)
        fit = df[(df["split"] == "fit") & df["has_ref"]]
        self.grid = threshold_grid(fit["r_oof"].values, st.H)
        self.fit_adm = {}

    def quotes(self, spec: QSpec, aq: float, idx):
        q = make_quoter(self.df, spec, aq, self.floors, self.rho, self.cal)
        return quote_frame(self.df, spec, q, idx)

    def run(self, spec: QSpec, aq: float | None = None, score="r", rule="agentquote", delta=None, mode="binom",
            cost=None, lam_override=None) -> dict:
        st, rho = self.st, self.rho
        aq = st.delta / 2 if aq is None else aq
        delta = st.delta if delta is None else delta
        cl = candidates(self.df, self.lam_idx, self.quotes(spec, aq, self.lam_idx), score, self.B)
        ct = candidates(self.df, self.test_idx, self.quotes(spec, aq, self.test_idx), score, self.B)
        grid = self.grid if score == "r" else threshold_grid(self.df.loc[self.df["split"] == "fit", score].values, st.H)
        kw = {"cost": cost} if cost else {}
        info = {}
        if lam_override is not None:
            lam = lam_override
        elif rule == "plugin":
            lam = plugin_threshold(cl, grid, st, rho, **kw)
        elif mode == "cond":
            keep = self._cond_keep(grid, delta, len(self.lam_idx.unique()))
            res = calibrate_threshold(cl, grid, st, rho, mode="cond", keep=keep, delta=delta, **kw)
            lam, info = res["lam"], res
        else:
            M_pool = len(cl["request_id"].unique()) + len(ct["request_id"].unique())
            res = calibrate_threshold(cl, grid, st, rho, mode=mode, M=M_pool, delta=delta, **kw)
            lam, info = res["lam"], res
        pick = choose(ct, lam, rho, **kw)
        out = choice_metrics(pick, ct, lam, st, rho)
        out["certified"] = bool(np.isfinite(lam) and lam > -math.inf) if lam_override is None else None
        out.update(breaks(pick, rho))
        out["_pick"] = pick
        return out

    def _cond_keep(self, grid, delta, n_lam):
        fit = self.df[(self.df["split"] == "fit") & self.df["has_ref"]]
        n_min = n_min_floor(delta, self.st.eta)
        share = np.array([(fit.groupby("request_id")["r_oof"].min() <= g).mean() for g in grid])
        return share * n_lam >= n_min

    def money_rule(self, spec: QSpec, aq=None, lam=None):
        """Quotes on money (and deadline) only, with AgentQuote's threshold."""
        st, rho = self.st, self.rho
        aq = st.delta / 2 if aq is None else aq
        ct = candidates(self.df, self.test_idx, self.quotes(spec, aq, self.test_idx), "r", self.B)
        if lam is None:
            cl = candidates(self.df, self.lam_idx, self.quotes(spec, aq, self.lam_idx), "r", self.B)
            lam = calibrate_threshold(cl, self.grid, st, rho)["lam"]
        pick = choose(ct, lam, rho)
        out = choice_metrics(pick, ct, lam, st, rho)
        out.update(breaks(pick, rho))
        return out

    def hard_caps(self, lam):
        """Run the candidate with the lowest money estimate under the threshold and abort at any meter's budget."""
        st, rho = self.st, self.rho
        ct = candidates(self.df, self.test_idx, pd.DataFrame(index=self.test_idx), "r", self.B)
        ct = ct.assign(_c=np.where(ct["r"].values <= lam, ct["pq_money"].values, np.inf))
        pick = ct.loc[ct.groupby("request_id")["_c"].idxmin()]
        run = np.isfinite(pick["_c"].values)
        over = np.any([pick[f"S_{k}"].values > pick[f"NB_{k}"].values for k in DIMS], axis=0)
        ok = run & ~over
        err = pick["err"].astype(float).values
        sm = np.minimum(money(pick, "S", rho), rho * pick["NB_comp"].values + pick["NB_tok"].values)
        return {"within": 100 * ok.mean(), "abort": 100 * (run & over).mean(),
                "risk": 100 * (ok & (err > st.tau)).mean(), "brk_any": 0.0, "brk_c": 0.0, "brk_t": 0.0,
                "brk_w": 0.0, "cents": 100 * float(sm[run].mean()) if run.any() else float("nan")}

    def template(self, templates: dict):
        st, rho = self.st, self.rho
        reqs = self.df.loc[self.test_idx, "request_id"].unique()
        rows = [templates.get(r) for r in reqs]
        serve = np.array([t is not None and not t.get("benchmark") for t in rows])
        err = np.array([float(t["err"]) if (t and t.get("err") is not None) else math.inf for t in rows])
        loss = serve & (err > st.tau)
        cents = np.array([100 * (rho * t["comp"] + t["tok"]) if t else np.nan for t in rows])
        return {"risk": 100 * loss.mean(), "adm": 100 * serve.mean(), "risk_served": 100 * loss[serve].mean()
                if serve.any() else float("nan"), "cents": float(np.nanmean(cents[serve])) if serve.any() else float("nan"),
                "cond": 100 * loss.sum() / max(serve.sum(), 1)}

    def oracle(self):
        st, rho = self.st, self.rho
        ct = candidates(self.df, self.test_idx, pd.DataFrame(index=self.test_idx), "r", self.B)
        ok = (ct["err"].astype(float).values <= st.tau) & np.all(
            [ct[f"S_{k}"].values <= ct[f"NB_{k}"].values for k in DIMS], axis=0)
        ct = ct.assign(_c=np.where(ok, money(ct, "S", rho), np.inf))
        pick = ct.loc[ct.groupby("request_id")["_c"].idxmin()]
        ans = np.isfinite(pick["_c"].values)
        return {"risk": 0.0, "adm": 100 * ans.mean(), "cond": 0.0,
                "cents": 100 * float(money(pick, "S", rho)[ans].mean()) if ans.any() else float("nan")}

    def select_alpha_q(self, spec: QSpec, n_pseudo: int = 3) -> float:
        """Share of delta for the quote, chosen on pseudo-splits of U_fit (out-of-fold scores)."""
        st, df, rho = self.st, self.df, self.rho
        fit = df[df["split"] == "fit"]
        units = fit["unit"].unique()
        shares = (st.delta / 4, st.delta / 2, 3 * st.delta / 4)
        res = {a: [] for a in shares}
        for p in range(n_pseudo):
            rng = np.random.default_rng(500 + p)
            perm = rng.permutation(units)
            a_u, b_u, c_u = np.array_split(perm, 3)
            Qr = fit[fit["unit"].isin(a_u)]
            L = fit[fit["unit"].isin(b_u)]
            T = fit[fit["unit"].isin(c_u)]
            cal = one_per_unit(Qr, rng, ref_only=True)
            li = one_per_unit(L, rng, ref_only=True, all_candidates=True)
            ti = one_per_unit(T, rng, ref_only=True, all_candidates=True)
            if not len(cal) or not len(li) or not len(ti):
                continue
            for a in shares:
                q = make_quoter(df, spec, a, self.floors, rho, cal)
                cl = candidates(df, li, quote_frame(df, spec, q, li), "r_oof", self.B)
                ctt = candidates(df, ti, quote_frame(df, spec, q, ti), "r_oof", self.B)
                lam = calibrate_threshold(cl, self.grid, st, rho)["lam"]
                pick = choose(ctt, lam, rho)
                m = choice_metrics(pick, ctt, lam, st, rho)
                res[a].append((m["adm"], m["risk"], m["cents"]))
        if not res[shares[1]]:
            return shares[1]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg = {a: np.nanmean(np.array(v, float), 0) for a, v in res.items() if v}
        ref = agg[shares[1]]
        best = shares[1]
        for a, (adm, risk, cents) in agg.items():
            if a == shares[1] or adm < ref[0] - 2 or risk > 100 * st.delta or not np.isfinite(cents):
                continue
            if cents < agg[best][2] and cents < 0.98 * ref[2]:
                best = a
        return best
