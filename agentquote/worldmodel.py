"""Session world model.

State before turn t: s_t = (compiled quantities v per operator class, attempt index j, last outcome y,
input context T_in, exploration calls n_X, data inspected iota, frailty nu). Actions: explore, execute,
and after a failed attempt give up or edit-and-retry (optionally after an inspection). The quoter does not
control the agent, so actions come from fitted policy heads:

  explore    logistic in (backbone, request type, log(1 + n_X), mu * nu)
  give up    logistic in (backbone, j, y_j, nu); otherwise inspect at the backbone's rate, then edit

Transition heads:

  hazard     P[y_j != success] = logit^-1(psi_b + Gamma_{y_{j-1}} + zeta j + nu
                                          + 1[j = 1] (omega' phi(P) + theta iota))
  outcome    flagged first attempts return a flag's outcome at the fitted share; otherwise the backbone's
             outcome frequencies (attempt 1 / later attempts), each count plus one half
  edit       v' = v * exp(g), g drawn from logged consecutive attempts after the same outcome
             (before attempt 1: logged executed-to-drafted ratios given iota)
  context    T_in grows by each drawn output and observation

Cost model: an attempt with outcome y costs f_y * E_{<= u_y}(v_j) times a residual drawn from the n_nb
successful fitted attempts nearest in predicted cost; tokens follow the context at list prices less a
re-read discount fitted per backbone; call latencies are fitted for completed and failed calls.

Every head is fitted on the logged sessions of U_fit only. ``estimate`` rolls the model out n_roll times
from s_0 and returns the per-meter (1 - alpha0) quantile and the share of failed rollouts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .adapters.base import DomainAdapter
from .config import Settings, price_in, price_out
from .types import Session

GRID_U = np.linspace(-2.5, 2.5, 21)
PRIOR_U = -GRID_U ** 2 / 2
VARIANTS = ("full", "noflags", "noprev", "nofrail", "pooled_resid", "reqonly", "success_only", "onecoef")


def ridge_logit(X, y, lam):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if len(y) == 0:
        return np.zeros(X.shape[1])

    def f(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + lam * w @ w / 2, X.T @ (expit(z) - y) + lam * w
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def loglik_bern(z, y):
    return -y * np.logaddexp(0, -z) - (1 - y) * np.logaddexp(0, z)


def fit_frailty(z, y, rows, off=False):
    """nu = sigma * u with u ~ N(0, 1) on a grid. sigma maximises the marginal likelihood; returns the grid
    posterior of every fitted session."""
    if off or not rows:
        return 0.0, [(GRID_U == 0).astype(float) for _ in rows]
    best, lps = (-np.inf, 1.0), None
    for sg in np.linspace(0.1, 2.0, 20):
        L = np.array([[loglik_bern(z[a:b] + sg * u, y[a:b]).sum() for u in GRID_U] for a, b in rows]) + PRIOR_U
        m = L.max(1, keepdims=True)
        ml = float(np.sum(m[:, 0] + np.log(np.exp(L - m).sum(1))))
        if ml > best[0]:
            best, lps = (ml, sg), L
    post = []
    for lp in lps:
        p = np.exp(lp - lp.max())
        post.append(p / p.sum())
    return best[1], post


def _arr(x, default):
    return np.asarray(x if len(x) else default, float)


@dataclass
class PreRun:
    """Pre-run information of one configuration: the compiled draft and the draft's known cost."""
    b: str
    rtype: str
    d: str
    v0: np.ndarray
    z0: np.ndarray
    flags: np.ndarray
    delta: np.ndarray
    draft_out: float


class SessionWorldModel:
    XCAP = 40

    def __init__(self, adapter: DomainAdapter, settings: Settings | None = None, variant: str = "full",
                 backbones: Sequence[str] | None = None, rtypes: Sequence[str] | None = None):
        assert variant in VARIANTS, variant
        self.ad, self.st, self.variant = adapter, settings or Settings(), variant
        self.backbones = list(backbones or [])
        self.rtypes = list(rtypes or [])
        self.Y = list(adapter.outcomes)
        self.nY = len(self.Y)
        self.nF = len(adapter.flag_names)
        self.nC = len(adapter.classes)
        self.upto = np.array([self.nC - 1] + [adapter.upto_index(y) for y in self.Y])  # code 0 = success
        self.is_tlim = np.array([False] + [y in adapter.timeout_outcomes for y in self.Y])

    # ------------------------------------------------------------------------------------- feature rows
    def bi(self, b):
        return self.backbones.index(b)

    @property
    def NB(self):
        return len(self.backbones)

    @property
    def NHZ(self):
        return self.NB + self.nF + 1 + self.nY + 1

    def hz_row(self, b, flags, iota, prev, j):
        f = np.zeros(self.NHZ)
        f[b] = 1
        if j == 1:
            f[self.NB:self.NB + self.nF] = flags
            f[self.NB + self.nF] = float(iota)
        if prev > 0:
            f[self.NB + self.nF + prev] = 1          # prev in 1..nY
        f[-1] = j - 1
        return self.mask_hz(f)

    def mask_hz(self, X):
        if self.variant == "noprev":
            X[..., self.NB + self.nF + 1:self.NB + self.nF + 1 + self.nY] = 0
        if self.variant in ("noflags", "reqonly"):
            X[..., self.NB:self.NB + self.nF] = 0
        return X

    def px_row(self, b, t, k, nu):
        f = np.zeros(self.NB + len(self.rtypes) + 2)
        f[b] = 1
        if t >= 0:
            f[self.NB + t] = 1
        f[-2], f[-1] = math.log1p(k), nu
        return f

    def pg_row(self, b, j, y, nu):
        f = np.zeros(self.NB + 1 + self.nY + 1)
        f[b], f[self.NB] = 1, j - 1
        f[self.NB + y] = 1                            # y in 1..nY
        f[-1] = nu
        return f

    def code(self, outcome: str) -> int:
        return 0 if outcome == "success" else 1 + (self.Y.index(outcome) if outcome in self.Y else self.nY - 1)

    @staticmethod
    def logratio(a, b):
        on = (a > 0) & (b > 0)
        return np.where(on, np.log(np.maximum(a, 1e-300) / np.maximum(b, 1e-300)), 0.0)

    def pre_run(self, s: Session) -> PreRun:
        v0, z0 = self.ad.quantities(s.plan)
        out = s.draft.turn.tok_out if s.draft else 0.0
        return PreRun(s.b, s.rtype, s.d, v0, z0, self.ad.flags(s.plan), s.delta(), float(out))

    @staticmethod
    def first_ctx(s: Session):
        if s.explore:
            return s.explore[0].turn.tok_in
        if s.attempts:
            return s.attempts[0].step.turn.tok_in
        return None

    # ------------------------------------------------------------------------------------- fitting
    def fit(self, sessions: Sequence[Session]) -> "SessionWorldModel":
        ad, st = self.ad, self.st
        sess = [s for s in sessions if s.attempts]
        for s in sess:
            if s.b not in self.backbones:
                self.backbones.append(s.b)
            if s.rtype not in self.rtypes:
                self.rtypes.append(s.rtype)
        self.seen = {s.b for s in sess}
        NB = self.NB
        if self.variant == "onecoef" and hasattr(ad, "single_coef"):
            ad.single_coef = True
        ad.fit(sess)
        pre = [self.pre_run(s) for s in sess]

        # ---- request-features-only variant: the type's median drafted quantities replace the compiled plan
        tmpl = {}
        for p in pre:
            tmpl.setdefault((p.rtype, p.d), []).append(p.v0)
        self.template = {k: np.median(np.array(v), 0) for k, v in tmpl.items()}

        # ---- context: system prompt per backbone, re-read discount per backbone
        sys_b = {b: [] for b in self.backbones}
        for s, p in zip(sess, pre):
            c = self.first_ctx(s)
            if c is not None:
                sys_b[s.b].append(c - p.draft_out)
        allsys = [x for v in sys_b.values() for x in v] or [3000.0]
        self.sys_tok = {b: float(np.median(v)) if v else float(np.median(allsys)) for b, v in sys_b.items()}
        self.kappa = {}
        for b in self.backbones:
            num = den = 0.0
            for s, p in zip(sess, pre):
                if s.b != b:
                    continue
                c0 = self.first_ctx(s)
                for t in self.turns_after_draft(s):
                    num += t.tok_in * price_in(b) + t.tok_out * price_out(b) - t.usd
                    den += max(t.tok_in - c0, 0.0) * price_in(b)
            self.kappa[b] = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0

        # ---- attempt-cost residuals of the adapter on successful attempts, keyed by log predicted cost
        ar, asz = [], []
        f_log = {}
        for s, p in zip(sess, pre):
            for a in s.attempts:
                v, _ = ad.quantities(a.plan) if a.plan else (p.v0, None)
                e = sum(c.comp for c in a.step.calls)
                y = self.code(a.outcome)
                if y == 0:
                    eh = float(ad.price(v, p.z0))
                    if eh > 0 and e > 0:
                        ar.append(math.log(e / eh))
                        asz.append(math.log(eh))
                else:
                    f_log.setdefault(y, []).append((e, float(ad.price(v, p.z0, self.upto[y]))))
        o = np.argsort(asz)
        self.att_resid = _arr(ar, [0.0])
        self.att_sz, self.att_res_sorted = _arr(np.array(asz)[o] if asz else [], [0.0]), _arr(
            np.array(ar)[o] if ar else [], [0.0])
        self.f = np.full(1 + self.nY, 0.3)
        self.f[0] = 1.0
        for y, lst in f_log.items():
            pos = [e > 0 for e, _ in lst]
            rr = [math.log(e / max(h, 1e-12)) for e, h in lst if e > 0 and h > 0]
            self.f[y] = float(np.mean(pos)) * math.exp(np.mean(rr)) if rr else 0.0
        for y in range(1, 1 + self.nY):
            if self.Y[y - 1] in ad.unbilled:
                self.f[y] = 0.0

        # ---- hazard head (ridge at nu = 0) and frailty
        hz_X, hz_y, rows = [], [], []
        for s, p in zip(sess, pre):
            r0 = len(hz_y)
            prev = 0
            for a in s.attempts:
                hz_X.append(self.hz_row(self.bi(s.b), p.flags, a.inspected, prev, a.j))
                y = self.code(a.outcome)
                hz_y.append(float(y != 0))
                prev = y
            rows.append((r0, len(hz_y)))
        hz_X, hz_y = np.array(hz_X), np.array(hz_y)
        self.w = ridge_logit(hz_X, hz_y, st.ridge)
        self.sigma, post = fit_frailty(hz_X @ self.w, hz_y, rows, self.variant == "nofrail")
        self.nu_grid = self.sigma * GRID_U
        nu_hat = [float(pp @ self.nu_grid) for pp in post]
        self.post_b = {b: [] for b in self.backbones}
        for s, pp in zip(sess, post):
            self.post_b[s.b].append(pp)
        self.post_b = {b: np.array(v) if v else np.tile((GRID_U == 0).astype(float), (1, 1))
                       for b, v in self.post_b.items()}

        # ---- outcome given failure
        fl, gen = [], {}
        flag_codes = np.array([self.code(ad.flag_outcome[f]) if f in ad.flag_outcome else -1
                               for f in ad.flag_names])
        self.flag_codes = flag_codes
        for s, p in zip(sess, pre):
            for a in s.attempts:
                y = self.code(a.outcome)
                if y == 0:
                    continue
                if a.j == 1 and (p.flags * (flag_codes > 0)).any():
                    hit = y in set(flag_codes[(p.flags > 0) & (flag_codes > 0)])
                    fl.append(hit)
                    if hit:
                        continue
                gen.setdefault((s.b, a.j == 1), []).append(y)
        self.flag_share = float(np.mean(fl)) if fl else 0.5
        self.gen_p = {}
        for b in self.backbones:
            for j1 in (True, False):
                cnt = np.bincount(gen.get((b, j1), []), minlength=1 + self.nY)[1:] + 0.5
                self.gen_p[(b, j1)] = cnt / cnt.sum()

        # ---- policy heads
        Xp, yp, Xg, yg = [], [], [], []
        ins_share = {b: [] for b in self.backbones}
        ins_n = {b: [] for b in self.backbones}
        ins_steps = {b: [] for b in self.backbones}
        for s, nu in zip(sess, nu_hat):
            b, t = self.bi(s.b), self.rtypes.index(s.rtype)
            nx = len(s.explore)
            for k in range(min(nx, self.XCAP) + 1):
                Xp.append(self.px_row(b, t, k, nu))
                yp.append(float(k < nx))
            for i, a in enumerate(s.attempts):
                y = self.code(a.outcome)
                if y == 0 or a.j >= st.n_max:
                    continue
                last = i == len(s.attempts) - 1
                if last and s.giveup is None:
                    continue                          # stopped by the turn cap, not a decision
                gave = last and s.giveup is not None
                Xg.append(self.pg_row(b, a.j, y, nu))
                yg.append(float(gave))
                if not gave:
                    ins_share[s.b].append(len(a.after) > 0)
                    if a.after:
                        ins_n[s.b].append(len(a.after))
                        ins_steps[s.b] += a.after
        self.wx = ridge_logit(np.array(Xp), np.array(yp), 1.0)
        self.wg = ridge_logit(np.array(Xg) if Xg else np.zeros((0, NB + self.nY + 2)), np.array(yg), 1.0)
        self.ins_share = {b: float(np.mean(v)) if v else 0.0 for b, v in ins_share.items()}
        self.ins_n = {b: _arr(v, [1]) for b, v in ins_n.items()}
        self.ins_e = {b: _arr([sum(c.comp for c in x.calls) for x in v], [0.0]) for b, v in ins_steps.items()}
        self.ins_out = {b: _arr([x.turn.tok_out for x in v], [250.0]) for b, v in ins_steps.items()}
        self.ins_obs = {b: _arr([x.turn.obs_tokens for x in v], [600.0]) for b, v in ins_steps.items()}
        self.ins_lat = {b: _arr([sum(c.latency for c in x.calls) for x in v], [1.5]) for b, v in ins_steps.items()}

        # ---- exploration: call kind, cost, observation and latency per call
        calls = {b: [] for b in self.backbones}
        heavy_r, hl_x, hl_y, light_lat = [], [], [], []
        light_all = []
        for s, p in zip(sess, pre):
            for x in s.explore:
                e = sum(c.comp for c in x.calls)
                lat = sum(c.latency for c in x.calls)
                heavy = any(c.reads_data for c in x.calls)
                calls[s.b].append((heavy, e, x.turn.tok_out, x.turn.obs_tokens, x.turn.latency, lat))
                if not heavy:
                    light_all.append(e)
                    light_lat.append(lat)
        lmed = float(np.median(light_all)) if light_all else 0.0
        for s, p in zip(sess, pre):
            E0 = float(ad.price(p.v0, p.z0))
            for x in s.explore:
                if any(c.reads_data for c in x.calls):
                    e = sum(c.comp for c in x.calls)
                    if E0 > 0:
                        heavy_r.append(math.log(max(e - lmed, 1e-9) / E0))
                    hl_x.append(math.log(max(e, 1e-3)))
                    hl_y.append(math.log(max(sum(c.latency for c in x.calls), 0.05)))
        self.p_heavy = {b: float(np.mean([c[0] for c in v])) if v else 0.1 for b, v in calls.items()}
        self.light_e = {b: _arr([c[1] for c in v if not c[0]], [lmed]) for b, v in calls.items()}
        self.xout = {b: _arr([c[2] for c in v], [250.0]) for b, v in calls.items()}
        allc = [c for v in calls.values() for c in v]
        self.xobs_h = _arr([c[3] for c in allc if c[0]], [900.0])
        self.xobs_l = _arr([c[3] for c in allc if not c[0]], [600.0])
        self.heavy_r = _arr(heavy_r, [math.log(0.08)])
        if len(hl_x) >= 5:
            A = np.c_[np.ones(len(hl_x)), hl_x]
            self.hlat = np.linalg.lstsq(A, np.array(hl_y), rcond=None)[0]
            o = np.argsort(hl_x)
            self.hlat_sz, self.hlat_res = np.array(hl_x)[o], (np.array(hl_y) - A @ self.hlat)[o]
        else:
            self.hlat, self.hlat_sz, self.hlat_res = np.array([0.5, 0.75]), np.zeros(1), np.zeros(1)
        self.light_lat = _arr(light_lat, [1.5])
        self.xturn = {}
        for b, v in calls.items():
            o_, l_ = (np.array([c[2] for c in v]), np.array([c[4] for c in v])) if len(v) >= 3 else \
                (np.array([250.0, 500.0]), np.array([6.0, 11.0]))
            self.xturn[b] = np.linalg.lstsq(np.c_[o_, np.ones(len(o_))], l_, rcond=None)[0]

        # ---- edits, observations, turn and call latencies
        start = {True: [], False: []}
        edits, gout, fout, fturn = {}, [], {b: [] for b in self.backbones}, {b: [] for b in self.backbones}
        lat_x, lat_y, latf_x, latf_y, tlim = [], [], [], [], []
        tok_out = {b: [] for b in self.backbones}
        obs_s, obs_f, turn_pairs = [], [], {b: [] for b in self.backbones}
        for s, p in zip(sess, pre):
            vs = [ad.quantities(a.plan)[0] if a.plan else p.v0 for a in s.attempts]
            start[bool(s.attempts[0].inspected)].append(self.logratio(vs[0], p.v0))
            for k in range(1, len(vs)):
                edits.setdefault(self.code(s.attempts[k - 1].outcome), []).append(self.logratio(vs[k], vs[k - 1]))
            if s.giveup is not None:
                gout.append(s.giveup.tok_out)
            for a in s.attempts:
                y = self.code(a.outcome)
                t = a.step.turn
                tok_out[s.b].append(t.tok_out)
                turn_pairs[s.b].append((t.tok_out, t.latency))
                (obs_s if y == 0 else obs_f).append(t.obs_tokens)
                e = sum(c.comp for c in a.step.calls)
                lat = sum(c.latency for c in a.step.calls)
                if a.final is not None:
                    fout[s.b].append(a.final.tok_out)
                    fturn[s.b].append(a.final.latency)
                if y == 0:
                    lat_x.append(math.log(max(e, 1e-3)))
                    lat_y.append(math.log(max(lat, 0.05)))
                elif self.Y[y - 1] in ad.timeout_outcomes:
                    tlim.append(lat)
                else:
                    latf_x.append(math.log(max(e, 1e-3)))
                    latf_y.append(math.log(max(lat, 0.05)))
        pooled = start[True] + start[False]
        self.start = {kn: np.array(v if len(v) >= 20 else pooled) for kn, v in start.items()}
        self.edits = {y: np.array(v) for y, v in edits.items()}
        self.tlim = _arr(tlim, [300.0])
        self.gout = _arr(gout, [150.0])
        self.fout = {b: _arr(v, [120.0]) for b, v in fout.items()}
        self.fturn = {b: _arr(v, [3.0]) for b, v in fturn.items()}
        self.has_final = any(len(v) for v in fout.values())
        allout = [x for v in tok_out.values() for x in v] or [600.0]
        self.tok_out = {b: _arr(v, allout) for b, v in tok_out.items()}
        self.obs_s, self.obs_f = _arr(obs_s, [250.0]), _arr(obs_f, [500.0])
        self.turn = {}
        allpairs = [x for v in turn_pairs.values() for x in v]
        for b in self.backbones:
            pr = turn_pairs[b] if len(turn_pairs[b]) >= 3 else allpairs
            o_, l_ = np.array(pr).T
            A_ = np.c_[o_, np.ones(len(o_))]
            sol = np.linalg.lstsq(A_, l_, rcond=None)[0]
            self.turn[b] = (sol, np.log(np.maximum(l_, 1e-3)) - np.log(np.maximum(A_ @ sol, 1e-3)))
        lx, ly = _arr(lat_x, [0.0, 1.0]), _arr(lat_y, [0.5, 1.0])
        A = np.c_[np.ones(len(lx)), lx]
        self.lat = np.linalg.lstsq(A, ly, rcond=None)[0]
        self.lat_sd = float(np.std(ly - A @ self.lat))
        o = np.argsort(lx)
        self.lat_sz, self.lat_res = lx[o], (ly - A @ self.lat)[o]
        if len(latf_x) >= 20:
            fx, fy = np.array(latf_x), np.array(latf_y)
            Af = np.c_[np.ones(len(fx)), fx]
            self.latf = np.linalg.lstsq(Af, fy, rcond=None)[0]
            o = np.argsort(fx)
            self.latf_sz, self.latf_res = fx[o], (fy - Af @ self.latf)[o]
        else:
            self.latf, self.latf_sz, self.latf_res = self.lat, self.lat_sz, self.lat_res
        return self

    @staticmethod
    def turns_after_draft(s: Session):
        for x in s.explore:
            yield x.turn
        for a in s.attempts:
            yield a.step.turn
            if a.final is not None:
                yield a.final
            for x in a.after:
                yield x.turn
        if s.giveup is not None:
            yield s.giveup

    # ------------------------------------------------------------------------------------- helpers
    def local_draw(self, keys, res, q, rng):
        n = len(keys)
        k = min(self.st.n_nb, n)
        p = np.searchsorted(keys, q)
        lo = np.clip(p - k // 2, 0, n - k)
        return res[lo + rng.integers(k, size=len(q))]

    def bill(self, ctx, ctx0, b):
        return (ctx - self.kappa[b] * (ctx - ctx0)) * price_in(b)

    def borrow(self, b: str):
        """A backbone held out of fitting borrows the other backbones' fitted heads."""
        i = self.bi(b)                    # pass every backbone to the constructor so its index exists
        o = [x for x in self.backbones if x != b and x in self.seen]
        oi = [self.bi(x) for x in o]
        for w in (self.w, self.wx, self.wg):
            w[i] = np.mean(w[oi])
        for d in (self.kappa, self.p_heavy, self.ins_share, self.sys_tok):
            d[b] = float(np.mean([d[x] for x in o]))
        for d in (self.tok_out, self.light_e, self.xout, self.ins_n, self.ins_e, self.ins_out, self.ins_obs,
                  self.ins_lat, self.fout, self.fturn):
            d[b] = np.concatenate([d[x] for x in o])
        self.post_b[b] = np.concatenate([self.post_b[x] for x in o])
        for j1 in (True, False):
            g = np.mean([self.gen_p[(x, j1)] for x in o], 0)
            self.gen_p[(b, j1)] = g / g.sum()
        self.turn[b], self.xturn[b] = self.turn[o[0]], self.xturn[o[0]]

    def prior_b(self, b):
        p = self.post_b[b].mean(0)
        return p / p.sum()

    def draw_nu(self, b, M, rng, probs=None):
        if self.variant in ("nofrail", "success_only"):
            return np.zeros(M)
        if probs is not None:
            return self.nu_grid[rng.choice(len(GRID_U), M, p=probs)]
        P = self.post_b[b][rng.integers(len(self.post_b[b]), size=M)]
        cum = P.cumsum(1)
        return self.nu_grid[np.minimum((cum < rng.random(M)[:, None]).sum(1), len(GRID_U) - 1)]

    def posterior(self, s: Session, upto_attempt: int) -> np.ndarray:
        """Frailty posterior given the exploration and the first ``upto_attempt`` attempts of a session."""
        b, t = self.bi(s.b), self.rtypes.index(s.rtype) if s.rtype in self.rtypes else -1
        if self.variant == "nofrail":
            return (GRID_U == 0).astype(float)
        lp = np.log(self.prior_b(s.b) + 1e-300)
        nx = len(s.explore)
        for k in range(min(nx, self.XCAP) + 1):
            z0 = self.px_row(b, t, k, 0.0) @ self.wx
            lp = lp + loglik_bern(z0 + self.wx[-1] * self.nu_grid, float(k < nx))
        flags = self.ad.flags(s.plan)
        prev = 0
        for a in s.attempts[:upto_attempt]:
            z0 = self.hz_row(b, flags, a.inspected, prev, a.j) @ self.w
            y = self.code(a.outcome)
            lp = lp + loglik_bern(z0 + self.nu_grid, float(y != 0))
            prev = y
        p = np.exp(lp - lp.max())
        return p / p.sum()

    # ------------------------------------------------------------------------------------- rollouts
    def rollout(self, pr: PreRun, M: int, rng: np.random.Generator, start: dict | None = None):
        """M imagined sessions from s_0 (or from an observed ``start`` state). Returns costs (M x 3),
        success indicators and attempt counts. From ``start`` the costs are the remaining ones."""
        ad, b, st = self.ad, pr.b, self.st
        bi = self.bi(b)
        t = self.rtypes.index(pr.rtype) if pr.rtype in self.rtypes else -1
        flags = pr.flags.copy()
        v0, z = pr.v0, pr.z0
        if self.variant in ("noflags", "reqonly"):
            flags[:] = 0
        if self.variant == "reqonly":
            v0 = self.template.get((pr.rtype, pr.d), v0)
        so = self.variant == "success_only"
        (sl, ic), tres = self.turn[b]
        ctx0 = self.sys_tok[b] + pr.draft_out
        nA = np.zeros(M, int)
        if start is None:
            S = np.tile(pr.delta, (M, 1)).astype(float)
            ctx = np.full(M, float(ctx0))
            nu = self.draw_nu(b, M, rng)
            iota = np.zeros(M, bool)
            if not so:
                E0 = float(ad.price(v0, z))
                act = np.ones(M, bool)
                base = self.px_row(bi, t, 0, 0.0) @ self.wx
                xs, xi = self.xturn[b]
                for k in range(self.XCAP):
                    ia = np.flatnonzero(act)
                    if not len(ia):
                        break
                    go = rng.random(len(ia)) < expit(base + self.wx[-2] * math.log1p(k) + self.wx[-1] * nu[ia])
                    act[ia[~go]] = False
                    g = ia[go]
                    if not len(g):
                        break
                    hv = rng.random(len(g)) < self.p_heavy[b]
                    e = rng.choice(self.light_e[b], len(g)) + hv * np.exp(rng.choice(self.heavy_r, len(g))) * E0
                    out = rng.choice(self.xout[b], len(g))
                    obs = np.where(hv, rng.choice(self.xobs_h, len(g)), rng.choice(self.xobs_l, len(g)))
                    le = np.log(np.maximum(e, 1e-3))
                    lat = np.where(hv, np.exp(self.hlat[0] + self.hlat[1] * le +
                                              self.local_draw(self.hlat_sz, self.hlat_res, le, rng)),
                                   rng.choice(self.light_lat, len(g)))
                    S[g] += np.c_[e, self.bill(ctx[g], ctx0, b) + out * price_out(b), out * xs + xi + lat]
                    ctx[g] += out + obs
                    iota[g] |= hv
            Vq = np.empty((M, self.nC))
            if so:
                Vq[:] = v0
            else:
                for kn in (False, True):
                    m = iota == kn
                    if m.any():
                        pool = self.start[kn]
                        Vq[m] = v0 * np.exp(pool[rng.integers(len(pool), size=m.sum())]) if len(pool) else v0
            j0, prev, alive = 1, np.zeros(M, int), np.ones(M, bool)
        else:
            S = np.zeros((M, 3))
            ctx = np.full(M, float(start["ctx"]))
            ctx0 = start.get("ctx0", ctx0)
            nu = self.draw_nu(b, M, rng, probs=start.get("nu_probs"))
            iota = np.full(M, bool(start.get("iota", False)))
            alive = np.ones(M, bool)
            if start["phase"] == "explored":
                pool = self.start[bool(start.get("iota", False))]
                Vq = v0 * np.exp(pool[rng.integers(len(pool), size=M)]) if len(pool) else np.tile(v0, (M, 1))
                j0, prev = 1, np.zeros(M, int)
            else:                                      # after a failed attempt j
                Vq = np.tile(start["v"], (M, 1)).astype(float)
                j0 = start["j"] + 1
                prev = np.full(M, self.code(start["y"]))
                nA[:] = start["j"]
                self._after_failure(np.arange(M), prev.copy(), start["j"], S, ctx, ctx0, nu, Vq, alive, prev, b, rng)
        succ = np.zeros(M, bool)
        pk = self.NB + self.nF
        for j in range(j0, st.n_max + 1):
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            n = len(idx)
            nA[idx] = j
            y = np.zeros(n, int)
            if not so:
                X = np.zeros((n, self.NHZ))
                X[:, bi] = 1
                if j == 1:
                    X[:, self.NB:pk] = flags
                    X[:, pk] = iota[idx]
                pv = prev[idx]
                X[pv > 0, pk + pv[pv > 0]] = 1
                X[:, -1] = j - 1
                X = self.mask_hz(X)
                fail = rng.random(n) < expit(X @ self.w + nu[idx])
                nf = int(fail.sum())
                if nf:
                    yf = 1 + rng.choice(self.nY, nf, p=self.gen_p[(b, j == 1)])
                    setf = self.flag_codes[(flags > 0) & (self.flag_codes > 0)]
                    if j == 1 and len(setf):
                        fl = rng.random(nf) < self.flag_share
                        yf[fl] = setf[rng.integers(len(setf), size=int(fl.sum()))]
                    y[fail] = yf
            Eh = ad.price(Vq[idx], z)
            Ep = ad.price(Vq[idx], z, self.upto[y])
            if self.variant == "pooled_resid":
                noise = np.exp(rng.choice(self.att_resid, n))
            else:
                noise = np.exp(self.local_draw(self.att_sz, self.att_res_sorted, np.log(np.maximum(Eh, 1e-12)), rng))
            ok = y == 0
            e = np.where(ok, Eh, Ep * self.f[y]) * noise
            out = rng.choice(self.tok_out[b], n)
            obs = np.where(ok, rng.choice(self.obs_s, n), rng.choice(self.obs_f, n))
            tok = self.bill(ctx[idx], ctx0, b) + out * price_out(b)
            turn = (out * sl + ic) * np.exp(rng.choice(tres, n))
            le = np.log(np.maximum(e, 1e-3))
            if self.variant == "pooled_resid":
                lat = np.exp(self.lat[0] + self.lat[1] * le + rng.normal(0, self.lat_sd, n))
            else:
                lat = np.where(ok, np.exp(self.lat[0] + self.lat[1] * le +
                                          self.local_draw(self.lat_sz, self.lat_res, le, rng)),
                               np.exp(self.latf[0] + self.latf[1] * le +
                                      self.local_draw(self.latf_sz, self.latf_res, le, rng)))
            tl = self.is_tlim[y]
            if tl.any():
                lat[tl] = rng.choice(self.tlim, int(tl.sum()))
            S[idx] += np.c_[e, tok, turn + lat]
            ctx[idx] += out + obs
            if self.has_final and ok.any():           # answer turn after a success
                g = idx[ok]
                fo = rng.choice(self.fout[b], len(g))
                S[g, 1] += self.bill(ctx[g], ctx0, b) + fo * price_out(b)
                S[g, 2] += rng.choice(self.fturn[b], len(g))
            succ[idx[ok]] = True
            stop = ok | (j == st.n_max)
            alive[idx[stop]] = False
            cont = idx[~stop]
            if len(cont) and not so:
                self._after_failure(cont, y[~stop], j, S, ctx, ctx0, nu, Vq, alive, prev, b, rng)
        return S, succ, nA

    def _after_failure(self, cont, yc, j, S, ctx, ctx0, nu, Vq, alive, prev, b, rng):
        (sl, ic), tres = self.turn[b]
        bi = self.bi(b)
        G = np.zeros((len(cont), self.NB + self.nY + 2))
        G[:, bi] = 1
        G[:, self.NB] = j - 1
        G[np.arange(len(cont)), self.NB + yc] = 1
        G[:, -1] = nu[cont]
        gu = rng.random(len(cont)) < expit(G @ self.wg)
        gi = cont[gu]
        if len(gi):
            out = rng.choice(self.gout, len(gi))
            S[gi, 1] += self.bill(ctx[gi], ctx0, b) + out * price_out(b)
            S[gi, 2] += (out * sl + ic) * np.exp(rng.choice(tres, len(gi)))
            alive[gi] = False
        ri, yr = cont[~gu], yc[~gu]
        if not len(ri):
            return
        ins = ri[rng.random(len(ri)) < self.ins_share[b]]
        if len(ins):
            ncall = rng.choice(self.ins_n[b], len(ins)).astype(int)
            xs, xi = self.xturn[b]
            for k in range(int(ncall.max())):
                g = ins[ncall > k]
                out = rng.choice(self.ins_out[b], len(g))
                S[g] += np.c_[rng.choice(self.ins_e[b], len(g)), self.bill(ctx[g], ctx0, b) + out * price_out(b),
                              out * xs + xi + rng.choice(self.ins_lat[b], len(g))]
                ctx[g] += out + rng.choice(self.ins_obs[b], len(g))
        for yy in np.unique(yr):
            m = ri[yr == yy]
            pool = self.edits.get(int(yy))
            if pool is not None and len(pool):
                Vq[m] = Vq[m] * np.exp(pool[rng.integers(len(pool), size=len(m))])
        prev[ri] = yr

    # ------------------------------------------------------------------------------------- estimates
    def estimate(self, pr: PreRun, rng: np.random.Generator, M: int | None = None, start: dict | None = None,
                 levels: Sequence[float] | None = None) -> dict:
        M = M or self.st.n_roll
        S, succ, nA = self.rollout(pr, M, rng, start)
        q = 1 - self.st.alpha0
        out = {"S_hat": np.quantile(S, q, axis=0), "p_fail": float(1 - succ.mean()), "mean_N": float(nA.mean()),
               "money_hat": float(np.quantile(self.ad.money(S), q))}
        for lv in levels or ():
            out[f"S_hat@{lv}"] = np.quantile(S, lv, axis=0)
        return out
