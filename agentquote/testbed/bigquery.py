"""BigQuery testbed: text-to-SQL agent sessions over public databases with metered bytes, tokens and time.

The layout follows the BigQuery part of Spider 2.0-Lite: 205 questions over 70 databases, each question naming
its database, so the candidates differ by backbone only. Table sizes follow the tables that gold queries read.
A session explores (schema lookups and previews; a SELECT * preview bills every column in full), runs its
drafted query and repairs it from the feedback. Billing follows the on-demand rule: bytes processed with a
10 MB minimum, failed queries unbilled, rejected results billed in full.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit

from ..config import BACKBONE_NAMES, BQ_MIN_BYTES
from ..types import Attempt, Call, CompiledPlan, Operator, Request, Session, Step, Turn, config_key
from .common import (FRAIL_X_CORR, NB, OUT_ATT, PIN, POUT, REWRITE_P, REWRITE_SD, SYS_TOK, TPS, TURN_SD, bill_in,
                     lognormal)

GB = 2.0 ** 30
PSI = np.array([1.0, 1.8, 1.1])
GIVEUP_SHIFT = 0.3
EXPL_MU = np.array([5.35, 6.0, 5.5])
STAR = np.array([0.85, 0.06, 0.24])
FLAGS = ["dry_run_fails", "full_scan_100gb", "select_star"]
LAT = (1.03, 0.10, 0.71)
PRUNE_P = 0.023
HZ_INTER = 0.5


def latency(q, rng):
    return float(np.exp(LAT[0] + LAT[1] * math.log(q / GB + 1e-3) + rng.normal(0, LAT[2])))


def billed(q, rng):
    """Bytes billed for a query whose dry run reports q bytes (clustered pruning is known only after the run)."""
    g = rng.uniform(0.01, 1.0) if rng.random() < PRUNE_P else 1.0 + abs(rng.normal(0, 0.015))
    return max(q * g, BQ_MIN_BYTES)


def db_question_counts(rng, n_db=70, n_q=205, n_single=32):
    counts = [1] * n_single
    rest = n_db - n_single
    w = rng.pareto(1.2, rest) + 1
    extra = np.floor(w / w.sum() * (n_q - n_single - rest)).astype(int) + 1
    while extra.sum() < n_q - n_single:
        extra[rng.integers(rest)] += 1
    return counts + list(extra)


class BigQueryTestbed:
    platform = "bigquery"

    def __init__(self, seed: int = 0, scale: float = 1.0, n_max: int = 5):
        self.seed, self.n_max = seed, n_max
        self.rng = rng = np.random.default_rng(seed + 7)
        counts = db_question_counts(rng)
        if scale != 1.0:                                  # a random subset of the databases
            keep = rng.permutation(len(counts))[:max(3, int(round(len(counts) * scale)))]
            counts = [counts[i] for i in sorted(keep)]
        self.dbs, self.tasks = [], []
        for d, n in enumerate(counts):
            nt = int(rng.integers(3, 30))
            sizes = np.clip(np.exp(rng.normal(math.log(0.128 * GB), 4.23, nt)), 1e-4 * GB, 171.2 * GB)
            self.dbs.append(dict(sizes=sizes, name=f"db{d:02d}"))
            for _ in range(n):
                touch = rng.choice(nt, size=min(nt, int(rng.integers(1, 5))), replace=False)
                frac = rng.beta(2, 5)
                part = rng.random() < 0.4
                correct = sizes[touch].sum() * frac * (rng.uniform(0.01, 0.3) if part else 1.0)
                draft = correct * float(lognormal(rng, 1, 0.4))
                flags = np.array([rng.random() < 0.18, draft > 100 * GB and not part, rng.random() < 0.15], float)
                self.tasks.append(dict(db=d, touch=touch, draft=max(draft, BQ_MIN_BYTES), flags=flags,
                                       d_out=float(lognormal(rng, 700, 0.3)), z=rng.normal(), zx=rng.normal(),
                                       part=part, n_tables=nt))

    def request(self, t: int) -> Request:
        task = self.tasks[t]
        db = self.dbs[task["db"]]["name"]
        return Request(f"bq-{t:04d}", db, "sql", f"Question {t} over database {db}.", "bigquery",
                       [config_key(b, db, "") for b in BACKBONE_NAMES], "gold", "gold", "table",
                       {"db": db, "n_tables": task["n_tables"]})

    def compiled(self, task, q, metered=None):
        f = {"volume": float(q)}
        if metered is not None:
            f["metered"] = float(metered)
        ops = [Operator(0, "scan", 0, {"bytes": float(self.dbs[task["db"]]["sizes"][task["touch"]].sum())}),
               Operator(1, "query", 0, f)]
        return CompiledPlan(ops, {k: int(x) for k, x in zip(FLAGS, task["flags"])},
                            {"n_obs": float(len(task["touch"])),
                             "quality": {"n_tables": float(task["n_tables"]),
                                         "log_bytes": math.log(max(task["draft"], 1.0)),
                                         "partitioned": float(task["part"])}})

    def _turn(self, kind, ctx, ctx0, out, obs, b, rng, noise=True):
        tok = bill_in(ctx, ctx0, b, rng) + out * POUT[b]
        lat = (out / TPS[b] + 1.0) * (float(lognormal(rng, 1, TURN_SD)) if noise else 1.0)
        return Turn(kind, float(ctx), float(out), float(obs), float(lat), float(tok))

    def run_session(self, t, b, rng=None):
        rng = rng or self.rng
        task = self.tasks[t]
        sizes = self.dbs[task["db"]]["sizes"]
        z = task["z"]
        zx = FRAIL_X_CORR * z + math.sqrt(1 - FRAIL_X_CORR ** 2) * task["zx"]
        d_out = task["d_out"]
        draft = Step(Turn("draft", 4500.0, d_out, 0.0, d_out / TPS[0] + 2.0, 4500 * PIN[0] + d_out * POUT[0]),
                     [Call("lookup", 0.0, 0.8), Call("lookup", 0.0, 0.8)])
        nx = rng.negative_binomial(2, 2 / (2 + EXPL_MU[b] * math.exp(0.3 * zx)))
        ctx0 = SYS_TOK[b] + d_out + float(lognormal(rng, 800, 0.3))
        ctx = ctx0
        explore, known = [], False
        for _ in range(nx):
            if rng.random() < 0.5:
                byt, heavy = float(BQ_MIN_BYTES), False
            else:
                tb = sizes[rng.choice(task["touch"])]
                frac = 1.0 if rng.random() < STAR[b] else float(rng.beta(1.5, 6) * 0.5)
                byt, heavy = max(tb * frac, BQ_MIN_BYTES), True
            out, obs = float(lognormal(rng, 250, 0.5)), float(lognormal(rng, 1200, 0.7))
            turn = self._turn("explore", ctx, ctx0, out, obs, b, rng, noise=False)
            explore.append(Step(turn, [Call("inspect" if heavy else "metadata", byt, latency(byt, rng), heavy)]))
            known = known or not heavy
            ctx += out + obs
        bf = float(lognormal(rng, 1, REWRITE_SD)) if rng.random() < REWRITE_P else 1.0
        flags = task["flags"].copy()
        if known and flags[0] and rng.random() < 0.5:      # the schema shows the invalid column before attempt 1
            flags[0] = 0
            bf *= float(lognormal(rng, 1, 0.2))
        gam = {"start": 0.0, "validation": -0.4, "runtime": 0.3, "rejected": 0.2}
        prev, atts, success, giveup = "start", [], False, None
        for j in range(1, self.n_max + 1):
            logit = PSI[b] + gam[prev] + 0.55 * z + (0.6 if j >= 3 else 0.0)
            if j == 1:
                logit += 3.0 * flags[0] + 0.6 * flags[1] + 0.3 * flags[2] + (0.4 * flags[0] if b == 0 else 0)
            elif task["flags"][1] and prev == "runtime":
                logit += HZ_INTER
            if rng.random() < expit(logit):
                if j == 1 and flags[0] and rng.random() < 0.8:
                    y = "validation"
                else:
                    p = np.array([0.30, 0.12, 0.43])
                    y = ["validation", "runtime", "rejected"][rng.choice(3, p=p / p.sum())]
            else:
                y = "success"
            q = task["draft"] * bf
            e = billed(q, rng) if y in ("success", "rejected") else 0.0
            lat = latency(q, rng) if y in ("success", "rejected", "runtime") else 0.5
            if y == "runtime":
                lat = min(lat * 3, 600)
            out = float(lognormal(rng, OUT_ATT[b] * 0.6, 0.35))
            obs = float(lognormal(rng, 900 if y in ("success", "rejected") else 400, 0.5))
            turn = self._turn("attempt", ctx, ctx0, out, obs, b, rng)
            a = Attempt(j, y, Step(turn, [Call("run", float(e), float(lat))]),
                        self.compiled(task, q, e).to_dict(), known)
            atts.append(a)
            ctx += out + obs
            if y == "success":
                a.final = self._turn("final", ctx, ctx0, float(lognormal(rng, 120, 0.4)), 0.0, b, rng)
                success = True
                break
            if j == self.n_max:
                break
            if rng.random() < expit(-2.4 + GIVEUP_SHIFT + 0.45 * (j - 1) + 0.4 * z + (0.4 if y == "runtime" else 0.0)):
                giveup = self._turn("giveup", ctx, ctx0, float(lognormal(rng, 150, 0.4)), 0.0, b, rng)
                break
            bf *= {"validation": float(lognormal(rng, 1, 0.2)), "runtime": rng.uniform(0.1, 0.6),
                   "rejected": float(lognormal(rng, 1, 0.5))}[y]
            prev = y
        req = self.request(t)
        key = config_key(BACKBONE_NAMES[b], req.meta["db"], "")
        s = Session(f"{req.request_id}#{key}", req.request_id, req.unit, "sql", "bigquery", key, draft,
                    self.compiled(task, task["draft"]).to_dict(), explore, atts, giveup, has_ref=True,
                    ref_kind="gold")
        correct = success and rng.random() < expit(-1.0 + [-0.4, 0.0, 0.3][b] - 0.5 * z)
        s.error = (0.0 if correct else 1.0) if success else math.inf
        s.extras = {"request_tokens": 1500.0}
        if rng.random() > 0.08:                              # the backbone's stated costs, from the plan alone
            s.extras["self_estimate"] = {
                "comp": float(task["draft"] * lognormal(rng, 1.0, 1.0)),
                "tok": float((ctx0 * PIN[b] + OUT_ATT[b] * 0.6 * POUT[b]) * lognormal(rng, 1.5, 0.8)),
                "time": float((OUT_ATT[b] * 0.6 / TPS[b] + 1 + latency(task["draft"], rng)) * lognormal(rng, 1.5, 0.8))}
        return s

    def generate(self, rng=None):
        rng = rng or self.rng
        reqs, sess = [], []
        for t in range(len(self.tasks)):
            reqs.append(self.request(t))
            for b in range(NB):
                sess.append(self.run_session(t, b, rng))
        return reqs, sess
