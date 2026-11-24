"""BigQuery adapter.

An SQL parser compiles the drafted queries into scan, filter, join, aggregate, sort and limit operators; the
free dry run validates each query and estimates its bytes processed, which is the attempt estimator. On-demand
queries bill the bytes processed with a 10 MB minimum. Failed queries are not billed; a rejected result is
billed in full. Flags: a failing dry run, a full scan of an unpartitioned table above 100 GB, SELECT *.
"""
from __future__ import annotations

import numpy as np

from ..config import BQ_MIN_BYTES, USD_PER_BYTE
from ..types import CompiledPlan, Operator, Request
from .base import DomainAdapter

FULL_SCAN = 100 * 2 ** 30


def split_statements(sql: str) -> list[str]:
    try:
        import sqlglot
        return [e.sql(dialect="bigquery") for e in sqlglot.parse(sql, read="bigquery") if e is not None]
    except Exception:
        return [q for q in sql.split(";") if q.strip()]


def _lookup(tables: dict, name: str) -> dict:
    if name in tables:
        return tables[name]
    short = name.split(".")[-1].lower()
    for k, v in tables.items():
        if k.split(".")[-1].lower() == short:
            return v
    return {}


class BigQueryAdapter(DomainAdapter):
    name = "bigquery"
    comp_unit = "bytes billed"
    rho_comp = USD_PER_BYTE
    classes = ("query",)
    outcomes = ("validation", "runtime", "rejected")
    flag_names = ("dry_run_fails", "full_scan_100gb", "select_star")
    flag_outcome = {"dry_run_fails": "validation"}
    unbilled = frozenset({"validation", "runtime"})
    floors = (float(BQ_MIN_BYTES), 1e-6, 1.0)

    def __init__(self, dry_run=None):
        """dry_run: callable(sql) -> (ok, bytes, message); None prices from table metadata."""
        self.dry_run = dry_run

    def compile(self, req: Request, sql: str, method: str = "") -> CompiledPlan:
        import sqlglot
        from sqlglot import exp
        tables = req.meta.get("tables", {})
        ops, flags, uid = [], {f: 0 for f in self.flag_names}, 0
        stmts = split_statements(sql)
        for call_id, q in enumerate(stmts):
            try:
                tree = sqlglot.parse_one(q, read="bigquery")
            except Exception:
                flags["dry_run_fails"] = 1
                continue
            if any(True for _ in tree.find_all(exp.Star)):
                flags["select_star"] = 1
            scanned = 0.0
            for t in tree.find_all(exp.Table):
                name = ".".join(p for p in (t.catalog, t.db, t.name) if p)
                meta = _lookup(tables, name)
                size = float(meta.get("bytes", 0.0))
                scanned += size
                ops.append(Operator(uid, "scan", call_id, {"table": name, "bytes": size}))
                uid += 1
                if size > FULL_SCAN and not meta.get("partitioned") and not tree.find(exp.Where):
                    flags["full_scan_100gb"] = 1
            for cls, node in (("filter", exp.Where), ("join", exp.Join), ("aggregate", exp.Group),
                              ("sort", exp.Order), ("limit", exp.Limit)):
                for _ in tree.find_all(node):
                    ops.append(Operator(uid, cls, call_id, {}))
                    uid += 1
            ok, byt = True, scanned
            if self.dry_run is not None:
                ok, byt, _ = self.dry_run(q)
            if not ok:
                flags["dry_run_fails"] = 1
            ops.append(Operator(uid, "query", call_id, {"sql": q, "volume": float(byt), "dry_ok": bool(ok)}))
            uid += 1
        if not stmts:
            flags["dry_run_fails"] = 1
        return CompiledPlan(ops, flags, {"n_obs": float(sum(o.cls == "scan" for o in ops)),
                                         "quality": {"n_tables": len(tables), "n_cols": req.meta.get("n_cols", 0)}})

    def classify(self, feedback: str) -> str:
        f = (feedback or "").lower()
        if any(k in f for k in ("syntax error", "not found", "unrecognized name", "no matching signature",
                                "invalid", "400")):
            return "validation"
        return "runtime"

    def price(self, V, z, upto=None):
        V = np.asarray(V, float)
        return np.where(V > 0, np.maximum(V, BQ_MIN_BYTES), 0.0).sum(-1)
