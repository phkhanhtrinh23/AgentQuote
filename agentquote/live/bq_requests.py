"""Spider 2.0-Lite, BigQuery part: 205 questions over 70 public databases.

Each question names its database, so the candidates differ by backbone only; correctness is the benchmark's
execution match against the gold result (ported from spider2-lite/evaluation_suite/evaluate.py).
Data: the Spider2 repository files go under data_cache/spider2/spider2-lite.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..config import BACKBONE_NAMES, ROOT
from ..types import Request, config_key

SPIDER = ROOT / "data_cache" / "spider2" / "spider2-lite"
BACKBONES = BACKBONE_NAMES
csv.field_size_limit(10 ** 9)


def _ddl_tables(db: str) -> dict:
    """{full table name: {columns: [(name, type)], ddl}} from the repo's DDL.csv files (free metadata)."""
    out = {}
    for f in sorted((SPIDER / "resource" / "databases" / "bigquery" / db).glob("*/DDL.csv")):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                ddl = row.get("ddl", "")
                m = re.search(r"CREATE (?:OR REPLACE )?(?:TABLE|VIEW) `([^`]+)`", ddl)
                name = m.group(1) if m else f"{f.parent.name}.{row['table_name']}"
                cols = re.findall(r"^\s{2}(\w+) (.+?),?\s*$", ddl, re.M)
                out[name] = {"columns": [(c, t.strip()) for c, t in cols], "ddl": ddl}
    return out


def schema_text(tables: dict, max_cols: int = 60) -> str:
    """Catalogue text; families of date-sharded tables (events_2021...) are collapsed into one entry."""
    groups = defaultdict(list)
    for name in tables:
        groups[re.sub(r"\d{6,8}$", "*", name)].append(name)
    lines = []
    for g, names in groups.items():
        t = tables[names[0]]
        cols = ", ".join(f"{c} {ty}" for c, ty in t["columns"][:max_cols])
        size = sum(tables[n].get("bytes", 0) for n in names)
        extra = f" [{len(names)} shards: {min(names).split('.')[-1]} .. {max(names).split('.')[-1]}]" if len(names) > 1 else ""
        sz = f", {size / 2 ** 30:.2f} GiB" if size else ""
        part = ", partitioned" if t.get("partitioned") else ""
        lines.append(f"TABLE `{g}`{extra}{sz}{part}: {cols}")
    return "\n".join(lines)


def load_requests(backbones=BACKBONES) -> list[Request]:
    rows = [json.loads(l) for l in open(SPIDER / "spider2-lite.jsonl")]
    rows = [r for r in rows if r["instance_id"].startswith(("bq", "ga"))]
    ev = {json.loads(l)["instance_id"]: json.loads(l) for l in open(SPIDER / "evaluation_suite/gold/spider2lite_eval.jsonl")}
    gold_dir = SPIDER / "evaluation_suite" / "gold" / "exec_result"
    cache: dict[str, dict] = {}
    out = []
    for r in rows:
        iid, db = r["instance_id"], r["db"]
        if db not in cache:
            cache[db] = _ddl_tables(db)
        golds = sorted(str(p) for p in gold_dir.glob(f"{iid}*.csv") if re.fullmatch(rf"{iid}(_[a-z])?\.csv", p.name))
        doc = r.get("external_knowledge")
        text = r["question"]
        if doc and (SPIDER / "resource" / "documents" / doc).exists():
            text += "\n\nExternal knowledge:\n" + (SPIDER / "resource" / "documents" / doc).read_text()[:6000]
        e = ev.get(iid, {})
        out.append(Request(iid, db, "sql", text, "bigquery", [config_key(b, db, "") for b in backbones],
                           golds or None, "gold" if golds else "", "table",
                           {"db": db, "tables": cache[db], "condition_cols": e.get("condition_cols", []),
                            "ignore_order": e.get("ignore_order", True)}))
    return out


def enrich_with_bigquery(reqs: list[Request], client) -> None:
    """Fill table sizes / partitioning from free table metadata (client.get_table)."""
    seen: dict[str, dict] = {}
    for r in reqs:
        for name, t in r.meta["tables"].items():
            if name not in seen:
                try:
                    tb = client.get_table(name)
                    seen[name] = {"bytes": int(tb.num_bytes or 0), "rows": int(tb.num_rows or 0),
                                  "partitioned": bool(tb.time_partitioning or tb.range_partitioning),
                                  "clustered": bool(tb.clustering_fields)}
                except Exception as e:
                    seen[name] = {"bytes": 0, "error": str(e)[:100]}
            t.update(seen[name])
        r.meta["schema_text"] = schema_text(r.meta["tables"])


# ------------------------------------------------------------------ execution match (Spider 2.0 evaluator)
def _vectors_match(v1, v2, tol=1e-2, ignore_order=False):
    norm = lambda x: 0 if pd.isna(x) else x  # noqa: E731
    v1, v2 = [norm(x) for x in v1], [norm(x) for x in v2]
    if ignore_order:
        key = lambda x: (x is None, str(x), isinstance(x, (int, float)))  # noqa: E731
        v1, v2 = sorted(v1, key=key), sorted(v2, key=key)
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True


def compare_table(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols=None, ignore_order=True) -> bool:
    gold_cols = gold.iloc[:, condition_cols] if condition_cols else gold
    preds = pred.transpose().values.tolist()
    return all(any(_vectors_match(g, p, ignore_order=ignore_order) for p in preds)
               for g in gold_cols.transpose().values.tolist())


def answer_error(req: Request, pred: pd.DataFrame | None) -> float:
    """epsilon = 1[not execution match]; a failed session has epsilon = inf."""
    if pred is None:
        return float("inf")
    cc = req.meta.get("condition_cols") or []
    golds = req.reference or []
    for i, g in enumerate(golds):
        c = cc[i] if cc and isinstance(cc[0], list) and i < len(cc) else cc
        try:
            if compare_table(pred, pd.read_csv(g), c if c else None, req.meta.get("ignore_order", True)):
                return 0.0
        except Exception:
            continue
    return 1.0
