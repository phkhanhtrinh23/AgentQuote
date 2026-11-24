"""BigQuery executor. Needs ``google-cloud-bigquery`` and credentials with a billing project.

Every job carries the session's label and runs with the cache off; bytes billed come from the job metadata.
Failed queries are not billed. A preview that reads a table (not INFORMATION_SCHEMA) is a data inspection.
"""
from __future__ import annotations

import re
import time

from ..adapters.bigquery import BigQueryAdapter, split_statements
from ..config import env
from ..types import Call, Request
from .platform import Platform, RunResult


def _label(tag: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", tag.lower())[:63]


class BigQueryPlatform(Platform):
    name = "bigquery"
    language = "sql"

    def __init__(self, project: str | None = None, max_rows: int = 50, timeout: float = 300):
        from google.cloud import bigquery
        self.bq = bigquery
        self.client = bigquery.Client(project=project or env("GOOGLE_CLOUD_PROJECT"))
        self.max_rows, self.timeout = max_rows, timeout
        self.adapter = BigQueryAdapter(dry_run=self.dry_run)

    def dry_run(self, sql: str):
        """Free validation and bytes estimate."""
        try:
            job = self.client.query(sql, job_config=self.bq.QueryJobConfig(dry_run=True, use_query_cache=False))
            return True, int(job.total_bytes_processed or 0), ""
        except Exception as e:
            return False, 0, str(e)

    def catalogue(self, req: Request, data_option: str):
        if req.meta.get("schema_text"):
            return req.meta["schema_text"], []
        lines = []
        for name, m in req.meta.get("tables", {}).items():
            cols = ", ".join(f"{c} {t}" for c, t in m.get("columns", [])[:80])
            lines.append(f"TABLE `{name}` ({m.get('bytes', 0) / 2 ** 30:.2f} GiB"
                         f"{', partitioned' if m.get('partitioned') else ''}): {cols}")
        return "\n".join(lines), []

    def _execute(self, req: Request, sql: str, tag: str, kind: str) -> RunResult:
        calls, last = [], None
        for q in split_statements(sql) or [sql]:
            cfg = self.bq.QueryJobConfig(use_query_cache=False, labels={"agentquote_session": _label(tag)})
            t0 = time.time()
            reads = kind == "inspect" and "information_schema" not in q.lower()
            try:
                job = self.client.query(q, job_config=cfg)
                last = job.result(timeout=self.timeout).to_dataframe()
                calls.append(Call(kind, float(job.total_bytes_billed or 0), time.time() - t0, reads,
                                  {"job_id": job.job_id}))
            except Exception as e:
                calls.append(Call(kind, 0.0, time.time() - t0, reads, {"error": str(e)[:200]}))
                y = self.adapter.classify(str(e))
                return RunResult(False, f"ERROR: {e}", calls, self._plan(req, sql, calls) if kind == "run" else None,
                                 outcome=y)
        text = last.head(self.max_rows).to_csv(index=False) if last is not None else ""
        return RunResult(True, text, calls, self._plan(req, sql, calls) if kind == "run" else None, last)

    def _plan(self, req, sql, calls):
        p = self.adapter.compile(req, sql)
        for o, c in zip([o for o in p.ops if o.cls == "query"], calls):
            o.feats["metered"] = c.comp
        return p.to_dict()

    def run(self, req, code, tag):
        return self._execute(req, code, tag, "run")

    def inspect(self, req, code, tag):
        return self._execute(req, code, tag, "inspect")

    def system_prompt(self, req: Request, config: str) -> str:
        return ("You are a data analyst answering questions with BigQuery Standard SQL on the database "
                f"`{req.meta.get('db', '')}`. Queries are billed by bytes processed. Use `inspect` for schema "
                "lookups or small previews and `run` for the query that produces the answer. When the result "
                "answers the question, call `answer` with a short summary; the last successful `run` result is "
                "taken as the answer table.")

    def draft_prompt(self, req: Request, data_option: str, catalogue: str) -> str:
        return (f"Question: {req.text}\nDatabase: {req.meta.get('db', data_option)}\n\nSchema:\n{catalogue}\n\n"
                "Write the complete BigQuery Standard SQL that answers the question.")
