"""LLM transition head on logged BigQuery sessions: after each retried failure, the backbone writes k imagined
repairs from the failed query and its error, a free dry run checks each one, and the failing share predicts the
next attempt's failure. Compared by Brier score with the fitted hazard head and the base rate, on databases
held out of fitting.

  python scripts/imagined_repairs.py --sessions data/bigquery/sessions.jsonl --requests data/bigquery/requests.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentquote.adapters import BigQueryAdapter  # noqa: E402
from agentquote.config import BACKBONE_NAMES, Settings  # noqa: E402
from agentquote.types import load_requests, load_sessions  # noqa: E402
from agentquote.worldmodel import SessionWorldModel  # noqa: E402

PROMPT = ("The BigQuery query below failed. Write a corrected query that answers the question.\n\nQuestion: {q}\n\n"
          "Query:\n```sql\n{sql}\n```\n\nError:\n{err}\n\nReply with one SQL code block.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from agentquote.live.agent import extract_code
    from agentquote.live.bq_platform import BigQueryPlatform
    from agentquote.live.llm import LLM
    P = BigQueryPlatform()
    reqs = {r.request_id: r for r in load_requests(a.requests)}
    sess = load_sessions(a.sessions)
    rng = np.random.default_rng(a.seed)
    dbs = sorted({s.unit for s in sess})
    fit_db = set(rng.choice(dbs, len(dbs) // 2, replace=False))
    wm = SessionWorldModel(BigQueryAdapter(), Settings(), "full", list(BACKBONE_NAMES)).fit(
        [s for s in sess if s.unit in fit_db])
    rows = []
    for s in sess:
        if s.unit in fit_db:
            continue
        flags = wm.ad.flags(s.plan)
        for i, att in enumerate(s.attempts[:-1]):
            if att.outcome == "success":
                continue
            nxt = s.attempts[i + 1]
            sql = next((o["feats"].get("sql", "") for o in (att.plan or {}).get("ops", []) if o["cls"] == "query"), "")
            llm = LLM(s.b, 0.7, 1024)
            t0, fails, tok = time.time(), 0, 0
            for _ in range(a.k):
                r = llm.chat([{"role": "user", "content": PROMPT.format(q=reqs[s.request_id].text[:3000], sql=sql,
                                                                        err=att.outcome)}])
                tok += r.tok_in + r.tok_out
                ok, _, _ = P.dry_run(extract_code(r.text))
                fails += not ok
            prev = wm.code(att.outcome)
            z = wm.hz_row(wm.bi(s.b), flags, nxt.inspected, prev, nxt.j) @ wm.w
            rows.append((float(nxt.outcome != "success"), fails / a.k, float(expit(z)), s.unit,
                         (time.time() - t0) / a.k, tok / a.k))
    R = np.array([r[:3] for r in rows], float)
    base = R[:, 0].mean()
    brier = {"llm": float(np.mean((R[:, 1] - R[:, 0]) ** 2)), "fitted": float(np.mean((R[:, 2] - R[:, 0]) ** 2)),
             "base": float(np.mean((base - R[:, 0]) ** 2))}
    print({"n": len(rows), "brier": brier, "sec_per_state": float(np.mean([r[4] for r in rows])),
           "tokens_per_state": float(np.mean([r[5] for r in rows]))})


if __name__ == "__main__":
    main()
