"""Run every candidate of every request once on a live platform and log the metered sessions.

  python scripts/build_ee_requests.py --out data/earthengine
  python scripts/collect_sessions.py --platform earthengine --out data/earthengine --max-usd 50
  python scripts/collect_sessions.py --platform bigquery --out data/bigquery --max-usd 50

Resumable: sessions already in <out>/sessions.jsonl are skipped. Stops when platform plus token spend passes
--max-usd. Each session runs under its own workload tag or job label.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentquote.config import DRAFTER, Settings  # noqa: E402
from agentquote.types import load_jsonl, load_requests, save_requests, save_sessions, split_key  # noqa: E402

SELF_EST = ("Before running anything, estimate what your whole session will consume for this request: platform "
            "compute ({unit}), LLM token spend in USD and wall time in seconds. Reply with JSON "
            '{{"comp": <number>, "tok": <number>, "time": <number>}} only.')


def self_estimate(backbone, req, code, unit) -> dict:
    """The backbone's stated costs (a separate call, not part of the session)."""
    from agentquote.live.llm import LLM
    try:
        r = LLM(backbone, 0.0, 300).chat([{"role": "user", "content": f"Request: {req.text[:3000]}\nDrafted plan:\n"
                                           f"{code[:4000]}\n\n" + SELF_EST.format(unit=unit)}])
        m = re.search(r"\{.*\}", r.text, re.S)
        d = json.loads(m.group()) if m else {}
        return {k: float(d.get(k, 0) or 0) for k in ("comp", "tok", "time")}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["earthengine", "bigquery"], required=True)
    ap.add_argument("--requests", help="requests.jsonl (default: <out>/requests.jsonl, or Spider 2.0-Lite for BigQuery)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--drafter", default=DRAFTER)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = Settings()
    from agentquote.live.agent import DataAgent, Drafter
    if a.platform == "bigquery":
        from agentquote.live.bq_platform import BigQueryPlatform
        from agentquote.live.bq_requests import answer_error, enrich_with_bigquery, load_requests as load_bq
        P = BigQueryPlatform()
        if a.requests or (out / "requests.jsonl").exists():
            reqs = load_requests(a.requests or out / "requests.jsonl")
        else:
            reqs = load_bq()
            enrich_with_bigquery(reqs, P.client)
            save_requests(out / "requests.jsonl", reqs)
    else:
        from agentquote.live.ee_platform import EarthEnginePlatform
        from agentquote.live.ee_requests import answer_error
        P = EarthEnginePlatform()
        reqs = load_requests(a.requests or out / "requests.jsonl")
    ad, rho, unit = P.adapter, P.adapter.rho_comp, P.adapter.comp_unit
    if a.limit:
        reqs = reqs[:a.limit]
    path = out / "sessions.jsonl"
    done = {d["session_id"] for d in load_jsonl(path)} if path.exists() else set()
    drafter = Drafter(P, ad, a.drafter, st)
    spent = 0.0
    for req in reqs:
        todo = [k for k in req.candidates if f"{req.request_id}#{k}" not in done]
        if not todo:
            continue
        drafts = {d: drafter.draft(req, d) for d in sorted({split_key(k)[1] for k in todo})}
        for k in todo:
            b, d, m = split_key(k)
            t0 = time.time()
            s = DataAgent(P, ad, b, st).run_session(req, k, drafts[d], tag=f"aq-{req.request_id}-{b}-{d}-{m}")
            value = s.extras.pop("last_value", None)
            if req.reference is not None:
                s.error = answer_error(req, value if a.platform == "bigquery" else s.answer) if s.success else math.inf
            s.extras.update({"self_estimate": self_estimate(b, req, drafts[d]["code"], unit),
                             "wall_clock": time.time() - t0, "request_tokens": len(req.text) / 4,
                             "continent": req.meta.get("continent", "")})
            save_sessions(path, [s], append=True)
            S = s.S()
            spent += rho * S[0] + S[1]
            print(f"{s.session_id}: N={s.N} {[x.outcome for x in s.attempts]} err={s.error} "
                  f"S=({S[0]:.3g}, ${S[1]:.4f}, {S[2]:.0f}s) total=${spent:.2f}", flush=True)
            if spent > a.max_usd:
                print(f"spend cap {a.max_usd} USD reached")
                return


if __name__ == "__main__":
    main()
