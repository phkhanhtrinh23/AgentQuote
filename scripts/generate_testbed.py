"""Generate metered agent sessions on the testbed platforms.

  python scripts/generate_testbed.py --out data                 # Earth Engine and BigQuery pools
  python scripts/generate_testbed.py --out data --scale 0.2     # a smaller pool

Writes <out>/<platform>/{requests,sessions}.jsonl, Earth Engine templates, re-executions and own-draft
sessions, an independently drawn second pool (<out>/pool2) and the weaker-adapter pool (<out>/hard).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentquote.testbed import BigQueryTestbed, EarthEngineTestbed  # noqa: E402
from agentquote.types import save_jsonl, save_requests, save_sessions  # noqa: E402


def write(out: Path, reqs, sess):
    out.mkdir(parents=True, exist_ok=True)
    save_requests(out / "requests.jsonl", reqs)
    save_sessions(out / "sessions.jsonl", sess)
    print(f"  {out}: {len(reqs)} requests, {len(sess)} sessions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="pool size relative to the full testbed")
    ap.add_argument("--platform", default="all", choices=["all", "earthengine", "bigquery"])
    ap.add_argument("--no-extra", action="store_true", help="skip the second pool and the weaker-adapter pool")
    a = ap.parse_args()
    out = Path(a.out)
    if a.platform in ("all", "earthengine"):
        tb = EarthEngineTestbed(seed=a.seed, scale=a.scale)
        reqs, sess = tb.generate()
        write(out / "earthengine", reqs, sess)
        rng = np.random.default_rng(a.seed + 99)
        save_jsonl(out / "earthengine" / "templates.jsonl",
                   ({"request_id": tb.request(r).request_id, **tb.template_run(r, rng)} for r in range(len(tb.regions))))
        _, rer = tb.rerun(range(len(tb.regions)), seed=a.seed + 1)
        for s in rer:
            s.session_id += "#rerun"
        save_sessions(out / "earthengine" / "rerun_sessions.jsonl", rer)
        own_regions = rng.choice(len(tb.regions), size=min(30 * 4, len(tb.regions)), replace=False)
        own = []
        for r in own_regions:
            _, ss = tb.generate([int(r)], rng, own_draft=True)
            for s in ss:
                s.session_id += "#own"
            own += ss
        save_sessions(out / "earthengine" / "owndraft_sessions.jsonl", own)
        print(f"  {out / 'earthengine'}: templates, {len(rer)} re-executions, {len(own)} own-draft sessions")
        if not a.no_extra:
            write(out / "pool2" / "earthengine", *EarthEngineTestbed(seed=a.seed + 1000, scale=a.scale).generate())
            write(out / "hard" / "earthengine", *EarthEngineTestbed(seed=a.seed, scale=a.scale, hard_adapter=True).generate())
    if a.platform in ("all", "bigquery"):
        write(out / "bigquery", *BigQueryTestbed(seed=a.seed, scale=a.scale).generate())


if __name__ == "__main__":
    main()
