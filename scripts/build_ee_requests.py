"""Build the live Earth Engine request set (benchmarks and templated analyses) and, with Earth Engine access,
its layer references and quality metadata.

  python scripts/build_ee_requests.py --out data/earthengine [--no-ee]

Inputs under data_cache/: cbgb/eval_set.csv, univearth/data.json, naturalearth/countries_110m.geojson and the
Sen1Floods11 hand labels (sen1floods11/LabelHand, sen1floods11/meta.geojson).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentquote.live.ee_requests import compute_layer_references, enrich_quality, load_all  # noqa: E402
from agentquote.types import save_requests  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/earthengine")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ee", action="store_true", help="skip references and quality metadata")
a = ap.parse_args()
reqs = load_all(a.seed)
if not a.no_ee:
    from agentquote.live.ee_platform import EarthEnginePlatform
    ee = EarthEnginePlatform().ee
    compute_layer_references(reqs, ee)
    enrich_quality(reqs, ee)
Path(a.out).mkdir(parents=True, exist_ok=True)
save_requests(Path(a.out) / "requests.jsonl", reqs)
print(f"{len(reqs)} requests, {sum(len(r.candidates) for r in reqs)} sessions to run -> {a.out}/requests.jsonl")
