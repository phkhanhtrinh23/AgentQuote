"""Earth Engine executor. Needs ``earthengine-api`` and a Cloud project registered for Earth Engine.

Metering: every session runs under its own workload tag and every call inside the profiler. A completed call
is charged its profiled EECU-seconds (or the monitored value when larger, if a monitor is attached), a failed
call its profiled EECU-seconds, and an out-of-memory call a fixed timeout. Each call runs at a sub-pixel offset
of the region so no cached result is reused.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import time
import traceback

import numpy as np

from ..adapters.earthengine import EarthEngineAdapter
from ..config import env
from ..types import Call, Request
from .platform import Platform, RunResult

OOM_EECU_S = 300.0
READS_DATA = ("reduceRegion", "reduceRegions", "sample", "getRegion", "toList", "aggregate_", "first()")
PROFILE_CLASS = [
    (r"reduceRegion|reduceRegions|sampleRegions|reduceToVectors|Reducer|sample", "reduce"),
    (r"classif|Classifier|cluster", "inference"),
    (r"median|mean|mosaic|qualityMosaic|Collection\.reduce|normalizedDifference|expression|composite", "composite"),
    (r"updateMask|mask|clip|map", "mask"),
    (r"filter|Filter|limit", "filter"),
    (r"Export|export", "export"),
    (r"load|Asset", "load"),
]


def _cls(algo: str) -> str:
    for pat, c in PROFILE_CLASS:
        if re.search(pat, algo):
            return c
    return "load"


def parse_profile(text: str) -> dict[str, float]:
    """EECU-seconds per operator class from Profile.getProfilesAsText output."""
    out: dict[str, float] = {}
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return out
    head = [h.strip() for h in lines[0].split(",")]
    try:
        ie = next(i for i, h in enumerate(head) if "EECU" in h)
        idd = next(i for i, h in enumerate(head) if "Description" in h or "Algorithm" in h)
    except StopIteration:
        return out
    for line in lines[1:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) <= max(ie, idd):
            continue
        try:
            e = float(p[ie])
        except ValueError:
            continue
        c = _cls(p[idd])
        out[c] = out.get(c, 0.0) + e
    return out


class EarthEnginePlatform(Platform):
    name = "earthengine"
    language = "python"

    def __init__(self, project: str | None = None, monitor=None):
        import ee
        self.ee = ee
        key = env("GOOGLE_APPLICATION_CREDENTIALS")
        if key:
            email = json.load(open(key))["client_email"]
            ee.Initialize(ee.ServiceAccountCredentials(email, key), project=project or env("GOOGLE_CLOUD_PROJECT"))
        else:
            ee.Initialize(project=project or env("GOOGLE_CLOUD_PROJECT"))
        self.monitor = monitor            # optional callable(tag) -> monitored EECU-s of that tag so far
        self.adapter = EarthEngineAdapter(resolver=self.resolve_collection)
        self.rng = np.random.default_rng(0)
        self._lookups: dict = {}

    def _ns(self, req: Request, offset=(0.0, 0.0)) -> dict:
        x0, y0, x1, y1 = req.meta["bbox"]
        dx, dy = offset
        g = self.ee.Geometry.Rectangle([x0 + dx, y0 + dy, x1 + dx, y1 + dy])
        return {"ee": self.ee, "region": g, "roi": g, "aoi": g, "geometry": g,
                "start_date": req.meta.get("start", ""), "end_date": req.meta.get("end", "")}

    def _profiled(self, fn):
        ids = []
        with self.ee.data.profiling(ids.append):
            t0 = time.time()
            val = fn()
            lat = time.time() - t0
        eecu = {}
        if ids:
            try:
                txt = self.ee.data.computeValue(self.ee.ApiFunction.call_("Profile.getProfilesAsText", ids=ids))
                eecu = parse_profile(txt)
            except Exception:
                pass
        return val, lat, eecu

    def resolve_collection(self, req: Request, coll: str, filters: list[str]) -> dict:
        """Metered catalogue lookup: images left after the plan's filters and their footprint. Repeated lookups
        of a draft (one per configuration) reuse the first result and cost nothing."""
        key = (req.request_id, coll, tuple(filters))
        if key in self._lookups:
            return dict(self._lookups[key], cost=0.0, latency=0.0)
        self._lookups[key] = r = self._resolve(req, coll, filters)
        return r

    def _resolve(self, req: Request, coll: str, filters: list[str]) -> dict:
        ns = self._ns(req)
        try:
            ic = eval(f"ee.ImageCollection('{coll}')" + "".join(filters), ns)  # noqa: S307
            (n, fp), lat, eecu = self._profiled(lambda: (
                ic.size().getInfo(), ic.geometry().intersection(ns["region"], 100).area(100).divide(1e6).getInfo()))
            return {"n_img": int(n), "footprint_km2": float(fp), "cost": float(sum(eecu.values())), "latency": lat}
        except Exception as e:
            return {"n_img": 0, "footprint_km2": 0.0, "cost": 0.0, "latency": 0.0, "error": str(e)[:200]}

    def catalogue(self, req: Request, data_option: str):
        cat = req.meta.get("catalogue", {})
        lines = [f"{cid}: bands={c.get('bands')}, pixel={c.get('pixel_m')} m, cadence={c.get('cadence', '?')}"
                 for cid, c in cat.items() if c.get("option", data_option) == data_option]
        lines.append(f"Region variable `region` ({req.meta.get('area_km2', 0):.1f} km2), dates "
                     f"{req.meta.get('start')} .. {req.meta.get('end')} (start_date, end_date).")
        return "\n".join(lines), []

    def _exec(self, req: Request, code: str, tag: str, kind: str) -> RunResult:
        ee = self.ee
        tag = re.sub(r"[^a-z0-9_-]", "-", tag.lower())[:63]
        try:
            ee.data.setWorkloadTag(tag)
        except Exception:
            pass
        ns = self._ns(req, tuple(self.rng.uniform(-1e-6, 1e-6, 2)))
        buf = io.StringIO()
        before = self.monitor(tag) if self.monitor else None

        def go():
            with contextlib.redirect_stdout(buf):
                exec(code, ns)  # noqa: S102
                res = ns.get("result")
                return res.getInfo() if isinstance(res, ee.ComputedObject) else res
        try:
            val, lat, eecu = self._profiled(go)
            ok, obs = True, buf.getvalue() + (f"\nresult = {val!r}" if val is not None else "")
        except Exception as e:
            val, lat, eecu = None, 0.0, {}
            ok, obs = False, f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"
        comp = float(sum(eecu.values()))
        y = "" if ok else self.adapter.classify(obs)
        if ok and self.monitor:
            comp = max(comp, float(self.monitor(tag) - (before or 0.0)))
        if y == "memory":
            comp = max(comp, OOM_EECU_S)
        reads = kind == "inspect" and any(k in code for k in READS_DATA)
        call = Call(kind, comp, lat, reads, {"area_km2": req.meta.get("area_km2")})
        plan = None
        if kind == "run":
            try:
                p = self.adapter.compile(req, code)
                by: dict[str, list] = {}
                for o in p.ops:
                    by.setdefault(o.cls, []).append(o)
                for c, ops in by.items():          # profiled cost per class shared by volume
                    tot = sum(o.feats.get("volume", 0.0) for o in ops) or len(ops)
                    for o in ops:
                        o.feats["metered"] = eecu.get(c, 0.0) * (o.feats.get("volume", 0.0) or 1.0) / tot
                plan = p.to_dict()
            except Exception:
                plan = None
        return RunResult(ok, obs, [call], plan, val, y)

    def run(self, req, code, tag):
        return self._exec(req, code, tag, "run")

    def inspect(self, req, code, tag):
        return self._exec(req, code, tag, "inspect")

    def system_prompt(self, req: Request, config: str) -> str:
        return ("You are a geospatial analyst using the Earth Engine Python API (`ee`, already initialised). "
                "The variable `region` holds the request's region and `start_date`, `end_date` its dates. "
                "Computation is billed in EECU. Use `inspect` for metadata (collection sizes, band names) and "
                "`run` for the analysis; store the final value in a variable named `result`. When the result "
                "answers the request, call `answer` with the value only (a number, yes/no or a category).")

    def draft_prompt(self, req: Request, data_option: str, catalogue: str) -> str:
        return (f"Request: {req.text}\nData option: {data_option}\nCatalogue:\n{catalogue}\n\n"
                "Write the complete Earth Engine Python script; leave the analysis method as a named step "
                "`classify(image)` that returns a binary image. Store the final value in `result`.")

    def parse_answer(self, req: Request, text: str):
        t = text.strip().lower()
        if req.answer_kind == "binary":
            return 1.0 if t.startswith(("yes", "true", "1")) else 0.0
        m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", t)
        return float(m.group()) if m else None
