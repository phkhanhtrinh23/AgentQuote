"""Earth Engine plan compiler.

Reads the drafted Python code as a syntax tree. Each Earth Engine method call becomes an operator of one of
seven classes (load, filter, mask, composite, inference, reduce, export) with the arguments the agent wrote,
grouped into the calls that materialise a result (getInfo, Export.*). Filters resolve the images left and
their footprints through metered catalogue lookups (``resolver``). Evaluation is lazy, so upstream raster
operators take the scale of their materialising call; an omitted scale takes the platform default.

Flags: an empty collection, a missing band, a reduction over the pixel limit, a client-side transfer and an
unparsed call.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..types import CompiledPlan, Operator, Request

CLASS_OF = {
    "filterBounds": "filter", "filterDate": "filter", "filter": "filter", "filterMetadata": "filter",
    "limit": "filter", "first": "filter",
    "updateMask": "mask", "unmask": "mask", "clip": "mask", "map": "mask",
    "median": "composite", "mean": "composite", "mosaic": "composite", "qualityMosaic": "composite",
    "min": "composite", "max": "composite", "sum": "composite", "reduce": "composite", "count": "composite",
    "normalizedDifference": "composite", "expression": "composite", "gt": "composite", "lt": "composite",
    "classify": "inference", "train": "inference", "cluster": "inference",
    "reduceRegion": "reduce", "reduceRegions": "reduce", "sampleRegions": "reduce", "reduceToVectors": "reduce",
    "sample": "reduce", "aggregate_array": "reduce", "size": "reduce",
}
MATERIALIZE = {"getInfo", "evaluate"}
EXPORTS = {"toDrive", "toAsset", "toCloudStorage", "table", "image"}
DEFAULT_MAX_PIXELS = 1e7
DEFAULT_SCALE_M = 1000.0   # EE default projection when no scale is given (coarse); compiled as such
CLIENT_LIMIT_ELEMENTS = 5000


@dataclass
class Chain:
    base: str                      # collection / image id, or variable name
    kind: str                      # ic | img | fc | var | other
    steps: list = field(default_factory=list)  # (method, args_src, kwargs_src dict)
    src: str = ""


def _src(node, code: str) -> str:
    return ast.get_source_segment(code, node) or ""


def flatten(node: ast.AST, code: str) -> Chain | None:
    steps = []
    cur = node
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        steps.append((cur.func.attr, [_src(a, code) for a in cur.args],
                      {k.arg: _src(k.value, code) for k in cur.keywords if k.arg}))
        inner = cur.func.value
        if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) and inner.value.id == "ee" \
                and inner.attr in ("ImageCollection", "Image", "FeatureCollection"):
            break
        cur = inner
    steps.reverse()
    if not steps:
        return None
    first = steps[0][0]
    if first in ("ImageCollection", "Image", "FeatureCollection"):
        kind = {"ImageCollection": "ic", "Image": "img", "FeatureCollection": "fc"}[first]
        base = steps[0][1][0].strip("'\"") if steps[0][1] else ""
        return Chain(base, kind, steps[1:], _src(node, code))
    if isinstance(cur, ast.Attribute) and isinstance(cur.value, ast.Name):
        return Chain(cur.value.id, "var", steps, _src(node, code))
    if isinstance(cur, ast.Name):
        return Chain(cur.id, "var", steps, _src(node, code))
    return Chain("", "other", steps, _src(node, code))


def _num(s: str | None, default: float) -> float:
    try:
        return float(ast.literal_eval(s)) if s else default
    except Exception:
        return default


def _bands(args: list[str]) -> list[str] | None:
    try:
        v = ast.literal_eval(args[0]) if args else None
    except Exception:
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return None


class EECompiler:
    """resolver(req, collection_id, filter_src: list[str]) -> dict(n_img, footprint_km2, cost, latency)."""

    def __init__(self, resolver=None):
        self.resolver = resolver

    def compile(self, req: Request, code: str, method: str = "") -> CompiledPlan:
        flags = {"empty_collection": 0, "missing_band": 0, "pixel_limit": 0, "client_transfer": 0, "unparsed": 0}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            flags["unparsed"] = 1
            return CompiledPlan([], flags, {"n_obs": 0})
        cat = req.meta.get("catalogue", {})
        region_km2 = float(req.meta.get("area_km2", 1.0))
        env: dict[str, dict] = {}   # variable -> symbolic state
        ops: list[Operator] = []
        meta_calls = []
        call_id = 0

        def state_of(ch: Chain) -> dict:
            if ch.kind in ("ic", "img", "fc"):
                c = cat.get(ch.base, {})
                return {"coll": ch.base, "kind": ch.kind, "filters": [], "bands": c.get("bands"),
                        "n_bands": len(c.get("bands", []) or [1]), "pixel_m": c.get("pixel_m", 30.0),
                        "n_img": 1 if ch.kind == "img" else None, "known": bool(c)}
            if ch.kind == "var" and ch.base in env:
                return dict(env[ch.base], filters=list(env[ch.base]["filters"]))
            return {"coll": "", "kind": "other", "filters": [], "bands": None, "n_bands": 1, "pixel_m": 30.0,
                    "n_img": 1, "known": False}

        def resolve(st: dict):
            if st["kind"] != "ic" or st["n_img"] is not None:
                return
            if self.resolver is not None:
                r = self.resolver(req, st["coll"], st["filters"])
                st["n_img"], st["footprint_km2"] = r.get("n_img", 0), r.get("footprint_km2", region_km2)
                meta_calls.append(r)
                if r.get("error"):
                    flags["unparsed"] = 1
            else:  # catalogue-average fallback (no metered lookup available)
                c = cat.get(st["coll"], {})
                st["n_img"] = int(c.get("img_per_month", 4) * req.meta.get("months", 1))
            if st["n_img"] == 0:
                flags["empty_collection"] = 1

        def volume(st: dict, scale_m: float) -> float:
            n = st["n_img"] if st["n_img"] is not None else 1
            area_m2 = min(st.get("footprint_km2", region_km2), region_km2) * 1e6
            return float(max(n, 0) * st["n_bands"] * area_m2 / scale_m ** 2)

        def process(ch: Chain, target: str | None):
            nonlocal call_id
            st = state_of(ch)
            pending: list[tuple[str, dict]] = []
            if ch.kind in ("ic", "img", "fc"):
                pending.append(("load", {}))
                if st["kind"] in ("ic", "img") and not st["known"]:
                    flags["unparsed"] = 1
            materialized = False
            for name, args, kw in ch.steps:
                if name in ("filterBounds", "filterDate", "filter", "filterMetadata", "limit"):
                    st["filters"].append(f".{name}({', '.join(args)})")
                    st["n_img"] = None if st["kind"] == "ic" else st["n_img"]
                if name == "select":
                    b = _bands(args)
                    if b is not None:
                        if st["bands"] is not None and any(x not in st["bands"] for x in b):
                            flags["missing_band"] = 1
                        st["n_bands"] = len(b)
                if name in ("median", "mean", "mosaic", "qualityMosaic", "min", "max", "sum", "count", "first"):
                    resolve(st)
                    st["kind_after"] = "img"
                if name in CLASS_OF:
                    pending.append((CLASS_OF[name], {"method": name, "args": args, "kw": kw}))
                if name in ("reduceRegion", "reduceRegions", "sampleRegions", "reduceToVectors", "sample"):
                    scale = _num(kw.get("scale"), st.get("pixel_m") or DEFAULT_SCALE_M)
                    max_px = _num(kw.get("maxPixels"), DEFAULT_MAX_PIXELS)
                    if region_km2 * 1e6 / scale ** 2 > max_px:
                        flags["pixel_limit"] = 1
                    st["scale"] = scale
                if name in MATERIALIZE:
                    materialized = True
                    if st["kind"] == "ic" and not any(p[0] in ("composite", "reduce") for p in pending):
                        resolve(st)
                        if (st["n_img"] or 0) > CLIENT_LIMIT_ELEMENTS or name == "getInfo" and \
                                any(s[0] == "toList" for s in ch.steps):
                            flags["client_transfer"] = 1
                if name in EXPORTS or name == "start":
                    materialized = True
            if st["kind"] == "ic":
                resolve(st) if any(p[0] in ("composite", "reduce", "inference") for p in pending) else None
            scale = st.get("scale") or st.get("pixel_m") or DEFAULT_SCALE_M
            if pending and (materialized or target is None):
                for cls, info in pending:
                    v = volume(st, scale) if cls != "reduce" else float(
                        st["n_bands"] * region_km2 * 1e6 / scale ** 2) * (1 if st.get("kind_after") else max(st["n_img"] or 1, 1))
                    if st["kind"] == "ic" and st["n_img"] == 0:
                        v = 0.0
                    ops.append(Operator(len(ops), cls, call_id, {
                        "volume": v, "scale": scale, "n_img": st["n_img"] or 0, "n_band": st["n_bands"],
                        "coll": st["coll"], "method": info.get("method", cls), "vector": float(st["kind"] == "fc"),
                        "tile_scale": _num(info.get("kw", {}).get("tileScale"), 1.0)}))
                call_id += 1
            if target is not None:
                st_saved = dict(st)
                if pending and not materialized:
                    st_saved["pending"] = st.get("pending", []) + pending
                env[target] = st_saved
            elif pending and not materialized:
                pass
            # materialising a variable defined earlier: emit its pending operators too
            if materialized and ch.kind == "var" and ch.base in env:
                for cls, info in env[ch.base].get("pending", []):
                    ops.append(Operator(len(ops), cls, call_id - 1, {
                        "volume": volume(st, scale), "scale": scale, "n_img": st["n_img"] or 0,
                        "n_band": st["n_bands"], "coll": st["coll"], "method": info.get("method", cls),
                        "vector": float(st["kind"] == "fc"), "tile_scale": 1.0}))

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                ch = flatten(node.value, code) if isinstance(node.value, ast.Call) else None
                if ch:
                    process(ch, node.targets[0].id)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                ch = flatten(node.value, code)
                if ch:
                    process(ch, None)
        if method and not any(o.cls == "inference" for o in ops) and method in ("rf", "random_forest", "classifier"):
            # instantiate the draft for the configuration by inserting the method's operator
            red = [o for o in ops if o.cls == "reduce"]
            v = max((o.feats["volume"] for o in ops if o.cls == "composite"), default=0.0)
            ops.insert(ops.index(red[0]) if red else len(ops),
                       Operator(0, "inference", red[0].call_id if red else 0, {"volume": v, "method": method}))
            for i, o in enumerate(ops):
                o.uid = i
        n_obs = max((o.feats.get("n_img", 0) for o in ops if o.cls == "load"), default=0)
        return CompiledPlan(ops, flags, {"n_obs": n_obs, "metadata_calls": meta_calls})
