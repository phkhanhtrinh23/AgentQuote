"""Earth Engine requests for live collection.

  CBGB         45 scenarios with numeric gold answers            (candidates differ by backbone only)
  UnivEARTH    a random 135 of its yes/no questions               (candidates differ by backbone only)
  buildings    area and count, 70 regions each: Open Buildings | Sentinel-2 classifier | Dynamic World
  builtup      built-up fraction over Sentinel-2 | Landsat | MODIS, method index threshold | random forest
  treecover    tree cover over Sentinel-2 | Landsat | MODIS,      method index threshold | random forest
  flood        flood extent over Sentinel-1 | Sentinel-2 | MODIS, method threshold | random forest
  ndvi_change  NDVI change over Sentinel-2 | Landsat | MODIS (no reference: quotes only)

Templated regions are squares of 1-500 km2 drawn from a land pool stratified by continent (and land type,
once Earth Engine is available). Flood regions are Sen1Floods11 hand-labelled chips with the event dates.
References: S1F hand labels (gold); Microsoft footprints, Hansen GFC and WorldCover 2021 (layer references),
computed with ``compute_layer_references`` on Earth Engine.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import BACKBONE_NAMES, ROOT
from ..types import Request, config_key

CACHE = ROOT / "data_cache"
BACKBONES = BACKBONE_NAMES

# data option -> catalogue entries the draft may read
CATALOGUE = {
    "s1": {"COPERNICUS/S1_GRD": {"bands": ["VV", "VH", "angle"], "pixel_m": 10, "img_per_month": 5, "cadence": "6-12 days"}},
    "s2": {"COPERNICUS/S2_SR_HARMONIZED": {"bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11",
                                                     "B12", "SCL", "QA60"], "pixel_m": 10, "img_per_month": 6,
                                           "cadence": "5 days"}},
    "landsat": {"LANDSAT/LC08/C02/T1_L2": {"bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                                                     "QA_PIXEL"], "pixel_m": 30, "img_per_month": 2, "cadence": "16 days"}},
    "modis": {"MODIS/061/MOD09GA": {"bands": [f"sur_refl_b0{i}" for i in range(1, 8)] + ["state_1km"], "pixel_m": 500,
                                    "img_per_month": 30, "cadence": "daily"},
              "MODIS/061/MOD13Q1": {"bands": ["NDVI", "EVI", "SummaryQA"], "pixel_m": 250, "img_per_month": 2,
                                    "cadence": "16 days"}},
    "open_buildings": {"GOOGLE/Research/open-buildings/v3/polygons": {"bands": ["area_in_meters", "confidence"],
                                                                      "pixel_m": 0.5, "img_per_month": 0, "kind": "fc"}},
    "dynamic_world": {"GOOGLE/DYNAMICWORLD/V1": {"bands": ["water", "trees", "grass", "flooded_vegetation", "crops",
                                                           "shrub_and_scrub", "built", "bare", "snow_and_ice", "label"],
                                                 "pixel_m": 10, "img_per_month": 6, "cadence": "5 days"}},
    "s2_classifier": {"COPERNICUS/S2_SR_HARMONIZED": {"bands": ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"],
                                                      "pixel_m": 10, "img_per_month": 6, "cadence": "5 days"}},
}
TEMPLATES = {
    "buildings_area": (("open_buildings", "s2_classifier", "dynamic_world"), ("",),
                       "What is the total building footprint area (km2) in the region during {year}?", "layer"),
    "buildings_count": (("open_buildings", "s2_classifier", "dynamic_world"), ("",),
                        "How many buildings are in the region during {year}?", "layer"),
    "builtup": (("s2", "landsat", "modis"), ("index", "rf"),
                "What fraction of the region is built-up land in {year}?", "layer"),
    "treecover": (("s2", "landsat", "modis"), ("index", "rf"),
                  "What fraction of the region is covered by trees in {year}?", "layer"),
    "flood": (("s1", "s2", "modis"), ("threshold", "rf"),
              "What share of the region was under water on {start} to {end}?", "gold"),
    "ndvi_change": (("s2", "landsat", "modis"), ("",),
                    "By how much did the mean growing-season NDVI of the region change between {y0} and {year}?", ""),
}
EPS_FLOOR = {"buildings_area": 0.01, "buildings_count": 10.0, "builtup": 0.01, "treecover": 0.01, "flood": 0.01,
             "ndvi_change": 0.01, "cbgb": 1e-3}


# ------------------------------------------------------------------ answer error: relative error with a floor
def answer_error(req: Request, ans) -> float:
    if ans is None:
        return float("inf")
    if req.answer_kind == "binary":
        ref = 1.0 if str(req.reference).strip().lower() in ("yes", "true", "1", "1.0") else 0.0
        a = str(ans).strip().lower()
        if a.startswith(("yes", "true")):
            return float(ref != 1.0)
        if a.startswith(("no", "false")):
            return float(ref != 0.0)
        try:
            return float(float(a) != ref)
        except ValueError:
            return 1.0
    try:
        a, ref = float(ans), float(req.reference)
    except (TypeError, ValueError):
        return float("inf")
    return abs(a - ref) / max(abs(ref), EPS_FLOOR.get(req.rtype, 1e-3))


def _cands(ds, ms, backbones=BACKBONES):
    return [config_key(b, d, m) for b in backbones for d in ds for m in ms]


def _catalogue(ds) -> dict:
    out = {}
    for d in ds:
        for cid, c in CATALOGUE[d].items():
            out[cid] = dict(c, option=d)
    return out


# ------------------------------------------------------------------ public benchmarks
def load_cbgb() -> list[Request]:
    rows = list(csv.DictReader(open(CACHE / "cbgb" / "eval_set.csv")))
    return [Request(f"cbgb-{r['id']}", f"cbgb-{r['id']}", "cbgb", r["question"], "earthengine",
                    _cands(("given",), ("",)), float(r["expected_answer"]), "gold", "numeric",
                    {"difficulty": r["difficulty"], "title": r["title"], "catalogue": {}}) for r in rows]


def load_univearth(n: int = 135, seed: int = 0) -> list[Request]:
    rows = json.load(open(CACHE / "univearth" / "data.json"))
    idx = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    out = []
    for i in sorted(idx):
        r = rows[i]
        out.append(Request(f"univ-{r['post_id']}-{i}", f"univ-{r['post_id']}", "univearth",
                           r["question"] + " Answer yes or no.", "earthengine", _cands(("given",), ("",)),
                           r["answer"], "gold", "binary", {"tag": r.get("tag", ""), "catalogue": {}}))
    return out


# ------------------------------------------------------------------ regions
def _land_polys():
    from shapely.geometry import shape
    d = json.load(open(CACHE / "naturalearth" / "countries_110m.geojson"))
    by = defaultdict(list)
    for f in d["features"]:
        c = f["properties"]["CONTINENT"]
        if c in ("Antarctica", "Seven seas (open ocean)"):
            continue
        by[c].append(shape(f["geometry"]))
    return by


def square(lon: float, lat: float, area_km2: float) -> list[float]:
    side = math.sqrt(area_km2)
    dlat = side / 111.32 / 2
    dlon = side / (111.32 * max(math.cos(math.radians(lat)), 0.05)) / 2
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def sample_regions(n: int, seed: int) -> list[dict]:
    """Squares of 1-500 km2 (log-uniform) on land, stratified evenly over continents."""
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    rng = np.random.default_rng(seed)
    polys = _land_polys()
    conts = sorted(polys)
    lands = {c: unary_union(polys[c]) for c in conts}
    out = []
    for i in range(n):
        c = conts[i % len(conts)]
        land = lands[c]
        x0, y0, x1, y1 = land.bounds
        while True:
            lon, lat = rng.uniform(x0, x1), rng.uniform(max(y0, -56), min(y1, 72))
            area = float(np.exp(rng.uniform(np.log(1), np.log(500))))
            bb = square(lon, lat, area)
            if land.contains(Point(lon, lat)) and land.contains(box(*bb)):
                out.append({"bbox": bb, "area_km2": area, "continent": c})
                break
    return out


def flood_chips() -> list[dict]:
    """The 446 Sen1Floods11 hand-labelled chips with their event dates and water share (gold)."""
    import rasterio
    meta = json.load(open(CACHE / "sen1floods11" / "meta.geojson"))["features"]
    dates = {f["properties"]["location"].replace(" ", "_").lower(): f["properties"]["s1_date"] for f in meta}
    dates["mekong"] = dates["cambodia"]  # the label files name the Cambodia event "Mekong"
    out = []
    for p in sorted((CACHE / "sen1floods11" / "LabelHand").glob("*_LabelHand.tif")):
        loc = p.name.rsplit("_", 2)[0]
        with rasterio.open(p) as f:
            a = f.read(1)
            b = f.bounds
        valid = a >= 0
        if valid.sum() == 0:
            continue
        d = dates.get(loc.lower())
        y, m, dd = (int(x) for x in d.split("/"))
        lat = (b.bottom + b.top) / 2
        area = (b.right - b.left) * 111.32 * math.cos(math.radians(lat)) * (b.top - b.bottom) * 111.32
        out.append({"chip": p.name.replace("_LabelHand.tif", ""), "event": loc, "bbox": [b.left, b.bottom, b.right, b.top],
                    "area_km2": area, "start": f"{y:04d}-{m:02d}-{dd:02d}", "end": _next_day(y, m, dd),
                    "water_share": float((a[valid] == 1).mean()), "valid_share": float(valid.mean())})
    return out


def _next_day(y, m, d):
    import datetime as dt
    return (dt.date(y, m, d) + dt.timedelta(days=1)).isoformat()


def load_templated(n_per_type: int = 70, seed: int = 0) -> list[Request]:
    rng = np.random.default_rng(seed)
    out = []
    types = [t for t in TEMPLATES if t != "flood"]
    regions = sample_regions(n_per_type * len(types), seed)
    k = 0
    for t in types:
        ds, ms, tmpl, ref_kind = TEMPLATES[t]
        for i in range(n_per_type):
            r = regions[k]; k += 1
            year = int(rng.choice([2019, 2020, 2021, 2022, 2023])) if t not in ("builtup",) else 2021
            if t in ("buildings_area", "buildings_count"):
                year = 2023
            y0 = year - int(rng.integers(2, 6))
            text = tmpl.format(year=year, y0=y0) + f" The region is a {r['area_km2']:.1f} km2 square in {r['continent']}."
            uid = f"{t}-{i:03d}"
            out.append(Request(uid, uid, t, text, "earthengine", _cands(ds, ms), None, ref_kind, "numeric",
                               {"bbox": r["bbox"], "area_km2": r["area_km2"], "continent": r["continent"],
                                "start": f"{year}-01-01", "end": f"{year}-12-31", "year": year, "y0": y0,
                                "months": 12, "catalogue": _catalogue(ds)}))
    chips = flood_chips()
    ds, ms, tmpl, _ = TEMPLATES["flood"]
    for i in rng.choice(len(chips), size=min(n_per_type, len(chips)), replace=False):
        c = chips[i]
        text = tmpl.format(start=c["start"], end=c["end"]) + f" The region is a {c['area_km2']:.1f} km2 tile ({c['event']})."
        out.append(Request(f"flood-{c['chip']}", f"flood-{c['chip']}", "flood", text, "earthengine", _cands(ds, ms),
                           c["water_share"], "gold", "numeric",
                           {"bbox": c["bbox"], "area_km2": c["area_km2"], "continent": c["event"], "event": c["event"],
                            "start": c["start"], "end": c["end"], "months": 1, "catalogue": _catalogue(ds)}))
    return out


def load_all(seed: int = 0) -> list[Request]:
    return load_cbgb() + load_univearth(135, seed) + load_templated(70, seed)


# ------------------------------------------------------------------ layer references (needs Earth Engine)
def compute_layer_references(reqs: list[Request], ee, ms_buildings_asset: str = "projects/sat-io/open-datasets/MSBuildings/global") -> None:
    """Fill ``reference`` for the layer-referenced types. Asset ids pinned here; check them before running."""
    for r in reqs:
        if r.reference is not None or r.ref_kind != "layer":
            continue
        g = ee.Geometry.Rectangle(r.meta["bbox"])
        area_m2 = r.meta["area_km2"] * 1e6
        if r.rtype == "builtup":
            wc = ee.ImageCollection("ESA/WorldCover/v200").first().eq(50)
            v = wc.reduceRegion(ee.Reducer.mean(), g, 10, maxPixels=1e10).get("Map").getInfo()
        elif r.rtype == "treecover":
            h = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
            y = r.meta["year"] - 2000
            tree = h.select("treecover2000").gte(30).And(h.select("lossyear").eq(0).Or(h.select("lossyear").gt(y)))
            v = tree.reduceRegion(ee.Reducer.mean(), g, 30, maxPixels=1e10).values().get(0).getInfo()
        elif r.rtype in ("buildings_area", "buildings_count"):
            fc = ee.FeatureCollection(ms_buildings_asset).filterBounds(g)
            v = fc.size().getInfo() if r.rtype == "buildings_count" else \
                fc.map(lambda f: f.set("a", f.geometry().intersection(g, 1).area(1))).aggregate_sum("a").getInfo() / 1e6
        else:
            continue
        r.reference = float(v) if v is not None else None
        _ = area_m2


def enrich_quality(reqs: list[Request], ee) -> None:
    """Quality metadata the error-risk score reads: mean cloud cover of optical options, land type."""
    for r in reqs:
        if "bbox" not in r.meta:
            continue
        g = ee.Geometry.Rectangle(r.meta["bbox"])
        q = {}
        try:
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(g).filterDate(r.meta["start"], _end(r))
            q["cloud_s2"] = s2.aggregate_mean("CLOUDY_PIXEL_PERCENTAGE").getInfo()
            lc = ee.ImageCollection("ESA/WorldCover/v200").first().reduceRegion(ee.Reducer.mode(), g, 100).get("Map")
            q["landcover"] = lc.getInfo()
        except Exception:
            pass
        r.meta["quality"] = q


def _end(r):
    return r.meta["end"] if r.meta["end"] > r.meta["start"] else r.meta["start"]
