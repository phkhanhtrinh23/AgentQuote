"""Earth Engine testbed: regions, drafted plans and fully metered agent sessions.

Request types follow the geospatial benchmarks and templated analyses: CBGB and UnivEARTH questions (their
data is named, so only the backbone varies), building area and count (Open Buildings, a Sentinel-2
classifier, Dynamic World), built-up fraction and tree cover (Sentinel-2, Landsat, MODIS, index threshold or
random forest), flood extent on Sen1Floods11-sized chips (Sentinel-1, Sentinel-2, MODIS) and NDVI change
(no reference). Every candidate configuration of every request is executed once under the agent protocol:
a shared draft, exploration, and up to N_max attempts that react to the previous attempt's feedback.

Costs are metered per operator class (EECU-s), per turn (list-price tokens) and per call (latency). Sessions
are written in the same log format as live collection, so every downstream step reads them unchanged.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit

from ..config import BACKBONE_NAMES
from ..types import (Attempt, Call, CompiledPlan, Operator, Request, Session, Step, Turn, config_key)
from .common import (CONTINENTS, FRAIL_X_CORR, NB, OUT_ATT, PIN, POUT, REWRITE_P, REWRITE_SD, SYS_TOK, TPS,
                     TURN_SD, bill_in, lognormal)

CLASSES = ["load", "filter", "mask", "composite", "inference", "reduce", "export"]
THETA = np.array([-16.6, -6.0, -15.9, -14.7, -14.0, -13.1, -9.0])
BETA = np.array([1.0, 0.5, 1.0, 1.0, 0.95, 0.95, 1.0])
SIG = np.array([0.25, 0.30, 0.30, 0.35, 0.40, 0.30, 0.30])
VEC_SHIFT = np.array([6.1, 0.0, 0.0, 0.0, 0.0, 3.9, 0.0])
CALL_CONST = np.array([0.05, 0.01, 0.03, 0.05, 0.05, 0.08, 0.03])
CALL_CONST_HARD = np.array([0.3, 0.05, 0.2, 0.4, 0.3, 0.5, 0.2])
VMIN_HARD = np.array([2e6, 50, 1e6, 2e6, 5e5, 2e5, 1e3])
FLAGS = ["pixel_limit", "missing_band", "empty_collection", "client_transfer", "unparsed"]
OMEGA = np.array([1.6, 2.2, 2.8, 1.8, 0.6])
PSI = np.array([-0.15, -0.75, -1.05])
WRONG_B = np.array([0.30, 0.0, -0.20])
EXPL_MU = np.array([2.0, 3.0, 1.5])
MID_INSPECT_P = 0.4
HZ_INTER = 0.5
OUT = ["success", "pixel_limit", "missing_band", "empty_collection", "client_transfer", "memory", "other"]
GAMMA_PREV = {"start": 0.0, "pixel_limit": 0.2, "missing_band": -0.6, "empty_collection": -0.3,
              "client_transfer": -0.5, "memory": 0.4, "other": 0.1}
UPTO = {"missing_band": 5, "empty_collection": 0, "client_transfer": 1}

OPT = {  # scale (m), revisit (days), bands, optical, vector
    "bench": dict(scale=30, revisit=8, bands=3, optical=1, vector=0),
    "open_buildings": dict(scale=1, revisit=0, bands=0, optical=0, vector=1),
    "s2_classifier": dict(scale=10, revisit=5, bands=6, optical=1, vector=0),
    "dynamic_world": dict(scale=10, revisit=5, bands=1, optical=0, vector=0),
    "sentinel2": dict(scale=10, revisit=5, bands=4, optical=1, vector=0),
    "landsat": dict(scale=30, revisit=8, bands=4, optical=1, vector=0),
    "modis": dict(scale=500, revisit=1, bands=3, optical=1, vector=0),
    "sentinel1": dict(scale=10, revisit=6, bands=2, optical=0, vector=0),
}
TYPES = {  # options, methods, reference kind, window (days), answer kind
    "cbgb": (["bench"], [""], "gold", 180, "numeric"),
    "univearth": (["bench"], [""], "gold", 90, "binary"),
    "building_area": (["open_buildings", "s2_classifier", "dynamic_world"], [""], "layer", 365, "numeric"),
    "building_count": (["open_buildings", "s2_classifier", "dynamic_world"], [""], "layer", 365, "numeric"),
    "builtup": (["sentinel2", "landsat", "modis"], ["index", "rf"], "layer", 365, "numeric"),
    "tree_cover": (["sentinel2", "landsat", "modis"], ["index", "rf"], "layer", 365, "numeric"),
    "flood": (["sentinel1", "sentinel2", "modis"], ["threshold", "rf"], "gold", 3, "numeric"),
    "ndvi_change": (["sentinel2", "landsat", "modis"], [""], "", 240, "numeric"),
}
COUNTS = {"cbgb": 45, "univearth": 135}
BASE_WRONG = {
    ("cbgb", "bench"): -0.4, ("univearth", "bench"): -0.9,
    ("building_area", "open_buildings"): -1.3, ("building_area", "s2_classifier"): 0.1,
    ("building_area", "dynamic_world"): -0.3, ("building_count", "open_buildings"): -1.0,
    ("building_count", "s2_classifier"): 0.6, ("building_count", "dynamic_world"): 0.9,
    ("builtup", "sentinel2"): -1.0, ("builtup", "landsat"): -0.6, ("builtup", "modis"): 0.4,
    ("tree_cover", "sentinel2"): -1.1, ("tree_cover", "landsat"): -0.9, ("tree_cover", "modis"): 0.2,
    ("flood", "sentinel1"): -1.2, ("flood", "sentinel2"): -0.4, ("flood", "modis"): 0.5,
    ("ndvi_change", "sentinel2"): -1.0, ("ndvi_change", "landsat"): -0.8, ("ndvi_change", "modis"): -0.2,
}
TEXT = {
    "cbgb": "Benchmark scenario {r}: compute the requested quantity over the region.",
    "univearth": "Benchmark question {r}: answer yes or no for the region.",
    "building_area": "What is the total building footprint area (km2) in the region?",
    "building_count": "How many buildings are in the region?",
    "builtup": "What fraction of the region is built-up land this year?",
    "tree_cover": "What fraction of the region is covered by trees this year?",
    "flood": "What share of the chip was under water during the event?",
    "ndvi_change": "By how much did the mean growing-season NDVI of the region change?",
}


class EarthEngineTestbed:
    platform = "earthengine"

    def __init__(self, seed: int = 0, per_type: int = 70, scale: float = 1.0, hard_adapter: bool = False,
                 area=(1.0, 500.0), n_max: int = 5):
        self.seed, self.n_max = seed, n_max
        self.rng = rng = np.random.default_rng(seed)
        self.const = CALL_CONST_HARD if hard_adapter else CALL_CONST
        self.vmin = VMIN_HARD if hard_adapter else np.ones(len(CLASSES))
        self.cont_fail = rng.normal(0, 0.25, len(CONTINENTS))
        self.cont_expl = rng.normal(0, 0.15, len(CONTINENTS))
        self.drift = np.zeros(NB)
        regions = []
        for t in TYPES:
            n = int(round(COUNTS.get(t, per_type) * scale))
            for _ in range(max(n, 2)):
                if t == "flood":
                    ar, cloud = 5.12 ** 2, rng.beta(4, 2)
                elif t in ("cbgb", "univearth"):
                    ar, cloud = math.exp(rng.uniform(math.log(5), math.log(2000))), rng.beta(2, 3)
                else:
                    ar, cloud = math.exp(rng.uniform(math.log(area[0]), math.log(area[1]))), rng.beta(2, 3)
                c = int(rng.integers(len(CONTINENTS)))
                regions.append(dict(type=t, area=ar, cloud=cloud, cont=c, z=rng.normal(), zx=rng.normal(),
                                    recall=float(np.clip(rng.normal(0.80, 0.09), 0.55, 0.97)),
                                    dens=math.exp(rng.normal(math.log(300), 1.0))))
        self.regions = regions
        self.plans = {}
        for r, reg in enumerate(regions):
            for opt in TYPES[reg["type"]][0]:
                self.plans[(r, opt)] = self.make_plan(reg, opt, rng)
            self.plans[(r, "__template__")] = self.make_plan(reg, TYPES[reg["type"]][0][0], rng, template=True)

    # ------------------------------------------------------------------------------------------ requests
    def request(self, r: int) -> Request:
        reg = self.regions[r]
        t = reg["type"]
        opts, methods, ref, _, kind = TYPES[t]
        cands = [config_key(b, o, m) for o in opts for m in methods for b in BACKBONE_NAMES]
        return Request(f"ee-{t}-{r:04d}", f"region-{r:04d}", t, TEXT[t].format(r=r), "earthengine", cands,
                       None if not ref else "reference", ref, kind,
                       {"area_km2": reg["area"], "continent": CONTINENTS[reg["cont"]]})

    # ------------------------------------------------------------------------------------------ plans
    def make_plan(self, reg, opt, rng, template=False):
        o, t = OPT[opt], reg["type"]
        v, vec = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
        smult = 1.0 if (o["vector"] or template) else rng.choice([1.0, 0.25, 4.0], p=[0.8, 0.1, 0.1])
        getinfo = False
        if o["vector"]:
            feats = max(1.0, reg["dens"] * reg["area"])
            v[[0, 1, 5]] = feats
            vec[[0, 5]] = 1
            getinfo = rng.random() < 0.5 and not template
            if getinfo:
                v[6] = feats
            n_img = 1
        else:
            px = reg["area"] * 1e6 / (o["scale"] ** 2) * smult
            window = TYPES[t][3] + (60 if (t == "flood" and opt == "sentinel1") else 0)
            raw = max(1, window / max(o["revisit"], 1))
            passr = 1 - 0.75 * reg["cloud"] if o["optical"] else 1.0
            n_img = rng.poisson(raw * passr) if raw * passr < 30 else int(raw * passr)
            if t == "flood" and opt == "sentinel2":
                n_img = rng.poisson(0.9 * (1 - reg["cloud"]))
            v[0] = max(n_img, 1) * o["bands"] * px
            v[1] = raw * 4
            v[2] = max(n_img, 1) * px if o["optical"] else 0
            v[3] = max(n_img, 1) * o["bands"] * px
            v[4] = o["bands"] * px * (3 if opt == "s2_classifier" else 1)
            v[5] = px
        flags = np.zeros(len(FLAGS))
        if not template:
            flags[0] = rng.random() < 0.85 * expit(math.log10(max(v[3], 1)) - 9.6)
            flags[1] = rng.random() < (0.12 if opt in ("landsat", "dynamic_world") else 0.06)
            flags[2] = (n_img == 0) or (rng.random() < 0.04)
            flags[3] = getinfo and v[6] > 5000
            flags[4] = rng.random() < 0.03
        eta = rng.normal(0, 0.2, len(CLASSES))
        d_out = float(lognormal(rng, 900, 0.3))
        lookups = [float(lognormal(rng, 0.2, 0.4)) for _ in range(3)]
        cloud_meta = float(np.clip(reg["cloud"] + rng.normal(0, 0.1), 0, 1)) if o["optical"] else 0.0
        return dict(v=np.maximum(v, 0.0), vec=vec, flags=flags, eta=eta, n_img=n_img, d_out=d_out,
                    lookups=lookups, cloud_meta=cloud_meta, opt=opt)

    def compiled(self, plan, v=None, metered=None, method=""):
        v = plan["v"] if v is None else v
        ops = []
        for u, c in enumerate(CLASSES):
            vol = float(v[u])
            if c == "inference" and method != "rf" and plan["opt"] != "s2_classifier":
                vol = 0.0
            if vol <= 0 and (metered is None or metered[u] <= 0):
                continue
            f = {"volume": vol, "vector": float(plan["vec"][u])}
            if metered is not None:
                f["metered"] = float(metered[u])
            ops.append(Operator(len(ops), c, 0, f))
        o = OPT[plan["opt"]]
        meta = {"n_obs": float(plan["n_img"]),
                "quality": {"cloud": plan["cloud_meta"], "scale_m": float(o["scale"]), "optical": float(o["optical"]),
                            "log_area": math.log(max(float(v[5]) * o["scale"] ** 2 / 1e6, 1e-3)) if v[5] > 0 else 0.0}}
        return CompiledPlan(ops, {f: int(x) for f, x in zip(FLAGS, plan["flags"])}, meta)

    def method_v(self, plan, m):
        v = plan["v"].copy()
        if m != "rf" and plan["opt"] != "s2_classifier":
            v[4] = 0.0
        return v

    # ------------------------------------------------------------------------------------------ metering
    def op_costs(self, plan, v, rng):
        on = v > 0
        lv = np.log(np.maximum(v, self.vmin))
        mu = THETA + BETA * lv + VEC_SHIFT * plan["vec"] + plan["eta"]
        return np.where(on, np.exp(mu + rng.normal(0, SIG)) + self.const, 0.0)

    def _turn(self, kind, ctx, ctx0, out, obs, b, rng, noise=True):
        tok = bill_in(ctx, ctx0, b, rng) + out * POUT[b]
        lat = (out / TPS[b] + 1.0) * (float(lognormal(rng, 1, TURN_SD)) if noise else 1.0)
        return Turn(kind, float(ctx), float(out), float(obs), float(lat), float(tok))

    def draft_step(self, plan, b=0, rng=None):
        rng = rng or self.rng
        d_out = plan["d_out"]
        t = Turn("draft", 4400.0, d_out, 0.0, d_out / TPS[b] + 2.0, 4400 * PIN[b] + d_out * POUT[b])
        calls = [Call("lookup", e, 1.5) for e in plan["lookups"]]
        return Step(t, calls)

    # ------------------------------------------------------------------------------------------ sessions
    def run_session(self, r, opt, m, b, rng=None, own_draft=False):
        rng = rng or self.rng
        reg = self.regions[r]
        t = reg["type"]
        base = self.plans[(r, opt)]
        plan = dict(base, v=self.method_v(base, m))
        bn = BACKBONE_NAMES[b]
        z = reg["z"]
        zx = FRAIL_X_CORR * z + math.sqrt(1 - FRAIL_X_CORR ** 2) * reg["zx"]
        draft = self.draft_step(plan, b if own_draft else 0, rng)
        if own_draft:                                        # the executing backbone writes its own plan
            plan = dict(plan, v=plan["v"] * float(lognormal(rng, 1.3, 0.6)))
        clean0 = self.op_costs(plan, plan["v"], rng).sum()
        mu = EXPL_MU[b] * math.exp(0.3 * zx + self.cont_expl[reg["cont"]])
        nx = rng.negative_binomial(2, 2 / (2 + mu))
        ctx0 = SYS_TOK[b] + plan["d_out"] + float(lognormal(rng, 400, 0.3))
        ctx = ctx0
        explore, known = [], False
        for _ in range(nx):
            heavy = rng.random() < 0.12
            e = float(lognormal(rng, 0.4, 0.7)) + (0.08 * clean0 if heavy else 0.0)
            out = float(lognormal(rng, 250, 0.5))
            obs = float(lognormal(rng, 900 if heavy else 600, 0.8))
            turn = self._turn("explore", ctx, ctx0, out, obs, b, rng, noise=False)
            lat = 1.5 * float(lognormal(rng, 1, 0.4)) + (0.9 * e ** 0.75 if heavy else 0.0)
            explore.append(Step(turn, [Call("inspect" if heavy else "metadata", e, lat, heavy)]))
            known = known or heavy
            ctx += out + obs
        q = np.ones(len(CLASSES))
        if rng.random() < REWRITE_P:
            q *= float(lognormal(rng, 1, REWRITE_SD))
        flags = plan["flags"].copy()
        if known and flags[2] and rng.random() < 0.5:        # an inspected collection reveals the empty filter
            flags[2] = 0
            q[[0, 1, 2, 3]] *= rng.uniform(1.5, 3.0)
        prev, atts, success, giveup = "start", [], False, None
        psi = PSI[b] + self.drift[b]
        for j in range(1, self.n_max + 1):
            logit = psi + GAMMA_PREV[prev] + 0.55 * z + self.cont_fail[reg["cont"]] + (0.6 if j >= 3 else 0.0)
            if j == 1:
                logit += OMEGA @ flags + (0.5 * flags[0] if b == 0 else 0.0)
            elif plan["flags"][0] and prev in ("pixel_limit", "memory"):
                logit += HZ_INTER
            if rng.random() < expit(logit):
                setf = [FLAGS[i] for i in range(4) if flags[i]]
                if j == 1 and setf and rng.random() < 0.75:
                    y = setf[rng.integers(len(setf))]
                else:
                    p = np.array([0.04, 0.08, 0.04, 0.03, 0.25, 0.40])
                    y = OUT[1:][rng.choice(6, p=p / p.sum())]
            else:
                y = "success"
            vq = plan["v"] * q
            ops = self.op_costs(plan, vq, rng)
            if y == "success":
                e, met = ops.sum(), ops
                lat = 2 + 0.9 * e ** 0.75 * lognormal(rng, 1, 0.3)
            else:
                f = {"pixel_limit": rng.uniform(0.6, 1.0), "memory": rng.uniform(1.0, 2.0),
                     "empty_collection": 0.01, "missing_band": 0.02, "client_transfer": 0.7,
                     "other": rng.uniform(0, 0.5)}[y]
                met = ops * f
                e = met.sum()
                lat = 300.0 if y == "memory" else 2 + 0.5 * e ** 0.75
            out = float(lognormal(rng, OUT_ATT[b], 0.35))
            obs = float(lognormal(rng, 250 if y == "success" else 500, 0.5))
            turn = self._turn("attempt", ctx, ctx0, out, obs, b, rng)
            a = Attempt(j, y, Step(turn, [Call("run", float(e), float(lat))]),
                        self.compiled(plan, vq, met, m).to_dict(), known)
            atts.append(a)
            ctx += out + obs
            if y == "success":
                fo = float(lognormal(rng, 120, 0.4))
                a.final = self._turn("final", ctx, ctx0, fo, 0.0, b, rng)
                success = True
                break
            if j == self.n_max:
                break
            if rng.random() < expit(-2.4 + 0.45 * (j - 1) + 0.4 * z + (0.5 if y == "other" else 0.0)):
                giveup = self._turn("giveup", ctx, ctx0, float(lognormal(rng, 150, 0.4)), 0.0, b, rng)
                break
            if rng.random() < MID_INSPECT_P:                  # inspection between attempts
                for _ in range(int(rng.integers(1, 3))):
                    o2, ob2 = float(lognormal(rng, 250, 0.5)), float(lognormal(rng, 600, 0.8))
                    e2 = float(lognormal(rng, 0.4, 0.7))
                    tt = self._turn("inspect", ctx, ctx0, o2, ob2, b, rng, noise=False)
                    a.after.append(Step(tt, [Call("inspect", e2, 1.5 * float(lognormal(rng, 1, 0.4)), False)]))
                    ctx += o2 + ob2
            g = np.ones(len(CLASSES))
            if y == "pixel_limit":
                g[[0, 2, 3, 4, 5]] = rng.uniform(0.08, 0.5)
            elif y == "memory":
                g[[0, 2, 3, 4, 5]] = rng.uniform(0.2, 0.8)
            elif y == "empty_collection":
                g[[0, 1, 2, 3]] = rng.uniform(1.5, 4.0)
            elif y == "client_transfer":
                g[6] = rng.uniform(0.05, 0.5)
            elif y == "other":
                g[:] = float(lognormal(rng, 1, 0.3))
            q = q * g
            prev = y
        req = self.request(r)
        bn_key = config_key(bn, opt, m)
        s = Session(f"{req.request_id}#{bn_key}", req.request_id, req.unit, t, "earthengine", bn_key,
                    draft, self.compiled(base, self.method_v(base, m), None, m).to_dict(), explore, atts, giveup,
                    has_ref=bool(TYPES[t][2]), ref_kind=TYPES[t][2])
        s.error = self.answer_error(reg, t, opt, m, b, success, rng) if s.has_ref else None
        s.answer = None if not success else "value"
        tmpl = self.plans[(r, "__template__")]
        s.extras = {"continent": CONTINENTS[reg["cont"]], "request_tokens": float(len(req.text) // 4 + 600),
                    "template_plan": self.compiled(tmpl, self.method_v(tmpl, m), None, m).to_dict()}
        s.extras["probe"] = self.probe(plan, b, ctx0, rng)
        if rng.random() > 0.08:                              # the backbone's stated costs, from the plan alone
            s.extras["self_estimate"] = {
                "comp": float(clean0 * lognormal(rng, 1.5, 0.9)),
                "tok": float((ctx0 * PIN[b] + OUT_ATT[b] * POUT[b]) * lognormal(rng, 1.5, 0.8)),
                "time": float((OUT_ATT[b] / TPS[b] + 2 + 0.9 * clean0 ** 0.75) * lognormal(rng, 1.5, 0.8))}
        return s

    def answer_error(self, reg, t, opt, m, b, success, rng):
        if not success:
            return math.inf
        lw = BASE_WRONG[(t, opt)] + 0.5 * reg["z"] + WRONG_B[b] - 0.25 * (m == "rf")
        if OPT[opt]["optical"]:
            lw += 2.0 * (reg["cloud"] - 0.4)
        if opt == "modis":
            lw += 0.5 - 0.35 * math.log10(reg["area"])
        if opt == "open_buildings":
            lw += 2.5 * (0.8 - reg["recall"])
        u = lw + rng.logistic()
        if TYPES[t][4] == "binary":
            return float(u > 0)
        return float(0.10 * math.exp(0.9 * u))       # P[error > 0.10] = logit^-1(lw)

    def probe(self, plan, b, ctx0, rng):
        """The draft run on a 1% tile, its compute scaled by area."""
        e = float(self.op_costs(plan, plan["v"] * 0.01, rng).sum()) * 100
        out = float(lognormal(rng, OUT_ATT[b], 0.35))
        tok = ctx0 * PIN[b] + out * POUT[b]
        return [e, tok, out / TPS[b] + 1 + 2 + 0.9 * e ** 0.75]

    def template_run(self, r, rng=None):
        """Non-agent template on the type's first data option: one fixed plan, no retries."""
        rng = rng or self.rng
        reg = self.regions[r]
        tmpl = self.plans[(r, "__template__")]
        opt = TYPES[reg["type"]][0][0]
        v = self.method_v(tmpl, "")
        e = float(self.op_costs(tmpl, v, rng).sum())
        ok = rng.random() > 0.03
        err = self.answer_error(reg, reg["type"], opt, "", 0, ok, rng) if TYPES[reg["type"]][2] else None
        return {"comp": e, "tok": 0.0, "time": 2 + 0.9 * e ** 0.75 * float(lognormal(rng, 1, 0.3)),
                "err": err, "ok": ok, "benchmark": reg["type"] in ("cbgb", "univearth")}

    # ------------------------------------------------------------------------------------------ generation
    def generate(self, regions=None, rng=None, **kw):
        rng = rng or self.rng
        regions = range(len(self.regions)) if regions is None else regions
        reqs, sess = [], []
        for r in regions:
            req = self.request(r)
            reqs.append(req)
            t = self.regions[r]["type"]
            for opt in TYPES[t][0]:
                for m in TYPES[t][1]:
                    for b in range(NB):
                        sess.append(self.run_session(r, opt, m, b, rng, **kw))
        return reqs, sess

    def rerun(self, regions, seed=1, psi_shift=0.2):
        """Later re-executions of the same requests, with backbone behaviour that may have changed."""
        rng = np.random.default_rng(seed)
        self.drift = rng.normal(0, psi_shift, NB)
        try:
            return self.generate(regions, rng)
        finally:
            self.drift = np.zeros(NB)
