"""Plots from the result files under <out>/<platform>/*.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import Ctx

C = {"pq": "#2a6f97", "b7": "#e07a5f", "base": "#8d99ae", "succ": "#bc4749", "grey": "#adb5bd"}


def _load(ctx, p, name):
    f = ctx.out / p / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _num(x):
    return float(x) if isinstance(x, (int, float)) else float("nan")


def run(ctx: Ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig_dir = ctx.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.pdf")
        fig.savefig(fig_dir / f"{name}.png", dpi=200)
        plt.close(fig)
        ctx.log(f"  wrote {fig_dir / name}.pdf")

    # cost by part and meter
    bd = {p: _load(ctx, p, "breakdown") for p in ("earthengine", "bigquery")}
    bd = {p: v for p, v in bd.items() if v}
    if bd:
        fig, axes = plt.subplots(1, len(bd), figsize=(3.4 * len(bd), 2.2), squeeze=False)
        for ax, (p, v) in zip(axes[0], bd.items()):
            parts = ("draft_explore", "failed", "final")
            lab = {"draft_explore": "draft + exploration", "failed": "failed / rejected", "final": "final attempt"}
            bottom = np.zeros(3)
            for part, col in zip(parts, ("#8ecae6", "#e76f51", "#2a9d8f")):
                vals = np.array([v["share"][part][k] for k in ("comp", "tok", "time")])
                ax.bar(["compute", "tokens", "wall time"], vals, bottom=bottom, color=col, label=lab[part])
                bottom += vals
            ax.set_title(p)
            ax.set_ylabel("% of session cost")
        axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, "fig_decomp")

    # coverage across nominal levels
    q = _load(ctx, "earthengine", "quotes")
    if q and "alpha_sweep" in q:
        lv = sorted(q["alpha_sweep"], key=float)
        x = np.array([100 * float(v) for v in lv])
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
        for name, col in (("pq", C["pq"]), ("b7", C["b7"]), ("succ_att", C["succ"])):
            if name not in q["alpha_sweep"][lv[0]]:
                continue
            m = np.array([_num(q["alpha_sweep"][v][name]["mean"]) for v in lv])
            lo = np.array([_num(q["alpha_sweep"][v][name]["lo"]) for v in lv])
            hi = np.array([_num(q["alpha_sweep"][v][name]["hi"]) for v in lv])
            label = {"pq": "AgentQuote", "b7": "B7", "succ_att": "calibrated on last attempts"}[name]
            axes[0].plot(x, m, "o-", color=col, label=label, ms=3)
            if name != "succ_att":
                axes[1].errorbar(x + (0.4 if name == "b7" else -0.4), m - x, yerr=[m - lo, hi - m], fmt="o",
                                 color=col, ms=3, capsize=2, label=label)
        axes[0].plot(x, x, ":", color="k", lw=0.8)
        axes[0].set_xlabel("nominal joint coverage (%)"), axes[0].set_ylabel("joint coverage (%)")
        axes[1].axhline(0, color="k", lw=0.8, ls=":")
        axes[1].set_xlabel("nominal (%)"), axes[1].set_ylabel("coverage - nominal (pp)")
        axes[0].legend(frameon=False, fontsize=7)
        save(fig, "fig_alpha")

    # width reduction against calibrated baselines
    qs = {p: _load(ctx, p, "quotes") for p in ("earthengine", "bigquery")}
    qs = {p: v for p, v in qs.items() if v and v.get("ratios")}
    if qs:
        fig, axes = plt.subplots(1, len(qs), figsize=(3.4 * len(qs), 2.3), squeeze=False)
        for ax, (p, v) in zip(axes[0], qs.items()):
            names = list(v["ratios"])
            for j, (k, col) in enumerate(zip(("c", "t", "w"), ("#264653", "#2a9d8f", "#e9c46a"))):
                m = np.array([_num(v["ratios"][n][k]["mean"]) for n in names])
                lo = np.array([_num(v["ratios"][n][k]["lo"]) for n in names])
                hi = np.array([_num(v["ratios"][n][k]["hi"]) for n in names])
                xs = np.arange(len(names)) + (j - 1) * 0.25
                ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o", color=col, ms=3, capsize=2,
                            label={"c": "compute", "t": "tokens", "w": "wall time"}[k])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(len(names)), [n.upper().replace("B7S", "B7 on Ŝ") for n in names])
            ax.set_ylabel("narrower than baseline (%)")
            ax.set_title(p)
        axes[0][0].legend(frameon=False, fontsize=7)
        save(fig, "fig_ratios")

    # choice: risk against cost, and risk/admission across delta
    ch = _load(ctx, "earthengine", "choice")
    if ch:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
        rules = [("agentquote", "AgentQuote", True), ("best_avg", "best on average", True),
                 ("most_acc", "most accurate", True), ("plugin", "plug-in", False),
                 ("cost_feasible", "cost-feasible", False), ("on_b7", "on B7 quotes", True)]
        for key, lab, cert in rules:
            if key not in ch:
                continue
            r = ch[key]
            axes[0].scatter(_num(r["cents"]), _num(r["risk"]), s=4 + 2 * _num(r["adm"]), label=lab,
                            facecolors=None if cert else "none", edgecolors="k" if not cert else None)
        axes[0].axhline(100 * ctx.st.delta, ls=":", color="k", lw=0.8)
        axes[0].set_xlabel("cost per answer (¢)"), axes[0].set_ylabel("risk (%)")
        axes[0].legend(frameon=False, fontsize=6)
        ds = sorted(ch.get("delta_sweep", {}), key=float)
        if ds:
            x = [100 * float(d) for d in ds]
            axes[1].plot(x, [_num(ch["delta_sweep"][d]["risk"]) for d in ds], "o-", color=C["pq"], ms=3, label="risk")
            axes[1].plot(x, [_num(ch["delta_sweep"][d]["adm"]) for d in ds], "s-", color=C["b7"], ms=3,
                         label="admission")
            axes[1].plot(x, x, ":", color="k", lw=0.8)
            axes[1].set_xlabel("δ (%)"), axes[1].set_ylabel("%")
            axes[1].legend(frameon=False, fontsize=7)
        save(fig, "fig_choice")

    # ablations
    ab = _load(ctx, "earthengine", "ablation")
    if ab:
        names = [n for n in ("full", "reqonly", "noflags", "onecoef", "succ_att", "succ_ses", "mult", "noprev",
                             "nofrail", "pooled_resid", "raw", "copula", "stdscore") if n in ab]
        cov = [_num(ab[n]["cov"]) for n in names]
        wid = [_num(ab[n]["wid"]) / _num(ab["full"]["wid"]) for n in names]
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
        axes[0].barh(names, cov, color=C["pq"])
        axes[0].axvline(100 * (1 - ctx.st.alpha), ls="--", color="k", lw=0.8)
        axes[0].set_xlabel("joint coverage (%)")
        bars = axes[1].barh(names, wid, color=C["base"], log=True)
        for b, c in zip(bars, cov):
            if c < 88:
                b.set_hatch("///")
        axes[1].axvline(1, color="k", lw=0.8)
        axes[1].set_xscale("log"), axes[1].set_xlabel("median width / full")
        axes[1].set_yticks([])
        save(fig, "fig_ablation")

    # shift
    sh = _load(ctx, "earthengine", "shift")
    if sh:
        items = [(f"cont. {k}", v) for k, v in sh.get("continent", {}).items()]
        if "rerun" in sh:
            items.append(("later reruns", sh["rerun"]))
        for b, v in sh.get("held_out_backbone", {}).items():
            items.append((f"{b} borrowed", {"cov": v["borrowed"], "band": v["band"]}))
            items.append((f"{b} recalibrated", {"cov": v["recal"], "band": v["band"]}))
        if "own_draft" in sh:
            items.append(("own draft", {"cov": sh["own_draft"]["cov_own"], "band": sh["own_draft"]["band"]}))
        if items:
            fig, ax = plt.subplots(figsize=(6.8, 2.4))
            for i, (lab, v) in enumerate(items):
                lo, hi = (_num(x) for x in v["band"])
                ax.plot([i, i], [lo, hi], color=C["grey"], lw=6, solid_capstyle="butt")
                ax.plot(i, _num(v["cov"]), "o", color=C["pq"], ms=4)
            ax.axhline(100 * (1 - ctx.st.alpha), ls=":", color="k", lw=0.8)
            ax.set_xticks(range(len(items)), [x[0] for x in items], rotation=45, ha="right", fontsize=6)
            ax.set_ylabel("joint coverage (%)")
            save(fig, "fig_shift")

    # efficiency
    ef = _load(ctx, "earthengine", "efficiency")
    if ef:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
        st = ef["stage"]
        axes[0].bar(["draft", "lookups", "rollouts", "score"],
                    [st["draft_s"], st["lookup_s"], st["rollout_ms"] / 1000, st["score_ms"] / 1000], color=C["pq"])
        axes[0].set_yscale("log"), axes[0].set_ylabel("seconds (median)")
        axes[1].loglog(ef["candidates"]["k"], ef["candidates"]["seconds"], "o-", color=C["pq"], ms=3)
        axes[1].set_xlabel("candidates"), axes[1].set_ylabel("rollout seconds")
        save(fig, "fig_eff")

    # sensitivity
    se = _load(ctx, "earthengine", "sensitivity")
    if se:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
        for ax, key, xl in ((axes[0], "fit_regions", "fitting regions"), (axes[1], "rollouts", "rollouts")):
            ks = list(se[key])
            for j, (k, col) in enumerate(zip(("c", "t", "w"), ("#264653", "#2a9d8f", "#e9c46a"))):
                m = np.array([_num(se[key][x][k]["mean"]) for x in ks])
                lo = np.array([_num(se[key][x][k]["lo"]) for x in ks])
                hi = np.array([_num(se[key][x][k]["hi"]) for x in ks])
                ax.errorbar(np.arange(len(ks)) + (j - 1) * 0.15, m, yerr=[m - lo, hi - m], fmt="o-", ms=3,
                            capsize=2, color=col, label={"c": "compute", "t": "tokens", "w": "wall time"}[k])
            ax.set_xticks(range(len(ks)), ks)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel(xl), ax.set_ylabel("narrower than B7 (%)")
        axes[0].legend(frameon=False, fontsize=7)
        save(fig, "fig_sens")

    # case
    cs = _load(ctx, "earthengine", "case")
    if cs and cs.get("candidates"):
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for c in cs["candidates"]:
            worst = max(q / b for q, b in zip(c["Q"], c["B"]))
            ax.scatter(worst, c["r"], s=40 if c["chosen"] else 12, facecolors="none" if c["chosen"] else C["pq"],
                       edgecolors="k" if c["chosen"] else C["pq"])
        ax.axvline(1, color="k", lw=0.8)
        ax.axhline(_num(cs["lam"]), color="k", lw=0.8, ls="--")
        ax.set_xscale("log"), ax.set_xlabel("max quote / budget"), ax.set_ylabel("error-risk score")
        save(fig, "fig_case")
