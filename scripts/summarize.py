"""Print the main result tables from <out>/<platform>/*.json and write them to <out>/summary.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {"b1raw": "B1 clean-run (raw)", "b5raw": "B5 self-estimate (raw)", "b1": "B1 clean-run",
         "b2": "B2 B1 x backbone", "b3": "B3 sample-and-extrapolate", "b4": "B4 per-group quantile",
         "b5": "B5 self-estimate", "b6": "B6 token forecaster + B1", "b7": "B7 learned quantile",
         "b7s": "B7 stacked on S_hat", "pq": "AgentQuote", "bonf": "AgentQuote, Bonferroni",
         "sel": "AgentQuote, estimate per meter", "pq_perq": "AgentQuote, one score per question"}


def f(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    md = []
    for p in ("earthengine", "bigquery"):
        q = out / p / "quotes.json"
        if not q.exists():
            continue
        d = json.loads(q.read_text())
        rows = []
        for k, v in d["methods"].items():
            rows.append([NAMES.get(k, k), f(v.get("cov_c")), f(v.get("cov_t")), f(v.get("cov_w")), f(v.get("cov_j")),
                         f(v.get("wid_c"), 2), f(v.get("wid_t"), 2), f(v.get("wid_w"), 2), f(v.get("miss"), 2)])
        md += [f"## Quotes, {p} (nominal {100 * (1 - 0.1):.0f}% joint)", "",
               table(["method", "cov C", "cov T", "cov W", "joint", "width C", "width T", "width W", "miss"], rows), ""]
        if d.get("ratios"):
            rr = [[k.upper(), *[f"{f(v[x]['mean'])} ({f(v[x]['lo'])}, {f(v[x]['hi'])})" for x in ("c", "t", "w")]]
                  for k, v in d["ratios"].items()]
            md += [f"Width reduction of AgentQuote against calibrated baselines, {p} (%)", "",
                   table(["vs", "compute", "tokens", "wall time"], rr), ""]
    c = out / "earthengine" / "choice.json"
    if c.exists():
        d = json.loads(c.read_text())
        rows = []
        for k in ("best_avg", "most_acc", "plugin", "cost_feasible", "agentquote", "on_b2", "on_b4", "on_b7",
                  "template", "oracle"):
            v = d.get(k)
            if v:
                rows.append([k, f(v.get("risk")), f(v.get("wrong")), f(v.get("overrun")), f(v.get("gold")),
                             f(v.get("layer")), f(v.get("cond")), f(v.get("adm")),
                             f"{f(v.get('decl_r'))}/{f(v.get('decl_b'))}", f(v.get("cents"), 2)])
        md += ["## Choice, earthengine", "",
               table(["rule", "risk", "wrong", "overrun", "gold", "layer", "cond", "adm", "decl r/b", "cents"], rows), ""]
        rows = []
        for rule in ("agentquote", "money_time", "money", "best_avg", "hard_caps"):
            r = [rule]
            for pct in ("90", "95", "99"):
                r.append(f(d["bind"][pct][rule].get("within")))
            for pct in ("90", "95", "99"):
                r.append(f(d["bind"][pct][rule].get("brk_any")))
            rows.append(r)
        md += ["Budgets that bind (within budget, breaks of any meter)", "",
               table(["rule", "in 90", "in 95", "in 99", "brk 90", "brk 95", "brk 99"], rows), ""]
    for p in ("earthengine", "bigquery"):
        ab = out / p / "ablation.json"
        if ab.exists():
            d = json.loads(ab.read_text())
            rows = [[k, f(v.get("cov")), f(v.get("wid"), 2), f(v.get("risk")), f(v.get("cents"), 2), f(v.get("adm"))]
                    for k, v in d.items() if isinstance(v, dict) and "cov" in v]
            md += [f"## Ablation, {p}", "", table(["variant", "cov", "width", "risk", "cents", "adm"], rows), ""]
    text = "\n".join(md)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
