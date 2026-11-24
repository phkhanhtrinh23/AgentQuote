"""python -m agentquote.experiments <task> [options]

Tasks: prepare, breakdown, quotes, choice, ablation, shift, efficiency, sensitivity, case, figures, all.
"""
from __future__ import annotations

import argparse
import time

from . import ablation, breakdown, case, choice_rules, efficiency, figures, quotes, sensitivity, shift
from .common import Ctx, add_args

TASKS = {"breakdown": breakdown.run, "quotes": quotes.run, "choice": choice_rules.run, "ablation": ablation.run,
         "shift": shift.run, "efficiency": efficiency.run, "sensitivity": sensitivity.run, "case": case.run,
         "figures": figures.run}


def prepare(ctx: Ctx):
    for p in ctx.platforms():
        for a in range(ctx.a.assign):
            ctx.table(p, a)


def main():
    ap = add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("task", choices=["prepare", "all", *TASKS])
    a = ap.parse_args()
    ctx = Ctx(a)
    todo = ["prepare", *TASKS] if a.task == "all" else [a.task]
    for t in todo:
        t0 = time.time()
        ctx.log(f"[{t}]")
        (prepare if t == "prepare" else TASKS[t])(ctx)
        ctx.log(f"[{t}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
