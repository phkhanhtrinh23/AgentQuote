#!/usr/bin/env bash
# End-to-end run: generate the session pools, fit and calibrate every method, run all analyses, draw figures.
#
#   bash run_all.sh                 # full run (10 assignments, 1,000 rollouts)
#   QUICK=1 bash run_all.sh         # a few minutes: smaller pools, 2 assignments, 200 rollouts
#
# Use your own logged sessions instead of the testbeds by placing them in $DATA/<platform>/sessions.jsonl
# (see scripts/collect_sessions.py) and setting SKIP_GENERATE=1.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
DATA=${DATA:-data}
OUT=${OUT:-results}
if [[ "${QUICK:-0}" == "1" ]]; then
  SCALE=${SCALE:-0.3}; ASSIGN=${ASSIGN:-2}; NROLL=${NROLL:-200}
else
  SCALE=${SCALE:-1.0}; ASSIGN=${ASSIGN:-10}; NROLL=${NROLL:-1000}
fi
ARGS="--data $DATA --out $OUT --assign $ASSIGN --n-roll $NROLL"

echo "== tests"
$PY -m pytest -q tests

if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "== session pools (scale $SCALE)"
  $PY scripts/generate_testbed.py --out "$DATA" --scale "$SCALE"
fi

echo "== fit, estimate and cache every assignment"
$PY -m agentquote.experiments prepare $ARGS

for task in breakdown quotes choice ablation shift efficiency sensitivity case figures; do
  echo "== $task"
  $PY -m agentquote.experiments "$task" $ARGS
done

$PY scripts/summarize.py --out "$OUT"
echo "== done: results in $OUT/, figures in $OUT/figures/"
