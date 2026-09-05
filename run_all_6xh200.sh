#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-all}"
export RUN_GPUS="${RUN_GPUS:-0,1,2,3,4,5}"
export SEEDS="${SEEDS:-10,42}"
export PAIRS="${PAIRS:-qwen,gemma}"

setup_all() {
  bash "${ROOT}/baselines/setup_env.sh" legacy
  bash "${ROOT}/baselines/setup_env.sh" speculative_kd
  bash "${ROOT}/baselines/setup_env.sh" evaluator
}

preflight_all() {
  python3 "${ROOT}/baselines/preflight_reproduction.py" --stage static
  python3 "${ROOT}/baselines/preflight_reproduction.py" --stage train
  "${ROOT}/baselines/vendor/lm-evaluation-harness/.venv/bin/python" \
    "${ROOT}/baselines/preflight_reproduction.py" --stage eval
}

train_all() {
  bash "${ROOT}/baselines/reproduce_2tables_6xh200.sh" train
  REUSE_MAIN_CST=1 bash "${ROOT}/ablations/run_6xh200.sh" train
}

eval_all() {
  bash "${ROOT}/baselines/reproduce_2tables_6xh200.sh" eval
  REUSE_MAIN_CST=1 bash "${ROOT}/ablations/run_6xh200.sh" eval
}

report_all() {
  bash "${ROOT}/baselines/reproduce_2tables_6xh200.sh" report
  REUSE_MAIN_CST=1 bash "${ROOT}/ablations/run_6xh200.sh" report
}

case "${PHASE}" in
  setup) setup_all ;;
  preflight) preflight_all ;;
  train) preflight_all; train_all ;;
  eval) preflight_all; eval_all ;;
  report) report_all ;;
  all) preflight_all; train_all; eval_all; report_all ;;
  *) echo "Usage: $0 {setup|preflight|train|eval|report|all}" >&2; exit 2 ;;
esac
