#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
GPU_CSV="${RUN_GPUS:-0,1,2,3,4,5}"
SEEDS_CSV="${SEEDS:-10,42}"
PAIRS_CSV="${PAIRS:-qwen,qwen3}"
METHODS_CSV="${METHODS:-seqkd,supervised_kd,distillm,speculative_kd,minillm,gkd,csd,amid,cst}"
REPORT_METHODS_CSV="${REPORT_METHODS:-teacher,student,${METHODS_CSV}}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/reproduce_tables}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/benchmark_results/reproduce_tables}"
TABLE_ROOT="${TABLE_ROOT:-${ROOT}/benchmark_results/reproduce_tables/tables}"
EVAL_PYTHON="${EVAL_PYTHON:-${ROOT}/baselines/vendor/lm-evaluation-harness/.venv/bin/python}"

if [[ -f "${ROOT}/baselines/lm_eval_commit.txt" ]]; then
  export LM_EVAL_COMMIT="$(tr -d '[:space:]' < "${ROOT}/baselines/lm_eval_commit.txt")"
fi

IFS=',' read -r -a SEED_LIST <<< "${SEEDS_CSV}"
IFS=',' read -r -a PAIR_LIST <<< "${PAIRS_CSV}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS_CSV}"
IFS=',' read -r -a REPORT_METHOD_LIST <<< "${REPORT_METHODS_CSV}"

COMMON_METHODS=" seqkd supervised_kd distillm gkd csd amid cst "

latest_checkpoint() {
  local run_dir="$1"
  python3 - "${run_dir}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.rglob("*"):
    if not path.is_dir():
        continue
    if not (path / "config.json").exists():
        continue
    name = path.name.removeprefix("checkpoint-")
    step = int(name) if name.isdigit() else -1
    candidates.append((step, path.stat().st_mtime_ns, path))
if (root / "config.json").exists():
    candidates.append((-1, root.stat().st_mtime_ns, root))
if not candidates:
    raise SystemExit(f"No Hugging Face checkpoint found under {root}")
print(max(candidates)[2])
PY
}

train_one() {
  local pair="$1" method="$2" seed="$3"
  local out="${RESULT_ROOT}/${pair}/${method}/seed${seed}"
  if [[ -f "${out}/TRAINING_COMPLETE" ]]; then
    echo "[skip train] ${pair}/${method}/seed${seed}"
    return
  fi
  if [[ "${COMMON_METHODS}" == *" ${method} "* ]]; then
    RESULT_ROOT="${RESULT_ROOT}" RUN_GPUS="${GPU_CSV}" \
      bash "${ROOT}/baselines/train_common_6xh200.sh" "${pair}" "${method}" "${seed}"
  else
    local adapter="${ROOT}/baselines/external/${method}_6xh200.sh"
    if [[ ! -x "${adapter}" ]]; then
      echo "Missing executable faithful adapter: ${adapter}" >&2
      exit 4
    fi
    RUN_GPUS="${GPU_CSV}" RESULT_ROOT="${RESULT_ROOT}" \
      "${adapter}" "${pair}" "${seed}" "${out}"
  fi
  latest_checkpoint "${out}" >/dev/null
  touch "${out}/TRAINING_COMPLETE"
}

eval_one() {
  local pair="$1" method="$2" seed="$3"
  local run_dir="${RESULT_ROOT}/${pair}/${method}/seed${seed}"
  local eval_dir="${EVAL_ROOT}/${pair}/${method}/seed${seed}"
  if [[ -f "${eval_dir}/scores.json" ]]; then
    echo "[skip eval] ${pair}/${method}/seed${seed}"
    return
  fi
  local checkpoint
  checkpoint="$(latest_checkpoint "${run_dir}")"
  "${EVAL_PYTHON}" "${ROOT}/baselines/eval_lm_harness.py" \
    --checkpoint "${checkpoint}" --output "${eval_dir}" --gpus "${GPU_CSV}"
}

eval_base_models() {
  local pair="$1" first_seed="${SEED_LIST[0]}" teacher student
  case "${pair}" in
    qwen)
      teacher="${QWEN_TEACHER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
      student="${QWEN_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
      ;;
    qwen3)
      teacher="${QWEN3_TEACHER_MODEL:-Qwen/Qwen3-8B}"
      student="${QWEN3_STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
      ;;
  esac
  for role in teacher student; do
    local checkpoint="${!role}"
    local first_dir="${EVAL_ROOT}/${pair}/${role}/seed${first_seed}"
    if [[ ! -f "${first_dir}/scores.json" ]]; then
      "${EVAL_PYTHON}" "${ROOT}/baselines/eval_lm_harness.py" \
        --checkpoint "${checkpoint}" --output "${first_dir}" --gpus "${GPU_CSV}"
    fi
    # Teacher/student are fixed checkpoints, not independently trained seeds.
    for seed in "${SEED_LIST[@]:1}"; do
      local target="${EVAL_ROOT}/${pair}/${role}/seed${seed}"
      mkdir -p "${target}"
      cp "${first_dir}/scores.json" "${target}/scores.json"
    done
  done
}

case "${PHASE}" in
  train|all)
    for pair in "${PAIR_LIST[@]}"; do
      for method in "${METHOD_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
          train_one "${pair}" "${method}" "${seed}"
        done
      done
    done
    [[ "${PHASE}" == train ]] && exit 0
    ;;
  eval|report) ;;
  *) echo "Usage: $0 {train|eval|report|all}" >&2; exit 2 ;;
esac

if [[ "${PHASE}" == eval || "${PHASE}" == all ]]; then
  if [[ ! -x "${EVAL_PYTHON}" ]]; then
    echo "Missing evaluator Python: ${EVAL_PYTHON}; run 'bash baselines/setup_env.sh evaluator'" >&2
    exit 5
  fi
  for pair in "${PAIR_LIST[@]}"; do
    eval_base_models "${pair}"
    for method in "${METHOD_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        eval_one "${pair}" "${method}" "${seed}"
      done
    done
  done
  [[ "${PHASE}" == eval ]] && exit 0
fi

mkdir -p "${TABLE_ROOT}"
python3 "${ROOT}/baselines/report_mean_std.py" \
  --eval-root "${EVAL_ROOT}" --output "${TABLE_ROOT}" \
  --pairs "${PAIR_LIST[@]}" --methods "${REPORT_METHOD_LIST[@]}" --seeds "${SEED_LIST[@]}"
