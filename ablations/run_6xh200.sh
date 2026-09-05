#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
PAIR="${PAIR:-qwen}"
SEEDS_CSV="${SEEDS:-10,42}"
GPU_CSV="${RUN_GPUS:-0,1,2,3,4,5}"
RESULT_ROOT="${ABLATION_RESULT_ROOT:-${ROOT}/results/ablations}"
EVAL_ROOT="${ABLATION_EVAL_ROOT:-${ROOT}/benchmark_results/ablations}"
TABLE_ROOT="${ABLATION_TABLE_ROOT:-${EVAL_ROOT}/tables}"
EVAL_PYTHON="${EVAL_PYTHON:-${ROOT}/baselines/vendor/lm-evaluation-harness/.venv/bin/python}"
REUSE_MAIN_CST="${REUSE_MAIN_CST:-0}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS_CSV}"

# table|run_name|variant|q|gamma_min|gamma_max|sampling|grid|layers|tokens
CONFIGS=(
  "objective|hidden_mse|hidden_mse|2|1e-2|1e2|log_uniform||4|64"
  "objective|gram|gram|2|1e-2|1e2|log_uniform||4|64"
  "objective|cka|cka|2|1e-2|1e2|log_uniform||4|64"
  "objective|normalized_spectrum|normalized_spectrum|2|1e-2|1e2|log_uniform||4|64"
  "objective|direct_spectrum|direct_spectrum|2|1e-2|1e2|log_uniform||4|64"
  "objective|cst_main|cst|2|1e-2|1e2|log_uniform||4|64"
  "num_gamma|q1|cst|1|1e-2|1e2|log_uniform||4|64"
  "num_gamma|cst_main|cst|2|1e-2|1e2|log_uniform||4|64"
  "num_gamma|q4|cst|4|1e-2|1e2|log_uniform||4|64"
  "num_gamma|q8|cst|8|1e-2|1e2|log_uniform||4|64"
  "gamma|range_small_random|cst|2|1e-2|1e0|log_uniform||4|64"
  "gamma|range_mid_random|cst|2|1e-1|1e1|log_uniform||4|64"
  "gamma|cst_main|cst|2|1e-2|1e2|log_uniform||4|64"
  "gamma|range_large_random|cst|2|1e0|1e2|log_uniform||4|64"
  "gamma|range_broad_fixed|cst|2|1e-2|1e2|fixed_grid|1e-2,1e2|4|64"
  "layers|layers1|cst|2|1e-2|1e2|log_uniform||1|64"
  "layers|layers2|cst|2|1e-2|1e2|log_uniform||2|64"
  "layers|cst_main|cst|2|1e-2|1e2|log_uniform||4|64"
  "tokens|tokens32|cst|2|1e-2|1e2|log_uniform||4|32"
  "tokens|cst_main|cst|2|1e-2|1e2|log_uniform||4|64"
  "tokens|tokens128|cst|2|1e-2|1e2|log_uniform||4|128"
)

latest_checkpoint() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
items = []
for path in root.rglob("config.json"):
    parent = path.parent
    name = parent.name.removeprefix("checkpoint-")
    items.append((int(name) if name.isdigit() else -1, parent.stat().st_mtime_ns, parent))
if not items:
    raise SystemExit(f"No checkpoint under {root}")
print(max(items)[2])
PY
}

train_config() {
  local name="$1" variant="$2" q="$3" gmin="$4" gmax="$5" sampling="$6" grid="$7" layers="$8" tokens="$9" seed="${10}"
  local out="${RESULT_ROOT}/${PAIR}/${name}/seed${seed}"
  if [[ "${name}" == cst_main && "${REUSE_MAIN_CST}" == 1 ]]; then
    local source="${MAIN_RESULT_ROOT:-${ROOT}/results/reproduce_tables}/${PAIR}/cst/seed${seed}"
    [[ -f "${source}/TRAINING_COMPLETE" ]] || { echo "Missing reusable main CST run: ${source}" >&2; exit 6; }
    mkdir -p "$(dirname "${out}")"
    [[ -e "${out}" || -L "${out}" ]] || ln -s "${source}" "${out}"
    echo "[reuse train] ${out} -> ${source}"
    return
  fi
  [[ -f "${out}/TRAINING_COMPLETE" ]] && { echo "[skip train] ${name}/seed${seed}"; return; }
  RUN_NAME="${name}" RESULT_ROOT="${RESULT_ROOT}" RUN_GPUS="${GPU_CSV}" \
  CST_NUM_GAMMA="${q}" CST_GAMMA_MIN="${gmin}" CST_GAMMA_MAX="${gmax}" \
  CST_GAMMA_SAMPLING="${sampling}" CST_FIXED_GAMMA_GRID="${grid}" \
  CST_NUM_LAYERS="${layers}" CST_MAX_TOKENS="${tokens}" AUX_WEIGHT="${AUX_WEIGHT:-0.003}" \
    bash "${ROOT}/baselines/train_common_6xh200.sh" "${PAIR}" "${variant}" "${seed}"
  latest_checkpoint "${out}" >/dev/null
  touch "${out}/TRAINING_COMPLETE"
}

eval_config() {
  local name="$1" seed="$2" out="${EVAL_ROOT}/${PAIR}/${name}/seed${seed}"
  if [[ "${name}" == cst_main && "${REUSE_MAIN_CST}" == 1 ]]; then
    local source="${MAIN_EVAL_ROOT:-${ROOT}/benchmark_results/reproduce_tables}/${PAIR}/cst/seed${seed}"
    [[ -f "${source}/scores.json" ]] || { echo "Missing reusable main CST evaluation: ${source}" >&2; exit 7; }
    mkdir -p "$(dirname "${out}")"
    [[ -e "${out}" || -L "${out}" ]] || ln -s "${source}" "${out}"
    echo "[reuse eval] ${out} -> ${source}"
    return
  fi
  [[ -f "${out}/scores.json" ]] && { echo "[skip eval] ${name}/seed${seed}"; return; }
  local checkpoint
  checkpoint="$(latest_checkpoint "${RESULT_ROOT}/${PAIR}/${name}/seed${seed}")"
  "${EVAL_PYTHON}" "${ROOT}/baselines/eval_lm_harness.py" \
    --checkpoint "${checkpoint}" --output "${out}" --gpus "${GPU_CSV}"
}

for encoded in "${CONFIGS[@]}"; do
  IFS='|' read -r table name variant q gmin gmax sampling grid layers tokens <<< "${encoded}"
  for seed in "${SEED_LIST[@]}"; do
    case "${PHASE}" in
      train) train_config "${name}" "${variant}" "${q}" "${gmin}" "${gmax}" "${sampling}" "${grid}" "${layers}" "${tokens}" "${seed}" ;;
      eval) eval_config "${name}" "${seed}" ;;
      all) train_config "${name}" "${variant}" "${q}" "${gmin}" "${gmax}" "${sampling}" "${grid}" "${layers}" "${tokens}" "${seed}"; eval_config "${name}" "${seed}" ;;
      report) : ;;
      *) echo "Usage: $0 {train|eval|report|all}" >&2; exit 2 ;;
    esac
  done
done

if [[ "${PHASE}" == report || "${PHASE}" == all ]]; then
  mkdir -p "${TABLE_ROOT}"
  for table in objective num_gamma gamma layers tokens; do
    methods=()
    for encoded in "${CONFIGS[@]}"; do
      IFS='|' read -r group name _ <<< "${encoded}"
      [[ "${group}" == "${table}" ]] && methods+=("${name}")
    done
    python3 "${ROOT}/baselines/report_mean_std.py" --eval-root "${EVAL_ROOT}" \
      --output "${TABLE_ROOT}/${table}" --pairs "${PAIR}" \
      --methods "${methods[@]}" --seeds "${SEED_LIST[@]}"
  done
fi
