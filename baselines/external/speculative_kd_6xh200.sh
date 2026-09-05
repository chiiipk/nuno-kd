#!/usr/bin/env bash
set -euo pipefail

PAIR="${1:?pair required}"
SEED="${2:?seed required}"
OUT="${3:?output directory required}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKD_ROOT="${ROOT}/baselines/vendor/google-research/speculative_kd"
SKD_PYTHON="${SKD_PYTHON:-${SKD_ROOT}/.venv/bin/python}"
if [[ ! -x "${SKD_PYTHON}" ]]; then
  echo "Missing Speculative-KD Python: ${SKD_PYTHON}; run 'bash baselines/setup_env.sh speculative_kd'" >&2
  exit 3
fi
case "${PAIR}" in
  qwen)
    STUDENT="${QWEN_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    TEACHER="${QWEN_TEACHER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
    SKD_TRAIN_JSONL="${SKD_QWEN_TRAIN_JSONL:-}"
    SKD_VALIDATION_JSONL="${SKD_QWEN_VALIDATION_JSONL:-${SKD_TRAIN_JSONL}}"
    DATASET_MANIFEST="${QWEN_DATASET_MANIFEST:-${ROOT}/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct/dataset_contract.json}"
    ;;
  qwen3)
    STUDENT="${QWEN3_STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
    TEACHER="${QWEN3_TEACHER_MODEL:-Qwen/Qwen3-8B}"
    SKD_TRAIN_JSONL="${SKD_QWEN3_TRAIN_JSONL:-}"
    SKD_VALIDATION_JSONL="${SKD_QWEN3_VALIDATION_JSONL:-${SKD_TRAIN_JSONL}}"
    DATASET_MANIFEST="${QWEN3_DATASET_MANIFEST:-${ROOT}/processed_data/ultraInteract/Qwen/Qwen3-8B/dataset_contract.json}"
    ;;
  *) echo "Unknown pair: ${PAIR}" >&2; exit 2 ;;
esac
: "${SKD_TRAIN_JSONL:?Set the pair-specific SKD_QWEN_TRAIN_JSONL or SKD_QWEN3_TRAIN_JSONL}"
mkdir -p "${OUT}"
python3 "${ROOT}/baselines/dataset_contract.py" check \
  --manifest "${DATASET_MANIFEST}" --candidate "${SKD_TRAIN_JSONL}" \
  2>&1 | tee "${OUT}/dataset_check.log"
CONFIG="${OUT}/skd_config.yaml"
"${SKD_PYTHON}" - "${ROOT}/baselines/configs/speculative_kd_6xh200.yaml" "${CONFIG}" \
  "${SEED}" "${STUDENT}" "${TEACHER}" "${OUT}" <<'PY'
import sys, yaml
source, target, seed, student, teacher, output = sys.argv[1:]
with open(source) as handle:
    cfg = yaml.safe_load(handle)
cfg["task_params"]["task_type"] = "ultrainteract"
cfg["training_params"]["seed"] = int(seed)
cfg["model_params"]["checkpoint_template"] = teacher
cfg["model_params"]["assistant_checkpoint_template"] = student
cfg["model_params"]["assistant_checkpoint_override"] = student
cfg["model_params"]["tokenizer_name"] = student
cfg["exec_params"]["output_root"] = output
with open(target, "w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

cd "${SKD_ROOT}"
export SKD_TRAIN_JSONL SKD_VALIDATION_JSONL
export PATH="${SKD_ROOT}/.venv/bin:${PATH}"
{
  echo "pair=${PAIR} method=speculative_kd seed=${SEED}"
  echo "train_jsonl=${SKD_TRAIN_JSONL} validation_jsonl=${SKD_VALIDATION_JSONL}"
  echo "COMMAND: ${SKD_PYTHON} train/run_kd_train.py ${CONFIG}"
  echo
  "${SKD_PYTHON}" train/run_kd_train.py "${CONFIG}"
} 2>&1 | tee "${OUT}/train.log"
