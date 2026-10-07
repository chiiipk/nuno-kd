#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
CONFIG="${2:-${ROOT}/configs/qwen25_tsd_structural.yaml}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"
python3 -c 'import yaml' >/dev/null 2>&1 || die "PyYAML is required (python3 -m pip install pyyaml)"

# Parse the YAML once and emit shell-quoted scalar values. shlex.quote prevents
# values from the config from being interpreted as shell syntax.
CONFIG_EXPORTS="$(python3 - "${CONFIG}" <<'PY'
import shlex
import sys
import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

def get(path):
    value = cfg
    for key in path.split("."):
        value = value[key]
    return value

def emit(name, value):
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif isinstance(value, list):
        value = ",".join(str(item) for item in value)
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")

mapping = {
    "EXP_NAME": "experiment.name",
    "PAIR": "experiment.pair",
    "TEACHER_MODEL": "models.teacher",
    "STUDENT_MODEL": "models.student",
    "MODEL_TYPE": "models.model_type",
    "TEACHER_FP16": "models.teacher_fp16",
    "DATA_REPO": "data.repo_id",
    "DATA_REVISION": "data.revision",
    "DATA_FILENAME": "data.filename",
    "EXPECTED_EXAMPLES": "data.expected_examples",
    "DATA_SHA256": "data.sha256",
    "RAW_DIR_REL": "data.raw_dir",
    "CANONICAL_REL": "data.canonical_file",
    "PROCESSED_ROOT_REL": "data.processed_root",
    "PROCESSED_DIR_REL": "data.processed_dir",
    "CONTRACT_REL": "data.contract_file",
    "PREPROCESS_WORKERS": "data.preprocess_workers",
    "DEV_EXAMPLES": "data.dev_examples",
    "GPU_CSV": "resources.gpus",
    "DEEPSPEED_REL": "resources.deepspeed_config",
    "DATALOADER_WORKERS": "resources.dataloader_workers",
    "SEEDS_CSV": "training.seeds",
    "METHODS_CSV": "training.methods",
    "EPOCHS": "training.epochs",
    "LEARNING_RATE": "training.learning_rate",
    "MIN_LEARNING_RATE": "training.min_learning_rate",
    "MICRO_BATCH": "training.micro_batch_size",
    "GRAD_ACC": "training.gradient_accumulation_steps",
    "WEIGHT_DECAY": "training.weight_decay",
    "GRAD_CLIP": "training.gradient_clip",
    "LR_SCHEDULER": "training.scheduler",
    "LR_WARMUP_RATIO": "training.lr_warmup_ratio",
    "MAX_LENGTH": "training.max_length",
    "MAX_PROMPT_LENGTH": "training.max_prompt_length",
    "OUTPUT_KD_ENABLED": "training.output_kd.enabled",
    "DISTILL_TYPE": "training.output_kd.type",
    "KD_RATIO": "training.output_kd.ratio",
    "SKEW_ALPHA": "training.output_kd.skew_alpha",
    "STUDENT_GENERATION": "training.output_kd.student_generation",
    "WEIGHT_HIDDEN_MSE": "training.auxiliary.hidden_mse",
    "WEIGHT_GRAM": "training.auxiliary.gram",
    "WEIGHT_CKA": "training.auxiliary.cka",
    "WEIGHT_NORMALIZED_SPECTRUM": "training.auxiliary.normalized_spectrum",
    "WEIGHT_DIRECT_SPECTRUM": "training.auxiliary.direct_spectrum",
    "WEIGHT_CST": "training.auxiliary.cst",
    "AUX_WARMUP": "training.auxiliary.warmup_steps",
    "AUX_RAMP": "training.auxiliary.ramp_steps",
    "CST_MAX_TOKENS": "training.auxiliary.max_tokens",
    "CST_NUM_LAYERS": "training.auxiliary.num_layers",
    "CST_LAYER_MIN": "training.auxiliary.layer_min",
    "CST_LAYER_MAX": "training.auxiliary.layer_max",
    "CST_GAMMA_MIN": "training.auxiliary.gamma_min",
    "CST_GAMMA_MAX": "training.auxiliary.gamma_max",
    "CST_NUM_GAMMA": "training.auxiliary.num_gamma_samples",
    "CST_GAMMA_SAMPLING": "training.auxiliary.gamma_sampling",
    "CST_DISTANCE": "training.auxiliary.distance",
    "EVAL_TASKS_CSV": "evaluation.tasks",
    "EVAL_TEACHER": "evaluation.include_teacher",
    "EVAL_STUDENT": "evaluation.include_base_student",
    "EVAL_PYTHON_REL": "evaluation.evaluator_python",
    "EVAL_MAX_NEW_TOKENS": "evaluation.max_new_tokens",
    "EVAL_GPU_MEMORY": "evaluation.gpu_memory_utilization",
    "TRAIN_ROOT_REL": "outputs.train_root",
    "EVAL_ROOT_REL": "outputs.eval_root",
    "TABLE_ROOT_REL": "outputs.table_root",
}
for shell_name, yaml_path in mapping.items():
    emit(shell_name, get(yaml_path))
# Optional: "<method>/seed<seed>: <global step>" for runs continued from an epoch checkpoint.
emit("RESUME_SPEC", [f"{run}:{step}" for run, step in (cfg["training"].get("resume") or {}).items()])
PY
)" || die "Could not parse YAML config: ${CONFIG}"
eval "${CONFIG_EXPORTS}"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${ROOT}" "$1" ;;
  esac
}

RAW_DIR="$(abs_path "${RAW_DIR_REL}")"
RAW_FILE="${RAW_DIR}/${DATA_FILENAME}"
CANONICAL_FILE="$(abs_path "${CANONICAL_REL}")"
PROCESSED_ROOT="$(abs_path "${PROCESSED_ROOT_REL}")"
PROCESSED_DIR="$(abs_path "${PROCESSED_DIR_REL}")"
CONTRACT_FILE="$(abs_path "${CONTRACT_REL}")"
DEEPSPEED_CONFIG="$(abs_path "${DEEPSPEED_REL}")"
EVAL_PYTHON="$(abs_path "${EVAL_PYTHON_REL}")"
TRAIN_ROOT="$(abs_path "${TRAIN_ROOT_REL}")"
EVAL_ROOT="$(abs_path "${EVAL_ROOT_REL}")"
TABLE_ROOT="$(abs_path "${TABLE_ROOT_REL}")"

IFS=',' read -r -a GPU_LIST <<< "${GPU_CSV}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS_CSV}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS_CSV}"
WORLD_SIZE="${#GPU_LIST[@]}"

[[ "${OUTPUT_KD_ENABLED}" == 1 ]] || die "This protocol requires output_kd.enabled=true"
[[ "${KD_RATIO}" == "1.0" || "${KD_RATIO}" == "1" ]] || die "This protocol requires output_kd.ratio=1"
if [[ "${DISTILL_TYPE}" == adaptive-* ]]; then
  [[ "${STUDENT_GENERATION}" == 1 ]] || die "adaptive output KD requires student_generation=true"
else
  [[ "${STUDENT_GENERATION}" == 0 ]] || die "student_generation=true requires an adaptive output KD type"
fi
[[ "${WORLD_SIZE}" -gt 0 ]] || die "resources.gpus is empty"
[[ -f "${DEEPSPEED_CONFIG}" ]] || die "DeepSpeed config not found: ${DEEPSPEED_CONFIG}"

EXPECTED_TASKS="gsm8k,gsm_plus,minerva_math,mbpp,sciq,mmlu_stem,mmlu_pro_math,bbh_cot_fewshot"
[[ "${EVAL_TASKS_CSV}" == "${EXPECTED_TASKS}" ]] || \
  die "evaluation.tasks must be exactly: ${EXPECTED_TASKS}"

method_weight() {
  case "$1" in
    hidden_mse) echo "${WEIGHT_HIDDEN_MSE}" ;;
    gram) echo "${WEIGHT_GRAM}" ;;
    cka) echo "${WEIGHT_CKA}" ;;
    normalized_spectrum) echo "${WEIGHT_NORMALIZED_SPECTRUM}" ;;
    direct_spectrum) echo "${WEIGHT_DIRECT_SPECTRUM}" ;;
    cst) echo "${WEIGHT_CST}" ;;
    *) die "Unsupported method in YAML: $1" ;;
  esac
}

resume_step_for() {
  local entry
  for entry in ${RESUME_SPEC//,/ }; do
    if [[ "${entry%%:*}" == "$1" ]]; then
      echo "${entry##*:}"
      return 0
    fi
  done
  return 0
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

prepare_data() {
  mkdir -p "${RAW_DIR}" "$(dirname "${CANONICAL_FILE}")" "${PROCESSED_DIR}"
  local url="https://huggingface.co/datasets/${DATA_REPO}/resolve/${DATA_REVISION}/${DATA_FILENAME}?download=true"

  if [[ ! -f "${RAW_FILE}" ]]; then
    echo "[download] ${DATA_REPO}@${DATA_REVISION}/${DATA_FILENAME}"
    curl -fL --retry 5 --retry-delay 2 "${url}" -o "${RAW_FILE}.partial"
    mv "${RAW_FILE}.partial" "${RAW_FILE}"
  else
    echo "[reuse raw] ${RAW_FILE}"
  fi

  local actual_sha
  actual_sha="$(sha256_file "${RAW_FILE}")"
  [[ "${actual_sha}" == "${DATA_SHA256}" ]] || \
    die "Dataset checksum mismatch: expected ${DATA_SHA256}, got ${actual_sha}"

  if [[ ! -f "${CANONICAL_FILE}" || ! -f "${CANONICAL_FILE}.source_sha256" || "$(<"${CANONICAL_FILE}.source_sha256")" != "${DATA_SHA256}" ]]; then
    echo "[convert] instruction/response -> prompt/generated_text"
    python3 - "${RAW_FILE}" "${CANONICAL_FILE}" "${EXPECTED_EXAMPLES}" <<'PY'
import json
import os
import sys

source, target, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
temporary = target + ".partial"
count = 0
with open(source, encoding="utf-8") as src, open(temporary, "w", encoding="utf-8") as dst:
    for line_number, line in enumerate(src, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt, response = row.get("instruction"), row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"{source}:{line_number}: expected string instruction/response")
        dst.write(json.dumps({"prompt": prompt, "generated_text": response}, ensure_ascii=False) + "\n")
        count += 1
if count != expected:
    raise SystemExit(f"Expected {expected} examples, found {count}")
os.replace(temporary, target)
print(f"[converted] {count} ordered examples -> {target}")
PY
    printf '%s' "${DATA_SHA256}" > "${CANONICAL_FILE}.source_sha256"
  else
    echo "[reuse canonical] ${CANONICAL_FILE}"
  fi

  python3 "${ROOT}/baselines/dataset_contract.py" create \
    --pair qwen --reference "${CANONICAL_FILE}" --output "${CONTRACT_FILE}"

  if ! python3 "${ROOT}/baselines/dataset_contract.py" check \
      --manifest "${CONTRACT_FILE}" --candidate "${PROCESSED_DIR}" >/dev/null 2>&1; then
    echo "[preprocess] tokenizer=${STUDENT_MODEL}"
    cd "${ROOT}"
    PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 tools/process_data_ultraInteract.py \
      --data-dir "${CANONICAL_FILE}" \
      --processed-data-dir "${PROCESSED_ROOT}" \
      --model-path "${STUDENT_MODEL}" \
      --model-type "${MODEL_TYPE}" \
      --data-process-workers "${PREPROCESS_WORKERS}" \
      --max-length "${MAX_LENGTH}" \
      --max-prompt-length "${MAX_PROMPT_LENGTH}" \
      --dev-num "${DEV_EXAMPLES}"
  else
    echo "[reuse processed] ${PROCESSED_DIR}"
  fi

  python3 "${ROOT}/baselines/dataset_contract.py" check \
    --manifest "${CONTRACT_FILE}" --candidate "${PROCESSED_DIR}"
}

latest_checkpoint() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for config in root.rglob("config.json"):
    checkpoint = config.parent
    suffix = checkpoint.name.removeprefix("checkpoint-")
    step = int(suffix) if suffix.isdigit() else -1
    candidates.append((step, checkpoint.stat().st_mtime_ns, checkpoint))
if not candidates:
    raise SystemExit(f"No Hugging Face checkpoint under {root}")
print(max(candidates)[2])
PY
}

train_one() {
  local method="$1" seed="$2" aux_weight run_dir
  aux_weight="$(method_weight "${method}")"
  run_dir="${TRAIN_ROOT}/${method}/seed${seed}"
  if [[ -f "${run_dir}/TRAINING_COMPLETE" ]]; then
    echo "[skip train] ${method}/seed${seed}"
    return
  fi
  mkdir -p "${run_dir}"
  export CUDA_VISIBLE_DEVICES="${GPU_CSV}"
  export WANDB_DISABLED=true
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  local cmd=(torchrun --standalone --nproc_per_node="${WORLD_SIZE}" "${ROOT}/finetune.py"
    --base-path "${ROOT}"
    --model-path "${STUDENT_MODEL}"
    --teacher-model-path "${TEACHER_MODEL}"
    --ckpt-name qwen2.5-1.5b-it
    --teacher-ckpt-name qwen2.5-14b-it
    --model-type "${MODEL_TYPE}"
    --teacher-model-type "${MODEL_TYPE}"
    --n-gpu "${WORLD_SIZE}"
    --data-dir "${PROCESSED_DIR}"
    --num-workers "${DATALOADER_WORKERS}"
    --train-num -1 --dev-num -1
    --lr "${LEARNING_RATE}" --lr-min "${MIN_LEARNING_RATE}"
    --batch-size "${MICRO_BATCH}"
    --eval-batch-size "${MICRO_BATCH}"
    --gradient-accumulation-steps "${GRAD_ACC}"
    --gradient-checkpointing
    --warmup-ratio "${LR_WARMUP_RATIO}"
    --lr-decay-style "${LR_SCHEDULER}"
    --weight-decay "${WEIGHT_DECAY}"
    --clip-grad "${GRAD_CLIP}"
    --epochs "${EPOCHS}"
    --kd-ratio "${KD_RATIO}"
    --temperature 1.0
    --max-length "${MAX_LENGTH}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --do-train --save-interval -1 --eval-interval -1
    --log-interval 10 --mid-log-num -1
    --save "${run_dir}" --seed "${seed}"
    --deepspeed --deepspeed_config "${DEEPSPEED_CONFIG}"
    --type "${DISTILL_TYPE}" --skew-alpha "${SKEW_ALPHA}"
    --do-sample --top-k 0 --top-p 1.0
    --nnm --loss-variant "${method}"
    --nnm-ratio "${aux_weight}"
    --cst-loss-weight "${aux_weight}"
    --nnm-warmup-steps "${AUX_WARMUP}"
    --nnm-ramp-steps "${AUX_RAMP}"
    --cst-max-tokens "${CST_MAX_TOKENS}"
    --cst-num-layers "${CST_NUM_LAYERS}"
    --cst-layer-min "${CST_LAYER_MIN}"
    --cst-layer-max "${CST_LAYER_MAX}"
    --cst-gamma-min "${CST_GAMMA_MIN}"
    --cst-gamma-max "${CST_GAMMA_MAX}"
    --cst-num-gamma-samples "${CST_NUM_GAMMA}"
    --cst-gamma-sampling "${CST_GAMMA_SAMPLING}"
    --cst-distance "${CST_DISTANCE}")
  [[ "${TEACHER_FP16}" == 1 ]] && cmd+=(--teacher-model-fp16)
  if [[ "${STUDENT_GENERATION}" == 1 ]]; then
    cmd+=(--student-gen --gen-num-beams 1 --gen-top-p 1.0
      --init-threshold 0.0 --loss-eps 0.1 --capacity 1000
      --replay-ratio decreasing --mixed-alpha 0.5)
  fi

  local resume_step archive name
  resume_step="$(resume_step_for "${method}/seed${seed}")"
  if [[ -n "${resume_step}" ]]; then
    [[ -f "${run_dir}/${resume_step}/pytorch_model.bin" ]] || \
      die "Resume checkpoint missing: ${run_dir}/${resume_step}/pytorch_model.bin"
    cmd+=(--resume-ckpt "${run_dir}/${resume_step}" --resume-global-step "${resume_step}")
    # Keep the interrupted attempt's logs; tee and finetune.py rewrite them.
    archive="${run_dir}/pre_resume_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${archive}"
    for name in train.log log.txt args.json; do
      if [[ -f "${run_dir}/${name}" ]]; then mv "${run_dir}/${name}" "${archive}/"; fi
    done
  fi

  {
    echo "method=${method} seed=${seed} output_kd_ratio=${KD_RATIO} auxiliary_weight=${aux_weight}"
    printf 'COMMAND: '; printf '%q ' "${cmd[@]}"; printf '\n\n'
    "${cmd[@]}"
  } 2>&1 | tee "${run_dir}/train.log"
  latest_checkpoint "${run_dir}" >/dev/null
  touch "${run_dir}/TRAINING_COMPLETE"
}

eval_checkpoint() {
  local checkpoint="$1" output="$2"
  if [[ -f "${output}/scores.json" ]]; then
    echo "[skip eval] ${output}"
    return
  fi
  "${EVAL_PYTHON}" "${ROOT}/baselines/eval_lm_harness.py" \
    --checkpoint "${checkpoint}" --output "${output}" --gpus "${GPU_CSV}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY}" --reuse-complete
}

eval_fixed_model() {
  local role="$1" checkpoint="$2" first_seed="${SEED_LIST[0]}"
  local first_dir="${EVAL_ROOT}/${PAIR}/${role}/seed${first_seed}"
  eval_checkpoint "${checkpoint}" "${first_dir}"
  local seed target
  for seed in "${SEED_LIST[@]:1}"; do
    target="${EVAL_ROOT}/${PAIR}/${role}/seed${seed}"
    mkdir -p "${target}"
    cp "${first_dir}/scores.json" "${target}/scores.json"
  done
}

run_train() {
  prepare_data
  local method seed
  for method in "${METHOD_LIST[@]}"; do
    method_weight "${method}" >/dev/null
    for seed in "${SEED_LIST[@]}"; do
      train_one "${method}" "${seed}"
    done
  done
}

run_eval() {
  [[ -x "${EVAL_PYTHON}" ]] || \
    die "Evaluator missing: ${EVAL_PYTHON}; run: bash baselines/setup_env.sh evaluator"
  # All models run concurrently; eval_lm_harness.py hands every task one GPU
  # from a shared lease pool, so the GPUs stay busy across model boundaries.
  local pids=() pid failed=0
  [[ "${EVAL_TEACHER}" == 1 ]] && { eval_fixed_model teacher "${TEACHER_MODEL}" & pids+=("$!"); }
  [[ "${EVAL_STUDENT}" == 1 ]] && { eval_fixed_model student "${STUDENT_MODEL}" & pids+=("$!"); }
  local method seed checkpoint
  for method in "${METHOD_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      checkpoint="$(latest_checkpoint "${TRAIN_ROOT}/${method}/seed${seed}")"
      eval_checkpoint "${checkpoint}" "${EVAL_ROOT}/${PAIR}/${method}/seed${seed}" &
      pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  [[ "${failed}" == 0 ]] || die "At least one evaluation failed; see the eval.log files under ${EVAL_ROOT}"
}

run_report() {
  mkdir -p "${TABLE_ROOT}"
  local methods=("${METHOD_LIST[@]}")
  [[ "${EVAL_STUDENT}" == 1 ]] && methods=(student "${methods[@]}")
  [[ "${EVAL_TEACHER}" == 1 ]] && methods=(teacher "${methods[@]}")
  python3 "${ROOT}/baselines/report_mean_std.py" \
    --eval-root "${EVAL_ROOT}" --output "${TABLE_ROOT}" \
    --pairs "${PAIR}" --methods "${methods[@]}" --seeds "${SEED_LIST[@]}"
}

print_config() {
  echo "experiment=${EXP_NAME}"
  echo "teacher=${TEACHER_MODEL}"
  echo "student=${STUDENT_MODEL}"
  echo "dataset=${DATA_REPO}@${DATA_REVISION} (${EXPECTED_EXAMPLES} examples)"
  echo "methods=${METHODS_CSV}"
  echo "seeds=${SEEDS_CSV} gpus=${GPU_CSV} world_size=${WORLD_SIZE}"
  echo "output_kd=${DISTILL_TYPE} ratio=${KD_RATIO} for every method"
  local method
  for method in "${METHOD_LIST[@]}"; do
    echo "auxiliary_weight[${method}]=$(method_weight "${method}")"
  done
  echo "benchmarks=${EVAL_TASKS_CSV}"
  echo "train_root=${TRAIN_ROOT}"
  echo "eval_root=${EVAL_ROOT}"
}

case "${PHASE}" in
  check) print_config ;;
  prepare) print_config; prepare_data ;;
  train) print_config; run_train ;;
  eval) print_config; run_eval ;;
  report) print_config; run_report ;;
  all) print_config; run_train; run_eval; run_report ;;
  *) die "Usage: $0 {check|prepare|train|eval|report|all} [config.yaml]" ;;
esac
