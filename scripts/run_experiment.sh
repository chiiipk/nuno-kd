#!/usr/bin/env bash
# Run one distillation experiment described by a YAML config:
#   prepare: download, convert and tokenize the teacher-generation dataset
#   train:   train every method x seed with all configured GPUs (one DDP job each)
#   eval:    evaluate the teacher, base student and trained checkpoints
#   report:  aggregate the scores into mean +- std tables
# Usage: bash scripts/run_experiment.sh {check|prepare|train|eval|report|all} [config.yaml]
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

with open(sys.argv[1], encoding="utf-8") as handle:
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
    print(f"{name}={shlex.quote(str(value))}")

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
    "MICRO_BATCH": "training.micro_batch_size",
    "GRAD_ACC": "training.gradient_accumulation_steps",
    "WEIGHT_DECAY": "training.weight_decay",
    "GRAD_CLIP": "training.gradient_clip",
    "LR_SCHEDULER": "training.scheduler",
    "LR_WARMUP_RATIO": "training.lr_warmup_ratio",
    "MAX_LENGTH": "training.max_length",
    "MAX_PROMPT_LENGTH": "training.max_prompt_length",
    "KD_TYPE": "training.output_kd.type",
    "KD_RATIO": "training.output_kd.ratio",
    "SKEW_ALPHA": "training.output_kd.skew_alpha",
    "AUX_WARMUP": "training.auxiliary.warmup_steps",
    "AUX_RAMP": "training.auxiliary.ramp_steps",
    "AUX_MAX_TOKENS": "training.auxiliary.max_tokens",
    "AUX_NUM_LAYERS": "training.auxiliary.num_layers",
    "AUX_LAYER_MIN": "training.auxiliary.layer_min",
    "AUX_LAYER_MAX": "training.auxiliary.layer_max",
    "CST_GAMMA_MIN": "training.auxiliary.cst.gamma_min",
    "CST_GAMMA_MAX": "training.auxiliary.cst.gamma_max",
    "CST_NUM_GAMMA": "training.auxiliary.cst.num_gamma_samples",
    "CST_DISTANCE": "training.auxiliary.cst.distance",
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
# Auxiliary-loss weight per method, as "<method>:<weight>".
emit("WEIGHT_SPEC", [f"{method}:{weight}" for method, weight in cfg["training"]["auxiliary"]["weights"].items()])
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

RAW_FILE="$(abs_path "${RAW_DIR_REL}")/${DATA_FILENAME}"
CANONICAL_FILE="$(abs_path "${CANONICAL_REL}")"
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

[[ "${WORLD_SIZE}" -gt 0 ]] || die "resources.gpus is empty"
[[ -f "${DEEPSPEED_CONFIG}" ]] || die "DeepSpeed config not found: ${DEEPSPEED_CONFIG}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Look up "<key>:<value>" in a comma-separated spec; prints nothing if absent.
spec_value() {
  local entry
  for entry in ${1//,/ }; do
    if [[ "${entry%%:*}" == "$2" ]]; then
      echo "${entry##*:}"
      return 0
    fi
  done
}

method_weight() {
  local weight
  weight="$(spec_value "${WEIGHT_SPEC}" "$1")"
  [[ -n "${weight}" ]] || die "No training.auxiliary.weights entry for method: $1"
  echo "${weight}"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

prepare_data() {
  mkdir -p "$(dirname "${RAW_FILE}")" "$(dirname "${CANONICAL_FILE}")" "${PROCESSED_DIR}"
  if [[ ! -f "${RAW_FILE}" ]]; then
    echo "[download] ${DATA_REPO}@${DATA_REVISION}/${DATA_FILENAME}"
    curl -fL --retry 5 --retry-delay 2 \
      "https://huggingface.co/datasets/${DATA_REPO}/resolve/${DATA_REVISION}/${DATA_FILENAME}?download=true" \
      -o "${RAW_FILE}.partial"
    mv "${RAW_FILE}.partial" "${RAW_FILE}"
  else
    echo "[reuse raw] ${RAW_FILE}"
  fi

  local actual_sha
  actual_sha="$(sha256_file "${RAW_FILE}")"
  [[ "${actual_sha}" == "${DATA_SHA256}" ]] || \
    die "Dataset checksum mismatch: expected ${DATA_SHA256}, got ${actual_sha}"

  if [[ ! -f "${CANONICAL_FILE}" || "$(cat "${CANONICAL_FILE}.source_sha256" 2>/dev/null)" != "${DATA_SHA256}" ]]; then
    echo "[convert] instruction/response -> prompt/generated_text"
    python3 -m nuno_kd.data.prepare convert \
      --source "${RAW_FILE}" --target "${CANONICAL_FILE}" --expected "${EXPECTED_EXAMPLES}"
    printf '%s' "${DATA_SHA256}" > "${CANONICAL_FILE}.source_sha256"
  else
    echo "[reuse canonical] ${CANONICAL_FILE}"
  fi

  python3 -m nuno_kd.data.contract create \
    --pair "${PAIR}" --reference "${CANONICAL_FILE}" --output "${CONTRACT_FILE}"

  if ! python3 -m nuno_kd.data.contract check \
      --manifest "${CONTRACT_FILE}" --candidate "${PROCESSED_DIR}" >/dev/null 2>&1; then
    echo "[tokenize] tokenizer=${STUDENT_MODEL}"
    python3 -m nuno_kd.data.prepare tokenize \
      --data-file "${CANONICAL_FILE}" \
      --output-dir "${PROCESSED_DIR}" \
      --model-path "${STUDENT_MODEL}" \
      --max-prompt-length "${MAX_PROMPT_LENGTH}" \
      --dev-num "${DEV_EXAMPLES}" \
      --workers "${PREPROCESS_WORKERS}"
  else
    echo "[reuse processed] ${PROCESSED_DIR}"
  fi

  python3 -m nuno_kd.data.contract check \
    --manifest "${CONTRACT_FILE}" --candidate "${PROCESSED_DIR}"
}

# The checkpoint directory with the highest global step under $1.
latest_checkpoint() {
  local dir best="" best_step=-1 step
  for dir in "$1"/*/; do
    dir="${dir%/}"
    step="$(basename "${dir}")"
    if [[ "${step}" =~ ^[0-9]+$ && -f "${dir}/config.json" ]] && (( step > best_step )); then
      best="${dir}" best_step="${step}"
    fi
  done
  [[ -n "${best}" ]] || die "No checkpoint under $1"
  echo "${best}"
}

# Block until every GPU in resources.gpus is completely idle: no compute process
# of any user or project, and less than 2048 MiB in use. Training never shares
# a GPU with another job.
wait_for_idle_gpus() {
  local announced=0 gpu busy
  while true; do
    busy=""
    for gpu in "${GPU_LIST[@]}"; do
      if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}")" ]] || \
          (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}") >= 2048 )); then
        busy+=" ${gpu}"
      fi
    done
    [[ -z "${busy}" ]] && return
    if [[ "${announced}" == 0 ]]; then
      echo "[wait] GPUs busy:${busy}; waiting until all of ${GPU_CSV} are idle"
      announced=1
    fi
    sleep 30
  done
}

train_one() {
  local method="$1" seed="$2" aux_weight run_dir
  aux_weight="$(method_weight "${method}")"
  run_dir="${TRAIN_ROOT}/${method}/seed${seed}"
  if [[ -f "${run_dir}/TRAINING_COMPLETE" ]]; then
    echo "[skip train] ${method}/seed${seed}"
    return
  fi
  wait_for_idle_gpus
  mkdir -p "${run_dir}"

  local cmd=(torchrun --standalone --nproc_per_node="${WORLD_SIZE}" -m nuno_kd.train
    --model-path "${STUDENT_MODEL}"
    --teacher-model-path "${TEACHER_MODEL}"
    --model-type "${MODEL_TYPE}"
    --data-dir "${PROCESSED_DIR}"
    --num-workers "${DATALOADER_WORKERS}"
    --max-length "${MAX_LENGTH}"
    --lr "${LEARNING_RATE}"
    --batch-size "${MICRO_BATCH}"
    --eval-batch-size "${MICRO_BATCH}"
    --gradient-accumulation-steps "${GRAD_ACC}"
    --gradient-checkpointing
    --warmup-ratio "${LR_WARMUP_RATIO}"
    --lr-decay-style "${LR_SCHEDULER}"
    --weight-decay "${WEIGHT_DECAY}"
    --clip-grad "${GRAD_CLIP}"
    --epochs "${EPOCHS}"
    --seed "${seed}"
    --kd-type "${KD_TYPE}" --kd-ratio "${KD_RATIO}" --skew-alpha "${SKEW_ALPHA}"
    --aux-loss "${method}" --aux-weight "${aux_weight}"
    --aux-warmup-steps "${AUX_WARMUP}" --aux-ramp-steps "${AUX_RAMP}"
    --aux-max-tokens "${AUX_MAX_TOKENS}" --aux-num-layers "${AUX_NUM_LAYERS}"
    --aux-layer-min "${AUX_LAYER_MIN}" --aux-layer-max "${AUX_LAYER_MAX}"
    --cst-gamma-min "${CST_GAMMA_MIN}" --cst-gamma-max "${CST_GAMMA_MAX}"
    --cst-num-gamma-samples "${CST_NUM_GAMMA}" --cst-distance "${CST_DISTANCE}"
    --save "${run_dir}"
    --deepspeed --deepspeed_config "${DEEPSPEED_CONFIG}")
  [[ "${TEACHER_FP16}" == 1 ]] && cmd+=(--teacher-fp16)

  local resume_step archive name
  resume_step="$(spec_value "${RESUME_SPEC}" "${method}/seed${seed}")"
  if [[ -n "${resume_step}" ]]; then
    [[ -f "${run_dir}/${resume_step}/pytorch_model.bin" ]] || \
      die "Resume checkpoint missing: ${run_dir}/${resume_step}/pytorch_model.bin"
    cmd+=(--resume-ckpt "${run_dir}/${resume_step}" --resume-global-step "${resume_step}")
    # Keep the interrupted attempt's logs; tee and the trainer rewrite them.
    archive="${run_dir}/pre_resume_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${archive}"
    for name in train.log log.txt args.json; do
      if [[ -f "${run_dir}/${name}" ]]; then mv "${run_dir}/${name}" "${archive}/"; fi
    done
  fi

  {
    echo "method=${method} seed=${seed} output_kd_ratio=${KD_RATIO} auxiliary_weight=${aux_weight}"
    printf 'COMMAND: '; printf '%q ' "${cmd[@]}"; printf '\n\n'
    # PCI bus order makes the CUDA indices match nvidia-smi's GPU ids.
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_CSV}" \
      WANDB_DISABLED=true PYTHONUNBUFFERED=1 "${cmd[@]}"
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
  "${EVAL_PYTHON}" "${ROOT}/evaluation/eval_lm_harness.py" \
    --checkpoint "${checkpoint}" --output "${output}" --gpus "${GPU_CSV}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY}" --reuse-complete
}

# The teacher and base student do not depend on the seed: evaluate them once
# and copy the scores to the other seeds.
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
    die "Evaluator missing: ${EVAL_PYTHON}; run: bash evaluation/setup.sh"
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
  python3 "${ROOT}/evaluation/report.py" \
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
  echo "output_kd=${KD_TYPE} ratio=${KD_RATIO} for every method"
  local method
  for method in "${METHOD_LIST[@]}"; do
    echo "auxiliary_weight[${method}]=$(method_weight "${method}")"
  done
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
