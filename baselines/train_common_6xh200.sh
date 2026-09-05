#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIR="${1:?pair required: qwen or gemma}"
METHOD="${2:?method required}"
SEED="${3:?seed required}"

GPU_CSV="${RUN_GPUS:-0,1,2,3,4,5}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_CSV}"
export CUDA_VISIBLE_DEVICES="${GPU_CSV}"
WORLD_SIZE="${#GPU_LIST[@]}"

case "${PAIR}" in
  qwen)
    STUDENT_MODEL="${QWEN_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    TEACHER_MODEL="${QWEN_TEACHER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
    STUDENT_NAME="qwen2.5-1.5b-it"
    TEACHER_NAME="qwen2.5-14b-it"
    DATA_DIR="${QWEN_DATA_DIR:-${ROOT}/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct}"
    DATASET_MANIFEST="${QWEN_DATASET_MANIFEST:-${DATA_DIR}/dataset_contract.json}"
    ;;
  gemma)
    STUDENT_MODEL="${GEMMA_STUDENT_MODEL:-google/gemma-2-2b-it}"
    TEACHER_MODEL="${GEMMA_TEACHER_MODEL:-google/gemma-2-9b-it}"
    STUDENT_NAME="gemma2-2b-it"
    TEACHER_NAME="gemma2-9b-it"
    DATA_DIR="${GEMMA_DATA_DIR:-${ROOT}/processed_data/ultraInteract/google/gemma-2-9b-it}"
    DATASET_MANIFEST="${GEMMA_DATASET_MANIFEST:-${DATA_DIR}/dataset_contract.json}"
    ;;
  *) echo "Unknown pair: ${PAIR}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  seqkd)          DISTILL_TYPE=fkl;           KD_RATIO=0.0; STUDENT_GEN=0 ;;
  supervised_kd)  DISTILL_TYPE=fkl;           KD_RATIO=1.0; STUDENT_GEN=0 ;;
  distillm)        DISTILL_TYPE=adaptive-sfkl; KD_RATIO=1.0; STUDENT_GEN=1 ;;
  gkd)             DISTILL_TYPE=jsd-mixed;     KD_RATIO=1.0; STUDENT_GEN=1 ;;
  csd)             DISTILL_TYPE=csd;           KD_RATIO=1.0; STUDENT_GEN=0 ;;
  amid)            DISTILL_TYPE=amid;          KD_RATIO=1.0; STUDENT_GEN=0 ;;
  cst)             DISTILL_TYPE=adaptive-sfkl; KD_RATIO=1.0; STUDENT_GEN=1 ;;
  hidden_mse|gram|cka|normalized_spectrum|direct_spectrum)
                    DISTILL_TYPE=adaptive-sfkl; KD_RATIO=1.0; STUDENT_GEN=1 ;;
  *)
    echo "train_common supports: seqkd supervised_kd distillm gkd csd amid cst" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing full processed dataset: ${DATA_DIR}" >&2
  exit 3
fi
RUN_NAME="${RUN_NAME:-${METHOD}}"
OUT="${RESULT_ROOT:-${ROOT}/results/reproduce_tables}/${PAIR}/${RUN_NAME}/seed${SEED}"
mkdir -p "${OUT}"
python3 "${ROOT}/baselines/dataset_contract.py" check \
  --manifest "${DATASET_MANIFEST}" --candidate "${DATA_DIR}" --split train \
  2>&1 | tee "${OUT}/dataset_check.log"

CMD=(torchrun --standalone --nproc_per_node="${WORLD_SIZE}" "${ROOT}/finetune.py"
  --base-path "${ROOT}"
  --model-path "${STUDENT_MODEL}"
  --teacher-model-path "${TEACHER_MODEL}"
  --ckpt-name "${STUDENT_NAME}"
  --teacher-ckpt-name "${TEACHER_NAME}"
  --teacher-model-fp16
  --n-gpu "${WORLD_SIZE}"
  --data-dir "${DATA_DIR}"
  --num-workers "${NUM_WORKERS:-4}"
  --train-num -1 --dev-num -1
  --lr "${LR:-5e-6}" --lr-min 0
  --batch-size "${MICRO_BATCH:-1}"
  --eval-batch-size "${EVAL_BATCH:-1}"
  --gradient-accumulation-steps "${GRAD_ACC:-10}"
  --gradient-checkpointing
  --warmup-ratio 0.1 --lr-decay-style cosine
  --weight-decay 1e-2 --clip-grad 1.0
  --epochs "${EPOCHS:-2}" --kd-ratio "${KD_RATIO}" --temperature 1.0
  --max-length "${MAX_LENGTH:-1024}" --max-prompt-length "${MAX_PROMPT_LENGTH:-512}"
  --do-train --save-interval -1 --eval-interval -1
  --log-interval 10 --mid-log-num -1 --save "${OUT}" --seed "${SEED}"
  --deepspeed --deepspeed_config "${DEEPSPEED_CONFIG:-${ROOT}/configs/deepspeed/ds_config_zero2_bf16.json}"
  --type "${DISTILL_TYPE}" --skew-alpha 0.1
  --do-sample --top-k 0 --top-p 1.0)

if [[ "${STUDENT_GEN}" == 1 ]]; then
  CMD+=(--student-gen --gen-num-beams 1 --gen-top-p 1.0
    --init-threshold 0.0 --loss-eps 0.1 --capacity 1000
    --replay-ratio decreasing --mixed-alpha 0.5)
fi

if [[ "${METHOD}" == cst || "${METHOD}" == hidden_mse || "${METHOD}" == gram || "${METHOD}" == cka || "${METHOD}" == normalized_spectrum || "${METHOD}" == direct_spectrum ]]; then
  AUX_WEIGHT="${AUX_WEIGHT:-${CST_LOSS_WEIGHT:-0.003}}"
  CMD+=(--nnm --loss-variant "${METHOD}" --cst-loss-weight "${AUX_WEIGHT}"
    --nnm-ratio "${AUX_WEIGHT}"
    --nnm-warmup-steps 100 --nnm-ramp-steps 200
    --cst-max-tokens "${CST_MAX_TOKENS:-64}" --cst-num-layers "${CST_NUM_LAYERS:-4}"
    --cst-layer-min "${CST_LAYER_MIN:-0.20}" --cst-layer-max "${CST_LAYER_MAX:-0.85}"
    --cst-gamma-min "${CST_GAMMA_MIN:-1e-2}" --cst-gamma-max "${CST_GAMMA_MAX:-1e2}"
    --cst-num-gamma-samples "${CST_NUM_GAMMA:-2}" --cst-gamma-sampling "${CST_GAMMA_SAMPLING:-log_uniform}"
    --cst-distance l2)
  if [[ -n "${CST_FIXED_GAMMA_GRID:-}" ]]; then
    CMD+=(--cst-fixed-gamma-grid "${CST_FIXED_GAMMA_GRID}")
  fi
else
  CMD+=(--no-nnm)
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
{
  echo "pair=${PAIR} method=${METHOD} seed=${SEED}"
  echo "data_dir=${DATA_DIR} world_size=${WORLD_SIZE} effective_global_batch=$(( WORLD_SIZE * ${MICRO_BATCH:-1} * ${GRAD_ACC:-10} ))"
  echo "dataset_manifest=${DATASET_MANIFEST}"
  printf 'COMMAND: '
  printf '%q ' "${CMD[@]}"
  printf '\n\n'
  "${CMD[@]}"
} 2>&1 | tee "${OUT}/train.log"
