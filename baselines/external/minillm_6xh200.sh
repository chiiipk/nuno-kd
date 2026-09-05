#!/usr/bin/env bash
set -euo pipefail

PAIR="${1:?pair required}"
SEED="${2:?seed required}"
OUT="${3:?output directory required}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE="${ROOT}/baselines/vendor/contra-kd"
GPU_CSV="${RUN_GPUS:-0,1,2,3,4,5}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_CSV}"
WORLD_SIZE="${#GPU_LIST[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_CSV}"
TORCHRUN="${MINILLM_TORCHRUN:-${CODE}/.venv/bin/torchrun}"
if [[ ! -x "${TORCHRUN}" ]]; then
  echo "Missing MiniLLM torchrun: ${TORCHRUN}; run 'bash baselines/setup_env.sh legacy'" >&2
  exit 3
fi

case "${PAIR}" in
  qwen)
    STUDENT="${QWEN_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    TEACHER="${QWEN_TEACHER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
    MODEL_TYPE=qwen
    PROMPT_DATA="${MINILLM_QWEN_PROMPT_DATA:-}"
    DATASET_MANIFEST="${QWEN_DATASET_MANIFEST:-${ROOT}/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct/dataset_contract.json}"
    ;;
  gemma)
    STUDENT="${GEMMA_STUDENT_MODEL:-google/gemma-2-2b-it}"
    TEACHER="${GEMMA_TEACHER_MODEL:-google/gemma-2-9b-it}"
    MODEL_TYPE=llama
    PROMPT_DATA="${MINILLM_GEMMA_PROMPT_DATA:-}"
    DATASET_MANIFEST="${GEMMA_DATASET_MANIFEST:-${ROOT}/processed_data/ultraInteract/google/gemma-2-9b-it/dataset_contract.json}"
    ;;
  *) echo "Unknown pair: ${PAIR}" >&2; exit 2 ;;
esac
: "${PROMPT_DATA:?Set the pair-specific MINILLM_*_PROMPT_DATA to the full CST prompt dataset in MiniLLM indexed format}"
: "${MINILLM_LM_DATA:?Set MINILLM_LM_DATA to the processed MiniLLM LM corpus}"
mkdir -p "${OUT}"
python3 "${ROOT}/baselines/dataset_contract.py" check \
  --manifest "${DATASET_MANIFEST}" --candidate "${PROMPT_DATA}" --split train \
  2>&1 | tee "${OUT}/dataset_check.log"

CMD=("${TORCHRUN}" --standalone --nproc_per_node="${WORLD_SIZE}" "${CODE}/train_minillm.py"
  --base-path "${CODE}" --model-path "${STUDENT}" --teacher-model-path "${TEACHER}"
  --ckpt-name "${PAIR}-student" --teacher-ckpt-name "${PAIR}-teacher"
  --n-gpu "${WORLD_SIZE}" --model-type "${MODEL_TYPE}" --teacher-model-fp16
  --gradient-checkpointing --prompt-data-dir "${PROMPT_DATA}"
  --json-data
  --lm-data-dir "${MINILLM_LM_DATA}" --dev-num -1 --num-workers 0
  --epochs 2 --total-iters "${MINILLM_TOTAL_ITERS:-2600}" --kd-ratio 0.5
  --batch-size 1 --lr 5e-6 --lr-min 5e-6 --gradient-accumulation-steps 10
  --max-length 1024 --max-prompt-length 512 --warmup-iters 100
  --scheduler-name cosine_trm --save "${OUT}" --seed "${SEED}"
  --seed-ppo "${SEED}" --seed-lm "${SEED}" --save-interval -1
  --eval-interval -1 --log-interval 10 --mid-log-num -1 --do-train
  --type minillm --ppo-epochs 4 --num-rollouts 256 --chunk-size 8
  --length-norm --single-step-reg --teacher-mixed-alpha 0.2
  --reward-scaling 0.5 --cliprange-reward 100
  --do-sample --top-k 0 --top-p 1.0 --temperature 1.0
  --deepspeed --deepspeed_config "${CODE}/configs/deepspeed/ds_config_zero2.json")

export PYTHONPATH="${CODE}"
export WANDB_DISABLED=true
{
  echo "pair=${PAIR} method=minillm seed=${SEED} world_size=${WORLD_SIZE}"
  echo "prompt_data=${PROMPT_DATA} lm_data=${MINILLM_LM_DATA}"
  echo "dataset_manifest=${DATASET_MANIFEST}"
  printf 'COMMAND: '
  printf '%q ' "${CMD[@]}"
  printf '\n\n'
  "${CMD[@]}"
} 2>&1 | tee "${OUT}/train.log"
