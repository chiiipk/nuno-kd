#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_CSV="${RUN_GPUS:-0,1,2,3,4,5}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_CSV}"
EXPECTED_COUNT="${EXPECTED_TRAIN_COUNT:-79751}"
QWEN_RAW="${QWEN_RAW_JSONL:-${ROOT}/data/dpo/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
QWEN3_RAW="${QWEN3_RAW_JSONL:-${ROOT}/data/dpo/Qwen/Qwen3-8B/generated_train.jsonl}"
QWEN_OUT="${QWEN_DATA_DIR:-${ROOT}/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct}"
QWEN3_OUT="${QWEN3_DATA_DIR:-${ROOT}/processed_data/ultraInteract/Qwen/Qwen3-8B}"

if [[ ! -f "${QWEN_RAW}" ]]; then
  echo "Missing existing Qwen2.5 teacher data: ${QWEN_RAW}" >&2
  exit 3
fi

if [[ ! -f "${QWEN3_RAW}" ]]; then
  echo "Generating Qwen3-8B teacher responses on ${#GPU_LIST[@]} GPUs..."
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_CSV}" python3 tools/generate_vllm.py \
    --model_path Qwen/Qwen3-8B \
    --output_dir "$(dirname "${QWEN3_RAW}")" \
    --output_file "$(basename "${QWEN3_RAW}")" \
    --num_gpus "${#GPU_LIST[@]}"
else
  echo "[reuse] ${QWEN3_RAW}"
fi

python3 "${ROOT}/baselines/dataset_contract.py" validate \
  --candidate "${QWEN_RAW}" --expected-count "${EXPECTED_COUNT}"
python3 "${ROOT}/baselines/dataset_contract.py" validate \
  --candidate "${QWEN3_RAW}" --expected-count "${EXPECTED_COUNT}"

python3 "${ROOT}/baselines/dataset_contract.py" create --pair qwen \
  --reference "${QWEN_RAW}" --output "${QWEN_OUT}/dataset_contract.json"
python3 "${ROOT}/baselines/dataset_contract.py" create --pair qwen3 \
  --reference "${QWEN3_RAW}" --output "${QWEN3_OUT}/dataset_contract.json"

preprocess_if_needed() {
  local raw="$1" model="$2" output="$3"
  if python3 "${ROOT}/baselines/dataset_contract.py" check \
      --manifest "${output}/dataset_contract.json" --candidate "${output}"; then
    echo "[reuse processed] ${output}"
    return
  fi
  echo "[preprocess] ${model}"
  cd "${ROOT}"
  PYTHONPATH=. python3 tools/process_data_ultraInteract.py \
    --data-dir "${raw}" --processed-data-dir "${ROOT}/processed_data/ultraInteract" \
    --model-path "${model}" --data-process-workers "${DATA_WORKERS:-32}" \
    --max-prompt-length 512 --dev-num 200 --only-prompt --model-type qwen
}

preprocess_if_needed "${QWEN_RAW}" Qwen/Qwen2.5-14B-Instruct "${QWEN_OUT}"
preprocess_if_needed "${QWEN3_RAW}" Qwen/Qwen3-8B "${QWEN3_OUT}"

cd "${ROOT}"
python3 baselines/dataset_contract.py check \
  --manifest "${QWEN_OUT}/dataset_contract.json" --candidate "${QWEN_OUT}"
python3 baselines/dataset_contract.py check \
  --manifest "${QWEN3_OUT}/dataset_contract.json" --candidate "${QWEN3_OUT}"

export_minillm_if_needed() {
  local raw="$1" manifest="$2" output="$3"
  if python3 baselines/dataset_contract.py check \
      --manifest "${manifest}" --candidate "${output}" --split train; then
    echo "[reuse MiniLLM] ${output}/train.jsonl"
  else
    python3 baselines/dataset_contract.py export-minillm \
      --reference "${raw}" --output "${output}" --split train
  fi
}

export_minillm_if_needed "${QWEN_RAW}" "${QWEN_OUT}/dataset_contract.json" \
  "${MINILLM_QWEN_PROMPT_DATA:-${ROOT}/processed_data/minillm/qwen/prompt_data}"
export_minillm_if_needed "${QWEN3_RAW}" "${QWEN3_OUT}/dataset_contract.json" \
  "${MINILLM_QWEN3_PROMPT_DATA:-${ROOT}/processed_data/minillm/qwen3/prompt_data}"

echo "Data generation, ordered preprocessing, contracts, and MiniLLM train exports complete."
