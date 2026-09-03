#!/bin/bash

TP=4

LOG_DIR="outputs/eval_results/logs"
OUT_DIR="outputs/eval_results/vllm"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

run_eval() {
    local LABEL=$1
    local MODEL_ARGS=$2
    local OUT="${OUT_DIR}/${LABEL}"
    local LOG="${LOG_DIR}/${LABEL}.log"

    mkdir -p "${OUT}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        --fewshot_as_multiturn
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=5120"
    )

    BASE_ARGS_CODE=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=5120,temperature=0.0"
    )

    BASE_ARGS_MATH=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=5120,temperature=0.0"
    )

    {
        echo "=========================================="
        echo "Label: ${LABEL}"
        echo "Start: $(date)"
        echo "=========================================="

        echo ">>> [1/10] GSM8K"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm8k
            
        echo ">>> [2/10] MATH (Minerva format)"
        lm_eval "${BASE_ARGS[@]}" --tasks minerva_math --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [4/10] SciQ"
        lm_eval "${BASE_ARGS[@]}" --tasks sciq

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS_CODE[@]}" --tasks mbpp --confirm_run_unsafe_code  --num_fewshot 3

        echo ">>> [6/10] GSM-Plus (5-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm_plus

        echo ">>> [7/10] MMLU-Pro-Math (5-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_pro_math

        echo ">>> [8/10] BBH CoT (3-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks bbh_cot_fewshot


        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0


CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"
