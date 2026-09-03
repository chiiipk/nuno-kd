#!/bin/bash
TP_SIZE=${1:-2}

MODEL_PATH="google/gemma-2-9b-it"
OUTPUT_DIR="data/dpo/google/gemma-2-9b-it"
OUTPUT_FILE="generated_train.jsonl"

echo "Start gen traces..."
echo "Model: $MODEL_PATH"

python ./tools/generate_vllm.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --output_file $OUTPUT_FILE \
    --num_gpus $TP_SIZE

echo "Done!"