export TF_CPP_MIN_LOG_LEVEL=3


PYTHONPATH=. python ./tools/process_data_ultraInteract.py \
    --data-dir ./data/dpo/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
    --processed-data-dir ./processed_data/ultraInteract \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num 200 \
    --only-prompt \
    --model-type qwen


PYTHONPATH=. python ./tools/process_data_ultraInteract.py \
    --data-dir ./data/dpo/google/gemma-2-9b-it/generated_train.jsonl \
    --processed-data-dir ./processed_data/ultraInteract \
    --model-path google/gemma-2-9b-it \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num 200 \
    --only-prompt \
    --model-type qwen
