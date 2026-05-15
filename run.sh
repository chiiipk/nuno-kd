bash install.sh

source .venv/bin/activate

bash ./scripts/download_data.sh

if [ ! -d "./processed_data/ultraInteract" ]; then
    bash ./scripts/process_data_ultraInteract.sh
fi

bash ./scripts/distillm-nnm/train_llama3.2_3B_it.sh
bash ./scripts/distillm-nnm/train_qwen2.5_0.5B.sh
bash ./scripts/distillm-nnm/train_qwen2.5_1.5B_it.sh
