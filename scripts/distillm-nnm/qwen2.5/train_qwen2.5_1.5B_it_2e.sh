#! /bin/bash

GPUS=(0 1 2 3 4 5 6 7)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# ───── model ─────
BASE_PATH=.
CKPT_NAME="qwen2.5-1.5B-Instruct"
CKPT="Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_CKPT_NAME="qwen2.5-14B-Instruct"
TEACHER_CKPT="Qwen/Qwen2.5-14B-Instruct"

# ───── data ─────
DATA_DIR="./processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct/"

BATCH_SIZE=8
LR=1e-4
GRAD_ACC=1
EVAL_BATCH_SIZE=32
MAX_LENGTH=1025
SEED=10
EPOCHS=2
KD_R=1.0

# ───── SFKL ─────
SKEW_ALPHA=0.1

NNM_RATIO=0.1
NNM_K=128
NNM_N_LAYERS=4
NNM_D_PRIME=256
NNM_CENTROID_BATCHES=500
LOSS_VARIANT=${LOSS_VARIANT:-nnm}

SAVE_PATH="./results/${CKPT_NAME}#sfkl_nnm_lora/${LOSS_VARIANT}${NNM_RATIO}_K${NNM_K}_L${NNM_N_LAYERS}_epoch${EPOCHS}_lr${LR}_kdr${KD_R}"


OPTS=""
# model
OPTS+=" --base-path ."
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio ${KD_R}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 512"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ./configs/deepspeed/ds_config_zero0_bf16.json"
# ───── type: adaptive + SFKL ─────
OPTS+=" --type adaptive-sfkl"
OPTS+=" --skew-alpha ${SKEW_ALPHA}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# distillm: student-gen + adaptive threshold + replay buffer
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
OPTS+=" --replay-ratio decreasing"
OPTS+=" --mixed-alpha 0.5"
# ───── NNM (default ON; use --no-nnm to disable) ─────
OPTS+=" --nnm"
OPTS+=" --loss-variant ${LOSS_VARIANT}"
OPTS+=" --nnm-ratio ${NNM_RATIO}"
OPTS+=" --nnm-K ${NNM_K}"
OPTS+=" --nnm-n-layers ${NNM_N_LAYERS}"
OPTS+=" --nnm-d-prime ${NNM_D_PRIME}"
OPTS+=" --nnm-centroid-batches ${NNM_CENTROID_BATCHES}"
OPTS+=" --nnm-eta 0.05"
OPTS+=" --nnm-T-dead 50"
OPTS+=" --nnm-ns-iters 5"
OPTS+=" --nnm-warmup-steps 100"
OPTS+=" --nnm-ramp-steps 200"
OPTS+=" --delta-threshold 0.03"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=.
CMD="torchrun ${DISTRIBUTED_ARGS} ./finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
