# NuNo-KD

Compact implementation of NuNo-KD: on-policy knowledge distillation with a  representation loss (Nuclear Norm Matching) to preserve intermediate hidden-state geometry.

What this repo provides
- Training scripts that combine standard output-layer KD with NNM for mid-layer spectral alignment
- Utilities to build teacher centroids, apply NNM at selected layers, and run evaluation

## Repository structure (short)

```
NuNo/
├─ configs/                # Deepspeed configs and hostfiles for distributed runs
├─ data/                   # Example processed datasets (not full corpora)
├─ data_utils/             # Dataset builders and indexed dataset utilities
├─ distillm/               # Distillation losses, buffers, samplers
├─ scripts/                # Shell wrappers: preprocessing, training, eval, data generation
│  ├─ distillm-nnm/        # Example train scripts per model (gemma2, qwen2.5)
│  ├─ gen-data/             # Data generation scripts
│  └─ eval/                # Evaluation script(s)
├─ tools/                  # Helper scripts (data processing, plotting, merging)
├─ nnm_module.py           # NNM module implementation (representation loss)
├─ nnm_variants.py         # Variants and helpers for NNM
├─ nnm_recovery_weight.py  # Weighting / recovery utilities for NNM
├─ finetune.py             # Fine-tuning utilities (distillation training)
├─ generate.py             # Generation utilities / wrappers
├─ run.sh                  # Example launcher
└─ README.md               # This file
```

Key notes on important files/folders

- `configs/deepspeed/`: ready-to-use Deepspeed JSONs for different ZeRO stages and fp/bf16/offload options.

- `scripts/process_data_ultraInteract.sh` and `tools/process_data_ultraInteract.py`: preprocess raw instruction-response data into training-ready JSONL.

- `scripts/gen-data/`: scripts for generating or formatting dataset examples prior to preprocessing or training.

- `scripts/distillm-nnm/*`: model-specific training wrappers — edit flags inside these scripts or call underlying Python entrypoints.

- `finetune.py`: primary training entrypoint (distillation). Wrapper scripts call this file — inspect it for parsed flags and defaults. Common flags used by the repo include `--nnm-weight`, `--nnm-layers`, and `--kd-weight`.

- `nnm_module.py` / `nnm_variants.py`: implementation details for the Nuclear Norm Matching objective and helper variants.

- `distillm/losses.py`: integrates NNM with standard KD losses; see this file to understand how NNM is weighted and combined with output distillation.

- `scripts/eval/eval.sh`: example evaluation invocation; edit the checkpoint path there or pass a path as argument.

 

## Quick start

### 1. Install dependencies:

```bash
bash install.sh
```

### 2. Process data (required before training):

```bash
bash scripts/process_data_ultraInteract.sh
```

### 3. Train (example):

Primary training entrypoint: `finetune.py` (the shell wrappers in `scripts/` call this Python entrypoint with model-specific flags).

Example (wrapper script):

```bash
bash ./scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e.sh
```

Key flags

- `--nnm-weight`    weight for the Nuclear Norm Matching loss
- `--nnm-layers`    comma-separated list of intermediate layers to apply NNM
- `--kd-weight`     weight for the output distillation loss


### 4. Evaluation

Run the evaluation script after updating the checkpoint path in `eval.sh`:

```bash
bash ./scripts/eval/eval.sh
```

### Notes

- `scripts/eval/eval.sh` contains a sample invocation and a placeholder checkpoint; edit the checkpoint path inside that script if you prefer to change the default behavior.
- `scripts/gen-data/` contains sample data generation and formatting utilities; use these to prepare raw inputs before running preprocessing.
 
This README is intentionally short — see script headers and source files for full options and experiment configurations.
 
