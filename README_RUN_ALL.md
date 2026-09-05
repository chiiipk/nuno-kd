# Run every baseline and CST ablation on 6 x H200

`run_all_6xh200.sh` is the single entrypoint for the complete experiment suite.
It runs the two-seed baseline tables first, then every CST ablation, evaluates
all checkpoints with the same pinned lm-eval-harness, saves raw samples, and
generates CSV, LaTeX, and JSON reports.

NuNo, TSD-KD, and RPT are excluded. No ablation uses teacher centroids. Hidden
MSE alone has a learned width-matching projector; Gram, CKA, normalized/direct
spectrum, and CST are projector-free.

## 1. Required data

Canonical raw files:

```text
data/dpo/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl
data/dpo/google/gemma-2-9b-it/generated_train.jsonl
```

The ordered dataset contracts must exist at:

```text
processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct/dataset_contract.json
processed_data/ultraInteract/google/gemma-2-9b-it/dataset_contract.json
```

Regenerate processed data with `scripts/process_data_ultraInteract.sh` after
the ordered-preprocessing change. Training stops if count, content hash, or
sample-order hash differs from the contract.

Configure pair-specific external inputs:

```bash
export SKD_QWEN_TRAIN_JSONL=/data/qwen/generated_train.jsonl
export SKD_QWEN_VALIDATION_JSONL=/data/qwen/generated_validation.jsonl
export SKD_GEMMA_TRAIN_JSONL=/data/gemma/generated_train.jsonl
export SKD_GEMMA_VALIDATION_JSONL=/data/gemma/generated_validation.jsonl

export MINILLM_QWEN_PROMPT_DATA=/data/minillm/qwen/prompt_data
export MINILLM_GEMMA_PROMPT_DATA=/data/minillm/gemma/prompt_data
export MINILLM_LM_DATA=/data/minillm/lm_corpus
```

See `baselines/README.md` for contract creation and MiniLLM JSON export.

## 2. Install isolated environments

On the Linux H200 node:

```bash
export LM_EVAL_REF=<exact-lm-eval-commit-or-tag>
bash run_all_6xh200.sh setup
```

The common CST trainer environment must already be active. Setup creates
separate MiniLLM, Speculative-KD, and evaluator environments. The resolved
lm-eval commit is recorded and included in every evaluation summary.

## 3. Preflight

```bash
bash run_all_6xh200.sh preflight
```

This checks six unique visible GPUs, all data paths, exact ordered hashes,
external inputs, Python dependencies, the evaluator commit, and all eight task
names. Do not start training if preflight fails.

## 4. Run everything

Recommended resumable sequence:

```bash
nohup bash run_all_6xh200.sh train  > run_all_train.log 2>&1 &
nohup bash run_all_6xh200.sh eval   > run_all_eval.log 2>&1 &
bash run_all_6xh200.sh report | tee run_all_report.log
```

Or, in a persistent tmux session:

```bash
bash run_all_6xh200.sh all |& tee run_all.log
```

Defaults are GPUs `0,1,2,3,4,5`, seeds `10,42`, both baseline model pairs, and
Qwen for ablations. Overrides:

```bash
RUN_GPUS=0,1,2,3,4,5 SEEDS=10,42 PAIRS=qwen,gemma \
  bash run_all_6xh200.sh train
```

The main Qwen CST checkpoints and evaluation samples are reused by the
ablation tables through symlinks. They are not retrained or reevaluated.

## What gets run

Baseline table, for Qwen and Gemma with two seeds:

- Sequence-Level KD, Supervised-KD, DistiLLM, Speculative KD, MiniLLM, GKD,
  CSD, AMiD, and CST;
- GSM8K, GSM-Plus, MATH, MBPP, SciQ, STEM, Pro-Math, and BBH-COT.

Qwen ablations, two seeds:

- objectives: Hidden MSE, Gram, CKA, normalized spectrum, direct spectrum, CST;
- gamma samples: `q={1,2,4,8}`;
- ranges/randomness: four ranges plus fixed-grid control;
- supervised layers: `{1,2,4}`;
- response tokens: `{32,64,128}`.

There are 16 additional unique ablation configurations after reusing the main
CST configuration, or 32 additional training runs for two seeds.

## Outputs and resuming

```text
results/reproduce_tables/                 baseline checkpoints/logs
results/ablations/                        ablation checkpoints/logs
benchmark_results/reproduce_tables/       baseline results and samples
benchmark_results/ablations/              ablation results and samples
benchmark_results/ablations/tables/       five ablation table groups
```

Every task retains `eval.log`, official `results*.json`, and `samples*.jsonl`.
Missing or empty sample logs fail evaluation. Training is skipped only when a
`TRAINING_COMPLETE` marker and usable checkpoint exist; evaluation is skipped
only when `scores.json` exists. Scripts do not delete checkpoints.

Reported `mean +- std` is across two independently trained seeds (`ddof=1`),
not lm-eval's per-example standard error.
