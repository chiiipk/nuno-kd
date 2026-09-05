# Reproduce the two KD tables on 6 x H200

This workflow trains every requested baseline with two independent seeds and
evaluates both Qwen2.5 and Qwen3 tables on the same eight lm-eval-harness tasks.
NuNo, TSD-KD, and RPT are deliberately excluded.

## Scope

Default matrix: Qwen2.5-14B-Instruct -> Qwen2.5-1.5B-Instruct and
Qwen3-8B -> Qwen3-1.7B; seeds `10,42`; full processed data
(`--train-num -1`); two epochs. Methods are Sequence-Level KD, Supervised-KD,
DistiLLM, Speculative KD, MiniLLM, GKD, CSD, AMiD, and CST.

Both pairs are evaluated on GSM8K, GSM-Plus, MATH, MBPP, SciQ, MMLU-STEM,
MMLU-Pro-Math, and BBH-COT. The launcher uses six workers, microbatch 1, and
gradient accumulation 10 (effective global batch 60). Equal integer batches
on six ranks cannot produce exactly 64; disclose this deviation.

## Method mapping

| Row | Implementation |
|---|---|
| Sequence-Level KD | teacher-generated responses, student SFT (`kd_ratio=0`) |
| Supervised-KD | parent trainer, forward KL on fixed generated data |
| DistiLLM | adaptive skew-forward KL with student generation |
| Speculative KD | official Google Research source and patched generation stack |
| MiniLLM | official `contra-kd` PPO entry point |
| GKD | parent trainer's on-policy mixed-JSD path |
| CSD / AMiD | corresponding loss ports in the parent trainer |
| CST | multi-scale LogDet objective, weight 0.003 |

GKD, CSD, and AMiD are controlled ports, not necessarily byte-identical
official launches. A shared budget/evaluator does not establish algorithmic
equivalence; verify configurations against each paper before calling the final
table an official reproduction. External snapshots live in ignored
`baselines/vendor/`; record their commits for the paper.

## Environments and data

Set up isolated environments on the Linux server because upstream pins conflict:

```bash
bash baselines/setup_env.sh legacy
bash baselines/setup_env.sh speculative_kd
bash baselines/setup_env.sh evaluator
```

The common trainer uses the normal NuNo/CST environment. Pin the official
EleutherAI harness with `LM_EVAL_REF`; setup records the resolved revision in
`baselines/lm_eval_commit.txt`. Speculative KD requires its upstream
Transformers 4.44.2 patch and must remain isolated.

Common data defaults:

```text
processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct
processed_data/ultraInteract/Qwen/Qwen3-8B
```

Override with `QWEN_DATA_DIR` and `QWEN3_DATA_DIR`. Create one ordered contract
per pair from the canonical raw teacher-generation file:

```bash
python3 baselines/dataset_contract.py create \
  --pair qwen \
  --reference data/dpo/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
  --output processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct/dataset_contract.json

python3 baselines/dataset_contract.py create \
  --pair qwen3 \
  --reference data/dpo/Qwen/Qwen3-8B/generated_train.jsonl \
  --output processed_data/ultraInteract/Qwen/Qwen3-8B/dataset_contract.json
```

Regenerate old processed data once: the preprocessor now uses ordered
`multiprocessing.Pool.imap`; its previous `imap_unordered` could permute samples.
After preprocessing, verify the binary sidecar JSONL against its contract:

```bash
python3 baselines/dataset_contract.py check \
  --manifest processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct/dataset_contract.json \
  --candidate processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct
```

Create MiniLLM's JSON prompt dataset from the same raw files (repeat for Qwen
and Qwen3, and export the corresponding validation file as `--split valid`):

```bash
python3 baselines/dataset_contract.py export-minillm \
  --reference data/dpo/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
  --output /data/minillm/qwen/prompt_data --split train
```

The MiniLLM adapter passes `--json-data`, so `train.jsonl`/`valid.jsonl` are the
actual inputs rather than unchecked binary or text files. External variables:

```bash
export SKD_QWEN_TRAIN_JSONL=/data/qwen/generated_train.jsonl
export SKD_QWEN_VALIDATION_JSONL=/data/qwen/generated_validation.jsonl
export SKD_QWEN3_TRAIN_JSONL=/data/qwen3/generated_train.jsonl
export SKD_QWEN3_VALIDATION_JSONL=/data/qwen3/generated_validation.jsonl
export MINILLM_QWEN_PROMPT_DATA=/data/minillm/qwen/prompt_data
export MINILLM_QWEN3_PROMPT_DATA=/data/minillm/qwen3/prompt_data
export MINILLM_LM_DATA=/data/minillm/lm_corpus
```

Speculative-KD JSONL rows must contain `prompt` and `generated_text`. Qwen and
Qwen3 cannot share one raw file because their teacher-generated outputs differ.
`QWEN_DATASET_MANIFEST` and `QWEN3_DATASET_MANIFEST` may override contract paths.

Every launcher now checks count, the SHA-256 hash of the complete sample
multiset, and the ordered SHA-256 hash before allocating the model. A changed,
missing, duplicated, or reordered example stops training. The MiniLLM LM corpus
is an algorithm-specific additional input and is logged separately; the prompt
training set itself must match the contract exactly.

## Preflight and run

To generate the missing Qwen3 teacher responses and rebuild both ordered
processed datasets in one command:

```bash
bash run_all_6xh200.sh prepare-data
```

The generation step requires the normal vLLM/Transformers training environment
and six visible GPUs. Existing complete raw data is reused rather than
overwritten.

Run the checks in each relevant environment before spending GPU time:

```bash
python3 baselines/preflight_reproduction.py --stage static
python3 baselines/preflight_reproduction.py --stage train
python3 baselines/preflight_reproduction.py --stage eval
```

Then use the resumable phases:

```bash
bash baselines/reproduce_2tables_6xh200.sh train
bash baselines/reproduce_2tables_6xh200.sh eval
bash baselines/reproduce_2tables_6xh200.sh report
```

`all` runs all phases. Overrides include `RUN_GPUS`, `SEEDS`, `PAIRS`,
`METHODS`, `RESULT_ROOT`, `EVAL_ROOT`, and `EVAL_PYTHON`; do not vary them between controlled
arms. `TRAINING_COMPLETE` is created only after a usable Hugging Face checkpoint
is found. Scripts never delete checkpoints.

## Logs and samples

Each training log includes the exact command and effective batch:

```text
results/reproduce_tables/<pair>/<method>/seed<seed>/train.log
results/reproduce_tables/<pair>/<method>/seed<seed>/dataset_check.log
```

`dataset_check.log` is written before model loading and records the successful
ordered-contract check for that exact run.

Each evaluation task stores:

```text
benchmark_results/reproduce_tables/<pair>/<method>/seed<seed>/<task>/eval.log
benchmark_results/reproduce_tables/<pair>/<method>/seed<seed>/<task>/results*.json
benchmark_results/reproduce_tables/<pair>/<method>/seed<seed>/<task>/samples*.jsonl
```

Evaluation always passes `--log_samples` and fails if sample files are missing
or empty. `scores.json` records metric, lm-eval stderr, sample paths/count,
checkpoint, harness commit, timestamp, GPUs, decoding length, and average.
Archive raw samples; they can be large.

MBPP runs with unsafe-code confirmation and no chat template. MATH is 4-shot,
MBPP 3-shot, and MMLU-STEM 5-shot. Other generation tasks use the chat template.
Decoding uses temperature 0 and `max_new_tokens=5120`.

## Reporting

Table `+-` is sample standard deviation (`ddof=1`) over two independently
trained seeds. lm-eval's printed `+-` is a per-example standard error for one
checkpoint; it is retained in `scores.json` but never substituted for seed
variation. Fixed teacher/student checkpoints are evaluated once and have zero
cross-seed standard deviation.

CSV, LaTeX, and JSON outputs are:

```text
benchmark_results/reproduce_tables/tables/qwen_full8_mean_std.*
benchmark_results/reproduce_tables/tables/qwen3_full8_mean_std.*
```

`Avg.` is calculated for each seed over all eight tasks, then summarized across
seeds. It is not calculated from rounded cells.

## Final checklist

- Pin model/source revisions, Python/CUDA/Torch, lm-eval, vLLM, task YAML, and prompts.
- Keep data order, examples, steps, optimizer, batch, decoding, and checkpoint selection fixed.
- Inspect failed/empty generations in every `samples*.jsonl`, especially MATH and MBPP.
- If discussing MATH parser mismatch, report exact and symbolic metrics consistently for all arms.
- Report end-to-end wall time and peak memory, not only loss-kernel time.
- Run these local checks before launch:

```bash
python3 -m unittest discover -s tests -p 'test_baseline_reporting.py' -v
bash -n baselines/*.sh baselines/external/*.sh
git diff --check
```
