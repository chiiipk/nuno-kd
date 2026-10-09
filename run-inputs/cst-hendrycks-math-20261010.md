# Project

- `status`: requested by the user on 2026-10-10. Launch follows the smoke test in Commands §2.
- `name`: Hendrycks MATH evaluation of the CST gram, cka and direct_spectrum checkpoints.
- `run_id`: `cst_hendrycks_math_20261010`.
- `user_request`: on 2026-10-10 the user asked to evaluate every gram, cka and direct_spectrum checkpoint on the Hendrycks MATH dataset, using all eight H200, and to download the results into the local project `/Users/savoxism/Documents/GitHub/nuno-kd/`. The user also asked for a GPU health check every 20 minutes and for any error to be fixed without waiting for confirmation.
- `parent_runs`:
  - [qwen25-tsd-structural-20261005.md](qwen25-tsd-structural-20261005.md) (weight 1.0, checkpoints 996 and 1992).
  - [qwen25-tsd-structural-w10-20261008.md](qwen25-tsd-structural-w10-20261008.md) (stronger weights, checkpoints 1022 and 2044).
  - Their checkpoints and benchmark artifacts are read-only for this run.
- `local_path`: `/Users/savoxism/Documents/GitHub/nuno-kd`.
- `ssh_host`: `vt-admin` (HGX47, user `vt_admin`).
- `remote_path`: `/nvme/annp36-home/work/CST`.
- `branch`: `qwen25-tsd-structural`.
- `source_revision`: the commit that adds this file.
- `terminal_report`: `run-outputs/out_cst-hendrycks-math-20261010.md`.

# Schedule

- `start_at`: as soon as the smoke test passes. All eight GPUs were idle (0 MiB, no compute process) at about 01:20 giờ Việt Nam on 2026-10-10.
- `timezone`: `Asia/Ho_Chi_Minh`.
- `max_runtime`: none; run to completion under the health monitor.
- `estimate`: the parents' 4-shot minerva_math (same 5,000 problems) took up to 17 min per model on one GPU. Twelve models on eight GPUs run in two waves, so about 30–45 min, plus a few minutes of rescoring.

# Data

- `benchmark`: lm-eval task `hendrycks_math`, dataset `EleutherAI/hendrycks_math` (7 subjects, full test split, 5,000 problems), cached offline under `HF_HOME=/nvme/annp36-home/.cache/huggingface`.
- `prompt`: the task's template `Problem: {problem}\nAnswer:`, zero-shot, with the Qwen chat template (`--apply_chat_template`), as for the other generative tasks in the parents.
- `generation`: greedy (temperature 0), `max_new_tokens` 5120, stop at `Problem:` (task default) and EOS.

# Model

Twelve checkpoints, seed 10, final and epoch-1 of each run:

| Run tag | Training run | Methods | Steps |
|---|---|---|---|
| `w1` | `results/qwen25_tsd_structural` | gram, cka, direct_spectrum | 996 (epoch 1), 1992 (final) |
| `w10` | `results/qwen25_tsd_structural_w10` | gram, cka, direct_spectrum | 1022 (epoch 1), 2044 (final) |

- `not_included`: hidden_mse, the teacher and the base student, which the user did not ask for.
- `GPU_policy`: GPUs 0–7. All twelve evaluations start together and share the lease pool of `baselines/eval_lm_harness.py`: each takes one completely idle GPU (no compute process, under 2048 MiB), so eight run at once and four wait. One vLLM engine per GPU, `tensor_parallel_size=1`, bf16, `gpu_memory_utilization` 0.85.

# Graders

The run produces one set of responses and grades it three ways (`baselines/score_hendrycks_math.py`):

1. `lm_eval_exact_match`: lm-eval's own `hendrycks_math` grader, the official task metric. It compares the text between the first and the last `$` of the whole response with the gold answer, which suits short base-model answers but rarely isolates the final answer of a chat model's worked solution. Expected to be near 0 for these chat checkpoints.
2. `boxed_exact_match`: the last `\boxed{}` of the response compared with the gold answer by the task's own `is_equiv` (Hendrycks string normalization). Reference grader.
3. `math_verify`: `math_verify.verify(parse(solution), parse(response))`, as in lm-eval's minerva_math. Reference grader.

# Commands

## 1. Source and deployment

```bash
cd /Users/savoxism/Documents/GitHub/nuno-kd
git push origin qwen25-tsd-structural
ssh vt-admin 'cd /nvme/annp36-home/work/CST && git pull --ff-only && git status --porcelain --untracked-files=no && git rev-parse HEAD'
```

## 2. Smoke test (server, GPU 7, about 3 min)

```bash
cd /nvme/annp36-home/work/CST
export HF_HOME=/nvme/annp36-home/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
S=benchmark_results/hendrycks_math_20261010_smoke
baselines/vendor/lm-evaluation-harness/.venv/bin/python baselines/eval_lm_harness.py \
  --checkpoint results/qwen25_tsd_structural/gram/seed10/1992 --output $S/w1/gram/seed10/1992 \
  --gpus 7 --tasks hendrycks_math --limit 5
baselines/vendor/lm-evaluation-harness/.venv/bin/python baselines/score_hendrycks_math.py --root $S
```

Passes if `scores.json` holds `hendrycks_math` with 35 samples, the rescorer reproduces lm-eval's aggregate, and responses end with a `\boxed{}` answer.

## 3. Durable launch (server)

```bash
cd /nvme/annp36-home/work/CST
setsid nohup bash scripts/run_cst_hendrycks_math_20261010.sh > logs/cst_hendrycks_math_20261010.chain.log 2>&1 < /dev/null &
echo $! > logs/cst_hendrycks_math_20261010.pid
```

## 4. Monitoring

- Watcher: `ssh -t vt-admin 'cd /nvme/annp36-home/work/CST && INTERVAL=10 bash scripts/watch_cst_hendrycks_math_20261010.sh'`.
- A read-only health check every 20 minutes: chain PID and exit file, per-checkpoint progress, GPU owner, utilization, VRAM, temperature and power, disk, and an error scan.

## 5. Verification, report and download

- Exit code `0`; twelve `scores.json` with `hendrycks_math` and 5,000 samples each; rescorer agreement with lm-eval's aggregates (enforced by the script).
- Write `run-outputs/out_cst-hendrycks-math-20261010.md`.
- Download `benchmark_results/hendrycks_math_20261010/` into the same path of the local project (git-ignored), with `rsync` and a SHA-256 check of the tables and per-sample CSV files.

# Outputs

- `eval_root`: `benchmark_results/hendrycks_math_20261010/{w1,w10}/<method>/seed10/<step>/` with `scores.json`, `driver.log`, `hendrycks_math/` (lm-eval results, samples, `eval.log`) and `hendrycks_math_rescored.json`.
- `per_sample_outputs`: `benchmark_results/hendrycks_math_20261010/data/hendrycks_math_samples_<run>_<method>_seed10_<step>.csv.gz`.
- `tables`: `benchmark_results/hendrycks_math_20261010/tables/hendrycks_math_rescored.{csv,json}`.
- `chain_log`, `pid_file`, `exit_file`, `started_at`, `finished_at`: `logs/cst_hendrycks_math_20261010.{chain.log,pid,exit,started_at,finished_at}`.
- `smoke`: `benchmark_results/hendrycks_math_20261010_smoke/`.

# Upload

- `destination`: none authorized. Results are downloaded to the local project only.

# Acceptance

- Every evaluation runs on its own idle GPU from 0–7; no GPU of another job is used.
- No traceback, OOM, NCCL error or full disk.
- Twelve complete results, 5,000 samples each, all three graders reported.
- Retry policy (authorized by the user): fix errors without waiting, rerunning only failed evaluations through `--reuse-complete`; never change the checkpoints, prompt, decoding or graders without an entry in the change history.

# Change history

- `2026-10-10`: plan created from the user's request.
