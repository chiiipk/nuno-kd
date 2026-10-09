# Run output: cst_hendrycks_math_20261010

All times are giờ Việt Nam (UTC+7). The server logs in UTC, and those times were converted.

## Header

- **Plan:** [run-inputs/cst-hendrycks-math-20261010.md](../run-inputs/cst-hendrycks-math-20261010.md).
- **Run id:** `cst_hendrycks_math_20261010`.
- **Server and root:** `vt-admin` (HGX47, 8× H200), `/nvme/annp36-home/work/CST`.
- **Source:** branch `qwen25-tsd-structural` at `9a24652548a80654b580ba6be3158b819676981e`. The server checkout was clean and at this commit before launch.
- **Status:** completed. Chain exit code `0`, written at 2026-10-10 01:29:53.
- **Timeline:** launched 2026-10-10 01:16:01; the last evaluation finished at 2026-10-10 01:24:12; rescoring finished 2026-10-10 01:29:53. Elapsed 13 min 52 s.
- **Gates:**
  - Exit code `0`.
  - 12 `scores.json` files, each holding `hendrycks_math` with 5,000 samples (1,187 / 474 / 479 / 903 / 540 / 871 / 546 per subject).
  - The rescorer reproduced lm-eval's own aggregate and per-subject `exact_match` for every checkpoint exactly (maximum difference 0.0); the script stops on any difference above 1e-9.
  - The scan for Traceback, CUDA OOM, NCCL, No space left and Killed found 0 matches in the 12 `driver.log` and 12 `eval.log` files and the chain log.
  - A smoke test (35 problems, GPU 7) passed before launch.
- **Retries:** none.

## Settings

| Item | Value |
|---|---|
| Benchmark | lm-eval task `hendrycks_math`, dataset `EleutherAI/hendrycks_math`, full test split of 7 subjects, 5,000 problems, cached offline |
| Evaluator | vendored lm-eval `6d642546f4688648fced259eb3302efd36ece5af`, vLLM 0.31.0, `baselines/eval_lm_harness.py --tasks hendrycks_math` |
| Prompt | task template `Problem: {problem}\nAnswer:`, zero-shot, Qwen chat template (`--apply_chat_template`) |
| Decoding | greedy (temperature 0), `max_new_tokens` 5120, stop at `Problem:` and EOS |
| Engine | one vLLM engine per GPU, `tensor_parallel_size=1`, bf16, `gpu_memory_utilization` 0.85 |
| GPUs | 0–7. The 12 evaluations shared the lease pool: 8 ran at once, the other 4 took the first GPUs to free up |
| Checkpoints | gram, cka and direct_spectrum, seed 10; epoch-1 and final of the weight-1.0 run (996, 1992) and of the w10 run (1022, 2044) |

No training was run; the checkpoints are the parents' (see the plan). Teacher, base student and hidden_mse were not evaluated, as requested.

### Graders

Each checkpoint generated one response per problem. The responses were graded three ways by `baselines/score_hendrycks_math.py`:

1. **lm-eval `exact_match`**, the official metric of the task. It compares the text between the first and the last `$` of the whole response with the gold answer. That suits short base-model answers, but in a chat model's worked solution this span covers most of the solution, so it almost never equals the answer.
2. **`\boxed{}` exact match**: the last `\boxed{...}` of the response, compared with the gold answer by the task's own `is_equiv` (Hendrycks string normalization). A response without `\boxed{}` scores 0. Reference grader.
3. **`math_verify`**: `math_verify.verify(parse(solution), parse(response))`, the grader of lm-eval's minerva_math task and of the MATH column in the parents' tables. Reference grader.

## Main Results

Single seed (10), 5,000 problems per checkpoint, scores in %.

| Checkpoint | Epoch | lm-eval `exact_match` | `\boxed{}` exact match | `math_verify` | Responses without `\boxed{}` |
|---|---|---:|---:|---:|---:|
| w1.0 gram 996 | epoch 1 | 0.02 | 31.90 | 54.08 | 1,778 (35.6%) |
| w1.0 gram 1992 | final | 0.02 | 31.60 | 54.42 | 1,773 (35.5%) |
| w1.0 cka 996 | epoch 1 | 0.02 | 27.94 | 51.46 | 1,955 (39.1%) |
| w1.0 cka 1992 | final | 0.02 | 27.16 | 52.06 | 2,041 (40.8%) |
| w1.0 direct_spectrum 996 | epoch 1 | 0.00 | 31.04 | 50.02 | 1,735 (34.7%) |
| w1.0 direct_spectrum 1992 | final | 0.00 | 29.68 | 49.46 | 1,810 (36.2%) |
| w10 gram 1022 | epoch 1 | 0.02 | 27.62 | 49.48 | 1,908 (38.2%) |
| w10 gram 2044 | final | 0.02 | 29.08 | 50.58 | 1,887 (37.7%) |
| w10 cka 1022 | epoch 1 | 0.02 | 25.38 | 45.80 | 1,932 (38.6%) |
| w10 cka 2044 | final | 0.02 | 25.66 | 45.96 | 1,871 (37.4%) |
| w10 direct_spectrum 1022 | epoch 1 | 0.00 | 28.24 | 40.66 | 1,766 (35.3%) |
| w10 direct_spectrum 2044 | final | 0.02 | 25.22 | 39.52 | 2,009 (40.2%) |

The lm-eval `exact_match` of 0.02 is 1 correct problem out of 5,000.

### `math_verify` by subject

| Checkpoint | Algebra (n=1187) | Count. & Prob. (n=474) | Geometry (n=479) | Interm. Algebra (n=903) | Num. Theory (n=540) | Prealgebra (n=871) | Precalc (n=546) | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w1.0 gram 996 | 75.91 | 45.36 | 45.09 | 33.33 | 45.74 | 68.20 | 42.12 | 54.08 |
| w1.0 gram 1992 | 76.33 | 45.99 | 44.26 | 34.88 | 45.19 | 69.00 | 41.21 | 54.42 |
| w1.0 cka 996 | 73.21 | 43.67 | 41.75 | 33.78 | 39.07 | 66.36 | 37.18 | 51.46 |
| w1.0 cka 1992 | 73.04 | 46.84 | 43.01 | 31.78 | 42.59 | 65.90 | 39.74 | 52.06 |
| w1.0 direct_spectrum 996 | 73.38 | 40.08 | 39.46 | 29.35 | 40.00 | 65.21 | 37.00 | 50.02 |
| w1.0 direct_spectrum 1992 | 71.27 | 40.30 | 39.04 | 28.57 | 39.26 | 65.79 | 37.73 | 49.46 |
| w10 gram 1022 | 70.51 | 41.77 | 42.59 | 31.23 | 37.59 | 64.41 | 34.62 | 49.48 |
| w10 gram 2044 | 72.54 | 40.51 | 44.26 | 28.90 | 42.04 | 66.25 | 36.45 | 50.58 |
| w10 cka 1022 | 66.30 | 36.08 | 39.04 | 25.80 | 36.30 | 61.77 | 32.60 | 45.80 |
| w10 cka 2044 | 65.63 | 37.76 | 37.79 | 26.47 | 37.78 | 61.65 | 32.78 | 45.96 |
| w10 direct_spectrum 1022 | 62.34 | 31.22 | 32.36 | 19.49 | 29.07 | 58.44 | 27.11 | 40.66 |
| w10 direct_spectrum 2044 | 57.88 | 31.86 | 32.36 | 20.49 | 30.74 | 57.63 | 23.81 | 39.52 |

### `\boxed{}` exact match by subject

| Checkpoint | Algebra | Count. & Prob. | Geometry | Interm. Algebra | Num. Theory | Prealgebra | Precalc | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w1.0 gram 996 | 45.24 | 24.89 | 30.90 | 24.25 | 33.33 | 28.82 | 26.01 | 31.90 |
| w1.0 gram 1992 | 46.59 | 24.47 | 26.93 | 26.02 | 31.30 | 27.10 | 26.01 | 31.60 |
| w1.0 cka 996 | 39.51 | 24.05 | 24.22 | 24.58 | 26.48 | 24.57 | 21.79 | 27.94 |
| w1.0 cka 1992 | 37.66 | 25.11 | 23.38 | 23.37 | 27.59 | 22.96 | 21.98 | 27.16 |
| w1.0 direct_spectrum 996 | 46.59 | 24.05 | 26.93 | 22.37 | 29.44 | 29.85 | 24.73 | 31.04 |
| w1.0 direct_spectrum 1992 | 43.64 | 22.78 | 24.84 | 20.93 | 27.41 | 29.97 | 25.82 | 29.68 |
| w10 gram 1022 | 40.44 | 21.52 | 25.68 | 22.48 | 26.30 | 26.52 | 18.32 | 27.62 |
| w10 gram 2044 | 42.21 | 21.52 | 29.02 | 21.37 | 30.37 | 26.52 | 22.71 | 29.08 |
| w10 cka 1022 | 38.84 | 16.67 | 23.17 | 18.16 | 25.74 | 23.31 | 20.51 | 25.38 |
| w10 cka 2044 | 39.01 | 18.14 | 23.59 | 18.05 | 27.22 | 23.19 | 19.96 | 25.66 |
| w10 direct_spectrum 1022 | 45.41 | 18.78 | 24.63 | 14.40 | 23.52 | 35.13 | 18.86 | 28.24 |
| w10 direct_spectrum 2044 | 37.74 | 19.20 | 24.43 | 14.62 | 22.59 | 29.97 | 16.48 | 25.22 |

### Final checkpoints against the parents' minerva_math

The parents' MATH column is the 4-shot minerva_math prompt graded by `math_verify`, on the same 5,000 problems. Teacher and base student are given from the parents for reference; they were not evaluated on Hendrycks MATH.

| Model | minerva_math `math_verify` (parents, 4-shot) | Hendrycks MATH `math_verify` (this run, 0-shot) | Difference |
|---|---:|---:|---:|
| w1.0 gram 1992 | 53.30 | 54.42 | +1.12 |
| w1.0 cka 1992 | 51.12 | 52.06 | +0.94 |
| w1.0 direct_spectrum 1992 | 48.94 | 49.46 | +0.52 |
| w10 gram 2044 | 50.82 | 50.58 | -0.24 |
| w10 cka 2044 | 46.18 | 45.96 | -0.22 |
| w10 direct_spectrum 2044 | 38.62 | 39.52 | +0.90 |
| Base student (reference) | 45.26 | not run | |
| Teacher (reference) | 76.32 | not run | |

## Insights

**The official lm-eval metric is unusable for these chat checkpoints.** All 12 score 0.00 or 0.02 (at most 1 of 5,000 problems), while `math_verify` on the same responses ranges from 39.52 to 54.42. The grader compares the span between the first and the last `$` of the response with the answer, so any multi-step solution fails it. It measures the answer format of a base model, not mathematical correctness.

**`math_verify` reproduces the parents' MATH ranking within about 1 point.** For the final checkpoints the 0-shot Hendrycks prompt gives w1.0 gram 1992 54.42, w1.0 cka 1992 52.06, w1.0 direct_spectrum 1992 49.46, w10 gram 2044 50.58, w10 cka 2044 45.96, w10 direct_spectrum 2044 39.52, against 53.30, 51.12, 48.94, 50.82, 46.18, 38.62 on 4-shot minerva_math. The order of the six models is the same in both. The best checkpoint is w1.0 gram 1992 (54.42).

**The stronger auxiliary weights (w10) are worse on every method.** gram: 54.42 at w1.0 against 50.58 at w10 (-3.84); cka: 52.06 at w1.0 against 45.96 at w10 (-6.10); direct_spectrum: 49.46 at w1.0 against 39.52 at w10 (-9.94) (`math_verify`, final checkpoints). As in the parents, the w10 run also changed the global batch (80 → 78), so the drop is not attributable to the weight alone.

**The epoch-1 and final checkpoints differ by at most 1.14 points.** w1.0 gram 996 54.08 → 1992 54.42; w1.0 cka 996 51.46 → 1992 52.06; w1.0 direct_spectrum 996 50.02 → 1992 49.46; w10 gram 1022 49.48 → 2044 50.58; w10 cka 1022 45.80 → 2044 45.96; w10 direct_spectrum 1022 40.66 → 2044 39.52. The second epoch did not change MATH accuracy beyond single-seed noise.

**About a third to two fifths of the responses have no `\boxed{}` answer, which caps the boxed grader.** The rate is 34.7–40.8%. The zero-shot prompt never asks for a boxed answer; such responses end, for example, with "Therefore, the value of $x$ is $4$." `math_verify` still reads the final expression, so it scores 12.4–24.9 points higher than the boxed grader on the same responses. In the smoke test every problem where the two graders disagreed was an unboxed response, not a grader error.

## Artifacts

All paths are relative to the run root on the server, and the same paths exist in the local project after download.

- **Per-checkpoint results:** `benchmark_results/hendrycks_math_20261010/{w1,w10}/<method>/seed10/<step>/`: `scores.json`, `hendrycks_math_rescored.json`, `driver.log`, and `hendrycks_math/` with lm-eval's `results_*.json`, 7 `samples_*.jsonl` and `eval.log`.
- **Per-problem outputs:** `benchmark_results/hendrycks_math_20261010/data/hendrycks_math_samples_<run>_<method>_seed10_<step>.csv.gz`, 5,000 rows each: subject, level, problem, gold answer, the span lm-eval graded, the last `\boxed{}`, the three 0/1 grades, and the full, untruncated response.
- **Tables:**

| File | SHA-256 |
|---|---|
| `benchmark_results/hendrycks_math_20261010/tables/hendrycks_math_rescored.csv` | `33f5861e14713479b8acbb2cc840ccb06d66ee7477c26025671c7240789025d2` |
| `benchmark_results/hendrycks_math_20261010/tables/hendrycks_math_rescored.json` | `059a2d3f5f2729d42287b10d33a6bb28fe1f86c6c9d2b7d2e658e7706195b975` |

| Checkpoint | GPU | Finished | Per-problem CSV SHA-256 |
|---|---:|---|---|
| w1.0 gram 996 | 5 | 01:19:56 | `1edfe3c7ed3ae8061f858a27001760c4aa4fa0273414171b7da18e59b182cf03` |
| w1.0 gram 1992 | 3 | 01:23:20 | `08c405ff7e2cf9900a3bbd530cd574354706a34b508204786c7505707a29ae0a` |
| w1.0 cka 996 | 5 | 01:23:55 | `59213023a3d6b67e5dccd364ffbf1b0a2a7ce04c20f00403225d94a457571e50` |
| w1.0 cka 1992 | 3 | 01:20:02 | `b2d4e46d5dd50077c5de7cf308c7ecd09fa74b4cb06873708e4a6ec7b4b365d3` |
| w1.0 direct_spectrum 996 | 0 | 01:20:05 | `3e5af9dd48097117ade9480f3ea03c1b63de26419ae015e3197d373a7730c146` |
| w1.0 direct_spectrum 1992 | 1 | 01:20:07 | `fc618edcb62f197fab4890ff542600bbc1f6c3538ace334d4551508dc416a566` |
| w10 gram 1022 | 0 | 01:24:12 | `4d667a0ba2080dfcbbf665c097bf29fbe7b83511ce9e2ad1b80c1fc7ec1d562b` |
| w10 gram 2044 | 6 | 01:23:24 | `df043ba7e0b9c508a20f60e767260ee0c3736e113b56790ba99893981450b6d8` |
| w10 cka 1022 | 2 | 01:20:13 | `b25be8a933cf500f00953e0ec331e1e6d646b7608e96721395492f889c1b3bc6` |
| w10 cka 2044 | 6 | 01:19:37 | `81adff1f2e7e9a831544fd16a331553689ed6714ec1e6926d2407162e789963b` |
| w10 direct_spectrum 1022 | 7 | 01:20:58 | `39f0d20842332c434e484335e30cacaed4350767a386fb62ac9d7a877e1c8508` |
| w10 direct_spectrum 2044 | 4 | 01:20:34 | `f9277f891774aba0b545dffe6de16bd2d8e30ba2b11318ef4e650eeaaf5f8efd` |

- **Logs:** `logs/cst_hendrycks_math_20261010.{chain.log,pid,exit,started_at,finished_at}` (server only).
- **Smoke test:** `benchmark_results/hendrycks_math_20261010_smoke/` (server only).
- **Size:** 485 MB on the server, 484 MB locally.
- **Download:** `rsync` to `/Users/savoxism/Documents/GitHub/nuno-kd/benchmark_results/hendrycks_math_20261010/` (git-ignored). The SHA-256 of all 134 result, sample, score and table files matched between server and local copy.
- **GPU release:** after the run all eight GPUs show 0 MiB and no compute process.
- **Upload:** none; no destination is authorized.

## Limitations

- **Single seed (10)** for every checkpoint; differences of about 1 point, such as between epoch-1 and final checkpoints, may be noise.
- **The official metric is not informative here** (see Insights). The two other graders are reference graders run on the same responses, not lm-eval's task metric.
- **No teacher, base student or hidden_mse** on Hendrycks MATH, as requested. The comparison with the base student relies on the parents' 4-shot minerva_math, a different prompt.
- **Zero-shot prompt without an answer-format instruction.** The boxed grader is therefore limited by how often a model happens to box its answer.
- **The w10 comparison also changes the global batch** (80 → 78) and the step count (1,992 → 2,044).

## Next Steps

None of these have been run.

1. Evaluate the base student and the teacher with the same task and graders, to give a baseline on this prompt.
2. If an exact-match score on Hendrycks MATH is needed, add a prompt instruction to put the final answer in `\boxed{}` and grade the boxed answer.
3. Add seeds for the methods worth keeping.
