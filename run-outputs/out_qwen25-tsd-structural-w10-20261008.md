# Header

- Plan: [qwen25-tsd-structural-w10-20261008.md](../run-inputs/qwen25-tsd-structural-w10-20261008.md).
- Run: `qwen25_tsd_structural_w10_20261008`, `vt-admin`, `/nvme/annp36-home/work/CST`.
- Pipeline status: train, eval and report completed; chain exit **0**. The additional raw-sample recomputation requested in the plan has not been performed by this completion check.
- Start: 2026-10-08 14:41:43 Vietnam time; finish: 21:23:02; elapsed **6 h 41 m 19 s**.
- Experiment source: `a7eff1967c92c6d89df196ed55abedb32b33343c`, branch `qwen25-tsd-structural`.
- Cleanup: Direct spectrum's Minerva MATH evaluator saved results at 20:48:30, then remained waiting for its vLLM EngineCore during exit. At approximately 21:23, all 24 result files were parsed and their numeric metrics checked finite. SIGTERM was sent only to EngineCore PID 653291 after checking owner, parent PID, completed log and stale timestamp. The evaluator then exited successfully; the existing chain generated the tables without rerunning training or inference.

# Settings

- Teacher: Qwen2.5-14B-Instruct revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`; student: Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Data: `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` revision `450c33f2d591f68e12ecb546c16cf79821ef0006`, 79,751 examples. The 200-example dev set overlaps training, as documented in the plan.
- Sequential six-GPU training, seed 10, two epochs, 2,044 steps per variant. Microbatch 1, accumulation 13: global batch 78.
- AdamW, LR 5e-6, weight decay 1e-2, cosine schedule, warmup 0.1, gradient clip 1.0. Frozen teacher fp16, student bf16 with ZeRO-2.
- Output KD: SFKL, ratio 1.0, skew alpha 0.1, no student generation. Auxiliary weights: Gram 100,000; CKA 10; direct spectrum 10, ramped from step 100 to 300.
- Final checkpoint only, eight benchmarks, generation limit 5,120, temperature 0. Teacher and base-student benchmark results reuse the unchanged parent run.

# Main Results

Scores are percentages. One seed; no variance estimate. Sources: `benchmark_results/qwen25_tsd_structural_w10/tables/qwen25_tsd_full8_mean_std.csv` and each variant's `seed10/scores.json`.

| Model | GSM8K | GSM-Plus | MATH | MBPP | SciQ | STEM | Pro-Math | BBH-COT | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher (reused) | 84.46 | 67.55 | 76.32 | 52.60 | 62.40 | 75.71 | 79.64 | 44.45 | 67.89 |
| Base student (reused) | 63.00 | 45.76 | 45.26 | 46.00 | 74.40 | 53.63 | 40.19 | 45.26 | 51.69 |
| Gram | 72.71 | 52.09 | 50.82 | 45.80 | 68.40 | 50.59 | 43.01 | 28.49 | 51.49 |
| CKA | 68.54 | 49.62 | 46.18 | 42.00 | 69.20 | 50.30 | 37.60 | 25.11 | 48.57 |
| Direct spectrum | 62.77 | 45.54 | 38.62 | 14.40 | 63.10 | 51.25 | 34.34 | 22.27 | 41.54 |

| Variant | Weight-1 parent mean | Current mean | Change (percentage points) |
|---|---:|---:|---:|
| Gram | 53.29 | 51.49 | -1.80 |
| CKA | 51.97 | 48.57 | -3.40 |
| Direct spectrum | 48.37 | 41.54 | -6.83 |

# Insights

**Stronger auxiliary weights did not improve the eight-task mean.** Gram, CKA and direct spectrum declined by 1.80, 3.40 and 6.83 points against their weight-1 parent. This comparison also changes global batch from 80 to 78, so it does not isolate the effect of auxiliary weight.

**Gram remains the strongest of the three variants.** Its mean is 51.49, compared with 48.57 for CKA and 41.54 for direct spectrum. Gram remains 0.20 points below the base student on the mean, although GSM8K improves from 63.00 to 72.71.

# Artifacts

- Final checkpoints: `results/qwen25_tsd_structural_w10/{gram,cka,direct_spectrum}/seed10/2044/`. All three config JSON files parse; each checkpoint has one nonempty weight file of 3,087,867,669 bytes. Full tensor loading and weight hashing were not repeated in this completion check.
- Benchmark artifacts: 24 parseable result JSON files, eight for each variant; all numeric metrics checked finite. Three aggregate `scores.json` files; output tables in CSV, JSON and LaTeX.
- SHA256 of aggregate scores: Gram `99eeae2a6ed05e40a1163e06729368fda8c2d021c760ca34a8e30f223a9703c3`; CKA `e14d5dedab96f7637d0df1e2e4e4092f3fcf2df6c3ae25cfcdf1faf983755520`; direct spectrum `74bdaeddfc1b5ee7a841b3091abcd22fa85fc1e185411960ce117dff48d5f089`.
- No matching traceback, CUDA OOM or NCCL error in the new evaluation logs in the completion scan.
- CST released GPUs 0–5. The user's requested ORBIT-KD handoff completed at 21:23:44, exit 0; eight GPU queues and one collector were verified running. Handoff log: `/nvme/annp36-home/work/ORBIT-KD/logs/analysis/after_cst8.log`.
- Drive upload: none authorized for CST; none performed.

# Limitations

- Single seed. The tables' `± 0.00` does not estimate variance.
- Teacher and student baselines were reused, not rerun.
- The pipeline succeeded after releasing a stuck evaluation cleanup worker; this intervention is disclosed above.
- Independent recomputation from all raw samples, detailed training-error review and full checkpoint tensor integrity checks remain unverified in this completion check.

# Next Steps

- Not run: recompute each headline metric from raw sample logs and complete the remaining integrity checks.
- ORBIT-KD continues on all eight GPUs under its own existing plan and acceptance criteria.
