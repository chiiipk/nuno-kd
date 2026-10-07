# Run output: qwen25_tsd_structural_20261005

All times are giờ Việt Nam (UTC+7). The server logs in UTC, and those times were converted.

## Header

- **Run id:** `qwen25_tsd_structural_20261005`.
- **Run input:** [run-inputs/qwen25-tsd-structural-20261005.md](../run-inputs/qwen25-tsd-structural-20261005.md).
- **Server and root:** `vt-admin` (HGX47, 8× H200), `/nvme/annp36-home/work/CST`. The checkout is branch `qwen25-tsd-structural` at `07f015f7b68104491f7c814474695e14210eb2df`.
- **Status:** completed. The final chain exit code is `0`, written at 2026-10-07 17:46:56.
- **Plan §9 gates:**
  - Exit code `0`.
  - Four final checkpoints (`1992/config.json`).
  - Six parseable `scores.json` files with all eight tasks each.
  - The non-empty sample logs.
  - The CSV/JSON/LaTeX tables.
  - The error scan for Traceback, CUDA OOM, NCCL, No space left, Killed and NaN/Inf, which found 0 matches. It covered the current chain logs (`chain.log`, `chain2.log`, `chain3.log`, `chain3.waiting.log`) and every `train.log`/`log.txt` under `results/qwen25_tsd_structural`.
  - The only traceback is in the archived attempt-1 log (`logs/attempt1_20261006/`), which is the documented exit-1 failure below.
- **Attempts:** the first chain exited `1`. It was continued with evaluation only, without retraining. The phase timestamps follow.

| Phase | Start | End | Duration |
|---|---|---|---|
| Chain attempt 1 launched | 2026-10-06 20:21:06 | | |
| Train `hidden_mse` | 2026-10-06 20:21:30 | 2026-10-06 21:51:30 | 1 h 30 m |
| Train `gram` | 2026-10-06 21:51:48 | 2026-10-06 23:18:46 | 1 h 27 m |
| Train `cka` | 2026-10-06 23:19:05 | 2026-10-07 00:46:53 | 1 h 28 m |
| Train `direct_spectrum` | 2026-10-07 00:47:12 | 2026-10-07 02:17:40 | 1 h 30 m |
| Attempt 1: teacher/student eval, then exit `1` at the `hidden_mse` eval | 2026-10-07 02:17:40 | 2026-10-07 02:40:00 | |
| Idle, diagnosis and fixes (no GPU use) | 2026-10-07 02:40 | 2026-10-07 10:04 | |
| Continuation: BBH reruns (teacher, student), trained-model eval on the shared GPU pool | 2026-10-07 10:04:56 | 2026-10-07 10:31 (stopped at the user's request) | 26 m |
| Stopped, all GPUs released | 2026-10-07 10:31 | 2026-10-07 17:39 | |
| Resume: reuse finished tasks, run 2 remaining gram tasks, report | 2026-10-07 17:39:44 | 2026-10-07 17:46:56 | 7 m |

- **Elapsed time:**
  - Wall clock from launch to exit `0`: 21 h 26 m.
  - Training: 5 h 56 m.
  - Most of the remaining time is the idle gap after the failure and the stop the user asked for.

## Settings

| Item | Actual value |
|---|---|
| Teacher | `Qwen/Qwen2.5-14B-Instruct` @ `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, frozen, fp16 |
| Student | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, DeepSpeed ZeRO-2 bf16 |
| Data | `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` @ `450c33f2…`, 79,751 examples (student-generated responses) |
| Output KD | `sfkl` (skew-forward KL), ratio `1.0`, skew alpha `0.1`, no student generation |
| Structural auxiliary | one per run: `hidden_mse`, `gram`, `cka` or `direct_spectrum`, each with weight `1.0` |
| Optimizer | AdamW, LR `5e-6`, weight decay `0.01`, gradient clip `1.0` |
| LR schedule | cosine, warmup ratio `0.1` (`args.json`: `warmup_ratio 0.1`, `warmup_iters 0`) |
| Sequence | max length `1024`, max prompt length `512` |
| Steps | 2 epochs = 1992 optimizer steps per run (19,920 micro-steps per rank) |
| Batch | $B_{\mathrm{global}} = 8 \times 1 \times 10 = 80$ (8 GPUs, microbatch 1, accumulation 10), one DDP job at a time |
| Seed | `10` only |
| Checkpoint | final epoch, `1992`; benchmark scores were not used for selection |
| In-run validation | 200-example dev set built by the processor; it overlaps the training collection |
| Benchmarks | lm-eval through vLLM (`baselines/eval_lm_harness.py`), `max_new_tokens` 5120, full test sets (`limit: None`) |

The values above were checked against each run's `args.json`. All four runs record the same epochs, LR, batch, accumulation, seed, `type: sfkl`, `student_gen: false`, clip, weight decay and sequence lengths.

There is no paper reference for this sweep.

### Deviations from the run input

All of these are recorded in the plan's change history.

- **`hidden_mse` evaluated through a copy.**
  - vLLM rejects the raw `hidden_mse` checkpoint because it holds four training-only `projectors.*` tensors.
  - That model was therefore evaluated through `1992-vllm`, a copy without the projectors (`EXPORT_COMPLETE` lists the dropped keys). The student weights are unchanged.
- **BBH stop sequence changed.**
  - BBH-COT stops only at `<|im_end|>` (plus tokenizer EOS), instead of the template's `\n\n`/`Q`. The template stops cut chat answers after their first paragraph.
  - Teacher and student BBH were rerun with this setting. The superseded runs are kept: teacher BBH `25.28` with 69% `[invalid]`.
  - All six models use the same BBH setting (`stop_overrides` in every `scores.json`).
- **Evaluation GPU scheduling changed.**
  - Each task leases one GPU from a pool shared by concurrent model evaluations.
  - Since 2026-10-07 17:36 a task runs only on a GPU with no compute process and less than 2048 MiB in use.
  - Tasks, decoding and metrics are unchanged. The `gpus` field in `scores.json` lists the pool (0–7), not the GPU each task used.
- **Finished task runs were reused on resume (`--reuse-complete`).** A task directory was scored without rerunning only when its `eval.log` recorded the identical command and a `results*.json` existed.
  - `hidden_mse` and `direct_spectrum`: all 8 tasks reused.
  - `gram`: 6 tasks reused. `gsm_plus` was rerun on GPU 0 (the interrupted run is kept as `gsm_plus.incomplete_20261007T103945Z`), and `mbpp` ran fresh on GPU 1.
  - `cka`: finished in the continuation at 10:27:55, before the stop.
- **Evaluator commit not recorded.** `scores.json` records `lm_eval_commit: "unknown"`. The plan pins lm-eval `6d64254`, and the vendored harness was not changed during the run.

## Main Results

These are single-seed scores (seed 10), multiplied by 100. The table script writes a `sample_std` of 0.0 with one seed; that is not a variance estimate, so no error bars are shown.

### Benchmarks

| Model | GSM8K | GSM-Plus | MATH | MBPP | SciQ | MMLU-STEM | MMLU-Pro-Math | BBH-COT | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Teacher (14B) | 84.46 | 67.55 | 76.32 | 52.60 | 62.40 | 75.71 | 79.64 | 44.45 | 67.89 |
| Student (1.5B, base) | 63.00 | 45.76 | 45.26 | 46.00 | 74.40 | 53.63 | 40.19 | 45.26 | 51.69 |
| `hidden_mse` | 4.85 | 3.61 | 3.94 | 0.80 | 57.20 | 31.21 | 3.92 | 3.98 | 13.69 |
| `gram` | **74.22** | **54.37** | **53.30** | **48.40** | 67.80 | **51.44** | **44.63** | **32.13** | **53.29** |
| `cka` | 71.72 | 52.65 | 51.12 | 48.20 | **69.90** | 51.19 | 42.71 | 28.31 | 51.97 |
| `direct_spectrum` | 70.05 | 51.73 | 48.94 | 34.00 | 64.40 | 49.79 | 40.56 | 27.48 | 48.37 |

Bold marks the best trained model per column.

Metrics, in column order:
- GSM8K and GSM-Plus: `flexible-extract`.
- MATH: `math_verify` (minerva_math).
- MBPP: pass@1.
- SciQ: `acc_norm`.
- MMLU-STEM: `acc`.
- MMLU-Pro-Math: `custom-extract`.
- BBH-COT: `exact_match,get-answer`.

### Score verification

All 48 cells (6 models × 8 tasks) and the 6 averages were cross-checked after the run. No cell differed by more than 1e-6.
- **Three sources agree.** Each `scores.json` value equals lm-eval's own aggregate in that task's `results_*.json`, and also equals the value recomputed from the raw `samples_*.jsonl`.
  - The recomputation uses only rows of the headline filter, for example 1,319 `flexible-extract` rows for GSM8K.
  - Group tasks (MATH, MMLU-STEM, BBH) are sample-weighted means over their subtasks, which is how lm-eval aggregates them.
- **The CSV table matches the score files.** Each table cell is the same value rounded to two decimals.
- **The averages are correct.** Each average is the plain mean of its model's eight task scores.
- **The `hidden_mse` export is the trained model.** The evaluated `1992-vllm` weights are identical to the 339 non-projector tensors of `1992/pytorch_model.bin`. The only tensors dropped are `projectors.0–3.weight`.
- **The check confirms the arithmetic, not the grading.** It does not re-grade answers: the correctness of lm-eval's extractors (for example the strict BBH regex) is taken as given.

### Change versus the base student

The table shows the trained model's score minus the base student's score.

| Model | GSM8K | GSM-Plus | MATH | MBPP | SciQ | MMLU-STEM | MMLU-Pro-Math | BBH-COT | Avg |
|---|---|---|---|---|---|---|---|---|---|
| `gram` | +11.22 | +8.61 | +8.04 | +2.40 | −6.60 | −2.19 | +4.44 | −13.13 | +1.60 |
| `cka` | +8.72 | +6.89 | +5.86 | +2.20 | −4.50 | −2.44 | +2.52 | −16.95 | +0.28 |
| `direct_spectrum` | +7.05 | +5.97 | +3.68 | −12.00 | −10.00 | −3.84 | +0.37 | −17.78 | −3.32 |
| `hidden_mse` | −58.15 | −42.15 | −41.32 | −45.20 | −17.20 | −22.42 | −36.27 | −41.28 | −38.00 |

### Sample counts

These counts are identical for all six models.

| Task | GSM8K | GSM-Plus | MATH | MBPP | SciQ | MMLU-STEM | MMLU-Pro-Math | BBH-COT |
|---|---|---|---|---|---|---|---|---|
| Logged samples | 2,638 | 21,104 | 5,000 | 500 | 1,000 | 3,153 | 1,351 | 6,511 |

GSM8K and GSM-Plus log one line per question for each of their two filters: 1,319 and 10,552 questions.

### Training and dev losses

| Method | Dev loss, pre-train | Dev loss, epoch 1 | Dev loss, final | Train `ds_loss`, step 1990 | Aux loss, step 1990 |
|---|---|---|---|---|---|
| `hidden_mse` | 0.3054 | 0.7152 | 0.6953 | 0.6522 | 3.0004 |
| `gram` | 0.3054 | 0.5124 | 0.5252 | 0.5213 | 0.0000 |
| `cka` | 0.3054 | 0.5140 | 0.5223 | 0.5239 | 0.0313 |
| `direct_spectrum` | 0.3054 | 0.5146 | 0.5259 | 0.5321 | 0.0543 |

The dev set overlaps training, so these losses are only a sanity check, not validation results. All losses are finite. Step time was 2.54–2.66 s at the end of every run.

### BBH answers without an extractable answer

BBH answers with no `the answer is …` match are scored `[invalid]`. These counts are a reference diagnostic, not a headline metric.

| Model | `[invalid]` / 6,511 | Rate |
|---|---|---|
| Teacher | 2,469 | 37.9% |
| Student | 443 | 6.8% |
| `hidden_mse` | 5,062 | 77.7% |
| `gram` | 1,799 | 27.6% |
| `cka` | 2,210 | 33.9% |
| `direct_spectrum` | 2,238 | 34.4% |

## Insights

**The best average comes from `gram`, but its structural loss never contributed.**
- `gram` averages 53.29 against the base student's 51.69 (+1.60) and is the best trained model on 7 of 8 tasks (all but SciQ, where `cka` leads).
- However, `gram_loss` logs `0.0000` from step 10 to step 1990 (`results/qwen25_tsd_structural/gram/seed10/log.txt`). Its total loss equals `ds_loss` (0.5213 at step 1990).
- This run is therefore effectively output-KD-only `sfkl`. Its gains cannot be attributed to the Gram objective.
- Because the sweep has no `sfkl`-only baseline, `gram` is the closest available stand-in for one. On that reading, `cka` (+0.28) and `direct_spectrum` (−3.32) did no better than plain output KD.

**Output KD helps math reasoning in every non-collapsed run.**
- GSM8K rises by 7.05 to 11.22 points, GSM-Plus by 5.97 to 8.61, and MATH by 3.68 to 8.04 for `gram`, `cka` and `direct_spectrum`.
- MMLU-Pro-Math rises by up to 4.44.
- Even so, the best run still trails the teacher by 10.24 on GSM8K and 23.02 on MATH.

**Every trained model regresses on SciQ, MMLU-STEM and BBH.**
- SciQ falls by 4.50 to 10.00, MMLU-STEM by 2.19 to 3.84, and BBH by 13.13 to 17.78.
- BBH follows answer format more than ability: the `[invalid]` rate rises from 6.8% (base student) to 27.6–34.4% in trained models.
- This rate is close to the teacher's own 37.9%. The training responses rarely end with the lowercase `the answer is …` phrase that the `get-answer` regex needs.
- Responses also get longer. The median GSM8K response is 626 characters for `gram`, against 285 for the base student.

**The `direct_spectrum` model loses 12.00 points on MBPP (34.00 against 46.00).**
- `gram` and `cka` gain slightly on the same benchmark.
- This drop was not investigated.

**`hidden_mse` collapsed.**
- Its average is 13.69, and it scores below 5 on every generative task except SciQ (57.20) and MMLU-STEM (31.21), both of which are log-likelihood tasks.
- With weight 1.0, its auxiliary loss is 3.0004 at step 1990, against 0.6522 for `ds_loss`, so the hidden-state term dominated the gradient.
- Its dev loss is the worst of the four: 0.6953, against 0.52–0.53.
- The outputs degenerate:
  - The median GSM8K response is 887 characters, against 285 for the base student.
  - 218 of 2,638 GSM8K responses have more than half their lines duplicated, against 4 for the base student.
  - Its BBH `[invalid]` rate is 77.7%.
  - Its longest GSM8K output loops on "the question asks for the difference in earnings in dollars, which is $00."
  - Its first GSM8K answer computes `4 * 13 = 48` and then `$48 * $2 = $88`.

**The teacher scores below the base student on SciQ and BBH.**
- SciQ: 62.40 against 74.40. BBH: 44.45 against 45.26.
- The teacher's BBH score is limited by its 37.9% `[invalid]` rate.
- The SciQ gap was not investigated. Both scores come from the same pipeline as the other models, so the comparison between models is still consistent.

**Dev loss rose during training for every method.**
- It went from 0.3054 before training to 0.52–0.70 at the end.
- The training `ds_loss` was already 0.5731 at step 10, so the pre-train dev evaluation and the training loss do not appear to be measured the same way.
- This was not investigated. The dev set overlaps training, so the number is not evidence of generalization either way.

## Artifacts

All paths are relative to `/nvme/annp36-home/work/CST` on `vt-admin`.

### Score files and tables

- **Score files:** `benchmark_results/qwen25_tsd_structural/qwen25_tsd/{teacher,student,hidden_mse,gram,cka,direct_spectrum}/seed10/scores.json`. Each task's directory holds its `eval.log` and `samples_*.jsonl`.
- **Tables and SHA-256:**

| File | SHA-256 |
|---|---|
| `benchmark_results/qwen25_tsd_structural/tables/qwen25_tsd_full8_mean_std.csv` | `a4fefa797f65f35b8abf7eafc72a9be7ae0553750dd3a96e0b70fffc7597f05e` |
| `…/qwen25_tsd_full8_mean_std.json` | `b50645df332d8db96c020f026d557530bda318dfec1112be1657856c4fa9be66` |
| `…/qwen25_tsd_full8_mean_std.tex` | `7120a2bf3ed2c42cc05e8f3fd0146bb8adabffc020c26a6cd33e3e1e5f1872bd` |

- **Kept superseded and partial runs:**
  - `teacher/seed10/{bbh_cot_fewshot,scores}.superseded_20261007T030456Z*`
  - `student/seed10/{bbh_cot_fewshot,scores}.superseded_20261007T030922Z*`
  - `gram/seed10/gsm_plus.incomplete_20261007T103945Z`

### Final checkpoints

Each final checkpoint is `results/qwen25_tsd_structural/<method>/seed10/1992/`. Each run directory also keeps its epoch-1 checkpoint `996`.

| Checkpoint | `pytorch_model.bin` SHA-256 | Size (dir) |
|---|---|---|
| `hidden_mse/seed10/1992` | `22e3b71f25612de34f2109c8ae37fc966556c56f4528be53554c8dc5190faa5e` | 3.0 G |
| `hidden_mse/seed10/1992-vllm` (evaluated copy, no projectors) | `5a687892dc01a008dddf80e3972d09bdefde2b2996c75c3aa918c3a0bbf93bc1` | |
| `gram/seed10/1992` | `b71db2c56e7902d492dfc738233abf14e26e92b9f2cecbb430898fcb7a8f059a` | 2.9 G |
| `cka/seed10/1992` | `ae9ccec97010fb520641243e0a680816cbe44e5615aed8271b509c562467650b` | 2.9 G |
| `direct_spectrum/seed10/1992` | `3ab210d34a83a5c2ac8812fa1a8b733e2e8d2bb2ace9fd17567ff94c79adb17c` | 2.9 G |

### Logs

- **Training logs:** `results/qwen25_tsd_structural/<method>/seed10/{log.txt,train.log,args.json,TRAINING_COMPLETE}`.
- **Chain logs:**
  - Attempt 1: `logs/attempt1_20261006/qwen25_tsd_structural_20261005.{chain.log,exit,finished_at,pid,started_at}`, with exit `1`.
  - Continuations: `logs/qwen25_tsd_structural_20261005.{chain.log,chain2.log,chain3.log,chain3.waiting.log}`.
  - Final state: `logs/qwen25_tsd_structural_20261005.exit` contains `0`, and `.finished_at` is 2026-10-07 17:46:56.

### Sizes, GPU state and upload

- **Sizes:**
  - `results/qwen25_tsd_structural`: 27 G. `hidden_mse` is 8.8 G, including the export; each other method is 5.8 G.
  - `benchmark_results/qwen25_tsd_structural`: 2.0 G.
  - `/nvme`: 19% used, 5.7 T free.
- **GPU release:**
  - After the run, no process from this run holds a GPU.
  - GPUs 0, 1 and 3–7 show 0 MiB.
  - GPU 2 shows 16,557 MiB, held by another project's MTEB job (`.venv-mteb`), which was never touched.
- **Upload:** none. No destination is authorized in the run input.

## Limitations

- **Single seed (10).** There is no variance estimate, and differences of a few points, for example `cka` against the base student (+0.28), may be noise.
- **No `sfkl`-only baseline.**
  - The effect of each structural loss cannot be separated from output KD.
  - `gram_loss` was identically 0, so the Gram objective was not tested at all. This looks like an implementation or scale bug, not yet diagnosed.
- **`hidden_mse` was not tuned.** Weight 1.0 let the auxiliary loss dominate (about 4.6× `ds_loss`), so its collapse says little about hidden-state matching at a sensible weight.
- **BBH extraction is strict.**
  - The `get-answer` regex marks 27.6–77.7% of trained-model answers invalid.
  - BBH-COT therefore largely measures answer format.
  - The stop-sequence change is disclosed above and applies equally to all six models.
- **The dev set overlaps training.** The pre-train dev loss and the training loss appear to be measured differently.
- **Only the final checkpoint `1992` was evaluated.** The epoch-1 checkpoints were not scored.
- **The run did not finish in one pass.**
  - Evaluation finished across a failed chain, a stopped continuation and a resume, with finished task runs reused by exact command match.
  - Scores are unaffected, but `scores.json` lists the GPU pool rather than the GPU each task used, and the evaluator commit is recorded as `unknown`.
- **The SciQ gap between teacher and student was not investigated.** The teacher scores 62.40 against the student's 74.40.

## Next Steps

None of these have been run.

1. Train an `sfkl`-only baseline (no auxiliary loss) with the same data, seed and hyperparameters, to measure what each structural loss adds.
2. Diagnose why `gram_loss` is exactly 0, for example from its normalization or scale, the layers it reads, or a detached tensor. Then retrain `gram`.
3. Retrain `hidden_mse` with a lower or normalized weight, so the auxiliary loss does not dominate `ds_loss`.
4. Add seeds (for example 42) for the methods worth keeping, to get real error bars.
5. Add a lenient BBH extractor as a secondary, reference-only metric, and inspect the trained models' BBH and SciQ outputs.
6. Upload the artifacts, if a destination is authorized.
