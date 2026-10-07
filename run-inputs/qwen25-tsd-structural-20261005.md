# Project

- `status`: first launch stopped on 2026-10-06; relaunched from scratch for all four methods on 2026-10-06 at the user's request; training completed, the chain then failed (exit 1) at the `hidden_mse` evaluation; an evaluation-only continuation was launched on 2026-10-07 at the user's request and switched to a shared eight-GPU pool for the trained models the same morning; at the user's request all evaluation was stopped at 10:31 giờ Việt Nam and the GPUs were released; a resume of the unfinished evaluation was first scheduled for 18:00 giờ Việt Nam, then started at 17:39 giờ Việt Nam on the GPUs that were completely idle; the run completed with exit `0` at 17:46:56 giờ Việt Nam on 2026-10-07 and the terminal report is `run-outputs/out_qwen25-tsd-structural-20261005.md`.
- `name`: Qwen2.5 structural hidden-state distillation on TSD-KD generations.
- `run_id`: `qwen25_tsd_structural_20261005`.
- `local_path`: `/Users/savoxism/Documents/GitHub/nuno-kd`.
- `ssh_host`: `vt-admin` (HGX47, user `vt_admin`); SSH gate passed on 2026-10-06.
- `remote_path`: `/nvme/annp36-home/work/CST`, chosen by the user on 2026-10-06. Its previous contents (the separate CST ablation project) were removed at the user's request; a backup without virtual environments is at `/nvme/annp36-home/backups/CST-before-nuno-kd-20261006T073338Z.tar.gz`. Never deploy this repository into the existing `ORBIT-KD` directory.
- `source_revision`: executable implementation commit `fabb04af3ad5a532c1a37b3a803d71699716cf25` on `main`; deployed checkout is branch `qwen25-tsd-structural` (a descendant of it). The executable files must match the hashes below.
- `source_hashes`:
  - `configs/qwen25_tsd_structural.yaml`: `6cf2c08d73ca312496051fa4f45fa513eca3964509fc92dcf2b2ef64844a1bf7` (four-method, single-seed, `sfkl` without student generation, no resume; revision of 2026-10-06; the resume revision hashed `d1676dd38279c7f0cb80d69ea23e2050581f50beea342cf4595eb4cc094d8006`; the four-method adaptive revision hashed `e9c18801749c131aaf83b447f15e074566b435fae57773e8886192014874ea86`; the six-method single-seed revision hashed `41f4f23e61b3d5c711b58369c098e27544efb0f2eeee140fd6ae3f14bde5f007`; the original two-seed file at `fabb04a` hashed `731491a1d3ebca05c00ec8dd75235b8b1b95aad37a263b9a2506a6c3b09a0bcc`).
  - `scripts/run_qwen25_tsd_structural.sh`: `8a293dc086ff46ed32a63a9af8cacf67586913d71ccdbc0226a8b34d4f5d70c2` (passes `--reuse-complete`, 2026-10-07; before that `a32c45cc9cac96a4796835c271b29976f1950cb4173e1df5a40fc61e48ceb348`: all evaluated models run concurrently on the shared GPU pool; the revision before hashed `e6bf040ca8674a329f030cbd90839c3a1f8cfe7e47c488d6e2177bf886dcb74b`: student-generation flags only for adaptive types, optional `training.resume`; the file at `fabb04a` hashed `573a31a2327baa54cdadc82446883a603ada4702801c63888054d3305003eba7`).
  - `baselines/eval_lm_harness.py`: `ea3856dc3fbe5acbdbbdedc9fab3b968aa84666aeb22eb1571b0cf23cf81cb94` (a GPU is used only when completely idle, 2026-10-07; before that `5ee3b289f015eb1844f0e944646efdd3c6726f19c457a31434c199b56967f045`: `--reuse-complete`; before that `321e11038cc4f7f4cbffbe3ed0a474877d0af6d56c54721e961d2283641c165c`: shared GPU lease pool; the revision before hashed `01235fc279d43eabdbf65ff6f9704a7c1f85941172f26db85f10799dd71d0531`: BBH stop at `<|im_end|>` only, projector-free vLLM export, `--tasks` rerun mode; 2026-10-07; the `get-answer` revision hashed `4a66eb3712bcc339e29f18838443875da338fd6109046a0cf6cd74e7a5387b1a`; the file at `fabb04a` hashed `4316a091680eb274dcd2773399af31bed856c7d9361536ec9c17225f5e95e4fe`).
  - `finetune.py`: `d40f75ec9f70f4d2bfb3c4d329e919e2cf04678129ed3cf722ad2ceb5c788e94` (weights-only epoch resume; the file at `fabb04a` hashed `8cfb7d7facbf03b0190af2f3f2b4ef95f05fe4190865dc73202d35221f6c4834`).
  - `arguments.py`: `8b7bbc6a866b7b82ff6918183b3d1f0f3689517ef7ff6a8045fb80320938cb7d` (adds `--resume-ckpt`, `--resume-global-step`; previously `57f5038477a63a0fbb9a6a2d389b39bee15a20259ef1fd0aed7abbcc8e626a0a`).
- `authoritative_input`: `run-inputs/qwen25-tsd-structural-20261005.md`.
- `terminal_report`: `run-outputs/out_qwen25-tsd-structural-20261005.md`; create or update it after terminal success or failure, before reporting completion.

# Schedule

- `start_at`: as soon as source, SSH, data, model, evaluator, disk, environment, and GPU preflight all pass.
- `timezone`: `Asia/Ho_Chi_Minh`.
- `max_runtime`: none set; run to completion under the health monitor.
- `execution_policy`: one sequential DDP training job at a time. Each job uses all eight H200 GPUs. Do not overlap methods or seeds.
- `job_order`: seed `10` only, one run per method: `hidden_mse`, `gram`, `cka`, `direct_spectrum`; evaluation follows training and the table report follows complete evaluation.

# Data

- `input`: `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` at revision `450c33f2d591f68e12ecb546c16cf79821ef0006`.
- `source_file`: `Qwen2.5-1.5B-Instruct_train_all.jsonl`, 79,751 ordered examples, 233,898,937 bytes, SHA-256 `c9b0a1cef6025334a9f947dc20d591b79baf88d6a1c19f2ac7a684d8330862bb`.
- `source_schema`: `instruction`, message-list `prompt`, and student-generated `response`.
- `canonical_train`: `data/tsd_kd_qwen25/canonical/train.jsonl`, converted without reordering to string `prompt` plus `generated_text`.
- `processed_train`: `processed_data/tsd_kd_qwen25/Qwen/Qwen2.5-1.5B-Instruct/{train_0.bin,train_0.idx,train.jsonl}`.
- `validation`: the processor creates a 200-example in-run sanity set. It overlaps the training collection and must not be reported as an unbiased validation result.
- `manifest`: `processed_data/tsd_kd_qwen25/Qwen/Qwen2.5-1.5B-Instruct/dataset_contract.json`; its ordered and multiset hashes must match the canonical train file.
- `benchmarks`: `gsm8k`, `gsm_plus`, `minerva_math`, `mbpp`, `sciq`, `mmlu_stem`, `mmlu_pro_math`, and `bbh_cot_fewshot` through `baselines/eval_lm_harness.py`.
- `headline_metrics`: GSM flexible extract, GSM-Plus flexible extract, MATH `math_verify`, MBPP pass@1, SciQ normalized accuracy, MMLU-STEM accuracy, MMLU-Pro-Math custom extract, and BBH-COT `exact_match,get-answer` (the pinned lm-eval commit defines no flexible-extract filter for `bbh_cot_fewshot`). Scores are multiplied by 100.

# Model

- `teacher_repo_id`: `Qwen/Qwen2.5-14B-Instruct`.
- `teacher_revision`: `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- `student_repo_id`: `Qwen/Qwen2.5-1.5B-Instruct`.
- `student_revision`: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- `model_paths`: pin each model with `snapshot_download` in the project virtual environment, record the returned local snapshot directories in `logs/qwen25_tsd_structural_20261005_model_paths.json`, and create a server-only runtime YAML that replaces Hub IDs with those snapshot paths before offline execution.
- `task`: causal-LM knowledge distillation with off-policy skew-forward KL output KD (dataset responses only, no student generation) plus one structural hidden-state auxiliary objective.
- `methods`: `hidden_mse`, `gram`, `cka`, and `direct_spectrum`.
- `output_kd`: enabled for every method; `sfkl`, ratio `1.0`, skew alpha `0.1`, student generation disabled (no `--student-gen`, no replay buffer).
- `auxiliary_weights`: `1.0` for Hidden MSE, Gram, CKA, and direct spectrum. The YAML still holds normalized spectrum `1.0` and CST `0.003` because the launcher reads every weight key; they are unused.
- `seeds`: `10` only (single seed, user decision 2026-10-06).
- `epochs`: `2`.
- `optimizer`: AdamW, learning rate `5e-6`, weight decay `1e-2`, cosine schedule, LR warmup ratio `0.1`, gradient clip `1.0`.
- `sequence`: max length `1024`, max prompt length `512`.
- `precision`: DeepSpeed ZeRO-2 bf16 configuration for the student; frozen teacher fp16.
- `GPU_policy`: exactly GPUs `0,1,2,3,4,5,6,7`, all verified as H200 and free before launch; do not fall back to fewer GPUs.
- `batch`: per-device microbatch `1`, gradient accumulation `10`, eight GPUs, therefore effective global batch

  $$B_{\mathrm{global}}=8\times1\times10=80.$$

- `checkpoint_selection`: final epoch checkpoint only; benchmark scores are never used for selection.

# Commands

## 1. Local source gate and push

Run from the local repository:

```bash
cd /Users/savoxism/Documents/GitHub/nuno-kd
git status --short
git log -2 --oneline
git push origin main
```

Do not continue until the push succeeds and `origin/main` contains the implementation and this plan.

## 2. SSH and deployment gate

The first command must succeed; its current failure to resolve `vt-admin` blocks the run.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 vt-admin 'hostname && id && test -d /nvme/annp36-home/work && echo REMOTE_BASE_OK'
```

Deploy without deleting remote files:

```bash
ssh vt-admin 'if test -d /nvme/annp36-home/work/CST/.git; then cd /nvme/annp36-home/work/CST && test -z "$(git status --porcelain)" && git pull --ff-only; else git clone --branch qwen25-tsd-structural https://github.com/chiiipk/nuno-kd.git /nvme/annp36-home/work/CST; fi'
```

If the remote tree is dirty, stop and report it; do not reset or overwrite it. Then remove AppleDouble files only inside the resolved project root and verify source hashes:

```bash
ssh vt-admin 'cd /nvme/annp36-home/work/CST && find "$PWD" -type f -name "._*" -print -delete && git merge-base --is-ancestor fabb04af3ad5a532c1a37b3a803d71699716cf25 HEAD && sha256sum configs/qwen25_tsd_structural.yaml scripts/run_qwen25_tsd_structural.sh baselines/eval_lm_harness.py finetune.py arguments.py'
```

## 3. Hardware and process gate

```bash
ssh vt-admin 'cd /nvme/annp36-home/work/CST && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader && nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader && df -h . /dev/shm && free -g'
```

Acceptance for this gate: exactly eight H200 GPUs; none owned or occupied by another user/process; sufficient disk for two model snapshots, dataset/cache, four final runs (two epoch checkpoints each), raw benchmark samples, and logs. If any required GPU is occupied, wait or ask the user; never stop another process.

## 4. Project environment and Hugging Face authentication

On `vt-admin`:

```bash
cd /nvme/annp36-home/work/CST
UV_CACHE_DIR=/nvme/annp36-home/.cache/uv UV_PYTHON_INSTALL_DIR=/nvme/annp36-home/.cache/uv/python \
  /nvme/annp36-home/.uvboot/bin/uv sync --python 3.10
source .venv/bin/activate
set +x
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER
hf auth whoami || python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    token_path = Path(os.environ.get("HF_TOKEN_FILE", str(Path.home() / ".config/huggingface/token")))
    if token_path.is_file():
        token = token_path.read_text().strip()
if not token:
    raise SystemExit("No HF_TOKEN or protected HF token file; stop and ask the user")
login(token=token, add_to_git_credential=False)
PY
```

Never print the token and never enable shell tracing.

## 5. Pinned online prefetch and preprocessing

Download the pinned raw dataset through the project Python so the subsequent launcher does not use its curl fallback:

```bash
cd /nvme/annp36-home/work/CST
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
python - <<'PY'
from huggingface_hub import hf_hub_download

print(hf_hub_download(
    repo_id="Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen",
    repo_type="dataset",
    revision="450c33f2d591f68e12ecb546c16cf79821ef0006",
    filename="Qwen2.5-1.5B-Instruct_train_all.jsonl",
    local_dir="data/tsd_kd_qwen25/raw",
))
PY
sha256sum data/tsd_kd_qwen25/raw/Qwen2.5-1.5B-Instruct_train_all.jsonl
bash scripts/run_qwen25_tsd_structural.sh prepare configs/qwen25_tsd_structural.yaml
```

Prefetch pinned model snapshots and produce a runtime YAML with local model paths:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

teacher = snapshot_download("Qwen/Qwen2.5-14B-Instruct", revision="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
student = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
paths = {"teacher": teacher, "student": student}
Path("logs").mkdir(exist_ok=True)
Path("logs/qwen25_tsd_structural_20261005_model_paths.json").write_text(json.dumps(paths, indent=2) + "\n")
cfg = yaml.safe_load(Path("configs/qwen25_tsd_structural.yaml").read_text())
cfg["models"]["teacher"] = teacher
cfg["models"]["student"] = student
Path("configs/qwen25_tsd_structural.remote.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(json.dumps(paths, indent=2))
PY
```

The processed dataset must already pass its contract before using the runtime YAML, because a local snapshot path must not be used as the processor's output suffix.

## 6. Pinned evaluator and offline smoke gate

Pin lm-eval v0.4.12 at commit `6d642546f4688648fced259eb3302efd36ece5af`:

```bash
LM_EVAL_REF=6d642546f4688648fced259eb3302efd36ece5af bash baselines/setup_env.sh evaluator
test "$(cat baselines/lm_eval_commit.txt)" = 6d642546f4688648fced259eb3302efd36ece5af
```

Warm all eight benchmark datasets online with one sample, using the pinned local student snapshot:

```bash
STUDENT_SNAPSHOT="$(python -c 'import json; print(json.load(open("logs/qwen25_tsd_structural_20261005_model_paths.json"))["student"])')"
baselines/vendor/lm-evaluation-harness/.venv/bin/python baselines/eval_lm_harness.py \
  --checkpoint "${STUDENT_SNAPSHOT}" \
  --output benchmark_results/qwen25_tsd_structural/.cache_warmup_online \
  --gpus 0,1,2,3,4,5,6,7 --limit 1
```

Repeat the smoke test offline into a different directory; all eight tasks must produce non-empty sample logs and a parseable `scores.json`:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
baselines/vendor/lm-evaluation-harness/.venv/bin/python baselines/eval_lm_harness.py \
  --checkpoint "${STUDENT_SNAPSHOT}" \
  --output benchmark_results/qwen25_tsd_structural/.cache_smoke_offline \
  --gpus 0,1,2,3,4,5,6,7 --limit 1
```

## 7. Durable sequential launch

Only after all gates pass:

```bash
cd /nvme/annp36-home/work/CST
mkdir -p logs run-outputs
test ! -f logs/qwen25_tsd_structural_20261005.pid || ! kill -0 "$(cat logs/qwen25_tsd_structural_20261005.pid)" 2>/dev/null
nohup bash -lc '
  cd /nvme/annp36-home/work/CST
  source .venv/bin/activate
  set +x
  export HF_HUB_DISABLE_XET=1
  unset HF_HUB_ENABLE_HF_TRANSFER
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  date --iso-8601=seconds > logs/qwen25_tsd_structural_20261005.started_at
  set +e
  bash scripts/run_qwen25_tsd_structural.sh all configs/qwen25_tsd_structural.remote.yaml
  rc=$?
  printf "%s\n" "$rc" > logs/qwen25_tsd_structural_20261005.exit
  date --iso-8601=seconds > logs/qwen25_tsd_structural_20261005.finished_at
  exit "$rc"
' > logs/qwen25_tsd_structural_20261005.chain.log 2>&1 < /dev/null &
echo $! > logs/qwen25_tsd_structural_20261005.pid
```

## 8. Monitoring

- Every 20 minutes, read only: PID/exit state; current method/seed/phase; optimizer step; finite loss and gradient; newest log/checkpoint timestamp; disk; and every GPU's owner, utilization, VRAM, temperature, and power.
- Scan logs for `Traceback`, `CUDA out of memory`, `NCCL`, `nan`, `inf`, killed workers, and disk-full errors. One idle sample is not a stall.
- Before launch, create `scripts/watch_qwen25_tsd_structural_20261005.sh` on the remote project. It must refresh every 30 seconds, display phase, method, seed, progress, latest loss/LR, elapsed time, ETA when derivable, PID/exit state, and all eight GPU health rows. `Ctrl-C` stops only the watcher.
- Direct watcher command:

  ```bash
  ssh -t vt-admin 'cd /nvme/annp36-home/work/CST && INTERVAL=10 bash scripts/watch_qwen25_tsd_structural_20261005.sh'
  ```

## 9. Terminal verification and report

After the PID exits, require exit code `0`, four valid final checkpoints, six parseable score files (teacher, student, and 4 trained runs), complete eight-task sample logs, and CSV/JSON/LaTeX tables:

```bash
cd /nvme/annp36-home/work/CST
cat logs/qwen25_tsd_structural_20261005.exit
find results/qwen25_tsd_structural -name config.json -print
find benchmark_results/qwen25_tsd_structural/qwen25_tsd -name scores.json -print
find benchmark_results/qwen25_tsd_structural/tables -maxdepth 1 -type f -print
grep -RniE 'Traceback|CUDA out of memory|NCCL|No space left|Killed|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
  logs/qwen25_tsd_structural_20261005.chain.log results/qwen25_tsd_structural || true
```

Create or update `run-outputs/out_qwen25-tsd-structural-20261005.md` with the required Header, Settings, Main Results, Insights, Artifacts, Limitations, and Next Steps sections. If the run failed, still write the report with the exit code and preserved partial artifacts before reporting failure.

# Outputs

- `run_root`: `/nvme/annp36-home/work/CST`.
- `train_root`: `results/qwen25_tsd_structural/<method>/seed<seed>/`.
- `checkpoints`: numeric Hugging Face checkpoint directories under each training run; the highest completed epoch checkpoint is final.
- `eval_root`: `benchmark_results/qwen25_tsd_structural/qwen25_tsd/{teacher,student,<method>}/seed<seed>/`.
- `tables`: `benchmark_results/qwen25_tsd_structural/tables/qwen25_tsd_full8_mean_std.{csv,json,tex}`.
- `chain_log`: `logs/qwen25_tsd_structural_20261005.chain.log`.
- `pid_file`: `logs/qwen25_tsd_structural_20261005.pid`.
- `exit_file`: `logs/qwen25_tsd_structural_20261005.exit`.
- `model_manifest`: `logs/qwen25_tsd_structural_20261005_model_paths.json`.
- `runtime_config`: `configs/qwen25_tsd_structural.remote.yaml`, server-only and containing local cache paths but no secret.
- `terminal_report`: `run-outputs/out_qwen25-tsd-structural-20261005.md`.

# Upload

- `destination`: none authorized for this run.
- `transport`: do not upload or infer a Drive path. Record `rclone listremotes` in the terminal report only if upload is later requested.
- `included_artifacts_if_authorized_later`: final checkpoints, `scores.json`, tables, raw sample logs, chain/training/evaluation logs, runtime configuration, model/data hashes, and terminal report. Never upload tokens, environment files, Hub caches, optimizer states, or unrelated repository data.

# Acceptance

- SSH connectivity to `vt-admin` and the remote base passes before deployment.
- The executable source hashes match the pinned implementation revision.
- Environment setup, Hugging Face authentication, exact dataset checksum/count, processed-data contract, pinned model snapshots, evaluator commit, online cache warmup, and offline smoke evaluation all pass before training.
- Exactly eight free H200 GPUs are used by one DDP training job at a time; no silent fallback and no concurrent methods/seeds.
- Every method uses `sfkl` output KD with ratio `1.0` and no student generation; every auxiliary weight is `1.0`.
- Every run records two epochs, seed, LR `5e-6`, microbatch `1`, gradient accumulation `10`, global batch `80`, and the declared objective in `args.json` and logs.
- All four method runs (seed `10`) exit successfully and have a complete final checkpoint.
- Teacher, base student, and every trained checkpoint have all eight benchmark scores with non-empty raw samples. Benchmark metrics are used only for reporting.
- Every loss and reported metric is finite; no traceback, CUDA/host OOM, NCCL failure, worker death, or disk-full error. Any occurrence fails the gate and is documented rather than silently retried.
- The final table has no missing dataset cell. With one seed, `report_mean_std.py` writes the seed-10 value as the mean and `sample_std` 0.0; that 0.0 is not a variance estimate and must be reported as single-seed, without ± error bars.
- Retry policy: preserve partial artifacts; diagnose first; resume only completed-safe phases. Never change data, seed, batch, objective, sampler, precision, GPU allocation, or hyperparameters without explicit user approval and an update to this same plan's change history.
- `run-outputs/out_qwen25-tsd-structural-20261005.md` exists in the project repository and reflects the terminal state before completion is declared.

# Change history

- `2026-10-05`: initial authoritative plan created from the user-approved eight-H200 policy. `ssh_host` and remote project base were taken from the user-provided OrbitKD example; upload is deliberately disabled because no destination was authorized.
- `2026-10-06`: `remote_path` changed from `/nvme/annp36-home/work/nuno-kd` to `/nvme/annp36-home/work/CST` at the user's request (old CST contents removed after a backup); `local_path` set to this Mac's checkout; deployment clones branch `qwen25-tsd-structural`; `uv` is not on the server PATH, so step 4 calls `/nvme/annp36-home/.uvboot/bin/uv` with caches on `/nvme`. Data, models, seeds, batch, objectives and hyperparameters are unchanged.
- `2026-10-06`: seeds reduced from `10`, `42` to `10` only at the user's explicit request (`training.seeds: [10]` in the YAML, new hash above; watcher `SEEDS=(10)`). The sweep is now 6 training runs and 8 evaluated models; results are single-seed with no seed variance. All other data, objectives, batch and hyperparameters are unchanged.
- `2026-10-06`: the online smoke gate failed with `KeyError: No supported metric found for bbh_cot_fewshot`, because lm-eval `6d64254` reports only `exact_match,get-answer` (regex `the answer is …`) for that group. With the user's approval, `baselines/eval_lm_harness.py` adds `exact_match,get-answer` after `exact_match,flexible-extract` in the BBH metric priority (new hash above); the BBH-COT headline is therefore `get-answer`. No other task, decoding setting or training setting changed.
- `2026-10-06`: at the user's request the first launch (started 15:14 giờ Việt Nam) was stopped at about 16:07 giờ Việt Nam while `hidden_mse/seed10` was near step 1010/1992 (dev loss rose 0.305 → 0.721 after epoch 1, so adaptive student generation had switched on and step time rose from ~2.6 s to 6–9 s). No run completed; the partial `results/qwen25_tsd_structural/hidden_mse/seed10/` (epoch-1 checkpoint `996`) is kept on the server and must be removed or moved before relaunch. Separately, at the user's explicit request `normalized_spectrum` and `cst` were removed from `training.methods` (new YAML hash above; watcher `METHODS` updated); their `training.auxiliary` weights stay in the YAML because the launcher reads every weight key. The sweep is now 4 training runs and 6 evaluated models. Objective (`adaptive-sfkl` with student generation), data, seed, batch and hyperparameters are unchanged.
- `2026-10-06`: at the user's explicit request student generation is removed: `training.output_kd` is now `type: sfkl`, `student_generation: false`. The launcher passes `--student-gen` and the adaptive/replay flags only when `student_generation` is true, and refuses an adaptive type without it or student generation with a non-adaptive type. In `finetune.py` the loss for `sfkl` and `adaptive-sfkl` is the same `skewed_forward_kl`; with `adaptive-sfkl` at threshold `0.0` no batch is generated or replayed, so epoch 1 of the stopped `hidden_mse/seed10` run was already pure off-policy SKL and matches the new objective.
- `2026-10-06`: also at the user's request, `hidden_mse/seed10` continues from its epoch-1 checkpoint instead of restarting. This supersedes the earlier note that the partial directory must be removed. `finetune.py`/`arguments.py` gain `--resume-ckpt`/`--resume-global-step`. These load the checkpoint's bf16 weights and projectors strictly before DeepSpeed init. They then start at epoch 1 with the uninterrupted run's counters (step 9969, global step 997), sampler epoch 1, and the cosine schedule replayed 996 steps (LR `2.9336e-06`). The YAML's `training.resume` maps `hidden_mse/seed10: 996`. A CPU-only check on the server confirmed four things: the replayed schedule reproduces the stopped run's logged LR at global iters 990/1000/1010 exactly; all 343 tensors load bit-exactly in bf16 with tied embeddings intact; the launcher passes the resume flags only to `hidden_mse/seed10`; and it archives the old logs. Known deviations of the resumed run from an uninterrupted one, to be disclosed in the report: AdamW moments restart from zero; the fp32 master weights were never saved, so training resumes from the bf16 copy; and the checkpoint was written after optimizer step 995, so epoch-1 micro-batches 9951–9968 on each rank (18 × 8 = 144 samples, 0.18% of an epoch) are not trained. Data, seed, batch, LR schedule, precision and GPU allocation are unchanged. Estimated time after launch: about 50 min to finish `hidden_mse`, about 4 h 25 m for the other three runs at the observed ~2.56 s/step, then 5–7.5 h of evaluation for 6 models. That is about 10.5–13 h in total.
- `2026-10-06`: the user decided against the resume. The partial `results/qwen25_tsd_structural/hidden_mse/seed10` directory (epoch-1 checkpoint `996` and logs of the stopped adaptive attempt) is deleted at the user's explicit request, the `training.resume` block is removed from the YAML, and all four methods train from scratch with `sfkl` and no student generation. The resume support in `finetune.py`, `arguments.py` and the launcher stays but is inactive. Estimated time after launch: about 5 h 40 m of training (4 × 1992 steps at ~2.56 s/step plus load/save), then 5–7.5 h of evaluation for 6 models, about 11–13 h in total.
- `2026-10-07`: training of all four methods completed (`TRAINING_COMPLETE`, final checkpoint `1992` each; final dev loss hidden_mse `0.6953`, gram `0.5252`, cka `0.5223`, direct_spectrum `0.5259`). Teacher and base-student evaluation completed, then the chain exited `1` at 02:40 giờ Việt Nam. vLLM refused the `hidden_mse` checkpoint because it also holds the four learned `projectors.*` tensors. gram, cka and direct_spectrum were not evaluated. Two problems were found and fixed in `baselines/eval_lm_harness.py`:
  - **Projector checkpoint.** A checkpoint with training-only tensors is now evaluated through a sibling `<step>-vllm` copy without them (`EXPORT_COMPLETE` lists the dropped keys). The student weights are unchanged.
  - **BBH stop sequences.** The lm-eval BBH CoT template stops at `\n\n` and `Q`, which truncated chat answers after their first paragraph; 4506/6511 teacher BBH samples were `[invalid]` (score 25.28). At the user's explicit request BBH now stops only at `<|im_end|>` (plus tokenizer EOS), passed as `until` through `--gen_kwargs`. The `get-answer` regex and every other task are unchanged.
  - **Smoke results** (`--limit`, scratch output, not reported). The `hidden_mse` export loads in vLLM for all eight tasks. On the teacher, BBH `[invalid]` fell from 69% to 32%; the rest are answers without the lowercase `the answer is` phrase.
  - **Continuation chain**, using the same pid/exit/chain-log names; the first attempt's files are moved to `logs/attempt1_20261006/`. It runs `eval_lm_harness.py --tasks bbh_cot_fewshot` into the existing teacher and student score files, keeping the superseded BBH directories and score files as `*.superseded_<UTC>`. It then runs `scripts/run_qwen25_tsd_structural.sh eval`, which skips teacher/student (`scores.json` present) and evaluates the four trained checkpoints, and finally `report`. Estimated 1.5–2.5 h.
- `2026-10-07`: the BBH reruns finished: teacher `44.45`, base student `45.26` (`exact_match,get-answer`, 6511 samples each). At the user's request ("dùng cả 8 H200 cho eval") evaluation GPU scheduling changed; no task, decoding setting, checkpoint or metric changed. Before, `eval_lm_harness.py` ran one model at a time in waves of one task per GPU, so GPUs sat idle until the slowest task of each model ended (and a single-task rerun used one GPU). Now every task takes an exclusive `flock` lease on one GPU from a pool shared by all concurrent evaluations (default lock dir `/tmp/eval_lm_harness_gpu_locks_<uid>`). It starts only when that GPU's memory is below 2048 MiB, and `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes the CUDA index match `nvidia-smi`. `run_eval` in the launcher starts every model's evaluation at once and fails if any of them fails. A local test with 4 simulated models × 8 tasks on 8 fake GPUs never double-booked a GPU and reached 8 concurrent tasks. Switch-over at 10:12 giờ Việt Nam: the continuation wrappers were stopped after the BBH reruns, while the `hidden_mse` evaluation (old code, one wave on GPUs 0–7) kept running untouched. A second continuation evaluates gram, cka and direct_spectrum concurrently on the pool; each new task waits until a `hidden_mse` task frees its GPU. When `hidden_mse` has its `scores.json`, the continuation runs `scripts/run_qwen25_tsd_structural.sh eval` (all six models must then be skipped) and `report`. Estimated finish about 50–80 min after the switch.
- `2026-10-07`: at the user's request ("Dừng eval và giải phóng hết cả 8 GPU H200") every evaluation process was stopped at 10:31 giờ Việt Nam: the second continuation, the four `eval_lm_harness.py` processes, their `lm_eval` children and vLLM engines. All eight GPUs then showed 0 MiB and no compute processes. No exit or finished_at file was written and `report` did not run. Complete `scores.json` exist for teacher, base student and cka only. hidden_mse, gram and direct_spectrum have partial task directories, kept on the server unchanged. Tables and the run-output report are not produced.
- `2026-10-07`: at the user's request the unfinished evaluation resumes on the eight-GPU pool about 30 minutes after their message, from 18:00 giờ Việt Nam. The stopped processes had already finished lm-eval for all eight tasks of hidden_mse and direct_spectrum, but were killed before writing `scores.json`. gram had six finished tasks, an interrupted `gsm_plus` and no `mbpp`. `eval_lm_harness.py --reuse-complete` (passed by the launcher) scores a task directory without rerunning it only if `eval.log` records the same command (interpreter dropped, paths resolved) and a `results*.json` exists. A finished run of a different command is an error. An unfinished directory is kept as `<task>.incomplete_<UTC>` and rerun. A CPU-only dry check on the server chose: reuse for all tasks of hidden_mse, cka and direct_spectrum; rerun gram `gsm_plus` (interrupted); run gram `mbpp`. Only these two tasks need GPUs, so at most two of the eight run at once. When the run was scheduled, all eight GPUs held another project's MTEB jobs (`.venv-mteb`, 3–24 GiB each). Those jobs are not touched: a task starts only on a GPU whose memory is below 2048 MiB, so it waits while they run. A third continuation (`logs/qwen25_tsd_structural_20261005.eval_continue3.sh`, log `chain3.log`, same pid/exit names) sleeps until 18:00 giờ Việt Nam, then runs `scripts/run_qwen25_tsd_structural.sh eval` (teacher, student and cka skipped through `scores.json`) and `report`. Expected about 10–15 min once two GPUs are free.
- `2026-10-07`: the user then asked to start right away, but only on GPUs that are completely idle, never sharing a GPU with another job. At 17:36 giờ Việt Nam GPUs 4 and 6 had no process and 0 MiB; the other six held MTEB jobs. `lease_gpu()` now treats a GPU as usable only when `nvidia-smi` lists no compute process on it and its memory is below 2048 MiB. A GPU that fails the check is released at once, so the task moves on to the next GPU instead of waiting on a busy one. A local test with GPUs 0–3, 5 and 7 marked busy placed the two gram tasks on GPUs 4 and 6. The 18:00 continuation was stopped while it was still waiting and before it had run anything. The same `eval_continue3.sh` was relaunched without the wait; the start-time line was removed.
- `2026-10-07`: the resume finished with exit `0` at 17:46:56 giờ Việt Nam. hidden_mse and direct_spectrum reused all eight finished task runs; gram reused six, reran `gsm_plus` on GPU 0 and ran `mbpp` on GPU 1; `report` wrote the CSV/JSON/LaTeX tables. The §9 gates passed: four final checkpoints, six `scores.json` with all eight tasks, and no Traceback/OOM/NCCL/NaN/Inf match in the current chain and training logs (the only traceback is the archived attempt-1 failure). Single-seed averages: teacher 67.89, base student 51.69, gram 53.29, cka 51.97, direct_spectrum 48.37, hidden_mse 13.69. The report also records two findings that need follow-up before any retraining: `gram_loss` logged `0.0000` throughout, so the gram run is effectively `sfkl`-only, and hidden_mse collapsed (its auxiliary loss is about 4.6× `ds_loss`). No upload.
