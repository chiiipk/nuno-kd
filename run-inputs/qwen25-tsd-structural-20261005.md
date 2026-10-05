# Project

- `status`: planned; not launched. Training may start only after every preflight gate below passes.
- `name`: Qwen2.5 structural hidden-state distillation on TSD-KD generations.
- `run_id`: `qwen25_tsd_structural_20261005`.
- `local_path`: `/Users/skye/Downloads/artifacts/nuno-kd`.
- `ssh_host`: `vt-admin`, taken from the user-provided OrbitKD example. The alias did not resolve from the current Mac on 2026-10-05, so SSH connectivity is a hard preflight gate rather than an assumed capability.
- `remote_path`: `/nvme/annp36-home/work/nuno-kd`, using the verified `/nvme/annp36-home/work` project base from the example and a project-specific directory. Never deploy this repository into the existing `ORBIT-KD` directory.
- `source_revision`: executable implementation commit `fabb04af3ad5a532c1a37b3a803d71699716cf25` on `main`. The later plan-only commit may be checked out too, but the executable files must match the hashes below.
- `source_hashes`:
  - `configs/qwen25_tsd_structural.yaml`: `731491a1d3ebca05c00ec8dd75235b8b1b95aad37a263b9a2506a6c3b09a0bcc`.
  - `scripts/run_qwen25_tsd_structural.sh`: `573a31a2327baa54cdadc82446883a603ada4702801c63888054d3305003eba7`.
  - `baselines/eval_lm_harness.py`: `4316a091680eb274dcd2773399af31bed856c7d9361536ec9c17225f5e95e4fe`.
  - `finetune.py`: `8cfb7d7facbf03b0190af2f3f2b4ef95f05fe4190865dc73202d35221f6c4834`.
- `authoritative_input`: `run-inputs/qwen25-tsd-structural-20261005.md`.
- `terminal_report`: `run-outputs/out_qwen25-tsd-structural-20261005.md`; create or update it after terminal success or failure, before reporting completion.

# Schedule

- `start_at`: as soon as source, SSH, data, model, evaluator, disk, environment, and GPU preflight all pass.
- `timezone`: `Asia/Ho_Chi_Minh`.
- `max_runtime`: none set; run to completion under the health monitor.
- `execution_policy`: one sequential DDP training job at a time. Each job uses all eight H200 GPUs. Do not overlap methods or seeds.
- `job_order`: for seed `10`, then seed `42` within each method: `hidden_mse`, `gram`, `cka`, `normalized_spectrum`, `direct_spectrum`, `cst`; evaluation follows training and the table report follows complete evaluation.

# Data

- `input`: `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` at revision `450c33f2d591f68e12ecb546c16cf79821ef0006`.
- `source_file`: `Qwen2.5-1.5B-Instruct_train_all.jsonl`, 79,751 ordered examples, 233,898,937 bytes, SHA-256 `c9b0a1cef6025334a9f947dc20d591b79baf88d6a1c19f2ac7a684d8330862bb`.
- `source_schema`: `instruction`, message-list `prompt`, and student-generated `response`.
- `canonical_train`: `data/tsd_kd_qwen25/canonical/train.jsonl`, converted without reordering to string `prompt` plus `generated_text`.
- `processed_train`: `processed_data/tsd_kd_qwen25/Qwen/Qwen2.5-1.5B-Instruct/{train_0.bin,train_0.idx,train.jsonl}`.
- `validation`: the processor creates a 200-example in-run sanity set. It overlaps the training collection and must not be reported as an unbiased validation result.
- `manifest`: `processed_data/tsd_kd_qwen25/Qwen/Qwen2.5-1.5B-Instruct/dataset_contract.json`; its ordered and multiset hashes must match the canonical train file.
- `benchmarks`: `gsm8k`, `gsm_plus`, `minerva_math`, `mbpp`, `sciq`, `mmlu_stem`, `mmlu_pro_math`, and `bbh_cot_fewshot` through `baselines/eval_lm_harness.py`.
- `headline_metrics`: GSM flexible extract, GSM-Plus flexible extract, MATH `math_verify`, MBPP pass@1, SciQ normalized accuracy, MMLU-STEM accuracy, MMLU-Pro-Math custom extract, and BBH-COT flexible extract. Scores are multiplied by 100.

# Model

- `teacher_repo_id`: `Qwen/Qwen2.5-14B-Instruct`.
- `teacher_revision`: `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- `student_repo_id`: `Qwen/Qwen2.5-1.5B-Instruct`.
- `student_revision`: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- `model_paths`: pin each model with `snapshot_download` in the project virtual environment, record the returned local snapshot directories in `logs/qwen25_tsd_structural_20261005_model_paths.json`, and create a server-only runtime YAML that replaces Hub IDs with those snapshot paths before offline execution.
- `task`: causal-LM knowledge distillation with adaptive skew-forward output KD plus one structural hidden-state auxiliary objective.
- `methods`: `hidden_mse`, `gram`, `cka`, `normalized_spectrum`, `direct_spectrum`, and `cst`.
- `output_kd`: enabled for every method; `adaptive-sfkl`, ratio `1.0`, skew alpha `0.1`, student generation enabled.
- `auxiliary_weights`: `1.0` for Hidden MSE, Gram, CKA, normalized spectrum, and direct spectrum; `0.003` for CST.
- `seeds`: `10`, `42`.
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
cd /Users/skye/Downloads/artifacts/nuno-kd
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
ssh vt-admin 'if test -d /nvme/annp36-home/work/nuno-kd/.git; then cd /nvme/annp36-home/work/nuno-kd && test -z "$(git status --porcelain)" && git pull --ff-only origin main; else git clone https://github.com/chiiipk/nuno-kd.git /nvme/annp36-home/work/nuno-kd; fi'
```

If the remote tree is dirty, stop and report it; do not reset or overwrite it. Then remove AppleDouble files only inside the resolved project root and verify source hashes:

```bash
ssh vt-admin 'cd /nvme/annp36-home/work/nuno-kd && find "$PWD" -type f -name "._*" -print -delete && git merge-base --is-ancestor fabb04af3ad5a532c1a37b3a803d71699716cf25 HEAD && shasum -a 256 configs/qwen25_tsd_structural.yaml scripts/run_qwen25_tsd_structural.sh baselines/eval_lm_harness.py finetune.py'
```

## 3. Hardware and process gate

```bash
ssh vt-admin 'cd /nvme/annp36-home/work/nuno-kd && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader && nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader && df -h . /dev/shm && free -g'
```

Acceptance for this gate: exactly eight H200 GPUs; none owned or occupied by another user/process; sufficient disk for two model snapshots, dataset/cache, twelve full checkpoints, raw benchmark samples, and logs. If any required GPU is occupied, wait or ask the user; never stop another process.

## 4. Project environment and Hugging Face authentication

On `vt-admin`:

```bash
cd /nvme/annp36-home/work/nuno-kd
uv sync
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
cd /nvme/annp36-home/work/nuno-kd
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
cd /nvme/annp36-home/work/nuno-kd
mkdir -p logs run-outputs
test ! -f logs/qwen25_tsd_structural_20261005.pid || ! kill -0 "$(cat logs/qwen25_tsd_structural_20261005.pid)" 2>/dev/null
nohup bash -lc '
  cd /nvme/annp36-home/work/nuno-kd
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
  ssh -t vt-admin 'cd /nvme/annp36-home/work/nuno-kd && INTERVAL=10 bash scripts/watch_qwen25_tsd_structural_20261005.sh'
  ```

## 9. Terminal verification and report

After the PID exits, require exit code `0`, twelve valid final checkpoints, fourteen parseable score files (teacher, student, and 12 trained runs), complete eight-task sample logs, and CSV/JSON/LaTeX tables:

```bash
cd /nvme/annp36-home/work/nuno-kd
cat logs/qwen25_tsd_structural_20261005.exit
find results/qwen25_tsd_structural -name config.json -print
find benchmark_results/qwen25_tsd_structural/qwen25_tsd -name scores.json -print
find benchmark_results/qwen25_tsd_structural/tables -maxdepth 1 -type f -print
grep -RniE 'Traceback|CUDA out of memory|NCCL|No space left|Killed|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
  logs/qwen25_tsd_structural_20261005.chain.log results/qwen25_tsd_structural || true
```

Create or update `run-outputs/out_qwen25-tsd-structural-20261005.md` with the required Header, Settings, Main Results, Insights, Artifacts, Limitations, and Next Steps sections. If the run failed, still write the report with the exit code and preserved partial artifacts before reporting failure.

# Outputs

- `run_root`: `/nvme/annp36-home/work/nuno-kd`.
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
- Every method uses output-KD ratio `1.0`; auxiliary weights are `1.0` except CST `0.003`.
- Every run records two epochs, seed, LR `5e-6`, microbatch `1`, gradient accumulation `10`, global batch `80`, and the declared objective in `args.json` and logs.
- All twelve method/seed runs exit successfully and have a complete final checkpoint. A partial seed set is not averaged.
- Teacher, base student, and every trained checkpoint have all eight benchmark scores with non-empty raw samples. Benchmark metrics are used only for reporting.
- Every loss and reported metric is finite; no traceback, CUDA/host OOM, NCCL failure, worker death, or disk-full error. Any occurrence fails the gate and is documented rather than silently retried.
- The final mean ± sample-standard-deviation table is computed across seeds `10` and `42` with no missing dataset cell.
- Retry policy: preserve partial artifacts; diagnose first; resume only completed-safe phases. Never change data, seed, batch, objective, sampler, precision, GPU allocation, or hyperparameters without explicit user approval and an update to this same plan's change history.
- `run-outputs/out_qwen25-tsd-structural-20261005.md` exists in the project repository and reflects the terminal state before completion is declared.

# Change history

- `2026-10-05`: initial authoritative plan created from the user-approved eight-H200 policy. `ssh_host` and remote project base were taken from the user-provided OrbitKD example; upload is deliberately disabled because no destination was authorized.
