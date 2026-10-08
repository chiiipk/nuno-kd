# Project

- `status`: approved by the user on 2026-10-08. Launch follows the source, config, GPU and smoke checks below.
- `name`: Qwen2.5 structural hidden-state distillation with stronger auxiliary weights.
- `run_id`: `qwen25_tsd_structural_w10_20261008`.
- `parent_run`: [qwen25-tsd-structural-20261005.md](qwen25-tsd-structural-20261005.md). Its report is [out_qwen25-tsd-structural-20261005.md](../run-outputs/out_qwen25-tsd-structural-20261005.md). Every setting not listed here is identical to the parent: data, models, revisions, objective, seed, LR schedule, precision, sequence lengths and benchmarks.
- `motivation`: in the parent run the weight-1 auxiliary losses were small or zero next to `ds_loss` at step 1990.
  - `gram_loss` logged `0.0000` throughout, and its trace-normalized Gram MSE is about 1e-5.
  - `cka` was `0.0313` and `direct_spectrum` `0.0543`, against a `ds_loss` of about 0.52.
- `user_request`: on 2026-10-08 the user asked for:
  - Weight 10 for gram, cka and direct_spectrum, and then specifically weight 10^5 for gram.
  - GPUs 0–5, all six used by each run in turn.
  - Then evaluation.
  - Gradient accumulation 13, so the global batch stays close to the parent's 80.
- `local_path`: `/Users/savoxism/Documents/GitHub/nuno-kd`.
- `ssh_host`: `vt-admin` (HGX47, user `vt_admin`).
- `remote_path`: `/nvme/annp36-home/work/CST`.
- `source_revision`: branch `qwen25-tsd-structural`, the commit that adds this file.
- `source_hashes`:
  - `configs/qwen25_tsd_structural_w10.yaml`: `bb913c72b4eab7a3bb0c60ee47ff5571ad7db24e3009e8a43eaffba1c882e757`.
  - `scripts/run_qwen25_tsd_structural.sh`: `7eb183412fe2853a3bf9a9da7e44ab67979a30fb9c1de8bf7868d6045c07b452`. It adds two things to `train_one`:
    - `wait_for_idle_gpus`, which starts a training run only when every listed GPU has no compute process and less than 2048 MiB in use.
    - `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so CUDA indices 0–5 are `nvidia-smi` GPUs 0–5.
  - `baselines/eval_lm_harness.py`: `ea3856dc3fbe5acbdbbdedc9fab3b968aa84666aeb22eb1571b0cf23cf81cb94` (unchanged).
  - `finetune.py`: `d40f75ec9f70f4d2bfb3c4d329e919e2cf04678129ed3cf722ad2ceb5c788e94` (unchanged).
  - `structural_ablation_losses.py`: `2abbed41cfa948c12b92b7b2e9f490ebc76d06d8dbb8815b62e736e400638aa6` (unchanged).
- `terminal_report`: `run-outputs/out_qwen25-tsd-structural-w10-20261008.md`.

# Schedule

- `start_at`: as soon as the checks in Commands §1–§2 pass. GPUs 0–5 were idle at 14:35 giờ Việt Nam on 2026-10-08.
- `timezone`: `Asia/Ho_Chi_Minh`.
- `max_runtime`: none; run to completion under the health monitor.
- `execution_policy`: one six-GPU DDP job at a time, in order `gram`, `cka`, `direct_spectrum`. Each run waits until GPUs 0–5 are completely idle, and evaluation follows training.
- `estimate`:
  - Each run takes about 2 h: 2044 steps at about 3.3 s/step, plus about 5 min of load, dev evaluation and save. The parent measured about 2.55 s/step with 10 micro-steps per step; 13 micro-steps scale this to about 3.3 s.
  - Training therefore takes about 6 h.
  - Evaluation takes about 20–30 min: 24 tasks on six GPUs, where MATH was the longest task in the parent at up to 17 min.
  - The total is about 6.5 h after launch.

# Data

Data is unchanged from the parent:
- `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` @ `450c33f2d591f68e12ecb546c16cf79821ef0006`, 79,751 examples.
- The same processed directory and `dataset_contract.json`, checked by `prepare_data` before training.
- The same 200-example dev set, which overlaps training.
- The same eight benchmarks and headline metrics.

# Model

- `teacher`: `Qwen/Qwen2.5-14B-Instruct` @ `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, fp16, frozen.
- `student`: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, ZeRO-2 bf16.
- `model_paths`: the parent's server-only runtime config already maps the Hub ids to local snapshot paths. The w10 runtime config `configs/qwen25_tsd_structural_w10.remote.yaml` is the w10 YAML with those two paths substituted, and runs offline.
- `methods`: `gram`, `cka` and `direct_spectrum`, with `sfkl` output KD (ratio 1.0, skew alpha 0.1, no student generation).
- `auxiliary_weights`: gram `100000.0` (10^5; written in full because PyYAML reads `1.0e5` as a string), cka `10.0`, direct_spectrum `10.0`. The weight ramps linearly from 0 at step 100 to full weight at step 300, as in the parent.
- `seed`: `10`.
- `epochs`: `2`.
- `optimizer`: AdamW, LR `5e-6`, weight decay `1e-2`, cosine schedule, warmup ratio `0.1`, gradient clip `1.0`.
- `GPU_policy`: exactly GPUs `0,1,2,3,4,5`, all six used by each run. GPUs 6 and 7 are not used.
- `batch`: microbatch `1`, gradient accumulation `13`, six GPUs, so

  $$B_{\mathrm{global}}=6\times1\times13=78$$

  against the parent's 80. This gives `int(79751/78)` = 1022 steps per epoch and 2044 in total.
- `checkpoint_selection`: final epoch checkpoint only.

# Commands

## 1. Source and deployment

```bash
cd /Users/savoxism/Documents/GitHub/nuno-kd
git push origin qwen25-tsd-structural
ssh vt-admin 'cd /nvme/annp36-home/work/CST && git pull --ff-only && sha256sum configs/qwen25_tsd_structural_w10.yaml scripts/run_qwen25_tsd_structural.sh'
```

## 2. Runtime config, fixed-model scores and checks (server)

```bash
cd /nvme/annp36-home/work/CST
python3 - <<'PY'
import yaml
parent = yaml.safe_load(open("configs/qwen25_tsd_structural.remote.yaml"))
text = open("configs/qwen25_tsd_structural_w10.yaml").read()
for role in ("teacher", "student"):
    hub = yaml.safe_load(text)["models"][role]
    text = text.replace(f"{role}: {hub}\n", f"{role}: {parent['models'][role]}\n", 1)
open("configs/qwen25_tsd_structural_w10.remote.yaml", "w").write(text)
PY
mkdir -p benchmark_results/qwen25_tsd_structural_w10/qwen25_tsd
ln -sfn ../../qwen25_tsd_structural/qwen25_tsd/teacher benchmark_results/qwen25_tsd_structural_w10/qwen25_tsd/teacher
ln -sfn ../../qwen25_tsd_structural/qwen25_tsd/student benchmark_results/qwen25_tsd_structural_w10/qwen25_tsd/student
bash scripts/run_qwen25_tsd_structural.sh check configs/qwen25_tsd_structural_w10.remote.yaml
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader
```

The teacher and base-student checkpoints and evaluation settings are unchanged. They are therefore not re-evaluated: their parent `scores.json` and sample logs are linked in, so `eval` skips them and `report` puts them in the table.

## 3. Durable launch (server)

`logs/qwen25_tsd_structural_w10_20261008.chain.sh` takes these steps:
1. Activate `.venv` and set `HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.
2. Write `started_at`.
3. Run `train`, then `eval`, then `report` with `configs/qwen25_tsd_structural_w10.remote.yaml`, each only if the previous step exited `0`.
4. Write `exit` and `finished_at`.

The script is launched with `setsid nohup … < /dev/null`, with its PID in `logs/qwen25_tsd_structural_w10_20261008.pid` and its log in `logs/qwen25_tsd_structural_w10_20261008.chain.log`.

## 4. Monitoring

- Watcher: `ssh -t vt-admin 'cd /nvme/annp36-home/work/CST && INTERVAL=10 bash scripts/watch_qwen25_tsd_structural_w10_20261008.sh'`.
- A read-only health check runs every 20 minutes.

## 5. Terminal verification and report

These checks mirror the parent's §9:
- Exit code `0`.
- Three final checkpoints (`2044/config.json`).
- Three new `scores.json` files with all eight tasks.
- The tables.
- The error grep.
- A recomputation of every score from the raw samples.

Then write `run-outputs/out_qwen25-tsd-structural-w10-20261008.md`, comparing each method against both its weight-1 parent run and the base student.

# Outputs

- `train_root`: `results/qwen25_tsd_structural_w10/<method>/seed10/`.
- `eval_root`: `benchmark_results/qwen25_tsd_structural_w10/qwen25_tsd/{teacher→parent,student→parent,<method>}/seed10/`.
- `tables`: `benchmark_results/qwen25_tsd_structural_w10/tables/qwen25_tsd_full8_mean_std.{csv,json,tex}`.
- `chain_log`, `pid_file`, `exit_file`, `started_at`, `finished_at`: `logs/qwen25_tsd_structural_w10_20261008.{chain.log,pid,exit,started_at,finished_at}`.
- `runtime_config`: `configs/qwen25_tsd_structural_w10.remote.yaml` (server only).
- The parent run's artifacts are read-only for this run and are never modified.

# Upload

- `destination`: none authorized.

# Acceptance

- Each of the three runs uses exactly GPUs 0–5 as one DDP job, after the idle check, never alongside another job.
- `args.json` of every run records:
  - epochs 2, LR `5e-6`, microbatch 1, gradient accumulation 13, seed 10;
  - `type: sfkl`, `student_gen: false`;
  - `nnm_ratio` 1e5 for gram, 10 for cka and 10 for direct_spectrum.
- All losses are finite, and there is no traceback, OOM, NCCL failure, worker death or disk-full error.
- All three runs have a complete final checkpoint.
- All eight benchmark scores exist for each, with non-empty samples.
- With one seed, the table's ± 0.00 is not a variance and is reported as single-seed.
- Retry policy: preserve artifacts and diagnose first. Never change data, seed, batch, objective, weights or GPUs without the user's approval and an entry in this plan's change history.

# Change history

- `2026-10-08`: plan created from the user's request. Gram weight 10^5, cka 10, direct_spectrum 10, GPUs 0–5 and accumulation 13 (global batch 78) were all chosen by the user.
