---
name: ssh-experiment-runner
description: Migrate projects to SSH GPU servers, run durable sequential H200 training/evaluation jobs, monitor health and progress, and upload validated artifacts. Use for remote ML experiments; not for local-only runs.
---

# SSH Experiment Runner

## 1. Run input: `run-inputs/<plan>.md`

One plan file per run, in the project repo. `<plan>` is its basename; the runtime `run_id` may differ. Sections:

- `# Project`: name, run_id, local_path, ssh_host, remote_path, git_remote, branch, source_revision (commit SHA).
- `# Schedule`: start_at, timezone, max_runtime.
- `# Data`: input, train, validation, benchmark, and manifest paths.
- `# Model`: repo_id, revision, model/cache path, task, method, seed, batch sizes, epochs, GPU policy.
- `# Commands`: exact preflight, train, eval, and verify commands, in order.
- `# Outputs`: unique output, checkpoint, log, PID, and exit-code paths.
- `# Upload`: Google Drive remote and folder, artifacts to upload, local download path.
- `# Acceptance`: expected artifacts and metrics, failure conditions, retry policy.

Ask only for missing fields that change the run. Never invent paths, commands, revisions, or destinations.

**Plan changes.** If a run is stopped to change the plan, update the same plan file before resuming. Add a dated history entry covering: why it stopped; what changed (old → new); what is reused; what remains. Keep old logs and manifests. Commit the plan, then pull it on the server. Do not create a replacement plan.

## 2. Code → server: git only

- Never send code to the server with rsync, scp, or edits on the server. Hotfixes follow the same path: edit locally, commit, push, pull.
- Commit on the branch you are currently working on and push it. Never force-push.
- Server: `git fetch origin && git switch <branch> && git pull --ff-only`.
  - If the pull is blocked by identical files that were copied earlier, check hashes, then run `git reset --mixed origin/<branch>`. This moves HEAD and the index only.
  - If a server change matches no commit, stop and ask.
- Never run these on the server: `reset --hard`, `checkout -- .`, `restore .`, `clean`, `stash`.
- Keep artifacts out of git by listing them in the server's `.git/info/exclude`: `.venv*/`, `logs/`, `outputs/`, caches, data, checkpoints. Never commit secrets, data, checkpoints, or logs.
- Before launching:
  - `git status --porcelain --untracked-files=no` is empty;
  - `git rev-parse HEAD` equals source_revision. Record that SHA in the plan, the launch log, and the report.
  - Also check the venv, model revision, free disk, and launch scripts.

## 3. Data: local ↔ server via rsync

- `rsync -avh --partial --progress -e "ssh -o ControlPath=none -o ServerAliveInterval=30" SRC DEST`
- No `--delete` or `--remove-source-files` unless the user asks.
- When sending from the Mac, add `--exclude '._*' --exclude .DS_Store`. Afterwards, delete any `._*` files inside that destination only.
- Copy only into git-ignored paths (e.g. local `data/`).
- Verify with sha256 on both ends, or with a `SHA256SUMS` file.

## 4. Hugging Face

- Use the project venv only. Set `HF_HUB_DISABLE_XET=1`; do not set `HF_HUB_ENABLE_HF_TRANSFER`.
- Run `hf auth whoami`. If not logged in, log in from `HF_TOKEN` or the token file through `huggingface_hub.login`. If neither exists, ask the user.
- Never print or store a token.
- Prefetch each pinned revision once, then set `HF_HUB_OFFLINE=1` for every run.

## 5. Training on H200

- Before launch, check GPU owners, processes, utilization, and VRAM.
- Each job uses all GPUs in one DDP job:
  - never split the GPUs across runs;
  - never silently use fewer GPUs;
  - never take GPUs or processes from another user — wait or ask.
- Record and verify the global batch: B_global = N_GPU × B_device × G_accum.
- Launcher requirements: `nohup`, unique paths, PID file, exit-code file, structured log, checkpoint validation.
- Run one phase at a time. A phase starts only after the previous one exits 0 and its artifacts are valid.
- A retry must not change data, seed, batch, objective, sampler, or hyperparameters without approval.

## 6. Monitoring

- Every 20 minutes, run a read-only health check of:
  - PID and exit file;
  - step, loss, and grad (finite);
  - log and checkpoint mtimes;
  - disk;
  - each GPU's owner, utilization, VRAM, temperature, and power.
- Scan for traceback, OOM, NCCL errors, NaN/Inf, dead workers, and a full disk. One idle sample is not a stall.
- Write `scripts/watch_<run_id>.sh`. It refreshes on `INTERVAL` and shows progress bar, phase, percent, step/total, epoch, loss, LR, elapsed, ETA, latest metric, and GPU health.
- Ctrl-C must stop only the watcher.
- Give the user exactly this command: `ssh -t SSH_HOST 'cd REMOTE_PATH && INTERVAL=10 bash scripts/watch_<run_id>.sh'`

## 7. Upload to Google Drive (rclone + your own OAuth client)

- Use an rclone `drive` remote with the user's own OAuth client (`client_id`/`client_secret`), not rclone's default client.
- The user does the authorization: `rclone authorize "drive" ...` on the Mac, then pastes the token into `rclone config` on the server.
- Never ask for, print, or store the secret or token. Do not run `config show` or `config dump`; only `rclone config redacted`.
- Before uploading, run `rclone listremotes --long` and `rclone lsd REMOTE:`. If the remote is missing, unauthenticated, or uses the default client, stop and ask. Never change the destination.
- Upload only after: exit code 0, complete checkpoints, parseable metrics, matching hashes.
- `rclone copy OUTPUT REMOTE:DEST --progress`. Never use `sync`, `move`, `delete`, `purge`, or `--delete*`.
- Verify with `rclone check OUTPUT REMOTE:DEST --one-way`. Report the path, destination, size, hashes, and status.
- Keep failed or partial artifacts unless the user asks to delete them.

## 8. Run output: `run-outputs/out_<plan>.md` (required)

- After every success or failure, create or update this report before telling the user the run is done.
  - The report covers the whole plan, including resumed phases.
  - Before updating an existing report, save a dated snapshot of it.
  - Ask before writing a report for an unfinished run.
- Use only real numbers from this run. Never invent values. Disclose every deviation from the plan.
- Sections, in this order:
  1. **Header**: plan, run_id(s), server, run root, link to the plan, status/exit code/gates, timestamps, elapsed time, commit SHA.
  2. **Settings**: model and revision, configuration, method, optimizer, LR schedule, steps, global batch, validation protocol. If there is a paper, add a comparison table and explain each deviation.
  3. **Main Results**: tables per metric family with raw counts and baselines. Label secondary graders as reference only.
  4. **Insights**: paragraphs that each start with a **bold hook sentence** and back every claim with numbers and paths.
  5. **Artifacts**: files, hashes, sizes, GPU release state, upload status.
  6. **Limitations**: e.g. single seed, checkpoint choice, missing baselines, known bugs.
  7. **Next Steps**: marked as not run.

## 9. Completion

Tell the user in one sentence: configuration, GPU count, global batch, elapsed time, final checkpoint, metrics, retries, commit SHA, output path, Drive destination, GPU release state, and the report path.
