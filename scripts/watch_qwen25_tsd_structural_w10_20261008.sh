#!/usr/bin/env bash
# Read-only watcher for run qwen25_tsd_structural_w10_20261008.
# Usage: INTERVAL=10 bash scripts/watch_qwen25_tsd_structural_w10_20261008.sh
# Ctrl-C stops only this watcher; it never signals the training chain.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=qwen25_tsd_structural_w10_20261008
INTERVAL="${INTERVAL:-30}"
METHODS=(gram cka direct_spectrum)
SEEDS=(10)
TOTAL_RUNS=$(( ${#METHODS[@]} * ${#SEEDS[@]} ))
TOTAL_EVALS=$(( TOTAL_RUNS + 2 ))
TRAIN_ROOT="${ROOT}/results/qwen25_tsd_structural_w10"
EVAL_ROOT="${ROOT}/benchmark_results/qwen25_tsd_structural_w10/qwen25_tsd"
TABLE_ROOT="${ROOT}/benchmark_results/qwen25_tsd_structural_w10/tables"
LOGS="${ROOT}/logs"

trap 'echo; echo "watcher stopped (run untouched)"; exit 0' INT TERM

vn_time() { TZ=Asia/Ho_Chi_Minh date -d "$1" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$1"; }

hms() {
  local s=${1%.*}
  [[ -z "${s}" || "${s}" -lt 0 ]] && { echo "--"; return; }
  printf '%dh %02dm' $(( s / 3600 )) $(( s % 3600 / 60 ))
}

bar() {
  local pct=$1 width=40 filled
  filled=$(( pct * width / 100 ))
  printf '['; printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' $(( width - filled )) '' | tr ' ' '.'; printf '] %3d%%' "${pct}"
}

render() {
  local now pid state started elapsed=0
  now=$(date +%s)
  echo "=== ${RUN_ID} | $(TZ=Asia/Ho_Chi_Minh date '+%Y-%m-%d %H:%M:%S') giờ Việt Nam | refresh ${INTERVAL}s ==="

  pid=$(cat "${LOGS}/${RUN_ID}.pid" 2>/dev/null || true)
  if [[ -f "${LOGS}/${RUN_ID}.exit" ]]; then
    state="EXITED rc=$(cat "${LOGS}/${RUN_ID}.exit") at $(vn_time "$(cat "${LOGS}/${RUN_ID}.finished_at" 2>/dev/null)")"
  elif [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    state="RUNNING pid=${pid}"
  elif [[ -n "${pid}" ]]; then
    state="DEAD pid=${pid} (no exit file)"
  else
    state="NOT LAUNCHED"
  fi
  if [[ -f "${LOGS}/${RUN_ID}.started_at" ]]; then
    started=$(cat "${LOGS}/${RUN_ID}.started_at")
    elapsed=$(( now - $(date -d "${started}" +%s) ))
    echo "state: ${state} | started $(vn_time "${started}") | elapsed $(hms "${elapsed}")"
  else
    echo "state: ${state}"
  fi

  # Training progress.
  local done_runs=0 cur_log="" cur_name="" m s
  for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
      if [[ -f "${TRAIN_ROOT}/${m}/seed${s}/TRAINING_COMPLETE" ]]; then
        done_runs=$(( done_runs + 1 ))
      elif [[ -z "${cur_log}" && -f "${TRAIN_ROOT}/${m}/seed${s}/train.log" ]]; then
        cur_log="${TRAIN_ROOT}/${m}/seed${s}/train.log"; cur_name="${m}/seed${s}"
      fi
    done
  done

  local n_scores phase
  n_scores=$(find -L "${EVAL_ROOT}" -name scores.json 2>/dev/null | wc -l)
  if ls "${TABLE_ROOT}"/*.csv >/dev/null 2>&1; then phase=report
  elif (( done_runs == TOTAL_RUNS )); then phase=eval
  else phase=train; fi
  echo "phase: ${phase} | train runs ${done_runs}/${TOTAL_RUNS} | eval scores ${n_scores}/${TOTAL_EVALS}"

  local line it tot step_time loss lr epoch run_pct=0 overall_pct
  if [[ -n "${cur_log}" ]]; then
    line=$(grep -a 'global iter:' "${cur_log}" | tail -1)
    if [[ -n "${line}" ]]; then
      it=$(sed -E 's/.*global iter: *([0-9]+)\/ *([0-9]+).*/\1/' <<< "${line}")
      tot=$(sed -E 's/.*global iter: *([0-9]+)\/ *([0-9]+).*/\2/' <<< "${line}")
      epoch=$(sed -E 's/.*epoch +([0-9]+).*/\1/' <<< "${line}")
      loss=$(sed -E 's/.*\| loss: ([^ ]+).*/\1/' <<< "${line}")
      lr=$(sed -E 's/.*\| lr: ([^ ]+).*/\1/' <<< "${line}")
      step_time=$(sed -E 's/.*step time: *([0-9.]+).*/\1/' <<< "${line}")
      run_pct=$(( it * 100 / tot ))
      local run_eta train_eta
      run_eta=$(awk -v r=$(( tot - it )) -v t="${step_time}" 'BEGIN{printf "%d", r*t}')
      train_eta=$(awk -v r=$(( tot - it + (TOTAL_RUNS - done_runs - 1) * tot )) -v t="${step_time}" 'BEGIN{printf "%d", r*t}')
      echo "current: ${cur_name} | epoch ${epoch} | step ${it}/${tot} | loss ${loss} | lr ${lr} | step ${step_time}s"
      echo "run    $(bar "${run_pct}")  ETA run $(hms "${run_eta}") | ETA all training ≈ $(hms "${train_eta}") (excl. load/eval)"
    else
      echo "current: ${cur_name} | loading / pre-training eval ($(tail -c 200 "${cur_log}" | tr '\r\n' '  ' | cut -c1-120))"
    fi
    echo "log age: $(( now - $(stat -c %Y "${cur_log}") ))s  (${cur_log#${ROOT}/})"
  fi
  overall_pct=$(( (done_runs * 100 + run_pct) * 70 / (TOTAL_RUNS * 100) + n_scores * 30 / TOTAL_EVALS ))
  echo "overall $(bar "${overall_pct}")  (train weighted 70%, eval 30%)"

  # Error scan on the chain log and current train log.
  local errs
  errs=$(grep -aciE 'Traceback|CUDA out of memory|NCCL error|No space left|Killed' \
    "${LOGS}/${RUN_ID}.chain.log" ${cur_log:+"${cur_log}"} 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
  echo "error lines: ${errs}"

  echo "--- GPUs ---"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader 2>/dev/null | awk -F', ' '{printf "GPU%s %-12s util %5s  mem %9s / %9s  %3s°C  %s\n",$1,$2,$3,$4,$5,$6,$7}'
  local owners
  owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | \
    xargs -r -I{} sh -c 'echo "{}:$(ps -o user= -p {} 2>/dev/null):$(readlink /proc/{}/cwd 2>/dev/null)"' | \
    awk -F: '{print $3}' | sort | uniq -c | tr '\n' ';')
  echo "GPU processes by cwd: ${owners:-none}"
  echo "disk: $(df -h "${ROOT}" | awk 'NR==2{print $4" free of "$2}')"
}

while true; do
  out=$(render 2>&1)
  clear 2>/dev/null || true
  printf '%s\n' "${out}"
  sleep "${INTERVAL}"
done
