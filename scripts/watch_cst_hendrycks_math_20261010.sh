#!/usr/bin/env bash
# Read-only watcher for cst_hendrycks_math_20261010. Ctrl-C stops only this watcher.
# Usage: INTERVAL=10 bash scripts/watch_cst_hendrycks_math_20261010.sh
cd /nvme/annp36-home/work/CST || exit 1
INTERVAL="${INTERVAL:-10}"
RUN=logs/cst_hendrycks_math_20261010
OUT=benchmark_results/hendrycks_math_20261010
trap 'echo; exit 0' INT TERM

bar() {  # bar <done> <total>
  local done=$1 total=$2 width=30 fill
  (( total > 0 )) || total=1
  fill=$(( done * width / total ))
  printf '[%s%s] %3d%%' "$(printf '#%.0s' $(seq 1 $fill) 2>/dev/null)" "$(printf '.%.0s' $(seq 1 $(( width - fill ))) 2>/dev/null)" $(( done * 100 / total ))
}

while true; do
  clear
  now=$(date +%s)
  start=$(date -d "$(cat ${RUN}.started_at 2>/dev/null)" +%s 2>/dev/null || echo "$now")
  echo "cst_hendrycks_math_20261010 | $(TZ=Asia/Ho_Chi_Minh date '+%Y-%m-%d %H:%M:%S') giờ Việt Nam | elapsed $(( (now - start) / 60 )) min"
  if [[ -f ${RUN}.exit ]]; then
    echo "FINISHED: exit $(cat ${RUN}.exit) at $(TZ=Asia/Ho_Chi_Minh date -d "$(cat ${RUN}.finished_at)" '+%H:%M:%S')"
  elif [[ -f ${RUN}.pid ]] && kill -0 "$(cat ${RUN}.pid)" 2>/dev/null; then
    echo "RUNNING: chain PID $(cat ${RUN}.pid)"
  else
    echo "NOT RUNNING and no exit file"
  fi
  echo
  finished=0 total=0
  for d in ${OUT}/w{1,10}/*/seed10/*/; do
    [[ -d "$d" ]] || continue
    total=$((total + 1))
    name=${d#${OUT}/}; name=${name%/}
    log="$d/hendrycks_math/eval.log"
    if [[ -f "$d/scores.json" ]]; then
      finished=$((finished + 1))
      val=$(python3 -c "import json;print('%.2f'%json.load(open('$d/scores.json'))['scores']['hendrycks_math']['value'])" 2>/dev/null)
      printf '  %-34s DONE     lm-eval exact_match %s\n' "$name" "$val"
    elif [[ -f "$log" ]]; then
      gpu=$(sed -n 's/^CUDA_VISIBLE_DEVICES=//p' "$log" | head -1)
      prog=$(tr '\r' '\n' < "$log" | grep -oE 'Processed prompts: +[0-9]+%[^|]*\| *[0-9]+/[0-9]+' | tail -1 | grep -oE '[0-9]+/[0-9]+$')
      if [[ -n "$prog" ]]; then
        printf '  %-34s GPU %-3s  %s %s\n' "$name" "$gpu" "$(bar ${prog%/*} ${prog#*/})" "$prog"
      else
        printf '  %-34s GPU %-3s  loading\n' "$name" "$gpu"
      fi
    else
      printf '  %-34s waiting for a free GPU\n' "$name"
    fi
  done
  echo
  echo "Checkpoints: $(bar $finished ${total:-1}) $finished/$total"
  echo
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader \
    | awk -F', ' '{printf "  GPU %s  util %5s  mem %10s / %s  %s C  %s\n", $1, $2, $3, $4, $5, $6}'
  errs=$(grep -lE "Traceback|OutOfMemoryError|CUDA out of memory|NCCL error|No space left" ${OUT}/w*/*/seed10/*/driver.log ${OUT}/w*/*/seed10/*/hendrycks_math/eval.log 2>/dev/null | wc -l)
  echo; echo "Logs with errors: ${errs}   Disk /nvme: $(df -h /nvme | awk 'NR==2{print $4" free"}')"
  sleep "$INTERVAL"
done
