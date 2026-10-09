#!/usr/bin/env bash
# Run cst_hendrycks_math_20261010 (plan run-inputs/cst-hendrycks-math-20261010.md):
# evaluate the 12 gram/cka/direct_spectrum checkpoints on lm-eval hendrycks_math.
# All 12 evaluations start at once and share the GPU lease pool 0-7 of
# eval_lm_harness.py: each takes one idle H200, the remaining four wait for a
# free one. Then the samples are rescored with three graders.
cd /nvme/annp36-home/work/CST
ROOT="$(pwd)"
export HF_HOME=/nvme/annp36-home/.cache/huggingface
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset HF_HUB_ENABLE_HF_TRANSFER
export LM_EVAL_COMMIT=6d642546f4688648fced259eb3302efd36ece5af
PY=baselines/vendor/lm-evaluation-harness/.venv/bin/python
RUN=logs/cst_hendrycks_math_20261010
OUT=benchmark_results/hendrycks_math_20261010
GPUS=0,1,2,3,4,5,6,7

CHECKPOINTS=(
  "w1 qwen25_tsd_structural 996 1992"
  "w10 qwen25_tsd_structural_w10 1022 2044"
)
METHODS=(gram cka direct_spectrum)

date --iso-8601=seconds > "${RUN}.started_at"
mkdir -p "${OUT}"
pids=() names=() outs=()
for spec in "${CHECKPOINTS[@]}"; do
  read -r tag train_run step_a step_b <<< "${spec}"
  for method in "${METHODS[@]}"; do
    for step in "${step_a}" "${step_b}"; do
      ckpt="${ROOT}/results/${train_run}/${method}/seed10/${step}"
      out="${ROOT}/${OUT}/${tag}/${method}/seed10/${step}"
      [[ -f "${ckpt}/config.json" && -f "${ckpt}/pytorch_model.bin" ]] || { echo "missing checkpoint ${ckpt}"; echo 1 > "${RUN}.exit"; exit 1; }
      mkdir -p "${out}"
      echo "=== start ${tag}/${method}/${step} $(date --iso-8601=seconds)"
      "${PY}" baselines/eval_lm_harness.py --checkpoint "${ckpt}" --output "${out}" --gpus "${GPUS}" \
        --tasks hendrycks_math --max-new-tokens 5120 --gpu-memory-utilization 0.85 --reuse-complete \
        > "${out}/driver.log" 2>&1 &
      pids+=("$!") names+=("${tag}/${method}/${step}") outs+=("${out}")
    done
  done
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "=== done ${names[$i]} $(date --iso-8601=seconds): $(tail -1 "${outs[$i]}/driver.log")"
  else
    rc_i=$?
    echo "=== FAILED ${names[$i]} rc=${rc_i} $(date --iso-8601=seconds)"
    failed=1
  fi
done

rc=${failed}
if [[ ${rc} == 0 ]]; then
  echo "=== rescore start $(date --iso-8601=seconds)"
  "${PY}" baselines/score_hendrycks_math.py --root "${OUT}" || rc=$?
fi
printf "%s\n" "${rc}" > "${RUN}.exit"
date --iso-8601=seconds > "${RUN}.finished_at"
echo "=== exit ${rc} $(date --iso-8601=seconds)"
exit "${rc}"
