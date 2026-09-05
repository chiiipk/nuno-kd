#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"

case "${TARGET}" in
  speculative_kd)
    cd "${ROOT}/baselines/vendor/google-research/speculative_kd"
    python3.10 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install torch
    python -m pip install -r requirements.txt
    echo "SKD additionally requires its two patched Transformers generation files; see README.md."
    ;;
  legacy)
    cd "${ROOT}/baselines/vendor/contra-kd"
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install torch==2.1.2 transformers==4.42.4
    bash install.sh
    ;;
  evaluator)
    HARNESS="${ROOT}/baselines/vendor/lm-evaluation-harness"
    if [[ ! -d "${HARNESS}/.git" ]]; then
      git clone https://github.com/EleutherAI/lm-evaluation-harness.git "${HARNESS}"
    fi
    cd "${HARNESS}"
    git fetch --tags origin
    if [[ -n "${LM_EVAL_REF:-}" ]]; then
      git checkout "${LM_EVAL_REF}"
    fi
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e '.[vllm,math]'
    git rev-parse HEAD > "${ROOT}/baselines/lm_eval_commit.txt"
    ;;
  *)
    echo "Usage: bash baselines/setup_env.sh {speculative_kd|legacy|evaluator}" >&2
    exit 2
    ;;
esac
