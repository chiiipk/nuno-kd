#!/usr/bin/env bash
# Install lm-evaluation-harness (with vLLM and math extras) in its own venv
# under evaluation/vendor/. Pin a revision with LM_EVAL_REF=<commit or tag>.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ROOT}/evaluation/vendor/lm-evaluation-harness"

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
git rev-parse HEAD > "${ROOT}/evaluation/lm_eval_commit.txt"
