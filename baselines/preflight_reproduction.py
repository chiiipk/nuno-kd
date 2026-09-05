#!/usr/bin/env python3
"""Fail-fast checks for the two-table 6xH200 reproduction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

from dataset_contract import check as check_contract


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "gsm8k", "gsm_plus", "minerva_math", "mbpp", "sciq",
    "mmlu_stem", "mmlu_pro_math", "bbh_cot_fewshot",
)
DEFAULT_METHODS = (
    "seqkd", "supervised_kd", "distillm", "speculative_kd", "minillm",
    "gkd", "csd", "amid", "cst",
)
FORBIDDEN = {"nuno", "tsd_kd", "tsd-kd", "rpt"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PRECHECK FAILED: {message}")


def check_jsonl(path: Path) -> None:
    require(path.is_file(), f"missing JSONL: {path}")
    count = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require({"prompt", "generated_text"} <= row.keys(),
                    f"{path}:{line_number} lacks prompt/generated_text")
            count += 1
    require(count > 0, f"empty JSONL: {path}")
    print(f"[ok] {path}: {count:,} non-empty rows")


def static_checks(methods: tuple[str, ...]) -> None:
    require(len(TASKS) == 8 and len(set(TASKS)) == 8, "task matrix must contain 8 unique tasks")
    require(not (set(methods) & FORBIDDEN), f"excluded method requested: {set(methods) & FORBIDDEN}")
    require(set(methods) <= set(DEFAULT_METHODS), f"unknown methods: {set(methods) - set(DEFAULT_METHODS)}")
    for relative in (
        "baselines/reproduce_2tables_6xh200.sh",
        "baselines/train_common_6xh200.sh",
        "baselines/eval_lm_harness.py",
        "baselines/report_mean_std.py",
    ):
        require((ROOT / relative).is_file(), f"missing orchestration file: {relative}")
    print(f"[ok] static matrix: methods={','.join(methods)} tasks={','.join(TASKS)}")


def train_checks(methods: tuple[str, ...]) -> None:
    require(shutil.which("torchrun") is not None, "torchrun is not on PATH")
    gpu_csv = os.environ.get("RUN_GPUS", "0,1,2,3,4,5")
    gpus = [item.strip() for item in gpu_csv.split(",") if item.strip()]
    require(len(gpus) == 6 and len(set(gpus)) == 6, f"RUN_GPUS must name 6 unique GPUs, got {gpus}")
    require(shutil.which("nvidia-smi") is not None, "nvidia-smi is not on PATH")
    detected = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    ).splitlines()
    require(all(gpu in detected for gpu in gpus), f"requested GPUs {gpus}, detected indices {detected}")

    qwen_data = Path(os.environ.get(
        "QWEN_DATA_DIR", str(ROOT / "processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct")
    ))
    gemma_data = Path(os.environ.get(
        "GEMMA_DATA_DIR", str(ROOT / "processed_data/ultraInteract/google/gemma-2-9b-it")
    ))
    require(qwen_data.is_dir(), f"missing Qwen processed data: {qwen_data}")
    require(gemma_data.is_dir(), f"missing Gemma processed data: {gemma_data}")

    pair_inputs = {
        "qwen": {
            "data": qwen_data,
            "manifest": Path(os.environ.get("QWEN_DATASET_MANIFEST", qwen_data / "dataset_contract.json")),
            "skd": os.environ.get("SKD_QWEN_TRAIN_JSONL"),
            "mini": os.environ.get("MINILLM_QWEN_PROMPT_DATA"),
        },
        "gemma": {
            "data": gemma_data,
            "manifest": Path(os.environ.get("GEMMA_DATASET_MANIFEST", gemma_data / "dataset_contract.json")),
            "skd": os.environ.get("SKD_GEMMA_TRAIN_JSONL"),
            "mini": os.environ.get("MINILLM_GEMMA_PROMPT_DATA"),
        },
    }
    for pair, inputs in pair_inputs.items():
        require(inputs["manifest"].is_file(), f"missing {pair} dataset contract: {inputs['manifest']}")
        check_contract(inputs["manifest"], inputs["data"], "train")

    if "speculative_kd" in methods:
        for pair, inputs in pair_inputs.items():
            require(bool(inputs["skd"]), f"SKD_{pair.upper()}_TRAIN_JSONL is unset")
            check_jsonl(Path(inputs["skd"]))
            check_contract(inputs["manifest"], Path(inputs["skd"]), "train")
    if "minillm" in methods:
        for pair, inputs in pair_inputs.items():
            require(bool(inputs["mini"]), f"MINILLM_{pair.upper()}_PROMPT_DATA is unset")
            require(Path(inputs["mini"]).exists(), f"MiniLLM {pair} prompt data does not exist")
            check_contract(inputs["manifest"], Path(inputs["mini"]), "train")
        lm_data = os.environ.get("MINILLM_LM_DATA")
        require(bool(lm_data) and Path(lm_data).exists(), "MINILLM_LM_DATA is missing or does not exist")
    print(f"[ok] six-GPU training preflight: GPUs={gpu_csv}")


def eval_checks() -> None:
    require(importlib.util.find_spec("lm_eval") is not None, "lm_eval is not importable")
    require(importlib.util.find_spec("vllm") is not None, "vllm is not importable")
    from lm_eval.tasks import TaskManager

    available = set(TaskManager().all_tasks)
    missing = [task for task in TASKS if task not in available]
    require(not missing, f"lm-eval commit lacks tasks: {missing}")
    commit_file = ROOT / "baselines/lm_eval_commit.txt"
    require(commit_file.is_file() and commit_file.read_text().strip(), "missing lm_eval_commit.txt")
    print(f"[ok] evaluator tasks and commit {commit_file.read_text().strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("static", "train", "eval", "all"), default="all")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    static_checks(methods)
    if args.stage in {"train", "all"}:
        train_checks(methods)
    if args.stage in {"eval", "all"}:
        eval_checks()


if __name__ == "__main__":
    main()
