#!/usr/bin/env python3
"""Evaluate one checkpoint on the eight NuNo tasks using lm-eval-harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TASKS = {
    "gsm8k": {"fewshot": None, "unsafe": False},
    "gsm_plus": {"fewshot": None, "unsafe": False},
    "minerva_math": {"fewshot": 4, "unsafe": False},
    "mbpp": {"fewshot": 3, "unsafe": True},
    "sciq": {"fewshot": None, "unsafe": False},
    "mmlu_stem": {"fewshot": 5, "unsafe": False},
    "mmlu_pro_math": {"fewshot": None, "unsafe": False},
    "bbh_cot_fewshot": {"fewshot": None, "unsafe": False},
}

METRIC_PRIORITY = {
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"),
    "gsm_plus": ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"),
    "minerva_math": ("exact_match,none", "exact_match", "math_verify,none", "math_verify"),
    "mbpp": ("pass_at_1,none", "pass_at_1", "exact_match,none", "exact_match"),
    "sciq": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "mmlu_stem": ("acc,none", "acc"),
    "mmlu_pro_math": ("acc,none", "acc"),
    "bbh_cot_fewshot": ("exact_match,flexible-extract", "exact_match,none", "exact_match"),
}


def newest_result(directory: Path) -> Path:
    files = list(directory.rglob("results*.json"))
    if not files:
        raise FileNotFoundError(f"lm-eval produced no results JSON under {directory}")
    return max(files, key=lambda p: p.stat().st_mtime_ns)


def find_metric(payload: dict, task: str) -> tuple[str, float, float | None]:
    results = payload.get("results", {})
    groups = payload.get("groups", {})
    candidates = []
    if task in groups and isinstance(groups[task], dict):
        candidates.append(groups[task])
    if task in results:
        candidates.append(results[task])
    candidates.extend(value for key, value in results.items() if key.startswith(task) and isinstance(value, dict))
    for metric in METRIC_PRIORITY[task]:
        values = [float(row[metric]) for row in candidates if metric in row and isinstance(row[metric], (int, float))]
        if values:
            # Prefer the aggregate emitted by lm-eval. A subtask macro is only
            # a compatibility fallback for harness versions without `groups`.
            value = values[0] if task in groups or task in results else sum(values) / len(values)
            if "," in metric:
                base, filter_name = metric.split(",", 1)
                stderr_key = f"{base}_stderr,{filter_name}"
            else:
                stderr_key = f"{metric}_stderr"
            stderr = None
            aggregate = groups.get(task, results.get(task, {}))
            if isinstance(aggregate, dict):
                raw_stderr = aggregate.get(stderr_key)
                if isinstance(raw_stderr, (int, float)):
                    stderr = float(raw_stderr)
            return metric, value, stderr
    raise KeyError(f"No supported metric found for {task}; available rows: {list(results)}")


def run_task(args: argparse.Namespace, task: str, gpu: str) -> tuple[str, str, float, float | None, list[str], int]:
    task_dir = args.output / task
    task_dir.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={args.checkpoint},tensor_parallel_size=1,dtype=bfloat16,"
        f"gpu_memory_utilization={args.gpu_memory_utilization},trust_remote_code=True"
    )
    cmd = [
        sys.executable, "-m", "lm_eval", "--model", "vllm",
        "--model_args", model_args, "--tasks", task, "--batch_size", "auto",
        "--log_samples", "--output_path", str(task_dir),
        "--gen_kwargs", f"max_new_tokens={args.max_new_tokens},temperature=0.0",
    ]
    spec = TASKS[task]
    if task != "mbpp":
        cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
    if spec["fewshot"] is not None:
        cmd += ["--num_fewshot", str(spec["fewshot"])]
    if spec["unsafe"]:
        cmd.append("--confirm_run_unsafe_code")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "HF_ALLOW_CODE_EVAL": "1",
        "PYTHONUNBUFFERED": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    with (task_dir / "eval.log").open("w") as log:
        log.write("COMMAND: " + json.dumps(cmd, ensure_ascii=False) + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n\n")
        log.flush()
        completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    result_file = newest_result(task_dir)
    payload = json.loads(result_file.read_text())
    metric, value, stderr = find_metric(payload, task)
    sample_files = sorted(task_dir.rglob("samples*.jsonl"))
    if not sample_files:
        raise FileNotFoundError(f"--log_samples produced no samples JSONL under {task_dir}")
    sample_count = 0
    for path in sample_files:
        with path.open() as handle:
            sample_count += sum(1 for line in handle if line.strip())
    if sample_count == 0:
        raise RuntimeError(f"All sample logs are empty under {task_dir}")
    return (
        task,
        metric,
        100.0 * value,
        None if stderr is None else 100.0 * stderr,
        [str(path.relative_to(args.output)) for path in sample_files],
        sample_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-new-tokens", type=int, default=5120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")

    scores: dict[str, dict[str, float | str]] = {}
    task_names = list(TASKS)
    # Run in waves so no two vLLM processes are ever assigned the same GPU.
    for start in range(0, len(task_names), len(gpus)):
        wave = task_names[start : start + len(gpus)]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            pending = {
                pool.submit(run_task, args, task, gpus[index]): task
                for index, task in enumerate(wave)
            }
            for future in as_completed(pending):
                task, metric, value, stderr, sample_files, sample_count = future.result()
                scores[task] = {
                    "metric": metric,
                    "value": value,
                    "lm_eval_stderr": stderr,
                    "sample_files": sample_files,
                    "sample_count": sample_count,
                }
                print(f"{task}: {value:.2f} ({metric}), samples={sample_count}", flush=True)

    ordered_values = [float(scores[task]["value"]) for task in TASKS]
    output = {
        "checkpoint": str(args.checkpoint),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lm_eval_commit": os.environ.get("LM_EVAL_COMMIT", "unknown"),
        "gpus": gpus,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "scores": {task: scores[task] for task in TASKS},
        "average": sum(ordered_values) / len(ordered_values),
    }
    (args.output / "scores.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
