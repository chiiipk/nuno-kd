#!/usr/bin/env python3
"""Evaluate one checkpoint on the eight-task benchmark suite with lm-eval-harness.

Run with the evaluator venv (see evaluation/setup.sh). Every task leases one idle
GPU from --gpus and writes lm-eval's results and samples under --output/<task>/;
the headline metric of each task goes to --output/scores.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
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

# Run only when named in --tasks; never part of the eight-task suite or its average.
EXTRA_TASKS = {
    "hendrycks_math": {"fewshot": None, "unsafe": False},
}
ALL_TASKS = {**TASKS, **EXTRA_TASKS}

# The BBH template stops at "\n\n" and "Q", which cuts chat-formatted answers
# after their first paragraph, before "the answer is ...". Stop only at the
# Qwen chat end-of-turn token (lm-eval also appends the tokenizer EOS).
STOP_OVERRIDES = {"bbh_cot_fewshot": "<|im_end|>"}

METRIC_PRIORITY = {
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"),
    "gsm_plus": ("exact_match,flexible-extract", "exact_match,strict-match", "exact_match"),
    "minerva_math": ("math_verify,none", "math_verify", "exact_match,none", "exact_match"),
    "mbpp": ("pass_at_1,none", "pass_at_1", "exact_match,none", "exact_match"),
    "sciq": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "mmlu_stem": ("acc,none", "acc"),
    "mmlu_pro_math": (
        "exact_match,custom-extract", "exact_match,flexible-extract",
        "acc,none", "acc", "exact_match,none", "exact_match",
    ),
    "bbh_cot_fewshot": ("exact_match,flexible-extract", "exact_match,get-answer", "exact_match,none", "exact_match"),
    "hendrycks_math": ("exact_match,none", "exact_match"),
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


def gpu_idle(gpu: str, free_mib: int) -> bool:
    """True if no process runs on the GPU and its memory use is below free_mib."""
    def query(*fields: str) -> str:
        return subprocess.run(
            ["nvidia-smi", *fields, "--format=csv,noheader,nounits", "-i", gpu],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    return not query("--query-compute-apps=pid") and int(query("--query-gpu=memory.used")) < free_mib


def lease_gpu(args: argparse.Namespace, task: str):
    """Block until this process holds an exclusive lease on an idle GPU.

    Leases are flock()s in --gpu-lock-dir, so concurrent evaluations of several
    checkpoints share one pool and never put two vLLM engines on the same GPU.
    A leased GPU is used only once it is completely idle: no process of any
    user or project on it, and its memory below --gpu-free-mib.
    """
    args.gpu_lock_dir.mkdir(parents=True, exist_ok=True)
    announced = False
    while True:
        for gpu in args.gpus:
            handle = (args.gpu_lock_dir / f"gpu{gpu}.lock").open("w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            if gpu_idle(gpu, args.gpu_free_mib):
                return gpu, handle
            # Busy with another job: release it so the next GPU can be tried.
            handle.close()
        if not announced:
            print(f"[{args.output}] {task}: waiting for an idle GPU", flush=True)
            announced = True
        time.sleep(10)


def run_task(args: argparse.Namespace, task: str) -> tuple[str, str, float, float | None, list[str], int]:
    task_dir = args.output / task
    if args.reuse_complete and completed_run(args, task):
        print(f"[{args.output}] {task}: reusing the completed lm-eval run", flush=True)
        args.reused_tasks.append(task)
        return collect_result(args, task)
    if args.reuse_complete and task_dir.exists():
        # Keep an interrupted run for audit; its partial files must not be read.
        task_dir.rename(args.output / f"{task}.incomplete_{args.stamp}")
    gpu, lease = lease_gpu(args, task)
    try:
        run_task_on_gpu(args, task, gpu)
    finally:
        lease.close()
    return collect_result(args, task)


def build_command(args: argparse.Namespace, task: str) -> list[str]:
    model_args = (
        f"pretrained={args.model_path},tensor_parallel_size=1,dtype=bfloat16,"
        f"gpu_memory_utilization={args.gpu_memory_utilization},trust_remote_code=True"
    )
    gen_kwargs = f"max_new_tokens={args.max_new_tokens},temperature=0.0"
    if task in STOP_OVERRIDES:
        gen_kwargs += f",until={STOP_OVERRIDES[task]}"
    cmd = [
        sys.executable, "-m", "lm_eval", "--model", "vllm",
        "--model_args", model_args, "--tasks", task, "--batch_size", "auto",
        "--log_samples", "--output_path", str(args.output / task),
        "--gen_kwargs", gen_kwargs,
    ]
    spec = ALL_TASKS[task]
    if task != "mbpp":
        cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
    if spec["fewshot"] is not None:
        cmd += ["--num_fewshot", str(spec["fewshot"])]
    if spec["unsafe"]:
        cmd.append("--confirm_run_unsafe_code")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def comparable(cmd: list[str]) -> list[str]:
    """Drop the interpreter and resolve paths, which may be relative or absolute."""
    norm = list(cmd[1:])
    for index, item in enumerate(norm):
        if index and norm[index - 1] == "--output_path":
            norm[index] = str(Path(item).resolve())
        elif item.startswith("pretrained="):
            path, _, rest = item.removeprefix("pretrained=").partition(",")
            norm[index] = f"pretrained={Path(path).resolve()},{rest}"
    return norm


def completed_run(args: argparse.Namespace, task: str) -> bool:
    """True if the task directory holds a finished run of exactly this command."""
    task_dir = args.output / task
    log = task_dir / "eval.log"
    if not log.is_file() or not list(task_dir.rglob("results*.json")):
        return False
    first = log.read_text(errors="ignore").split("\n", 1)[0]
    if not first.startswith("COMMAND: "):
        return False
    if comparable(json.loads(first.removeprefix("COMMAND: "))) != comparable(build_command(args, task)):
        raise RuntimeError(f"{task_dir} holds a finished run of a different command; refusing to reuse it")
    return True


def run_task_on_gpu(args: argparse.Namespace, task: str, gpu: str) -> None:
    task_dir = args.output / task
    task_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, task)
    env = os.environ.copy()
    env.update({
        # Match nvidia-smi numbering, which lease_gpu() uses for the idle check.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
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


def collect_result(args: argparse.Namespace, task: str) -> tuple[str, str, float, float | None, list[str], int]:
    task_dir = args.output / task
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


def vllm_model_path(checkpoint: Path) -> Path:
    """Return a checkpoint vLLM can load, without training-only modules.

    nuno_kd.train saves the hidden_mse projectors (``projectors.*``) with the student
    weights; vLLM rejects unknown parameters, so evaluate a copy without them.
    """
    weights = checkpoint / "pytorch_model.bin"
    if not weights.is_file():
        return checkpoint
    import torch

    state = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
    extra = [key for key in state if not key.startswith(("model.", "lm_head."))]
    if not extra:
        return checkpoint
    export = checkpoint.with_name(checkpoint.name + "-vllm")
    done = export / "EXPORT_COMPLETE"
    if not done.is_file():
        shutil.rmtree(export, ignore_errors=True)
        export.mkdir(parents=True)
        for path in checkpoint.iterdir():
            if path.is_file() and path.name != weights.name:
                shutil.copy2(path, export / path.name)
        torch.save({k: v for k, v in state.items() if k not in extra}, export / weights.name)
        done.write_text(json.dumps({"source": str(checkpoint), "dropped": extra}, indent=2) + "\n")
    print(f"evaluating {export} ({len(extra)} training-only tensors dropped)", flush=True)
    return export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-new-tokens", type=int, default=5120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks", help="Comma-separated tasks: rerun into an existing scores.json, or run alone in a fresh output.")
    parser.add_argument(
        "--gpu-lock-dir", type=Path, default=Path(f"/tmp/eval_lm_harness_gpu_locks_{os.getuid()}"),
        help="Shared by concurrent evaluations so they draw from one GPU pool.",
    )
    parser.add_argument("--gpu-free-mib", type=int, default=2048, help="A leased GPU is used below this memory use.")
    parser.add_argument(
        "--reuse-complete", action="store_true",
        help="Reuse task directories holding a finished run of the same command; rerun the rest.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")
    args.gpus = gpus
    args.reused_tasks = []
    args.stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    scores: dict[str, dict[str, float | str]] = {}
    previous = None
    task_names = list(TASKS)
    if args.tasks:
        task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
        unknown = sorted(set(task_names) - set(ALL_TASKS))
        if unknown:
            parser.error(f"unknown tasks: {unknown}")
    if args.tasks and (args.output / "scores.json").is_file():
        previous = json.loads((args.output / "scores.json").read_text())
        scores = dict(previous["scores"])
        # Keep the superseded run for audit; its samples must not be recounted.
        for task in task_names:
            if (args.output / task).exists():
                (args.output / task).rename(args.output / f"{task}.superseded_{args.stamp}")
        (args.output / "scores.json").rename(args.output / f"scores.superseded_{args.stamp}.json")
    args.model_path = vllm_model_path(args.checkpoint)
    # Each task leases one GPU from the shared pool as soon as one is free.
    with ThreadPoolExecutor(max_workers=len(task_names)) as pool:
        pending = {pool.submit(run_task, args, task): task for task in task_names}
        for future in as_completed(pending):
            task, metric, value, stderr, sample_files, sample_count = future.result()
            scores[task] = {
                "metric": metric,
                "value": value,
                "lm_eval_stderr": stderr,
                "sample_files": sample_files,
                "sample_count": sample_count,
            }
            print(f"[{args.output}] {task}: {value:.2f} ({metric}), samples={sample_count}", flush=True)

    suite = [task for task in ALL_TASKS if task in scores]
    # The average is the eight-task mean when all eight are present; a run of
    # other tasks only (e.g. --tasks hendrycks_math in a fresh output) averages those.
    averaged = list(TASKS) if all(task in scores for task in TASKS) else suite
    ordered_values = [float(scores[task]["value"]) for task in averaged]
    output = {
        "checkpoint": str(args.checkpoint),
        "evaluated_path": str(args.model_path),
        "stop_overrides": STOP_OVERRIDES,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lm_eval_commit": os.environ.get("LM_EVAL_COMMIT", "unknown"),
        "gpus": gpus,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "scores": {task: scores[task] for task in suite},
        "average": sum(ordered_values) / len(ordered_values),
    }
    if args.reused_tasks:
        output["reused_tasks"] = sorted(args.reused_tasks)
    if previous is not None:
        output["rerun_tasks"] = task_names
        output["previous_created_at_utc"] = previous.get("created_at_utc")
    (args.output / "scores.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
