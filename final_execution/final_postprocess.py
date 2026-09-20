#!/usr/bin/env python3
"""Finalize verifier audits, coverage gates, tables, and Figure E."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_execution.math_verifier import VERIFIER_VERSION, verify_answers


EVAL = ROOT / "artifacts" / "final_execution" / "final_eval"
TASK5 = ROOT / "artifacts" / "final_execution" / "task5"
STATUS = ROOT / "artifacts" / "final_execution" / "status.json"
VERIFY_TASKS = ["aime24", "aime25", "hendrycks_math500", "olympiadbench"]
EXPECTED = {"aime24": 30, "aime25": 30, "hendrycks_math500": 500, "olympiadbench": 674}
ALL_TASKS = [
    "gsm8k", "gsm_plus", "mmlu_pro_math", "aime24", "aime25",
    "hendrycks_math500", "olympiadbench", "mmlu_stem",
    "gpqa_diamond_zeroshot", "mbpp",
]
METRICS = {
    "gsm8k": ("results", "exact_match,flexible-extract"),
    "gsm_plus": ("results", "exact_match,flexible-extract"),
    "mmlu_pro_math": ("results", "exact_match,custom-extract"),
    "mmlu_stem": ("groups", "acc,none"),
    "gpqa_diamond_zeroshot": ("results", "acc_norm,none"),
    "mbpp": ("results", "pass_at_1,none"),
}
METHOD_LABEL = {
    "teacher": "Teacher", "base": "Base", "distillm": "DistiLLM",
    "csd": "CSD", "cka": "CKA", "cst_w005_l2": "CST-w0.005-L2",
    "cst_w005_l4": "CST-w0.005-L4", "cst_w005_l8": "CST-w0.005-L8",
    "cst_w008_l4": "CST-w0.008-L4",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def update_status(phase: str, **extra: Any) -> None:
    atomic_json(STATUS, {"phase": phase, "updated": time.time(), **extra})


def latest(directory: Path, pattern: str) -> Path:
    paths = list(directory.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def raw_response(row: dict) -> str:
    responses = row.get("resps") or []
    if responses and isinstance(responses[0], list) and responses[0]:
        return str(responses[0][0])
    if responses:
        return str(responses[0])
    return ""


def references(row: dict, task: str) -> list[str]:
    if task == "olympiadbench":
        values = row.get("doc", {}).get("final_answer", [])
        if not isinstance(values, list):
            values = [values]
        return [str(value) for value in values if str(value).strip()]
    return [str(row.get("target", ""))]


def verify_item(item: tuple[int, str, dict, str]) -> tuple[int, dict]:
    index, task, row, source = item
    prediction = raw_response(row)
    refs = references(row, task)
    attempts = [verify_answers(reference, prediction) for reference in refs] or [verify_answers("", prediction)]
    chosen = next((result for result in attempts if result["correct"]), attempts[0])
    record = {
        "task": task,
        "doc_id": row.get("doc_id"),
        "doc_hash": row.get("doc_hash"),
        "prompt_hash": row.get("prompt_hash"),
        "target_hash": row.get("target_hash"),
        "references": refs,
        "generation": prediction,
        "generation_hash": hashlib.sha256(prediction.encode()).hexdigest(),
        "generation_present": bool(prediction.strip()),
        "built_in_exact_match": row.get("exact_match"),
        "correct": any(result["correct"] for result in attempts),
        "verification_method": chosen.get("verification_method"),
        "failure_type": None if any(result["correct"] for result in attempts) else chosen.get("failure_type"),
        "extracted_reference": chosen.get("extracted_reference", []),
        "extracted_prediction": chosen.get("extracted_prediction", []),
        "normalized_reference": chosen.get("normalized_reference", []),
        "normalized_prediction": chosen.get("normalized_prediction", []),
        "verifier_version": VERIFIER_VERSION,
        "source_sample": source,
    }
    return index, record


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" + "r" * (len(fields) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(latex_escape(x) for x in fields) + r" \\", "\\midrule"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            values.append(latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def dataset_revisions(configs: dict[str, dict]) -> dict[str, dict]:
    output = {}
    for task in VERIFY_TASKS:
        config = configs[task]
        repo = config.get("dataset_path")
        revision = None
        revision_source = None
        try:
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info(repo).sha
            revision_source = "huggingface_hub.dataset_info"
        except Exception as error:
            revision_source = f"unavailable: {type(error).__name__}"
        output[task] = {
            "dataset_path": repo,
            "dataset_name": config.get("dataset_name"),
            "split": config.get("test_split"),
            "resolved_revision": revision,
            "revision_source": revision_source,
            "expected_count": EXPECTED[task],
        }
    return output


def collect_audits(out: Path, jobs: list[dict], workers: int, limit: int | None) -> tuple[list[dict], list[dict], dict]:
    audit_root = out / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict] = []
    score_rows: list[dict] = []
    first_results = json.loads(latest(EVAL / "runs" / jobs[0]["id"] / "generative", "results*.json").read_text())
    identities = dataset_revisions(first_results["configs"])
    total_units = len(jobs) * len(VERIFY_TASKS)
    completed_units = 0
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for job in jobs:
            job_dir = EVAL / "runs" / job["id"]
            if not (job_dir / "complete.json").exists():
                raise RuntimeError(f"incomplete evaluation: {job['id']}")
            for task in VERIFY_TASKS:
                sample_path = latest(job_dir / "generative", f"samples_{task}_*.jsonl")
                samples = read_jsonl(sample_path)
                if limit is not None:
                    samples = samples[:limit]
                items = [(index, task, row, str(sample_path)) for index, row in enumerate(samples)]
                verified = sorted(pool.imap_unordered(verify_item, items, chunksize=8), key=lambda item: item[0])
                records = [record for _, record in verified]
                audit_path = audit_root / job["id"] / f"{task}.jsonl"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("w") as sink:
                    for record in records:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")

                doc_ids = [record["doc_id"] for record in records]
                failures: dict[str, int] = {}
                methods: dict[str, int] = {}
                for record in records:
                    key = record["failure_type"] or "accepted"
                    failures[key] = failures.get(key, 0) + 1
                    method = record["verification_method"] or "none"
                    methods[method] = methods.get(method, 0) + 1
                expected = min(EXPECTED[task], limit) if limit is not None else EXPECTED[task]
                coverage_rows.append({
                    "job_id": job["id"], "pair": job["pair"], "method": job["method"],
                    "train_seed": job["train_seed"], "task": task, "expected": expected,
                    "sample_count": len(samples), "unique_doc_ids": len(set(doc_ids)),
                    "duplicates": len(doc_ids) - len(set(doc_ids)),
                    "generation_present": sum(record["generation_present"] for record in records),
                    "audit_count": len(records), "correct": sum(record["correct"] for record in records),
                    "score": sum(record["correct"] for record in records) / len(records) if records else float("nan"),
                    "failure_counts": json.dumps(failures, sort_keys=True),
                    "verification_methods": json.dumps(methods, sort_keys=True),
                    "source_sample": str(sample_path), "audit_path": str(audit_path),
                    "audit_sha256": sha256(audit_path),
                    "passed": len(samples) == expected == len(set(doc_ids)) == len(records) == sum(record["generation_present"] for record in records),
                })
                score_rows.append({
                    "job_id": job["id"], "pair": job["pair"], "method": job["method"],
                    "method_label": METHOD_LABEL[job["method"]], "train_seed": job["train_seed"],
                    "benchmark": task, "score": coverage_rows[-1]["score"], "score_source": VERIFIER_VERSION,
                })
                completed_units += 1
                update_status("final_postprocess_verifier", completed=completed_units, total=total_units, current=f"{job['id']}:{task}")
    return coverage_rows, score_rows, identities


def metric_value(payload: dict, task: str) -> float:
    section, metric = METRICS[task]
    values = payload[section][task]
    if metric not in values:
        raise KeyError(f"{task}: {metric} not found in {list(values)}")
    return float(values[metric])


def collect_builtin_scores(jobs: list[dict]) -> list[dict]:
    rows = []
    phase_for = {task: "generative" for task in ALL_TASKS}
    phase_for.update({"mmlu_stem": "likelihood", "gpqa_diamond_zeroshot": "likelihood", "mbpp": "code"})
    for job in jobs:
        job_dir = EVAL / "runs" / job["id"]
        payloads = {phase: json.loads(latest(job_dir / phase, "results*.json").read_text()) for phase in {phase_for[t] for t in ALL_TASKS}}
        for task in ALL_TASKS:
            if task in VERIFY_TASKS:
                continue
            rows.append({
                "job_id": job["id"], "pair": job["pair"], "method": job["method"],
                "method_label": METHOD_LABEL[job["method"]], "train_seed": job["train_seed"],
                "benchmark": task, "score": metric_value(payloads[phase_for[task]], task),
                "score_source": "lm-eval-canonical",
            })
    return rows


def aggregate_scores(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    macro_by_job: dict[str, list[float]] = {}
    job_meta: dict[str, dict] = {}
    for row in rows:
        key = (row["pair"], row["method"], row["method_label"], row["benchmark"])
        grouped.setdefault(key, []).append(float(row["score"]))
        macro_by_job.setdefault(row["job_id"], []).append(float(row["score"]))
        job_meta[row["job_id"]] = row
    summary = []
    for (pair, method, label, benchmark), values in sorted(grouped.items()):
        summary.append({
            "pair": pair, "method": method, "method_label": label, "benchmark": benchmark,
            "mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        })
    macro = []
    for job_id, values in sorted(macro_by_job.items()):
        if len(values) != len(ALL_TASKS):
            raise RuntimeError(f"{job_id}: expected {len(ALL_TASKS)} benchmark scores, got {len(values)}")
        meta = job_meta[job_id]
        macro.append({
            "job_id": job_id, "pair": meta["pair"], "method": meta["method"],
            "method_label": meta["method_label"], "train_seed": meta["train_seed"],
            "macro_average": statistics.fmean(values), "n_benchmarks": len(values),
        })
    return summary, macro


def make_tables(out: Path, per_seed: list[dict], summary: list[dict], macro: list[dict]) -> None:
    tables = out / "tables"
    write_csv(tables / "benchmark_per_seed.csv", per_seed)
    atomic_json(tables / "benchmark_per_seed.json", per_seed)
    write_csv(tables / "benchmark_mean_std.csv", summary)
    atomic_json(tables / "benchmark_mean_std.json", summary)
    write_latex(tables / "benchmark_mean_std.tex", summary, ["pair", "method_label", "benchmark", "mean", "std", "n_seeds"])
    write_csv(tables / "benchmark_macro_by_checkpoint.csv", macro)
    atomic_json(tables / "benchmark_macro_by_checkpoint.json", macro)

    selected = {"qwen": "cst_w005_l4", "qwen3": "cst_w005_l8"}
    for pair in ["qwen", "qwen3"]:
        keep = {"teacher", "base", "distillm", "csd", "cka", selected[pair]}
        subset = [row for row in summary if row["pair"] == pair and row["method"] in keep]
        write_csv(tables / f"{pair}_final_comparison.csv", subset)
        write_latex(tables / f"{pair}_final_comparison.tex", subset, ["method_label", "benchmark", "mean", "std", "n_seeds"])
    ablation = [row for row in summary if row["pair"] == "qwen3" and row["method"].startswith("cst_")]
    write_csv(tables / "qwen3_cst_ablation.csv", ablation)
    write_latex(tables / "qwen3_cst_ablation.tex", ablation, ["method_label", "benchmark", "mean", "std", "n_seeds"])


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            result[position] = rank
        index = end
    return result


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")
    lx, rx = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - lx) * (y - rx) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - lx) ** 2 for x in left) * sum((y - rx) ** 2 for y in right))
    return numerator / denominator if denominator else float("nan")


def make_figure_e(out: Path, macro: list[dict]) -> list[dict]:
    by_layer = TASK5 / "tables" / "intrinsic_by_layer.csv"
    gaps: dict[tuple[str, str, str], list[float]] = {}
    with by_layer.open() as source:
        for row in csv.DictReader(source):
            if row["method"] not in {"Base", "DistiLLM", "CSD", "CKA", "CST"}:
                continue
            gaps.setdefault((row["pair"], row["method"], row["seed"]), []).append(float(row["cst_curve_gap"]))
    gap_mean = {key: statistics.fmean(values) for key, values in gaps.items()}
    base = {pair: value for (pair, method, seed), value in gap_mean.items() if method == "Base"}
    intrinsic_method = {"distillm": "DistiLLM", "csd": "CSD", "cka": "CKA"}
    selected_cst = {"qwen": "cst_w005_l4", "qwen3": "cst_w005_l8"}
    points = []
    for row in macro:
        method = row["method"]
        if method == selected_cst[row["pair"]]:
            intrinsic = "CST"
        elif method in intrinsic_method:
            intrinsic = intrinsic_method[method]
        else:
            continue
        key = (row["pair"], intrinsic, row["train_seed"])
        if key not in gap_mean:
            raise KeyError(f"missing intrinsic checkpoint {key}")
        points.append({**row, "intrinsic_method": intrinsic, "normalized_cst_curve_gap": gap_mean[key] / (base[row["pair"]] + 1e-12)})

    x = [row["normalized_cst_curve_gap"] for row in points]
    y = [row["macro_average"] for row in points]
    correlations = [{
        "analysis": "exploratory", "pearson": pearson(x, y),
        "spearman": pearson(ranks(x), ranks(y)), "n": len(points),
        "included_checkpoints": [row["job_id"] for row in points],
    }]
    tables = out / "tables"
    write_csv(tables / "intrinsic_downstream_correlation.csv", correlations, ["analysis", "pearson", "spearman", "n", "included_checkpoints"])
    atomic_json(tables / "intrinsic_downstream_correlation.json", correlations)
    write_csv(tables / "figure_e_plot_data.csv", points)
    atomic_json(tables / "figure_e_plot_data.json", points)

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ROOT / ".venvs" / "train" / "bin" / "python"),
        str(ROOT / "final_execution" / "plot_figure_e.py"),
        "--data", str(tables / "figure_e_plot_data.json"),
        "--output", str(figures),
    ], check=True, cwd=ROOT)
    return correlations


def select_manual_audit(out: Path) -> list[dict]:
    records = []
    for path in sorted((out / "audits").rglob("*.jsonl")):
        job_id = path.parent.name
        for row in read_jsonl(path):
            row["job_id"] = job_id
            row["audit_path"] = str(path)
            records.append(row)
    stable = lambda row: hashlib.sha256(f"{row['job_id']}:{row['task']}:{row['doc_id']}".encode()).hexdigest()

    def stratified(pool: list[dict], count: int, fields: tuple[str, ...]) -> list[dict]:
        groups: dict[tuple, list[dict]] = {}
        for row in pool:
            groups.setdefault(tuple(row.get(field) for field in fields), []).append(row)
        for values in groups.values():
            values.sort(key=stable)
        selected = []
        keys = sorted(groups, key=str)
        while len(selected) < count and keys:
            remaining = []
            for key in keys:
                if groups[key] and len(selected) < count:
                    selected.append(groups[key].pop(0))
                if groups[key]:
                    remaining.append(key)
            keys = remaining
        return selected

    accepted = stratified([row for row in records if row["correct"]], 25, ("task", "verification_method"))
    rejected = stratified([row for row in records if not row["correct"]], 25, ("task", "failure_type"))
    parser_errors = sorted((row for row in records if row["failure_type"] == "parse_error"), key=stable)[:50]
    timeouts = sorted((row for row in records if row["failure_type"] == "verification_timeout"), key=stable)[:50]
    numeric_accept = stratified([row for row in records if row["correct"] and row["verification_method"] == "numeric"], 25, ("task",))
    symbolic_accept = stratified([row for row in records if row["correct"] and row["verification_method"] == "symbolic"], 25, ("task",))
    chosen = {}
    categories = [
        ("accepted", accepted), ("rejected", rejected), ("parser_error", parser_errors),
        ("verification_timeout", timeouts), ("numeric_nonexact_accept", numeric_accept),
        ("symbolic_nonexact_accept", symbolic_accept),
    ]
    for category, rows in categories:
        for row in rows:
            key = (row["job_id"], row["task"], row["doc_id"])
            entry = chosen.setdefault(key, dict(row, audit_categories=[]))
            entry["audit_categories"].append(category)
    queue = list(chosen.values())
    path = out / "manual_audit" / "queue.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        for row in queue:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(out / "manual_audit" / "selection.json", {
        "selection_hash": "sha256(job_id:task:doc_id)", "required_accepted": 25,
        "required_rejected": 25, "per_parser_error_or_timeout_category_cap": 50,
        "accepted": len(accepted), "rejected": len(rejected),
        "parser_error": len(parser_errors), "verification_timeout": len(timeouts),
        "numeric_nonexact_accept": len(numeric_accept), "symbolic_nonexact_accept": len(symbolic_accept),
        "unique_records": len(queue), "queue_sha256": sha256(path),
    })
    return queue


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in ["math-verify", "sympy", "latex2sympy2-extended", "lm-eval", "transformers", "vllm"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(EVAL / "postprocess"))
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--limit-per-task", type=int)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((EVAL / "manifest.json").read_text())
    jobs = manifest["models"]
    if len(jobs) != 22:
        raise RuntimeError(f"expected 22 jobs, got {len(jobs)}")

    update_status("final_postprocess_starting", completed=0, total=len(jobs) * len(VERIFY_TASKS))
    coverage, verifier_scores, identities = collect_audits(out, jobs, args.workers, args.limit_per_task)
    write_csv(out / "coverage_report.csv", coverage)
    atomic_json(out / "coverage_report.json", {"dataset_identities": identities, "rows": coverage})
    per_seed = collect_builtin_scores(jobs) + verifier_scores
    per_seed.sort(key=lambda row: (row["pair"], row["method"], row["train_seed"], ALL_TASKS.index(row["benchmark"])))
    summary, macro = aggregate_scores(per_seed)
    make_tables(out, per_seed, summary, macro)
    correlations = make_figure_e(out, macro)
    queue = select_manual_audit(out)

    verifier_path = ROOT / "final_execution" / "math_verifier.py"
    gate_passed = all(row["passed"] for row in coverage) and len(coverage) == len(jobs) * len(VERIFY_TASKS)
    gate = {
        "status": "awaiting_manual_audit" if gate_passed else "failed",
        "automated_coverage_passed": gate_passed,
        "coverage_cells": len(coverage), "expected_cells": len(jobs) * len(VERIFY_TASKS),
        "manual_audit_queue_records": len(queue), "manual_audit_reviewed": False,
        "verifier_version": VERIFIER_VERSION, "verifier_source": str(verifier_path),
        "verifier_sha256": sha256(verifier_path), "dependency_versions": dependency_versions(),
        "dataset_identities": identities, "correlations": correlations,
    }
    atomic_json(out / "verification_gate.json", gate)
    report = [
        "# Final verification gate", "", f"- Automated coverage: {'PASS' if gate_passed else 'FAIL'}",
        f"- Coverage cells: {len(coverage)}/{len(jobs) * len(VERIFY_TASKS)}",
        f"- Manual-audit queue: {len(queue)} records (review pending)",
        f"- Verifier: `{VERIFIER_VERSION}` / `{gate['verifier_sha256']}`", "",
        "Unified verifier scores are authoritative for AIME 2024, AIME 2025, MATH-500, and OlympiadBench. Built-in scores are cross-checks only.",
    ]
    (out / "verification_gate.md").write_text("\n".join(report) + "\n")
    atomic_json(out / "postprocess_manifest.json", {
        "source_eval_manifest": str(EVAL / "manifest.json"), "source_eval_manifest_sha256": sha256(EVAL / "manifest.json"),
        "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "verifier_sha256": sha256(verifier_path), "workers": args.workers,
        "limit_per_task": args.limit_per_task, "generated_at": time.time(),
    })
    update_status("final_postprocess_manual_audit_pending", automated_coverage_passed=gate_passed, manual_audit_records=len(queue))
    print(json.dumps({"coverage_passed": gate_passed, "coverage_cells": len(coverage), "manual_audit_records": len(queue), "output": str(out)}))


if __name__ == "__main__":
    main()
