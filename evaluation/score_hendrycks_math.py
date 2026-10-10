#!/usr/bin/env python3
"""Rescore lm-eval hendrycks_math sample logs with three graders.

lm-eval's own hendrycks_math grader compares the text between the first and
the last ``$`` of the whole response with the gold answer. That suits short
base-model answers, but a chat model's worked solution contains many ``$``
pairs, so the grader almost never extracts the final answer. This script keeps
that score and adds two graders on the same, unchanged responses:

- ``lm_eval_exact_match``: the score lm-eval logged for each sample.
- ``boxed_exact_match``: the last ``\\boxed{...}`` of the response, compared
  with the gold answer by the task's own ``is_equiv`` (Hendrycks string
  normalization). This is the answer extraction of the MATH paper.
- ``math_verify``: ``math_verify.verify(parse(solution), parse(response))``,
  as lm-eval's minerva_math task does.

Run with the evaluator venv, which provides lm-eval's task utils and math_verify.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed
from math_verify import parse, verify

ROOT = Path(__file__).resolve().parents[1]

SUBTASKS = ["algebra", "counting_and_prob", "geometry", "intermediate_algebra", "num_theory", "prealgebra", "precalc"]
GRADERS = ["lm_eval_exact_match", "boxed_exact_match", "math_verify"]
FIELDS = [
    "subtask", "doc_id", "level", "type", "problem", "gold_answer", "lm_eval_extracted", "last_boxed",
    "lm_eval_exact_match", "boxed_exact_match", "math_verify", "response_chars", "response",
]


def boxed_answer(text: str) -> str | None:
    try:
        boxed = last_boxed_only_string(text)
        return remove_boxed(boxed) if boxed else None
    except AssertionError:
        return None


def lm_eval_extracted(text: str) -> str:
    """The span lm-eval's process_results compares with the gold answer."""
    indices = [pos for pos, char in enumerate(text) if char == "$"]
    return text if len(indices) <= 1 else text[indices[0] + 1 : indices[-1]]


def rescore(model_dir: Path, label: str, data_dir: Path) -> dict:
    scores = json.loads((model_dir / "scores.json").read_text())
    entry = scores["scores"]["hendrycks_math"]
    results = json.loads(max((model_dir / "hendrycks_math").rglob("results_*.json"), key=lambda p: p.stat().st_mtime_ns).read_text())
    per = {sub: defaultdict(float) for sub in SUBTASKS}
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"hendrycks_math_samples_{label}.csv.gz"
    with gzip.open(out, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        for rel in entry["sample_files"]:
            sub = re.search(r"samples_hendrycks_math_(.+?)_\d{4}-", rel).group(1)
            for line in (model_dir / rel).open():
                row = json.loads(line)
                response = row["resps"][0][0]
                doc = row["doc"]
                boxed = boxed_answer(response)
                grades = {
                    "lm_eval_exact_match": int(row["exact_match"]),
                    "boxed_exact_match": int(boxed is not None and is_equiv(boxed, doc["answer"])),
                    "math_verify": int(bool(verify(gold=parse(doc["solution"]), target=parse(response)))),
                }
                stats = per[sub]
                stats["n"] += 1
                stats["no_boxed"] += boxed is None
                for grader, value in grades.items():
                    stats[grader] += value
                writer.writerow({
                    "subtask": sub, "doc_id": row["doc_id"], "level": doc.get("level"), "type": doc.get("type"),
                    "problem": doc["problem"], "gold_answer": doc["answer"],
                    "lm_eval_extracted": lm_eval_extracted(response), "last_boxed": "" if boxed is None else boxed,
                    **grades, "response_chars": len(response), "response": response,
                })
    total = defaultdict(float)
    for stats in per.values():
        for key, value in stats.items():
            total[key] += value
    # The rescored lm-eval column must reproduce lm-eval's own aggregates exactly.
    group = (results.get("groups") or results["results"])["hendrycks_math"]["exact_match,none"]
    diffs = [abs(total["lm_eval_exact_match"] / total["n"] - group), abs(100 * group - entry["value"])]
    diffs += [abs(per[s]["lm_eval_exact_match"] / per[s]["n"] - results["results"][f"hendrycks_math_{s}"]["exact_match,none"]) for s in SUBTASKS]
    if max(diffs) > 1e-9:
        raise RuntimeError(f"{model_dir}: rescored lm-eval exact_match differs from results JSON by {max(diffs)}")
    summary = {
        "label": label,
        "model_dir": str(model_dir.relative_to(ROOT)),
        "checkpoint": scores["checkpoint"],
        "n": int(total["n"]),
        "no_boxed": int(total["no_boxed"]),
        "overall": {g: 100 * total[g] / total["n"] for g in GRADERS},
        "subtasks": {s: {"n": int(per[s]["n"]), **{g: 100 * per[s][g] / per[s]["n"] for g in GRADERS}} for s in SUBTASKS},
        "max_abs_diff_vs_lm_eval": max(diffs),
        "samples_csv": str(out.relative_to(ROOT)),
    }
    (model_dir / "hendrycks_math_rescored.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Output root holding <run>/<method>/seed10/<step>/.")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for scores in sorted(root.glob("*/*/seed*/*/scores.json")):
        model_dir = scores.parent
        run, method, seed, step = model_dir.relative_to(root).parts
        label = f"{run}_{method}_{seed}_{step}"
        summary = rescore(model_dir, label, root / "data")
        rows.append(summary)
        print(f"{label}: " + ", ".join(f"{g}={summary['overall'][g]:.2f}" for g in GRADERS), flush=True)
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "hendrycks_math_rescored.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (tables / "hendrycks_math_rescored.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "n", "no_boxed", *GRADERS, *(f"{g}:{s}" for g in GRADERS for s in SUBTASKS)])
        for r in rows:
            writer.writerow([r["label"], r["n"], r["no_boxed"], *(f"{r['overall'][g]:.4f}" for g in GRADERS),
                             *(f"{r['subtasks'][s][g]:.4f}" for g in GRADERS for s in SUBTASKS)])


if __name__ == "__main__":
    main()
