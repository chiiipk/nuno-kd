#!/usr/bin/env python3
"""Aggregate two-seed lm-eval scores into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


TASKS = ["gsm8k", "gsm_plus", "minerva_math", "mbpp", "sciq", "mmlu_stem", "mmlu_pro_math", "bbh_cot_fewshot"]
LABELS = ["GSM8K", "GSM-Plus", "MATH", "MBPP", "SciQ", "STEM", "Pro-Math", "BBH-COT"]


def summarize(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    if not values:
        raise ValueError("Cannot summarize an empty result list")
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=["qwen", "qwen3"])
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 42])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for pair in args.pairs:
        rows = []
        machine_rows = []
        for method in args.methods:
            payloads = []
            for seed in args.seeds:
                path = args.eval_root / pair / method / f"seed{seed}" / "scores.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                payloads.append(json.loads(path.read_text()))
            cells = []
            seed_averages = []
            for task in TASKS:
                vals = [float(item["scores"][task]["value"]) for item in payloads]
                cells.append(summarize(vals))
            for item in payloads:
                seed_averages.append(sum(float(item["scores"][task]["value"]) for task in TASKS) / len(TASKS))
            rows.append((method, cells, summarize(seed_averages)))
            machine_rows.append({
                "method": method,
                "seeds": args.seeds,
                "per_seed": {
                    str(seed): {
                        **{task: float(payload["scores"][task]["value"]) for task in TASKS},
                        "average": sum(float(payload["scores"][task]["value"]) for task in TASKS) / len(TASKS),
                    }
                    for seed, payload in zip(args.seeds, payloads)
                },
                "summary": {
                    **{
                        task: {"mean": mean, "sample_std": std}
                        for task, (mean, std) in zip(TASKS, cells)
                    },
                    "average": {"mean": rows[-1][2][0], "sample_std": rows[-1][2][1]},
                },
            })

        csv_path = args.output / f"{pair}_full8_mean_std.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Method", *LABELS, "Avg"])
            for method, cells, avg in rows:
                writer.writerow([method, *[f"{m:.2f} ± {s:.2f}" for m, s in cells], f"{avg[0]:.2f} ± {avg[1]:.2f}"])

        tex_path = args.output / f"{pair}_full8_mean_std.tex"
        lines = [
            "\\begin{tabular}{l" + "c" * 9 + "}",
            "\\toprule",
            "Method & " + " & ".join(LABELS) + " & Avg. \\\\",
            "\\midrule",
        ]
        for method, cells, avg in rows:
            formatted = [f"${m:.2f} \\pm {s:.2f}$" for m, s in cells]
            formatted.append(f"${avg[0]:.2f} \\pm {avg[1]:.2f}$")
            lines.append(method.replace("_", "\\_") + " & " + " & ".join(formatted) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        tex_path.write_text("\n".join(lines) + "\n")
        json_path = args.output / f"{pair}_full8_mean_std.json"
        json_path.write_text(json.dumps({"pair": pair, "ddof": 1, "rows": machine_rows}, indent=2) + "\n")
        print(csv_path)
        print(tex_path)
        print(json_path)


if __name__ == "__main__":
    main()
