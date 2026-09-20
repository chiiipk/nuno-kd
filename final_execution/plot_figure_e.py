#!/usr/bin/env python3
"""Render deterministic Figure E from stored plotting data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_path = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    points = json.loads(data_path.read_text())
    colors = {"DistiLLM": "#0072B2", "CSD": "#E69F00", "CKA": "#009E73", "CST": "#CC79A7"}
    markers = {"qwen": "o", "qwen3": "s"}

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for row in points:
        axis.scatter(
            row["normalized_cst_curve_gap"], row["macro_average"],
            color=colors[row["intrinsic_method"]], marker=markers[row["pair"]],
            s=62, edgecolor="black", linewidth=0.5,
        )
    for method, color in colors.items():
        axis.scatter([], [], color=color, marker="o", label=method)
    for pair, marker in markers.items():
        axis.scatter([], [], facecolor="white", edgecolor="black", marker=marker, label=pair)
    axis.set_xlabel("Normalized CST curve gap (lower is closer to teacher)")
    axis.set_ylabel("Final benchmark macro-average")
    axis.set_title(f"Exploratory intrinsic/downstream relationship (n={len(points)})")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    for suffix in ["pdf", "svg"]:
        figure.savefig(output / f"figure_e_intrinsic_downstream.{suffix}", bbox_inches="tight")
    figure.savefig(output / "figure_e_intrinsic_downstream.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    config = {
        "analysis": "exploratory", "colors": colors, "markers": markers,
        "trend_line": False, "data_sha256": sha256(data_path),
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "figure_e_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
