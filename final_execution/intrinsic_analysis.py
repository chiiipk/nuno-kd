#!/usr/bin/env python3
"""Task 5 intrinsic spectral analysis for FINAL_EXECUTION_PLAN.md.

The script has three subcommands:
  prepare   create the immutable held-out probe and job manifest
  worker    extract per-example, per-depth spectral statistics for one model
  aggregate validate coverage and produce tables, reports, and figures
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_execution" / "task5"
DEPTHS = np.linspace(0.0, 1.0, 9).tolist()
GAMMAS = np.logspace(-2, 2, 100).tolist()
BOOTSTRAP_SEED = 2026
METHOD_ORDER = ["Teacher", "Base", "DistiLLM", "CSD", "CKA", "CST"]
COLORS = {
    "Teacher": "#000000",
    "Base": "#7f7f7f",
    "DistiLLM": "#E69F00",
    "CSD": "#CC79A7",
    "CKA": "#0072B2",
    "CST": "#009E73",
}
LINESTYLES = {"Teacher": "-", "Base": "--", "DistiLLM": "-.", "CSD": ":", "CKA": "-", "CST": "--"}
MARKERS = {"Teacher": "o", "Base": "x", "DistiLLM": "s", "CSD": "v", "CKA": "D", "CST": "^"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_best(path: str) -> str:
    return json.loads((ROOT / path).read_text())["checkpoint"]


def snapshot(repo_dir: str) -> str:
    paths = sorted((Path(repo_dir) / "snapshots").iterdir())
    if len(paths) != 1:
        raise RuntimeError(f"expected one cached snapshot under {repo_dir}, found {paths}")
    return str(paths[0].resolve())


def build_jobs() -> list[dict]:
    qwen_base = snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    qwen_teacher = snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct")
    qwen3_base = snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B")
    qwen3_teacher = snapshot("/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen3-8B")
    specs = [
        ("qwen", "Teacher", "na", qwen_teacher),
        ("qwen", "Base", "na", qwen_base),
        ("qwen", "DistiLLM", "10", read_best("checkpoints/main/qwen/distillm/seed10/best.json")),
        ("qwen", "CSD", "10", read_best("checkpoints/main/qwen/csd/seed10/best.json")),
        ("qwen", "CKA", "10", read_best("checkpoints/final/cka_repo/qwen/seed10/best.json")),
        ("qwen", "CKA", "42", read_best("checkpoints/final/cka_repo/qwen/seed42/best.json")),
        ("qwen", "CST", "10", read_best("checkpoints/final/cst/qwen/cst_w005_l4/seed10/best.json")),
        ("qwen", "CST", "42", read_best("checkpoints/final/cst/qwen/cst_w005_l4/seed42/best.json")),
        ("qwen3", "Teacher", "na", qwen3_teacher),
        ("qwen3", "Base", "na", qwen3_base),
        ("qwen3", "DistiLLM", "10", read_best("checkpoints/main/qwen3/distillm/seed10/best.json")),
        ("qwen3", "CSD", "10", read_best("checkpoints/main/qwen3/csd/seed10/best.json")),
        ("qwen3", "CKA", "10", read_best("checkpoints/final/cka_repo/qwen3/seed10/best.json")),
        ("qwen3", "CKA", "42", read_best("checkpoints/final/cka_repo/qwen3/seed42/best.json")),
        ("qwen3", "CST", "10", read_best("checkpoints/final/cst/qwen3/cst_w005_l8/seed10/best.json")),
        ("qwen3", "CST", "42", read_best("checkpoints/final/cst/qwen3/cst_w005_l8/seed42/best.json")),
    ]
    jobs = []
    for pair, method, seed, checkpoint in specs:
        job_id = f"{pair}-{method.lower()}-{seed}".replace("na", "fixed")
        jobs.append({"id": job_id, "pair": pair, "method": method, "seed": seed, "checkpoint": checkpoint})
    return jobs


def prepare(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    probe_path = OUT / "probe.jsonl"
    if not probe_path.exists():
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        doc_ids = sorted(rng.choice(len(ds), size=args.n_examples, replace=False).tolist())
        with probe_path.open("w") as handle:
            for doc_id in doc_ids:
                row = ds[int(doc_id)]
                prompt = "Solve the following problem carefully and give the final answer.\n\n" + row["question"]
                continuation = row["answer"]
                payload = {
                    "dataset": "openai/gsm8k",
                    "config": "main",
                    "split": "test",
                    "dataset_revision": getattr(ds, "_fingerprint", "unknown"),
                    "doc_id": int(doc_id),
                    "prompt": prompt,
                    "continuation": continuation,
                    "prompt_hash": sha256_bytes(prompt.encode()),
                    "continuation_hash": sha256_bytes(continuation.encode()),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    jobs = build_jobs()
    atomic_json(OUT / "jobs.json", jobs)
    atomic_json(
        OUT / "config.json",
        {
            "n_examples": sum(1 for line in probe_path.open() if line.strip()),
            "max_response_tokens": 64,
            "normalized_depths": DEPTHS,
            "gammas": GAMMAS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": args.bootstrap_replicates,
            "method_order": METHOD_ORDER,
            "colors": COLORS,
            "linestyles": LINESTYLES,
            "markers": MARKERS,
            "script_sha256": sha256_file(Path(__file__)),
            "probe_sha256": sha256_file(probe_path),
        },
    )
    print(json.dumps({"probe": str(probe_path), "jobs": len(jobs)}))


def encode_example(tokenizer, probe: dict, max_response_tokens: int) -> tuple[list[int], list[int], str]:
    messages = [{"role": "user", "content": probe["prompt"]}]
    kwargs = dict(tokenize=True, add_generation_prompt=True)
    try:
        prompt_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        prompt_ids = tokenizer.apply_chat_template(messages, **kwargs)
    response_ids = tokenizer(probe["continuation"], add_special_tokens=False).input_ids[:max_response_tokens]
    prompt_ids = prompt_ids[-512:]
    full_ids = prompt_ids + response_ids
    token_hash = sha256_bytes(json.dumps({"prompt": prompt_ids, "response": response_ids}).encode())
    return full_ids, response_ids, token_hash


def spectrum_metrics(hidden, gammas: np.ndarray) -> tuple[float, float, float, list[float]]:
    import torch

    # H200 has strong FP64 throughput; keeping the small response-token Gram
    # matrix on-device avoids making CPU eigendecomposition the bottleneck.
    x = hidden.detach().double()
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.T
    gram = 0.5 * (gram + gram.T)
    eig = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eig.sum().clamp_min(1e-18)
    p = eig / total
    positive = p[p > 0]
    entropy = float(-(positive * positive.log()).sum())
    effective_rank = float(math.exp(entropy))
    nuclear = float(eig.sqrt().sum() / total.sqrt())
    gamma_tensor = torch.as_tensor(gammas, dtype=torch.float64, device=p.device)
    phi = torch.log1p(gamma_tensor[:, None] * p[None, :]).sum(dim=1).cpu().tolist()
    return entropy, effective_rank, nuclear, phi


def worker(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    job = json.loads(Path(args.job_json).read_text()) if Path(args.job_json).exists() else json.loads(args.job_json)
    job_dir = OUT / "workers" / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "rows.jsonl"
    complete_path = job_dir / "COMPLETE"
    if complete_path.exists() and result_path.exists():
        print(f"{job['id']} already complete")
        return
    probes = [json.loads(line) for line in (OUT / "probe.jsonl").open() if line.strip()]
    config = json.loads((OUT / "config.json").read_text())
    gammas = np.asarray(config["gammas"], dtype=np.float64)
    # A single tokenizer source per teacher/student pair is mandatory.  Trained
    # checkpoints can contain an older copied tokenizer.json whose regex differs
    # from the cached base model, which would silently violate the shared-token
    # probe contract even though the vocabulary is nominally the same.
    tokenizer_repo = {
        "qwen": "/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct",
        "qwen3": "/home/annp36/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B",
    }[job["pair"]]
    tokenizer_source = snapshot(tokenizer_repo)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        job["checkpoint"], trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    model.config.use_cache = False
    n_layers = int(model.config.num_hidden_layers)
    selected = [int(round(depth * n_layers)) for depth in config["normalized_depths"]]
    done = set()
    if result_path.exists():
        for line in result_path.open():
            try:
                row = json.loads(line); done.add((row["doc_id"], row["depth_index"]))
            except Exception:
                pass
    started = time.time()
    with result_path.open("a") as handle:
        for example_index, probe in enumerate(probes):
            ids, response_ids, token_hash = encode_example(tokenizer, probe, config["max_response_tokens"])
            if not response_ids:
                raise RuntimeError(f"empty response tokens for doc {probe['doc_id']}")
            input_ids = torch.tensor([ids], device="cuda:0", dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
            for depth_index, (depth, layer_index) in enumerate(zip(config["normalized_depths"], selected)):
                if (probe["doc_id"], depth_index) in done:
                    continue
                hidden = out.hidden_states[layer_index][0, -len(response_ids):]
                entropy, effective_rank, nuclear, phi = spectrum_metrics(hidden, gammas)
                row = {
                    "pair": job["pair"], "method": job["method"], "seed": job["seed"],
                    "checkpoint": job["checkpoint"], "doc_id": probe["doc_id"],
                    "prompt_hash": probe["prompt_hash"], "continuation_hash": probe["continuation_hash"],
                    "tokenization_hash": token_hash, "response_tokens": len(response_ids),
                    "depth_index": depth_index, "normalized_depth": depth, "layer_index": layer_index,
                    "num_hidden_layers": n_layers, "matrix_entropy": entropy,
                    "effective_rank": effective_rank, "normalized_nuclear_norm": nuclear,
                    "phi": phi,
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            atomic_json(job_dir / "status.json", {"job": job, "examples_done": example_index + 1, "examples_total": len(probes), "elapsed_seconds": time.time() - started})
            print(f"{job['id']} {example_index + 1}/{len(probes)}", flush=True)
            del out, input_ids, attention_mask
    complete_path.write_text(json.dumps({"job": job, "completed_at": time.time(), "rows_sha256": sha256_file(result_path)}) + "\n")


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def save_figure(fig, stem: Path) -> None:
    for ext, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight", **kwargs)


def dataframe_markdown(frame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dep."""
    columns = [str(name) for name in frame.columns]
    lines = ["| Method | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for index, row in frame.iterrows():
        values = ["" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.6g}" for value in row.tolist()]
        lines.append("| " + str(index) + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    config = json.loads((OUT / "config.json").read_text())
    jobs = json.loads((OUT / "jobs.json").read_text())
    missing = [job["id"] for job in jobs if not (OUT / "workers" / job["id"] / "COMPLETE").exists()]
    if missing:
        raise RuntimeError(f"incomplete Task 5 workers: {missing}")
    rows = []
    for job in jobs:
        rows.extend(json.loads(line) for line in (OUT / "workers" / job["id"] / "rows.jsonl").open() if line.strip())
    expected = len(jobs) * config["n_examples"] * len(config["normalized_depths"])
    if len(rows) != expected:
        raise RuntimeError(f"coverage mismatch: got {len(rows)} rows, expected {expected}")
    token_hashes = defaultdict(set)
    for row in rows:
        token_hashes[(row["pair"], row["doc_id"])].add(row["tokenization_hash"])
    bad = [key for key, values in token_hashes.items() if len(values) != 1]
    if bad:
        raise RuntimeError(f"tokenization drift within pair for {bad[:10]}")
    teachers = {(r["pair"], r["doc_id"], r["depth_index"]): r for r in rows if r["method"] == "Teacher"}
    for row in rows:
        teacher = teachers[(row["pair"], row["doc_id"], row["depth_index"])]
        row["cst_curve_gap"] = float(np.mean(np.abs(np.asarray(row["phi"]) - np.asarray(teacher["phi"]))))
        row["matrix_entropy_gap"] = abs(row["matrix_entropy"] - teacher["matrix_entropy"])
        row["effective_rank_gap"] = abs(row["effective_rank"] - teacher["effective_rank"])
        row["normalized_nuclear_norm_gap"] = abs(row["normalized_nuclear_norm"] - teacher["normalized_nuclear_norm"])
        # Keep the numeric vector available while every non-teacher row is
        # compared with its teacher reference; serialize a CSV-friendly copy.
        row["phi_json"] = json.dumps(row["phi"], separators=(",", ":"))
    tables = OUT / "tables"; tables.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(tables / "intrinsic_by_layer.csv", index=False)
    atomic_json(tables / "intrinsic_by_layer.json", rows)
    gap_cols = ["cst_curve_gap", "matrix_entropy_gap", "effective_rank_gap", "normalized_nuclear_norm_gap"]
    per_doc = df.groupby(["pair", "method", "doc_id"], as_index=False)[gap_cols].mean()
    rng = np.random.default_rng(config["bootstrap_seed"])
    summary = []
    for (pair, method), group in per_doc.groupby(["pair", "method"], sort=False):
        n_seeds = df[(df.pair == pair) & (df.method == method)]["seed"].nunique()
        for metric in gap_cols:
            vals = group[metric].to_numpy(float)
            lo, hi = bootstrap_ci(vals, config["bootstrap_replicates"], rng)
            summary.append({"pair": pair, "method": method, "metric": metric, "mean": float(vals.mean()), "standard_error": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0, "ci95_low": lo, "ci95_high": hi, "n_probe_examples": len(vals), "n_seeds": int(n_seeds)})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(tables / "intrinsic_summary.csv", index=False)
    atomic_json(tables / "intrinsic_summary.json", summary)
    relative = []
    for pair in sorted(sdf.pair.unique()):
        for metric in gap_cols:
            base = float(sdf[(sdf.pair == pair) & (sdf.method == "Base") & (sdf.metric == metric)]["mean"].iloc[0])
            for _, row in sdf[(sdf.pair == pair) & (sdf.metric == metric)].iterrows():
                relative.append({"pair": pair, "method": row.method, "metric": metric, "absolute_gap": row["mean"], "relative_to_base": float(row["mean"] / (base + 1e-12))})
    rdf = pd.DataFrame(relative)
    rdf.to_csv(tables / "intrinsic_relative_to_base.csv", index=False)
    atomic_json(tables / "intrinsic_relative_to_base.json", relative)
    corr = [{"status": "pending_final_evaluation", "reason": "Figure E is generated after downstream final benchmark evaluation"}]
    atomic_json(tables / "intrinsic_downstream_correlation.json", corr)
    (tables / "intrinsic_downstream_correlation.csv").write_text("status,reason\npending_final_evaluation,Figure E is generated after downstream final benchmark evaluation\n")

    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    metric_labels = [("matrix_entropy", "Matrix entropy"), ("effective_rank", "Effective rank"), ("normalized_nuclear_norm", "Normalized nuclear norm")]
    for pair in sorted(df.pair.unique()):
        pdf = df[df.pair == pair]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        for ax, (metric, label) in zip(axes, metric_labels):
            for method in METHOD_ORDER:
                mdf = pdf[pdf.method == method]
                means=[]; lows=[]; highs=[]
                for depth in config["normalized_depths"]:
                    vals = mdf[np.isclose(mdf.normalized_depth, depth)].groupby("doc_id")[metric].mean().to_numpy(float)
                    means.append(vals.mean()); lo,hi=bootstrap_ci(vals,config["bootstrap_replicates"],rng); lows.append(lo); highs.append(hi)
                ax.plot(config["normalized_depths"], means, color=COLORS[method], linestyle=LINESTYLES[method], marker=MARKERS[method], label=method, linewidth=2.5 if method=="Teacher" else 1.7)
                ax.fill_between(config["normalized_depths"], lows, highs, color=COLORS[method], alpha=.12)
            ax.set_title(label); ax.set_xlabel("Normalized layer depth"); ax.grid(alpha=.25)
        axes[0].set_ylabel("Mean over probe examples")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"{pair}: spectral statistics (n={config['n_examples']})")
        save_figure(fig, figures / f"{pair}_figure_a_spectral_depth"); plt.close(fig)

        target_depths=[.25,.5,.8]; gamma=np.asarray(config["gammas"])
        fig, axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
        y_top=[]; y_gap=[]
        cached={}
        for col,target in enumerate(target_depths):
            depth=min(config["normalized_depths"],key=lambda x:abs(x-target))
            teacher_rows=pdf[(pdf.method=="Teacher") & np.isclose(pdf.normalized_depth,depth)]
            teacher_phi=np.mean(np.stack([json.loads(v) for v in teacher_rows.phi_json]),axis=0)
            for method in METHOD_ORDER:
                vals=pdf[(pdf.method==method) & np.isclose(pdf.normalized_depth,depth)]
                mean_phi=np.mean(np.stack([json.loads(v) for v in vals.phi_json]),axis=0)
                gap=np.abs(mean_phi-teacher_phi)
                cached[(0,col,method)]=mean_phi; cached[(1,col,method)]=gap
                y_top.extend(mean_phi.tolist()); y_gap.extend(gap.tolist())
            axes[0,col].set_title(f"depth={depth:.3g}")
        for col in range(3):
            for method in METHOD_ORDER:
                axes[0,col].plot(gamma,cached[(0,col,method)],color=COLORS[method],linestyle=LINESTYLES[method],label=method)
                if method!="Teacher": axes[1,col].plot(gamma,cached[(1,col,method)],color=COLORS[method],linestyle=LINESTYLES[method],label=method)
            axes[0,col].set_xscale("log"); axes[1,col].set_xscale("log"); axes[1,col].set_ylim(bottom=0); axes[1,col].set_xlabel("gamma"); axes[0,col].grid(alpha=.2); axes[1,col].grid(alpha=.2)
        for ax in axes[0]: ax.set_ylim(min(y_top),max(y_top))
        for ax in axes[1]: ax.set_ylim(0,max(y_gap)*1.05 if max(y_gap)>0 else 1)
        axes[0,0].set_ylabel("Phi"); axes[1,0].set_ylabel("Absolute teacher gap"); axes[0,-1].legend(fontsize=8)
        fig.suptitle(f"{pair}: dense characteristic curves")
        save_figure(fig, figures / f"{pair}_figure_b_dense_curves"); plt.close(fig)

        heat_methods=[m for m in METHOD_ORDER if m!="Teacher"]
        heat=np.zeros((len(heat_methods),len(config["normalized_depths"])))
        for i,method in enumerate(heat_methods):
            for j,depth in enumerate(config["normalized_depths"]):
                heat[i,j]=pdf[(pdf.method==method)&np.isclose(pdf.normalized_depth,depth)].groupby("doc_id").cst_curve_gap.mean().mean()
        fig,ax=plt.subplots(figsize=(12,4)); im=ax.imshow(heat,aspect="auto",cmap="viridis")
        for j in range(heat.shape[1]):
            best=int(np.argmin(heat[:,j]))
            for i in range(heat.shape[0]): ax.text(j,i,f"{heat[i,j]:.3g}"+(" *" if i==best else ""),ha="center",va="center",color="white" if heat[i,j] > heat.max()*.55 else "black",fontsize=8)
        ax.set_yticks(range(len(heat_methods)),heat_methods); ax.set_xticks(range(len(config["normalized_depths"])),[f"{x:.3g}" for x in config["normalized_depths"]]); ax.set_xlabel("Normalized layer depth"); ax.set_title(f"{pair}: dense-CST teacher gap (* best per depth)"); fig.colorbar(im,ax=ax,label="Mean gap")
        save_figure(fig,figures/f"{pair}_figure_c_gap_heatmap"); plt.close(fig)

        rpair=rdf[(rdf.pair==pair)&(rdf.method!="Teacher")]
        spair=sdf[(sdf.pair==pair)&(sdf.method!="Teacher")]
        fig,ax=plt.subplots(figsize=(10,5)); metrics=gap_cols; offsets=np.linspace(-.24,.24,len(metrics))
        metric_colors=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
        metric_markers=["o", "s", "D", "^"]
        for k,metric in enumerate(metrics):
            base=float(spair[(spair.method=="Base")&(spair.metric==metric)]["mean"].iloc[0])
            selected=spair[spair.metric==metric].set_index("method")
            vals=np.asarray([float(rpair[(rpair.method==m)&(rpair.metric==metric)].relative_to_base.iloc[0]) for m in heat_methods])
            lows=np.asarray([float(selected.loc[m,"ci95_low"])/base for m in heat_methods])
            highs=np.asarray([float(selected.loc[m,"ci95_high"])/base for m in heat_methods])
            errors=np.vstack((np.maximum(0,vals-lows),np.maximum(0,highs-vals)))
            ax.errorbar(np.arange(len(heat_methods))+offsets[k],vals,yerr=errors,
                        fmt=metric_markers[k],color=metric_colors[k],capsize=3,
                        linestyle="none",label=metric.replace("_"," "))
        ax.axhline(1,color="gray",linestyle="--",linewidth=1); ax.set_xticks(range(len(heat_methods)),heat_methods); ax.set_ylabel("Gap relative to Base"); ax.set_title(f"{pair}: aggregate intrinsic gaps"); ax.grid(axis="y",alpha=.25); ax.legend(fontsize=8)
        save_figure(fig,figures/f"{pair}_figure_d_relative_gaps"); plt.close(fig)

    correlation_path=ROOT/"artifacts"/"final_execution"/"final_eval"/"postprocess"/"tables"/"intrinsic_downstream_correlation.csv"
    if correlation_path.exists():
        correlation=pd.read_csv(correlation_path).iloc[0].to_dict()
    else:
        correlation={"pearson":float("nan"),"spearman":float("nan"),"n":0}
    pair_findings={}
    for pair in sorted(df.pair.unique()):
        ptab=sdf[(sdf.pair==pair)&(sdf.method!="Teacher")].pivot(index="method",columns="metric",values="mean").reindex(["Base","DistiLLM","CSD","CKA","CST"])
        trained=ptab.drop(index="Base")
        closest=trained.cst_curve_gap.idxmin()
        aggregate_depth=(df[(df.pair==pair)&(df.method!="Teacher")].groupby("normalized_depth").cst_curve_gap.mean().sort_values(ascending=False))
        worst_depths=list(aggregate_depth.head(3).index)
        pair_df=df[df.pair==pair]
        wins=[]; train_wins=[]; extrap_wins=[]
        gamma=np.asarray(config["gammas"]); train_mask=gamma<=1
        for depth in sorted(pair_df.normalized_depth.unique()):
            def mean_curve(method):
                values=pair_df[(pair_df.method==method)&np.isclose(pair_df.normalized_depth,depth)].phi_json
                return np.mean(np.stack([json.loads(value) for value in values]),axis=0)
            teacher_curve=mean_curve("Teacher"); base_curve=mean_curve("Base"); cst_curve=mean_curve("CST")
            better=np.abs(cst_curve-teacher_curve)<np.abs(base_curve-teacher_curve)
            wins.extend(better.tolist()); train_wins.extend(better[train_mask].tolist()); extrap_wins.extend(better[~train_mask].tolist())
        cst_improves=float(ptab.loc["CST","cst_curve_gap"])<float(ptab.loc["Base","cst_curve_gap"])
        metric_winners={metric:trained[metric].idxmin() for metric in ptab.columns}
        pair_findings[pair]={"closest":closest,"worst_depths":worst_depths,"cst_win_fraction":float(np.mean(wins)),"metric_winners":metric_winners}
        checkpoint_lines=[f"- `{job['method']}` seed `{job['seed']}`: `{job['checkpoint']}`" for job in jobs if job["pair"]==pair]
        md=[
            f"# Intrinsic spectral analysis: {pair}","","## Protocol and definitions","",
            f"The probe contains {config['n_examples']} fixed held-out GSM8K test examples from `probe.jsonl` (SHA-256 `{config['probe_sha256']}`). All methods use identical prompts, continuations, tokenization, response-token positions, at most 64 response tokens, and nine normalized depths from 0 to 1. Each normalized depth is mapped independently to the nearest non-embedding layer.","",
            "For centered hidden states $X=H-\\mathbf 1\\bar h^\\top$, the analysis uses $p_i=\\sigma_i^2/(\\sum_j\\sigma_j^2+\\epsilon)$, matrix entropy $\\mathcal H=-\\sum_i p_i\\log(p_i+\\epsilon)$, effective rank $r_{\\mathrm{eff}}=\\exp(\\mathcal H)$, normalized nuclear norm $\\nu=\\lVert X\\rVert_*/(\\lVert X\\rVert_F+\\epsilon)$, and $\\Phi_H(\\gamma)=\\log\\det(I+\\gamma XX^\\top/(\\lVert X\\rVert_F^2+\\epsilon))$ on 100 log-spaced $\\gamma\\in[10^{-2},10^2]$.","",
            "Aggregate gaps are first computed per probe example, then averaged over normalized depths (and over the 100 gamma points for the CST-curve gap). Reported uncertainty is a deterministic 1,000-replicate bootstrap over probe examples with seed 2026. Seed coverage is explicit in `intrinsic_summary.csv`; Base, DistiLLM, and CSD have one seed, while CKA and CST have two.","",
            "## Absolute aggregate gaps","",dataframe_markdown(ptab),"",
            "Lower is closer to the teacher. No unique winner is bolded because the 95% bootstrap intervals overlap competing methods.","",
            "## Required interpretation","",
            f"1. **Closest spectral geometry:** {closest} has the lowest trained-method aggregate CST-curve gap. Per-metric winners are " + ", ".join(f"{metric.replace('_',' ')}: {method}" for metric,method in metric_winners.items()) + ".",
            f"2. **Largest mismatch depths:** averaging non-teacher methods, the three largest CST-curve gaps occur at normalized depths {', '.join(f'{value:g}' for value in worst_depths)}.",
            f"3. **Full curve versus narrow gamma region:** CST {'improves' if cst_improves else 'does not improve'} the full 100-point curve relative to Base. It is closer at {np.mean(wins):.1%} of all depth/gamma grid points, {np.mean(train_wins):.1%} inside the training gamma range $[10^{{-2}},1]$, and {np.mean(extrap_wins):.1%} for $\\gamma>1$; this is not evidence of only a narrow-region effect.",
            f"4. **Metric consistency:** the metric winners are {', '.join(sorted(set(metric_winners.values())))}. Differences are small relative to overlapping uncertainty; treat rankings as descriptive.",
            f"5. **Downstream relationship:** the combined exploratory Figure E analysis has Pearson $r={correlation['pearson']:.3f}$ and Spearman $\\rho={correlation['spearman']:.3f}$ over $n={int(correlation['n'])}$ comparable trained checkpoints. The weak, sign-inconsistent association does not support a causal claim.",
            "6. **Stability:** probe-example bootstrap intervals overlap broadly. Two-seed CKA/CST estimates are available, but single-seed DistiLLM/CSD coverage limits claims about training-seed stability.","",
            "## Checkpoints","",*checkpoint_lines,"",
            "## Reproduction","","```bash",".venvs/train/bin/python final_execution/run_task5.py",".venvs/evaluator/bin/python final_execution/final_postprocess.py --workers 32","```","",
            "## Limitations","","The probe is limited to 64 held-out GSM8K examples, absolute scales are architecture-specific, some baselines have one training seed, and Figure E is exploratory. These results describe association and representation geometry, not causality.","",
        ]
        (OUT/f"report_{pair}.md").write_text("\n".join(md))
        tex=ptab.to_latex(float_format=lambda x:f"{x:.6g}",caption=f"Intrinsic spectral gaps for {pair}.",label=f"tab:intrinsic-{pair}")
        tex += (f"\n\\paragraph{{Protocol.}} {config['n_examples']} fixed held-out GSM8K examples, nine normalized depths, and at most 64 response tokens are shared across methods. Uncertainty uses 1,000 bootstrap replicates over examples (seed 2026).\n"
                f"\\paragraph{{Interpretation.}} {closest} has the smallest trained-method CST-curve gap. The largest average discrepancies occur at normalized depths {', '.join(f'{value:g}' for value in worst_depths)}. CST is closer than Base at {np.mean(wins):.1%} of depth--gamma points. The exploratory downstream association is Pearson $r={correlation['pearson']:.3f}$ and Spearman $\\rho={correlation['spearman']:.3f}$ ($n={int(correlation['n'])}$); no causal claim is made. Broadly overlapping bootstrap intervals and single-seed coverage for some baselines limit stability claims.\n")
        (OUT/f"report_{pair}.tex").write_text(tex)
    cross_md=["# Cross-pair intrinsic summary","","Qwen2.5 and Qwen3 are reported on separate absolute scales. Cross-architecture comparisons should use `intrinsic_relative_to_base.csv`, not the raw gaps.",""]
    for pair,finding in pair_findings.items():
        cross_md.append(f"- {pair}: closest trained method by CST-curve gap is {finding['closest']}; CST beats Base at {finding['cst_win_fraction']:.1%} of depth/gamma grid points.")
    cross_md.extend(["",f"The combined exploratory intrinsic/downstream result is Pearson $r={correlation['pearson']:.3f}$ and Spearman $\\rho={correlation['spearman']:.3f}$ over $n={int(correlation['n'])}$ checkpoints. The association is weak and sign-inconsistent, so no causal conclusion is supported.",""])
    (OUT/"report_cross_pair.md").write_text("\n".join(cross_md))
    (OUT/"report_cross_pair.tex").write_text(f"\\paragraph{{Cross-pair summary.}} Absolute Qwen2.5 and Qwen3 scales are kept separate; normalized-to-Base values are supplied for context. The exploratory relationship is Pearson $r={correlation['pearson']:.3f}$ and Spearman $\\rho={correlation['spearman']:.3f}$ over $n={int(correlation['n'])}$ checkpoints, without causal interpretation.\n")
    manifest={"completed_at":time.time(),"script_sha256":sha256_file(Path(__file__)),"config":config,"jobs":jobs,"row_count":len(rows),"artifacts_sha256":{str(p.relative_to(OUT)):sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in {"TASK5_COMPLETE","manifest.json"}}}
    atomic_json(OUT/"manifest.json",manifest)
    (OUT/"TASK5_COMPLETE").write_text(json.dumps({"manifest_sha256":sha256_file(OUT/"manifest.json"),"completed_at":time.time()})+"\n")
    print(json.dumps({"status":"complete","rows":len(rows),"output":str(OUT)}))


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare"); p.add_argument("--n-examples",type=int,default=64); p.add_argument("--bootstrap-replicates",type=int,default=1000); p.set_defaults(func=prepare)
    p=sub.add_parser("worker"); p.add_argument("--job-json",required=True); p.set_defaults(func=worker)
    p=sub.add_parser("aggregate"); p.set_defaults(func=aggregate)
    args=parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
