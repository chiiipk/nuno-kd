"""
analyze_layers.py — Layer-wise representation quality analysis.

Computes 3 metrics across all layers for 3 models:
    1. Nuclear Norm     — ||Z||_*  (raw + normalized)
    2. Effective Rank   — exp(H(σ_normalized))   (Roy & Vetterli 2007)
    3. Curvature        — avg angle between consecutive token diffs
                          (Hosseini & Fedorenko 2023)

Models compared:
    • Teacher          (Qwen2.5-Math-1.5B-Instruct)
    • Student-base     (Qwen2.5-0.5B, no distillation)
    • Student-distilled (Qwen2.5-0.5B + NNM-KD checkpoint)

Reference:
    "Layer by Layer: Uncovering Hidden Representations in Language Models"
    (Skean et al., ICML 2025, arxiv:2502.02013)

Usage:
    python analyze.py                                  # tsd_kd + prompt+response (defaults)
    python analyze.py --input-mode prompt              # tsd_kd, prompt-only
    python analyze.py --dataset math500                # held-out generalization check
    python analyze.py --n-samples 100 --max-len 1024
    python analyze.py --device cuda:7 --student-ckpt results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492 --n-samples 100 --save-dir ./layer_analysis/new_w_0.2
    python analyze.py --device cuda:7 --student-ckpt results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.5_K128_L4_epoch2_lr1e-4_kdr1.0/2492 --n-samples 100 --save-dir ./layer_analysis/new_w_0.5
"""

import os
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import transformers.integrations.peft as peft_integration


peft_integration.is_peft_available = lambda: False

# ════════════════════════════════════════════════════════════════
#  Metric computations
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def nuclear_norm(Z: torch.Tensor) -> tuple[float, float]:
    """
    Nuclear norm ||Z||_* = sum of singular values.
    Returns (raw_nuc, normalized_nuc) where normalized = nuc / sqrt(N*D).
    """
    Z = Z.float()
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return 0.0, 0.0
    try:
        # Subsample if too large for SVD speed
        if Z.shape[0] > 512:
            idx = torch.randperm(Z.shape[0])[:512]
            Z = Z[idx]
        Z = Z - Z.mean(dim=0, keepdim=True)
        S = torch.linalg.svdvals(Z)
        nuc = S.sum().item()
        n_normalized = nuc / math.sqrt(Z.shape[0] * Z.shape[1])
        return nuc, n_normalized
    except Exception:
        return 0.0, 0.0


@torch.no_grad()
def effective_rank(Z: torch.Tensor) -> float:
    """
    Effective Rank (Roy & Vetterli 2007):
        EffRank = exp(H(p))   where p_i = σ_i / sum_j σ_j  (normalized singulars)
    Lower bound for exp(matrix-based entropy α=1).
    """
    Z = Z.float()
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return 0.0
    try:
        if Z.shape[0] > 512:
            idx = torch.randperm(Z.shape[0])[:512]
            Z = Z[idx]
        S = torch.linalg.svdvals(Z)
        S = S[S > 1e-10]
        if S.numel() == 0:
            return 0.0
        p = S / S.sum()
        H = -(p * (p + 1e-12).log()).sum().item()
        return math.exp(H)
    except Exception:
        return 0.0


@torch.no_grad()
def matrix_entropy_alpha1(Z: torch.Tensor) -> float:
    """
    Matrix-based entropy (α=1, von Neumann-style) on Gram matrix K = Z @ Z.T.
        S_1(Z) = -Σ p_i log(p_i)   where p_i = λ_i(K) / tr(K)
    This is what the paper plots as "prompt entropy".
    """
    Z = Z.float()
    if Z.shape[0] < 2:
        return 0.0
    if Z.shape[0] > 512:
        idx = torch.randperm(Z.shape[0])[:512]
        Z = Z[idx]
    try:
        S = torch.linalg.svdvals(Z)
        eig = S ** 2  # eigenvalues of Gram = singular values squared
        eig = eig[eig > 1e-10]
        if eig.numel() == 0:
            return 0.0
        p = eig / eig.sum()
        return (-p * (p + 1e-12).log()).sum().item()
    except Exception:
        return 0.0


@torch.no_grad()
def curvature(Z: torch.Tensor) -> float:
    """
    Curvature (Hosseini & Fedorenko 2023):
        v_k = z_{k+1} - z_k
        C_avg = mean_k arccos( <v_{k+1}, v_k> / (||v_{k+1}|| ||v_k||) )
    Z: [L, D] for a single sequence.
    """
    Z = Z.float()
    if Z.shape[0] < 3:
        return 0.0
    v = Z[1:] - Z[:-1]                  # [L-1, D]
    v_norm = F.normalize(v, dim=-1)
    cos = (v_norm[1:] * v_norm[:-1]).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(cos).mean().item()


# ════════════════════════════════════════════════════════════════
#  Per-layer metric extraction
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_layer_metrics(model, tokenizer, prompts, device, max_len=256):
    """
    For each layer of `model`, compute mean of:
        - nuclear_norm (raw, normalized)
        - effective_rank
        - matrix_entropy
        - curvature
    over the given prompts.

    Returns dict[metric_name] -> list of values per layer (length = n_layers + 1
    because hidden_states includes embedding layer at index 0).
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_total  = n_layers + 1   # +1 for embedding layer

    metrics = {
        "nuclear_norm":      [[] for _ in range(n_total)],
        "nuclear_norm_norm": [[] for _ in range(n_total)],
        "effective_rank":    [[] for _ in range(n_total)],
        "matrix_entropy":    [[] for _ in range(n_total)],
        "curvature":         [[] for _ in range(n_total)],
    }

    for prompt in tqdm(prompts, desc=f"  layers"):
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len,
        ).to(device)
        if enc["input_ids"].shape[1] < 5:
            continue

        out = model(**enc, output_hidden_states=True, return_dict=True)
        # out.hidden_states: tuple of (n_layers + 1) tensors, each [1, T, D]

        for lid in range(n_total):
            Z = out.hidden_states[lid].squeeze(0).float().cpu()   # [T, D]
            nuc, nuc_n = nuclear_norm(Z)
            er         = effective_rank(Z)
            ent        = matrix_entropy_alpha1(Z)
            curv       = curvature(Z)

            metrics["nuclear_norm"][lid].append(nuc)
            metrics["nuclear_norm_norm"][lid].append(nuc_n)
            metrics["effective_rank"][lid].append(er)
            metrics["matrix_entropy"][lid].append(ent)
            metrics["curvature"][lid].append(curv)

        del out
        torch.cuda.empty_cache()

    # Average across prompts
    return {
        k: [float(np.mean(v)) if len(v) > 0 else 0.0 for v in vals]
        for k, vals in metrics.items()
    }


# ════════════════════════════════════════════════════════════════
#  Model loading helpers
# ════════════════════════════════════════════════════════════════

def load_model_safely(name_or_path: str, device: str, dtype=torch.float16):
    """Load a model with proper error handling."""
    print(f"  Loading {name_or_path} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": device},
            use_safetensors=True,
        )
    except Exception as e:
        print(f"  device_map failed ({e}), falling back to .to(device)")
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  → {n_params:.2f}B params, {model.config.num_hidden_layers} layers")
    return model


# ════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════

def plot_metric(metric_name: str, results: dict, save_dir: str,
                ylabel: str = None, log_scale: bool = False):
    """
    Plot one metric across layers for all 3 models.
    x-axis: layer depth percentage (0-100%) for fair cross-model comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    style_map = {
        "Teacher":            {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "Student-base":       {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "Student-distilled":  {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    }

    for label, layer_vals in results.items():
        n = len(layer_vals)
        x_pct = np.linspace(0, 100, n)
        st = style_map.get(label, {})
        ax.plot(x_pct, layer_vals,
                label=label,
                color=st.get("color"),
                marker=st.get("marker"),
                linestyle=st.get("linestyle"),
                markersize=5, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Layer Depth (%)", fontsize=12)
    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} across Layers",
                 fontsize=13, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{metric_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → saved {out_path}")
    plt.close()


def plot_combined(all_results: dict, save_dir: str):
    """3-panel figure: Nuclear Norm | Effective Rank | Curvature."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = [
        ("nuclear_norm_norm", "Nuclear Norm (normalized)", False),
        ("effective_rank",    "Effective Rank",            False),
        ("curvature",         "Curvature (rad)",           False),
    ]
    style_map = {
        "Teacher":            {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "Student-base":       {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "Student-distilled":  {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    }

    for ax, (mkey, mlabel, log) in zip(axes, metrics_to_plot):
        for model_label, model_res in all_results.items():
            vals = model_res[mkey]
            n = len(vals)
            x_pct = np.linspace(0, 100, n)
            st = style_map.get(model_label, {})
            ax.plot(x_pct, vals,
                    label=model_label,
                    color=st.get("color"),
                    marker=st.get("marker"),
                    linestyle=st.get("linestyle"),
                    markersize=4, linewidth=1.6, alpha=0.85)
        ax.set_xlabel("Layer Depth (%)", fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle("Layer-wise Representation Quality: Teacher vs Student vs Distilled",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(save_dir, "combined.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → saved {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════
#  Data preparation
# ════════════════════════════════════════════════════════════════

def _format_chat_prompt(messages, tokenizer, response: str = None):
    """
    Convert a list-of-messages [{role, content}, ...] into a string the model
    actually sees. Use the tokenizer's chat template if available, otherwise
    concatenate raw contents.

    If `response` is provided (non-empty string), it is appended as an
    assistant turn so the formatted text becomes prompt + response — the
    full sequence the student is trained on during KD.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        return None
    # Keep only the input side (user/system); drop any pre-existing assistant
    # turns so we can attach the canonical teacher response cleanly.
    input_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") in ("user", "system")
    ]
    if not input_msgs:
        input_msgs = messages

    has_response = isinstance(response, str) and len(response.strip()) > 0

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            if has_response:
                full_msgs = input_msgs + [{"role": "assistant", "content": response}]
                # add_generation_prompt=False — we already supplied the assistant turn.
                return tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False,
                )
            return tokenizer.apply_chat_template(
                input_msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback when no chat template is available.
    text = "\n".join(
        m.get("content", "") for m in input_msgs if isinstance(m, dict)
    )
    if has_response:
        text = text + "\n" + response
    return text


def get_eval_prompts(tokenizer, n_samples: int = 50,
                     dataset_name: str = "tsd_kd",
                     input_mode: str = "prompt_response"):
    """
    Get evaluation prompts.

    Args:
        input_mode:
            - "prompt"          : only the user/system prompt (ends with the
                                  assistant generation marker). Matches what the
                                  model sees at inference time before decoding.
            - "prompt_response" : prompt + teacher's response (the full sequence
                                  the student sees under KD teacher forcing).
                                  Recommended for distillation analysis since
                                  the KD loss is computed on response tokens.

    Datasets:
        - gsm8k    : openai/gsm8k                              (prompt only)
        - math500  : HuggingFaceH4/MATH-500                    (prompt only)
        - wikitext : Salesforce/wikitext                       (plain text)
        - tsd_kd   : Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen
                     columns: instruction, prompt[messages], response.
                     Supports both input modes.
    """
    if dataset_name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]

    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        texts = ds["question"]

    elif dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        texts = ds["problem"]

    elif dataset_name == "tsd_kd":
        repo = "Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen"
        # KD generation datasets usually only ship a 'train' split.
        try:
            ds = load_dataset(repo, split="train")
        except Exception:
            ds_all = load_dataset(repo)
            first_split = list(ds_all.keys())[0]
            print(f"  'train' split not found, using '{first_split}'")
            ds = ds_all[first_split]

        cols = set(ds.column_names)
        print(f"  Loaded {repo}  rows={len(ds)}  cols={ds.column_names}")
        print(f"  input_mode = {input_mode!r}")

        use_response = (input_mode == "prompt_response") and ("response" in cols)
        if input_mode == "prompt_response" and "response" not in cols:
            print("  WARNING: 'response' column missing — falling back to prompt-only.")

        texts = []
        # Preferred path: the chat-format 'prompt' column rendered with the
        # tokenizer's chat template, optionally followed by 'response'.
        if "prompt" in cols:
            for row in ds:
                p = row.get("prompt")
                resp = row.get("response") if use_response else None
                if isinstance(p, list):
                    t = _format_chat_prompt(p, tokenizer, response=resp)
                elif isinstance(p, str):
                    t = p + ("\n" + resp if (use_response and resp) else "")
                else:
                    t = None
                if t and len(t.strip()) > 20:
                    texts.append(t)
        # Fallback path: build from 'instruction' (+ optional 'response')
        if len(texts) == 0 and "instruction" in cols:
            for row in ds:
                instr = row.get("instruction")
                if not isinstance(instr, str) or len(instr.strip()) <= 20:
                    continue
                resp = row.get("response") if use_response else None
                t = instr + ("\n" + resp if (use_response and resp) else "")
                texts.append(t)

        if len(texts) == 0:
            raise RuntimeError(
                f"Could not extract any usable prompts from {repo}. "
                f"Columns were: {ds.column_names}"
            )
        print(f"  Extracted {len(texts)} sequences from {repo} "
              f"({'prompt+response' if use_response else 'prompt-only'})")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    np.random.seed(42)
    indices = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    prompts = [texts[i] for i in indices]
    print(f"  Selected {len(prompts)} sequences from {dataset_name}")
    return prompts


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise metric analysis")
    p.add_argument("--teacher-id",  type=str,
                   default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--student-id",  type=str,
                   default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--student-ckpt", type=str,
                   default="results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0/1246",
                   help="Path to distilled student checkpoint.")
    p.add_argument("--n-samples",   type=int, default=500,
                   help="Number of prompts to average over.")
    p.add_argument("--max-len",     type=int, default=1024,
                   help="Max token length per sequence. Bump up for "
                        "prompt+response mode since CoT responses are long.")
    p.add_argument("--dataset", type=str, default="tsd_kd",
                   choices=["wikitext", "gsm8k", "math500", "tsd_kd"])
    p.add_argument("--input-mode", type=str, default="prompt_response",
                   choices=["prompt", "prompt_response"],
                   help="Whether to analyze representations on the prompt only "
                        "or on the full prompt+response sequence (matches what "
                        "the student sees during KD teacher forcing). "
                        "Only affects tsd_kd.")
    p.add_argument("--save-dir",    type=str, default="./layer_analysis")
    p.add_argument("--device",      type=str, default="cuda:7")
    p.add_argument("--skip-distilled", action="store_true",
                   help="Skip distilled student (if checkpoint not available).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  Layer-wise representation analysis")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset}, input_mode={args.input_mode}, "
          f"n_samples={args.n_samples}, max_len={args.max_len}")
    print(f"{'='*70}\n")

    # ── Tokenizer (shared — Qwen family) ────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prompts ─────────────────────────────────────────────────
    print("Preparing eval prompts...")
    prompts = get_eval_prompts(
        tokenizer, args.n_samples, args.dataset, args.input_mode,
    )

    # ── Run each model ──────────────────────────────────────────
    all_results = {}

    print("\n[1/3] Teacher model")
    teacher = load_model_safely(args.teacher_id, device)
    all_results["Teacher"] = compute_layer_metrics(
        teacher, tokenizer, prompts, device, args.max_len,
    )
    del teacher
    torch.cuda.empty_cache()

    print("\n[2/3] Student-base (no distillation)")
    student_base = load_model_safely(args.student_id, device)
    all_results["Student-base"] = compute_layer_metrics(
        student_base, tokenizer, prompts, device, args.max_len,
    )
    del student_base
    torch.cuda.empty_cache()

    if not args.skip_distilled and os.path.isdir(args.student_ckpt):
        print(f"\n[3/3] Student-distilled ({args.student_ckpt})")
        student_d = load_model_safely(args.student_ckpt, device)
        s_tokenizer = AutoTokenizer.from_pretrained(args.student_ckpt, trust_remote_code=True, padding_side="right")
        all_results["Student-distilled"] = compute_layer_metrics(
            student_d, s_tokenizer, prompts, device, args.max_len,
        )
        del student_d
        torch.cuda.empty_cache()
    else:
        print(f"\n[3/3] SKIPPED — checkpoint not found at {args.student_ckpt}")

    # ── Save raw numbers ────────────────────────────────────────
    json_path = os.path.join(args.save_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw metrics → {json_path}")

    # ── Plots ───────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_metric("nuclear_norm",      {k: v["nuclear_norm"]      for k, v in all_results.items()},
                args.save_dir, ylabel="Nuclear Norm ||Z||_*")
    plot_metric("nuclear_norm_norm", {k: v["nuclear_norm_norm"] for k, v in all_results.items()},
                args.save_dir, ylabel="Nuclear Norm / sqrt(N*D)")
    plot_metric("effective_rank",    {k: v["effective_rank"]    for k, v in all_results.items()},
                args.save_dir, ylabel="Effective Rank")
    plot_metric("matrix_entropy",    {k: v["matrix_entropy"]    for k, v in all_results.items()},
                args.save_dir, ylabel="Matrix Entropy (α=1)")
    plot_metric("curvature",         {k: v["curvature"]         for k, v in all_results.items()},
                args.save_dir, ylabel="Curvature (rad)")

    plot_combined(all_results, args.save_dir)

    print(f"\nAll outputs in: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()