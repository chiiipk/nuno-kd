"""
analyze_layers.py — Layer-wise representation quality analysis.

Computes 5 metrics across all layers for 6 models:
    1. Nuclear Norm (raw)
    2. Nuclear Norm (normalized by sqrt(N*D))
    3. Effective Rank        — exp(H(σ_normalized))   (Roy & Vetterli 2007)
    4. Matrix Entropy α=1    — von Neumann entropy on Gram matrix
    5. Curvature             — avg angle between consecutive token diffs
                               (Hosseini & Fedorenko 2023)

Models compared (6 total):
    • Teacher          (Qwen2.5-14B-Instruct or similar)
    • Student-base     (no distillation)
    • DistiLLM         (SFKL baseline)
    • AMID
    • CSD
    • NNM (ours)

Reference:
    "Layer by Layer: Uncovering Hidden Representations in Language Models"
    (Skean et al., ICML 2025, arxiv:2502.02013)

Usage:
    python analyze_layers.py
    python analyze_layers.py --n-samples 100 --max-len 256 --batch-size 8
    python analyze_layers.py --device cuda:7 --n-samples 100 --save-dir ./layer_analysis/6_method
    python analyze_6_methods.py --device cuda:1 --n-samples 100 --save-dir ./layer_analysis/6_method_tsd \
        --ckpt-amid      results/qwen2.5-1.5B-Instruct#amid/ab_pr_0.5_0.5_4_1e-4 \
        --ckpt-csd       results/qwen2.5-1.5B-Instruct#csd/ab_pr_0.5_0.5_8_1e-4 \
        --ckpt-nnm       results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0 \
        --save-hidden-states --dataset tsd_kd --batch-size 128 --max-len 512 
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
#  Style map — single source of truth for color/marker/order
# ════════════════════════════════════════════════════════════════

MODEL_STYLE = {
    "Teacher":      {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "lw": 2.2, "alpha": 0.95},
    "Student-base": {"color": "#8c564b", "marker": "x", "linestyle": "--", "lw": 1.5, "alpha": 0.75},
    "DistiLLM":     {"color": "#ff7f0e", "marker": "s", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "AMID":         {"color": "#9467bd", "marker": "D", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "CSD":          {"color": "#e377c2", "marker": "v", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "NNM (ours)":   {"color": "#2ca02c", "marker": "^", "linestyle": "-",  "lw": 2.0, "alpha": 0.95},
}
MODEL_ORDER = list(MODEL_STYLE.keys())


# ════════════════════════════════════════════════════════════════
#  Shared helper
# ════════════════════════════════════════════════════════════════

def _subsample(Z: torch.Tensor, max_n: int = 512) -> torch.Tensor:
    """
    Subsample rows of Z to at most max_n rows using a deterministic generator
    seeded on the tensor shape — no global random state is touched.
    """
    if Z.shape[0] <= max_n:
        return Z
    g = torch.Generator()
    g.manual_seed(Z.shape[0] * Z.shape[1])
    idx = torch.randperm(Z.shape[0], generator=g)[:max_n]
    return Z[idx]


def _compute_metrics_for_Z(Z_raw: torch.Tensor) -> dict:
    """
    Compute all 5 metrics for a single [T, D] float32 tensor.

    Z_raw  — raw (uncentered) token representations for one sample at one layer.

    Centering strategy:
        • nuclear_norm, effective_rank, matrix_entropy — need centered Z so that
          singular values reflect variance around the mean, not the mean itself.
          Centering is done once here and shared by all three.
        • curvature — measures angles between consecutive *difference* vectors
          (Z[i+1] - Z[i]); centering has no effect on differences, so Z_sub
          (uncentered) is passed directly.
    """
    Z_sub = _subsample(Z_raw)                          # [min(T,512), D]
    Z     = Z_sub - Z_sub.mean(dim=0, keepdim=True)   # centered, shared

    nuc, nuc_n = nuclear_norm(Z)
    return {
        "nuclear_norm":      nuc,
        "nuclear_norm_norm": nuc_n,
        "effective_rank":    effective_rank(Z),
        "matrix_entropy":    matrix_entropy_alpha1(Z),
        "curvature":         curvature(Z_sub),          # uncentered intentionally
    }


# ════════════════════════════════════════════════════════════════
#  Metric computations
#  All functions receive an already-centered, already-subsampled Z
#  (except curvature which receives uncentered Z_sub).
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def nuclear_norm(Z: torch.Tensor) -> tuple[float, float]:
    """||Z||_* and ||Z||_* / sqrt(N*D).  Z must already be centered."""
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return 0.0, 0.0
    try:
        S = torch.linalg.svdvals(Z)
        nuc = S.sum().item()
        nuc_n = nuc / math.sqrt(Z.shape[0] * Z.shape[1])
        return nuc, nuc_n
    except Exception:
        return 0.0, 0.0


@torch.no_grad()
def effective_rank(Z: torch.Tensor) -> float:
    """exp(H(p)), p_i = σ_i / Σσ_j.  Z must already be centered."""
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return 0.0
    try:
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
    """Von Neumann entropy on Gram K = Z Z^T.  Z must already be centered."""
    if Z.shape[0] < 2:
        return 0.0
    try:
        S = torch.linalg.svdvals(Z)
        eig = S ** 2
        eig = eig[eig > 1e-10]
        if eig.numel() == 0:
            return 0.0
        p = eig / eig.sum()
        return (-p * (p + 1e-12).log()).sum().item()
    except Exception:
        return 0.0


@torch.no_grad()
def curvature(Z: torch.Tensor) -> float:
    """Mean arccos cosine between consecutive token diff vectors.
    Z should NOT be centered (centering does not affect differences)."""
    if Z.shape[0] < 3:
        return 0.0
    v = Z[1:] - Z[:-1]
    v_norm = F.normalize(v, dim=-1)
    cos = (v_norm[1:] * v_norm[:-1]).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(cos).mean().item()


# ════════════════════════════════════════════════════════════════
#  Per-layer metric extraction  — BATCH VERSION
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_layer_metrics(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    max_len: int = 256,
    batch_size: int = 8,
    save_hs_dir: str = None,
) -> dict:
    """
    Return dict[metric] -> list of (n_layers + 1) averaged values.

    Processes prompts in batches of `batch_size`.  Each batch is tokenized
    with right-padding; after the forward pass the attention mask is used to
    strip pad tokens before computing metrics, so each sample only contributes
    its real tokens to the statistics.

    Args:
        model       : HuggingFace CausalLM, already on `device`.
        tokenizer   : corresponding tokenizer.
        prompts     : list of raw text strings.
        device      : torch device string, e.g. "cuda:0".
        max_len     : maximum token length per sample (truncation).
        batch_size  : number of prompts per forward pass.
        save_hs_dir : if given, saves per-sample hidden states as .pt files
                      with shape [n_layers+1, real_len, D] (no padding).
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_total  = n_layers + 1

    metric_keys = ("nuclear_norm", "nuclear_norm_norm",
                   "effective_rank", "matrix_entropy", "curvature")
    metrics = {k: [[] for _ in range(n_total)] for k in metric_keys}

    if save_hs_dir:
        os.makedirs(save_hs_dir, exist_ok=True)

    # Right-pad so that real tokens are contiguous at the start of each row,
    # making it trivial to unpad via attention_mask.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    sample_idx = 0

    for batch in tqdm(batches, desc="  batches", leave=False):
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,       # pad to longest sequence in the batch
        ).to(device)

        # out.hidden_states: tuple of (n_layers+1) tensors, each [B, T_pad, D]
        out = model(**enc, output_hidden_states=True, return_dict=True)

        # Move masks to CPU once — used for unpadding every layer/sample.
        masks = enc["attention_mask"].bool()  # [B, T_pad]

        print("encode done!")

        for b_idx in range(len(batch)):
            real_len = masks[b_idx].sum().item()
            if real_len < 5:
                sample_idx += 1
                continue

            sample_hidden_states = []

            for lid in range(n_total):
                # Unpad: keep only real tokens → [real_len, D]
                Z_raw = out.hidden_states[lid][b_idx]  # [T_pad, D]
                Z_raw = Z_raw[masks[b_idx]].float()          # [real_len, D]

                if save_hs_dir:
                    sample_hidden_states.append(Z_raw.clone())

                m = _compute_metrics_for_Z(Z_raw)
                for k in metric_keys:
                    metrics[k][lid].append(m[k])

            if save_hs_dir:
                # Shape: [n_layers+1, real_len, D]  — no padding tokens saved.
                stacked = torch.stack(sample_hidden_states)
                out_path = os.path.join(save_hs_dir, f"sample_{sample_idx}.pt")
                torch.save(stacked, out_path)

            sample_idx += 1

        del out
        torch.cuda.empty_cache()

    tokenizer.padding_side = orig_padding_side

    return {
        k: [float(np.mean(v)) if v else 0.0 for v in vals]
        for k, vals in metrics.items()
    }


# ════════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════════

def load_model_safely(name_or_path: str, device: str, dtype=torch.float16):
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


def get_tokenizer_for(ckpt_path: str, fallback_id: str):
    """Try to load tokenizer from ckpt folder; fall back to model id."""
    try:
        return AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True, padding_side="right",
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            fallback_id, trust_remote_code=True, padding_side="right",
        )


# ════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════

def plot_metric(metric_name: str, results: dict, save_dir: str,
                ylabel: str = None, log_scale: bool = False):
    """One metric, all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label in MODEL_ORDER:
        if label not in results:
            continue
        layer_vals = results[label]
        n = len(layer_vals)
        x_pct = np.linspace(0, 100, n)
        st = MODEL_STYLE[label]
        ax.plot(x_pct, layer_vals,
                label=label,
                color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                markersize=5, linewidth=st["lw"], alpha=st["alpha"])

    ax.set_xlabel("Layer Depth (%)", fontsize=12)
    ax.set_ylabel(ylabel or metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} across Layers",
                 fontsize=13, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{metric_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → saved {out_path}")
    plt.close()


def plot_combined_1x5(all_results: dict, save_dir: str):
    """1x5 panel — all 5 metrics side by side, 6 curves each."""
    metrics_spec = [
        ("nuclear_norm",      "Nuclear Norm ||Z||_*",      False),
        ("nuclear_norm_norm", "Nuclear Norm (normalized)", False),
        ("effective_rank",    "Effective Rank",            False),
        ("matrix_entropy",    "Matrix Entropy (α=1)",      False),
        ("curvature",         "Curvature (rad)",           False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))

    for ax, (mkey, mlabel, log) in zip(axes, metrics_spec):
        for label in MODEL_ORDER:
            if label not in all_results:
                continue
            vals = all_results[label][mkey]
            n = len(vals)
            x_pct = np.linspace(0, 100, n)
            st = MODEL_STYLE[label]
            ax.plot(x_pct, vals,
                    label=label,
                    color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                    markersize=4, linewidth=st["lw"], alpha=st["alpha"])
        ax.set_xlabel("Layer Depth (%)", fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels), fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.suptitle("Layer-wise Representation Quality (Teacher vs Baselines vs NNM)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = os.path.join(save_dir, "combined_1x5.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → saved {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════
#  Data preparation
# ════════════════════════════════════════════════════════════════

def _format_chat_prompt(messages, tokenizer, response: str = None):
    """
    Convert a list-of-messages [{role, content}, ...] into a string the model
    actually sees.  If `response` is provided it is appended as an assistant
    turn so the formatted text is prompt + response — the full sequence the
    student is trained on during KD.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        return None
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
                return tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False,
                )
            return tokenizer.apply_chat_template(
                input_msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass

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
            - "prompt"          : only the user/system prompt.
            - "prompt_response" : prompt + teacher's response (full KD sequence).
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
    print(f"  Selected {len(prompts)} prompts from {dataset_name}")
    return prompts


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise metric analysis (6 models)")

    # base model IDs
    p.add_argument("--teacher-id",  type=str, default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--student-id",  type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    # distilled checkpoints
    p.add_argument("--ckpt-distillm", type=str,
                   default="results/PLACEHOLDER_distillm/checkpoint",
                   help="Path to DistiLLM checkpoint.")
    p.add_argument("--ckpt-amid",     type=str,
                   default="results/PLACEHOLDER_amid/checkpoint",
                   help="Path to AMID checkpoint.")
    p.add_argument("--ckpt-csd",      type=str,
                   default="results/PLACEHOLDER_csd/checkpoint",
                   help="Path to CSD checkpoint.")
    p.add_argument("--ckpt-nnm",      type=str,
                   default="results/PLACEHOLDER_nnm/checkpoint",
                   help="Path to NNM (ours) checkpoint.")

    # data
    p.add_argument("--n-samples",   type=int, default=100)
    p.add_argument("--max-len",     type=int, default=512)
    p.add_argument("--dataset",     type=str, default="math500",
                   choices=["wikitext", "gsm8k", "math500", "tsd_kd"])

    # output / device
    p.add_argument("--save-dir",    type=str, default="./layer_analysis")
    p.add_argument("--device",      type=str, default="cuda:0")

    # batch size
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Number of prompts per forward pass. "
                        "Reduce if OOM; increase for speed if VRAM allows.")

    # save hidden states
    p.add_argument("--save-hidden-states", action="store_true",
                   help="Save raw hidden states to disk for later reuse.")

    # skip flags
    p.add_argument("--skip-teacher",      action="store_true")
    p.add_argument("--skip-student-base", action="store_true")
    p.add_argument("--skip-distillm",     action="store_true")
    p.add_argument("--skip-amid",         action="store_true")
    p.add_argument("--skip-csd",          action="store_true")
    p.add_argument("--skip-nnm",          action="store_true")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def _ckpt_available(path: str) -> bool:
    return os.path.isdir(path) and "PLACEHOLDER" not in path


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"  Layer-wise representation analysis (6 models)")
    print(f"  Device     : {device}")
    print(f"  Dataset    : {args.dataset}, n_samples={args.n_samples}, max_len={args.max_len}")
    print(f"  Batch size : {args.batch_size}")
    if args.save_hidden_states:
        print(f"  [!] Will save hidden states to disk.")
    print(f"{'='*70}\n")

    # Shared tokenizer (Qwen family default)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_id, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing eval prompts...")
    prompts = get_eval_prompts(tokenizer, args.n_samples, args.dataset)

    # (label, ckpt_path, tokenizer_fallback_id, skip_flag)
    model_configs = [
        ("Teacher",      args.teacher_id,    args.teacher_id,  args.skip_teacher),
        ("Student-base", args.student_id,    args.student_id,  args.skip_student_base),
        # ("DistiLLM",   args.ckpt_distillm, args.student_id,  args.skip_distillm),
        ("AMID",         args.ckpt_amid,     args.student_id,  args.skip_amid),
        ("CSD",          args.ckpt_csd,      args.student_id,  args.skip_csd),
        ("NNM (ours)",   args.ckpt_nnm,      args.student_id,  args.skip_nnm),
    ]

    all_results = {}

    for i, (label, path, tok_fallback, skip) in enumerate(model_configs, start=1):
        print(f"\n[{i}/{len(model_configs)}] {label}")

        if skip:
            print(f"  ⏭  skipped (--skip flag)")
            continue

        is_base = label in ("Teacher", "Student-base")
        if not is_base and not _ckpt_available(path):
            print(f"  ⏭  skipped (checkpoint not found / placeholder: {path})")
            continue

        save_hs_dir = None
        if args.save_hidden_states:
            safe_label = (label.replace(" ", "_")
                               .replace("(", "").replace(")", "").lower())
            save_hs_dir = os.path.join(args.save_dir, "hidden_states", safe_label)

        try:
            model = load_model_safely(path, device)
            tok   = tokenizer if is_base else get_tokenizer_for(path, tok_fallback)
            all_results[label] = compute_layer_metrics(
                model, tok, prompts, device,
                max_len=args.max_len,
                batch_size=args.batch_size,
                save_hs_dir=save_hs_dir,
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ failed to process {label}: {e}")
            continue

    # ── Save raw numbers ──────────────────────────────────────────
    json_path = os.path.join(args.save_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw metrics → {json_path}")

    if not all_results:
        print("No results to plot — exiting.")
        return

    # ── Individual plots ──────────────────────────────────────────
    print("\nGenerating plots...")
    plot_metric("nuclear_norm",
                {k: v["nuclear_norm"]      for k, v in all_results.items()},
                args.save_dir, ylabel="Nuclear Norm ||Z||_*")
    plot_metric("nuclear_norm_norm",
                {k: v["nuclear_norm_norm"] for k, v in all_results.items()},
                args.save_dir, ylabel="Nuclear Norm / sqrt(N*D)")
    plot_metric("effective_rank",
                {k: v["effective_rank"]    for k, v in all_results.items()},
                args.save_dir, ylabel="Effective Rank")
    plot_metric("matrix_entropy",
                {k: v["matrix_entropy"]    for k, v in all_results.items()},
                args.save_dir, ylabel="Matrix Entropy (α=1)")
    plot_metric("curvature",
                {k: v["curvature"]         for k, v in all_results.items()},
                args.save_dir, ylabel="Curvature (rad)")

    # ── Combined 1x5 figure ───────────────────────────────────────
    plot_combined_1x5(all_results, args.save_dir)

    print(f"\nAll outputs in: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()