"""
plot_collapse_dist.py — Layer-wise representation collapse analysis (inter-sequence).

Reads the .pt files saved by analyze_layers.py (--save-hidden-states).
For each sample, extracts a single representative vector per layer via mean pooling,
then stacks all samples → Z_inter shape [N, D].  All metrics are computed on Z_inter,
so they measure whether the MODEL can distinguish different inputs (inter-sequence
diversity), NOT whether tokens within one sentence are diverse (intra-sequence).

Metrics computed:
    1. entropy_norm   — normalized Von Neumann entropy ∈ [0,1]  (low → collapse)
    2. effective_rank — exp(H(σ/Σσ))                            (low → collapse)
    3. sv_decay_top10 — energy in top-10 singular values ∈ [0,1] (high → collapse)
    4. mean_cos_sim   — mean pairwise cosine similarity ∈ [0,1]  (high → collapse)

For each metric, one figure is saved:
    - Grid of subplots (1 per layer), each showing overlapping histograms
      of per-sample scalar values, one color per model.

Wait — inter-sequence metrics are computed on Z_inter [N, D] (one vec per sample),
so they yield ONE scalar per layer per model, not a distribution over samples.
To get a *distribution* to histogram (like Figure 4), we compute the metric on
rolling windows / bootstrap subsets of samples, OR we plot the per-sample
projected value (e.g. projection onto top-1 PC, cosine to mean, etc.).

Concretely, for each sample i and layer l we compute:
    • entropy_norm / effective_rank / sv_decay_top10:
          computed on Z_inter built from ALL samples EXCEPT i is not tractable.
          Instead we use a per-sample proxy:
              - cos_to_mean : cosine similarity of sample_i's repr to the dataset mean
                              (high → sample collapsed toward the mean → collapse)
              - sv_proj_top1: |projection of sample_i onto top-1 PC| / ||sample_i||
                              (high → sample lies on the dominant direction → collapse)
    • mean_cos_sim: per-sample mean cosine similarity to all other samples.

These per-sample scalars produce a distribution over the dataset that can be
histogrammed per layer — matching the Figure 4 style — while measuring
inter-sequence diversity.

In addition, for the dataset-level (single scalar per layer per model) metrics
(entropy_norm, effective_rank, sv_decay_top10, mean_cos_sim on Z_inter),
a separate line-plot figure is generated (like the analyze_layers.py output).

Directory layout:
    <hs-root>/
        teacher/        sample_0.pt  sample_1.pt  ...
        student_base/   ...
        amid/           ...
        csd/            ...
        nnm_ours/       ...

Each .pt file: [n_layers+1, seq_len, hidden_dim]

Usage:
    python plot_collapse_dist.py --hs-root ./layer_analysis/hidden_states
    python plot_entropy.py \
        --hs-root    ./results/layer_analysis/6_method/hidden_states \
        --save-dir   ./results/layer_analysis/plots \
        --pooling    mean \
        --bins       100 \
        --alpha      0.55 \
        --max-samples 200 \
        --cos-subsample 256
"""

import os
import argparse
import math
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════
#  Style
# ════════════════════════════════════════════════════════════════

MODEL_STYLE = {
    "Teacher":      {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "lw": 2.2, "alpha": 0.95},
    "Student-base": {"color": "#8c564b", "marker": "x", "linestyle": "--", "lw": 1.5, "alpha": 0.75},
    "DistiLLM":     {"color": "#ff7f0e", "marker": "s", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "AMID":         {"color": "#9467bd", "marker": "D", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "CSD":          {"color": "#e377c2", "marker": "v", "linestyle": "--", "lw": 1.6, "alpha": 0.85},
    "NNM (ours)":   {"color": "#2ca02c", "marker": "^", "linestyle": "-",  "lw": 2.0, "alpha": 0.95},
}

FOLDER_TO_LABEL = {
    # "teacher":      "Teacher",
    "student_base": "Student-base",
    "distillm":     "DistiLLM",
    "amid":         "AMID",
    "csd":          "CSD",
    "nnm_ours":     "NNM (ours)",
}

MODEL_ORDER = list(MODEL_STYLE.keys())


# ════════════════════════════════════════════════════════════════
#  Dataset-level collapse metrics  (input: Z_inter [N, D])
# ════════════════════════════════════════════════════════════════

def _subsample_rows(Z: torch.Tensor, max_n: int, seed: int = 0) -> torch.Tensor:
    if Z.shape[0] <= max_n:
        return Z
    g = torch.Generator()
    g.manual_seed(seed + Z.shape[0] * Z.shape[1])
    idx = torch.randperm(Z.shape[0], generator=g)[:max_n]
    return Z[idx]


@torch.no_grad()
def _svd_metrics(Z_inter: torch.Tensor) -> dict:
    """
    Compute entropy_norm, effective_rank, sv_decay_top10 on Z_inter [N, D].
    Returns dict of scalars.
    """
    Z = Z_inter.float()
    Z = Z - Z.mean(dim=0, keepdim=True)   # center across samples

    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return {"entropy_norm": float("nan"),
                "effective_rank": float("nan"),
                "sv_decay_top10": float("nan")}
    try:
        S = torch.linalg.svdvals(Z)        # [min(N,D)]
        eig = S ** 2
        eig_pos = eig[eig > 1e-10]
        if eig_pos.numel() == 0:
            return {"entropy_norm": float("nan"),
                    "effective_rank": float("nan"),
                    "sv_decay_top10": float("nan")}

        r = eig_pos.numel()

        # entropy_norm
        p_eig = eig_pos / eig_pos.sum()
        H = (-p_eig * (p_eig + 1e-12).log()).sum().item()
        H_max = math.log(r) if r > 1 else 1.0
        entropy_norm = H / H_max

        # effective_rank (uses σ not σ²)
        p_sv = S[S > 1e-10] / S[S > 1e-10].sum()
        eff_rank = math.exp(-(p_sv * (p_sv + 1e-12).log()).sum().item())

        # sv_decay_top10
        k = min(10, r)
        sv_decay = (eig_pos[:k].sum() / eig_pos.sum()).item()

        return {
            "entropy_norm":    entropy_norm,
            "effective_rank":  eff_rank,
            "sv_decay_top10":  sv_decay,
        }
    except Exception:
        return {"entropy_norm": float("nan"),
                "effective_rank": float("nan"),
                "sv_decay_top10": float("nan")}


@torch.no_grad()
def _mean_cos_sim(Z_inter: torch.Tensor, max_n: int = 256) -> float:
    """Mean pairwise cosine similarity on Z_inter [N, D]."""
    Z = _subsample_rows(Z_inter.float(), max_n)
    Z_norm = F.normalize(Z, dim=-1)
    sim = Z_norm @ Z_norm.T          # [N, N]
    N = sim.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)
    return sim[mask].mean().item()


# ════════════════════════════════════════════════════════════════
#  Per-sample proxy metrics  (produce distributions to histogram)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def _per_sample_cos_to_mean(Z_inter: torch.Tensor) -> torch.Tensor:
    """
    For each sample i, cosine similarity to the dataset mean vector.
    Shape: [N]   high → sample collapsed toward mean → bad
    """
    Z = Z_inter.float()
    mean_vec = Z.mean(dim=0)                          # [D]
    mean_norm = F.normalize(mean_vec.unsqueeze(0), dim=-1)   # [1, D]
    Z_norm = F.normalize(Z, dim=-1)                   # [N, D]
    return (Z_norm * mean_norm).sum(dim=-1)            # [N]


@torch.no_grad()
def _per_sample_sv_proj_top1(Z_inter: torch.Tensor) -> torch.Tensor:
    """
    For each sample i, |projection onto top-1 principal component| / ||x_i||.
    Shape: [N]   high → sample lies along dominant direction → collapse
    """
    Z = Z_inter.float()
    Z_c = Z - Z.mean(dim=0, keepdim=True)
    try:
        _, _, Vt = torch.linalg.svd(Z_c, full_matrices=False)
        top1 = Vt[0]                                  # [D]
        top1 = F.normalize(top1.unsqueeze(0), dim=-1) # [1, D]
        Z_norm = F.normalize(Z_c, dim=-1)
        return (Z_norm * top1).sum(dim=-1).abs()      # [N]
    except Exception:
        return torch.full((Z.shape[0],), float("nan"))


@torch.no_grad()
def _per_sample_mean_cos_to_others(Z_inter: torch.Tensor, max_n: int = 256) -> torch.Tensor:
    """
    For each sample i, mean cosine similarity to all other samples.
    Shape: [N]   high → sample is similar to everyone → collapse
    Uses subsampling if N > max_n to keep it tractable.
    """
    Z = _subsample_rows(Z_inter.float(), max_n)
    Z_norm = F.normalize(Z, dim=-1)
    sim = Z_norm @ Z_norm.T          # [N, N]
    N = sim.shape[0]
    # Zero diagonal, average off-diagonal per row
    sim.fill_diagonal_(0.0)
    return sim.sum(dim=1) / (N - 1)  # [N]


# ════════════════════════════════════════════════════════════════
#  Load hidden states → build Z_inter and per-sample proxies
# ════════════════════════════════════════════════════════════════

def load_model_data(
    model_dir: str,
    pooling: str = "mean",
    max_samples: int = None,
    cos_subsample: int = 256,
) -> dict | None:
    """
    Load all sample_*.pt files, pool each to [D], stack → Z_inter [N, D].
    Returns per-layer dict of:
        dataset_metrics : dict of scalar collapse metrics on Z_inter
        per_sample      : dict of [N] tensors (per-sample proxies)
    Shape of output: list over layers of the above dict.
    """
    pt_files = sorted(glob(os.path.join(model_dir, "sample_*.pt")))
    if not pt_files:
        return None
    if max_samples:
        pt_files = pt_files[:max_samples]

    # First pass: determine n_layers from first file
    hs0 = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    n_layers_plus1 = hs0.shape[0]

    # Collect pooled vectors: layer_vecs[lid] = list of [D] tensors
    layer_vecs = [[] for _ in range(n_layers_plus1)]

    for fpath in tqdm(pt_files, desc=f"  {os.path.basename(model_dir)}", leave=False):
        try:
            hs = torch.load(fpath, map_location="cpu", weights_only=True).float()
            # hs: [n_layers+1, T, D]
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")
            continue
        for lid in range(n_layers_plus1):
            Z = hs[lid]           # [T, D]
            if pooling == "mean":
                vec = Z.mean(dim=0)
            elif pooling == "cls":
                vec = Z[0]
            elif pooling == "last":
                vec = Z[-1]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            layer_vecs[lid].append(vec)

    if not layer_vecs[0]:
        return None

    # Second pass: compute metrics per layer
    results = []
    for lid in tqdm(range(n_layers_plus1), desc="  computing metrics", leave=False):
        Z_inter = torch.stack(layer_vecs[lid])   # [N, D]

        svd_m = _svd_metrics(Z_inter)
        mcs   = _mean_cos_sim(Z_inter, max_n=cos_subsample)

        ps_cos_mean  = _per_sample_cos_to_mean(Z_inter)
        ps_sv_proj   = _per_sample_sv_proj_top1(Z_inter)
        ps_cos_other = _per_sample_mean_cos_to_others(Z_inter, max_n=cos_subsample)

        results.append({
            "dataset": {**svd_m, "mean_cos_sim": mcs},
            "per_sample": {
                "cos_to_mean":        ps_cos_mean.numpy(),   # [N]
                "sv_proj_top1":       ps_sv_proj.numpy(),    # [N]
                "mean_cos_to_others": ps_cos_other.numpy(),  # [N] (possibly < N if subsampled)
            },
        })

    return results   # list[n_layers+1] of dicts


# ════════════════════════════════════════════════════════════════
#  Plotting helpers
# ════════════════════════════════════════════════════════════════

def _make_legend_handles(labels_present):
    return [
        mpatches.Patch(facecolor=MODEL_STYLE[l]["color"], alpha=0.7, label=l)
        for l in MODEL_ORDER if l in labels_present
    ]


def plot_histogram_grid(
    data: dict,          # label -> list[n_layers+1] of np.ndarray [N] (per-sample values)
    title: str,
    xlabel: str,
    save_path: str,
    bins: int = 40,
    alpha: float = 0.55,
    collapse_direction: str = "high",   # "high" = higher value means more collapse
):
    """Generic histogram grid: one subplot per layer."""
    n_layers_plus1 = len(next(iter(data.values())))
    ncols = 3
    nrows = math.ceil(n_layers_plus1 / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.0, nrows * 3.0),
                             constrained_layout=True)
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    # Global x-range from all models and layers
    all_vals = np.concatenate([
        np.concatenate([v for v in layer_list if v is not None])
        for layer_list in data.values()
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        plt.close(fig)
        return
    xmin = float(np.percentile(all_vals, 0.5))
    xmax = float(np.percentile(all_vals, 99.5))
    pad  = (xmax - xmin) * 0.05
    xmin -= pad;  xmax += pad

    for lid in range(n_layers_plus1):
        ax = axes_flat[lid]
        for label in reversed(MODEL_ORDER):
            if label not in data:
                continue
            vals = data[label][lid]
            if vals is None:
                continue
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, range=(xmin, xmax),
                    color=MODEL_STYLE[label]["color"],
                    alpha=alpha, edgecolor="none", label=label)

        ax.set_title(f"layer{lid}", fontsize=9, pad=2)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("Frequency", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(xmin, xmax)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(n_layers_plus1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = _make_legend_handles(set(data.keys()))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.01))

    collapse_note = "↑ = more collapse" if collapse_direction == "high" else "↓ = more collapse"
    fig.suptitle(f"{title}  ({collapse_note})",
                 fontsize=12, fontweight="bold", y=1.03)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → saved {save_path}")
    plt.close(fig)


def plot_line_grid(
    data: dict,          # label -> np.ndarray [n_layers+1]  (scalar per layer)
    title: str,
    ylabel: str,
    save_path: str,
    collapse_direction: str = "low",    # "low" = lower means more collapse
):
    """Single plot: all models, x = layer depth %, y = scalar metric."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for label in MODEL_ORDER:
        if label not in data:
            continue
        vals = data[label]
        x = np.linspace(0, 100, len(vals))
        st = MODEL_STYLE[label]
        ax.plot(x, vals, label=label,
                color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], markersize=5,
                linewidth=st["lw"], alpha=st["alpha"])

    ax.set_xlabel("Layer Depth (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    collapse_note = "↓ = collapse" if collapse_direction == "low" else "↑ = collapse"
    ax.set_title(f"{title}  ({collapse_note})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → saved {save_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Inter-sequence collapse analysis from saved hidden states."
    )
    p.add_argument("--hs-root",      type=str, default="./layer_analysis/hidden_states")
    p.add_argument("--save-dir",     type=str, default="./layer_analysis/plots")
    p.add_argument("--pooling",      type=str, default="mean",
                   choices=["mean", "cls", "last"],
                   help="How to pool token dim → 1 vector per sample.")
    p.add_argument("--bins",         type=int,   default=40)
    p.add_argument("--alpha",        type=float, default=0.55)
    p.add_argument("--max-samples",  type=int,   default=None)
    p.add_argument("--cos-subsample",type=int,   default=256,
                   help="Max samples used for cosine similarity computation.")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print(f"  Inter-sequence collapse plots")
    print(f"  hs-root      : {args.hs_root}")
    print(f"  save-dir     : {args.save_dir}")
    print(f"  pooling      : {args.pooling}")
    print(f"  bins / alpha : {args.bins} / {args.alpha}")
    if args.max_samples:
        print(f"  max-samples  : {args.max_samples}")
    print(f"{'='*65}\n")

    if not os.path.isdir(args.hs_root):
        raise FileNotFoundError(
            f"Hidden-state root not found: {args.hs_root}\n"
            "Run analyze_layers.py with --save-hidden-states first."
        )

    subdirs = sorted([
        d for d in os.listdir(args.hs_root)
        if os.path.isdir(os.path.join(args.hs_root, d))
    ])
    print(f"Found {len(subdirs)} folder(s): {subdirs}\n")

    # ── Load all models ──────────────────────────────────────────
    all_model_data = {}   # label -> list[n_layers+1] of dicts

    for folder in subdirs:
        label = FOLDER_TO_LABEL.get(folder.lower()) or \
                FOLDER_TO_LABEL.get(folder.lower().replace("-", "_").replace(" ", "_"))
        if label is None:
            print(f"  ⚠  Unknown folder '{folder}' — add to FOLDER_TO_LABEL to include.")
            continue

        model_dir = os.path.join(args.hs_root, folder)
        print(f"[{label}]  {model_dir}")
        result = load_model_data(
            model_dir,
            pooling=args.pooling,
            max_samples=args.max_samples,
            cos_subsample=args.cos_subsample,
        )
        if result is None:
            print(f"  ⚠  No .pt files found — skipping.")
            continue

        n_layers_plus1 = len(result)
        print(f"  → {n_layers_plus1} layers, {len(glob(os.path.join(model_dir, 'sample_*.pt')))} samples")
        all_model_data[label] = result

    if not all_model_data:
        print("No data loaded — exiting.")
        return

    # ── Reorganize data for plotting ─────────────────────────────
    # dataset_series[metric][label] = np.ndarray [n_layers+1]  (one scalar per layer)
    dataset_series = {
        "entropy_norm":    {},
        "effective_rank":  {},
        "sv_decay_top10":  {},
        "mean_cos_sim":    {},
    }
    # per_sample_series[metric][label] = list[n_layers+1] of np.ndarray [N]
    per_sample_series = {
        "cos_to_mean":        {},
        "sv_proj_top1":       {},
        "mean_cos_to_others": {},
    }

    for label, layer_list in all_model_data.items():
        n = len(layer_list)
        for metric in dataset_series:
            dataset_series[metric][label] = np.array(
                [layer_list[lid]["dataset"][metric] for lid in range(n)]
            )
        for metric in per_sample_series:
            per_sample_series[metric][label] = [
                layer_list[lid]["per_sample"][metric] for lid in range(n)
            ]

    # ── 1. Line plots — dataset-level metrics ────────────────────
    print("\n── Dataset-level line plots ──")
    sd = args.save_dir

    plot_line_grid(
        dataset_series["entropy_norm"],
        title="Normalized Entropy (inter-sequence)",
        ylabel="Entropy norm [0,1]",
        save_path=os.path.join(sd, "line_entropy_norm.png"),
        collapse_direction="low",
    )
    plot_line_grid(
        dataset_series["effective_rank"],
        title="Effective Rank (inter-sequence)",
        ylabel="Effective Rank",
        save_path=os.path.join(sd, "line_effective_rank.png"),
        collapse_direction="low",
    )
    plot_line_grid(
        dataset_series["sv_decay_top10"],
        title="SV Decay Top-10 (inter-sequence)",
        ylabel="Energy in top-10 SVs [0,1]",
        save_path=os.path.join(sd, "line_sv_decay_top10.png"),
        collapse_direction="high",
    )
    plot_line_grid(
        dataset_series["mean_cos_sim"],
        title="Mean Pairwise Cosine Similarity (inter-sequence)",
        ylabel="Mean cosine sim [0,1]",
        save_path=os.path.join(sd, "line_mean_cos_sim.png"),
        collapse_direction="high",
    )

    # ── 2. Histogram grids — per-sample proxy distributions ──────
    print("\n── Per-sample distribution histograms ──")

    plot_histogram_grid(
        per_sample_series["cos_to_mean"],
        title="Cosine similarity to dataset mean",
        xlabel="cos(xᵢ, μ)",
        save_path=os.path.join(sd, "hist_cos_to_mean.png"),
        bins=args.bins, alpha=args.alpha,
        collapse_direction="high",
    )
    plot_histogram_grid(
        per_sample_series["sv_proj_top1"],
        title="Projection onto top-1 PC",
        xlabel="|xᵢ · PC₁| / ‖xᵢ‖",
        save_path=os.path.join(sd, "hist_sv_proj_top1.png"),
        bins=args.bins, alpha=args.alpha,
        collapse_direction="high",
    )
    plot_histogram_grid(
        per_sample_series["mean_cos_to_others"],
        title="Mean cosine similarity to other samples",
        xlabel="mean cos(xᵢ, xⱼ)",
        save_path=os.path.join(sd, "hist_mean_cos_to_others.png"),
        bins=args.bins, alpha=args.alpha,
        collapse_direction="high",
    )

    print(f"\nAll outputs in: {sd}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()