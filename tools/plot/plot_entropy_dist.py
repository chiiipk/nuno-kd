"""
plot_entropy_dist.py — Layer-wise entropy distribution plots from saved hidden states.

Reads the .pt files saved by analyze_layers.py (--save-hidden-states) and plots,
for each layer, a histogram of per-sample matrix entropy values across all models —
one subplot per layer, styled like Figure 4 in the reference image.

Each .pt file has shape [n_layers+1, seq_len, hidden_dim] (no padding tokens).

Directory layout expected (mirrors what analyze_layers.py writes):
    <root>/
        hidden_states/
            teacher/
                sample_0.pt
                sample_1.pt
                ...
            student_base/
                sample_0.pt
                ...
            amid/
            csd/
            nnm_ours/

Usage:
    python plot_entropy_dist.py --hs-root ./layer_analysis/6_method_tsd/hidden_states
    python plot_entropy_dist.py \
        --hs-root   ./results/layer_analysis/6_method_tsd/hidden_states \
        --save-dir  ./results/layer_analysis/plots \
        --bins      100 \
        --alpha     0.55 \
        --max-samples 200 --layers 1 5 10 13 16 19 22 25 28
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
#  Style — must match analyze_layers.py MODEL_STYLE keys
# ════════════════════════════════════════════════════════════════

MODEL_STYLE = {
    "Teacher":      {"color": "#1f77b4", "label": "Teacher"},
    "Student-base": {"color": "#8c564b", "label": "Student-base"},
    "DistiLLM":     {"color": "#ff7f0e", "label": "DistiLLM"},
    "AMID":         {"color": "#9467bd", "label": "AMID"},
    "CSD":          {"color": "#e377c2", "label": "CSD"},
    "NNM (ours)":   {"color": "#2ca02c", "label": "NNM (ours)"},
}

# Map from folder name (lowercased, spaces→underscore, parens removed)
# to display label.  Extend if you add more models.
FOLDER_TO_LABEL = {
    "teacher":      "Teacher",
    "student_base": "Student-base",
    "distillm":     "DistiLLM",
    "amid":         "AMID",
    "csd":          "CSD",
    "nnm_ours":     "NNM (ours)",
}

# Plot order (determines legend order and draw order)
MODEL_ORDER = list(MODEL_STYLE.keys())

# ════════════════════════════════════════════════════════════════
#  Entropy metric (Tối ưu hóa: Batched + Covariance + GPU)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def calculate_batched_entropy(hs: torch.Tensor) -> list[float]:
    """
    Tính Entropy cho tất cả các layer cùng lúc.
    hs shape: [n_layers, seq_len, hidden_dim]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, seq_len, D = hs.shape

    # 1. Lấy mẫu chung cho tất cả các layers (giống hệt logic cũ để không sai lệch)
    if seq_len > 512:
        g = torch.Generator()
        g.manual_seed(seq_len * D)
        idx = torch.randperm(seq_len, generator=g)[:512]
        hs_sub = hs[:, idx, :]  # [L, 512, D]
    else:
        hs_sub = hs

    # Đưa lên GPU (nếu có) để tính cho nhanh, chuyển sang Float
    hs_sub = hs_sub.to(device, dtype=torch.float32)

    # 2. Mean centering
    hs_sub = hs_sub - hs_sub.mean(dim=1, keepdim=True)

    # 3. Covariance Trick: Tính ma trận Z @ Z.T (Batched qua tất cả các layers)
    # Shape sẽ là [L, 512, 512] -> Tính Trị riêng (Eigenvalues) trên ma trận này cực nhanh
    C = torch.matmul(hs_sub, hs_sub.mT)
    
    # Tính Eigenvalues (eigh tối ưu cho ma trận đối xứng)
    # Trị riêng của Z @ Z.T chính là bình phương các giá trị suy biến (Singular values squared)
    eigvals = torch.linalg.eigvalsh(C)  # Shape: [L, seq_len]

    # 4. Tính toán entropy cho từng layer
    # Chuyển về CPU để xử lý số liệu cuối cùng (rất nhẹ)
    eigvals = eigvals.cpu()
    
    sample_ents = []
    for lid in range(L):
        eig = eigvals[lid]
        eig = eig[eig > 1e-10]  # Lọc các giá trị nhiễu hoặc <= 0
        
        if eig.numel() == 0:
            sample_ents.append(float("nan"))
            continue
            
        r = eig.numel()
        if r == 1:
            sample_ents.append(0.0)
            continue
            
        p = eig / eig.sum()
        H = (-p * (p + 1e-12).log()).sum().item()
        H_max = math.log(r)
        
        sample_ents.append(H / H_max)
        
    return sample_ents

# ════════════════════════════════════════════════════════════════
#  Load hidden states
# ════════════════════════════════════════════════════════════════

def load_entropy_per_layer(model_dir: str, max_samples: int = None) -> np.ndarray | None:
    pt_files = sorted(glob(os.path.join(model_dir, "sample_*.pt")))
    if not pt_files:
        return None

    if max_samples is not None:
        pt_files = pt_files[:max_samples]

    all_entropies = []
    for fpath in tqdm(pt_files, desc=f"  {os.path.basename(model_dir)}", leave=False):
        try:
            # Load toàn bộ layer của 1 sample (chỉ dùng CPU lúc load để tránh nghẽn RAM GPU)
            hs = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")
            continue

        # Tính entropy cho tất cả các layer cùng 1 lúc bằng hàm batched
        sample_ents = calculate_batched_entropy(hs)
        all_entropies.append(sample_ents)

    if not all_entropies:
        return None

    return np.array(all_entropies, dtype=np.float32)

# ════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════
def plot_entropy_distributions(
    entropy_data: dict,   # label -> np.ndarray [n_samples, n_layers+1]
    save_dir: str,
    target_layers: list[int] = None,  # Thêm tham số này
    bins: int = 40,
    alpha: float = 0.55,
    filename: str = "entropy_distributions.png",
):
    # Lấy tổng số layer hiện có từ model đầu tiên
    n_layers_plus1 = next(iter(entropy_data.values())).shape[1]
    
    # Nếu không chỉ định, mặc định vẽ tất cả các layer
    if target_layers is None:
        target_layers = list(range(n_layers_plus1))
        
    # Lọc bỏ các layer ID vượt quá giới hạn thực tế (để tránh lỗi Index Out of Bounds)
    target_layers = [lid for lid in target_layers if lid < n_layers_plus1]
    num_plots = len(target_layers)

    if num_plots == 0:
        print("  ⚠  Không có layer nào hợp lệ để vẽ!")
        return

    # Tính toán layout cho Grid (3 cột)
    ncols = 3 if num_plots >= 3 else num_plots
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 2.5),
                             constrained_layout=True)
                             
    # Xử lý trường hợp chỉ có 1 plot thì axes không phải là mảng
    if num_plots == 1:
        axes_flat = [axes]
    elif num_plots <= 3 and nrows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()

    # Xây dựng trục X chung (chỉ dựa trên các layer được chọn để scale đẹp hơn)
    all_vals = []
    for label in entropy_data:
        # Chỉ lấy dữ liệu của các target_layers
        all_vals.append(entropy_data[label][:, target_layers].flatten())
    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[np.isfinite(all_vals)]
    
    global_xmin = float(np.percentile(all_vals, 0.5))
    global_xmax = float(np.percentile(all_vals, 99.5))

    # Lặp qua các layer được chỉ định
    for idx, lid in enumerate(target_layers):
        ax = axes_flat[idx]
        layer_label = f"Layer {lid}"

        layer_xmin, layer_xmax = global_xmax, global_xmin
        for label in MODEL_ORDER:
            if label not in entropy_data:
                continue
            vals = entropy_data[label][:, lid]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            layer_xmin = min(layer_xmin, float(vals.min()))
            layer_xmax = max(layer_xmax, float(vals.max()))

        span = layer_xmax - layer_xmin
        pad = span * 0.05 if span > 0 else 0.01
        xmin = layer_xmin - pad
        xmax = layer_xmax + pad

        for label in reversed(MODEL_ORDER):
            if label not in entropy_data:
                continue
            vals = entropy_data[label][:, lid]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            color = MODEL_STYLE[label]["color"]
            ax.hist(
                vals,
                bins=bins,
                range=(xmin, xmax),
                color=color,
                alpha=alpha,
                edgecolor="none",
                label=label,
            )

        ax.set_title(layer_label, fontsize=10, pad=3)
        ax.set_xlabel("Entropy", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(xmin, xmax)
        ax.spines[["top", "right"]].set_visible(False)

    # Ẩn các ô trống trong grid nếu số lượng subplot không chia hết cho số cột
    for idx in range(num_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_handles = []
    for label in MODEL_ORDER:
        if label not in entropy_data:
            continue
        patch = mpatches.Patch(
            facecolor=MODEL_STYLE[label]["color"],
            alpha=0.7,
            label=label,
        )
        legend_handles.append(patch)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05), # Nhích lên một chút để không đè vào title
    )

    fig.suptitle(
        "Layer-wise Entropy Distributions",
        fontsize=13, fontweight="bold", y=1.08,
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  → saved {out_path}")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot layer-wise entropy distributions from saved hidden states."
    )
    p.add_argument(
        "--hs-root", type=str,
        default="./layer_analysis/hidden_states",
        help="Root directory containing one sub-folder per model "
             "(e.g. teacher/, student_base/, amid/, …).",
    )
    p.add_argument(
        "--save-dir", type=str,
        default="./layer_analysis/plots",
        help="Where to write the output PNG.",
    )
    p.add_argument(
        "--bins", type=int, default=40,
        help="Number of histogram bins per subplot.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.55,
        help="Histogram transparency (0–1).  Lower = more see-through.",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of .pt files loaded per model (useful for quick checks).",
    )
    p.add_argument(
        "--filename", type=str, default="entropy_distributions_9.png",
        help="Output filename.",
    )
    # TÙY CHỌN MỚI: Chỉ định các layer cần vẽ
    p.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Danh sách các layer cần vẽ (ví dụ: --layers 0 16 32). Nếu không truyền, vẽ tất cả.",
    )
    return p.parse_args()

# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Entropy distribution plots")
    print(f"  hs-root    : {args.hs_root}")
    print(f"  save-dir   : {args.save_dir}")
    print(f"  bins       : {args.bins}  alpha: {args.alpha}")
    if args.max_samples:
        print(f"  max-samples: {args.max_samples}")
    print(f"{'='*60}\n")

    if not os.path.isdir(args.hs_root):
        raise FileNotFoundError(
            f"Hidden-state root not found: {args.hs_root}\n"
            "Run analyze_layers.py with --save-hidden-states first."
        )

    # Discover model folders
    subdirs = sorted([
        d for d in os.listdir(args.hs_root)
        if os.path.isdir(os.path.join(args.hs_root, d))
    ])
    print(f"Found {len(subdirs)} model folder(s): {subdirs}\n")

    entropy_data = {}

    for folder in subdirs:
        label = FOLDER_TO_LABEL.get(folder.lower())
        if label is None:
            # Try a fuzzy match: strip common suffixes and retry
            label = FOLDER_TO_LABEL.get(
                folder.lower().replace("-", "_").replace(" ", "_")
            )
        if label is None:
            print(f"  ⚠  Unknown folder '{folder}' — skipping "
                  f"(add it to FOLDER_TO_LABEL to include it).")
            continue

        model_dir = os.path.join(args.hs_root, folder)
        print(f"[{label}]  loading from {model_dir}")
        arr = load_entropy_per_layer(model_dir, max_samples=args.max_samples)
        if arr is None:
            print(f"  ⚠  No .pt files found in {model_dir} — skipping.")
            continue

        print(f"  → loaded {arr.shape[0]} samples × {arr.shape[1]} layers")
        entropy_data[label] = arr

    if not entropy_data:
        print("No data loaded — nothing to plot.")
        return

    print(f"\nPlotting {len(entropy_data)} model(s) …")
    if args.layers:
        print(f"Chỉ vẽ các layers: {args.layers}")
        
    plot_entropy_distributions(
        entropy_data,
        save_dir=args.save_dir,
        target_layers=args.layers, # TRUYỀN THAM SỐ VÀO ĐÂY
        bins=args.bins,
        alpha=args.alpha,
        filename=args.filename,
    )
    print(f"\nDone.  Output in: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()