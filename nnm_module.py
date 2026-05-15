"""
nnm_module.py — Nuclear Norm Matching add-on for DistiLLM trainer (SFT-style).

Reuses model.projectors (Linear d_s -> d_t per layer) created in trainer.
Pre-computes teacher centroids once before training.

Notes:
  - This is the SFT / standard distillation version. There is no chosen/rejected
    split (no DPO). All response tokens (labels != -100) contribute equally.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Newton-Schulz polar factor + nuclear norm
# ═══════════════════════════════════════════════════════════════

_NS_COEFFS = (15 / 8, -10 / 8, 3 / 8)


def newton_schulz_polar(M: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    assert M.ndim == 2
    dtype = M.dtype
    transposed = False
    if M.shape[0] < M.shape[1]:
        M = M.T
        transposed = True
    X = M / (M.norm() + 1e-7)
    a, b, c = _NS_COEFFS
    for _ in range(n_iters):
        A = X.T @ X
        X = a * X + b * (X @ A) + c * (X @ (A @ A))
    if transposed:
        X = X.T
    return X.to(dtype)


class _NuclearNormNS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, n_iters):
        with torch.no_grad():
            P = newton_schulz_polar(M.detach(), n_iters)
        ctx.save_for_backward(P)
        return (P * M).sum()

    @staticmethod
    def backward(ctx, grad_output):
        (P,) = ctx.saved_tensors
        return grad_output * P, None


def nuclear_norm_ns(M, n_iters=5):
    return _NuclearNormNS.apply(M, n_iters)


# ═══════════════════════════════════════════════════════════════
#  Running centroids (used only during pre-pass)
# ═══════════════════════════════════════════════════════════════

class RunningCentroids:
    def __init__(self, K, d, eta, T_dead, device):
        self.K, self.d, self.eta, self.T_dead = K, d, eta, T_dead
        self.device = device
        self.C = torch.randn(K, d, device=device, dtype=torch.float32) * 0.01
        self.dead = torch.zeros(K, device=device, dtype=torch.int32)
        self._step = 0

    @torch.no_grad()
    def update(self, H):
        H = H.to(self.device).float()
        if H.shape[0] == 0:
            return
        self._step += 1
        eta = self.eta / (1 + 0.001 * self._step)
        dists = torch.cdist(H, self.C)
        assign = dists.argmin(dim=1)

        counts = torch.zeros(self.K, device=self.device)
        sums   = torch.zeros(self.K, self.d, device=self.device)
        sums.scatter_add_(0, assign.unsqueeze(1).expand(-1, self.d), H)
        counts.scatter_add_(0, assign, torch.ones(len(H), device=self.device))

        active = counts > 0
        self.C[active] = (1 - eta) * self.C[active] + eta * (sums[active] / counts[active].unsqueeze(1))

        # Dead centroid tracking (vectorized)
        self.dead[active] = 0
        self.dead[~active] += 1

        # Reset dead centroids with random sample from H
        dead_mask = self.dead >= self.T_dead
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            indices = torch.randint(len(H), (n_dead,), device=self.device)
            self.C[dead_mask] = H[indices]
            self.dead[dead_mask] = 0


# ═══════════════════════════════════════════════════════════════
#  Utils
# ═══════════════════════════════════════════════════════════════

def make_R(d: int, d_prime: int, device, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, d_prime, generator=g) / math.sqrt(d_prime)
    return R.to(device).float()


def layer_weight(l: int, L: int, sigma: float = 0.15) -> float:
    return math.exp(-((l / L - 0.5) ** 2) / (2 * sigma ** 2))


def select_mid_layers(n_layers: int, n_mid: int = 4) -> list:
    """40-85% range, dedup."""
    import numpy as np
    lo = max(1, int(0.4 * n_layers))
    hi = min(n_layers, int(0.85 * n_layers))
    if lo >= hi:
        lo = max(0, hi - n_mid)
    return sorted(set(int(i) for i in np.linspace(lo, hi, n_mid, dtype=int).tolist()))


# ═══════════════════════════════════════════════════════════════
#  NNM loss per layer
# ═══════════════════════════════════════════════════════════════

def nnm_loss_one_layer(
    H_s_proj: torch.Tensor,   # student hidden projected to teacher dim, [N, d_t] (with grad)
    H_t:      torch.Tensor,   # teacher hidden [N, d_t]                           (no grad)
    C_t:      torch.Tensor,   # teacher centroids [K, d_t]                        (precomputed, no grad)
    R:        torch.Tensor,   # random projection [d_t, d_prime]
    lw:       float,
    ns_iters: int = 5,
) -> torch.Tensor:
    H_s_proj = H_s_proj.float()
    H_t      = H_t.float().detach()
    C_t      = C_t.float().detach()
    R        = R.float()

    M_s = torch.cat([C_t, H_s_proj], dim=0) @ R
    m, n = M_s.shape
    scale = math.sqrt(m * n)
    nn_s = nuclear_norm_ns(M_s, ns_iters) / scale

    with torch.no_grad():
        M_t = torch.cat([C_t, H_t], dim=0) @ R
        nn_t = (nuclear_norm_ns(M_t, ns_iters) / scale).detach()

    return lw * (torch.log(nn_s + 1e-8) - math.log(nn_t.item() + 1e-8)) ** 2


# ═══════════════════════════════════════════════════════════════
#  Main NNM loss across selected layers (SFT / DistiLLM)
# ═══════════════════════════════════════════════════════════════

def compute_nnm_loss(
    projectors,                 # nn.ModuleList — model.projectors (Linear d_s -> d_t)
    s_hidden_states,            # tuple of [B, T, d_s] (output_hidden_states=True)
    t_hidden_states,            # tuple of [B, T, d_t]
    labels,                     # [B, T] with -100 for prompt/padding tokens
    student_layer_mapping,
    teacher_layer_mapping,
    t_centroids,                # dict[s_lid -> tensor [K, d_t]]
    R,                          # [d_t, d_prime]
    layer_weights,              # dict[s_lid -> float]
    ns_iters=5,
):
    """
    Compute average NNM loss across the selected layer pairs.
    Only positions where labels != -100 (i.e. response tokens) contribute.
    """
    device = labels.device
    total_loss = torch.tensor(0.0, device=device)
    n_layers = len(student_layer_mapping)
    if n_layers == 0:
        return total_loss

    flat_mask = (labels != -100).reshape(-1)
    if not flat_mask.any():
        return total_loss

    for s_lid, t_lid, projector in zip(student_layer_mapping, teacher_layer_mapping, projectors):
        s_h = s_hidden_states[s_lid]
        t_h = t_hidden_states[t_lid]
        C_t = t_centroids[s_lid]
        lw  = layer_weights.get(s_lid, 1.0)

        d_s = s_h.shape[-1]
        d_t = t_h.shape[-1]

        proj_dtype = projector.weight.dtype
        s_flat = s_h.reshape(-1, d_s)
        t_flat = t_h.reshape(-1, d_t)

        s_proj = projector(s_flat[flat_mask].to(proj_dtype))
        t_act  = t_flat[flat_mask]

        total_loss = total_loss + nnm_loss_one_layer(
            s_proj, t_act, C_t, R, lw, ns_iters,
        )

    return total_loss / n_layers


# ═══════════════════════════════════════════════════════════════
#  Pre-compute teacher centroids (called once before training)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def build_teacher_centroids(
    teacher,
    dataloader,
    student_layer_mapping,
    teacher_layer_mapping,
    K=128,
    eta=0.05,
    T_dead=50,
    max_batches=500,
    device=None,
):
    """
    Pre-compute teacher centroids in TEACHER hidden space (d_t).
    Returns dict: s_lid -> frozen [K, d_t] tensor.

    Hidden states are filtered with the LABEL mask (`labels != -100`) — i.e.
    response tokens only — so the centroid distribution matches the hidden
    distribution that NNM loss compares against at training time. Falls back
    to attention_mask only when labels are absent (this introduces prompt-
    token bias but is better than skipping the batch).
    """
    from tqdm import tqdm

    if device is None:
        device = next(teacher.parameters()).device

    teacher.eval()

    # Probe d_t
    sample_batch = next(iter(dataloader))
    sample_ids = sample_batch.get("input_ids")
    if sample_ids is None:
        raise ValueError("Cannot find input_ids in dataloader batch")
    sample_ids  = sample_ids[:1].to(device)
    sample_mask = torch.ones_like(sample_ids)
    out = teacher(sample_ids, attention_mask=sample_mask,
                  output_hidden_states=True, return_dict=True)
    d_t = out.hidden_states[teacher_layer_mapping[0]].shape[-1]

    centroids = {
        s_lid: RunningCentroids(K, d_t, eta, T_dead, device)
        for s_lid in student_layer_mapping
    }

    used_label_mask = False
    used_attn_mask_fallback = False

    for i, batch in enumerate(tqdm(dataloader,
                                   desc="NNM teacher centroid pre-pass",
                                   total=max_batches)):
        if i >= max_batches:
            break
        ids    = batch.get("input_ids")
        mask   = batch.get("attention_mask")
        labels = batch.get("labels")

        if ids is None or mask is None:
            continue
        ids  = ids.to(device)
        mask = mask.to(device)

        # Forward pass uses attention_mask (so padding is ignored inside the
        # transformer); the mask used to *select* hidden positions for
        # centroid updates is the label mask if available.
        out = teacher(ids, attention_mask=mask,
                      output_hidden_states=True, return_dict=True)

        if labels is not None:
            labels = labels.to(device)
            if labels.shape == ids.shape:
                flat_mask = (labels.reshape(-1) != -100)
                used_label_mask = True
            else:
                flat_mask = mask.reshape(-1).bool()
                used_attn_mask_fallback = True
        else:
            flat_mask = mask.reshape(-1).bool()
            used_attn_mask_fallback = True

        if not flat_mask.any():
            continue

        for s_lid, t_lid in zip(student_layer_mapping, teacher_layer_mapping):
            h = out.hidden_states[t_lid].reshape(-1, d_t)[flat_mask]
            centroids[s_lid].update(h)

    if used_attn_mask_fallback and not used_label_mask:
        print("[NNM] WARNING: labels not found in dataloader; centroids were "
              "built using attention_mask (includes prompt tokens). This "
              "biases centroids vs. training-time NNM mask.")
    elif used_attn_mask_fallback:
        print("[NNM] NOTE: some batches fell back to attention_mask due to "
              "shape mismatch.")

    return {s_lid: rc.C.detach().clone() for s_lid, rc in centroids.items()}