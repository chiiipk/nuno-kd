"""
nnm_variants.py — Loss variants for the NNM ablation study.

Drop-in replacement for nnm_module.compute_nnm_loss. Same signature, different
formula:

       Variant       Description                                                 Teacher ref?
       -------       ---------------------------------------------------------   ------------
   1.  bnm           Maximize the raw nuclear norm of student hidden alone        NO
                     (no centroid, no teacher).
                     L = -||H_s||_* / sqrt(md)

   2.  bnmm          Match the raw nuclear norm of student to teacher's           YES (norm only)
                     (no centroid).
                     L = ||H_t||_* / sqrt(md) - ||H_s||_* / sqrt(md)

   3.  nnm (= nuno)  Centroid-anchored nuclear-norm matching (log-squared).       YES (centroids)
                     Already implemented in nnm_module.compute_nnm_loss.
                     L = (log ||[C_t | H_s]||_* - log ||[C_t | H_t]||_*)^2

   4.  erank         Maximize effective rank of student hidden (entropy of        NO
                     normalized singular values). Differentiable via SVD.
                     L = -erank(H_s)
"""

import math
import torch
import torch.nn as nn

from nnm_module import (
    newton_schulz_polar,
    nuclear_norm_ns,
)
from rank_profile import rank_profile_loss_one_layer
from cst_module import compute_cst_loss


# ═══════════════════════════════════════════════════════════════
#  Variant 1: BNM (no teacher) — maximize student nuclear norm
# ═══════════════════════════════════════════════════════════════

def bnm_loss_one_layer(
    H_s_proj: torch.Tensor,
    H_t:      torch.Tensor,   # UNUSED but kept for API compatibility
    C_t:      torch.Tensor,   # UNUSED
    R:        torch.Tensor,
    lw:       float,
    ns_iters: int = 5,
) -> torch.Tensor:
    """
    BNM: maximize the nuclear norm of student features alone.
    Hidden vectors are L2-normalized per row so nuclear norm is bounded
    by sqrt(N * d_prime); without this normalization, the projector can
    inflate H_s_proj's magnitude to drive ||·||_* to infinity, which
    blows up training (loss ~ -1e13).
    """
    import torch.nn.functional as F
    H_s_proj = H_s_proj.float()
    R        = R.float()

    H_norm = F.normalize(H_s_proj, dim=-1)   # each row → unit L2 norm
    M_s = H_norm @ R                          # [N, d_prime]
    m, n = M_s.shape
    scale = math.sqrt(m * n)
    nn_s = nuclear_norm_ns(M_s, ns_iters) / scale

    # Negative — we MAXIMIZE the nuclear norm, so loss decreases as it grows
    return -lw * nn_s


# ═══════════════════════════════════════════════════════════════
#  Variant 2: BNMM — match raw nuclear norm without centroids
# ═══════════════════════════════════════════════════════════════

def bnmm_loss_one_layer(
    H_s_proj: torch.Tensor,
    H_t:      torch.Tensor,
    C_t:      torch.Tensor,   # UNUSED
    R:        torch.Tensor,
    lw:       float,
    ns_iters: int = 5,
) -> torch.Tensor:
    """
    BNMM: match the raw nuclear norm of student to teacher's.
    Both student and teacher hidden vectors are L2-normalized per row so
    nuclear norms are bounded and the gap loss is on a stable scale.
    """
    import torch.nn.functional as F
    H_s_proj = H_s_proj.float()
    H_t      = H_t.float().detach()
    R        = R.float()

    H_s_norm = F.normalize(H_s_proj, dim=-1)
    H_t_norm = F.normalize(H_t,      dim=-1)

    M_s = H_s_norm @ R
    m, n = M_s.shape
    scale = math.sqrt(m * n)
    nn_s = nuclear_norm_ns(M_s, ns_iters) / scale

    with torch.no_grad():
        M_t = H_t_norm @ R
        nn_t = (nuclear_norm_ns(M_t, ns_iters) / scale).detach()

    # Linear gap: drives nn_s up toward nn_t
    return lw * (nn_t - nn_s)


# ═══════════════════════════════════════════════════════════════
#  Variant 3: NNM (ours)
# ═══════════════════════════════════════════════════════════════

def nnm_loss_one_layer(
    H_s_proj: torch.Tensor,
    H_t:      torch.Tensor,
    C_t:      torch.Tensor,
    R:        torch.Tensor,
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
#  Variant 4: erank — effective rank via SVD
# ═══════════════════════════════════════════════════════════════

def effective_rank(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    erank(M) = exp(H(p)),  p_i = sigma_i / sum_j sigma_j
    Differentiable through SVD.
    """
    s = torch.linalg.svdvals(M)
    s = s / (s.sum() + eps)
    H = -(s * (s + eps).log()).sum()
    return torch.exp(H)


def erank_loss_one_layer(
    H_s_proj: torch.Tensor,
    H_t:      torch.Tensor,   # UNUSED
    C_t:      torch.Tensor,   # UNUSED
    R:        torch.Tensor,
    lw:       float,
    ns_iters: int = 5,        # UNUSED
) -> torch.Tensor:
    H_s_proj = H_s_proj.float()
    R        = R.float()

    M_s = H_s_proj @ R
    er  = effective_rank(M_s)
    return -lw * er


# ═══════════════════════════════════════════════════════════════
#  Dispatch + unified compute
# ═══════════════════════════════════════════════════════════════

_VARIANT_FNS = {
    "bnm":   bnm_loss_one_layer,
    "bnmm":  bnmm_loss_one_layer,
    "nnm":   nnm_loss_one_layer,
    "nuno":  nnm_loss_one_layer,   # alias
    "erank": erank_loss_one_layer,
    "rpt":   rank_profile_loss_one_layer,
    "cst":   None,
}

_PROJECTOR_FREE_VARIANTS = {"rpt", "cst"}


def get_loss_fn(variant: str):
    if variant not in _VARIANT_FNS:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Choose from {list(_VARIANT_FNS)}.")
    return _VARIANT_FNS[variant]


def compute_variant_loss(
    variant: str,
    projectors,
    s_hidden_states,
    t_hidden_states,
    labels,
    student_layer_mapping,
    teacher_layer_mapping,
    t_centroids,
    R,
    layer_weights,
    ns_iters=5,
    cst_options=None,
):
    """
    Drop-in replacement for nnm_module.compute_nnm_loss. Identical signature,
    only the per-layer aggregation formula changes based on `variant`.
    """
    device = labels.device
    total_loss = torch.tensor(0.0, device=device)
    n_layers = len(student_layer_mapping)
    if variant == "cst":
        return compute_cst_loss(
            s_hidden_states, t_hidden_states, labels,
            student_layer_mapping, teacher_layer_mapping,
            **(cst_options or {}),
        )
    if n_layers == 0:
        return total_loss

    flat_mask = (labels != -100).reshape(-1)
    if not flat_mask.any():
        return total_loss

    loss_fn = get_loss_fn(variant)

    for s_lid, t_lid, projector in zip(student_layer_mapping,
                                       teacher_layer_mapping, projectors):
        s_h = s_hidden_states[s_lid]
        t_h = t_hidden_states[t_lid]
        lw  = layer_weights.get(s_lid, 1.0)

        d_s = s_h.shape[-1]
        d_t = t_h.shape[-1]

        s_flat = s_h.reshape(-1, d_s)
        t_flat = t_h.reshape(-1, d_t)
        s_act  = s_flat[flat_mask]
        t_act  = t_flat[flat_mask]

        if variant in _PROJECTOR_FREE_VARIANTS:
            total_loss = total_loss + loss_fn(s_act, t_act, lw)
        else:
            C_t = t_centroids[s_lid]
            proj_dtype = projector.weight.dtype
            s_proj = projector(s_act.to(proj_dtype))
            total_loss = total_loss + loss_fn(
                s_proj, t_act, C_t, R, lw, ns_iters,
            )

    return total_loss / n_layers
