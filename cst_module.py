"""Characteristic Spectral Transform distillation (training path).

This module intentionally contains no eigendecomposition, SVD, learned
projector, centroid memory, or random feature projection.
"""

import math
import time

import torch
import torch.nn.functional as F


def center_hidden_states(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 2:
        raise ValueError(f"expected [tokens, width], got {tuple(hidden.shape)}")
    hidden = hidden.float()
    return hidden - hidden.mean(dim=0, keepdim=True)


def sample_cst_gammas(
    num_samples: int,
    gamma_min: float,
    gamma_max: float,
    *,
    device,
    fixed_grid=None,
) -> torch.Tensor:
    if fixed_grid:
        gamma = torch.as_tensor(fixed_grid, device=device, dtype=torch.float32)
    else:
        if num_samples < 1 or not (0 < gamma_min <= gamma_max):
            raise ValueError("CST requires q >= 1 and 0 < gamma_min <= gamma_max")
        u = torch.rand(num_samples, device=device, dtype=torch.float32)
        gamma = torch.exp(math.log(gamma_min) + u * math.log(gamma_max / gamma_min))
    if gamma.ndim != 1 or gamma.numel() == 0 or not torch.all(gamma > 0):
        raise ValueError("all CST gamma values must be positive")
    return gamma


def _normalized_smaller_gram(hidden: torch.Tensor, eps: float, center: bool):
    x = center_hidden_states(hidden) if center else hidden.float()
    energy = (x * x).sum()
    if not torch.isfinite(energy) or energy.detach().item() <= eps:
        return None
    base = x @ x.mT if x.shape[0] <= x.shape[1] else x.mT @ x
    base = base / (energy + eps)
    return 0.5 * (base + base.mT)


def compute_cst_transform(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    *,
    eps: float = 1e-8,
    jitter: float = 1e-6,
    center: bool = True,
):
    """Return Phi_H(gamma) using one Gram and one batched Cholesky call."""
    base = _normalized_smaller_gram(hidden, eps, center)
    if base is None:
        return None
    rank_dim = base.shape[0]
    eye = torch.eye(rank_dim, device=base.device, dtype=torch.float32)
    matrices = eye.unsqueeze(0) + gamma[:, None, None] * base.unsqueeze(0)
    chol, info = torch.linalg.cholesky_ex(matrices)
    if torch.any(info != 0):
        matrices = matrices + jitter * eye.unsqueeze(0)
        chol, info = torch.linalg.cholesky_ex(matrices)
        if torch.any(info != 0):
            raise RuntimeError(f"CST Cholesky failed with info={info.tolist()}")
    return 2.0 * torch.log(chol.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)


def cst_distance(student_phi, teacher_phi, distance: str = "l2"):
    gap = student_phi - teacher_phi.detach()
    if distance == "l2":
        return gap.square().mean()
    if distance == "smooth_l1":
        return F.smooth_l1_loss(student_phi, teacher_phi.detach())
    raise ValueError(f"unknown CST distance: {distance}")


def compute_cst_loss(
    s_hidden_states,
    t_hidden_states,
    labels,
    student_layer_mapping,
    teacher_layer_mapping,
    *,
    max_tokens=64,
    num_gamma_samples=2,
    gamma_min=1e-2,
    gamma_max=1e2,
    gamma_sampling="log_uniform",
    distance="l2",
    eps=1e-8,
    jitter=1e-6,
    fixed_gamma_grid=None,
    profile=False,
    center_hidden=True,
):
    """Multi-layer CST with one shared token subset and gamma sample."""
    zero = s_hidden_states[student_layer_mapping[0]].sum() * 0.0
    if gamma_sampling != "log_uniform" and not fixed_gamma_grid:
        raise ValueError("only log_uniform CST gamma sampling is supported")
    valid_indices = (labels != -100).reshape(-1).nonzero(as_tuple=False).flatten()
    if valid_indices.numel() < 2:
        return zero, {"valid_layers": 0}
    if valid_indices.numel() > max_tokens:
        order = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:max_tokens]
        valid_indices = valid_indices[order]

    gamma = sample_cst_gammas(
        num_gamma_samples, gamma_min, gamma_max,
        device=labels.device, fixed_grid=fixed_gamma_grid,
    )
    if profile and labels.is_cuda:
        torch.cuda.synchronize(labels.device)
    started = time.perf_counter()
    losses, student_means, teacher_means, rank_dims = [], [], [], []
    for s_lid, t_lid in zip(student_layer_mapping, teacher_layer_mapping):
        s_h = s_hidden_states[s_lid]
        t_h = t_hidden_states[t_lid]
        s_selected = s_h.reshape(-1, s_h.shape[-1])[valid_indices]
        t_selected = t_h.reshape(-1, t_h.shape[-1])[valid_indices].detach()
        s_phi = compute_cst_transform(
            s_selected, gamma, eps=eps, jitter=jitter, center=center_hidden
        )
        with torch.no_grad():
            t_phi = compute_cst_transform(
                t_selected, gamma, eps=eps, jitter=jitter, center=center_hidden
            )
        if s_phi is None or t_phi is None:
            continue
        losses.append(cst_distance(s_phi, t_phi, distance))
        student_means.append(s_phi.detach().mean())
        teacher_means.append(t_phi.detach().mean())
        rank_dims.append(min(s_selected.shape[0], s_selected.shape[1], t_selected.shape[1]))
    if profile and labels.is_cuda:
        torch.cuda.synchronize(labels.device)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not losses:
        return zero, {"valid_layers": 0}
    loss = torch.stack(losses).mean()
    s_mean = torch.stack(student_means).mean()
    t_mean = torch.stack(teacher_means).mean()
    diagnostics = {
        "valid_layers": len(losses),
        "tokens": valid_indices.numel(),
        "rank_dim": max(rank_dims),
        "phi_student_mean": s_mean.item(),
        "phi_teacher_mean": t_mean.item(),
        "phi_gap_mean": (s_mean - t_mean).abs().item(),
        "gamma_mean": gamma.mean().item(),
        "gamma_min": gamma.min().item(),
        "gamma_max": gamma.max().item(),
        "compute_ms": elapsed_ms,
    }
    return loss, diagnostics
