"""Characteristic Spectral Transform (CST) loss.

Phi_H(gamma) = logdet(I + gamma * G), with G the centered, energy-normalized
Gram matrix of the selected hidden states. One Gram and one batched Cholesky
call per layer; no eigendecomposition, SVD, projector or random projection.
"""

import math
import time

import torch
import torch.nn.functional as F

from .structural import sample_response_tokens

EPS = 1e-8
JITTER = 1e-6


def sample_gammas(num_samples, gamma_min, gamma_max, device):
    """Log-uniform samples in [gamma_min, gamma_max]."""
    if num_samples < 1 or not (0 < gamma_min <= gamma_max):
        raise ValueError("CST requires num_samples >= 1 and 0 < gamma_min <= gamma_max")
    u = torch.rand(num_samples, device=device, dtype=torch.float32)
    return torch.exp(math.log(gamma_min) + u * math.log(gamma_max / gamma_min))


def _normalized_smaller_gram(hidden):
    x = hidden.float()
    x = x - x.mean(dim=0, keepdim=True)
    energy = (x * x).sum()
    if not torch.isfinite(energy) or energy.detach().item() <= EPS:
        return None
    base = x @ x.mT if x.shape[0] <= x.shape[1] else x.mT @ x
    base = base / (energy + EPS)
    return 0.5 * (base + base.mT)


def cst_transform(hidden, gamma):
    """Phi_H(gamma) for every gamma, or None for a degenerate (zero-energy) input."""
    base = _normalized_smaller_gram(hidden)
    if base is None:
        return None
    eye = torch.eye(base.shape[0], device=base.device, dtype=torch.float32)
    matrices = eye.unsqueeze(0) + gamma[:, None, None] * base.unsqueeze(0)
    chol, info = torch.linalg.cholesky_ex(matrices)
    if torch.any(info != 0):
        matrices = matrices + JITTER * eye.unsqueeze(0)
        chol, info = torch.linalg.cholesky_ex(matrices)
        if torch.any(info != 0):
            raise RuntimeError(f"CST Cholesky failed with info={info.tolist()}")
    return 2.0 * torch.log(chol.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)


def cst_distance(student_phi, teacher_phi, distance="l2"):
    if distance == "l2":
        return (student_phi - teacher_phi.detach()).square().mean()
    if distance == "smooth_l1":
        return F.smooth_l1_loss(student_phi, teacher_phi.detach())
    raise ValueError(f"unknown CST distance: {distance}")


def cst_loss(student_hidden, teacher_hidden, labels, student_layers, teacher_layers, *,
             max_tokens=64, num_gamma_samples=2, gamma_min=1e-2, gamma_max=1e2, distance="l2"):
    """Multi-layer CST on one shared token subset and gamma sample.

    Returns (loss, diagnostics); diagnostics is None when no layer was usable.
    """
    zero = student_hidden[student_layers[0]].sum() * 0.0
    indices = sample_response_tokens(labels, max_tokens)
    if indices.numel() < 2:
        return zero, None
    gamma = sample_gammas(num_gamma_samples, gamma_min, gamma_max, labels.device)

    started = time.perf_counter()
    losses, student_means, teacher_means = [], [], []
    for s_layer, t_layer in zip(student_layers, teacher_layers):
        student = student_hidden[s_layer]
        teacher = teacher_hidden[t_layer]
        student = student.reshape(-1, student.shape[-1])[indices]
        teacher = teacher.reshape(-1, teacher.shape[-1])[indices].detach()
        student_phi = cst_transform(student, gamma)
        with torch.no_grad():
            teacher_phi = cst_transform(teacher, gamma)
        if student_phi is None or teacher_phi is None:
            continue
        losses.append(cst_distance(student_phi, teacher_phi, distance))
        student_means.append(student_phi.detach().mean())
        teacher_means.append(teacher_phi.detach().mean())
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not losses:
        return zero, None

    s_mean = torch.stack(student_means).mean()
    t_mean = torch.stack(teacher_means).mean()
    diagnostics = {
        "tokens": indices.numel(),
        "phi_student_mean": s_mean.item(),
        "phi_teacher_mean": t_mean.item(),
        "phi_gap_mean": (s_mean - t_mean).abs().item(),
        "gamma_mean": gamma.mean().item(),
        "compute_ms": elapsed_ms,
    }
    return torch.stack(losses).mean(), diagnostics
