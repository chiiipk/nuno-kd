"""Projector-free rank-profile distillation losses.

The public loss accepts the same per-layer inputs as NuNo so it can be selected
inside the existing training/evaluation harness.  Unlike NuNo, it intentionally
uses the unprojected student and teacher representations.
"""

import torch


def normalized_token_gram(hidden: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the centered, trace-normalized token Gram matrix."""
    if hidden.ndim != 2:
        raise ValueError(f"expected [n_tokens, hidden_dim], got {tuple(hidden.shape)}")

    hidden = hidden.float()
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    gram = centered @ centered.mT
    gram = 0.5 * (gram + gram.mT)
    trace = gram.diagonal().sum()
    return gram / trace.clamp_min(eps)


def rank_profile(hidden: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cumulative normalized spectrum, ordered from largest to smallest."""
    gram = normalized_token_gram(hidden, eps=eps)
    eigenvalues = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
    # Renormalizing removes tiny numerical drift (and negative round-off).
    spectrum = eigenvalues / eigenvalues.sum().clamp_min(eps)
    return spectrum.cumsum(dim=0)


def rank_profile_loss_one_layer(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    layer_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """One-dimensional Wasserstein/L1-CDF distance between rank profiles."""
    student_profile = rank_profile(student_hidden, eps=eps)
    with torch.no_grad():
        teacher_profile = rank_profile(teacher_hidden.detach(), eps=eps)

    if student_profile.shape != teacher_profile.shape:
        raise ValueError(
            "student and teacher must contain the same selected tokens; "
            f"got {student_profile.numel()} and {teacher_profile.numel()}"
        )
    return layer_weight * (student_profile - teacher_profile).abs().mean()


def compute_rank_profile_loss(
    s_hidden_states,
    t_hidden_states,
    labels,
    student_layer_mapping,
    teacher_layer_mapping,
    *,
    max_tokens: int = 64,
    eps: float = 1e-12,
):
    """Multi-layer RPT using one shared response-token subset per step."""
    if max_tokens < 2:
        raise ValueError(f"max_tokens must be at least 2, got {max_tokens}")
    if len(student_layer_mapping) != len(teacher_layer_mapping):
        raise ValueError("student and teacher layer mappings must have equal length")
    if not student_layer_mapping:
        raise ValueError("RPT requires at least one mapped layer")
    zero = s_hidden_states[student_layer_mapping[0]].sum() * 0.0
    indices = (labels != -100).reshape(-1).nonzero(as_tuple=False).flatten()
    if indices.numel() < 2:
        return zero
    if indices.numel() > max_tokens:
        order = torch.randperm(indices.numel(), device=indices.device)[:max_tokens]
        indices = indices[order]

    losses = []
    for s_lid, t_lid in zip(student_layer_mapping, teacher_layer_mapping):
        s_h = s_hidden_states[s_lid]
        t_h = t_hidden_states[t_lid]
        s_selected = s_h.reshape(-1, s_h.shape[-1])[indices]
        t_selected = t_h.reshape(-1, t_h.shape[-1])[indices]
        losses.append(rank_profile_loss_one_layer(
            s_selected, t_selected, layer_weight=1.0, eps=eps
        ))
    return torch.stack(losses).mean() if losses else zero
