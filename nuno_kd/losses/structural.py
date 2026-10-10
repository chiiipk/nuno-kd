"""Hidden-state structural losses: hidden MSE and token-geometry matching.

Except for hidden_mse, the losses compare token-by-token Gram matrices, so the
student and teacher widths may differ and no projector is needed.
"""

import torch
import torch.nn.functional as F

STRUCTURAL_LOSSES = ("hidden_mse", "gram", "cka", "normalized_spectrum", "direct_spectrum")


def normalized_token_gram(hidden, eps=1e-12):
    """Centered, trace-normalized token Gram matrix of a [tokens, width] tensor."""
    if hidden.ndim != 2:
        raise ValueError(f"expected [n_tokens, hidden_dim], got {tuple(hidden.shape)}")
    hidden = hidden.float()
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    gram = centered @ centered.mT
    gram = 0.5 * (gram + gram.mT)
    return gram / gram.diagonal().sum().clamp_min(eps)


def one_layer(name, student, teacher, projector=None, eps=1e-8):
    student = student.float()
    teacher = teacher.float().detach()
    if name == "hidden_mse":
        return F.mse_loss(projector(student.to(projector.weight.dtype)).float(), teacher)

    if name == "direct_spectrum":
        xs = student - student.mean(0, keepdim=True)
        xt = teacher - teacher.mean(0, keepdim=True)
        # Per-feature token covariance spectra keep the activation scale while
        # staying comparable when the student and teacher widths differ.
        gs_raw = (xs @ xs.mT) / max(student.shape[1], 1)
        with torch.no_grad():
            gt_raw = (xt @ xt.mT) / max(teacher.shape[1], 1)
        es = torch.linalg.eigvalsh(0.5 * (gs_raw + gs_raw.mT)).flip(0).clamp_min(0)
        with torch.no_grad():
            et = torch.linalg.eigvalsh(0.5 * (gt_raw + gt_raw.mT)).flip(0).clamp_min(0)
        return F.smooth_l1_loss(es, et)

    gs = normalized_token_gram(student, eps=eps)
    with torch.no_grad():
        gt = normalized_token_gram(teacher, eps=eps)
    if name == "gram":
        return (gs - gt).square().mean()
    if name == "cka":
        similarity = (gs * gt).sum() / (gs.norm() * gt.norm()).clamp_min(eps)
        return 1.0 - similarity
    if name == "normalized_spectrum":
        es = torch.linalg.eigvalsh(gs).flip(0).clamp_min(0)
        with torch.no_grad():
            et = torch.linalg.eigvalsh(gt).flip(0).clamp_min(0)
        es = es / es.sum().clamp_min(eps)
        et = et / et.sum().clamp_min(eps)
        return (es - et).square().mean()
    raise ValueError(f"unknown structural loss: {name}")


def sample_response_tokens(labels, max_tokens):
    """Flat indices of at most max_tokens random response tokens (labels != -100)."""
    indices = (labels != -100).reshape(-1).nonzero(as_tuple=False).flatten()
    if indices.numel() > max_tokens:
        indices = indices[torch.randperm(indices.numel(), device=indices.device)[:max_tokens]]
    return indices


def structural_loss(name, student_hidden, teacher_hidden, labels, student_layers, teacher_layers,
                    projectors, *, max_tokens=64, eps=1e-8):
    """Mean of the per-layer loss over the matched layers, on one shared token subset."""
    if max_tokens < 2:
        raise ValueError("max_tokens must be >= 2")
    zero = student_hidden[student_layers[0]].sum() * 0.0
    indices = sample_response_tokens(labels, max_tokens)
    if indices.numel() < 2:
        return zero
    losses = []
    for index, (s_layer, t_layer) in enumerate(zip(student_layers, teacher_layers)):
        student = student_hidden[s_layer]
        teacher = teacher_hidden[t_layer]
        student = student.reshape(-1, student.shape[-1])[indices]
        teacher = teacher.reshape(-1, teacher.shape[-1])[indices]
        projector = projectors[index] if name == "hidden_mse" else None
        losses.append(one_layer(name, student, teacher, projector, eps))
    return torch.stack(losses).mean()
