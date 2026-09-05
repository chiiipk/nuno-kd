"""Structural baselines for the CST ablation study."""

import torch
import torch.nn.functional as F

from rank_profile import normalized_token_gram


PROJECTOR_FREE = {"gram", "cka", "normalized_spectrum", "direct_spectrum"}


def one_layer(variant, student, teacher, projector=None, eps=1e-8):
    student = student.float()
    teacher = teacher.float().detach()
    if variant == "hidden_mse":
        if projector is None:
            raise ValueError("hidden_mse requires a learned student-to-teacher projector")
        return F.mse_loss(projector(student.to(projector.weight.dtype)).float(), teacher)

    if variant == "direct_spectrum":
        xs = student - student.mean(0, keepdim=True)
        xt = teacher - teacher.mean(0, keepdim=True)
        # Per-feature token covariance spectra retain activation scale while
        # remaining comparable when student/teacher widths differ.
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
    if variant == "gram":
        return (gs - gt).square().mean()
    if variant == "cka":
        similarity = (gs * gt).sum() / (gs.norm() * gt.norm()).clamp_min(eps)
        return 1.0 - similarity

    es = torch.linalg.eigvalsh(gs).flip(0).clamp_min(0)
    with torch.no_grad():
        et = torch.linalg.eigvalsh(gt).flip(0).clamp_min(0)
    if variant == "normalized_spectrum":
        es = es / es.sum().clamp_min(eps)
        et = et / et.sum().clamp_min(eps)
        return (es - et).square().mean()
    raise ValueError(f"unknown structural ablation: {variant}")


def compute_structural_ablation_loss(
    variant, projectors, s_hidden_states, t_hidden_states, labels,
    student_layer_mapping, teacher_layer_mapping, *, max_tokens=64, eps=1e-8,
):
    if max_tokens < 2:
        raise ValueError("max_tokens must be >= 2")
    indices = (labels != -100).reshape(-1).nonzero(as_tuple=False).flatten()
    zero = s_hidden_states[student_layer_mapping[0]].sum() * 0.0
    if indices.numel() < 2:
        return zero
    if indices.numel() > max_tokens:
        indices = indices[torch.randperm(indices.numel(), device=indices.device)[:max_tokens]]
    losses = []
    for index, (s_lid, t_lid) in enumerate(zip(student_layer_mapping, teacher_layer_mapping)):
        student = s_hidden_states[s_lid]
        teacher = t_hidden_states[t_lid]
        student = student.reshape(-1, student.shape[-1])[indices]
        teacher = teacher.reshape(-1, teacher.shape[-1])[indices]
        projector = None if variant in PROJECTOR_FREE else projectors[index]
        losses.append(one_layer(variant, student, teacher, projector, eps))
    return torch.stack(losses).mean() if losses else zero
