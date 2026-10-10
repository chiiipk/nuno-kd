"""Token-level divergences between student and teacher next-token distributions.

Every loss averages over response tokens, i.e. positions whose label is not -100.
"""

import torch
import torch.nn.functional as F


def _masked_mean(per_token, labels):
    mask = (labels != -100).int().view(-1)
    return torch.sum(per_token.view(-1) * mask, dim=0) / torch.sum(mask, dim=0)


def forward_kl(logits, teacher_logits, labels):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    return -_masked_mean(torch.sum(prod_probs, dim=-1), labels)


def reverse_kl(logits, teacher_logits, labels):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    return -_masked_mean(torch.sum(prod_probs, dim=-1), labels)


def skewed_forward_kl(logits, teacher_logits, labels, lam=0.1):
    """KL(p_teacher || lam * p_teacher + (1 - lam) * p_student)."""
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(lam * teacher_probs + (1 - lam) * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    return -_masked_mean(torch.sum(prod_probs, dim=-1), labels)


def skewed_reverse_kl(logits, teacher_logits, labels, lam=0.1):
    """KL(p_student || (1 - lam) * p_teacher + lam * p_student)."""
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    return -_masked_mean(torch.sum(prod_probs, dim=-1), labels)


def js_distance(logits, teacher_logits, labels, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    loss = lam * -_masked_mean(torch.sum(prod_probs, dim=-1), labels)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    return loss + (1 - lam) * -_masked_mean(torch.sum(prod_probs, dim=-1), labels)


def tv_distance(logits, teacher_logits, labels):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    return _masked_mean(torch.sum(prod_probs, dim=-1), labels)


KD_LOSSES = {
    "fkl": forward_kl,
    "rkl": reverse_kl,
    "sfkl": skewed_forward_kl,
    "srkl": skewed_reverse_kl,
    "jsd": js_distance,
    "tvd": tv_distance,
}


def output_kd_loss(kind, logits, teacher_logits, labels, skew_alpha):
    if kind in ("sfkl", "srkl"):
        return KD_LOSSES[kind](logits, teacher_logits, labels, lam=skew_alpha)
    return KD_LOSSES[kind](logits, teacher_logits, labels)
