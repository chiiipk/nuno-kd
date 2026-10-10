"""Auxiliary hidden-state losses, selected by name."""

from .cst import cst_loss
from .structural import STRUCTURAL_LOSSES, structural_loss

AUX_LOSSES = (*STRUCTURAL_LOSSES, "cst")


def auxiliary_loss(name, student_hidden, teacher_hidden, labels, student_layers, teacher_layers,
                   projectors, args):
    """Return (loss, diagnostics); diagnostics is a dict for CST and None otherwise.

    student_hidden / teacher_hidden map a hidden-state index to a [batch, seq, width] tensor.
    """
    if name == "cst":
        return cst_loss(
            student_hidden, teacher_hidden, labels, student_layers, teacher_layers,
            max_tokens=args.aux_max_tokens,
            num_gamma_samples=args.cst_num_gamma_samples,
            gamma_min=args.cst_gamma_min,
            gamma_max=args.cst_gamma_max,
            distance=args.cst_distance,
        )
    loss = structural_loss(
        name, student_hidden, teacher_hidden, labels, student_layers, teacher_layers,
        projectors, max_tokens=args.aux_max_tokens,
    )
    return loss, None
