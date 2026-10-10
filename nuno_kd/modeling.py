"""Student/teacher loading, decoder-layer lookup and hidden_mse projectors."""

import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .distributed import print_rank0

# Qwen2.5 <|endoftext|>: the end-of-sequence and padding token during training.
QWEN_EOS_ID = 151643


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type == "qwen":
        tokenizer.eos_token_id = QWEN_EOS_ID
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _count_parameters(model):
    return sum(p.nelement() for p in model.parameters())


def load_student(args, device):
    started = time.time()
    # Loaded in fp16; DeepSpeed's bf16 engine casts the weights when it wraps the model.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map={"": device}, torch_dtype=torch.float16
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    print_rank0(f" > student parameters: {_count_parameters(model)}, "
                f"load time {time.time() - started:.1f}s", flush=True)
    return model


def load_teacher(args, device):
    dtype = torch.float16 if args.teacher_fp16 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path, device_map={"": device}, torch_dtype=dtype
    )
    model.eval()
    print_rank0(f" > teacher parameters: {_count_parameters(model)}", flush=True)
    return model


def decoder_layers(model):
    """The decoder blocks of a (possibly DeepSpeed-wrapped) causal LM."""
    while hasattr(model, "module"):
        model = model.module
    return model.model.layers


def select_layers(n_layers, n_select, layer_min, layer_max):
    """Uniformly spaced hidden-state indices in a relative-depth range.

    Index 0 is the embedding output and index i the output of decoder block i-1.
    """
    lo = max(1, int(layer_min * n_layers))
    hi = min(n_layers, int(layer_max * n_layers))
    if lo >= hi:
        lo = max(0, hi - n_select)
    if n_select == 1:
        return [int(round((lo + hi) / 2))]
    return sorted(set(int(i) for i in np.linspace(lo, hi, n_select, dtype=int).tolist()))


def attach_projectors(student, count, teacher_width, seed):
    """Attach learned student-to-teacher-width projectors as `student.projectors`.

    Must run before deepspeed.initialize so the projectors are trained with the
    student. A dedicated CPU generator gives every rank the same initialization.
    """
    param = next(student.parameters())
    student_width = student.config.hidden_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projectors = nn.ModuleList([nn.Linear(student_width, teacher_width, bias=False) for _ in range(count)])
    with torch.no_grad():
        for projector in projectors:
            projector.weight.copy_(torch.randn(teacher_width, student_width, generator=generator) * 0.02)
    student.projectors = projectors.to(device=param.device, dtype=param.dtype)
    print_rank0(f"[aux] attached {count} learned projectors ({student_width} -> {teacher_width})")
