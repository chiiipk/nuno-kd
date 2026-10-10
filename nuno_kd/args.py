"""Command-line arguments of the trainer."""

import argparse

import deepspeed

from .losses import AUX_LOSSES
from .losses.output_kd import KD_LOSSES


def get_args():
    parser = argparse.ArgumentParser(
        description="Distil a student LM from a teacher with an output KD loss "
                    "plus an optional auxiliary hidden-state loss."
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model-path", required=True, help="Student model (HF id or path).")
    model.add_argument("--teacher-model-path", required=True, help="Teacher model (HF id or path).")
    model.add_argument("--model-type", default="qwen", help="'qwen' pins Qwen2.5's <|endoftext|> as EOS/pad.")
    model.add_argument("--teacher-fp16", action="store_true", help="Run the teacher in fp16 instead of bf16.")
    model.add_argument("--gradient-checkpointing", action="store_true")

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", required=True, help="Output of `python -m nuno_kd.data.prepare tokenize`.")
    data.add_argument("--train-num", type=int, default=-1, help="Use the first N training sequences (-1: all).")
    data.add_argument("--dev-num", type=int, default=-1, help="Use the first N dev sequences (-1: all).")
    data.add_argument("--num-workers", type=int, default=1)
    data.add_argument("--max-length", type=int, default=1024)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--lr", type=float, required=True)
    optim.add_argument("--weight-decay", type=float, default=1e-2)
    optim.add_argument("--clip-grad", type=float, default=1.0)
    optim.add_argument("--batch-size", type=int, default=1, help="Micro batch per GPU.")
    optim.add_argument("--eval-batch-size", type=int, default=1)
    optim.add_argument("--gradient-accumulation-steps", type=int, default=1)
    optim.add_argument("--epochs", type=int, default=1)
    optim.add_argument("--total-iters", type=int, default=None, help="Optimizer steps; default: all epochs.")
    optim.add_argument("--warmup-ratio", type=float, default=0.0, help="Fraction of optimizer steps for LR warmup.")
    optim.add_argument("--lr-decay-style", choices=["cosine", "constant"], default="cosine")
    optim.add_argument("--seed", type=int, default=1234)

    kd = parser.add_argument_group("output-level KD")
    kd.add_argument("--kd-type", choices=sorted(KD_LOSSES), default="sfkl")
    kd.add_argument("--kd-ratio", type=float, default=1.0,
                    help="loss = (1 - r) * LM loss + r * KD loss.")
    kd.add_argument("--skew-alpha", type=float, default=0.1, help="Mixing weight of sfkl/srkl.")

    aux = parser.add_argument_group("auxiliary hidden-state loss")
    aux.add_argument("--aux-loss", choices=AUX_LOSSES, default=None, help="Default: no auxiliary loss.")
    aux.add_argument("--aux-weight", type=float, default=1.0)
    aux.add_argument("--aux-warmup-steps", type=int, default=0,
                     help="Optimizer steps before the auxiliary loss is switched on.")
    aux.add_argument("--aux-ramp-steps", type=int, default=0,
                     help="Then ramp its weight linearly from 0 over this many steps (0: hard step).")
    aux.add_argument("--aux-max-tokens", type=int, default=64,
                     help="Response tokens sampled per step for the token-geometry losses.")
    aux.add_argument("--aux-num-layers", type=int, default=4)
    aux.add_argument("--aux-layer-min", type=float, default=0.20, help="Relative depth of the first matched layer.")
    aux.add_argument("--aux-layer-max", type=float, default=0.85, help="Relative depth of the last matched layer.")
    aux.add_argument("--cst-gamma-min", type=float, default=1e-2)
    aux.add_argument("--cst-gamma-max", type=float, default=1e2)
    aux.add_argument("--cst-num-gamma-samples", type=int, default=2)
    aux.add_argument("--cst-distance", choices=["l2", "smooth_l1"], default="l2")

    run = parser.add_argument_group("runtime")
    run.add_argument("--save", required=True, help="Output directory: logs and checkpoints.")
    run.add_argument("--log-interval", type=int, default=10)
    run.add_argument("--save-interval", type=int, default=-1, help="Optimizer steps; -1: every epoch, 0: never.")
    run.add_argument("--eval-interval", type=int, default=-1, help="Optimizer steps; -1: every epoch, 0: never.")
    run.add_argument("--resume-ckpt", default=None,
                     help="Epoch-boundary checkpoint saved by this trainer; restores weights only.")
    run.add_argument("--resume-global-step", type=int, default=0,
                     help="Global step at which --resume-ckpt was saved.")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()
