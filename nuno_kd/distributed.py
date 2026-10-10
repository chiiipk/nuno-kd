"""Process-group setup and rank-0 logging."""

import os
import random
from datetime import timedelta

import deepspeed
import numpy as np
import torch
import torch.distributed as dist


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)


def log_rank0(line, path):
    """Append one line to a log file, from rank 0 only."""
    if is_rank0():
        with open(path, "a") as handle:
            handle.write(line + "\n")


def initialize(args):
    """Join the torchrun process group, pin this rank's GPU and seed every RNG."""
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    deepspeed.init_distributed(timeout=timedelta(minutes=300))
    # Each rank draws its own random stream.
    seed = args.seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(args.save, exist_ok=True)
