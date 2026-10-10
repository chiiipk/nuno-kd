"""Distillation trainer: output-level KD plus an optional auxiliary hidden-state loss.

    torchrun --standalone --nproc_per_node N -m nuno_kd.train --deepspeed --deepspeed_config CFG ...

scripts/run_experiment.sh builds the full command from a YAML config. The loss
per micro batch is

    (1 - kd_ratio) * LM loss + kd_ratio * KD loss + w(step) * auxiliary loss

where w ramps from 0 to --aux-weight after --aux-warmup-steps optimizer steps.
"""

import json
import os
import time

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from .args import get_args
from .data.dataset import DistillationDataset
from .distributed import initialize, log_rank0, print_rank0
from .losses import auxiliary_loss
from .losses.output_kd import output_kd_loss
from .modeling import (attach_projectors, decoder_layers, load_student, load_teacher,
                       load_tokenizer, select_layers)

torch.set_num_threads(4)


def build_optimizer(args, model):
    """AdamW with no weight decay on biases; projectors get their own group."""
    decay, no_decay, projector = [], [], []
    for name, param in model.named_parameters():
        if "projector" in name:
            projector.append(param)
        elif "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
    if projector:
        groups.append({"params": projector, "weight_decay": args.weight_decay, "lr": args.lr})
    return AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer):
    warmup_steps = round(args.total_iters * args.warmup_ratio)
    if args.lr_decay_style == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.total_iters
    )


def aux_weight_at(global_step, args):
    """0 during warmup, then a linear ramp to --aux-weight over --aux-ramp-steps."""
    if global_step < args.aux_warmup_steps:
        return 0.0
    if args.aux_ramp_steps <= 0:
        return args.aux_weight
    progress = (global_step - args.aux_warmup_steps) / args.aux_ramp_steps
    return args.aux_weight * min(progress, 1.0)


def load_resume_weights(args, model):
    """Load student weights (and projectors) from an epoch checkpoint.

    The checkpoint holds bf16 module weights only: AdamW moments and the fp32
    master copy were never saved, so the optimizer restarts from zero moments.
    """
    path = os.path.join(args.resume_ckpt, "pytorch_model.bin")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    # Match the checkpoint dtype so no value is rounded through fp16.
    model.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"resume checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    print_rank0(f"[resume] loaded {len(state)} tensors from {path}")


def to_device(model_batch, labels, device):
    return {k: v.to(device) for k, v in model_batch.items()}, labels.to(device)


@torch.no_grad()
def evaluate(args, engine, dataset, device):
    """Mean LM loss on the dev split, logged to log.txt."""
    world = dist.get_world_size()
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False,
                                 rank=dist.get_rank(), num_replicas=world)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                        num_workers=args.num_workers, collate_fn=dataset.collate)
    engine.eval()
    total, batches = 0.0, 0
    for model_batch, labels in loader:
        model_batch, labels = to_device(model_batch, labels, device)
        logits = engine(**model_batch).logits
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        total += loss.item() / world
        batches += 1
    line = f"dev | avg_loss: {total / batches}"
    print_rank0(line)
    log_rank0(line, os.path.join(args.save, "log.txt"))
    return total / batches


def train(args, tokenizer, engine, optimizer, lr_scheduler, datasets, teacher, layers, device):
    world = dist.get_world_size()
    train_set = datasets["train"]
    sampler = DistributedSampler(train_set, shuffle=True, drop_last=True,
                                 rank=dist.get_rank(), num_replicas=world)
    loader = DataLoader(train_set, sampler=sampler, batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=train_set.collate)
    log_path = os.path.join(args.save, "log.txt")
    gas = args.gradient_accumulation_steps

    # `step` counts micro batches, `global_step` optimizer steps; both start at 1.
    step, global_step, start_epoch = 1, 1, 0
    if args.resume_ckpt:
        # Start the next epoch with the counters, LR schedule position and
        # sampler epoch an uninterrupted run would have at that point.
        start_epoch = args.resume_global_step // args.train_iters_per_epoch
        skipped_micro_steps = start_epoch * len(loader)
        step = 1 + skipped_micro_steps
        global_step = 1 + step // gas
        for _ in range(skipped_micro_steps // gas):
            lr_scheduler.step()
        print_rank0(f"[resume] epoch {start_epoch} | step {step} | global step {global_step} | "
                    f"lr {lr_scheduler.get_last_lr()[0]:.4e}")

    evaluate(args, engine, datasets["dev"], device)

    # Forward hooks capture the student's hidden states at the matched layers.
    student_hidden = {}
    student_layers, teacher_layers = layers or ((), ())
    use_aux_loss = args.aux_loss is not None and args.aux_weight > 0
    if use_aux_loss:
        blocks = decoder_layers(engine)

        def capture(layer):
            def hook(module, inputs, output):
                if module.training:
                    student_hidden[layer] = output[0] if isinstance(output, tuple) else output
            return hook

        for layer in student_layers:
            # Hidden-state index i is the output of decoder block i - 1.
            blocks[max(0, layer - 1)].register_forward_hook(capture(layer))
        print_rank0(f"[aux] {args.aux_loss}: warmup={args.aux_warmup_steps} steps, "
                    f"ramp={args.aux_ramp_steps} steps, weight={args.aux_weight}")

    totals = {"loss": 0.0, "kd": 0.0, "aux": 0.0, "time": 0.0}
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        engine.train()
        for model_batch, labels in loader:
            model_batch, labels = to_device(model_batch, labels, device)
            student_hidden.clear()
            torch.cuda.synchronize()
            started = time.time()

            aux_weight = aux_weight_at(global_step, args) if use_aux_loss else 0.0
            use_aux = aux_weight > 0.0

            logits = engine(**model_batch, use_cache=False).logits
            lm_loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1))
            loss = lm_loss
            kd_loss = torch.tensor(0.0, device=device)
            teacher_out = None
            if args.kd_ratio > 0.0 or use_aux:
                with torch.no_grad():
                    teacher_out = teacher(**model_batch, output_hidden_states=use_aux, use_cache=False)
                if args.kd_ratio > 0.0:
                    kd_loss = output_kd_loss(args.kd_type, logits, teacher_out.logits, labels, args.skew_alpha)
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * kd_loss

            aux_loss = torch.tensor(0.0, device=device)
            diagnostics = None
            if use_aux:
                aux_loss, diagnostics = auxiliary_loss(
                    args.aux_loss, student_hidden, teacher_out.hidden_states, labels,
                    student_layers, teacher_layers,
                    getattr(engine.module, "projectors", None), args,
                )
                loss = loss + aux_weight * aux_loss

            engine.backward(loss)
            engine.step()

            dist.all_reduce(loss, dist.ReduceOp.SUM)
            dist.all_reduce(kd_loss, dist.ReduceOp.SUM)
            totals["loss"] += loss.item() / world
            totals["kd"] += kd_loss.item() / world
            if use_aux:
                dist.all_reduce(aux_loss, dist.ReduceOp.SUM)
                totals["aux"] += aux_loss.item() / world
            torch.cuda.synchronize()
            elapsed = time.time() - started
            totals["time"] += elapsed

            at_step_boundary = step % gas == 0
            if at_step_boundary and global_step % args.log_interval == 0:
                denom = args.log_interval * gas
                line = (
                    "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
                    "loss: {:.4f} | ds_loss: {:.4f} | {}_loss: {:.4f} | lr: {:.4e} | "
                    "scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}"
                ).format(
                    epoch, step, args.total_iters * gas, global_step, args.total_iters,
                    totals["loss"] / denom, totals["kd"] / denom, args.aux_loss, totals["aux"] / denom,
                    lr_scheduler.get_last_lr()[0], getattr(optimizer, "cur_scale", 0),
                    elapsed, totals["time"] / args.log_interval,
                )
                if diagnostics:
                    line += (" | phi_s: {phi_student_mean:.4f} | phi_t: {phi_teacher_mean:.4f} | "
                             "phi_gap: {phi_gap_mean:.4f} | gamma: {gamma_mean:.3g} | "
                             "cst_tokens: {tokens} | cst_ms: {compute_ms:.1f}").format(**diagnostics)
                print_rank0(line)
                log_rank0(line, log_path)
                totals = dict.fromkeys(totals, 0.0)

            if at_step_boundary and args.save_interval and global_step % args.save_interval == 0:
                save_dir = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir, exist_ok=True)
                    print_rank0(f"Model save to {save_dir}")
                    tokenizer.save_pretrained(save_dir)
                    engine.module.save_pretrained(save_dir, safe_serialization=False)
                dist.barrier()

            if at_step_boundary and args.eval_interval and global_step % args.eval_interval == 0:
                evaluate(args, engine, datasets["dev"], device)
                engine.train()

            step += 1
            if step % gas == 0:
                global_step += 1
            if global_step > args.total_iters:
                break


def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    if dist.get_rank() == 0:
        print("arguments:", flush=True)
        for name, value in vars(args).items():
            print(f"  {name} {'.' * (29 - len(name))} {value}", flush=True)
        with open(os.path.join(args.save, "args.json"), "w") as handle:
            json.dump(vars(args), handle)
    device = torch.cuda.current_device()
    log_rank0("\n\n" + "=" * 30 + f" EXP at {time.strftime('%Y-%m-%d %H:%M:%S')} " + "=" * 30,
              os.path.join(args.save, "log.txt"))

    with open(args.deepspeed_config) as handle:
        ds_config = json.load(handle)
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    # The config is passed as config_params; DeepSpeed rejects both at once.
    args.deepspeed_config = None

    tokenizer = load_tokenizer(args)
    datasets = {
        split: DistillationDataset(args.data_dir, name, args.max_length, tokenizer.eos_token_id, num)
        for split, name, num in (("train", "train", args.train_num), ("dev", "valid", args.dev_num))
    }
    print_rank0(f"train examples: {len(datasets['train'])}, dev examples: {len(datasets['dev'])}")

    args.train_iters_per_epoch = len(datasets["train"]) // (
        args.batch_size * dist.get_world_size() * args.gradient_accumulation_steps)
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.save_interval == -1:
        args.save_interval = args.train_iters_per_epoch
    if args.eval_interval == -1:
        args.eval_interval = args.train_iters_per_epoch
    print_rank0(f"train iters per epoch: {args.train_iters_per_epoch}, total iters: {args.total_iters}")

    student = load_student(args, device)
    teacher = load_teacher(args, device)
    student.resize_token_embeddings(teacher.config.vocab_size)

    layers = None
    if args.aux_loss is not None:
        layers = (
            select_layers(student.config.num_hidden_layers, args.aux_num_layers,
                          args.aux_layer_min, args.aux_layer_max),
            select_layers(teacher.config.num_hidden_layers, args.aux_num_layers,
                          args.aux_layer_min, args.aux_layer_max),
        )
        print_rank0(f"[aux] student layers {layers[0]}, teacher layers {layers[1]}")
        if args.aux_loss == "hidden_mse":
            attach_projectors(student, len(layers[0]), teacher.config.hidden_size, args.seed + 1)

    if args.resume_ckpt:
        if args.resume_global_step <= 0 or args.resume_global_step % args.train_iters_per_epoch:
            raise ValueError("--resume-global-step must be a positive epoch boundary "
                             f"(multiple of {args.train_iters_per_epoch})")
        load_resume_weights(args, student)

    optimizer = build_optimizer(args, student)
    lr_scheduler = build_scheduler(args, optimizer)
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=student, optimizer=optimizer, args=args, lr_scheduler=lr_scheduler,
        config_params=ds_config,
    )
    train(args, tokenizer, engine, optimizer, lr_scheduler, datasets, teacher, layers, device)


if __name__ == "__main__":
    main()
