"""
Optimizer and learning rate scheduler for Card/ID ROI Detection.

AdamW + Polynomial Decay with Warmup
-------------------------------------
- AdamW  : weight decay regularisation, betas=(0.9, 0.95)
- Poly   : smooth decay from lr → end_lr over max_steps
- Warmup : linear ramp-up for warmup_steps to avoid early instability
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_scheduler(model: nn.Module, args) -> tuple:
    """
    Build AdamW optimizer and polynomial LR scheduler.

    Parameters
    ----------
    model : nn.Module
    args  : TrainConfig

    Returns
    -------
    optimizer, scheduler
    """
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Separate backbone lr if specified
    if args.lr_backbone_ratio > 0:
        param_groups = [
            {
                'params': [p for n, p in model_without_ddp.named_parameters()
                           if 'vit_enc' not in n and p.requires_grad],
                'lr': args.lr,
            },
            {
                'params': [p for n, p in model_without_ddp.named_parameters()
                           if 'vit_enc' in n and p.requires_grad],
                'lr': args.lr * args.lr_backbone_ratio,
            },
        ]
    else:
        param_groups = model_without_ddp.parameters()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _poly_decay_with_warmup(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_steps,
            power=args.decay_power,
            lr_end_ratio=args.end_lr / args.lr if args.lr != 0 else 1e-7,
        )
    )

    return optimizer, scheduler


def _poly_decay_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    power: float = 1.0,
    lr_end_ratio: float = 1e-7,
) -> float:
    """
    Learning rate schedule: linear warmup → polynomial decay.

    Parameters
    ----------
    step         : current training step
    warmup_steps : linear ramp-up duration
    total_steps  : total training steps
    power        : polynomial decay exponent (1.0 = linear)
    lr_end_ratio : end_lr / initial_lr

    Returns
    -------
    float  multiplicative factor for LambdaLR
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)

    decay_steps = total_steps - warmup_steps
    step_in_decay = min(step - warmup_steps, decay_steps)
    decay_factor = (1 - step_in_decay / decay_steps) ** power
    return decay_factor * (1 - lr_end_ratio) + lr_end_ratio