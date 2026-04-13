"""
Card/ID ROI Detection - Training Entry Point

Usage
-----
# Single GPU
python train.py

# Multi GPU (DDP)
torchrun --nproc_per_node=4 train.py

# Resume from checkpoint
python train.py --resume ./output/roi_detection/checkpoints/checkpoint_step10000.pth

# Use performer attention instead of linear
python train.py --global_attn_type performer
"""

import os
import json
import time
import random
import datetime
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import RoiDetectionModel, ModelEMA
from src.loss import DetectionLoss, LossWeightScheduler
from src.optimizer import build_optimizer_scheduler
from src.config import TrainConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer) -> float:
    for pg in optimizer.param_groups:
        return pg['lr']


def is_main_process(args) -> bool:
    if not args.distributed:
        return True
    return torch.distributed.get_rank() == 0


def save_checkpoint(path: str, model, optimizer, scheduler,
                    epoch: int, global_step: int, args):
    state = {
        'model'       : model.module.state_dict() if args.distributed else model.state_dict(),
        'optimizer'   : optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch'       : epoch,
        'global_step' : global_step,
    }
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if 'lr_scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['lr_scheduler'])
    return ckpt.get('epoch', 0), ckpt.get('global_step', 0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    ema, loss_scheduler, epoch, global_step,
                    checkpoint_dir, args, writer):
    model.train()
    device = next(model.parameters()).device

    for idx, batch in enumerate(dataloader):
        images      = batch['image'].to(device)
        seg_targets = batch['seg_map'].to(device)
        cls_targets = batch['label'].to(device)

        seg_pred, cls_pred = model(images)

        loss_weights = loss_scheduler.step() if loss_scheduler else None
        loss = criterion(seg_pred, seg_targets, cls_pred, cls_targets,
                         loss_weights=loss_weights)

        optimizer.zero_grad()
        loss.backward()
        if args.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        global_step += 1

        if is_main_process(args) and (idx + 1) % args.print_freq == 0:
            lr = get_lr(optimizer)
            print(f"Epoch [{epoch}] Step [{idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}  LR: {lr:.2e}")
            if writer:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr',   lr,          global_step)

        # Periodic checkpoint
        if is_main_process(args) and global_step % (len(dataloader) * args.checkpoint_freq) == 0:
            ckpt_path = os.path.join(checkpoint_dir,
                                     f'checkpoint_step{global_step}.pth')
            save_checkpoint(ckpt_path, model, optimizer, scheduler,
                            epoch, global_step, args)

        if global_step >= args.max_steps:
            break

    return global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Distributed setup
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed + (torch.distributed.get_rank() if args.distributed else 0))

    # Model
    model = RoiDetectionModel(args).to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device])

    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process(args):
        print(f"Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion      = DetectionLoss()
    optimizer, lr_scheduler = build_optimizer_scheduler(model, args)
    loss_scheduler = LossWeightScheduler(total_steps=args.max_steps)
    ema            = ModelEMA(model, decay=0.999)

    # Checkpoints
    checkpoint_dir = os.path.join(args.output_folder, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume
    last_epoch, global_step = 0, 0
    if args.resume:
        last_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, device)
        ema = ModelEMA(model, decay=0.999)
        if is_main_process(args):
            print(f"Resumed from {args.resume} "
                  f"(epoch {last_epoch}, step {global_step})")
    elif args.pretrain_ckpt:
        load_checkpoint(args.pretrain_ckpt, model,
                        optimizer, lr_scheduler, device)
        if is_main_process(args):
            print(f"Loaded pretrain ckpt: {args.pretrain_ckpt}")

    # Dataset — plug in your own DataLoader here
    # train_dataset  = build_dataset('train', args)
    # train_loader   = build_dataloader(train_dataset, args)
    raise NotImplementedError(
        "Dataset loading is not included in this repository.\n"
        "Implement build_dataset() and build_dataloader() for your data."
    )

    # Logging
    writer = None
    if is_main_process(args):
        writer = SummaryWriter(
            log_dir=os.path.join(args.output_folder, 'logs/train'))
        with open(os.path.join(args.output_folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    # Training
    if is_main_process(args):
        print('Start training')
    start_time = time.time()

    for epoch in range(last_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, lr_scheduler,
            ema, loss_scheduler, epoch, global_step,
            checkpoint_dir, args, writer)

        if is_main_process(args):
            log = {'epoch': epoch, 'global_step': global_step}
            with open(os.path.join(args.output_folder, 'log.txt'), 'a') as f:
                f.write(json.dumps(log) + '\n')

        if global_step >= args.max_steps:
            break

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if is_main_process(args):
        print(f'Training time: {total_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Card/ID ROI Detection Training')
    parser.add_argument('--resume',           type=str,   default=None)
    parser.add_argument('--pretrain_ckpt',    type=str,   default=None)
    parser.add_argument('--output_folder',    type=str,   default='./output/roi_detection')
    parser.add_argument('--data_root',        type=str,   default='./data')
    parser.add_argument('--global_attn_type', type=str,   default='linear',
                        choices=['softmax', 'linear', 'performer'])
    parser.add_argument('--batch_size',       type=int,   default=None)
    parser.add_argument('--lr',               type=float, default=None)
    parser.add_argument('--distributed',      action='store_true')
    cmd_args = parser.parse_args()

    # Start from config defaults, override with CLI args
    args = TrainConfig()
    for k, v in vars(cmd_args).items():
        if v is not None:
            setattr(args, k, v)

    os.makedirs(args.output_folder, exist_ok=True)
    main(args)