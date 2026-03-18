"""
train_moment.py
───────────────
    python train_moment.py
    python train_moment.py --base_cfg  configs/base_config.yaml \
                           --model_cfg configs/moment_config.yaml \
                           --data_cfg  configs/dataset_config.yaml
    python train_moment.py --resume checkpoints/ckpt_step=0002000.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
from torch import Tensor

from common._utils import load_model_config, load_dataset_config
from ts_dataloader import DataLoaderFactory
from _base_model import BaseModel
from moment import MOMENT
from .common.train import train, eval_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_cfg",  default="configs/base_config.yaml")
    p.add_argument("--model_cfg", default="configs/moment_config.yaml")
    p.add_argument("--data_cfg",  default="configs/dataset_config.yaml")
    p.add_argument("--resume",    default=None)
    p.add_argument("--seed",      default=None)
    p.add_argument("--device",    default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device) if args.device else None

    # ── 1. Configs ───────────────────────────────────────────────
    mcfg = load_model_config(args.base_cfg, args.model_cfg)
    dcfg = load_dataset_config(args.data_cfg)

    # horizon comes straight from the dataset config — no batch peeking needed
    mcfg.h = dcfg.train[0].horizon

    # ── 2. Data ──────────────────────────────────────────────────
    factory      = DataLoaderFactory(mcfg, dcfg)
    train_loader = factory.train_dataloader()
    val_loaders  = factory.val_dataloaders()

    # ── 3. Model ─────────────────────────────────────────────────
    # n_channels is NOT passed — MOMENT infers it from each batch in forward()
    model    = MOMENT(mcfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MOMENT parameters: %s", f"{n_params:,}")

    # ── 5-8. Train ───────────────────────────────────────────────
    train(
        model        = model,
        mcfg         = mcfg,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        device       = device,
        seed         = args.seed,
        resume       = args.resume,
    )

    # ── 9. Evaluate on test set ──────────────────────────────────
    eval_test(model, factory)


if __name__ == "__main__":
    main()