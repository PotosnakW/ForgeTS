from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
from torch import Tensor

from ts_dataloader import DataLoaderFactory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train(
    model,
    mcfg:         SimpleNamespace,
    train_loader,
    val_loaders:  dict,
    device:       torch.device | None = None,
    seed:         int = 42,
    resume:       str | None = None,
) -> Dict[str, Dict[str, float]]:

    # ── 5. Optimizer + scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=mcfg.learning_rate, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=mcfg.max_steps, eta_min=mcfg.learning_rate / 10
    )

    # ── 6. Wire training state ───────────────────────────────────
    model.setup_training(
        mcfg         = mcfg,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        optimizer    = optimizer,
        scheduler    = scheduler,
        device       = device,
        seed         = seed,
    )

    if resume:
        model.load_train(resume)
        logger.info("Resumed from step %d", model.global_step)

    # ── 7. Fit ───────────────────────────────────────────────────
    model.fit()
    logger.info("Training complete.")

    # ── 8. Save ──────────────────────────────────────────────────
    save_path = Path(mcfg.checkpoint_dir) / "final.pt"
    model.save_state(save_path)
    logger.info("Saved → %s", save_path)

def eval_test(
    model,
    factory:      DataLoaderFactory,
) -> Dict[str, Tensor]:
    """
    Run inference on all test loaders.

    Returns {dataset_name: predictions tensor}.
    """
    test_loaders = factory.test_dataloaders()
    test_preds   = {name: model.predict(loader) for name, loader in test_loaders.items()}
    logger.info("Test shapes: %s", {k: tuple(v.shape) for k, v in test_preds.items()})
    return test_preds
