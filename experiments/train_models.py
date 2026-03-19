import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

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


# ── Custom resolver: allows ${load:conf/dataset/simglucose.yaml} in split configs ──
def _load_dataset_file(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

OmegaConf.register_new_resolver("load", _load_dataset_file)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # ── 0. Log full resolved config ──────────────────────────
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── 1. Device ────────────────────────────────────────────
    device = torch.device(cfg.device)

    # ── 2. Horizon (from first train dataset) ────────────────
    cfg.model.h = cfg.dataset.train[0].horizon

    # ── 3. Data ──────────────────────────────────────────────
    factory      = DataLoaderFactory(cfg.model, cfg.dataset)
    train_loader = factory.train_dataloader()
    val_loaders  = factory.val_dataloaders()

    # ── 4. Model ─────────────────────────────────────────────
    model    = MOMENT(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MOMENT parameters: %s", f"{n_params:,}")

    # ── 5. Train ─────────────────────────────────────────────
    train(
        model        = model,
        mcfg         = cfg.model,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        device       = device,
        seed         = cfg.base.seed,
        resume       = cfg.get("resume", None),
    )

    # ── 6. Test ──────────────────────────────────────────────
    eval_test(model, factory)


if __name__ == "__main__":
    main()