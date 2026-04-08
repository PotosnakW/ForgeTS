import logging
import os
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import yaml

from models.transformer import Transformer
from dataloaders._ts_dataloader import DataLoaderFactory
from common.train import train, eval_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_TYPES = {
    "transformer": Transformer,
    # "rnn": RNN,
    # "cnn": CNN,
}

def get_model(name: str):
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_TYPES.keys())}")
    return MODEL_TYPES[name]


# ── Custom resolver: allows ${load:conf/dataset/simglucose.yaml} in split configs ──
def _load_dataset_file(path: str) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, path)
    with open(full_path) as f:
        return yaml.safe_load(f)  # plain dict, not OmegaConf.create()

OmegaConf.register_new_resolver("load", _load_dataset_file)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Merge base into model so model config has all training params too
    mcfg = OmegaConf.merge(
        OmegaConf.to_container(cfg.base,  resolve=True),
        OmegaConf.to_container(cfg.model, resolve=True),
    )
    mcfg = OmegaConf.create(mcfg)
    mcfg.horizon_override = getattr(cfg.dataset, "horizon_override", None)
    mcfg.horizon = mcfg.horizon_override  # None = dynamic, int = fixed
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))

    factory = DataLoaderFactory(mcfg, dcfg)
    train_loader = factory.train_dataloader()
    val_loaders = factory.val_dataloaders()

    model_cls = get_model(mcfg.model_type)
    model = model_cls(mcfg)
    
    train(
        model        = model,
        mcfg         = mcfg,
        train_loader = train_loader,
        val_loaders  = val_loaders,
        device       = torch.device(cfg.device),
        seed         = cfg.base.seed,
        resume       = cfg.get("resume", None),
    )
    eval_test(model, factory)


if __name__ == "__main__":
    main()