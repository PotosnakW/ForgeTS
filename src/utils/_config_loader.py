from pathlib import Path
from types import SimpleNamespace
from typing import Union

import yaml


def _load_yaml(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _dataset_entry(d: dict) -> SimpleNamespace:
    """One dataset entry with sensible defaults for optional fields."""
    defaults = dict(
        weight           = 1.0,
        hist_exog_cols   = [],
        futr_exog_cols   = [],
        stat_exog_cols   = [],
        per_series_split = False,
        use_context_head = False,
    )
    return _to_namespace({**defaults, **d})


def load_model_config(
    base_cfg_path:  Union[str, Path],
    model_cfg_path: Union[str, Path],
) -> SimpleNamespace:
    """
    Merge base_config + model_config → one flat SimpleNamespace.
    Model-specific values win on any key conflict.

    Also derives:
        mcfg.h                          — horizon from first train dataset
                                          NOTE: requires dataset_config to be
                                          loaded first, or set manually after.
        mcfg.loss.outputsize_multiplier — len(quantiles) or 1 for point forecast
    """
    base   = _load_yaml(base_cfg_path)
    model  = _load_yaml(model_cfg_path)
    merged = {**base, **model}
    mcfg   = _to_namespace(merged)

    # Derive loss stub so MOMENT can read loss.outputsize_multiplier at init
    quantiles = getattr(mcfg, "quantiles", None) or []
    mcfg.loss = SimpleNamespace(outputsize_multiplier=len(quantiles) or 1)

    return mcfg


def load_dataset_config(dataset_cfg_path: Union[str, Path]) -> SimpleNamespace:
    """
    Load dataset_config.yaml → SimpleNamespace with
    .train / .validation / .test as lists of entry namespaces.
    """
    raw = _load_yaml(dataset_cfg_path)
    return SimpleNamespace(
        train      = [_dataset_entry(e) for e in (raw.get("train")      or [])],
        validation = [_dataset_entry(e) for e in (raw.get("validation") or [])],
        test       = [_dataset_entry(e) for e in (raw.get("test")       or [])],
    )
