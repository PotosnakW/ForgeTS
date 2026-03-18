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
    """
    Convert one dataset entry dict to a namespace with sensible defaults
    for optional fields so callers never have to guard with hasattr().
    """
    defaults = dict(
        weight = 1.0,
        hist_exog_cols = [],
        futr_exog_cols = [],
        stat_exog_cols = [],
        per_series_split = False,
        use_context_head = False,
        multivariate = False,
    )
    return _to_namespace({**defaults, **d})


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_model_config(
    base_cfg_path:  Union[str, Path],
    model_cfg_path: Union[str, Path],
) -> SimpleNamespace:
    """
    Merge base_config + model_config → one flat SimpleNamespace.
    Model-specific values win on any key conflict.

    Parameters
    ----------
    base_cfg_path  : path to base_config.yaml  (shared training knobs)
    model_cfg_path : path to model_config.yaml (architecture overrides)

    Returns
    -------
    SimpleNamespace with every key from both files accessible as attributes.
    """
    base  = _load_yaml(base_cfg_path)
    model = _load_yaml(model_cfg_path)
    merged = {**base, **model}   # model wins on conflict
    return _to_namespace(merged)


def load_dataset_config(
    dataset_cfg_path: Union[str, Path],
) -> SimpleNamespace:
    """
    Load dataset_config.yaml → SimpleNamespace with .train / .validation / .test
    each being a list of per-dataset SimpleNamespaces.

    Parameters
    ----------
    dataset_cfg_path : path to dataset_config.yaml

    Returns
    -------
    SimpleNamespace:
        .train      : list[SimpleNamespace]
        .validation : list[SimpleNamespace]
        .test       : list[SimpleNamespace]
    """
    raw = _load_yaml(dataset_cfg_path)
    return SimpleNamespace(
        train      = [_dataset_entry(e) for e in (raw.get("train")      or [])],
        validation = [_dataset_entry(e) for e in (raw.get("validation") or [])],
        test       = [_dataset_entry(e) for e in (raw.get("test")       or [])],
    )