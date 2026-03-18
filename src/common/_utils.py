from pathlib import Path
from types import SimpleNamespace
from typing import Union
import math
import yaml


class EarlyStopper:
    """
    Counts validation checks without improvement.
    Patience is in units of *checks*, not steps.

    Parameters
    ----------
    patience : int   — number of checks allowed without improvement
    mode     : str   — "min" (lower is better) or "max"
    min_delta: float — minimum change to count as improvement
    """

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self._best      = math.inf if mode == "min" else -math.inf
        self._counter   = 0

    @property
    def best(self) -> float:
        return self._best

    def step(self, metric: float) -> bool:
        """
        Call after each validation check.
        Returns True if training should stop.
        """
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if improved:
            self._best    = metric
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    def state_dict(self) -> dict:
        return {"best": self._best, "counter": self._counter}

    def load_state_dict(self, d: dict):
        self._best    = d["best"]
        self._counter = d["counter"]

# ─────────────────────────────────────────────────────────────────────────────
# Load config utils
# ─────────────────────────────────────────────────────────────────────────────

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

# Public API
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