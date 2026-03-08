import logging
from typing import Literal

logger = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(
        self,
        patience: int = 3,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._counter = 0
        self._best: float = float("inf") if mode == "min" else float("-inf")

    @property
    def should_stop(self) -> bool:
        return self._counter >= self.patience

    def step(self, metric: float) -> bool:
        improved = (
            metric < self._best - self.min_delta if self.mode == "min"
            else metric > self._best + self.min_delta
        )

        if improved:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
            logger.debug("No improvement %d/%d (best=%.6f)", self._counter, self.patience, self._best)

        return self.should_stop

    def reset(self) -> None:
        self._counter = 0
        self._best = float("inf") if self.mode == "min" else float("-inf")
