"""Public configurable training API for descriptor-free slow coordinates."""

from .batches import (
    LaggedBatch,
    ModelBatch,
    collate_lagged_batches,
    collate_model_batches,
)
from .objectives import SlowCVLossResult, slow_cv_loss, trajectory_centered_pairs
from .trainer import SlowCVTrainer, TrainingResult

__all__ = [
    "LaggedBatch",
    "ModelBatch",
    "SlowCVLossResult",
    "SlowCVTrainer",
    "TrainingResult",
    "collate_lagged_batches",
    "collate_model_batches",
    "slow_cv_loss",
    "trajectory_centered_pairs",
]
