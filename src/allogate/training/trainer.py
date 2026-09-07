"""Small, deterministic training loop with auditable public artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from allogate.config.hashing import canonical_json, stable_digest
from allogate.config.schemas import ExperimentConfig
from allogate.models.model import AlloGateGeometryModel

from .batches import LaggedBatch
from .objectives import slow_cv_loss


BatchSource = Iterable[LaggedBatch] | Callable[[], Iterable[LaggedBatch]]


def _state_digest(state: dict[str, Tensor]) -> str:
    digest = sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".allogate-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".allogate-", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _batches(source: BatchSource) -> Iterable[LaggedBatch]:
    return source() if callable(source) else source


@dataclass(frozen=True, slots=True)
class TrainingResult:
    experiment_digest: str
    run_digest: str
    initial_model_digest: str
    model_digest: str
    best_epoch: int
    completed_epochs: int
    best_validation_loss: float
    history: tuple[dict[str, float | int], ...]
    output_directory: Path | None


class SlowCVTrainer:
    """Train Direct and Interface Route CVs on user-supplied lagged graph batches."""

    def __init__(
        self,
        model: AlloGateGeometryModel,
        config: ExperimentConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
        }[config.optimization.dtype]
        self.experiment_digest = stable_digest(config)

    def _run_epoch(
        self,
        source: BatchSource,
        *,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals: dict[str, float] = {}
        pair_total = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in _batches(source):
                if batch.pair_count < 2:
                    raise ValueError("every kinetic batch must contain at least two lagged pairs")
                if batch.pair_count > self.config.optimization.batch_size:
                    raise ValueError(
                        "a supplied batch exceeds optimization.batch_size"
                    )
                moved = batch.to(self.device, dtype=self.dtype)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                time = self.model(
                    moved.time.graph, **moved.time.model_arguments()
                )
                lagged = self.model(
                    moved.lagged.graph, **moved.lagged.model_arguments()
                )
                result = slow_cv_loss(
                    time,
                    lagged,
                    self.config.objective,
                    moved.trajectory_index,
                )
                if not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError("the slow-CV objective became non-finite")
                if training:
                    result.loss.backward()
                    gradient_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimization.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                else:
                    gradient_norm = result.loss.new_zeros(())
                metrics = result.metrics()
                metrics["gradient_norm"] = float(gradient_norm)
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * batch.pair_count
                pair_total += batch.pair_count
        if pair_total == 0:
            raise ValueError("a training or validation epoch received no lagged batches")
        return {name: value / pair_total for name, value in totals.items()}

    def fit(
        self,
        train_batches: BatchSource,
        validation_batches: BatchSource,
        *,
        output_directory: str | Path | None = None,
    ) -> TrainingResult:
        optimization = self.config.optimization
        output_path = None if output_directory is None else Path(output_directory)
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            if any(output_path.iterdir()):
                raise FileExistsError(
                    "output_directory must be empty to preserve immutable run artifacts"
                )
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        initial_model_digest = _state_digest(initial_state)
        run_digest = stable_digest(
            {
                "schema_version": "allogate.training_run_identity.v1",
                "experiment_digest": self.experiment_digest,
                "initial_model_digest": initial_model_digest,
            }
        )
        random.seed(optimization.seed)
        np.random.seed(optimization.seed)
        torch.manual_seed(optimization.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(optimization.seed)
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(optimization.deterministic_algorithms)
        self.model.to(device=self.device, dtype=self.dtype)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimization.learning_rate,
            weight_decay=optimization.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=optimization.learning_rate_factor,
            patience=optimization.learning_rate_patience,
            min_lr=optimization.minimum_learning_rate,
        )

        history: list[dict[str, float | int]] = []
        best_state: dict[str, Tensor] | None = None
        best_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        for epoch in range(optimization.epochs):
            train_metrics = self._run_epoch(train_batches, optimizer=optimizer)
            validation_metrics = self._run_epoch(
                validation_batches, optimizer=None
            )
            validation_loss = validation_metrics["loss"]
            scheduler.step(validation_loss)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            history.append(row)
            if validation_loss < best_loss - optimization.early_stopping_min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                epoch + 1 >= optimization.minimum_epochs
                and epochs_without_improvement >= optimization.early_stopping_patience
            ):
                break
        if best_state is None:
            raise RuntimeError("training completed without a finite validation checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(device=self.device, dtype=self.dtype)
        model_digest = _state_digest(best_state)

        if output_path is not None:
            config_payload = json.loads(canonical_json(self.config))
            checkpoint_name = "best-model.pt"
            _atomic_checkpoint(
                output_path / checkpoint_name,
                {
                    "schema_version": "allogate.training_checkpoint.v1",
                    "experiment_config": config_payload,
                    "experiment_digest": self.experiment_digest,
                    "run_digest": run_digest,
                    "initial_model_digest": initial_model_digest,
                    "model_digest": model_digest,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "model_state": best_state,
                },
            )
            _atomic_json(output_path / "history.json", history)
            _atomic_json(
                output_path / "run.json",
                {
                    "schema_version": "allogate.training_run.v1",
                    "experiment_digest": self.experiment_digest,
                    "run_digest": run_digest,
                    "initial_model_digest": initial_model_digest,
                    "model_digest": model_digest,
                    "best_epoch": best_epoch,
                    "completed_epochs": len(history),
                    "best_validation_loss": best_loss,
                    "checkpoint": checkpoint_name,
                    "history": "history.json",
                    "runtime": {
                        "device_type": self.device.type,
                        "torch_version": torch.__version__,
                    },
                },
            )
        torch.use_deterministic_algorithms(previous_deterministic)
        return TrainingResult(
            experiment_digest=self.experiment_digest,
            run_digest=run_digest,
            initial_model_digest=initial_model_digest,
            model_digest=model_digest,
            best_epoch=best_epoch,
            completed_epochs=len(history),
            best_validation_loss=best_loss,
            history=tuple(history),
            output_directory=output_path,
        )
