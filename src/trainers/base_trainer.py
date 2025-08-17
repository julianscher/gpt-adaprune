import os
import re
from abc import ABC, abstractmethod
from typing import Tuple, Callable

import torch
from torch import nn

from src.data.linear_regression import LinearRegressionDataLoader


class Trainer(ABC):
    def __init__(self, model: nn.Module, device: str, logger) -> None:
        self.model = model
        self.device = device
        self.logger = logger

    @abstractmethod
    def train(self, **kwargs) -> None:
        """Train the model """
        pass

    @abstractmethod
    def validate(self, **kwargs) -> Tuple[float, float]:
        """Validate the model on a validation set."""
        pass

    @abstractmethod
    def test(self, **kwargs) -> Tuple[float, float]:
        """Evaluate the model on a test set."""
        pass

    @abstractmethod
    def predict(self, X_batch: torch.Tensor, y_batch: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Run inference using the trained model."""
        pass

    def save(self, path=None, **kwargs):
        model_path = os.path.join(self.PATH if path is None else path, "model.pt")
        torch.save(self.model.state_dict(), model_path)

    def load(self, ckpt_dir_or_file: str, **kwargs) -> int:
        return self._load_checkpoint(ckpt_dir_or_file, **kwargs)

    def _evaluate(self, loader: LinearRegressionDataLoader, metric: Callable, eval_steps: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss, total_metric, count = 0.0, 0.0, 0
        with torch.no_grad():
            for _ in range(int(eval_steps)):
                X_batch, y_batch = next(loader)
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                out = self.model(X_batch, y_batch)
                loss = self._compute_loss(out, y_batch)
                total_loss += float(loss)
                if metric:
                    m = metric(out, y_batch, self.device) if callable(metric) else None
                    if m is not None:
                        total_metric += float(m if not isinstance(m, torch.Tensor) else m.item())
                count += 1
        avg_loss = total_loss / max(count, 1)
        avg_metric = (total_metric / count) if (metric and count > 0) else None
        return avg_loss, avg_metric

    def _get_state_dir(self, ckpt_dir_or_file: str) -> Tuple[int, str, str]:

        def _find_latest_step_file(d: str) -> Tuple[int, str]:
            pats = [f for f in os.listdir(d) if f.startswith("step") and f.endswith(".pt")]
            if not pats:
                raise FileNotFoundError(f"No step*.pt checkpoints found in {d}")
            steps = []
            for f in pats:
                m = re.search(r"step(\d+)\.pt$", f)
                if m:
                    steps.append((int(m.group(1)), f))
            if not steps:
                raise FileNotFoundError(f"No parsable step*.pt files in {d}")
            steps.sort(key=lambda x: x[0])
            return steps[-1][0], os.path.join(d, steps[-1][1])

        if os.path.isdir(ckpt_dir_or_file):
            step, model_path = _find_latest_step_file(ckpt_dir_or_file)
            state_dir = ckpt_dir_or_file
        else:
            model_path = ckpt_dir_or_file
            m = re.search(r"step(\d+)\.pt$", os.path.basename(model_path))
            if not m:
                raise ValueError("When passing a file, name must look like 'stepNNN.pt'")
            step = int(m.group(1))
            state_dir = os.path.dirname(model_path)

        return step, model_path, state_dir