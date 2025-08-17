import os
import random
from typing import Callable, Dict, Tuple

import dill
import numpy as np
import torch
from torch import nn

from src.data.linear_regression import LinearRegressionDataLoader
from src.trainers.base_trainer import Trainer


class BackpropTrainer(Trainer):
    def __init__(self, model: nn.Module, criterion: Callable, train_loader: LinearRegressionDataLoader,
                 val_loader: LinearRegressionDataLoader=None, test_loader: LinearRegressionDataLoader=None,
                 optimizer: torch.optim.Optimizer=None, max_grad_norm: float=1.0,
                 scaler: torch.cuda.amp.GradScaler=None, lr_scheduler=None, validate_every_steps: int=0,
                 early_stopping_cfg: Dict={}, device: str="cpu", worker=None, logger=None, PATH: str=None,
                 log_every_steps: int=10, save_every_steps: int=0, **kwargs) -> None:
        super().__init__(model, device, logger)

        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if optimizer is None:
            raise ValueError("BackpropTrainer requires a torch optimizer (passed by Worker).")
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.validate_every_steps = validate_every_steps

        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.early_stopping_cfg = early_stopping_cfg
        self.device = device
        self.worker = worker
        self.logger = logger
        self.PATH = PATH
        self.log_every_steps = log_every_steps
        self.save_every_steps = save_every_steps

        self._global_step = 0


    def train(self, train_steps: int, val_steps: int) -> None:
        self.model.train()
        if not self.worker.args.resume_ckpt:
            self._save_checkpoint(step=0)

        use_plateau = isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        loss_window = []
        best_val = float("inf")
        patience = 0
        es = self.early_stopping_cfg
        es_patience  = int(es.get("patience", 0))
        es_min_delta = float(es.get("min_delta", 0.0))

        for _ in range(train_steps):
            self._global_step += 1

            X_batch, y_batch = next(self.train_loader)
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            loss, grad_norm = self._run_step(X_batch, y_batch, scaler=self.scaler, max_grad_norm=self.max_grad_norm)
            loss_window.append(loss)

            if self._global_step % self.log_every_steps == 0 or self._global_step == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                avg_loss = sum(loss_window[-self.log_every_steps:]) / min(len(loss_window), self.log_every_steps)
                self.logger.log(f"Step {self._global_step} | Loss: {loss:.4f} | "
                      f"Avg({self.log_every_steps}): {avg_loss:.4f} | GradNorm: {grad_norm:.2f} | LR: {lr:.2e}")

                results = {
                    "train_loss": loss,
                    "train_avg_loss": avg_loss,
                }
                self.logger.log_step(self._global_step, results)

            if self.save_every_steps and self._global_step % self.save_every_steps == 0:
                self._save_checkpoint(step=self._global_step)

            if self.lr_scheduler is not None and not use_plateau:
                self.lr_scheduler.step()

            if self.validate_every_steps > 0 and self._global_step % self.validate_every_steps == 0:
                if self.val_loader is not None:
                    val_loss, _ = self.validate(metric=None, val_steps=val_steps)
                    self.logger.log(f"[Validation] Step {self._global_step} | val_loss: {val_loss:.6f}")

                    if use_plateau:
                        self.lr_scheduler.step(val_loss)

                    # early stopping on val_loss
                    if es_patience > 0:
                        improved = (best_val - val_loss) > es_min_delta
                        if improved:
                            best_val = val_loss
                            patience = 0
                        else:
                            patience += 1
                            if patience >= es_patience:
                                self.logger.log(f"Early stopping at step {self._global_step} "
                                                f"(best val_loss={best_val:.6f})")
                                break

    def validate(self, metric: Callable, val_steps: int) -> Tuple[float, float]:
        if self.val_loader is None:
            raise ValueError("Validation loader is not provided.")
        avg_loss, avg_metric = self._evaluate(self.val_loader, metric, val_steps)
        results = {
            "val_loss": avg_loss,
            "val_avg_loss": avg_metric,
        }
        self.logger.log_validation(results)
        return avg_loss, avg_metric

    def test(self, metric: Callable, test_steps: int, loader: LinearRegressionDataLoader=None) -> Tuple[float, float]:
        loader = loader or self.test_loader
        if loader is None:
            raise ValueError("Test loader is not provided.")
        avg_loss, avg_metric = self._evaluate(loader, metric, test_steps)
        results = {
            "train_loss": avg_loss,
            "train_avg_loss": avg_metric,
        }
        self.logger.log_test(results)
        return avg_loss, avg_metric

    def predict(self, X_batch: torch.Tensor, y_batch: torch.Tensor=None, **kwargs) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            X_batch = X_batch.to(self.device)
            return self.model(X_batch)

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        try:
            return self.criterion(output, target, self.device)
        except TypeError:
            return self.criterion(output, target)

    def _run_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor, *, scaler: torch.cuda.amp.GradScaler=None,
                  max_grad_norm: float = 0.0) -> Tuple[float, float]:
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = self.model(X_batch, y_batch)
            loss = self._compute_loss(output, y_batch)

        grad_norm = 0.0
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm).item()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm).item()
            self.optimizer.step()

        return float(loss.detach()), grad_norm

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = os.path.join(self.PATH, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        model_path = os.path.join(ckpt_dir, f"step{step}.pt")
        torch.save(self.model.state_dict(), model_path)

        state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        state_path = os.path.join(ckpt_dir, f"train_state_step{step}.pkl")
        with open(state_path, "wb") as f:
            dill.dump(state, f)

        self.logger.log(f"Step {step} | Training checkpoint saved at {ckpt_dir}")

    def _load_checkpoint(self, ckpt_dir_or_file: str, *, load_optimizer: bool = True,
                         map_location=None, strict: bool = True) -> int:

        map_location = map_location or self.device

        step, model_path, state_dir = self._get_state_dir(ckpt_dir_or_file)

        state_dict = torch.load(model_path, map_location=map_location)
        self.model.load_state_dict(state_dict, strict=strict)

        state_pkl = os.path.join(state_dir, f"train_state_step{step}.pkl")
        if os.path.isfile(state_pkl):
            with open(state_pkl, "rb") as f:
                train_state = dill.load(f)

            if load_optimizer:
                if "optimizer" in train_state and self.optimizer is not None:
                    self.optimizer.load_state_dict(train_state["optimizer"])
                if "lr_scheduler" in train_state and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(train_state["lr_scheduler"])
                if "scaler" in train_state and self.scaler is not None:
                    self.scaler.load_state_dict(train_state["scaler"])

            if "torch_rng_state" in train_state:
                torch.set_rng_state(train_state["torch_rng_state"])
            if "cuda_rng_state" in train_state and train_state["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(train_state["cuda_rng_state"])
            if "numpy_rng_state" in train_state:
                np.random.set_state(train_state["numpy_rng_state"])
            if "python_rng_state" in train_state:
                random.setstate(train_state["python_rng_state"])

            self._global_step = int(train_state.get("step", step))
        else:
            self._global_step = int(step)

        self.logger.log(f"Loaded checkpoint from '{model_path}' (step {self._global_step})")
        return self._global_step

