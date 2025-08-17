import os
from typing import Callable, Dict, Tuple

import dill
import numpy as np
import torch
from torch import nn

from src.data.linear_regression import LinearRegressionDataLoader
from src.trainers.base_trainer import Trainer
from src.utils.custom_lr_scheduler import CustomLRScheduler


class AdaPruneTrainer(Trainer):
    def __init__(self, model: nn.Module, criterion: Callable, train_loader: LinearRegressionDataLoader,
                 val_loader: LinearRegressionDataLoader=None, test_loader: LinearRegressionDataLoader=None,
                 optimizer: str="sgd", validate_every_steps: int=0, max_grad_norm: float=1.0,
                 lr_scheduler: CustomLRScheduler=None, early_stopping_cfg: Dict={},
                 wanted_density: float=1.0, device: str="cpu", worker=None, logger=None, PATH: str=None,
                 log_every_steps: int=10, save_every_steps: int=0, **kwargs) -> None:
        super().__init__(model, device, logger)

        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.validate_every_steps = validate_every_steps
        self.max_grad_norm = max_grad_norm

        self.lr_scheduler = lr_scheduler
        self.early_stopping_cfg = early_stopping_cfg
        self.wanted_density = wanted_density

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

        loss_window = []
        best_val = float("inf")
        patience = 0
        es = self.early_stopping_cfg
        es_patience = int(es.get("patience", 0))
        es_min_delta = float(es.get("min_delta", 0.0))

        for _ in range(int(train_steps)):
            self._global_step += 1

            X_batch, y_batch = next(self.train_loader)
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            loss, grad_w, grad_m = self._run_step(X_batch, y_batch)
            loss_window.append(loss)

            ones_pct = float(self.model.count_ones())
            if self._global_step % self.log_every_steps == 0 or self._global_step == 1:
                weights_lr = self.model.weights_lr
                mask_lr = self.model.mask_lr
                avg_loss = sum(loss_window[-self.log_every_steps:]) / min(len(loss_window), self.log_every_steps)
                self.logger.log(f"Step {self._global_step} | Loss: {loss:.4f} | "
                       f"Avg({self.log_every_steps}): {avg_loss:.4f} | "
                       f"GradNorm W: {grad_w:.2f} | GradNorm M: {grad_m:.2f} | "
                       f"Weights_lr: {weights_lr:.2e} | Mask_lr: {mask_lr:.2e} | OnesPercent: {ones_pct:.6f}")

                results = {
                    "train_loss": loss,
                    "train_avg_loss": avg_loss,
                    "grad_norm_weights": grad_w,
                    "grad_norm_mask": grad_m,
                    "weights_lr": weights_lr,
                    "mask_lr": mask_lr,
                    "ones_percent": ones_pct,
                }
                self.logger.log_step(self._global_step, results)

            if self.save_every_steps and self._global_step % self.save_every_steps == 0:
                self._save_checkpoint(step=self._global_step)

            if self.lr_scheduler:
                self.lr_scheduler.step()
                self.model.weights_lr = self.lr_scheduler.get_lr()
            else:
                if hasattr(self.model, "weights_lr"):
                    self.model.weights_lr *= 0.5

            self._adjust_decay(ones_pct)

            if self.validate_every_steps > 0 and self._global_step % self.validate_every_steps == 0:
                if self.val_loader is not None:
                    val_loss, _ = self._evaluate(self.val_loader, metric=None, eval_steps=val_steps)
                    self.logger.log(f"[Validation] Step {self._global_step} | val_loss: {val_loss:.6f}")

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

    def save(self, path: str=None, **kwargs) -> None:
        model = self.model.put_on_masks()
        dst_dir = self.PATH if path is None else path
        os.makedirs(dst_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dst_dir, "model.pt"))

        info = {"mask_weights": self.model.mask_weights, "b_fnc": self.model.binarise}
        with open(os.path.join(dst_dir, "pickled_info_dict"), "wb") as f:
            dill.dump(info, f)


    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        try:
            return self.criterion(output, target, self.device)
        except TypeError:
            return self.criterion(output, target)

    def _run_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[float, float, float]:
        out = self.model(X_batch, y_batch)
        loss = self._compute_loss(out, y_batch)

        loss.backward()

        grad_norm_weights = torch.nn.utils.clip_grad_norm_(
            [self.model.update_weights[k] for k in self.model.update_weights], self.max_grad_norm).item()
        grad_norm_mask = torch.nn.utils.clip_grad_norm_([self.model.mask_weights[k] for k in self.model.mask_weights],
                                                        self.max_grad_norm).item()

        if self.optimizer == "sgd":
            self.model.sgd_step()
        elif self.optimizer == "adamw":
            self.model.adamw_step()
        else:
            if self._global_step == 1:
                self.logger.log(f"WARN: Unknown optimizer '{self.optimizer}'.")

        return float(loss.detach()), grad_norm_weights, grad_norm_mask

    def _adjust_decay(self, ones_percent: float) -> None:
        if self.wanted_density:
            if ones_percent < self.wanted_density + 0.005:
                self.model.mask_decay = 0.
                self.model.mask_lr = 0.

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = os.path.join(self.PATH, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        model = self.model.put_on_masks()
        model_path = os.path.join(ckpt_dir, f"step{step}.pt")
        torch.save(model.state_dict(), model_path)

        ckpt = {
            "step": step,
            "mask_weights": self.model.mask_weights,
            "update_weights": self.model.update_weights,
            "adam_weights": self.model.adam_weights,
            "b_fnc": self.model.binarise,
            "weights_lr": self.model.weights_lr,
            "mask_lr": self.model.mask_lr,
            "mask_decay": self.model.mask_decay,
            "current_lr": self.lr_scheduler.current_lr,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy_rng_state": np.random.get_state(),
        }

        info_path = os.path.join(ckpt_dir, f"training_state_step{step}.pkl")
        with open(info_path, "wb") as f:
            dill.dump(ckpt, f)

        self.logger.log(f"Step {step} | Training checkpoint saved at {ckpt_dir}")

    def _load_checkpoint(self, ckpt_dir_or_file: str, *, map_location: str=None, strict: bool = True) -> int:
        step, model_path, state_dir = self._get_state_dir(ckpt_dir_or_file)

        info_path = os.path.join(state_dir, f"training_state_step{step}.pkl")
        if os.path.isfile(info_path):
            with open(info_path, "rb") as f:
                ckpt = dill.load(f)

            self.model.load_weights(ckpt["mask_weights"])
            self.model.update_weights = ckpt["update_weights"]
            self.model.adam_weights = ckpt["adam_weights"]
            self.model.binarise = ckpt["b_fnc"]
            self.model.weights_lr = ckpt.get("weights_lr", self.model.weights_lr)
            self.model.mask_lr = ckpt.get("mask_lr", self.model.mask_lr)
            self.model.mask_decay = ckpt.get("mask_decay", self.model.mask_decay)

            scheduler_state = ckpt.get("current_lr")
            if self.lr_scheduler is not None and scheduler_state is not None:
                self.lr_scheduler.current_lr = scheduler_state

            if "torch_rng_state" in ckpt:
                torch.set_rng_state(ckpt["torch_rng_state"])
            if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
            if "numpy_rng_state" in ckpt:
                np.random.set_state(ckpt["numpy_rng_state"])

            self._global_step = int(ckpt.get("step", step))
        else:
            self._global_step = int(step)

        self.logger.log(f"Loaded AdaPrune checkpoint from '{model_path}' (step {step})")
        return step


