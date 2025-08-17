import argparse
import os, time, json
from dataclasses import asdict
from typing import Dict, Any

import torch

from src.trainers.adaprune_trainer import AdaPruneTrainer
from src.utils.arg_parser import get_criterion, get_optimizer, get_scheduler, get_amp_scaler, load_config, get_test_metric
from src.trainers.backprop_trainer import BackpropTrainer
from src.trainers.base_trainer import Trainer
from src.data.linear_regression import get_dataloaders
from src.utils.logger import SimpleLogger
from src.model.builder import build_model
from src.utils.utils import get_results_path, set_seed

TRAINERS = {
    "backpropagation_trainer": lambda **kwargs: BackpropTrainer(**kwargs),
    "adaprune_trainer":        lambda **kwargs: AdaPruneTrainer(**kwargs),
}

class Worker:
    def __init__(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        self.args = args
        self._init(args, parser)

        model = build_model(args)
        model.train()
        self.device = args.device
        if self.device == "cpu":
            self.logger.log("Use CPU for training")
        else:
            self.logger.log(f"Use GPU-{self.device} for training")
            model.to(self.device)
        self.model = model

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(args)

        self.logger.log(f"Using criterion: {args.criterion} for training")
        self.criterion = get_criterion(args)

        self.logger.log(f"Loading optimizer: {args.optimizer}")
        self.optimizer = get_optimizer(self.model, args)

        self.logger.log(f"Initializing learning rate scheduler: {self.args.lr_scheduler}")
        self.scheduler = get_scheduler(self.optimizer, args)

        if self.args.amp_scaler:
            self.logger.log(f"Using amp scaler.")
            self.scaler = get_amp_scaler(args)
        else:
            self.scaler = None

        self.trainer = self._build_trainer(args)

        if args.resume_ckpt:
            self._try_resume(args.resume_ckpt)

    def run(self) -> None:
        args = self.args
        trainer = self.trainer

        self.logger.log(f"Trainer: {args.trainer}")
        self.logger.log(f"Start training: steps={args.train_steps}")

        if not args.only_inference:
            start = time.time()
            trainer.train(train_steps=args.train_steps, val_steps=args.val_steps)
            tr_time = time.time() - start
            self.logger.log(f"Training time: {tr_time:.2f}s")

        start = time.time()
        test_metric = get_test_metric(args)
        total_loss, total_metric = trainer.test(test_steps=args.test_steps, metric=test_metric, loader=self.test_loader)
        if isinstance(total_metric, torch.Tensor):
            total_metric = total_metric.item()
        self.logger.log(f"[Test] — loss: {total_loss:.6f} | metric: {total_metric}")
        te_time = time.time() - start
        self.logger.log(f"Testing time: {te_time:.2f}s")

        trainer.save()
        self.logger.log("Saved final checkpoint.")

        self.logger.close()

    def _init(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        load_config(args, parser)
        set_seed(args.seed)

        base = f"{get_results_path()}/logs/{args.name}"
        if args.resume_ckpt:
            run_dir = self._infer_resume_run_dir(args.resume_ckpt)
            print(f"Resume run: {run_dir}")
        elif args.log:
            run_dir = self._next_run_dir(base)
        else:
            run_dir = base
            os.makedirs(run_dir, exist_ok=True)

        self._save_json(self._args_to_dict(args), os.path.join(run_dir, "settings.json"))
        self.log_path = run_dir
        self.logger = SimpleLogger(run_dir if args.log else None)

    def _args_to_dict(self, args) -> Dict[str, Any]:
        if hasattr(args, "__dict__"):
            d = {k: v for k, v in vars(args).items()}
        elif hasattr(args, "__dataclass_fields__"):
            d = asdict(args)
        else:
            d = {"args_repr": str(args)}
        return d

    def _infer_resume_run_dir(self, resume_ckpt_or_dir: str, base_name: str = "checkpoints") -> str:
        p = os.path.abspath(resume_ckpt_or_dir)
        if os.path.isdir(p):
            # If they pointed at the checkpoints subdir, go one up
            return os.path.dirname(p) if os.path.basename(p) == base_name else p
        d = os.path.dirname(p)
        return os.path.dirname(d) if os.path.basename(d) == base_name else d

    def _next_run_dir(self, base: str) -> str:
        os.makedirs(base, exist_ok=True)
        runs = [d for d in os.listdir(base) if d.startswith("run")]
        run_numbers = [int(d[3:]) for d in runs if d[3:].isdigit()]
        next_run_number = max(run_numbers) + 1 if run_numbers else 0
        run_dir = os.path.join(base, f"run{next_run_number}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _save_json(self, obj: Dict[str, Any], path: str):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def _build_trainer(self, args: argparse.Namespace) -> Trainer:
        trainer_name = args.trainer
        if trainer_name not in TRAINERS:
            raise ValueError(f"Unknown trainer '{trainer_name}'. "
                             f"Available: {', '.join(TRAINERS.keys())}")
        trainer_args = dict(
            model=self.model,
            criterion=self.criterion,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            optimizer=self.optimizer,
            max_grad_norm=self.args.max_grad_norm,
            scaler=self.scaler,
            lr_scheduler=self.scheduler,
            validate_every_steps=args.validate_every_steps,
            early_stopping_cfg=self.args.early_stopping_cfg,
            device=self.device,
            worker=self,
            logger=self.logger,
            PATH=self.log_path,
            log_every_steps=args.log_every_steps,
            save_every_steps=args.save_every_steps,
        )

        if trainer_name == "adaprune_trainer":
            trainer_args.update(dict(
                wanted_density=args.wanted_density,
            ))
        return TRAINERS[trainer_name](**trainer_args)

    def _try_resume(self, ckpt_path: str) -> None:
        try:
            self.trainer.load(ckpt_path)
            self.logger.log(f"Resumed from checkpoint: {ckpt_path}")
        except Exception as e:
            self.logger.log(f"WARN: resume failed ({e}). Continuing fresh.")

