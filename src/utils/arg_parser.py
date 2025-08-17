import argparse
import pathlib
from argparse import ArgumentParser, Namespace
from typing import Union, Callable

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, ExponentialLR

from src.utils.custom_lr_scheduler import CustomLRScheduler
from src.utils.utils import get_project_root_path, mean_squared_error, mean_absolute_error


class ArgParser(ArgumentParser):
    """
    Argument parser for all configs (pretraining, fine-tuning, pruning).
    Mirrors the fields in the YAML config files.
    """
    def __init__(self) -> None:
        super().__init__()

        # ===== General ===== #
        self.add_argument("--config", type=str, default="configs/pretraining/linear_regression.yml")
        self.add_argument("--log_dir", type=str, default="results/logs")
        self.add_argument("--seed", type=int, default=0)
        self.add_argument("--name", type=str, default="ap_tail_linear_regression")
        self.add_argument("--device", type=str, default="cpu")

        # ===== Dataset ===== #
        self.add_argument("--data", type=str, default="gaussian")
        self.add_argument("--data_sampler_args", type=dict, default={})
        self.add_argument("--batch_size", type=int, default=64)
        self.add_argument("--normalized", type=bool, default=False)
        self.add_argument("--splits", type=dict, default={
            "train": {"n_batches": 16, "shuffle": True},
            "val": {"n_batches": 16, "shuffle": False},
            "test": {"n_batches": 16, "shuffle": False},
        })
        self.add_argument("--task", type=str, default="linear_regression")
        self.add_argument("--task_kwargs", type=dict, default={"scale": 1.0})
        self.add_argument("--curriculum", type=dict, default={
            "dims": {"start": 5, "end": 20, "inc": 1, "interval": 2000},
            "points": {"start": 11, "end": 41, "inc": 2, "interval": 2000},
        })

        # ===== Training ===== #
        self.add_argument("--only_inference", type=bool, default=False)
        self.add_argument("--trainer", type=str, default="backpropagation_trainer")
        self.add_argument("--train_steps", type=int, default=500001)
        self.add_argument("--val_steps", type=int, default=10)
        self.add_argument("--test_steps", type=int, default=1000)
        self.add_argument("--validate_every_steps", type=int, default=1000)

        # Optimizer (for backprop trainer)
        self.add_argument("--optimizer", type=str, default="adam")
        self.add_argument("--lr", type=float, default=1e-4)
        self.add_argument("--weight_decay", type=float, default=1e-4)
        self.add_argument("--momentum", type=float, default=0.9)
        self.add_argument("--nesterov", type=bool, default=False)
        self.add_argument("--max_grad_norm", type=float, default=1.0)

        # Optimizer (additional for adaprune trainer)
        self.add_argument("--weights_lr", type=float, default=1e-3)
        self.add_argument("--mask_lr", type=float, default=1e-2)
        self.add_argument("--mask_decay", type=float, default=1e-4)
        self.add_argument("--wanted_density", type=float, default=1.0)

        # LR scheduler
        self.add_argument("--lr_scheduler", type=str, default="cosine")  # can also be "custom"
        self.add_argument("--min_lr", type=float, default=0.0)
        self.add_argument("--scheduler_cfg", type=dict, default=None)  # used for custom schedulers

        # Mixed precision / early stopping
        self.add_argument("--amp_scaler", type=bool, default=False)
        self.add_argument("--early_stopping_cfg", type=dict, default={"patience": 0, "min_delta": 0.0})

        # ===== Loss fn ===== #
        self.add_argument("--criterion", type=str, default="mse")

        # ===== Testing ===== #
        self.add_argument("--test_metric", type=str, default="mse")

        # ===== Architecture ===== #
        self.add_argument("--model", type=dict, default={
            "family": "gpt2",
            "n_embd": 256,
            "n_layer": 12,
            "n_head": 8,
            "n_dims": 20,
            "n_positions": 101,
        })

        # ===== Logging and Checkpointing ===== #
        self.add_argument("--log", type=bool, default=True)
        self.add_argument("--log_every_steps", type=int, default=10)
        self.add_argument("--save_every_steps", type=int, default=1000)
        self.add_argument("--resume_ckpt", type=str, default=None)
        self.add_argument("--out_dir", type=str, default="../results/logs/linear_regression")


    def parse(self) -> Namespace:
        return self.parse_args()



CUSTOM_TYPE_KEYS = {
}


def load_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.config:
        print(f"Loading arguments from {args.config}")
        config_path = pathlib.Path(get_project_root_path()) / args.config
        if not config_path.suffix:
            config_path = config_path.with_suffix(".yml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config_name = config.get("name")
        filename_without_ext = config_path.stem
        if config_name and config_name != filename_without_ext:
            raise ValueError(f"Configuration name '{config_name}' does not match filename '{filename_without_ext}'")

        base_config_name = config.get("base_config")
        if base_config_name:
            base_path = pathlib.Path(get_project_root_path()) / base_config_name
            if not base_path.suffix:
                base_path = base_path.with_suffix(".yml")
            with open(base_path, "r") as f:
                base_config = yaml.load(f, Loader=yaml.FullLoader)
            base_config.update(config)
            config = base_config

        none_keys = {key: None for key, value in config.items() if value == "None"}
        config.update(none_keys)

        for key, target_type in CUSTOM_TYPE_KEYS.items():
            if key in config and config[key] is not None:
                config[key] = target_type(config[key])

        parser.set_defaults(**config)
        args = parser.parse_args()  # CLI overrides YAML

    return args


def get_optimizer(model: nn.Module, args: argparse.Namespace) -> Union[Optimizer, str]:
    if args.trainer == "backpropagation_trainer":
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer '{args.optimizer}'")
    elif args.trainer in ["adaprune_trainer"]:
        optimizer = args.optimizer # optimization step is handled by model
    else:
        raise ValueError(f"There exists no optimizer for trainer '{args.trainer}'")

    return optimizer

def get_criterion(args: argparse.Namespace) -> Callable:
    if args.criterion == "mse":
        return mean_squared_error
    elif args.criterion == "mae":
        return mean_absolute_error
    else:
        raise ValueError(f"Unknown criterion '{args.criterion}'")


def get_test_metric(args: argparse.Namespace) -> Callable:
    if args.test_metric == "mse":
        return mean_squared_error


def get_scheduler(optimizer: Union[Optimizer, str], args: argparse.Namespace) \
        -> Union[None, ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR, CustomLRScheduler]:
    if args.trainer == "backpropagation_trainer":
        if args.lr_scheduler == "step":
            return StepLR(
                optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_gamma
            )
        elif args.lr_scheduler == "exponential":
            return ExponentialLR(
                optimizer,
                gamma=args.lr_gamma
            )
        elif args.lr_scheduler == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=args.train_steps,
                eta_min=args.min_lr
            )
        elif args.lr_scheduler == "":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=args.early_stopping_cfg["patience"],
                min_lr=args.min_lr,
            )
        elif args.lr_scheduler in ["none", ""]:
            return None
        else:
            raise ValueError(f"Specified lr scheduler {args.lr_scheduler} is not supported.")
    elif args.trainer in ["adaprune_trainer"]:
        if args.lr_scheduler == "custom":
            return CustomLRScheduler(**args.scheduler_cfg)
        elif not args.lr_scheduler:
            return None
    else:
        raise ValueError(f"There exists no learning rate scheduler for trainer '{args.trainer}'")


def get_amp_scaler(args: argparse.Namespace) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=args.amp_scaler)
