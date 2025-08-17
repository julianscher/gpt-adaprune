import os
import random
from pathlib import Path

import torch
import numpy as np
from torch import nn

def get_project_root_path() -> Path:
    return Path(__file__).parent.parent.parent

def get_results_path(results_path=None) -> Path:
    return Path(os.path.join(get_project_root_path(), "results"))  \
        if not results_path else results_path

def get_configs_path() -> Path:
    return Path(os.path.join(get_project_root_path(), "configs"))


def set_seed(seed: int=None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)


def cross_entropy_loss(output: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    return nn.CrossEntropyLoss().to(device)(output, target.long())

def mean_squared_error(output: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    return nn.MSELoss().to(device)(output, target.float())

def mean_absolute_error(output: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    return nn.L1Loss().to(device)(output, target.float())

def binary_cross_entropy_loss(output: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    return nn.BCELoss().to(device)(output, target.long())


