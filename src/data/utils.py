import numpy as np
import torch


def normalize_dataset_over_interval(dataset: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return (b - a) * (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset)) + a

def normalize_dataset_over_interval_min_max_given(dataset: torch.Tensor, a: float, b: float,
                                                  min: float, max: float) -> torch.Tensor:
    return (b - a) * (dataset - min) / (max - min) + a


def normalize_dataset(dataset: torch.Tensor) -> torch.Tensor:
    return (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset))


def upscale_probs(probs: torch.Tensor) -> torch.Tensor:
    # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    return ((probs - np.min(probs)) / (np.max(probs) - np.min(probs))) * (1 - 0) + 0
