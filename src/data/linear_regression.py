import argparse
from typing import Dict, Iterable, Tuple, Callable

import numpy as np
import torch

from torch.utils.data import IterableDataset, DataLoader

from src.data.curriculum import Curriculum
from src.data.samplers import get_data_sampler, DataSampler
from src.data.tasks import get_task_sampler
from src.data.utils import normalize_dataset_over_interval


class LinearRegressionDataLoader:
    def __init__(self, loader_type: str, data_sampler: DataSampler, task_sampler: Callable,
                 n_points: int, n_dims_truncated: int,
                 data_sampler_args: Dict, task_sampler_args: Dict,
                 norm: bool, device: str,
                 batch_size: int, buff_size: int,
                 shuffle: bool=False, seed: int=0,
                 curriculum:Curriculum=None) -> None:

        if buff_size % batch_size != 0:
            raise Exception("buff_size must be evenly div by bat_size")

        self.loader_type = loader_type
        self.device = device
        self.batch_size = batch_size
        self.buff_size = buff_size
        self.shuffle = shuffle
        self.rnd = np.random.RandomState(seed)

        self.ds = LinearRegressionDataset(
            data_sampler=data_sampler,
            task_sampler=task_sampler,
            n_points=n_points,
            batch_size=batch_size,
            n_dims_truncated=n_dims_truncated,
            data_sampler_args=data_sampler_args,
            task_sampler_args=task_sampler_args,
            norm=norm,
            device=device,
        )
        self._it = iter(DataLoader(self.ds, batch_size=None, pin_memory=True))

        # optional
        self.curriculum = curriculum
        self._cur_n_points = self.curriculum.n_points if curriculum else n_points
        self._cur_n_dims = self.curriculum.n_dims_truncated if curriculum else n_dims_truncated

        # ==== Buffer ====#
        self.ptr = 0
        self.x_data = None  # [buff_size, D]
        self.y_data = None  # [buff_size]
        self.refill_buffer()

    def _advance_curriculum(self) -> None:
        if not self.curriculum or self.loader_type != "train":
            return
        self.curriculum.update()
        n_points_new = self.curriculum.n_points
        n_dims_new = self.curriculum.n_dims_truncated
        if n_points_new != self._cur_n_points or n_dims_new != self._cur_n_dims:
            self._cur_n_points, self._cur_n_dims = n_points_new, n_dims_new
            # reconfigure dataset + reset iterator + refill buffer
            self.ds.configure(n_points=n_points_new, n_dims_truncated=n_dims_new)
            self._it = iter(DataLoader(self.ds, batch_size=None, pin_memory=True))
            self.refill_buffer()

    def refill_buffer(self) -> None:
        self.ptr = 0
        xs, ys, ct = [], [], 0
        while ct < self.buff_size:  # While buffer is not completely filled
            try:
                x, y = next(self._it)  # sample one batch from the dynamic dataset
            except StopIteration:
                # This branch will not be reached, because the dynamic dataset has potentially infinite elements
                self._it = iter(DataLoader(self.ds, batch_size=None, pin_memory=True))
                x, y = next(self._it)
            xs.append(x.detach())
            ys.append(y.detach())
            ct += x.shape[0]  # + batch_size

        # Concatenate tensors along 0 dimension and drop excess items to match buffer size
        X = torch.cat(xs, 0)[:self.buff_size]
        y = torch.cat(ys, 0)[:self.buff_size]

        if self.shuffle:
            idx = torch.tensor(self.rnd.permutation(self.buff_size), device=X.device)
            X, y = X.index_select(0, idx), y.index_select(0, idx)

        self.x_data = X.to(self.device, dtype=torch.float32)
        self.y_data = y.to(self.device, dtype=torch.float32)

        return 0  # buffer successfully loaded

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # We encapsulate the curriculum update step inside the data loader, no need to explicitly call it in the trainer
        self._advance_curriculum()

        # Get next batch from the buffer
        res = 0
        if self.ptr + self.batch_size > self.buff_size:
            # print(" ** refilling buffer ** ")
            res = self.refill_buffer()  # 0 = success

        if res == 0:
            start = self.ptr
            end = self.ptr + self.batch_size
            x = self.x_data[start:end, ...]
            y = self.y_data[start:end]
            self.ptr += self.batch_size
            return x, y

        self.refill_buffer()  # Refill buffer
        raise StopIteration


class LinearRegressionDataset(IterableDataset):
    def __init__(self, data_sampler: DataSampler, task_sampler: Callable, n_points: int, batch_size: int,
                 n_dims_truncated: int, data_sampler_args: Dict, task_sampler_args: Dict,
                 norm: bool, device: str) -> None:
        self.data_sampler = data_sampler
        self.task_sampler = task_sampler
        self.n_points = n_points
        self.batch_size = batch_size
        self.n_dims_truncated = n_dims_truncated
        self.data_sampler_args = data_sampler_args
        self.task_sampler_args = task_sampler_args
        self.norm = norm
        self.device = device

    def configure(self, *, n_points:int=None, n_dims_truncated:int=None, batch_size:int=None) -> None:
        if n_points is not None:
            self.n_points = n_points
        if n_dims_truncated is not None:
            self.n_dims_truncated = n_dims_truncated
        if batch_size is not None:
            self.batch_size = batch_size

    def __iter__(self) -> Iterable:
        while True:
            xs = self.data_sampler.sample_xs(
                self.batch_size,
                self.n_points,
                self.n_dims_truncated,
                **self.data_sampler_args,
            ).to(self.device)
            task = self.task_sampler(**self.task_sampler_args)
            ys = task.evaluate(xs).to(self.device)

            if self.norm:
                xs = normalize_dataset_over_interval(xs, -0.7, 0.7)

            yield xs, ys


def get_dataloaders(args: argparse.Namespace) -> Tuple[
    LinearRegressionDataLoader, LinearRegressionDataLoader, LinearRegressionDataLoader]:
    n_dims_model = args.model["n_dims"]
    n_points_start = args.curriculum["points"]["start"]
    n_dims_trunc_start = args.curriculum["dims"]["start"]

    batch_size = args.batch_size
    data_name = args.data
    task_name = args.task
    task_kwargs = args.task_kwargs
    data_sampler_args = args.data_sampler_args
    normalized = args.normalized
    device = args.device
    seed = args.seed

    # Samplers (build at max dims; truncation is handled by dataset)
    data_sampler = get_data_sampler(data_name=data_name, n_dims=n_dims_model, **data_sampler_args)
    task_sampler = get_task_sampler(task_name=task_name, n_dims=n_dims_model, batch_size=batch_size, **task_kwargs)

    curriculum = Curriculum(args)

    def build(split_key: str, curriculum: Curriculum) -> LinearRegressionDataLoader:
        split = args.splits[split_key]
        n_batches = split["n_batches"]
        shuffle = split["shuffle"]
        # start at curriculum.start values; training will advance automatically per batch
        return LinearRegressionDataLoader(
            loader_type=split_key,
            data_sampler=data_sampler,
            task_sampler=task_sampler,
            n_points=n_points_start,
            n_dims_truncated=n_dims_trunc_start,
            data_sampler_args={},
            task_sampler_args={},
            norm=normalized,
            device=device,
            batch_size=batch_size,
            buff_size=batch_size * n_batches,
            shuffle=shuffle,
            seed=seed,
            curriculum=curriculum,
        )

    train_loader = build("train", curriculum)
    val_loader = build("val", curriculum)
    test_loader = build("test", curriculum)
    return train_loader, val_loader, test_loader
