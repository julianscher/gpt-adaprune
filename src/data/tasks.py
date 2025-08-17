import torch
from typing import Dict, Optional, Callable, List, Generator


class Task:
    def __init__(self, n_dims: int, batch_size: int, pool_dict: Dict=None, seeds: List[int]=None) -> None:
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

def get_task_sampler(task_name: str, n_dims: int, batch_size: int, pool_dict: Dict=None,
                     num_tasks: int=None, **kwargs) -> Callable:
    task_names_to_classes = {
        "linear_regression":            LinearRegression,
        "dense_zero_linear_regression":     DenseZeroLinearRegression,
        "scaling_linear_regression": ScalingLinearRegression,
        "tail_linear_regression":       TailLinearRegression,
        "noisy_linear_regression":      NoisyLinearRegression,
    }

    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims: int, batch_size: int, pool_dict: Dict=None,
                 seeds: List[int]=None, scale: float=1) -> None:
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        def sampler_batch(batch):
            return torch.randn(batch, n_dims, 1)

        def sampler_single(generator):
            return torch.randn(n_dims, 1, generator=generator)

        self.w_b = _init_w_b(n_dims, batch_size, pool_dict, seeds, sampler_batch, sampler_single)

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        return _evaluate_linear(xs_b, self.w_b, self.scale)

    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, **kwargs) -> Dict[str, torch.Tensor]:  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}


class DenseZeroLinearRegression(Task):
    def __init__(self, n_dims: int, batch_size: int, pool_dict: Dict=None, seeds: List[int]=None,
                 scale: float=1.0, sample_fraction: float=0.1) -> None:
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = float(scale)
        sample_fraction = float(sample_fraction)

        def sampler_batch(batch: int) -> torch.Tensor:
            w = torch.zeros(batch, n_dims, 1)
            k = max(1, int(sample_fraction * batch))
            w[torch.randperm(batch)[:k]] = torch.randn(k, n_dims, 1)
            return w

        def sampler_single(generator: Generator) -> torch.Tensor:
            keep = torch.rand(1, generator=generator).item() < sample_fraction
            return torch.randn(n_dims, 1, generator=generator) if keep else torch.zeros(n_dims, 1)

        self.w_b = _init_w_b(n_dims, batch_size, pool_dict, seeds, sampler_batch, sampler_single)

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:  # [B,N,D] x [B,D,1] -> [B,N]
        return _evaluate_linear(xs_b, self.w_b, self.scale)

    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, sample_fraction: float=0.1, **kwargs) -> Dict[str, torch.Tensor]:
        sf = float(sample_fraction)
        w = torch.zeros(num_tasks, n_dims, 1)
        k = max(1, int(sf * num_tasks))
        w[torch.randperm(num_tasks)[:k]] = torch.randn(k, n_dims, 1)
        return {"w": w}


class ScalingLinearRegression(Task):
    def __init__(self, n_dims: int, batch_size: int, pool_dict: Dict=None, seeds: List[int]=None,
                 linear_scale: float=1.0, global_scale: float=0.1) -> None:
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = float(linear_scale)
        global_scale = float(global_scale)

        def sampler_batch(batch: int) -> torch.Tensor:
            return torch.randn(batch, n_dims, 1) * global_scale

        def sampler_single(generator: Generator) -> torch.Tensor:
            return torch.randn(n_dims, 1, generator=generator) * global_scale

        self.w_b = _init_w_b(n_dims, batch_size, pool_dict, seeds, sampler_batch, sampler_single)

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        return _evaluate_linear(xs_b, self.w_b, self.scale)

    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, global_scale: float=0.1, **kwargs) -> Dict[str, torch.Tensor]:
        global_scale = float(global_scale)
        return {"w": torch.randn(num_tasks, n_dims, 1) * global_scale}


class TailLinearRegression(Task):
    def __init__(self, n_dims: int, batch_size: int, pool_dict: Dict=None,
                 seeds: List[int]=None, scale: float=1.0) -> None:
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = float(scale)

        def sampler_batch(batch: int) -> torch.Tensor:
            return (torch.rand(batch, n_dims, 1) * -1.0) - 1.0  # U([-2,-1])

        def sampler_single(generator: Generator) -> torch.Tensor:
            return (torch.rand(n_dims, 1, generator=generator) * -1.0) - 1.0

        self.w_b = _init_w_b(n_dims, batch_size, pool_dict, seeds, sampler_batch, sampler_single)

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        return _evaluate_linear(xs_b, self.w_b, self.scale)

    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, **kwargs) -> Dict[str, torch.Tensor]:
        return {"w": (torch.rand(num_tasks, n_dims, 1) * -1.0) - 1.0}


class NoisyLinearRegression(Task):
    def __init__(self, n_dims: int, batch_size: int, pool_dict: Dict=None,
                 seeds: List[int]=None, scale: float=1.0) -> None:
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = float(scale)
        noise_std = n_dims / 20.0

        def sampler_batch(batch: int) -> torch.Tensor:
            return torch.randn(batch, n_dims, 1) + torch.rand(batch, n_dims, 1) * noise_std

        def sampler_single(generator: Generator) -> torch.Tensor:
            base = torch.randn(n_dims, 1, generator=generator)
            noise = torch.rand(n_dims, 1, generator=generator) * noise_std
            return base + noise

        self.w_b = _init_w_b(n_dims, batch_size, pool_dict, seeds, sampler_batch, sampler_single)

    def evaluate(self, xs_b: torch.Tensor) -> torch.Tensor:
        return _evaluate_linear(xs_b, self.w_b, self.scale)

    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, **kwargs) -> Dict[str, torch.Tensor]:
        noise_std = n_dims / 20.0
        return {"w": torch.randn(num_tasks, n_dims, 1) + torch.rand(num_tasks, n_dims, 1) * noise_std}


#=========== Helper functions ===========#

def _evaluate_linear(xs_b: torch.Tensor, w_b: torch.Tensor, scale: float) -> torch.Tensor:
    return scale * (xs_b @ w_b.to(xs_b.device))[:, :, 0]

def _init_w_b(n_dims: int, batch_size: int, pool_dict: Optional[Dict[str, torch.Tensor]], seeds: Optional[list],
              sampler_batch: Callable[[int], torch.Tensor],
              sampler_single: Callable[[torch.Generator], torch.Tensor]) -> torch.Tensor:

    if pool_dict is not None:
        assert "w" in pool_dict
        indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
        w_b = pool_dict["w"][indices]
        return w_b
    if seeds is not None:
        assert len(seeds) == batch_size
        w_b, generator = torch.zeros(batch_size, n_dims, 1), torch.Generator()
        for i, seed in enumerate(seeds):
            generator.manual_seed(int(seed))
            w_b[i] = sampler_single(generator)
        return w_b
    return sampler_batch(batch_size)
