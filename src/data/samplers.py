import ast
from typing import List, Tuple, Generator

import torch


class DataSampler:
    def __init__(self, n_dims: int) -> None:
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name: str, n_dims: int, **kwargs) -> DataSampler:
    names_to_classes = {
        "gaussian": GaussianSampler,
        "gaussian_gap_sampler": GaussianGapSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


class GaussianSampler(DataSampler):
    def __init__(self, n_dims: int, bias: float=None, scale: float=None) -> None:
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, batch_size: int, n_points: int, n_dims_truncated: int=None, seeds: List[int]=None) -> torch.Tensor:
        if seeds is None:
            xs_b = torch.randn(batch_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(batch_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class GaussianGapSampler(DataSampler):
    """
    Standard normal features with controllable 'gaps' (low-density intervals).
    Gaps are specified in standard-normal space; scale/bias are applied AFTER thinning.
    """
    def __init__(self, n_dims: int, bias: float=None, scale: float=None, gaps: List[Tuple[float, float]]=None,
                 gap_rate: float=0.5) -> None:
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        if gaps is not None:
            self.gaps = [ast.literal_eval(gap) for gap in gaps] # e.g., [(-1.0, -0.2), (0.2, 1.0)]
        else:
            self.gaps = []
        self.gap_rate = float(gap_rate)

    @staticmethod
    def _thin_gaps_(xs: torch.Tensor, gaps: List[Tuple[float, float]], gap_rate: float,
                    generator: Generator=None) -> torch.Tensor:
        """In-place: resample coordinates falling in any gap with prob = gap_rate."""
        if not gaps or gap_rate <= 0.0:
            return xs
        # build 'in-gap' mask
        in_gap = torch.zeros_like(xs, dtype=torch.bool)
        for lo, hi in gaps:
            in_gap |= (xs >= lo) & (xs <= hi)
        # Bernoulli thinning
        resample_mask = in_gap & (torch.rand_like(xs) < gap_rate)
        if resample_mask.any():
            new_vals = torch.randn(int(resample_mask.sum()), generator=generator, dtype=xs.dtype, device=xs.device)
            xs.view(-1)[resample_mask.view(-1)] = new_vals
        return xs

    def sample_xs(self, batch_size: int, n_points: int, n_dims_truncated: int=None,
                  seeds: List[int]=None) -> torch.Tensor:
        if seeds is None:
            xs_b = torch.randn(batch_size, n_points, self.n_dims)
            self._thin_gaps_(xs_b, self.gaps, self.gap_rate, generator=None)
        else:
            xs_b = torch.zeros(batch_size, n_points, self.n_dims)
            gen = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                gen.manual_seed(int(seed))
                x = torch.randn(n_points, self.n_dims, generator=gen)
                self._thin_gaps_(x, self.gaps, self.gap_rate, generator=gen)
                xs_b[i] = x

        # affine transform (shifts gaps accordingly in observed space)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b = xs_b + self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
