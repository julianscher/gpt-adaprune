from typing import List

import torch
import torch.nn as nn
from transformers import GPT2Model

class PretrainedModel(nn.Module):
    def __init__(self, n_dims: int, n_positions: int, n_embd: int=768) -> None:
        super(PretrainedModel, self).__init__()
        self.name = f"pretrained_gpt2_embd={n_embd}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model.from_pretrained('gpt2')
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b: torch.Tensor, ys_b: torch.Tensor) -> torch.Tensor:
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs: torch.Tensor, ys: torch.Tensor, inds: List[int]=None) -> torch.Tensor:
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs