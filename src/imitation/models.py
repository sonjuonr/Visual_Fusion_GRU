"""PyTorch models for heatmap-based action classification."""

from __future__ import annotations

import torch
from torch import nn

from imitation.constants import NUM_ACTIONS


def _activation_module(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class HeatmapActionMLP(nn.Module):
    """
    Simple classifier head on top of the existing heatmap backbone output.

    The backbone remains the repo's current heatmap-generation path; this model
    only consumes the flattened heatmap and predicts one of three actions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        dropout: float = 0.0,
        activation: str = "relu",
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(value) for value in hidden_dims], int(num_actions)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(_activation_module(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs.reshape(obs.shape[0], -1)
        return self.network(obs.float())
