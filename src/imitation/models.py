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


class HeatmapActionGRU(nn.Module):
    """
    Recurrent student for target-loss recovery.

    It consumes a sequence of flattened heatmaps and predicts an action at each
    timestep. During rollout, the policy adapter keeps the GRU hidden state
    across environment steps, giving the student memory of where the target was
    before it disappeared from the current frame.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        mlp_hidden_dims: list[int] | tuple[int, ...] = (),
        activation: str = "relu",
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

        head_dims = [self.hidden_dim, *[int(value) for value in mlp_hidden_dims], int(num_actions)]
        head_layers: list[nn.Module] = []
        for idx in range(len(head_dims) - 2):
            head_layers.append(nn.Linear(head_dims[idx], head_dims[idx + 1]))
            head_layers.append(_activation_module(activation))
            if dropout > 0.0:
                head_layers.append(nn.Dropout(float(dropout)))
        head_layers.append(nn.Linear(head_dims[-2], head_dims[-1]))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.reshape(1, 1, -1)
        elif obs.ndim == 2:
            obs = obs.unsqueeze(1)
        elif obs.ndim != 3:
            raise ValueError(f"Expected obs with 1, 2, or 3 dims, got shape={tuple(obs.shape)}")

        obs = obs.reshape(obs.shape[0], obs.shape[1], -1).float()
        recurrent_out, next_hidden = self.gru(obs, hidden_state)
        logits = self.head(recurrent_out)
        return logits, next_hidden


def build_heatmap_action_model(model_type: str, model_config: dict[str, object]) -> nn.Module:
    normalized = str(model_type).strip().lower()
    if normalized == "mlp":
        return HeatmapActionMLP(**model_config)
    if normalized == "gru":
        return HeatmapActionGRU(**model_config)
    raise ValueError(f"Unsupported heatmap action model_type: {model_type}")
