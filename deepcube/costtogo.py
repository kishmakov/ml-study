"""Cost-to-go approximations for search.

``cost_to_go(state_float) -> float``.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from puzzle import Puzzle, StateFloat


VALUE_NET_HIDDEN_DIM = 512
VALUE_NET_RESNET_DIM = 384
VALUE_NET_NUM_RESNET_BLOCKS = 2


class CostToGo:
    """Trivial heuristic that turns A* into uniform-cost search."""

    def __call__(self, state: StateFloat) -> float:
        return 0.0


class ResnetBlock(nn.Module):
    """Two-layer residual block used by the value head."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: Any) -> Any:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x + residual)


class CostToGoNet(nn.Module):
    """DeepCube-style value network for flat puzzle state features."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        resnet_dim: int,
        num_resnet_blocks: int,
    ) -> None:
        super().__init__()
        assert input_size > 0, "input size must be positive"
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, resnet_dim)
        self.blocks = nn.ModuleList(
            [ResnetBlock(resnet_dim) for _ in range(num_resnet_blocks)]
        )
        self.fc_out = nn.Linear(resnet_dim, 1)

    @classmethod
    def from_puzzle(cls, puzzle: Puzzle) -> Self:
        return cls(
            len(puzzle.cost_to_go_input()),
            VALUE_NET_HIDDEN_DIM,
            VALUE_NET_RESNET_DIM,
            VALUE_NET_NUM_RESNET_BLOCKS,
        )

    def forward(self, state: Any) -> Any:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        assert state.shape[1] == self.input_size, (
            f"CostToGoNet expects {self.input_size} inputs, "
            f"got {state.shape[1]}"
        )

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        for block in self.blocks:
            x = block(x)

        return self.fc_out(x).squeeze(-1)


class NeuralCostToGo:
    """Callable A* heuristic backed by ``CostToGoNet``."""

    def __init__(self, model: CostToGoNet, device: str) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, puzzle: Puzzle, device: str) -> Self:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = CostToGoNet.from_puzzle(puzzle)
        state_dict = _state_dict_from_checkpoint(checkpoint)
        model.load_state_dict(state_dict)
        return cls(model, device)

    def __call__(self, state: StateFloat) -> float:
        state_array = np.asarray(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_array).to(self.device)
        with torch.no_grad():
            value = self.model(state_tensor)
        return max(float(value.item()), 0.0)


def count_params(model: CostToGoNet) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _state_dict_from_checkpoint(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint
