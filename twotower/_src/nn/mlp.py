from typing import Callable

import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        bias: bool = True,
        activation: Callable[[], nn.Module] = nn.ReLU,
        bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.bn: nn.Module = nn.BatchNorm1d(out_size) if bn else nn.Identity()
        self.activation = activation()
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.bn(self.linear(x))))


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: tuple[int, ...],
        bias: bool = True,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        sizes = (in_size,) + layer_sizes
        self.layers = nn.ModuleList(
            [
                Perceptron(sizes[i], sizes[i + 1], bias=bias, activation=activation, bn=True, dropout=dropout)
                for i in range(len(sizes) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
