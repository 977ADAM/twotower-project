import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build a linear projection or MLP depending on whether hidden_dims is provided.

    When hidden_dims=(), returns a single Linear(input_dim, output_dim).
    When hidden_dims=(d1, d2, ...), inserts BatchNorm1d + ReLU + optional Dropout
    between each pair of layers and a final Linear(..., output_dim) with no activation.
    """
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.BatchNorm1d(h_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)
