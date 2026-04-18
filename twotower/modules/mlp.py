from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        bias: bool = True,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._out_size = out_size
        self._in_size = in_size
        self._linear: nn.Linear = nn.Linear(
            self._in_size,
            self._out_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._activation_fn(self._linear(input))


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        bias: bool = True,
        activation: Union[
            str,
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        activation_on_last: bool = True,
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation = torch.relu
        elif activation == "sigmoid":
            activation = torch.sigmoid

        if not isinstance(activation, str):
            self._mlp: torch.nn.Module = torch.nn.Sequential(
                *[
                    Perceptron(
                        layer_sizes[i - 1] if i > 0 else in_size,
                        layer_sizes[i],
                        bias=bias,
                        activation=torch.nn.Identity(),
                        device=device,
                        dtype=dtype,
                    )
                    for i in range(len(layer_sizes))
                ]
            )
        else:
            if activation == "swish_layernorm":
                self._mlp: torch.nn.Module = torch.nn.Sequential(
                    *[
                        Perceptron(
                            layer_sizes[i - 1] if i > 0 else in_size,
                            layer_sizes[i],
                            bias=bias,
                            activation=torch.nn.Identity(),
                            device=device,
                        )
                        for i in range(len(layer_sizes))
                    ]
                )
            else:
                assert (
                    ValueError
                ), "This MLP only support str version activation function of relu, sigmoid, and swish_layernorm"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._mlp(input)