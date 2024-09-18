"""
Copyright 2022 Lauri Juvela

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import glotnet.cpp_extensions as ext

def _gated_activation(x: torch.Tensor) -> torch.Tensor:

    assert x.size(1) % 2 == 0, f"Input must have an even number of channels, shape was {x.shape}"
    half = x.size(1) // 2
    return torch.tanh(x[:, :half, :]) * torch.sigmoid(x[:, half:, :])

class Activation(torch.nn.Module):
    """ Activation class """

    def __init__(self, activation="gated"):
        super().__init__()
        self.activation_str = activation
        if activation == "gated":
            self.activation_func = _gated_activation
        elif activation == "tanh":
            self.activation_func = torch.tanh
        elif activation == "linear":
            self.activation_func = torch.nn.Identity()

    def forward(self, input, use_extension=True):

        if use_extension:
            return ActivationFunction.apply(input, self.activation_str)
        else:
            return self.activation_func(input)


class ActivationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, 
                activation: str, time_major: bool = True
                ) -> torch.Tensor:

        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        
        input = input.contiguous()
        ctx.save_for_backward(input)
        ctx.activation_type = activation

        output, = ext.activations_forward(input, activation)

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        return output 

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError("Backward function not implemented for sequential processing")



class Tanh(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return TanhFunction.apply(input)


class TanhFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        
        output = torch.tanh(input)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, d_output: torch.Tensor):

        input, output = ctx.saved_tensors
        # d tanh(x) / dx = 1 - tanh(x) ** 2
        d_input = (1 - output ** 2) * d_output
        return d_input