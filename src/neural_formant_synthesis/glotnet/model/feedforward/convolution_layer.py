import torch
from typing import Tuple
from .convolution import Convolution


class ConvolutionLayer(torch.nn.Module):
    """
    Wavenet Convolution Layer (also known as Residual Block)

    Uses a gated activation and a 1x1 output transformation by default

    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 bias=True, device=None, dtype=None,
                 causal=True,
                 activation="gated",
                 use_output_transform=True,
                 cond_channels=None,
                 skip_channels=None,
                 ):
        super().__init__()

        residual_channels = out_channels
        self.activation = activation
        self.activation_fun, self.channel_mul = self._parse_activation(activation)
        self.use_output_transform = use_output_transform
        self.use_conditioning = cond_channels is not None
        self.cond_channels = cond_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.conv = Convolution(
            in_channels=in_channels,
            out_channels=self.channel_mul * residual_channels,
            kernel_size=kernel_size, dilation=dilation, bias=bias,
            device=device, dtype=dtype, causal=causal)
        # TODO: make parameter alloc conditional on use_output_transform
        self.out = Convolution(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype)
        if self.skip_channels is not None:
            self.skip = Convolution(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype)
        if self.use_conditioning:
            self.cond_1x1 = torch.nn.Conv1d(
                cond_channels, self.channel_mul * residual_channels,
                kernel_size=1, bias=True, device=device, dtype=dtype)
    
    @property
    def receptive_field(self):
        return self.conv.receptive_field + self.out.receptive_field

    activations = {
        "gated": ((torch.tanh, torch.sigmoid), 2),
        "tanh": (torch.tanh, 1),
        "linear": (torch.nn.Identity(), 1)
    }
    def _parse_activation(self, activation):
        activation_fun, channel_mul = ConvolutionLayer.activations.get(activation, (None, None))
        if channel_mul is None:
            raise NotImplementedError
        return activation_fun, channel_mul


    def forward(self, input, cond_input=None, sequential=False):
        """
        Args:
            input, torch.Tensor of shape (batch_size, in_channels, timesteps)
            sequential (optional),
                if True, use CUDA compatible parallel implementation
                if False, use custom C++ sequential implementation

        Returns:
            output, torch.Tensor of shape (batch_size, out_channels, timesteps)
            skip, torch.Tensor of shape (batch_size, out_channels, timesteps)

        """

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, \
                but conditioning input was provided at forward pass")

        if sequential:
            raise NotImplementedError
        else:
            return self._forward_native(input=input, cond_input=cond_input)


    def _forward_native(self, input, cond_input):
        c = self.cond_1x1(cond_input) if self.use_conditioning else None
        x = self.conv(input, cond_input=c)
        if self.channel_mul == 2:
            R = self.residual_channels
            x = self.activation_fun[0](x[:, :R, :]) * self.activation_fun[1](x[:, R:, :])
        else:
            x = self.activation_fun(x)

        if self.skip_channels is not None:
            skip = self.skip(x)

        if self.use_output_transform:
            output = self.out(x)
        else:
            output = x

        if self.skip_channels is None:
            return output
        else:
            return output, skip

