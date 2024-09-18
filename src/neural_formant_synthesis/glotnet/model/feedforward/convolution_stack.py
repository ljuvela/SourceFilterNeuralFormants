import torch
from typing import List
from .convolution_layer import ConvolutionLayer


class ConvolutionStack(torch.nn.Module):
    """
    Wavenet Convolution Stack

    Uses a gated activation and residual connections by default
    """

    def __init__(self, channels, skip_channels, kernel_size, dilations=[1], bias=True, device=None, dtype=None,
                 causal=True,
                 activation="gated",
                 use_residual=True,
                 use_1x1_block_out=True,
                 cond_channels=None,
                 ):
        super().__init__()

        self.channels = channels
        self.skip_channels = skip_channels
        self.activation = activation
        self.dilations = dilations
        self.use_residual = use_residual
        self.use_1x1_block_out = use_1x1_block_out
        self.use_conditioning = cond_channels is not None
        self.cond_channels = cond_channels
        self.causal = causal
        self.num_layers = len(dilations)

        self.layers = torch.nn.ModuleList()
        for i, d in enumerate(dilations):
            use_output_transform = self.use_1x1_block_out
            # Always disable output 1x1 for last layer
            if i == self.num_layers - 1:
                use_output_transform = False
            # Add ConvolutionLayer to Stack
            self.layers.append(
                ConvolutionLayer(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, dilation=d, bias=bias, device=device, dtype=dtype,
                    causal=causal,
                    activation=activation,
                    use_output_transform=use_output_transform,
                    cond_channels=self.cond_channels,
                    skip_channels=self.skip_channels,
                )
            )

    @property
    def weights_conv(self):
        return [layer.conv.weight for layer in self.layers]
    
    @property
    def biases_conv(self):
        return [layer.conv.bias for layer in self.layers]

    @property
    def weights_out(self):
        return [layer.out.weight for layer in self.layers]

    @property
    def biases_out(self):
        return [layer.out.bias for layer in self.layers]

    @property
    def weights_skip(self):
        return [layer.skip.weight for layer in self.layers]

    @property
    def biases_skip(self):
        return [layer.skip.bias for layer in self.layers]

    @property
    def weights_cond(self):
        if self.use_conditioning:
            return [layer.cond_1x1.weight for layer in self.layers]
        else:
            return None

    @property
    def biases_cond(self):
        if self.use_conditioning:
            return [layer.cond_1x1.bias for layer in self.layers]
        else:
            return None

    @property
    def receptive_field(self):
        return sum([l.receptive_field for l in self.layers])

    def forward(self, input, cond_input=None, sequential=False):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, channels, timesteps)
            cond_input (optional),
                torch.Tensor of shape (batch_size, cond_channels, timesteps)
            sequential (optional), 
                if True, use CUDA compatible parallel implementation
                if False, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, channels, timesteps)
            skips, list of torch.Tensor of shape (batch_size, out_channels, timesteps)
        
        """

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if sequential:
            raise NotImplementedError("Sequential mode not implemented")
        else:
            return self._forward_native(input=input, cond_input=cond_input)

    def _forward_native(self, input, cond_input):
        x = input
        skips = []
        for layer in self.layers:
            h = x
            x, s = layer(x, cond_input, sequential=False)
            x = x + h  # residual connection
            skips.append(s)
        return x, skips
