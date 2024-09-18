import torch
from typing import List
import glotnet.cpp_extensions as ext
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
            return ConvolutionStackFunction.apply(
                input,
                self.weights_conv, self.biases_conv,
                self.weights_out, self.biases_out,
                self.weights_skip, self.biases_skip,
                self.weights_cond, self.biases_cond,
                self.dilations, self.activation, self.use_residual,
                cond_input)
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

class ConvolutionStackFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                weights_conv: List[torch.Tensor], biases_conv: List[torch.Tensor],
                weights_out: List[torch.Tensor], biases_out: List[torch.Tensor],
                weights_skip: List[torch.Tensor], biases_skip: List[torch.Tensor],
                weights_cond: List[torch.Tensor], biases_cond: List[torch.Tensor],
                dilations: List[int], activation: str, use_residual: bool,
                cond_input: torch.Tensor = None, time_major: bool = True):

        num_layers = len(dilations)

        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            if cond_input is not None:
                cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)

        input = input.contiguous()
        if cond_input is not None:
            cond_input = cond_input.contiguous()

        ctx.save_for_backward(input, *weights_conv, *biases_conv,
                              *weights_out, *biases_out)
        ctx.dilations = dilations

        if cond_input is None:
            output, skips = ext.convolution_stack_forward(
                input, weights_conv, biases_conv, weights_out, biases_out,
                weights_skip, biases_skip,
                dilations, use_residual, activation)
        else:
            output, skips = ext.convolution_stack_cond_forward(
                input, cond_input, weights_conv, biases_conv, weights_out, biases_out,
                weights_skip, biases_skip, weights_cond, biases_cond,
                dilations, use_residual, activation)

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
            skips = skips.split(1, dim=1) # (B, L, T, C) -> L * (B, 1, T, C)
            skips = [s.squeeze(1).permute(0, 2, 1) for s in skips] # L * (B, 1, T, C) -> L * (B, C, T)
        else:
            raise NotImplementedError

        return output, skips

    def backward(self, d_output, d_skip):
        raise NotImplementedError("Backward function not implemented for sequential processing")
