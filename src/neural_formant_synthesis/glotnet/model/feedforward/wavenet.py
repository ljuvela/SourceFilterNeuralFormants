import torch
from typing import List
import glotnet.cpp_extensions as ext
from .convolution_layer import ConvolutionLayer
from .convolution_stack import ConvolutionStack

class WaveNet(torch.nn.Module):
    """ Feedforward WaveNet """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 residual_channels: int,
                 skip_channels: int,
                 kernel_size: int,
                 dilations: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
                 causal: bool = True,
                 activation: str = "gated",
                 use_residual: bool = True,
                 cond_channels: int = None,
                 cond_net: torch.nn.Module = None,
                 ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels
        self.use_conditioning = cond_channels is not None
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.num_layers = len(dilations)
        self.causal = causal

        # Layers
        self.input = ConvolutionLayer(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=1, activation="tanh", use_output_transform=False)
        self.stack = ConvolutionStack(
            channels=self.residual_channels,
            skip_channels=self.skip_channels,
            kernel_size=self.kernel_size,
            dilations=dilations,
            activation=self.activation,
            use_residual=True,
            causal=self.causal,
            cond_channels=cond_channels)
        # TODO: output layers should be just convolution, These hanve conv and Out
        self.output1 = ConvolutionLayer(
            in_channels=self.skip_channels,
            out_channels=self.residual_channels,
            kernel_size=1, activation="tanh", use_output_transform=False)
        self.output2 = ConvolutionLayer(
            in_channels=self.residual_channels,
            out_channels=self.output_channels,
            kernel_size=1, activation="linear", use_output_transform=False)

        self.cond_net = cond_net

    @property
    def output_weights(self):
        return [self.output1.conv.weight, self.output2.conv.weight]

    @property
    def output_biases(self):
        return [self.output1.conv.bias, self.output2.conv.bias]
    
    @property
    def receptive_field(self):
        return self.input.receptive_field + self.stack.receptive_field \
            + self.output1.receptive_field + self.output2.receptive_field


    def forward(self, input, cond_input=None):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, input_channels, timesteps)
            cond_input (optional),
                torch.Tensor of shape (batch_size, cond_channels, timesteps)

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)

        """

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        x = input
        x = self.input(x)
        _, skips = self.stack(x, cond_input) # TODO self.stack must be called something different, torch.stack is different
        x = torch.stack(skips, dim=0).sum(dim=0)
        x = self.output1(x)
        x = self.output2(x)
        return x


    def inference(self, input, cond_input=None):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, input_channels, timesteps)
            cond_input (optional),
                torch.Tensor of shape (batch_size, cond_channels, timesteps)

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)

        """

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        output = WaveNetFunction.apply(
            input,
            self.stack.weights_conv, self.stack.biases_conv,
            self.stack.weights_out, self.stack.biases_out,
            self.stack.weights_skip, self.stack.biases_skip,
            self.stack.weights_cond, self.stack.biases_cond,
            self.input.conv.weight, self.input.conv.bias,
            self.output_weights, self.output_biases,
            self.dilations, self.use_residual, self.activation,
            cond_input
        )
        return output

class WaveNetFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                stack_weights_conv: List[torch.Tensor], stack_biases_conv: List[torch.Tensor],
                stack_weights_out: List[torch.Tensor], stack_biases_out: List[torch.Tensor],
                stack_weights_skip: List[torch.Tensor], stack_biases_skip: List[torch.Tensor],
                stack_weights_cond: List[torch.Tensor], stack_biases_cond: List[torch.Tensor],
                input_weight: torch.Tensor, input_bias: torch.Tensor,
                output_weights: List[torch.Tensor], output_biases: List[torch.Tensor],
                dilations: List[int], use_residual: bool, activation: str,
                cond_input: torch.Tensor = None, time_major: bool = True):

        num_layers = len(dilations)

        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            if cond_input is not None:
                cond_input = cond_input.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        input = input.contiguous()
        if cond_input is not None:
            cond_input = cond_input.contiguous()

        if cond_input is None:
            output, = ext.wavenet_forward(input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation)
        else:
            output, = ext.wavenet_cond_forward(input, cond_input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                stack_weights_cond, stack_biases_cond,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation)

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError
