import torch
import glotnet.cpp_extensions as ext

class Convolution(torch.nn.Conv1d):
    """ Causal convolution with optional FILM conditioning """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 causal: bool = True,
                 use_film: bool = False
                 ):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride,
                         padding, dilation,
                         groups, bias, padding_mode,
                         device, dtype)
        self.causal = causal
        self.use_film = use_film
        self._impl = ext.Convolution(in_channels, out_channels,
                                     kernel_size, dilation, use_film)

    @property
    def receptive_field(self) -> int:
        """ Receptive field length """
        return (self.kernel_size[0] - 1) * self.dilation[0] + 1

    def forward(self,
                input: torch.Tensor,
                cond_input: torch.Tensor = None,
                sequential: bool = False
                ) -> torch.Tensor:
        """
        Args:
            input shape is (batch, in_channels, time)
            cond_input: conditioning input
                
                shape = (batch, 2*out_channels, time) if self.use_film else (batch, out_channels, time)
        Returns:
            output shape is (batch, out_channels, time)
        """

        if cond_input is not None:
            if self.use_film and cond_input.size(1) != 2 * self.out_channels:
                raise ValueError(f"Cond input number of channels mismatch."
                                 f"Expected {2*self.out_channels}, got {cond_input.size(1)}")
            if not self.use_film and cond_input.size(1) != self.out_channels:
                raise ValueError(f"Cond input number of channels mismatch."
                                 f"Expected {self.out_channels}, got {cond_input.size(1)}")
            if cond_input.size(2) != input.size(2):
                raise ValueError(f"Mismatching timesteps, "
                                 f"input has {input.size(2)}, cond_input has {cond_input.size(2)}")

        if sequential:
            return ConvolutionFunction.apply(
                self._impl,
                input, cond_input,
                *self.parameters())
        else:
            return self._forward_native(input=input, cond_input=cond_input, causal=self.causal)

    def _forward_native(self, input: torch.Tensor,
                     cond_input: torch.Tensor,
                     causal:bool=True) -> torch.Tensor:
        """ Native torch conv1d with causal padding

        Args:
            input shape is (batch, in_channels, time)
            cond_input: conditioning input
                shape = (batch, 2*out_channels, time) if self.use_film else (batch, out_channels, time)
        Returns:
            output shape is (batch, out_channels, time)

        """
        
        if causal:
            padding = self.dilation[0] * self.stride[0] * (self.kernel_size[0]-1)
            if padding > 0:
                input = torch.nn.functional.pad(input, (padding, 0))
            output = torch.nn.functional.conv1d(
                input, self.weight, bias=self.bias,
                stride=self.stride, padding=0,
                dilation=self.dilation, groups=self.groups)
        else:
            output = torch.nn.functional.conv1d(
                input, self.weight, bias=self.bias,
                stride=self.stride, padding='same',
                dilation=self.dilation, groups=self.groups)

        if cond_input is not None:
            if self.use_film:
                b, a = torch.chunk(cond_input, 2, dim=1)
                output = a * output + b
            else:
                output = output + cond_input
        return output

class ConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, impl, input, cond_input=None, *params):
        """ Dilated covolution bindings forward pass

        Args:
            input: tensor of shape (batch, channels, time)
            cond_input: (default = None)

        """
        weight, bias = params

        input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        input = input.contiguous()
        if cond_input is not None:
            cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            cond_input = cond_input.contiguous()
       
        conv = impl
        conv.set_weight(weight)
        conv.set_bias(bias)
        if cond_input is None:
            output, = conv.forward(input)
        else:
            output, = conv.forward_cond(input, cond_input)

        output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        return output 

    def backward(self, *output_grads):
        raise NotImplementedError