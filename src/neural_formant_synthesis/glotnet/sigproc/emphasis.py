from .lfilter import LFilter
import torch
from .lfilter import ceil_division

class Emphasis(torch.nn.Module):
    """ Pre-emphasis and de-emphasis filter"""

    def __init__(self, alpha=0.85) -> None:
        """
        Args:
            alpha : pre-emphasis coefficient
        """
        super().__init__()
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.alpha = alpha
        self.lfilter = LFilter(n_fft=512, hop_length=256, win_length=512)
        self.register_buffer('coeffs', torch.tensor([1, -alpha], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """" Apply pre-emphasis to signal
        Args:
            x : input signal, shape (batch, channels, timesteps)

        Returns:
            y : output signal, shape (batch, channels, timesteps)

        """
        if not (self.alpha > 0):
            return x
        # expand coefficients to batch size and number of frames
        b = self.coeffs.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, ceil_division(x.size(-1), self.lfilter.hop_length))
        # filter
        return self.lfilter(x, b=b, a=None)

    def emphasis(self, x: torch.Tensor) -> torch.Tensor:
        """" Apply pre-emphasis to signal
        Args:
            x : input signal, shape (batch, channels, timesteps)

        Returns:
            y : output signal, shape (batch, channels, timesteps)

        """
        return self.forward(x)

    def deemphasis(self, x: torch.Tensor) -> torch.Tensor:
        """ Remove emphasis from signal
        Args:
            x : input signal, shape (batch, channels, timesteps)
        Returns:
            y : output signal, shape (batch, channels, timesteps)
        """
        if not (self.alpha > 0):
            return x
        # expand coefficients to batch size and number of frames
        a = self.coeffs.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, ceil_division(x.size(-1), self.lfilter.hop_length))
        # filter
        return self.lfilter(x, b=None, a=a)