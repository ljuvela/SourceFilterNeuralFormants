import torch
from neural_formant_synthesis.glotnet.sigproc.lfilter import LFilter
from .levinson import spectrum_to_allpole


class LinearPredictor(torch.nn.Module):

    def __init__(self,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 win_length=512,
                 order=10):
        """
        Args:
            n_fft (int): FFT size
            hop_length (int): Hop length
            win_length (int): Window length
            order (int): Allpole order
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.order = order
        self.lfilter = LFilter(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.register_buffer('window', torch.hann_window(win_length))

    def estimate(self, x: torch.Tensor, root_scale:float=1.0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input audio (batch, time)

        Returns:
            torch.Tensor: Allpole coefficients (batch, order+1, time)
            torch.Tensor: Gain (batch, 1, time)
        """
        X = torch.stft(x, n_fft=self.n_fft,
                       hop_length=self.hop_length, win_length=self.win_length,
                       window=self.window,
                       return_complex=True)
        
        # power spectrum
        X = torch.abs(X)**2

        # transpose to (batch, time, freq)
        X = X.transpose(1, 2)

        # allpole coefficients
        a, g = spectrum_to_allpole(X, order=self.order, root_scale=root_scale)

        # transpose to (batch, order, num_frames)
        a = a.transpose(1, 2)
        # transpose to (batch, 1, num_frames)
        g = g.transpose(1, 2)

        A = torch.fft.rfft(a, n=512, dim=1).abs()
        H = g / (A + 1e-6)
        # H = 1 / (A + 1e-6) 

        return a, g, H

    def inverse_filter(self,
            x: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input audio (batch, time)
            a: Allpole coefficients (batch, order+1, time)

        Returns:
            Inverse filtered audio (batch, time)
        """
        # inverse filter
        e = self.lfilter.forward(x=x, b=a, a=None)

        return e

    def synthesis_filter(self,
            e: torch.Tensor,
            a: torch.Tensor,
            g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: Excitation signal (batch, channels, time)
            a: Allpole coefficients (batch, order+1, time)
            g: Gain for filters (batch, 1, time)

        Returns:
            torch.Tensor: Synthesis filtered audio (batch, time)
        """
        # inverse filter
        x = self.lfilter.forward(x=e, b=g, a=a)
        return x


    def prediction(self,
            x: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input audio (batch, channels, time)
            a: Allpole coefficients (batch, order+1, time)

        Returns:
            p: Linear prediction signal (batch, time)
        """
        a_pred = -a
        a_pred[:, 0, :] = 0.0

        # predictor filter
        p = self.lfilter.forward(x=x, b=a_pred, a=None)

        return p