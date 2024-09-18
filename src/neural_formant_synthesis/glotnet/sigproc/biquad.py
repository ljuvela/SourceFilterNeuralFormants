import torch

from .lfilter import LFilter
from typing import Tuple, Union


class BiquadBaseFunctional(torch.nn.Module):

    def __init__(self):
        """ Initialize Biquad"""

        super().__init__()

        # TODO: pass STFT arguments to LFilter
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 1024
        self.lfilter = LFilter(n_fft=self.n_fft,
                               hop_length=self.hop_length,
                               win_length=self.win_length)

    def _check_param_sizes(self, freq, gain, Q):
        """ Check that parameter sizes are compatible
        
        Args:
            freq: center frequencies 
                shape = (batch, out_channels, in_channels, n_frames)
            gain: gains in decibels,
                shape = (batch, out_channels, in_channels, n_frames)
            Q: filter resonance (quality factor)
                shape = (batch, out_channels, in_channels, n_frames)

        Returns:
            batch, out_channels, in_channels, n_frames
        
        """

        # dimensions must be flat
        if freq.ndim != 4:
            raise ValueError("freq must be 4D")
        if gain.ndim != 4:
            raise ValueError("gain must be 4D")
        if Q.ndim != 4:
            raise ValueError("Q must be 4D")

        if freq.shape != gain.shape != Q.shape:
            raise ValueError("freq, gain, and Q must have the same shape")
        
        return freq.shape

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency,
                shape is (..., n_frames)
            gain, gain in decibels
                shape is (..., n_frames)
            Q, resonance sharpness
                shape is (..., n_frames)

        Returns:
            b, filter numerator coefficients
                shape is (..., n_taps==3, n_frames)
            a, filter denominator coefficients
                shape is (..., n_taps==3, n_frames)

        """
        raise NotImplementedError("Subclasses must implement this method")


    def forward(self,
                x: torch.Tensor,
                freq: torch.Tensor,
                gain: torch.Tensor,
                Q: torch.Tensor,
                ) -> torch.Tensor:
        """ 
        Args:
            x: input signal
                shape = (batch, in_channels, time)
            freq: center frequencies 
                shape = (batch, out_channels, in_channels, n_frames)
                n_frames is expected to be (time // hop_size)
            gain: gains in decibels,
                shape = (batch, out_channels, in_channels, n_frames)
            Q: filter resonance (quality factor)
                shape = (batch, out_channels, in_channels, n_frames)
        
        Returns:
            y: output signal
                shape = (batch, channels, n_filters, time)
        """

        batch, out_channels, in_channels, n_frames = self._check_param_sizes(freq=freq, gain=gain, Q=Q)
        timesteps = x.size(-1)

        freq = freq.reshape(batch * out_channels * in_channels, n_frames)
        gain = gain.reshape(batch * out_channels * in_channels, n_frames)
        Q = Q.reshape(batch * out_channels * in_channels, n_frames)

        b, a = self._params_to_direct_form(freq=freq, gain=gain, Q=Q)

        # expand x: (batch, in_channels, time) -> (batch, out_channels, in_channels, time)
        x_exp = x.unsqueeze(1).expand(-1, out_channels, -1, -1)
        # apply filtering
        x_exp = x_exp.reshape(batch * out_channels * in_channels, 1, timesteps)
        y = self.lfilter.forward(x_exp, b=b, a=a)
        # reshape 
        y = y.reshape(batch, out_channels, in_channels, timesteps)
        # sum over input channels
        y = y.sum(dim=2)

        return y
    

class BiquadPeakFunctional(BiquadBaseFunctional):

    def __init__(self):
        """ Initialize Biquad"""
        super().__init__()

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency,
                shape is (..., n_frames)
            gain, gain in decibels
                shape is (..., n_frames)
            Q, resonance sharpness
                shape is (..., n_frames)

        Returns:
            b, filter numerator coefficients
                shape is (..., n_taps==3, n_frames)
            a, filter denominator coefficients
                shape is (..., n_taps==3, n_frames)

        """
        if torch.any(freq > 1.0):
            raise ValueError(f"Normalized frequency must be below 1.0, max was {freq.max()}")
        if torch.any(freq < 0.0):
            raise ValueError(f"Normalized frequency must be above 0.0, min was {freq.min()}")
        
        freq = freq.unsqueeze(-2)
        gain = gain.unsqueeze(-2)
        Q = Q.unsqueeze(-2)

        omega = torch.pi * freq
        A = torch.pow(10.0, 0.025 * gain)
        alpha = 0.5 * torch.sin(omega) / Q

        b0 = 1.0 + alpha * A
        b1 = -2.0 * torch.cos(omega)
        b2 = 1.0 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2.0 * torch.cos(omega)
        a2 = 1 - alpha / A

        a = torch.cat([a0, a1, a2], dim=-2)
        b = torch.cat([b0, b1, b2], dim=-2)

        b = b / a0
        a = a / a0
        return b, a


class BiquadModule(torch.nn.Module):


    @property 
    def freq(self):
        return self._freq
    
    
    def set_freq(self, freq):
        if type(freq) != torch.Tensor:
            freq = torch.tensor([freq], dtype=torch.float32)

        # convert to normalized frequency
        freq = 2.0 * freq / self.fs

        if freq.max() > 1.0:
            raise ValueError(
                "Maximum normalized frequency is larger than 1.0. "
                "Please provide a sample rate or input normalized frequencies")
        if freq.min() < 0.0:
            raise ValueError(
                "Maximum normalized frequency is smaller than 0.0.")


        self._freq.data = freq.broadcast_to(self._freq.shape)

    @property 
    def gain_dB(self):
        return self._gain_dB

    def set_gain_dB(self, gain):
        if type(gain) != torch.Tensor:
            gain = torch.tensor([gain], dtype=torch.float32)
        self._gain_dB.data = gain.broadcast_to(self._gain_dB.shape)

    @property
    def Q(self):
        return self._Q
    
    def set_Q(self, Q):
        if type(Q) != torch.Tensor:
            Q = torch.tensor([Q], dtype=torch.float32)
        self._Q.data = Q.broadcast_to(self._Q.shape)

    def _init_freq(self):
        freq = torch.rand(self.out_channels, self.in_channels)
        self._freq = torch.nn.Parameter(freq)

    def _init_gain_dB(self):
        gain_dB = torch.zeros(self.out_channels, self.in_channels)
        self._gain_dB = torch.nn.Parameter(gain_dB)

    def _init_Q(self):
        Q = torch.ones(self.out_channels, self.in_channels)
        self._Q = torch.nn.Parameter(0.7071 * Q)



    def __init__(self,
                 in_channels: int=1,
                 out_channels: int=1,
                 fs: float = None,
                 func: BiquadBaseFunctional = BiquadPeakFunctional()
                ):
        """
        Args:
            func: BiquadBaseFunctional subclass
            freq: center frequency 
            gain: gain in dB
            Q: quality factor determining filter resonance bandwidth
            fs: sample rate, if not provided freq is assumed as normalized from 0 to 1 (Nyquist)

        """
        super().__init__()

        self.func = func

        # if no sample rate provided, assume normalized frequency
        if fs is None:
            fs = 2.0

        self.fs = fs

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._init_freq()
        self._init_gain_dB()
        self._init_Q()


    def get_impulse_response(self, n_timesteps: int = 2048) -> torch.Tensor:
        """ Get impulse response of filter

        Args:
            n_timesteps: number of timesteps to evaluate

        Returns:
            h, shape is (batch, channels, n_timesteps)
        """
        x = torch.zeros(1, 1, n_timesteps)
        x[:, :, 0] = 1.0
        h = self.forward(x)
        return h
    
    def get_frequency_response(self, n_timesteps: int = 2048, n_fft: int = 2048) -> torch.Tensor:
        """ Get frequency response of filter

        Args:
            n_timesteps: number of timesteps to evaluate

        Returns:
            H, shape is (batch, channels, n_timesteps)
        """
        h = self.get_impulse_response(n_timesteps=n_timesteps)
        H = torch.fft.rfft(h, n=n_fft, dim=-1)
        H = torch.abs(H)
        return H


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, shape is (batch, in_channels, timesteps)

        Returns:
            y, shape is (batch, out_channels, timesteps)
        """

        batch, in_channels, timesteps = x.size()

        num_frames = timesteps // self.func.hop_length
        # map size: (out_channels, in_channels) -> (batch, out_channels, in_channels, n_frames)
        freq = self.freq.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, num_frames)
        gain = self.gain_dB.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, num_frames)
        q_factor = self.Q.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, num_frames)

        y = self.func.forward(x, freq, gain, q_factor)

        return y

class BiquadParallelBankModule(torch.nn.Module):

    def __init__(self, 
                 num_filters:int=10,
                 fs=None,
                 func: BiquadBaseFunctional = BiquadPeakFunctional()
                ):
        """
        Args:
            num_filters: number of filters in bank
            func: BiquadBaseFunctional subclass

        """
        super().__init__()

        self.num_filters = num_filters

        self.fs = fs
        self.filter_bank = BiquadModule(in_channels=num_filters, out_channels=1, fs=fs, func=func)

        # flat initialization
        freq = torch.linspace(0.0, 1.0, num_filters+2)[1:-1]
        gain = torch.zeros_like(freq)
        Q = 0.7071 * torch.ones_like(freq)

        self.filter_bank.set_freq(freq)
        self.filter_bank.set_gain_dB(gain)
        self.filter_bank.set_Q(Q)

    def set_freq(self, freq: torch.Tensor):
        """ Set center frequency of each filter
        Args: 
            freq, shape is (num_filters,)
        """
        self.filter_bank.set_freq(freq)

    def set_gain_dB(self, gain_dB: torch.Tensor):
        self.filter_bank.set_gain_dB(gain_dB)

    def set_Q(self, Q: torch.Tensor):
        self.filter_bank.set_Q(Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, shape is (batch, channels=1, timesteps)

        Returns:
            y, shape is (batch, channels=1, timesteps)
        """

        if x.size(1) != 1:
            raise ValueError(f"Input must have 1 channel, got {x.size(1)}")

        # expand channels to match filter bank
        x = x.expand(-1, self.num_filters, -1)

        # output shape is (batch, channels, , timesteps)
        y = self.filter_bank(x)

        # normalize output by number of filters
        # y = y / self.num_filters

        return y

    def get_impulse_response(self, n_timesteps: int = 2048) -> torch.Tensor:
        """ Get impulse response of filter

        Args:
            n_timesteps: number of timesteps to evaluate

        Returns:
            h, shape is (batch, channels, n_timesteps)
        """
        x = torch.zeros(1, 1, n_timesteps)
        x[:, :, 0] = 1.0
        h = self.forward(x)
        return h
    
    def get_frequency_response(self, n_timesteps: int = 2048, n_fft: int = 2048) -> torch.Tensor:
        """ Get frequency response of filter

        Args:
            n_timesteps: number of timesteps to evaluate

        Returns:
            H, shape is (batch, channels, n_timesteps)
        """
        h = self.get_impulse_response(n_timesteps=n_timesteps)
        H = torch.fft.rfft(h, n=n_fft, dim=-1)
        H = torch.abs(H)
        return H



class BiquadResonatorFunctional(BiquadBaseFunctional):

    def __init__(self):
        """ Initialize Biquad"""
        super().__init__()

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency,
                shape is (..., n_frames)
            gain, gain in decibels
                shape is (..., n_frames)
            Q, resonance sharpness
                shape is (..., n_frames)

        Returns:
            b, filter numerator coefficients
                shape is (..., n_taps==3, n_frames)
            a, filter denominator coefficients
                shape is (..., n_taps==3, n_frames)

        """
        if torch.any(freq > 1.0):
            raise ValueError(f"Normalized frequency must be below 1.0, max was {freq.max()}")
        if torch.any(freq < 0.0):
            raise ValueError(f"Normalized frequency must be above 0.0, min was {freq.min()}")
        
        freq = freq.unsqueeze(-2)
        gain = gain.unsqueeze(-2)
        Q = Q.unsqueeze(-2)

        omega = torch.pi * freq
        A = torch.pow(10.0, 0.025 * gain)
        alpha = 0.5 * torch.sin(omega) / Q

        b0 = torch.ones_like(freq)
        b1 = torch.zeros_like(freq)
        b2 = torch.zeros_like(freq)
        a0 = 1 + alpha / A
        a1 = -2.0 * torch.cos(omega)
        a2 = 1 - alpha / A

        a = torch.cat([a0, a1, a2], dim=1)
        b = torch.cat([b0, b1, b2], dim=1)

        b = b / a0
        a = a / a0
        return b, a


class BiquadBandpassFunctional(BiquadBaseFunctional):

    def __init__(self):
        """ Initialize Biquad"""
        super().__init__()

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency,
                shape is (..., n_frames)
            gain, gain in decibels
                shape is (..., n_frames)
            Q, resonance sharpness
                shape is (..., n_frames)

        Returns:
            b, filter numerator coefficients
                shape is (..., n_taps==3, n_frames)
            a, filter denominator coefficients
                shape is (..., n_taps==3, n_frames)

        """
        if torch.any(freq > 1.0):
            raise ValueError(f"Normalized frequency must be below 1.0, max was {freq.max()}")
        if torch.any(freq < 0.0):
            raise ValueError(f"Normalized frequency must be above 0.0, min was {freq.min()}")
        
        freq = freq.unsqueeze(-2)
        gain = gain.unsqueeze(-2)
        Q = Q.unsqueeze(-2)

        omega = torch.pi * freq
        A = torch.pow(10.0, 0.025 * gain)
        alpha = 0.5 * torch.sin(omega) / Q

        b0 = alpha
        b1 = torch.zeros_like(freq)
        b2 = -1.0 * alpha
        a0 = 1 + alpha / A
        a1 = -2.0 * torch.cos(omega)
        a2 = 1 - alpha / A

        a = torch.cat([a0, a1, a2], dim=1)
        b = torch.cat([b0, b1, b2], dim=1)

        b = b / a0
        a = a / a0
        return b, a