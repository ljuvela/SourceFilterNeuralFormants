import torch
import torchaudio

from .levinson import levinson, spectrum_to_allpole

class SpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.log(torch.clamp(input, min=1e-5))

class InverseSpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.exp(input)

# Tacotron 2 reference configuration
#https://github.com/pytorch/audio/blob/6b2b6c79ca029b4aa9bdb72d12ad061b144c2410/examples/pipeline_tacotron2/train.py#L284
class LogMelSpectrogram(torch.nn.Module):
    """ Log Mel Spectrogram """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = None,
                 hop_length: int = None,
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: float = None,
                 mel_scale: str = "slaney",
                 normalized: bool = False,
                 power: float = 1.0,
                 norm: str = "slaney"
                 ):
        super().__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
            norm=norm,
        )
        self.log = SpectralNormalization()
        self.exp = InverseSpectralNormalization()

        fb_pinv = torch.linalg.pinv(self.mel_spectrogram.mel_scale.fb)
        self.register_buffer('fb_pinv', fb_pinv)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : input signal
                (batch, channels, timesteps)
        """
        X = self.mel_spectrogram(x)

        return self.log(X)

    def allpole(self, X: torch.Tensor, order:int = 20) -> torch.Tensor:
        """
        Args:
            X : input mel spectrogram
            order : all pole model order

        Returns:
            a : allpole filter coefficients

        """

        # invert normalization
        X = self.exp(X)
 
        # (..., F, T) -> (..., T, F)
        X = X.transpose(-1, -2)
        # pseudoinvert mel filterbank
        X = torch.matmul(X, self.fb_pinv).clamp(min=1e-9)
        
        # power spectrum (squared magnitude) spectrum
        X = torch.pow(X, 2.0 / self.mel_spectrogram.power)
        X = X.clamp(min=1e-9)

        g, a = spectrum_to_allpole(X, order=order)
        # (..., T, order) -> (..., order, T)
        a = a.transpose(-1, -2)
        return a

