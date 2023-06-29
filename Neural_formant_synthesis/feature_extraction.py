import torch
import diffsptk
from functions import frame_energy, spectral_centroid, tilt_levinson, root_to_formant, levinson, pitch_extraction

class feature_extractor(torch.nn.Module):
    """
    Class to extract the features for neural formant synthesis.
    Params:
        sr: sample rate
        window_samples: window size in samples
        step_samples: step size in samples
        formant_ceiling: formant ceiling in Hz
        max_formants: maximum number of formants to estimate
        allpole_order: allpole order for the LPC analysis
    """
    def __init__(self, sr, window_samples, step_samples, formant_ceiling, max_formants, allpole_order = 10, r_coeff_order = 30):
        super().__init__()
        self.sr = sr
        self.window_samples = window_samples
        self.step_samples = step_samples
        self.formant_ceiling = formant_ceiling
        self.max_formants = max_formants
        self.allpole_order = allpole_order
        self.r_coeff_order = r_coeff_order

        self.framer = diffsptk.Frame(
        frame_length = self.window_samples,
        frame_period = self.step_samples
        )
        self.windower = diffsptk.Window(
        in_length = self.window_samples
        )

        self.root_finder = diffsptk.DurandKernerMethod(self.allpole_order)

    def forward(self, x):
        # Signal windowing
        x_frame = self.framer(x)
        x_window = self.windower(x_frame)

        # STFT

        x_spec = torch.fft.rfft(x_window, dim = -1)

        ds_samples = int(self.formant_ceiling/self.sr * x_spec.size(-1))
        x_ds = x_spec[...,:ds_samples]

        x_ds_acorr = torch.fft.irfft(x_ds * torch.conj(x_ds), dim = -1)
    
        x_us_acorr = torch.fft.irfft(x_spec * torch.conj(x_spec), dim = -1)

        # Calculate formants

        ap_env, _ = levinson(x_ds_acorr, self.allpole_order)
        _, r_coeff_ref = levinson(x_us_acorr, self.r_coeff_order)

        roots_env, converge_flag = self.root_finder(ap_env)
        formants = root_to_formant(roots_env, self.formant_ceiling, self.max_formants)
        # Calculate other features
        energy = frame_energy(x_frame)
        centroid = spectral_centroid(x_frame, self.sr)
        tilt = tilt_levinson(x_ds_acorr)

        # Extract pitch from audio signal
        pitch, voicing = pitch_extraction(x = x, sr = self.sr, window_samples = self.window_samples, step_samples = self.step_samples, fmin = 50, fmax = 500)#penn.dsp.dio.from_audio(audio = x, sample_rate = self.sr, hopsize = hopsize, fmin = 50, fmax = 500)

        return formants, energy, centroid, tilt, pitch, voicing, r_coeff_ref, x_ds

class MedianPool1d(torch.nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel
         stride: pool stride
         padding: pool padding
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool1d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.same = same

    def _padding(self, x):
        if self.same:
            iw = x.size()[-1]
            if iw % self.stride == 0:
                pw = max(self.k - self.stride, 0)
            else:
                pw = max(self.k - (iw % self.stride), 0)
            pl = pw // 2
            pr = pw - pl

            padding = (pl, pr)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        x = torch.nn.functional.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k, self.stride)
        x = x.contiguous().view(x.size()[:-1] + (-1,)).median(dim=-1)[0]
        return x