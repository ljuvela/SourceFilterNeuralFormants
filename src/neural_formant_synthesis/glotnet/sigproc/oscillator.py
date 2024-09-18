import torch

class Oscillator(torch.nn.Module):
    """ Sinusoidal oscillator """
    def __init__(self, 
                 audio_rate:int=48000,
                 control_rate:int=200,
                 shape:str='sin'):
        """
        Args:
            audio_rate: audio sample rate in samples per second
            control_rate: control sample rate in samples per second
                typically equal to 1 / frame_length
        """
        super().__init__()

        self.audio_rate = audio_rate
        self.control_rate = control_rate
        self.nyquist_rate = audio_rate // 2

        upsample_factor = self.audio_rate // self.control_rate
        self.upsampler = torch.nn.modules.Upsample(
            mode='linear',
            scale_factor=upsample_factor,
            align_corners=False) 

        self.audio_step = 1.0 / audio_rate
        self.control_step = 1.0 / control_rate
        self.shape = shape


    def forward(self, f0, init_phase=None):
        """ 
        Args:
            f0 : fundamental frequency, shape (batch_size, channels, num_frames)
        Returns:
            x : sinusoid, shape (batch_size, channels, num_samples)
        """

        f0 = torch.clamp(f0, min=0.0, max=self.nyquist_rate)
        inst_freq = self.upsampler(f0)
        if_shape = inst_freq.shape
        if init_phase is None:
            # random initial phase in range [-pi, pi]
            init_phase =  2 * torch.pi * (torch.rand(if_shape[0], if_shape[1], 1) - 0.5)
        # integrate instantaneous frequency for phase
        phase = torch.cumsum(2 * torch.pi * inst_freq * self.audio_step, dim=-1)

        if self.shape == 'sin':
            return torch.sin(phase + init_phase)
        elif self.shape == 'saw':
            return (torch.fmod(phase + init_phase, 2 * torch.pi) - torch.pi) /  torch.pi
        