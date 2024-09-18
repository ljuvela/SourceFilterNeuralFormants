import os
import torch
from torchaudio.models import Conformer

from neural_formant_synthesis.glotnet.model.feedforward.wavenet import WaveNet
from neural_formant_synthesis.glotnet.sigproc.emphasis import Emphasis

from .sigproc.levinson import forward_levinson, spectrum_to_allpole
from .sigproc.lpc import LinearPredictor



from neural_formant_synthesis.third_party.hifi_gan.models import Generator
from neural_formant_synthesis.third_party.hifi_gan.utils import load_checkpoint

import diffsptk
from .functions import root_to_formant
from .feature_extraction import Normaliser
# from neural_formant_synthesis.functions import root_to_formant
# from neural_formant_synthesis.feature_extraction import Normaliser

class NeuralFormant_Envelope(torch.nn.Module):
    """
    Envelope estimation model for neural formant synthesis.
    """
    def __init__(self, config, pre_trained_file: str = None, freeze_weights: bool = True, device:str = 'cpu'):
        """
        Params:
            path: Config object with parameters for the model.
            load_pretrained: Path to the file with the pre-trained weights.
            freeze_weights: Freeze weights of model.
        """
        super().__init__()
        self.config = config
        self.pre_trained = pre_trained_file
        self.freeze_weights = freeze_weights
        self.device = device

        self.model = None
        self.create_model()

        if os.path.exists(self.pre_trained):
            self.load_NF_checkpoint(self.pre_trained)
        
        if self.freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def create_model(self):

        self.model = WaveNet(
            input_channels = self.config.n_feat,
            output_channels = self.config.n_out,
            residual_channels = self.config.res_channels,
            skip_channels = self.config.skip_channels,
            kernel_size = self.config.kernel_size,
            dilations = self.config.dilations,
            causal = self.config.causal
        )

    def load_NF_checkpoint(self,checkpoint_file):
        fm_cp_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.feature_mapping.load_state_dict(fm_cp_dict["model_state_dict"])

    def forward(self, x):
        y = self.model(x)
        return y

class FM_Hifi_Generator(torch.nn.Module):
    """
    Combination of HiFi-GAN generator with Feature mapping model.
    Feature mapping transforms input features into Mel spectrogram, which is fed to the HiFi generator.
    """
    def __init__(self, fm_config, g_config, pretrained_fm:str = None, freeze_fm:bool = True, device:str = 'cpu'):
        """
        fm_config: Config object for feature mapping model:
        g_config: Config object for HiFi generator.
        """
        super().__init__()

        self.fm_config = fm_config
        self.g_config = g_config

        self.pretrained_fm = pretrained_fm

        self.freeze_fm = freeze_fm

        self.device = device

        self.create_models()

        if self.pretrained_fm is not None:
            print("Freezing pre-trained feature mapping.")
            self.load_fm_checkpoint(self.pretrained_fm) 
            if self.freeze_fm:
                for param in self.feature_mapping.parameters():
                    param.requires_grad = False

    def create_models(self):

        self.feature_mapping = WaveNet(
            input_channels = self.fm_config.n_feat,
            output_channels = self.fm_config.n_out,
            residual_channels = self.fm_config.res_channels,
            skip_channels = self.fm_config.skip_channels,
            kernel_size = self.fm_config.kernel_size,
            dilations = self.fm_config.dilations,
            causal = self.fm_config.causal
        )

        self.hifi_generator = Generator(h = self.g_config, input_channels = self.fm_config.n_out)

    def load_fm_checkpoint(self,checkpoint_file):
        fm_cp_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.feature_mapping.load_state_dict(fm_cp_dict["model_state_dict"])

    def load_generator_checkpoint(self,checkpoint_file):
        generator_state_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.hifi_generator.load_state_dict(generator_state_dict['generator'])

    def forward(self,input_features):
        
        x = self.feature_mapping(input_features)

        y = self.hifi_generator(x)

        return y
    
class SourceFilterFormantSynthesisGenerator(torch.nn.Module):

    def __init__(self, fm_config, g_config, pretrained_fm:str = None, freeze_fm:bool = True, device:str = 'cpu'):
        """
        fm_config: Config object for feature mapping model:
        g_config: Config object for HiFi generator.
        """
        super().__init__()

        self.fm_config = fm_config
        self.g_config = g_config

        self.pretrained_fm = pretrained_fm

        self.freeze_fm = freeze_fm

        self.device = device

        self.allpole_order = self.g_config['allpole_order']
        self.num_mels = self.g_config['num_mels']
        self.noise_channels = self.fm_config.get('output_noise_channels', 0)

        self.create_models()

        if self.pretrained_fm is not None:
            print("Freezing pre-trained feature mapping.")
            self.load_fm_checkpoint(self.pretrained_fm) 
            if self.freeze_fm:
                for param in self.feature_mapping.parameters():
                    param.requires_grad = False

    def create_models(self):

        self.feature_mapping = WaveNet(
            input_channels = self.fm_config.n_feat,
            output_channels = self.fm_config.n_out,
            residual_channels = self.fm_config.res_channels,
            skip_channels = self.fm_config.skip_channels,
            kernel_size = self.fm_config.kernel_size,
            dilations = self.fm_config.dilations,
            causal = self.fm_config.causal
        )

        for p in self.feature_mapping.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)

        self.hifi_generator = Generator(h = self.g_config, input_channels = self.fm_config.n_out)

        h = self.g_config
        self.pre_emphasis = Emphasis(alpha=h.pre_emph_coeff)
        self.lpc = LinearPredictor(
            n_fft=h.n_fft,
            hop_length=h.hop_size,
            win_length=h.win_size,
            order=h.allpole_order)


    def load_fm_checkpoint(self,checkpoint_file):
        fm_cp_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.feature_mapping.load_state_dict(fm_cp_dict["model_state_dict"])

    def load_generator_checkpoint(self,checkpoint_file):
        generator_state_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.hifi_generator.load_state_dict(generator_state_dict['generator'])

    def load_generator_e2e_checkpoint(self,checkpoint_file):
        generator_state_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.load_state_dict(generator_state_dict['generator'])

    def forward(self, input_features, feature_map_only = False):

        x = self.feature_mapping(input_features)

        x_allpole, x_mel, x_noise = torch.split(x, (self.allpole_order+1, self.num_mels, self.noise_channels), dim=1)

        x_lars, x_gain = torch.split(x_allpole, (self.allpole_order, 1), dim=1)
        gain_lin = torch.exp(x_gain + 1e-5)
        x_rcoef = torch.tanh(0.5 * x_lars)

        allpole = torch.transpose(forward_levinson(torch.transpose(x_rcoef,1,2)), 1,2)

        A = torch.fft.rfft(allpole, n=512, dim=1).abs()
        H = gain_lin / (A + 1e-6)
        # H = 1 / (A + 1e-6)

        if feature_map_only:
            return H, x_mel

        # s = torch.exp(torch.clamp(x_noise, min=-7.0, max=0.0))
        s = torch.sigmoid(x_noise)
        x_noise = s * torch.randn_like(x_noise)

        # gan_input = torch.cat([x_mel, x_allpole, x_noise], dim=1)
        gan_input = torch.cat([x_mel, x_rcoef, x_gain, x_noise], dim=1)

        # y = self.hifi_generator(gan_input)

        excitation = self.hifi_generator(gan_input)
        # apply synthesis filter
        y = self.lpc.synthesis_filter(excitation, allpole, gain_lin)

        return y, H, x_mel
    
class Envelope_wavenet(torch.nn.Module):
    """
    Envelope estimation using gated convolution architecture based on wavenet
    """
    def __init__(self, config, use_pretrained:bool = False, freeze_weights:bool = True, device = 'cpu'):
        super().__init__()

        self.n_feat = config.n_feat
        self.n_out = config.n_out
        self.res_channels = config.res_channels
        self.skip_channels = config.skip_channels
        self.kernel_size = config.kernel_size
        self.dilations = config.dilations
        self.causal = config.causal

        self.pretrained_path = config.model_path
        self.use_pretrained = use_pretrained
        self.freeze_weights = freeze_weights

        self.device = device

        self.create_model()

        if self.use_pretrained:
            self.load_pretrained(self.pretrained_path)
            if self.freeze_weights:
                self.freeze()

    def create_model(self):

        self.model = WaveNet(
                input_channels = self.n_feat,
                output_channels = self.n_out,
                residual_channels = self.res_channels,
                skip_channels = self.skip_channels,
                kernel_size = self.kernel_size,
                dilations = self.dilations,
                causal = self.causal
            )
        self.out_activation = torch.nn.Tanh()
        
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def load_pretrained(self,checkpoint_file):
        cp_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.model.load_state_dict(cp_dict["model_state_dict"])

    def save_pretrained(self, checkpoint_file, param_dict):
        param_dict["model_state_dict"] = self.model.state_dict()
        torch.save(param_dict, checkpoint_file)

    def forward(self,x):
        y = self.model(x)
        y = self.out_activation(y)
        return y

class Envelope_conformer(torch.nn.Module):
    """
    Envelope prediction model based on conformer network.
    """
    def __init__(self, config, pre_trained:str = None, freeze_weights:bool = False, device:str = 'cpu'):
        """
        Params:
            Config: Dictionary containing config parameters for the network
            pre_trained: Path to a file with pre-trained information for the model.
            freeze_weights: Bool to freeze weights of the model.
        """
        super().__init__()

        self.input_size = config.input_size
        self.output_size = config.output_size
        self.conformer_features = config.conformer_size

        self.num_heads = config.num_heads
        self.ffn_dim = config.ffn_dim
        self.num_conformer_layers = config.num_conformer_layers
        self.depthwise_conv_kernel_size = config.depthwise_kernel_size

        self.pretrained_path = pre_trained
        self.freeze_weights = freeze_weights
        self.device = device

        self.create_model()

        if self.pretrained_path is not None:
            self.load_pretrained(self.pretrained_path)

        if freeze_weights:
            self.freeze()

    def freeze(self):
        for param in self.input_embeddings.parameters():
            param.requires_grad = False
        for param in self.conformer.parameters():
            param.requires_grad = False
        for param in self.output_embeddings.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.input_embeddings.parameters():
            param.requires_grad = True
        for param in self.conformer.parameters():
            param.requires_grad = True
        for param in self.output_embeddings.parameters():
            param.requires_grad = True

    def load_pretrained(self,checkpoint_file):
        fm_cp_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.input_embeddings.load_state_dict(fm_cp_dict["emb_state_dict"])
        self.conformer.load_state_dict(fm_cp_dict["conf_state_dict"])
        self.output_embeddings.load_state_dict(fm_cp_dict["outemb_state_dict"])

    def save_pretrained(self, checkpoint_file, param_dict):
        param_dict["emb_state_dict"] = self.input_embeddings.state_dict()
        param_dict["conf_state_dict"] = self.conformer.state_dict()
        param_dict["outemb_state_dict"] = self.output_embeddings.state_dict()
        torch.save(param_dict, checkpoint_file)
    
    def create_model(self):
        self.input_embeddings = torch.nn.Sequential(torch.nn.Linear(self.input_size, self.conformer_features),
                                                     torch.nn.Tanh()
                                                    )
        self.conformer = Conformer(input_dim = self.conformer_features,
                                   num_heads = self.num_heads,
                                   ffn_dim = self.ffn_dim,
                                   num_layers = self.num_conformer_layers,
                                   depthwise_conv_kernel_size = self.depthwise_conv_kernel_size
                                  )
        self.output_embeddings = torch.nn.Sequential(torch.nn.Linear(self.conformer_features, self.output_size),
                                                     torch.nn.Tanh()
                                                     )
        
    def forward(self, x, lengths = None):
        # Expect input with size (batch_size, input_dims, sequence_len) as it was used in wavenet. We need to tranpose input

        if lengths == None:
            lengths = torch.ones((x.size(0),), device = self.device) * x.size(-1)

        x = torch.transpose(x,1,2)
        x_emb = self.input_embeddings(x)
        y_conf, _ = self.conformer(x_emb, lengths)
        y = self.output_embeddings(y_conf)
        y = torch.transpose(y,1,2)
        return y

class fm_config_obj(object):
    def __init__(self, dict):
        self.type = dict["type"]
        if self.type == "WaveNet":
            self.model_path = dict.get("model_path", None)
            self.n_feat = dict["n_feat"]
            self.n_out = dict["n_out"]
            self.batch_size = dict["batch_size"]
            self.res_channels = dict["res_channels"]
            self.skip_channels = dict["skip_channels"]
            self.kernel_size = dict["kernel_size"]
            self.dilations = dict["dilations"]
            self.causal = dict["causal"]
        elif self.type == "Conformer":
            self.model_path = dict["model_path"]
            self.input_size = dict["input_size"]
            self.output_size = dict["output_size"]
            self.conformer_size = dict["conformer_size"]
            self.num_heads = dict["num_heads"]
            self.ffn_dim = dict["ffn_dim"]
            self.num_conformer_layers = dict["num_conformer_layers"]
            self.depthwise_kernel_size = dict["depthwise_kernel_size"]
        else:
            raise ValueError("Envelope model type not supported.")

class LAR_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k_ref, k_hat):
        """
        Calculate loss function as the error in Log Area Ratio domain from reflection coefficients as input.
        Params:
            k_hat: Estimated reflection coefficients with shape (batch_size, num_frames, num_coeffs)
            k_ref: Reference reflection coefficients with shape (batch_size, num_frames, num_coeffs)
        Return:
            Loss value
        """
        batch_size = k_hat.size(0)

        lar_hat = torch.log(torch.divide(1 - k_hat, 1 + k_hat))
        lar_ref = torch.log(torch.divide(1 - k_ref, 1 + k_ref))

        abs_err = torch.sum(torch.square(lar_ref - lar_hat))

        loss = torch.divide(abs_err, batch_size)

        return loss

class Spectrum_loss(torch.nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.loss_fn = torch.nn.L1Loss()

    def forward(self,k_ref, k_hat):
        """
        Calculate mse betwee the magnitude spectra extracted from reflection coefficients.
        Params:
            k_hat: Estimated reflection coefficients with shape (batch_size, num_frames, num_coeffs)
            k_ref: Reference reflection coefficients with shape (batch_size, num_frames, num_coeffs)
        Return:
            Loss value
        """

        ap_hat = forward_levinson(k_hat)
        ap_ref = forward_levinson(k_ref)

        #
        spec_hat = -20 * torch.log10(torch.abs(torch.fft.rfft(ap_hat, n = self.n_fft, dim=- 1)) + 1e-6)
        spec_ref = -20 * torch.log10(torch.abs(torch.fft.rfft(ap_ref, n = self.n_fft, dim=- 1)) + 1e-6)

        loss = self.loss_fn(spec_hat, spec_ref)#torch.mean(torch.square(spec_ref - spec_hat))
        return loss

class Composite_Loss(torch.nn.Module):
    def __init__(self, n_fft, scale_spec:float = 20):
        super().__init__()
        self.n_fft = n_fft
        self.lar_loss = LAR_loss()
        self.spec_loss = Spectrum_loss(self.n_fft)
        self.scale_spec = scale_spec

    def forward(self, k_ref, k_hat):
        lar_loss_value = self.lar_loss(k_ref, k_hat)
        spec_loss_value = self.spec_loss(k_ref, k_hat)

        return lar_loss_value + self.scale_spec * spec_loss_value

class CycleConsistency(torch.nn.Module):
    """
    Cycle consistency function to return from estimated reflection coefficients to formant values (And other features)
    Params:
        r_coeff: Estimated reflection coefficients.
    Returns:
        Estimated formant values.
    """
    def __init__(self, allpole_order, num_formants:int = 4, n_fft:int = 512, rcoeff_sampling_rate:int = 22050, formant_sampling_rate:int = 10000, device:str = 'cpu'):
        super().__init__()
        self.num_formants = num_formants
        self.allpole_order = allpole_order

        self.device = device

        self.root_finder = diffsptk.DurandKernerMethod(self.allpole_order).to(self.device)


        self.n_fft = n_fft
        self.freq_samples = int(self.n_fft/2 + 1)

        self.rcoeff_sampling_rate = rcoeff_sampling_rate
        self.formant_sampling_rate = formant_sampling_rate

        self.resample_segment = int(self.freq_samples * self.formant_sampling_rate / self.rcoeff_sampling_rate)

        self.normaliser = Normaliser(self.rcoeff_sampling_rate).to(self.device)

    def forward(self, r_coeff: torch.Tensor):

        r_coeff = torch.transpose(r_coeff,1,2)

        batch_size = r_coeff.size(0)
        seq_len = r_coeff.size(1)

        # Forward-levinson from r_coeffs to allpole
        all_pole = forward_levinson(r_coeff)

        all_pole_freq = torch.fft.rfft(all_pole, n = self.n_fft, dim=-1)

        # Downsample allpole spectrum
        freq_ds = all_pole_freq[:,:,:self.resample_segment]

        # Get autocorrelation in frequency

        acorr_ds = torch.abs(freq_ds) ** 2

        # Calculate allpole

        all_pole_ds,_ = spectrum_to_allpole(spectrum = acorr_ds, order = self.allpole_order)

        roots, _ = self.root_finder(all_pole_ds)
        roots = torch.reshape(roots,(batch_size*seq_len,-1))
        formants = root_to_formant(roots = roots, sr = self.formant_sampling_rate, max_formants = self.num_formants)
        
        _,formants, _,_,_ = self.normaliser(torch.tensor([0]), formants, torch.tensor([0]),torch.tensor([0]),torch.tensor([0]))
        formants = torch.transpose(torch.reshape(formants,(batch_size,seq_len,-1)),1,2)
        

        return formants
    

 
class NeuralFormantSynthesisGenerator(torch.nn.Module):

    def __init__(self, fm_config, g_config, pretrained_fm:str = None, freeze_fm:bool = True, device:str = 'cpu'):
        """
        fm_config: Config object for feature mapping model:
        g_config: Config object for HiFi generator.
        """
        super().__init__()

        self.fm_config = fm_config
        self.g_config = g_config

        self.pretrained_fm = pretrained_fm

        self.freeze_fm = freeze_fm

        self.device = device

        self.num_mels = self.g_config['num_mels']

        self.create_models()

        if self.pretrained_fm is not None:
            print("Freezing pre-trained feature mapping.")
            self.load_fm_checkpoint(self.pretrained_fm) 
            if self.freeze_fm:
                for param in self.feature_mapping.parameters():
                    param.requires_grad = False

    def create_models(self):

        self.feature_mapping = WaveNet(
            input_channels = self.fm_config.n_feat,
            output_channels = self.fm_config.n_out,
            residual_channels = self.fm_config.res_channels,
            skip_channels = self.fm_config.skip_channels,
            kernel_size = self.fm_config.kernel_size,
            dilations = self.fm_config.dilations,
            causal = self.fm_config.causal
        )

        for p in self.feature_mapping.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)

        self.hifi_generator = Generator(h = self.g_config, input_channels = self.fm_config.n_out)


    def load_fm_checkpoint(self,checkpoint_file):
        fm_cp_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.feature_mapping.load_state_dict(fm_cp_dict["model_state_dict"])

    def load_generator_checkpoint(self,checkpoint_file):
        generator_state_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.hifi_generator.load_state_dict(generator_state_dict['generator'])

    def load_generator_e2e_checkpoint(self,checkpoint_file):
        generator_state_dict = load_checkpoint(checkpoint_file, device = self.device)
        self.load_state_dict(generator_state_dict['generator'])

    def forward(self, input_features, feature_map_only=False, detach_feature_map=False):

        x = self.feature_mapping(input_features)

        if feature_map_only:
            return x
        
        gen_input = x
        if detach_feature_map:
            gen_input = gen_input.detach()

        y = self.hifi_generator(gen_input)
 
        return y, x
    
