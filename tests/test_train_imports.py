import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse

import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from neural_formant_synthesis.third_party.hifi_gan.env import AttrDict, build_env
from neural_formant_synthesis.third_party.hifi_gan.meldataset import mel_spectrogram
from neural_formant_synthesis.third_party.hifi_gan.models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_adversarial_loss,\
    discriminator_loss
from neural_formant_synthesis.third_party.hifi_gan.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint


from neural_formant_synthesis.glotnet.sigproc.lpc import LinearPredictor
from neural_formant_synthesis.glotnet.sigproc.emphasis import Emphasis

from neural_formant_synthesis.dataset import FeatureDataset_List
from neural_formant_synthesis.models import FM_Hifi_Generator, fm_config_obj, Envelope_wavenet, Envelope_conformer
from neural_formant_synthesis.models import SourceFilterFormantSynthesisGenerator

from neural_formant_synthesis.glotnet.sigproc.levinson import forward_levinson

import torchaudio as ta
