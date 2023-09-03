import os
from glob import glob
from logging import warning
import random
import torch
import torchaudio as ta
from torch.utils.data import Dataset
from typing import List, Tuple
from Neural_formant_synthesis.feature_extraction import Normaliser
from glotnet.sigproc.levinson import forward_levinson

from hifi_gan.meldataset import mel_spectrogram
from tqdm import tqdm

class FeatureDataset(Dataset):
    """ Dataset for audio files """

    def __init__(self,
                 dataset_dir: str,
                 segment_len: int,
                 sampling_rate: int,
                 feature_ext: str = '.pt',
                 audio_ext: str = '.wav',
                 causal: bool = True,
                 non_causal_segment: int = 0,
                 normalise: bool = True,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        """
        Args:
            config: Config object
            audio_dir: directory containing audio files
            audio_ext: file extension of audio files
            file_list: list of audio files
            transforms: transforms to apply to audio, output as auxiliary feature for conditioning
            dtype: data type of output
        """

        self.dataset_dir = dataset_dir
        self.segment_len = segment_len
        self.sampling_rate = sampling_rate
        self.feature_ext = feature_ext
        self.audio_ext = audio_ext
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.causal = causal
        self.non_causal_segment = non_causal_segment

        self.total_segment_len = segment_len + non_causal_segment

        self.normaliser = Normaliser(self.sampling_rate)

        self.audio_files = glob(os.path.join(
            self.dataset_dir, f"*{self.feature_ext}"))

        # elements are (filename, start, stop)
        self.segment_index: List[Tuple(str, int, int)] = []

        for f in tqdm(self.audio_files, total=len(self.audio_files)):
            self._check_feature_file(f)

    def _check_feature_file(self, f):


        features = torch.load(f)
        input_features = features['Energy']
        num_samples = input_features.size(0)

        # read sements from the end (to minimize zero padding)
        stop = num_samples
        start = stop - self.segment_len # Only consider causal length to set start index.
        start = max(start, 0)
        while stop > 0:
            #if self.causal:
            self.segment_index.append(
                (os.path.realpath(f), start, stop))
            #else:
            #    stop_noncausal = min(stop + self.non_causal_segment, num_samples)
            #    self.segment_index.append(
            #        (os.path.realpath(f), start, stop_noncausal))

            stop = stop - self.segment_len
            start = stop - self.segment_len
            start = max(start, 0)

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, i):
        f, start, stop = self.segment_index[i]
        data = torch.load(f)
        features = torch.cat((data["Pitch"].unsqueeze(1), data["Formants"], data["Tilt"].unsqueeze(1), data["Centroid"].unsqueeze(1), data["Energy"].unsqueeze(1), data["Voicing"].unsqueeze(1)), dim = -1)
        out_features = data["R_Coeff"]

        x = features[start:stop,:]
        y = out_features[start:stop,:]

        if torch.any(torch.isnan(data["Pitch"])):
            raise ValueError("Pitch features are NaN before normalisation.")
        if torch.any(torch.isnan(data["Formants"])):
            raise ValueError("Formants features are NaN before normalisation.")
        if torch.any(torch.isnan(data["Tilt"])):
            raise ValueError("Tilt features are NaN before normalisation.")
        if torch.any(torch.isnan(data["Centroid"])):
            raise ValueError("Centroid features are NaN before normalisation.")
        if torch.any(torch.isnan(data["Energy"])):
            raise ValueError("Energy features are NaN before normalisation.")

        num_samples = features.size(0)
        pad_left = 0
        pad_right = 0

        if start == 0:
            pad_left = self.segment_len - x.size(0)
            
            # zero pad to segment_len + padding
            x = torch.transpose(torch.nn.functional.pad(torch.transpose(x, 0, 1), ( pad_left, 0), mode='replicate'), 0, 1) # seq_len, n_channels
            y = torch.transpose(torch.nn.functional.pad(torch.transpose(y, 0, 1), ( pad_left, 0), mode='replicate'), 0, 1)

        if not self.causal:
            remaining_samples = min(num_samples - stop, self.non_causal_segment)
            x = torch.cat((x, features[stop:stop + remaining_samples,:]), dim=0)
            y = torch.cat((y, out_features[stop:stop + remaining_samples,:]), dim=0)

            pad_right = self.non_causal_segment - remaining_samples
            if pad_right > 0:
                x = torch.transpose(torch.nn.functional.pad(torch.transpose(x, 0, 1), ( 0, pad_right), mode='replicate'), 0, 1) # seq_len, n_channels
                y = torch.transpose(torch.nn.functional.pad(torch.transpose(y, 0, 1), ( 0, pad_right), mode='replicate'), 0, 1)

        if not(x.size(0) == self.total_segment_len):
            raise ValueError('Padding in the wrong dimension')

        x = torch.transpose(torch.cat((torch.cat(self.normaliser(x[...,0:1], x[...,1:5], x[...,5:6], x[...,6:7], x[...,7:8]),dim = -1), x[...,8:9]),dim = -1), 0, 1)
        y = torch.transpose(y, 0, 1)

        if torch.any(torch.isnan(x)):
            raise ValueError("Output x features are NaN.")
        if torch.any(torch.isnan(y)):
            raise ValueError("Output y features are NaN.")
        # Set sequence len as last dimension.
        return x.type(torch.FloatTensor).to(self.device), y.type(torch.FloatTensor).to(self.device)
    
class FeatureDataset_with_Mel(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 segment_len: int,
                 sampling_rate: int,
                 frame_size: int,
                 hop_size:int,
                 feature_ext: str = '.pt',
                 audio_ext: str = '.wav',
                 causal: bool = True,
                 non_causal_segment: int = 0,
                 normalise: bool = True,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        """
        Args:
            config: Config object
            audio_dir: directory containing audio files
            audio_ext: file extension of audio files
            file_list: list of audio files
            transforms: transforms to apply to audio, output as auxiliary feature for conditioning
            dtype: data type of output
        """

        self.dataset_dir = dataset_dir
        self.segment_len = segment_len
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.feature_ext = feature_ext
        self.audio_ext = audio_ext
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.causal = causal
        self.non_causal_segment = non_causal_segment

        self.total_segment_len = segment_len + non_causal_segment

        self.melspec = ta.transforms.MelSpectrogram(sample_rate = self.sampling_rate, n_fft = self.frame_size, win_length = self.frame_size, hop_length = self.hop_size, n_mels = 80)

        self.normaliser = Normaliser(self.sampling_rate)

        self.audio_files = glob(os.path.join(
            self.dataset_dir, f"*{self.feature_ext}"))

        # elements are (filename, start, stop)
        self.segment_index: List[Tuple(str, int, int)] = []

        for f in tqdm(self.audio_files, total=len(self.audio_files)):
            self._check_feature_file(f)
    
    def _check_feature_file(self, f):


        features = torch.load(f)
        input_features = features['Energy']
        num_samples = input_features.size(0)

        # read sements from the end (to minimize zero padding)
        stop = num_samples
        start = stop - self.segment_len # Only consider causal length to set start index.
        start = max(start, 0)
        while stop > 0:
            #if self.causal:
            self.segment_index.append(
                (os.path.realpath(f), start, stop))
            #else:
            #    stop_noncausal = min(stop + self.non_causal_segment, num_samples)
            #    self.segment_index.append(
            #        (os.path.realpath(f), start, stop_noncausal))

            stop = stop - self.segment_len
            start = stop - self.segment_len
            start = max(start, 0)

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, i):

        f, start, stop = self.segment_index[i]

        audio_file = os.path.splitext(f)[0] + self.audio_ext
        start_samples = start * self.hop_size
        stop_samples = (stop - 1) * self.hop_size

        data = torch.load(f)
        features = torch.cat((torch.log(torch.clamp(data["Pitch"],50, 2000)).unsqueeze(1), data["Formants"], data["Tilt"].unsqueeze(1), data["Centroid"].unsqueeze(1), data["Energy"].unsqueeze(1), data["Voicing"].unsqueeze(1)), dim = -1)
        out_features = data["R_Coeff"]

        x = features[start:stop,:]
        y = out_features[start:stop,:]

        audio, sample_rate = ta.load(audio_file)

        if sample_rate != self.sampling_rate:
            audio = ta.functional.resample(audio, sample_rate, self.sampling_rate)
        audio_segment = audio[...,start_samples:stop_samples]

        num_samples = features.size(0)
        pad_left = 0
        pad_right = 0

        if start == 0:
            pad_left = self.segment_len - x.size(0)
            pad_left_audio = pad_left * self.hop_size
            
            # zero pad to segment_len + padding
            x = torch.transpose(torch.nn.functional.pad(torch.transpose(x, 0, 1), ( pad_left, 0), mode='replicate'), 0, 1) # seq_len, n_channels
            y = torch.transpose(torch.nn.functional.pad(torch.transpose(y, 0, 1), ( pad_left, 0), mode='replicate'), 0, 1)
            audio_segment = torch.nn.functional.pad(audio_segment, ( pad_left_audio, 0), mode='constant', value=0)

        if not self.causal:
            remaining_samples = min(num_samples - stop, self.non_causal_segment)
            remaining_samples_audio = remaining_samples * self.hop_size

            x = torch.cat((x, features[stop:stop + remaining_samples,:]), dim=0)
            y = torch.cat((y, out_features[stop:stop + remaining_samples,:]), dim=0)
            audio_segment = torch.cat((audio_segment, audio[...,stop_samples:stop_samples + remaining_samples_audio]), dim=-1)

            pad_right = self.non_causal_segment - remaining_samples
            if pad_right > 0:
                x = torch.transpose(torch.nn.functional.pad(torch.transpose(x, 0, 1), ( 0, pad_right), mode='replicate'), 0, 1) # seq_len, n_channels
                y = torch.transpose(torch.nn.functional.pad(torch.transpose(y, 0, 1), ( 0, pad_right), mode='replicate'), 0, 1)
                pad_right_audio = pad_right * self.hop_size
                audio_segment = torch.nn.functional.pad(audio_segment, ( pad_right_audio, 0), mode='constant', value=0)

        if not(x.size(0) == self.total_segment_len):
            raise ValueError('Padding in the wrong dimension')

        x = torch.transpose(torch.cat((torch.cat(self.normaliser(x[...,0:1], x[...,1:5], x[...,5:6], x[...,6:7], x[...,7:8]),dim = -1), x[...,8:9]),dim = -1), 0, 1)
        y = torch.transpose(y, 0, 1)

        mels = self.melspec(audio_segment).squeeze(0)

        if torch.any(torch.isnan(x)):
            raise ValueError("Output x features are NaN.")
        if torch.any(torch.isnan(y)):
            raise ValueError("Output y features are NaN.")
        # Set sequence len as last dimension.
        return x.type(torch.FloatTensor).to(self.device), y.type(torch.FloatTensor).to(self.device), audio_segment.squeeze(0).type(torch.FloatTensor).to(self.device), mels.type(torch.FloatTensor).to(self.device)
    
class FeatureDataset_List(Dataset):
    def __init__(self, 
                 dataset_dir:str,
                 config,
                 sampling_rate: int,
                 frame_size: int,
                 hop_size:int,
                 feature_ext: str = '.pt',
                 audio_ext: str = '.wav',
                 segment_length: int = None,
                 normalise: bool = True,
                 shuffle: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        
        self.config = config

        self.dataset_dir = dataset_dir
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.feature_ext = feature_ext
        self.audio_ext = audio_ext
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.shuffle = shuffle

        self.segment_length = segment_length

        self.mel_spectrogram = ta.transforms.MelSpectrogram(sample_rate = self.sampling_rate, n_fft=self.frame_size, win_length = self.frame_size, hop_length = self.hop_size, f_min = 0.0, f_max = 8000, n_mels = 80)
        self.normaliser = Normaliser(self.sampling_rate)

        self.get_file_list()

    def get_file_list(self):

        self.file_list = glob(os.path.join(self.dataset_dir, '*' + self.feature_ext))
        if self.shuffle:
            #indices = torch.randperm(len(file_list))
            #file_list = file_list[indices]
            random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):

        feature_file = self.file_list[index]
        audio_file = os.path.splitext(feature_file)[0] + self.audio_ext

        data = torch.load(feature_file)
        x = torch.cat((data["Pitch"].unsqueeze(1), data["Formants"], data["Tilt"].unsqueeze(1), data["Centroid"].unsqueeze(1), data["Energy"].unsqueeze(1), data["Voicing"].unsqueeze(1)), dim = -1)
        y = data["R_Coeff"]

        audio, sample_rate = ta.load(audio_file)

        if sample_rate != self.sampling_rate:
            audio = ta.functional.resample(audio, sample_rate, self.sampling_rate)

        if self.segment_length is None:
            self.segment_length = x.size(0)

        audio_total_len = int(x.size(0) * self.hop_size)

        if audio.size(1) < audio_total_len:
            audio = torch.unsqueeze(torch.nn.functional.pad(audio.squeeze(0), (0,int(audio_total_len - audio.size(1))),'constant'),0)
        else:
            audio = audio[:,:audio_total_len]

        if self.segment_length <= x.size(0):

            max_segment_start = x.size(0) - self.segment_length
            segment_start = random.randint(0, max_segment_start)

            x = x[segment_start:segment_start + self.segment_length,:]
            y = y[segment_start:segment_start + self.segment_length,:]

            audio_start = int(segment_start * self.hop_size)
            audio_segment_len = int(self.segment_length * self.hop_size)
            audio = audio[:,audio_start:audio_start + audio_segment_len]
        elif self.segment_length > x.size(0):
            diff = self.segment_length - x.size(0)
            x = torch.transpose(torch.nn.functional.pad(torch.transpose(x,0,1),(0,diff),'replicate'),0,1)
            y = torch.transpose(torch.nn.functional.pad(torch.transpose(y,0,1),(0,diff),'replicate'),0,1)

            audio_segment_diff = int(self.hop_size * diff)
            audio = torch.unsqueeze(torch.nn.functional.pad(audio.squeeze(0), (0,audio_segment_diff),'constant'),0)

        x = torch.transpose(torch.cat((torch.cat(self.normaliser(x[...,0:1], x[...,1:5], x[...,5:6], x[...,6:7], x[...,7:8]),dim = -1), x[...,8:9], y),dim = -1), 0, 1)
        #x = torch.transpose(torch.cat((torch.cat(self.normaliser(x[...,0:1], x[...,1:5], x[...,5:6], x[...,6:7], x[...,7:8]),dim = -1), x[...,8:9]),dim = -1), 0, 1)

        y = forward_levinson(y)

        y = torch.transpose(y, 0, 1)

        y_mel = mel_spectrogram(audio,sampling_rate = self.sampling_rate, n_fft=self.frame_size, win_size = self.frame_size, hop_size = self.hop_size, fmin = 0.0, fmax = self.config.fmax_for_loss, num_mels = 80)#self.mel_spectrogram(audio)

        return x.type(torch.FloatTensor).to(self.device), y.type(torch.FloatTensor).to(self.device), audio.squeeze(0).type(torch.FloatTensor).to(self.device), y_mel.squeeze().type(torch.FloatTensor).to(self.device)