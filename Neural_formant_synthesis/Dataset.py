import os, glob
from logging import warning
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class FeatureDataset(Dataset):
    """ Dataset for audio files """

    def __init__(self,
                 dataset_dir: str,
                 segment_len: int,
                 feature_scaler,
                 output_scaler,
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
        self.feature_scaler = feature_scaler
        self.output_scaler = output_scaler
        self.feature_ext = feature_ext
        self.audio_ext = audio_ext
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.causal = causal
        self.non_causal_segment = non_causal_segment

        self.total_segment_len = segment_len + non_causal_segment

        self.audio_files = glob(os.path.join(
            self.dataset_dir, f"*{self.feature_ext}"))

        # elements are (filename, start, stop)
        self.segment_index: List[Tuple(str, int, int)] = []

        for f in self.audio_files:
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
        features = torch.cat((data["Formants"],data["Energy"], data["Centroid"], data["Tilt"], data["Pitch"], data["Voicing"]), dim = -1)
        out_features = data["R_Coeff"]
        
        x = features[start:stop,:]
        y = out_features[start:stop,:]

        num_samples = features.size(0)
        pad_left = 0
        pad_right = 0

        if start == 0:
            pad_left = self.segment_len - x.size(0)
            
            # zero pad to segment_len + padding
            x = torch.transpose(torch.nn.functional.pad(torch.transpose(x,0,1), ( pad_left, 0), mode='replicate'),0,1) # seq_len, n_channels
            y = torch.transpose(torch.nn.functional.pad(torch.transpose(y,0,1), ( pad_left, 0), mode='replicate'),0,1)

        if not self.causal:
            remaining_samples = min(num_samples - stop, self.non_causal_segment)
            x = torch.cat((x, features[stop:stop + remaining_samples,:]),dim=0)
            y = torch.cat((y, out_features[stop:stop + remaining_samples,:]),dim=0)

            pad_right = self.non_causal_segment - remaining_samples
            if pad_right > 0:
                x = torch.transpose(torch.nn.functional.pad(torch.transpose(x,0,1), ( 0, pad_right), mode='replicate'),0,1) # seq_len, n_channels
                y = torch.transpose(torch.nn.functional.pad(torch.transpose(y,0,1), ( 0, pad_right), mode='replicate'),0,1)

        if not(x.size(0) == self.total_segment_len):
            raise ValueError('Padding in the wrong dimension')

        x = torch.transpose(torch.cat((x[:,:-2],x[:,-1:],x[:,-2:-1]), dim = 1), 0, 1)
        y = torch.transpose(y, 0, 1)

        if torch.any(torch.isnan(x)):
            raise ValueError("Output x features are NaN.")
        if torch.any(torch.isnan(y)):
            raise ValueError("Output y features are NaN.")
        # Set sequence len as last dimension.
        return x.type(torch.FloatTensor), y.type(torch.FloatTensor)