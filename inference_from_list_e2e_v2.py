import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import argparse
import json
import torch
from neural_formant_synthesis.third_party.hifi_gan.env import AttrDict, build_env
#from neural_formant_synthesis.third_party.hifi_gan.models import discriminator_metrics
from neural_formant_synthesis.third_party.hifi_gan.utils import scan_checkpoint


from neural_formant_synthesis.glotnetsigproc.lpc import LinearPredictor
from neural_formant_synthesis.glotnetsigproc.emphasis import Emphasis

from Neural_formant_synthesis.models import FM_Hifi_Generator, fm_config_obj, Envelope_wavenet,  Envelope_conformer
from Neural_formant_synthesis.feature_extraction import feature_extractor, Normaliser, MedianPool1d
from Neural_formant_synthesis.models import SourceFilterFormantSynthesisGenerator


from neural_formant_synthesis.glotnetsigproc.levinson import forward_levinson

import torchaudio as ta
import pandas as pd
from tqdm import tqdm
from glob import glob


torch.backends.cudnn.benchmark = True


def generate_wave_list(file_list, scale_list, a, h, fm_h):

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    target_sr = h.sampling_rate
    win_size = h.win_size
    hop_size = h.hop_size

    feat_extractor = feature_extractor(sr = target_sr,window_samples = win_size, step_samples = hop_size, formant_ceiling = 10000, max_formants = 4)
    median_filter = MedianPool1d(kernel_size = 3, stride = 1, padding = 0, same = True)
    pre_emphasis_cpu = Emphasis(alpha = h.pre_emph_coeff)

    normalise_features = Normaliser(target_sr)

    generator = SourceFilterFormantSynthesisGenerator(
        fm_config=fm_h,
        g_config=h,
        pretrained_fm=None,
        freeze_fm=False,
        device=device)

    
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')


    generator.load_generator_e2e_checkpoint(cp_g)

    generator = generator.to(device)

    generator.eval()

            
    # Read files from list
    for file in tqdm(file_list, total = len(file_list)):
        # Read audio and resample if necessary
        x, sample_rate = ta.load(file)
        x = x[0:1].type(torch.DoubleTensor)
        
        x = ta.functional.resample(x, sample_rate, target_sr)

        # Get features using feature extractor

        x_preemph = pre_emphasis_cpu(x.unsqueeze(0))
        x_preemph = x_preemph.squeeze(0).squeeze(0)
        formants, energy, centroid, tilt, pitch, voicing_flag,_, _,_ = feat_extractor(x_preemph)

        # Parameter smoothing and length matching

        formants = median_filter(formants.T.unsqueeze(1)).squeeze(1).T

        pitch = pitch.squeeze(0)
        voicing_flag = voicing_flag.squeeze(0)
        
        # If pitch length is smaller than formants, pad pitch and voicing flag with last value
        if pitch.size(0) < formants.size(0):
            pitch = torch.nn.functional.pad(pitch, (0, formants.size(0) - pitch.size(0)), mode = 'constant', value = pitch[-1])
            voicing_flag = torch.nn.functional.pad(voicing_flag, (0, formants.size(0) - voicing_flag.size(0)), mode = 'constant', value = voicing_flag[-1])
        # If pitch length is larger than formants, truncate pitch and voicing flag
        elif pitch.size(0) > formants.size(0):
            pitch = pitch[:formants.size(0)]
            voicing_flag = voicing_flag[:formants.size(0)]

        # We can apply manipulation HERE
        
        log_pitch = torch.log(pitch)

        #pitch = pitch * scale_list[0]
        for i in range(voicing_flag.size(0)):
            if voicing_flag[i] == 1:
                log_pitch[i] = log_pitch[i] + torch.log(torch.tensor(scale_list[0]))   
                formants[i,0] = formants[i,0] * scale_list[1]
                formants[i,1] = formants[i,1] * scale_list[2]
                formants[i,2] = formants[i,2] * scale_list[3]
                formants[i,3] = formants[i,3] * scale_list[4]

        # Normalise data
        log_pitch, formants, tilt, centroid, energy = normalise_features(log_pitch, formants, tilt, centroid, energy)
       
        #Create input data
        #size --> (Batch, features, sequence)
        norm_feat = torch.transpose(torch.cat((log_pitch.unsqueeze(1), formants, tilt.unsqueeze(1), centroid.unsqueeze(1), energy.unsqueeze(1), voicing_flag.unsqueeze(1)),dim = -1), 0, 1)

        norm_feat = norm_feat.type(torch.FloatTensor).unsqueeze(0).to(device)

        y_g_hat, _, _ = generator(norm_feat)

        output_file = os.path.splitext(os.path.basename(file))[0] + '_wave_' + str(scale_list[0]) + '_' + str(scale_list[1]) + '_' + str(scale_list[2]) + '_' + str(scale_list[3]) + '_' + str(scale_list[4]) + '.wav'
        output_orig = os.path.splitext(os.path.basename(file))[0] + '_orig.wav'
        out_path = os.path.join(a.output_path, output_file)
        out_orig_path = os.path.join(a.output_path, output_orig)

        ta.save(out_path, y_g_hat.detach().cpu().squeeze(0), target_sr)
        if not os.path.exists(out_orig_path):
            ta.save(out_orig_path, x.type(torch.FloatTensor), target_sr)

def parse_file_list(list_file):
    """
    Read text file with paths to the files to process.
    """
    file1 = open(list_file, 'r')
    lines = file1.read().splitlines()
    return lines

def str_to_list(in_str):
    return list(map(float, in_str.strip('[]').split(',')))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default = None, help="Path to directory containing files to process.")
    parser.add_argument('--list_file', default = None, help="Text file containing list of files to process. Optional argument to use instead of input_path.")
    parser.add_argument('--output_path', default='test_output', help="Path to directory to save processed files")
    parser.add_argument('--config', default='', help="Path to HiFi-GAN config json file")
    parser.add_argument('--fm_config', default='', help="Path to feature mapping model config json file")
    parser.add_argument('--env_config', default='', help="Path to envelope estimation model config json file")
    parser.add_argument('--audio_ext', default = '.wav', help="Extension of the audio files to process")
    parser.add_argument('--checkpoint_path', help="Path to pre-trained HiFi-GAN model")
    parser.add_argument('--feature_scale', help="List of scales for pitch and formant frequencies -- [F0, F1, F2, F3, F4]")


    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    with open(a.fm_config) as f:
        data = f.read()
    json_fm_config = json.loads(data)
    fm_h = AttrDict(json_fm_config)
    # fm_h = fm_config_obj(json_fm_config)

    build_env(a.config, 'config.json', a.checkpoint_path)
    if a.input_path is not None:
        file_list = glob(os.path.join(a.input_path,'*' + a.audio_ext))
    elif a.list_file is not None:
        file_list = parse_file_list(a.list_file)
    else:
        raise ValueError('Input arguments should include either input_path or file_list')

    if not os.path.exists(a.output_path):
        os.makedirs(a.output_path, exist_ok=True)

    scale_list = str_to_list(a.feature_scale)
    if len(scale_list) != 5:
        raise ValueError('The scaling vector must contain 5 features: [F0, F1, F2, F3, F4]')

    torch.manual_seed(h.seed)

    generate_wave_list(file_list, scale_list, a, h, fm_h)


if __name__ == '__main__':
    main()
