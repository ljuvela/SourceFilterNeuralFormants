import os, glob
import numpy as np
import torch
import torchaudio as ta
from tqdm import tqdm

from feature_extraction import feature_extractor, MedianPool1d
from glotnet.sigproc.emphasis import Emphasis

ta.set_audio_backend("sox_io")

def main(vctk_path, target_dir):
    # Set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # Define initial parameters
    window_length = 1024
    step_length = 256

    target_sr = 22050

    file_ext = '.flac'

    # Declare feature extractor
    feat_extractor = feature_extractor(sr = target_sr,window_samples = window_length, step_samples = step_length, formant_ceiling = 10000, max_formants = 4)
    median_filter = MedianPool1d(kernel_size = 3, stride = 1, padding = 0, same = True)
    pre_emphasis = Emphasis(alpha=0.97)

    # Divide vctk dataset into train, validation and test sets
    divide_vctk(vctk_path, target_dir, file_ext = file_ext)

    # Process train, validation and test sets
    print("Processing separated sets")
    train_dir = os.path.join(target_dir, 'train')
    process_directory(train_dir, target_sr, feat_extractor, median_filter, pre_emphasis = pre_emphasis, file_ext = file_ext)
    val_dir = os.path.join(target_dir, 'val')
    process_directory(val_dir, target_sr, feat_extractor, median_filter, pre_emphasis = pre_emphasis, file_ext = file_ext)
    test_dir = os.path.join(target_dir, 'test')
    process_directory(test_dir, target_sr, feat_extractor, median_filter, pre_emphasis = pre_emphasis, file_ext = file_ext)

def divide_vctk(vctk_path, target_dir, file_ext = '.wav'):
    """
    Divide original vctk dataset into train, validation and test sets with a ratio of 80:10:10 with different speakers.
    """
    #Empty target directory if it exists
    if os.path.exists(target_dir):
        os.system('rm -rf ' + target_dir)
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    # Create train, validation and test directories if they don't exist
    train_dir = os.path.join(target_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    val_dir = os.path.join(target_dir, 'val')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    test_dir = os.path.join(target_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Get speakers as directory names in vctk_path
    speakers = os.listdir(vctk_path)

    #Shuffle speakers list
    speakers = np.random.permutation(speakers)
    # Divide speakers list into train, validation and test sets
    train_speakers = speakers[:int(len(speakers) * 0.8)]
    val_speakers = speakers[int(len(speakers) * 0.8):int(len(speakers) * 0.9)]
    test_speakers = speakers[int(len(speakers) * 0.9):]

    # Save train, validation and test speakers lists in one file in target_dir
    speakers_dict = {"train": train_speakers, "val": val_speakers, "test": test_speakers}
    torch.save(speakers_dict, os.path.join(target_dir, 'speakers.pt'))

    print("Dividing Speakers")

    # Copy audio files in each speaker directory to train, validation and test directories
    trainfile = open(os.path.join(target_dir, "train_files.txt"), 'w')
    for speaker in tqdm(train_speakers, total = len(train_speakers)):
        speaker_dir = os.path.join(vctk_path, speaker)
        speaker_files = glob.glob(os.path.join(speaker_dir, '*' + file_ext))
        for file in speaker_files:
            os.system('cp ' + file + ' ' + train_dir)
        trainfile.writelines([str(i)+'\n' for i in speaker_files])
    trainfile.close()

    valfile = open(os.path.join(target_dir, "val_files.txt"), 'w')
    for speaker in tqdm(val_speakers, total = len(val_speakers)):
        speaker_dir = os.path.join(vctk_path, speaker)
        speaker_files = glob.glob(os.path.join(speaker_dir, '*' + file_ext))
        for file in speaker_files:
            os.system('cp ' + file + ' ' + val_dir)
        valfile.writelines([str(i)+'\n' for i in speaker_files])
    valfile.close()

    testfile = open(os.path.join(target_dir, "test_files.txt"), 'w')
    for speaker in tqdm(test_speakers,total = len(test_speakers)):
        speaker_dir = os.path.join(vctk_path, speaker)
        speaker_files = glob.glob(os.path.join(speaker_dir, '*' + file_ext))
        for file in speaker_files:
            os.system('cp ' + file + ' ' + test_dir)
        testfile.writelines([str(i)+'\n' for i in speaker_files])
    testfile.close()


def process_directory(path, target_sr, feature_extractor, median_filter, pre_emphasis = None, file_ext = '.wav'):
    file_list = glob.glob(os.path.join(path, '*' + file_ext))
    for file in tqdm(file_list, total=len(file_list)):
        basename = os.path.basename(file)
        no_ext = os.path.splitext(basename)[0]

        formants, energy, centroid, tilt, log_pitch, voicing_flag, r_coeff, ignored = process_file(file, target_sr, feature_extractor, median_filter, pre_emphasis = pre_emphasis)

        if formants.size(0) < log_pitch.size(0):
            raise ValueError("Formants size is different than pitch size for file: " + file)

        feature_dict = {"Formants": formants, "Energy": energy, "Centroid": centroid, "Tilt": tilt, "Pitch": log_pitch, "Voicing": voicing_flag, "R_Coeff": r_coeff} 
        if not ignored:    
            torch.save(feature_dict, os.path.join(path, no_ext + '.pt'))
        else:
            print("File: " + basename + " ignored.")

def process_file(file, target_sr, feature_extractor, median_filter, pre_emphasis = None):
    """
    Extract features for a single audio file and return feature arrays with the same length.
    Params:
        file: path to file
        target_sr: target sample rate
    Returns:
        formants: formants
        energy: energy in log scale
        centroid: spectral centroid
        tilt: spectral tilt
        tilt_ref: spectral tilt from reference
        pitch: pitch in log scale
        voicing_flag: voicing flag
        r_coeff: reflection coefficients from reference
    """
    # Read audio file
    x, sample_rate = ta.load(file)
    x = x[0:1].type(torch.DoubleTensor)
    # Resample to target sr
    x = ta.functional.resample(x, sample_rate, target_sr)

    if pre_emphasis is not None:
        x = pre_emphasis(x.unsqueeze(0))
    x = x.squeeze(0).squeeze(0)
    formants, energy, centroid, tilt, pitch, voicing_flag,r_coeff, _, ignored = feature_extractor(x)

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
    
    log_pitch = torch.log(pitch)

    return formants, energy, centroid, tilt, log_pitch, voicing_flag, r_coeff, ignored

if __name__ == "__main__":
    vctk_path = '/workspace/Dataset/wav48_silence_trimmed'
    target_dir = '/workspace/Dataset/vctk_features'
    main(vctk_path, target_dir)