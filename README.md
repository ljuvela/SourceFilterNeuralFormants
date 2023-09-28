# Neural Formant Synthesis with Differentiable Resonant Filters

Neural formant synthesis using differtiable resonant filters and source-filter model structure.

Authors: [Pablo Pérez Zarazaga][pablo_profile], [Zofia Malisz][zofia_profile], [Gustav Eje Henter][gustav_profile], [Lauri Juvela][lauri_profile]

[HiFi_link]: https://github.com/jik876/hifi-gan
[GlotNet_link]: https://github.com/ljuvela/GlotNet
[arxiv_link]: http://arxiv.org/abs/placeholder_link
[demopage_link]: https://perezpoz.github.io/DDSPneuralformants
[gustav_profile]: https://people.kth.se/~ghe/
[pablo_profile]: https://www.kth.se/profile/pablopz
[zofia_profile]: https://www.kth.se/profile/malisz
[lauri_profile]: https://research.aalto.fi/en/persons/lauri-juvela

[lfs_link]:https://git-lfs.com

## Model examples

A description of the presented model and sound samples compared to other synthesis/manipulation systems can be found in the [project's demo webpage][demopage_link]



Link to demo page and small example of generated sounds.

## Repository installation

#### Conda environment

First, we need to create a conda environment to install our dependencies.
```sh
conda create -n neuralformants python=3.10 
conda activate neuralformants
```

#### GlotNet
GlotNet module is required for some models and functions. Available here [here][GlotNet_link]

We can clone the Glotnet repository in the root directory of this project and follow the instructions in it for installation.

```sh
git clone git@github.com:ljuvela/GlotNet.git
cd GlotNet

# Install GlotNet requirements in with conda
conda install -c pytorch -c conda-forge pytorch torchaudio tensorboard scikit-build matplotlib pandas cmake eigen ninja pytest

# Clone git submodules
git submodule update --init --recursive

# Build extensions and install
pip install -v .

# Run pytest unit tests to check everthing works correctly
pytest test

# Return to root directory
cd ..

```

#### HiFi-GAN
HiFi-GAN is included in the `hifi_gan` subdirectory. Original source code is available [here][HiFi_link]

#### Additional libraries

Additional libraries that can't be found through conda need to be installed using pip.

```sh
pip install diffsptk pyworld
```

## Pre-trained model

Pre-trained models for every module of the proposed system are stored in HuggingFace. In order to download them, first it is necessary to have [git-lfs][lfs_link] installed.

```sh
# Activate git-lfs and download pre-trained models
git lfs install
git clone git@hf.co:pablopz/SourceFilterNeuralFormants
```

The config.json files in nthe pretrained models are adapted to work if the models are cloned in the root directory of the project. Change the path defined in those files if necessary.

## Inference

We provide a script to run inference on the end-to-end architecture, such that an audio file can be provided as input and a wav file with the manipulated features is stored as output.

In order to run inference on the proposed model, run the following command:

```sh
python inference_from_list.py --list_file "[path to txt file with file list]" --output_path "[path to output directory]" --config "[path to HiFi-GAN config file]" --fm_config "[path to feature mapping model config file]" --env_config "[path to envelope estimation config file]" --checkpoint_path "[path to checkpoint file]" --feature_scale "[scale array]"
```

The input to the inference script is provided as a txt file containing a list of paths to each of the audio files to process, separated by end of line. Additionally, the parameter [scale array] is a string with the format "[F0, F1, F2, F3, F4]", where each of the elements represents the scaling factor applied to each of the corresponding parameters.

## Model training

Training of the HiFi-GAN model excitation is possible in the end-to-end architecture by using the the script "train_e2e_DDSPNF.py".

More details on how to train the different models will be provided in the future.

## Citation information

Citation information will be added when a pre-print is available.