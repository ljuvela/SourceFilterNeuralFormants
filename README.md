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

## Table of contents
1. [Model structure](#model_struct)
2. [Repository installation](#install)
    1. [Conda environment](#conda)
    2. [GlotNet](#glotnet)
    3. [HiFi-GAN](#hifi)
    4. [Additional libraries](#additional)
3. [Pre-trained models](#pretrained)
4. [Inference](#inference)
5. [Training](#training)
6. [Citation information](#citation)

## Model structure <a name="model_struct"></a>

We present a model that performs neural speech synthesis using the structure of the source-filter model, allowing to independently inspect and manipulate the spectral envelope and glottal excitation:

![Neural formant pipeline follwing the source-filter model architectrue](./Images/DDSPNF_Diagram.png "Neural formant pipeline follwing the source-filter model architectrue.")

A description of the presented model and sound samples compared to other synthesis/manipulation systems can be found in the [project's demo webpage][demopage_link]

## Repository installation <a name="install"></a>

#### Conda environment <a name="conda"></a>

First, we need to create a conda environment to install our dependencies.
```sh
conda create -n neuralformants python=3.10 
conda activate neuralformants
```

#### GlotNet <a name="glotnet"></a>
GlotNet module is required for WaveNet models and DSP functions. Full repository is available [here][GlotNet_link]

We can clone the Glotnet repository in the root directory of this project and follow the instructions in it for installation. The process involves building some PyTorch C++ extensions and will take a few minutes.

```sh
git clone https://github.com/ljuvela/GlotNet.git
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

#### HiFi-GAN <a name="hifi"></a>
HiFi-GAN is included in the `hifi_gan` subdirectory. Original source code is available [here][HiFi_link]

#### Additional libraries <a name="additional"></a>

Additional libraries that can't be found through conda need to be installed using pip.

```sh
pip install diffsptk pyworld
```

## Pre-trained model <a name="pretrained"></a>

Pre-trained models for every module of the proposed system are stored in HuggingFace. Check the model card at https://huggingface.co/pablopz/SourceFilterNeuralFormants

In order to download them, first it is necessary to have [git-lfs][lfs_link] installed.

```sh
# install git-lfs with conda in case it's missing
conda install -c conda-forge git-lfs
```

```sh
# Activate git-lfs and download pre-trained models
git lfs install
git clone https://huggingface.co/pablopz/SourceFilterNeuralFormants 
```

The config.json files in the pretrained models are adapted to work if the models are cloned in the root directory of the project. Change the path defined in those files if necessary.

## Inference <a name="inference"></a>

We provide a script to run inference on the end-to-end architecture, such that an audio file can be provided as input and a wav file with the manipulated features is stored as output.

In order to run inference on the proposed model, run the following command:

```sh
python inference_from_list.py --input_path "[path to directory with audio samples to process]" --output_path "[path to output directory]" --config "[path to HiFi-GAN config file]" --fm_config "[path to feature mapping model config file]" --env_config "[path to envelope estimation config file]" --checkpoint_path "[path to checkpoint file]" --feature_scale "[scale array]"
```

An example with the provided audio samples from the VCTK dataset can be run using:

```sh
python inference_from_list.py --input_path "./Samples" --output_path "./Generated_Samples" --config "./SourceFilterNeuralFormants/HiFiExcitation/config.json" --fm_config "./SourceFilterNeuralFormants/FeatureMapping/config.json" --env_config "./SourceFilterNeuralFormants/EnvelopeEstimator/config.json" --checkpoint_path "./SourceFilterNeuralFormants/HiFiExcitation" --feature_scale "[1.0,1.0,1.0,1.0,1.0]"
```

The input to the inference script is provided as a txt file containing a list of paths to each of the audio files to process, separated by end of line. Additionally, the parameter [scale array] is a string with the format "[F0, F1, F2, F3, F4]", where each of the elements represents the scaling factor applied to each of the corresponding parameters.

## Model training <a name="training"></a>

Training of the HiFi-GAN model excitation is possible in the end-to-end architecture by using the the script "train_e2e_DDSPNF.py".

More details on how to train the different models will be provided in the future.

## Citation information <a name="citation"></a>

Citation information will be added when a pre-print is available.
