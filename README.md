# Neural Formant Synthesis with Differentiable Resonant Filters

Neural formant synthesis using differtiable resonant filters and source-filter model structure.

Authors: [Lauri Juvela][lauri_profile], [Pablo Pérez Zarazaga][pablo_profile], [Gustav Eje Henter][gustav_profile], [Zofia Malisz][zofia_profile]

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

![Neural formant pipeline follwing the source-filter model architectrue](./Images/LPC-NFS.png "Neural formant pipeline follwing the source-filter model architectrue.")

A description of the presented model and sound samples compared to other synthesis/manipulation systems can be found in the [project's demo webpage][demopage_link]

## Repository installation <a name="install"></a>

#### Conda environment <a name="conda"></a>

First, we need to create a conda environment to install our dependencies. Use mamba to speed up the process if possible.
```sh
mamba env create -n neuralformants -f environment.yml
conda activate neuralformants
```

Pre-trained models are available in HuggingFace, and can be downloaded using git-lfs. If you don't have git-lfs installed (it's included in `environment.yml`), you can find it [here][lfs_link]. Use the following command to download the pre-trained models:
```sh
git submodule update --init --recursive
```

Install the package in development mode:
```sh
pip install -e .
```


#### GlotNet <a name="glotnet"></a>
GlotNet is included partially for WaveNet models and DSP functions. Full repository is available [here][GlotNet_link]


#### HiFi-GAN <a name="hifi"></a>
HiFi-GAN is included in the `hifi_gan` subdirectory. Original source code is available [here][HiFi_link]

## Inference <a name="inference"></a>

We provide a script to run inference on the end-to-end architecture, such that an audio file can be provided as input and a wav file with the manipulated features is stored as output.

Change the feature scaling to modify pitch (with F0) or formants. The scales are provided as a list of 5 elements with the following order:
```python
[F0, F1, F2, F3, F4]
```
An example with the provided audio samples from the VCTK dataset can be run using:

HiFi-Glot
```sh
python inference_hifiglot.py \
    --input_path "./Samples" \
    --output_path "./output/hifi-glot" \
    --config "./checkpoints/HiFi-Glot/config_hifigan.json" \
    --fm_config "./checkpoints/HiFi-Glot/config_feature_map.json" \
    --checkpoint_path "./checkpoints/HiFi-Glot" \
    --feature_scale "[1.0, 1.0, 1.0, 1.0, 1.0]"
```

NFS
```sh
python inference_hifigan.py \
    --input_path "./Samples" \
    --output_path "./output/nfs" \
    --config "./checkpoints/NFS/config_hifigan.json" \
    --fm_config "./checkpoints/NFS/config_feature_map.json" \
    --checkpoint_path "./checkpoints/NFS" \
    --feature_scale "[1.0, 1.0, 1.0, 1.0, 1.0]"
```

NFS-E2E
```sh
python inference_hifigan.py \
    --input_path "./Samples" \
    --output_path "./output/nfs-e2e" \
    --config "./checkpoints/NFS-E2E/config_hifigan.json" \
    --fm_config "./checkpoints/NFS-E2E/config_feature_map.json" \
    --checkpoint_path "./checkpoints/NFS-E2E" \
    --feature_scale "[1.0, 1.0, 1.0, 1.0, 1.0]"
```


## Model training <a name="training"></a>

Training of the HiFi-GAN and HiFi-Glot models is possible with the end-to-end architecture by using the the scripts `train_e2e_hifigan.py` and `train_e2e_hifiglot.py`.


## Citation information <a name="citation"></a>

Citation information will be added when a pre-print is available.
