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

## Model examples

A description of the presented model and sound samples compared to other synthesis/manipulation systems can be found in the [project's demo webpage][demopage_link]



Link to demo page and small example of generated sounds.

## Repository installation

#### GlotNet
GlotNet module is required for some models and functions. Available here [here][GlotNet_link]

ToDo: Create conda environment for the project. Also repeated in GlotNet

HiFi-GAN is included in the `hifi_gan` subdirectory. Original source code is available [here][HiFi_link]

Clone HiFi-GAN repository in the root directory of this project:

```sh
git clone git@github.com:jik876/hifi-gan.git
```




## Pre-trained model

The pre-trained models can be found in HuggingFace.
ToDo: Upload pre-trained models to huggingface
Pre-trained models can be downloaded with:
ToDo: Code snippet to download models

## Inference

In order to run inference on the proposed model and manipulate speech parameters on utterance-level:

ToDo: modify python script to run using just a list of paths to a file, not parsing vctk samples.
```sh
python inference_from_list.py --input_file "[path to csv file with input wav files in vctk dataset]" --dataset_path "[path to vctk dataset]" --output_path ./test_generated_new --config "[path to HiFi-GAN config file]" --fm_config "[path to feature mapping model config file]" --env_config "[path to envelope estimation config file]" --checkpoint_path "[path to checkpoint file]" --feature_scale "[scale array]";
```
ToDo: Move sh scripts into code snippets.

Describe use for manipulation.

Include sound samples.

ToDo: complete pipeline of commands.

## Model training

## Citation information

Citation information will be added when a pre-print is available.