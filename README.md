# TRM: Phrase-level Temporal Relationship Mining for Temporal Sentence Localization

Official Pytorch implementation of Phrase-level Temporal Relationship Mining for Temporal Sentence Localization (AAAI2023).

## Abstract

In this paper, we address the problem of video temporal sentence localization, which aims to localize a target moment from videos according to a given language query. We observe that existing models suffer from a sheer performance drop when dealing with simple phrases contained in the sentence. It reveals the limitation that existing models only capture the annotation bias of the datasets but lack sufficient understanding of the semantic phrases in the query. To address this problem, we propose a phrase-level Temporal Relationship Mining (TRM) framework employing the temporal relationship relevant to the phrase and the whole sentence to have a better understanding of each semantic entity in the sentence. Specifically, we use phrase-level predictions to refine the sentence-level prediction, and use Multiple Instance Learning to improve the quality of phrase-level predictions. We also exploit the consistency and exclusiveness constraints of phrase-level and sentence-level predictions to regularize the training process, thus alleviating the ambiguity of each phrase prediction. The proposed approach sheds light on how machines can understand detailed phrases in a sentence and their compositions in their generality rather than learning the annotation biases. Experiments on the ActivityNet Captions and Charades-STA datasets show the effectiveness of our method on both phrase and sentence temporal localization and enable better model interpretability and generalization when dealing with unseen compositions of seen concepts.

## Pipeline

![pipeline](imgs/pipeline.png)

## Requiments

- pytorch
- h5py
- yacs
- terminaltables
- tqdm
- transformers

## Quick Start

### Data Preparation

We use the C3D feature for the ActivityNet Captions dataset. Please download from [here](http://activity-net.org/challenges/2016/download.html) and save as `dataset/ActivityNet/sub_activitynet_v1-3.c3d.hdf5`. We use the VGG feature provided by [2D-TAN](https://github.com/microsoft/VideoX) for the Charades-STA dataset, which can be downloaded from [here](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471415879). Please save it as `dataset/Charades-STA/vgg_rgb_features.hdf5`.

### Training

To train on the ActivityNet Captions dataset:
```bash
sh scripts/anet_train.sh
```

To train on the Charades-STA dataset:
```bash
sh scripts/charades_train.sh
```

You can change the options in the shell scripts, such as the GPU id, configuration file, et al.

### Inference

Run the following commands for evaluation:

```bash
sh scripts/eval.sh
```

Please change the configuration file and the directory of the saved weight in the shell script.
