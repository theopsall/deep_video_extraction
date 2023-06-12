# Deep Feature Extraction

![APM](https://img.shields.io/apm/l/vim-mode?style=plastic)
[![Generic badge](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)](https://shields.io/)

[comment]: <> (![GitHub issues]&#40;https://img.shields.io/github/issues/theopsall/deep_video_extraction?style=plastic&#41;)

## Description

This repo is used to extract deep features representation for video inputs using pretrained models. More specifically
VGG 19 is used as deep feature extractor.
The code is structured in pyTorch.

## Installation

```bash
git clone https://github.com/theopsall/deep_video_extraction.git
cd deep_video_extraction
pip install -r requirements.txt
```

## Usage

### 1. Get visual deep video features

To extract only the visual deep video features run the following command:

```bash
python deep_feature_extraction extractVisual -i <input_directory> [-m <model>] [-l <layers>] [-f] [-s] [-o <output_directory>]
```

`-i`, `--input`: Input directory with videos (required)</br>
`-m`, `--model`: The pretrained model (default: "vgg")</br>
`-l`, `--layers`: Number of layers to exclude from the pretrained model (default: -1)</br>
`-f`, `--flatten`: Flatten the last layer of the feature vector</br>
`-s`, `--store`: Store feature vectors</br>
`-o`, `--output`: Output directory</br>

### 2. Get aural deep video features

To extract only the aural deep video features run the following command:

```bash
python deep_feature_extraction extractAural -i <input_directory> [-m <model>] [-l <layers>] [-f] [-s] [-o <output_directory>]
```

`-i` , `--input`: Input directory with audios (required)</br>
`-m` , `--model`: The pretrained model (default: "resnet")</br>
`-l` , `--layers`: Number of layers to exclude from the pretrained model (default: -1)</br>
`-f` , `--flatten`: Flatten the last layer of the feature vector</br>
`-s` , `--store`: Store feature vectors</br>
`-o` , `--output`: Output directory</br>

### 3. Isolate audio from videos

```bash
python    deep_feature_extraction soundIsolation [-i <input_directory>] [-o <output_directory>]
```

`-i` , `--input`: Input directory with videos</br>
`-o` , `--output`: Output directory</br>

### 4.Extract spectrograms from audio files

```bash
python    deep_feature_extraction spectro [-i <input_directory>] [-o <output_directory>]
```

`-i`, `--input`: Input directory with audio files</br>
`-o`, `--output`: Output directory</br>
