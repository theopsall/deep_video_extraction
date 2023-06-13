# Deep Feature Extraction

![APM](https://img.shields.io/apm/l/vim-mode?style=plastic)
[![Generic badge](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)](https://shields.io/)

[comment]: <> (![GitHub issues]&#40;https://img.shields.io/github/issues/theopsall/deep_video_extraction?style=plastic&#41;)

## Description

Welcome to the "Deep Video Extraction" repository! This repository is designed to extract deep feature representations from video inputs using pre-trained models, such as VGG19 and ResNet, among others. In addition to processing videos, this repository also enables the extraction of deep features from audio by converting them into spectrograms. Furthermore, it provides the capability to extract audio from videos and spectograms.

The core functionality of this repository lies in its ability to iterate through a directory containing videos, applying deep feature extraction using state-of-the-art pre-trained models. Specifically, the renowned VGG19 model is employed as the primary deep feature extractor. The codebase is built using the popular deep learning framework, PyTorch, ensuring robustness and ease of use.

By leveraging the power of pre-trained models, this repository enables you to extract high-level representations from video inputs. These deep feature representations capture meaningful visual patterns, aiding in various tasks such as object recognition, video summarization, and content-based retrieval.

Moreover, the repository offers an additional dimension of processing by extracting deep features from audio. By converting audio signals into spectrograms, you can analyze the temporal and frequency components, unlocking valuable insights for audio-related tasks such as speech recognition, music classification, and audio event detection.

The repository also provides functionalities to extract the audio track from videos and the corresponding spectrograms. This empowers users to explore the audio content independently and perform tasks such as audio synthesis, audio captioning, or combining audio and visual modalities for multimodal analysis.

Overall, "Deep Video Extraction" serves as a comprehensive toolbox for deep feature extraction, catering to both video and audio inputs. By harnessing pre-trained models and leveraging PyTorch's flexibility, this repository empowers researchers, developers, and practitioners to unlock the rich information embedded in videos and audio signals, facilitating a wide range of applications in computer vision, multimedia analysis, and beyond.

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
