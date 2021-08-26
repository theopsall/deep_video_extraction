# Deep Feature Extraction 

![APM](https://img.shields.io/apm/l/vim-mode?style=plastic)
[![Generic badge](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)](https://shields.io/)

[comment]: <> (![GitHub issues]&#40;https://img.shields.io/github/issues/theopsall/deep_video_extraction?style=plastic&#41;)
## Description
This repo is used to extract deep features representation for video inputs using pretrained models. More specifically 
VGG 19 is used as deep feature extractor. 
The code is structured in pyTorch.

##Installation
```bash
git clone https://github.com/theopsall/deep_video_extraction.git
cd deep_video_extraction
pip install -r requirements.txt
```

##Usage
### 1. Get deep video features
To extract the deep video features run the following command:  

```bash
python3 extract
```
`-d`:

`-v`:

`-l`:

`-m`:


OR call the python function: 


```python

```
### 2. Get visual deep video features
To extract only the visual deep video features run the following command:  


```bash
python3 extractVisual
```
`-d`:

`-v`:

`-l`:

`-m`:


OR call the python function: 



```python

```
### 3. Get aural deep video features
To extract only the aural deep video features run the following command:  


```bash
python3 extractAural
```
`-d`:

`-v`:

`-l`:

`-m`:


OR call the python function: 



```python

```