# CrackDetection-Utils-for-Python

[简体中文](README_ZH.md) · [English](README_EN.md)

This repository contains some modules for Crack Detection and Semantic Segmentation, mainly using Window 11 + Conda virtual environment. 

The environment configuration is [here](#config).

----
## Structure

```python
/crack_detection_util
    /image_util.py
        class MyImage
        class ChunkAndMerge
    /evaluate_util.py
        class Evaluator
```

Refer to [usage](USAGE.md) for package usage.

----

<a id='config'></a>
## Environment Configuration

The environment configuration of modules is listed below, including system environment and python module dependencies. 

**Generally**, even when the versions of referred python modules are different, the program is able to run normally.

```js
Python version: 3.10.13
OS: Windows

opencv-contrib-python-headless==4.9.0.80
opencv-python-headless==4.9.0.80

numpy==1.23.5
pandas==2.2.1
Pillow==9.5.0 (important)

torch==1.13.1+cu116
torch-tb-profiler==0.4.3
torchaudio==0.13.1+cu116
torchvision==0.14.1+cu116

tensorboard==2.15.2
tensorboard-data-server==0.7.2
tensorflow==2.15.0
tensorflow-estimator==2.15.0
tensorflow-intel==2.15.0
tensorflow-io-gcs-filesystem==0.31.0
```

Using a higher version of the Pillow module may cause errors in the use of the pycocotools module. It is recommended to install 9.5.0 version.

Check [this file](../docs/FULLLIST.md) if any critical module is missing.