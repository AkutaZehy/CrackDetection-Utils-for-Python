# CrackDetection-Utils-for-Python

[简体中文](README_ZH.md) · [English](README_EN.md)

该存储库包含一些用于裂缝检测和语义分割的库，主要使用Windows 11 + Conda虚拟环境。

环境配置在[这里](#config)。

----
## 结构

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
## 环境配置

下面列出了库的环境配置，包括系统环境和Python库依赖。

**通常情况下**，即使所引用的Python库的版本不同，程序仍然能够正常运行。

```js
Python version: 3.10.13
OS: Windows

opencv-contrib-python-headless==4.9.0.80
opencv-python-headless==4.9.0.80

numpy==1.23.5
pandas==2.2.1
Pillow==9.5.0 (关键)

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

使用较高版本的Pillow库可能会导致在使用pycocotools库时出现错误。建议安装9.5.0版本。

如果仍旧缺少任何关键库，请查看[此文件](../docs/FULLLIST.md)。
