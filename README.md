# Antipodal Robotic Grasping
We present a novel generative residual convolutional neural network based model architecture which detects objects in the camera’s field of view and predicts a suitable antipodal grasp configuration for the objects in the image.

This repository contains the implementation of the Generative Residual Convolutional Neural Network (GR-ConvNet) from the paper:

#### Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/antipodal-robotic-grasping-using-generative/robotic-grasping-on-cornell-grasp-dataset)](https://paperswithcode.com/sota/robotic-grasping-on-cornell-grasp-dataset?p=antipodal-robotic-grasping-using-generative)

If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry:

```
@inproceedings{kumra2020antipodal,
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network}, 
  year={2020},
  pages={9626-9633},
  doi={10.1109/IROS45743.2020.9340777}}
}
```

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/skumra/robotic-grasping.git
```

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd robotic-grasping
$ pip install -r requirements.txt
```

## Datasets

This repository supports both the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

#### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

Example for Jacquard dataset:

```bash
python train_network.py --dataset jacquard --dataset-path <Path To Dataset> --description training_jacquard --use-dropout 0 --input-size 300
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```

Example for Jacquard dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --iou-eval --use-dropout 0 --input-size 300
```

## Run Tasks
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:
```bash
python run_grasp_generator.py
```

## Run on a Robot
To run the grasp generator with a robot, please use our ROS implementation for Baxter robot. It is available at: https://github.com/skumra/baxter-pnp


- 建置
```bash
git clone [<https://github.com/skumra/robotic-grasping.git>](<https://github.com/skumra/robotic-grasping.git>)
python3.6 -m venv --system-site-packages venv
source venv/bin/activate
cd robotic-grasping
pip install --upgrade pip
pip install -r requirements.txt
pip install imagecodecs

```

- check torch use gpu

    ```python
    import torch

    torch.cuda.is_available()
    >>> True

    torch.cuda.current_device()
    >>> 0

    torch.cuda.device(0)
    >>> <torch.cuda.device at 0x7efce0b03be0>

    torch.cuda.device_count()
    >>> 1

    torch.cuda.get_device_name(0)
    >>> 'GeForce GTX 950M'

    ```

- Download Cornell Grasping Dataset

    [https://www.kaggle.com/oneoneliu/cornell-grasp](https://www.kaggle.com/oneoneliu/cornell-grasp)

### Offline 實作(選擇Cornell)

```python
python run_offline.py --network trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2add7aff-86c0-4f55-84e3-0046fd6685a4/Screenshot_2021-06-01_170025.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2add7aff-86c0-4f55-84e3-0046fd6685a4/Screenshot_2021-06-01_170025.png)

- ~~目前使用攝影機realsense還有一些問題,還須修改程式,原因它內建程式會去吃校正檔是一個原因,第二個原因是它校正檔的程式有問題(目前不解決,已使用其他程式)~~

### Run realtime use realsense

```python
python run_realtime.py --network trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ff490b3-7e1c-4749-8ab4-545f37ad880f/Screenshot_from_2021-06-04_16-55-13.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ff490b3-7e1c-4749-8ab4-545f37ad880f/Screenshot_from_2021-06-04_16-55-13.png)

### 使用心得：

1. 當然對於未知物件,偵測出來的夾取框能使用率極低
2. 非常容易受環境複雜度影響
3. 目前還未開始訓練不知道效果如何,不過是可訓練
4. 環境目前都沒有什麼問題
5. 偵測出來的夾取框只有一個(亦即不像YOLO一次吐出很多bounding boxes)
6. 目前有開啟cuda run程式 ,看似fps不高,目測大約10~15左右

### 未來改善：

1. 可先用YOLO或YOLACT辨識物件後,拉近鏡頭距離,減少背景干擾
2. 還需要現場訓練才知道訓練效果,使用的開源資料集,背景十分單純乾淨
3. 下一步,先使用三指夾具使用的那一包,查看效果