# BBAVectors-Oriented-Object-Detection
[WACV2021] Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors ([arXiv](https://arxiv.org/pdf/2008.07043.pdf))

Please cite the article in your publications if it helps your research:

	@inproceedings{yi2021oriented,
	title={Oriented object detection in aerial images with box boundary-aware vectors},
	author={Yi, Jingru and Wu, Pengxiang and Liu, Bo and Huang, Qiaoying and Qu, Hui and Metaxas, Dimitris},
	booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
	pages={2150--2159},
	year={2021}
	}


# Introduction

Oriented object detection in aerial images is a challenging task as the objects in aerial images are displayed in arbitrary directions and are usually densely packed. Current oriented object detection methods mainly rely on two-stage anchor-based detectors. However, the anchor-based detectors typically suffer from a severe imbalance issue between the positive and negative anchor boxes. To address this issue, in this work we extend the horizontal keypoint-based object detector to the oriented object detection task. In particular, we first detect the center keypoints of the objects, based on which we then regress the box boundary-aware vectors (BBAVectors) to capture the oriented bounding boxes. The box boundary-aware vectors are distributed in the four quadrants of a Cartesian coordinate system for all arbitrarily oriented objects. To relieve the difficulty of learning the vectors in the corner cases, we further classify the oriented bounding boxes into horizontal and rotational bounding boxes. In the experiment, we show that learning the box boundary-aware vectors is superior to directly predicting the width, height, and angle of an oriented bounding box, as adopted in the baseline method. Besides, the proposed method competes favorably with state-of-the-art methods.

<p align="center">
	<img src="assets/img1.png", width="800">
</p>

# How To Start

Install the [DOTA development kit](/datasets/DOTA_devkit/README.md) and the BBA Vectors modules running the following commands.

1. get in the project folder
```bash
    cd BBAVectors-Oriented-Object-Detection
    export PROJECT_FOLDER=$(pwd)
```

2. install swig and create the c++ extension for python
```bash
    cd $PROJECT_FOLDER/DOTA_devkit
    sudo apt-get install swig
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```
3. install the project modules
```bash
    cd $PROJECT_FOLDER
    pip install -e .
```

After install this package, you can start your experiments. I suggest you start by the `demo.ipynb` this file has a step by step instructions to guide you through the basics commands.

## About DOTA
### Split Image
Split the DOTA images from [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) before training, testing and evaluation.

The dota ```trainval``` and ```test``` datasets are cropped into ```600×600``` patches with a stride of `100` and two scales `0.5` and `1`. 

### About txt Files
The `trainval.txt` and `test.txt` used in `datasets/dataset_dota.py` contain the list of image names without suffix, example:
```
P0000__0.5__0___0
P0000__0.5__0___1000
P0000__0.5__0___1500
P0000__0.5__0___2000
P0000__0.5__0___2151
P0000__0.5__0___500
P0000__0.5__1000___0
```
Some people would be interested in the format of the ground-truth, I provide some examples for DOTA dataset:
Format: `x1, y1, x2, y2, x3, y3, x4, y4, category, difficulty`

Examples: 
```
275.0 463.0 411.0 587.0 312.0 600.0 222.0 532.0 tennis-court 0
341.0 376.0 487.0 487.0 434.0 556.0 287.0 444.0 tennis-court 0
428.0 6.0 519.0 66.0 492.0 108.0 405.0 50.0 bridge 0
```
## Data Arrangment
### DOTA
```
data_dir/
        images/*.png
        labelTxt/*.txt
        trainval.txt
        test.txt
```
you may modify `datasets/dataset_dota.py` to adapt code to your own data.
### HRSC
```
data_dir/
        AllImages/*.bmp
        Annotations/*.xml
        train.txt
        test.txt
        val.txt
```
you may modify `datasets/dataset_hrsc.py` to adapt code to your own data.


## Train Model
```ruby
python main.py --data_dir dataPath --epochs 80 --batch_size 16 --dataset dota --phase train
```

## Test Model
```ruby
python main.py --data_dir dataPath --batch_size 16 --dataset dota --phase test
```


## Evaluate Model
```ruby
python main.py --data_dir dataPath --conf_thresh 0.1 --batch_size 16 --dataset dota --phase eval
```

You may change `conf_thresh` to get a better `mAP`. 

Please zip and upload the generated `merge_dota` for DOTA [Task1](https://captain-whu.github.io/DOTA/evaluation.html) evaluation.


## Docker
### Building image
```bash
docker build -t "bbavectors" .
```

### Testing
```bash
bash run-docker.sh <model-dir> <image-path> <drone-altitude>  

Ex: bash run-docker.sh /home/docs/model_weights/ /home/docs/image-test.jpg 45
```
