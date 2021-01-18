[![CodeFactor](https://www.codefactor.io/repository/github/etomoscow/gmad_project/badge/main)](https://www.codefactor.io/repository/github/etomoscow/gmad_project/overview/main)

# GMAD 2020 Course Project.

This repository contains code for COVID-19 prediction on chest X-ray images. 

Used dataset is [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset). Link for download is [here](https://www.dropbox.com/s/4gnp630blcre0za/images.rar?dl=0).

## Prerequisites
```
python >= 3.6
pytorch >= 1.1.0
torchvision >= 0.2.2
numpy >= 1.14.3
imageio >= 2.4.1
pandas >= 0.22.0
opencv-python >= 3.4.2
matplotlib >= 3.0.2
sklearn >= 0.24.0
mogutda
```
## Installation and Running

1. Clone this repository: `git clone https://github.com/etomoscow/gmad_project`
2. `pip install requirements.txt`
3. Download [dataset](https://www.dropbox.com/s/4gnp630blcre0za/images.rar?dl=0) and unpack archive into `images` folder inside `project` folder, so the path to the images should be `../project/images/`.
4. Run `src/train_model.py` to train models and show performance. Example of usage:
```
  python train_model.py \
  --batch_size 32 \
  --n_epochs 100 \
  --learning_rate 0.0007 \
  --model_type resnet18 \
  --save_model
```
5. Obtain results.
