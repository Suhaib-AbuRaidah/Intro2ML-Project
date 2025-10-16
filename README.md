# Intro2ML-Project

# 6D Object Pose Estimation from RGB-D Data (Our Proposed Model)

## Abstract
This repository implements a deep learning framework for **6D object pose estimation** using **RGB-D input**. The model predicts object transformations directly from completed or enhanced depth data. The framework is still under active development to support object segmentation and raw depth processing in future.

## Project Structure

6DPOSE/
│
├── data/ # Datasets and preprocessed inputs
├── model.py/ # Network architectures
├── utils.py/ # Helper scripts and visualization tools
├── datasets.py/ # Pytorch dataset and dataloader objects
├── train.py # Training script
├── evaluate.py # Evaluation script
├── inference.py # Inference and visualization
└── pre_trained_models/ # Containing checkpoints and pretraind models

## Installation
```bash
# Clone the repository
git clone https://github.com/Suhaib-AbuRaidah/Intro2ML-Project.git
cd 6DPOSE

# Create a conda environment
conda create -n 6dpose python=3.8
conda activate 6dpose

### Install dependencies
pip install -r requirements.txt
```
## Training
```bash
python train.py \
  --data /path/to/dataset \
  --epochs 50 \
  --batch_size 4
```
To check Tensorboard during training
``` bash
tensorboard --logdir "./runs"
```
## Evaluation
```bash
python evaluate.py \
  --data /path/to/dataset \
  --model "pre_trained_models/2025-10-16 15:17:10/chkpt_best_model_val.pth"\
  --use_depth
```
## Inference and Visualization
``` bash
python inference.py \
  --input /path/to/dataset \
  --model "pre_trained_models/2025-10-16 15:17:10/chkpt_best_model_val.pth" \
```

