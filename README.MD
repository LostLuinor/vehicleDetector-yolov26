# Multi-Class Vehicle Detection

Computer vision project for preparing, analyzing, and training a multi-class vehicle detection system built around YOLO-format labels.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Structure & Statistics](#dataset-structure--statistics)
3. [Data Preparation & Preprocessing](#data-preparation--preprocessing)
4. [Dataset Analysis & Visualization](#dataset-analysis--visualization)
5. [Model Architecture](#model-architecture)
6. [Loss Functions](#loss-functions)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation](#evaluation)
9. [Project Layout](#project-layout)
10. [Future Work](#future-work)

## Project Overview

This repository focuses on the data engineering side of a multi-class vehicle detection workflow. It includes dataset splitting, preprocessing, augmentation, and inspection tools built around YOLO labels.

Key details:

- 4 vehicle classes: `0`, `1`, `2`, and `3`
- YOLO annotation format: class index, normalized center `x`, normalized center `y`, normalized width, normalized height
- Main tools: PyTorch, OpenCV, Albumentations, Pandas, Matplotlib, and Seaborn
- Base checkpoint: `yolo26m.pt`
- Target input resolution: `640 x 640`

## Dataset Structure & Statistics

The dataset is already organized into training, validation, and test splits.

### Split Summary

| Split | Images |
| --- | ---: |
| Train | 7,066 |
| Val | 982 |
| Test | 786 |

### Train Class Distribution

| Class | Instances |
| --- | ---: |
| 0 | 1,052 |
| 1 | 47,517 |
| 2 | 4,529 |
| 3 | 5,004 |

The split logic uses a `carve_test_split` approach that tries to preserve class presence when possible and falls back to a random split when stratification is not feasible.

The current training notebook and analysis workflow are centered on the processed dataset already present in the repository, with the same preprocessing assumptions used across the notebooks and starter scripts.

## Data Preparation & Preprocessing

### Data Loading

- A custom `VehicleDataset` class loads image and YOLO label pairs.
- Images are returned as normalized float tensors with shape `[3, H, W]`.
- Inputs are resized to `640 x 640` for training and evaluation consistency.

### Augmentation Pipeline

The project uses Albumentations for both disk-based export transforms and in-memory training transforms.

Training augmentations include:

- `RandomResizedCrop` to `640 x 640` with scale range `0.5` to `1.0`
- `HorizontalFlip` with 50% probability
- `ColorJitter` for brightness, contrast, saturation, and hue variation
- `GaussNoise` with 20% probability
- `Blur` with blur limit `3` and 10% probability
- `RandomRain` with 10% probability
- ImageNet normalization

Validation augmentations include:

- `LongestMaxSize` to 640 pixels
- `PadIfNeeded` to `640 x 640` with a constant fill value of `114`
- ImageNet normalization

## Model Architecture

The planned detector is organized as a `VehicleDetectorYOLOv26` pipeline that combines a frozen YOLOv26 backbone with a custom neck and attention bridge.

Architecture highlights:

- Modified FPN + PAN neck with deformable `3 x 3` fusion blocks at merge points
- Dual-Domain Attention that mixes spatial and frequency-domain features
- `ChannelAttention`, `SpatialAttention` with kernel size `7`, and `FrequencyAttention` using 2D FFTs
- Separate attention weights across pyramid scales before the detection heads

The model design is intended to improve feature fusion and make the detection heads more robust on dense and imbalanced vehicle scenes.

## Loss Functions

The planned `VehicleDetectionLoss` combines multiple objectives to balance localization and classification quality.

- Classification loss: Varifocal loss with weight `0.5`
- Box loss: CIoU loss with weight `7.5`
- Distribution Focal Loss: weight `1.5` with `reg_max = 16`
- Class imbalance handling: class weights tuned as `1.2`, `1.0`, `1.3`, and `1.4`

This setup is aimed at improving box regression quality while keeping the classifier stable under uneven class frequencies.

## Training Pipeline

The training plan uses a custom `Trainer` loop with modern optimization and stability features.

- Optimizer: AdamW
- Learning-rate schedule: OneCycleLR
- Mixed precision: AMP with gradient scaling
- EMA: model weight smoothing with decay `0.9999`
- Gradient accumulation: 4 steps
- Early stopping: stop after 20 epochs without mAP@0.5 improvement

A short smoke-training run was also used to validate the pipeline end to end:

- 1 epoch at image size `256`
- Batch size `4`
- Baseline validation mAP@0.5: `0.155`

## Evaluation

Evaluation is intended to focus on detection quality, class balance behavior, and whether the custom neck and attention bridge improve results over a plain baseline.

The current notebook workflow already supports visual inspection, dataset sanity checks, and small validation runs, which makes it a practical starting point for later full-scale evaluation experiments.

## Dataset Analysis & Visualization

The notebooks include checks that help verify the quality of the dataset before training:

- Class-count bar charts built with Seaborn and Matplotlib
- Split-wise aggregation of total class instances
- Bounding-box overlays for randomly selected training images to verify YOLO coordinates visually

The visualization step uses clear bounding boxes and labels to make annotation issues easier to spot quickly.

## Project Layout

- `augmentedDataset/` - processed dataset split into train, val, and test
- `data/` - dataset copy used by notebooks and experiments
- `competition_starter/` - starter scripts for training, prediction, and setup verification
- `shortTrainerWithGPU.ipynb` - GPU-focused notebook workflow
- `shortTrainerWithPlot.ipynb` - notebook with plotting and dataset inspection utilities
- `otherFiles/` - exploratory notebooks and supporting analysis

## Future Work

The current repository centers on preprocessing and analysis. Planned next steps are:

- Phase 3: Refine the model architecture and detection head design
- Phase 4: Tune the loss functions and optimization strategy
- Phase 5: Expand the training loop and experiment tracking
- Phase 6: Run full evaluation and testing experiments
- Phase 7: Deployment for an intelligent transportation system

If you want, the next logical improvement is to add a short "How to Run" section with notebook order, dependencies, and expected inputs.