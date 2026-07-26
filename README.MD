# Multi-Class Vehicle Detection

A comprehensive computer vision project for computing dataset statistics, visualizations, and data preprocessing for a multi-class vehicle detection task using YOLO-format labels.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Structure & Statistics](#dataset-structure--statistics)
3. [Phase 1: Data Preparation & Preprocessing](#phase-1-data-preparation--preprocessing)
4. [Phase 2: Dataset Analysis & Visualization](#phase-2-dataset-analysis--visualization)
5. [Future Phases (Model Architecture & Training)](#future-phases)

---

## Project Overview

This project implements a data preparation and analysis pipeline for a **multi-class vehicle detection system** using:

* **Vehicle Classes:** 4 distinct classes (0, 1, 2, 3).


* **Label Format:** YOLO standard format representing class index, normalized center X, normalized center Y, normalized width, and normalized height.


* **Core Libraries:** The pipeline relies on PyTorch, OpenCV, Albumentations, and Pandas for processing and visualization.



---

## Dataset Structure & Statistics

The dataset has been successfully processed and split into training, validation, and testing sets.

### Splits and Organization

The dataset is divided using a `carve_test_split` function, which attempts a stratified split based on class presence and falls back to a random split if stratification fails.

**Image Counts by Split:**

* **Train:** 7,066 images.


* **Val:** 982 images.


* **Test:** 786 images.



**Class Distribution (Train Split):**

* **Class 0:** 1,052 instances.


* **Class 1:** 47,517 instances.


* **Class 2:** 4,529 instances.


* **Class 3:** 5,004 instances.



---

## Phase 1: Data Preparation & Preprocessing

### Data Loading

* A custom `VehicleDataset` PyTorch class loads image and YOLO-format label pairs.


* The dataset returns images as a `FloatTensor` of shape `[3, H, W]` that are normalized to a `[0,1]` range.


* Images are resized to a default target dimension of 640x640.



### Data Augmentation Pipeline

The project utilizes `Albumentations` for both disk-export transforms and memory transforms during training.

**Training Augmentations (`train_transforms`):**

* `RandomResizedCrop` with a target size of 640x640 and a scale range of 0.5 to 1.0.


* `HorizontalFlip` applied with a 50% probability.


* `ColorJitter` adjusting brightness (0.3), contrast (0.3), saturation (0.3), and hue (0.1).


* `GaussNoise` applied with a 20% probability.


* `Blur` with a blur limit of 3, applied with a 10% probability.


* `RandomRain` applied with a 10% probability to simulate weather variations.


* ImageNet `Normalize` standardizations.



**Validation Augmentations (`val_transforms`):**

* `LongestMaxSize` scaling to 640 pixels.


* `PadIfNeeded` expanding to 640x640 using a constant border mode filled with a value of 114 to maintain aspect ratios.


* ImageNet `Normalize` standardizations.



---

## Phase 2: Dataset Analysis & Visualization

The notebook includes a robust analysis phase to verify data integrity before model training:

* Bar charts generated via `seaborn` and `matplotlib` plot class counts across the augmented dataset by split.


* Total class instances per split are aggregated and visualized to ensure proper distribution.


* A bounding box visualization tool overlays YOLO labels onto random training images using lime green rectangle patches, drawing the class labels with a semi-transparent black background to verify coordinate accuracy.



---

## Future Phases

*(Note: The following sections represent the intended model architecture and deployment phases to be implemented alongside the completed data pipeline.)*

* **Phase 3:** Model Architecture (Targeting YOLOv26 / Advanced attention mechanisms)
* **Phase 4:** Loss Functions
* **Phase 5:** Training Loop & Execution
* **Phase 6:** Evaluation & Testing
* **Phase 7:** Deployment (Intelligent Transportation System)