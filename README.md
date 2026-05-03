# American Sign Language (ASL) Alphabet Recognition Using Deep Learning

**Course:** Image and Video Processing with Deep Learning (DS3273)  
**Author:** Shweta Shankar (20231241) 

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Image Processing Pipeline](#6-image-processing-pipeline)
7. [How to Run](#7-how-to-run)
8. [File Descriptions](#8-file-descriptions)
9. [Results](#9-results)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Outputs and Visualizations](#11-outputs-and-visualizations)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

This project implements a **Deep Learning Image Classification pipeline** to recognize American Sign Language (ASL) alphabets. The objective is to learn a mapping from input images of hand gestures to their corresponding output classes (29 total classes: A-Z, space, del, nothing).

### Why This Problem Matters
Sign language recognition bridges the communication gap between the Deaf community and the hearing majority. An automated deep learning system enables real-time interpretation of gestures, which can be applied to:
- **Assistive communication tools**
- **Educational software for learning ASL**
- **Human-computer interaction**

### The 29 Classes
The model predicts hand gestures belonging to 29 categories:
- **Alphabets**: A to Z (26 classes)
- **Special Gestures**: `space`, `del`, `nothing` (3 classes)

---

## 2. Directory Structure

```text
project_shweta_shankar/
│
├── checkpoints/
│   ├── custom_cnn_best.pth        ← Trained baseline CNN model weights
│   └── final_weights.pth          ← Trained MobileNetV2 weights (best model)
│
├── data/                          ← Contains sample images from Kaggle dataset
│   ├── A/                         
│   ├── B/                         
│   └── ...                        
│
├── config.py                      ← All hyperparameters and paths
├── dataset.py                     ← Custom Dataset class and DataLoader logic
├── model.py                       ← Definitions for CustomASLNet & MobileNetV2
├── train.py                       ← Training loop and evaluation
├── predict.py                     ← Inference on image file paths
└── interface.py                   ← Standardised interface for grading
```

---

## 3. Dataset

### ASL Alphabet Dataset (Kaggle)
- **Source:** [Kaggle — grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Image Size:** Originally 200x200, resized to 224x224 pixels
- **Format:** RGB `.jpg` images organized in class-specific folders
- **Labels:** Fully annotated — 29 total classes

### Dataset Split
The full dataset used during the Kaggle training session was split as follows:
- **Train Split:** 80% of the training dataset.
- **Validation Split:** 20% of the training dataset.

*(Note: A separate unseen Test dataset was provided on Kaggle. This was not used during training but was evaluated separately on Kaggle to generate the confusion matrices and final test metrics).*

### Note on Sample Data (`data/` directory)
Due to the massive size of the dataset, the `data/` directory provided in this local submission contains a quick sample set (10 images per class) strictly to fulfill the submission structure requirements and demonstrate that the data-loading and inference pipelines work correctly locally.

---

## 4. Model Architecture

Two models were developed and compared for this task:

### Baseline: Custom 4-Block CNN
A custom convolutional neural network was built from scratch:
- **Features**: 4 blocks of Conv2d (32, 64, 128, 256 filters) -> BatchNorm2d -> ReLU -> MaxPool2d.
- **Classifier**: AdaptiveAvgPool2d(4x4) -> Flatten -> Linear(1024) -> Dropout(0.5) -> Linear(29).

### Fine-Tuned Model: MobileNetV2
MobileNetV2, a highly efficient convolutional neural network pretrained on ImageNet (1000 classes), was adapted for this 29-way classification.
- **Why MobileNetV2?**: It offers an excellent trade-off between parameter size and accuracy, using depthwise separable convolutions that make it lightweight and highly effective for edge devices.
- **Custom Head**: The original classifier was replaced with a custom head consisting of a Dropout(0.3) layer and a Linear layer mapping from 1280 features to the 29 ASL classes.

---

## 5. Training Strategy

### Model 1: Custom CNN (Baseline)
- **Epochs:** 20
- **Learning Rate:** `1e-3` with ReduceLROnPlateau scheduler
- **Optimizer:** Adam with weight decay `1e-4`

### Model 2: MobileNetV2 (Two-Phase Transfer Learning)
Training was done in two phases to effectively transfer ImageNet features without catastrophic forgetting:

**Phase 1 — Head Training (10 Epochs)**
- **Backbone:** Frozen (weights locked)
- **Trainable:** Only the custom classifier head
- **Learning Rate:** `1e-3` (ReduceLROnPlateau scheduler)
- **Purpose:** Train the new classification head to map pretrained features to ASL classes.

**Phase 2 — Full Fine-tuning (10 Epochs)**
- **Backbone:** Unfrozen (all layers trainable)
- **Trainable:** Full network
- **Learning Rate:** `1e-4` (CosineAnnealingLR scheduler)
- **Purpose:** Fine-tune the entire network on the ASL imagery to adapt low-level features.

### Loss Function
**CrossEntropyLoss with Label Smoothing (α=0.1)**  
Label smoothing improves model generalization by preventing the network from becoming overconfident in its predictions.

---

## 6. Image Processing Pipeline

### Training Transforms (with Augmentation)
```
Input: Raw ASL image
  ↓  Resize → 224×224
  ↓  RandomHorizontalFlip (p=0.5)
  ↓  RandomRotation (±15°)
  ↓  ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2)
  ↓  RandomAffine (translate=0.1)
  ↓  ToTensor → [0,1] float tensor
  ↓  Normalize (ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Output: [3, 224, 224] normalized tensor
```
**Why these augmentations?**
- They simulate variations in hand orientation, lighting, and position, which are common in real-world scenarios, thereby preventing overfitting.

### Validation/Test Transforms
```
Input: Raw ASL image
  ↓  Resize → 224×224
  ↓  ToTensor
  ↓  Normalize (ImageNet stats)
Output: [3, 224, 224] normalized tensor
```

---

## 7. How to Run

### Option A — View Full Training on Kaggle
Because the complete training was performed on Kaggle due to hardware constraints, the easiest way to view the full execution, logs, outputs, and Grad-CAM visualizations is to view the Kaggle Notebook directly:
- **[Kaggle Notebook: notebookb94d182b2d](https://www.kaggle.com/code/shshta/notebookb94d182b2d)**

### Option B — Run Locally (For Grading)

**Prerequisites:**
```bash
pip install torch torchvision tqdm scikit-learn matplotlib seaborn pillow numpy
```

**1. Run Full Training via Command Line**
The hyperparameters are controlled via `config.py`. To execute the training pipeline locally, simply run:
```bash
python train.py
```

**2. Example: Importing as a Python Module (For Grading)**
The following examples demonstrate how the provided interface can be imported and used within an external script (such as an automated grading script):

*Example A: Running inference in a separate script*
```python
from predict import classify_images
from model import get_mobilenet_model
import torch

# Load model
model = get_mobilenet_model()
model.load_state_dict(torch.load('checkpoints/final_weights.pth'))

# Example inference
paths = ['data/A/img01.jpg', 'data/B/img01.jpg']
class_names = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

predictions = classify_images(model, paths, class_names)
print(predictions)
```

*Example B: Using the Standardised Interface*
```python
from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader
import torch.nn as nn
import torch.optim as optim

train_loader, val_loader = the_dataloader()
model = TheModel(num_classes=29)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# the_trainer(...) can be used following the interface definitions
```
*(Note: MobileNetV2 architecture is aliased to `TheModel` in `interface.py` to satisfy the grading script).*

---

## 8. File Descriptions

- **`config.py`**: Central configuration file containing hyperparameters (`BATCH_SIZE`, `IMG_SIZE`, learning rates, epochs, etc.).
- **`dataset.py`**: Contains data loading logic and PyTorch transformations.
- **`model.py`**: Contains PyTorch class definitions for `CustomASLNet` and the MobileNetV2 architecture setup.
- **`train.py`**: The training loop containing logic for phase 1 & 2 fine-tuning and evaluation.
- **`predict.py`**: Contains functions to run inference on a list of raw image paths.
- **`interface.py`**: Standardised interface mapping custom objects to grading program's expected names.

---

## 9. Results

### Test Set Performance
| Model | Validation Accuracy |
|-------|---------------------|
| Custom CNN (Baseline) | ~90.0% |
| MobileNetV2 (Fine-tuned) | **~99.0%+** |

The MobileNetV2 architecture significantly outperformed the custom baseline, converging faster and generalizing much better to the unseen validation and test data.

---

## 10. Evaluation Metrics

- **Accuracy**: The primary metric measuring the fraction of correct predictions across all 29 classes on both the validation set and the separate Kaggle test set.
- **Confusion Matrix**: A 29x29 matrix generated on the Kaggle test set to reveal specific class confusions. This analysis was crucial in identifying visually similar alphabets that the model occasionally struggled with (e.g., distinguishing 'M' vs 'N' or 'O' vs 'Q').
- **Per-Class Accuracy**: Extracted from the confusion matrix to identify specific classes that act as bottlenecks to the model's overall performance.

---

## 11. Outputs and Visualizations

Detailed exploratory data analysis, training curves, test set confusion matrices, and Grad-CAM visualizations were generated in the Kaggle environment. 
- **Confusion Matrix Visualization**: Displayed the model's performance across all 29 classes on the separate Kaggle test split, demonstrating near-perfect diagonal activations for the MobileNetV2 model.
- **Grad-CAM**: Applied to the final convolutional layer of MobileNetV2 to interpret the model's spatial attention. Visualizations confirmed the model focuses precisely on the hand gestures rather than background artifacts.

---

## 12. Dependencies

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework, models, and transforms |
| `numpy` | Numerical operations |
| `matplotlib`, `seaborn` | Plotting training curves, heatmaps, and Grad-CAM |
| `scikit-learn` | Metrics (accuracy, confusion matrix, classification report) |
| `Pillow` | Image loading |
