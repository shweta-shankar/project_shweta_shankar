# ASL Alphabet Recognition

## Overview
This project implements a Deep Learning Image Classification pipeline to recognize American Sign Language (ASL) alphabets. The objective is to learn a mapping from input images of hand gestures to their corresponding output classes (29 total classes: A-Z, space, del, nothing).

For this project, two different model architectures were compared:
1. **Custom CNN (Baseline)**: A 4-block Convolutional Neural Network trained from scratch.
2. **MobileNetV2 (Fine-Tuned)**: A pretrained MobileNetV2 architecture adapted for 29-way classification. Transfer learning was conducted in two phases (head-only training followed by full unfreezing and fine-tuning).

## Directory Structure
Because this is a standard classification (input/output pair) task, the `project_shweta_shankar` repository strictly adheres to the suggested nomenclature and structure:

*   `config.py`: Contains all hyperparameters including `resize_x`, `resize_y`, `input_channels`, `batch_size`, `number_of_epochs`, learning rates, and dataset normalization stats.
*   `dataset.py`: Contains the data loading logic (`get_dataloaders`) and the custom dataset class, handling image resizing and data augmentation.
*   `model.py`: Contains the PyTorch definitions for both `CustomASLNet` and the `MobileNetV2` setup.
*   `train.py`: Contains the `train_model` function, running the training and evaluation loops, and automatically saving the best model to the `checkpoints/` folder.
*   `predict.py`: Contains `inferloader` and `classify_images` functions to take a batch of raw image paths, preprocess them, and yield classification labels.
*   `interface.py`: The standardized interface file mapping the custom functions and variables to the required grading variables (e.g., `TheModel`, `the_trainer`, `the_predictor`).
*   `data/`: Contains a sample of 10 raw `.jpg` images per class from the Kaggle dataset.
*   `checkpoints/`: Contains the best trained model weights for both models. To satisfy the automated grading script, the MobileNetV2 weights (the best performing model) are saved as `final_weights.pth`. The baseline CNN weights are also included as `custom_cnn_best.pth`.

### Note for the Automated Grader
Because the automated grading script strictly looks for `TheModel` and `final_weights.pth`, MobileNetV2 architecture is aliased to `TheModel` in `interface.py`, and `final_weights.pth` contains the corresponding MobileNetV2 weights. If you wish to test the baseline `CustomASLNet` model, you can change the import alias in `interface.py` and point it to the included `custom_cnn_best.pth` checkpoint.

## How to Run

### 1. Training
The hyperparameters are controlled via `config.py`. To initiate the training loop, which will run for the specified epochs and save the best weights to the `checkpoints` directory, execute:
```bash
python train.py
```

### 2. Inference / Prediction
Inference can be run on a list of image paths using the functions inside `predict.py`. It automatically resizes and normalizes the input before passing it to the model.

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

## Training Context & Evaluation
**Note on Local Setup:** The `data/` directory provided in this submission contains a quick sample set (10 images per class) strictly to fulfill the submission structure requirements and demonstrate that the data-loading and inference pipelines work correctly.

The actual training on the full ASL dataset was conducted entirely in a cloud environment on Kaggle (as outlined in the project proposal), because a standard local laptop does not have the hardware capacity to handle the massive dataset and intensive training workload. The `final_weights.pth` file provided in the `checkpoints/` directory was downloaded directly from those completed Kaggle training sessions.

From the full-scale Kaggle experiments, the following were achieved:
*   **Custom CNN**: Reached a validation accuracy of ~90%.
*   **MobileNetV2**: Achieved near 100% validation accuracy after fine-tuning.

As outlined in the project proposal, a comprehensive error analysis was conducted. Detailed **confusion matrices** and **per-class accuracy plots** were generated to understand misclassifications. Additionally, **Grad-CAM** visualizations were performed in the Kaggle development notebook to verify that the MobileNetV2 model accurately focused its attention on the hand gestures rather than background noise.

For full training logs, exploratory data analysis, and Grad-CAM visualizations on the entire dataset, please refer to the original [Kaggle Notebook](https://www.kaggle.com/code/shshta/notebookb94d182b2d).

