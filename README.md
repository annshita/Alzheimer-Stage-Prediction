# Alzheimer's Stage Prediction

This project focuses on predicting Alzheimer's disease stages using deep learning models applied to brain MRI data. Specifically, I have utilised the ConvNeXt architecture on ADNI and OASIS datasets, and MedViT on the ADNI dataset.

---

## Table of Contents

* [Introduction](#introduction)
* [Datasets](#datasets)
* [Model Architectures](#model-architectures)
* [Methodology](#methodology)
* [Results](#results)
* [Dependencies](#dependencies)
* [How to Run](#how-to-run)
* [Acknowledgements](#acknowledgements)

---

## Introduction

Alzheimer's Disease (AD) is a neurodegenerative disorder characterized by progressive cognitive decline. Early and accurate stage prediction can significantly help in timely interventions. This project explores deep learning-based MRI analysis for multi-class classification of Alzheimer's stages.

---

## Datasets

### ADNI (Alzheimer's Disease Neuroimaging Initiative)

* Multi-class MRI dataset categorized into: Alzheimer's Disease (AD), Cognitive Impairment (CI), and Cognitively Normal (CN).
* Preprocessed axial slices were used for training.

### OASIS (Open Access Series of Imaging Studies)

* Contains MRI scans categorized similarly into stages of cognitive decline.
* Used for additional training and validation with ConvNeXt.

---

## Model Architectures

### ConvNeXt

* Applied on both ADNI and OASIS datasets.
* Modern convolutional neural network architecture inspired by vision transformers.
* Used pretrained weights for transfer learning.

### MedViT

* Applied on ADNI dataset.
* Medical Vision Transformer architecture designed specifically for medical imaging tasks.
* Used pretrained weights to leverage learned features from large-scale medical image data.

---

## Methodology

1. **Data Preparation:**

   * Loaded and preprocessed MRI slices.
   * Resized images according to the requirement of tghe given models.

2. **Model Training:**

   * Fine-tuned pretrained ConvNeXt and MedViT models.
   * Hyperparameter tuning for optimal performance.
   * Training progress, losses, and accuracies logged at each epoch.

3. **Evaluation:**

   * Evaluated using metrics like accuracy, precision, recall, and confusion matrix.
   * Separate train/validation/test splits maintained.

---

## Results

| Model    | Dataset | Accuracy |
| -------- | ------- | -------- |
| ConvNeXt | ADNI    | 99.33%      |
| ConvNeXt | OASIS   | 99.45%      |
| MedViT   | ADNI    | 96.51%      |

---

## Dependencies

* Python 3.x
* PyTorch
* scikit-learn
* numpy
* pandas
* matplotlib
* torchvision
* timm (for ConvNeXt pretrained models)
* MedViT package (or custom MedViT implementation)
* tqdm

---

## How to Run

1. Clone this repository:

```bash
git clone <your-repo-url>
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the datasets (place ADNI and OASIS datasets in the `data/` directory following the expected folder structure).

4. Train the models:

```bash
python train_convnext.py  # For ConvNeXt training
python train_medvit.py    # For MedViT training
```

5. Evaluate the models:

```bash
python evaluate.py
```

---

## Acknowledgements

* ADNI: Alzheimer's Disease Neuroimaging Initiative
* OASIS: Open Access Series of Imaging Studies
* MedViT authors for pretrained models
* ConvNeXt authors for open-source pretrained models

---

## Contact

For any queries, feel free to contact: **Anshita Verma**

---

*Note: This project was conducted purely for academic and research purposes.*
