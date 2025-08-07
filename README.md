# Hierarchical Multimodal Fusion for Whole Slide Image Analysis

This repository contains the official implementation of our paper:

**Hierarchical Multimodal Fusion of Whole Slide Images and Text for Patient-Level Cancer Staging**

> We propose a novel hierarchical framework that integrates whole slide images (WSIs) and GPT-generated textual prompts using Bi-Cross Attention for robust patient-level classification.

---

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Introduction

Recent deep learning methods for WSI analysis rely heavily on MIL-based models that lack patient-level understanding.

In this work, we propose a **Multi-modal Hierarchical (MH)** framework that:
- Integrates WSI features with textual descriptions via **Bi-Cross Attention (BCA)**
- Aggregates slide-level features for **patient-level** prediction
- Uses contrastive learning and hierarchical fusion to improve interpretability

---

## Model Architecture
![Model Architexture](./assets/MM-MIL.png)

## Dataset
We evaluate our method on three publicly available datasets:
### 1.[CAMELYON17](https://camelyon17.grand-challenge.org/)
- Task: Lymph node metastasis classification
- Data level: Slide-level & Patient-level
- Source: Grand Challenge

### 2.[TCGA-BRCA](https://portal.gdc.cancer.gov/)
- Task: Breast cancer subtyping
- Subtypes: IDC (Invasive Ductal Carcinoma), ILC (Invasive Lobular Carcinoma)

### 3.[TCGA-NSCLC](https://portal.gdc.cancer.gov/)
- Task: Non-small cell lung cancer subtyping
- Subtypes: LUAD (Lung Adenocarcinoma), LUSC (Lung Squamous Cell Carcinoma)
  
Label Distribution
| Dataset       | Level         | Class Label     | Count |
|---------------|---------------|------------------|-------|
| **CAMELYON17** | Slide-level   | Negative         | 318   |
|               |               | ITC              | 36    |
|               |               | Micro            | 59    |
|               |               | Macro            | 87    |
| **CAMELYON17** | Patient-level | pN0              | 120   |
|               |               | pN0(i+)          | 55    |
|               |               | pN1mi            | 105   |
|               |               | pN1              | 150   |
|               |               | pN2              | 70    |
| **TCGA-BRCA**  | Patient-level | IDC              | 779   |
|               |               | ILC              | 198   |
| **TCGA-NSCLC** | Patient-level | LUAD             | 478   |
|               |               | LUSC             | 478   |

To extract features from WSIs, we follow the pipeline proposed by the [CLAM framework (Mahmood Lab)](https://github.com/mahmoodlab/CLAM/tree/master?tab=readme-ov-file#wsi-segmentation-and-patching), which includes tissue segmentation, patching, and feature extraction using ResNet.
- **CAMELYON17**
  - We use `CLAM/presets/bwh_biopsy.csv` as configuration
  - Extracted features are saved as `.h5` files
  - Saved under the folder: `./h5_files/`

- **TCGA (BRCA & NSCLC)**
  - We use `CLAM/presets/tcga.csv` as configuration
  - Extracted features are saved as `.pt` files
  - Saved under the folder: `./pt_files/`
    
  > If you want to reproduce feature extraction, please refer to CLAM’s [wsi_feature_extraction.py](https://github.com/mahmoodlab/CLAM/blob/master/create_patches_fp.py) and modify the config CSVs accordingly.
  > 
## Installation
- Python >= 3.10
- PyTorch >= 2.3.0 (with CUDA 11.8 support)
- Torchvision == 0.18.0+cu118

- pip install -r requirements.txt

## Project Structure
Download the T5 weights from HuggingFace:

> [FLAN-T5 Small (google/flan-t5-small)](https://huggingface.co/google/flan-t5-small/tree/main)

After downloading, place the following files into:
/used_checkpoint/T5/

The whole structure are followed:
- `data_splits_stage_folds/`: CAMELYON17 slide & patient-level CSVs
- `data_splits_tcga_brca_folds/`: TCGA-BRCA subtype classification folds
- `data_splits_tcga_nsclc_folds/`: TCGA-NSCLC subtype classification folds
- `text/`: GPT-generated textual descriptions per class
- `used_checkpoint/`: Pretrained weights and downloaded T5 model files
- `h5_files/`: Extracted HDF5 features from WSIs
- `pt_files/`: Saved `.pt` model checkpoints

**Core Scripts**
- `dataloader.py`: Data loading and transformation
- `main.py`: Entry point for training and evaluation
- `model.py`: HMF model architecture with Bi-Cross Attention
- `train.py`: Training loop, scheduler, optimizer
- `test.py`: Test script for model evaluation
- `utils.py`: Utility functions

## Usage
### CAMELYON17
If you're using **Windows (single GPU)**:
```bash
python main.py --task_target 'camelyon' --h5_path 'your_h5_directory'
```

If you're using multi-GPU setup (e.g., 2 GPUs):
```bash
torchrun --nproc_per_node=2 main.py --task_target 'camelyon' --h5_path 'your_h5_directory'
```

### TCGA Subtyping(BRCA/NSCLC) Windows for example
BRCA:
```bash
python main.py --task_target 'tcga' --subtyping_task 'brca' --h5_path 'your_h5_directory'
```
NSCLC:
```bash
python main.py --task_target 'tcga' --subtyping_task 'nsclc' --h5_path 'your_h5_directory'
```
### Evaluate/Test
Evaluate a trained model using saved checkpoint:
```bash
python test.py --checkpoint used_checkpoint/model_epochX.pth
```
The extra Arguments:
### ⚙️ Arguments

| Argument               | Description                                       | Example                                  |
|------------------------|---------------------------------------------------|------------------------------------------|
| `--task_target`        | Task type: `camelyon` or `tcga`                   | `--task_target camelyon`                 |
| `--subtyping_task`     | (For TCGA only) Subtype: `brca` or `nsclc`        | `--subtyping_task brca`                  |
| `--fold`               | Fold index for cross-validation (1–5)             | `--fold 1`                                |
| `--ratio`              | Attention filter ratio (range: 0.1 to 0.9)        | `--ratio 0.5`                             |
| `--h5_path`            | Path to `.h5` feature files                       | `--h5_path ./h5_files/`                   |
| `--pt_path`            | Path to `.pt` feature files                       | `--pt_path ./pt_files/`                   |
| `--accumulation_steps` | Gradient accumulation steps to reduce GPU usage   | `--accumulation_steps 5`                 |


