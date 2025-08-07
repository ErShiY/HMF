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
