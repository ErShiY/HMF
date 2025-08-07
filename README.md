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
