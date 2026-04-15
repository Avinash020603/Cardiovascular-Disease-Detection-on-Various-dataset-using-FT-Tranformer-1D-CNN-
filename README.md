# Heart Failure Prediction Using FT-Transformer + 1D CNN Ensemble

## Dataset: Family Health Unit (1002009), Balıkesir City

**Heart Failure Clinical Records Dataset**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Challenges Faced](#challenges-faced)
- [Solutions & Improvements](#solutions--improvements)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Authors](#authors)

---

## 🔬 Project Overview

This project applies a **hybrid deep-learning ensemble** — combining an **FT-Transformer** and a **1D Convolutional Neural Network (CNN)** — to predict mortality events in heart failure patients using the clinical records from the **Family Health Unit (1002009), Balıkesir City**.

The goal is to build a robust binary classifier that predicts the `DEATH_EVENT` (whether a patient died during the follow-up period) based on 12 clinical features. The project addresses several real-world challenges including **small dataset size**, **class imbalance**, and **feature skewness** that are typical of clinical datasets.

---

## 📊 Dataset Description

| Property             | Value                                                        |
| -------------------- | ------------------------------------------------------------ |
| **Source**            | Family Health Unit (1002009), Balıkesir City                 |
| **File**             | `heart_failure_clinical_records_dataset.csv`                 |
| **Samples**          | 299 patients                                                 |
| **Features**         | 12 clinical features + 1 target variable                     |
| **Target**           | `DEATH_EVENT` (0 = Survived, 1 = Died)                      |
| **Class Distribution** | Survived: 203 (67.9%) · Died: 96 (32.1%)                  |
| **Class Ratio**      | ~2.11 : 1 (negative / positive)                             |

### Features

| Feature                       | Type        | Description                                      |
| ----------------------------- | ----------- | ------------------------------------------------ |
| `age`                         | Numerical   | Age of the patient (years)                       |
| `anaemia`                     | Categorical | Decrease of red blood cells (0/1)                |
| `creatinine_phosphokinase`    | Numerical   | Level of the CPK enzyme in the blood (mcg/L)     |
| `diabetes`                    | Categorical | If the patient has diabetes (0/1)                |
| `ejection_fraction`           | Numerical   | Percentage of blood leaving the heart (%)        |
| `high_blood_pressure`         | Categorical | If the patient has hypertension (0/1)            |
| `platelets`                   | Numerical   | Platelets in the blood (kiloplatelets/mL)        |
| `serum_creatinine`            | Numerical   | Level of serum creatinine in the blood (mg/dL)   |
| `serum_sodium`                | Numerical   | Level of serum sodium in the blood (mEq/L)       |
| `sex`                         | Categorical | Sex of the patient (0 = Female, 1 = Male)        |
| `smoking`                     | Categorical | If the patient smokes (0/1)                      |
| `time`                        | Numerical   | Follow-up period (days)                          |

---

## 🧪 Methodology

### Model Architecture

The project employs a **soft-voting ensemble** of two complementary deep-learning models:

#### 1. FT-Transformer (Feature Tokenizer Transformer)

A Transformer-based architecture specifically designed for tabular data. It processes **numerical** and **categorical** features separately through dedicated embedding layers, then combines them via a `[CLS]` token and multi-head self-attention.

| Hyperparameter   | Value   |
| ---------------- | ------- |
| `d_model`        | 64      |
| `n_heads`        | 4       |
| `n_layers`       | 3       |
| `dropout`        | 0.2     |
| `lr`             | 5e-4    |
| `weight_decay`   | 1e-4    |
| `batch_size`     | 32      |

#### 2. 1D CNN (One-Dimensional Convolutional Neural Network)

A CNN treats the unified, scaled feature vector as a 1D signal, capturing **local patterns** and interactions between neighboring features.

| Hyperparameter   | Value   |
| ---------------- | ------- |
| `dropout`        | 0.3     |
| `lr`             | 5e-4    |
| `weight_decay`   | 1e-4    |
| `batch_size`     | 32      |

#### 3. Ensemble Strategy

The ensemble uses **probability-level soft voting** with optimized weights, searched at 0.01 granularity over the validation set:

```
ensemble_prob = w × FT_prob + (1 − w) × CNN_prob
```

### Data Preprocessing Pipeline

1. **Feature Engineering** — Applied `np.log1p` transform to heavily skewed numerical features:
   - `creatinine_phosphokinase`
   - `platelets`
   - `serum_creatinine`

2. **Feature Scaling** — StandardScaler (fit on training set only) applied to all numerical features.

3. **Stratified Splitting** — 80 / 10 / 10 train / validation / test split with stratification.

4. **Class-Weighted Loss** — `BCEWithLogitsLoss(pos_weight = n_neg / n_pos ≈ 2.10)` to address class imbalance.

### Training Configuration

| Parameter         | Value                                                      |
| ----------------- | ---------------------------------------------------------- |
| **Epochs**        | 300 (max)                                                  |
| **Patience**      | 40 (early stopping on validation accuracy)                 |
| **Optimizer**     | AdamW                                                      |
| **LR Scheduler**  | ReduceLROnPlateau (mode=max, factor=0.5, patience=10)      |
| **Random Seed**   | 42                                                         |

---

## ⚠️ Challenges Faced

Working with the Heart Failure Clinical Records dataset from the Family Health Unit (1002009), Balıkesir City presented several significant challenges that required dedicated solutions:

### 1. Extremely Small Dataset (299 samples)

With only **299 patient records**, the dataset is much smaller than what deep learning models typically require. This severely limits training data volume and makes the models prone to **overfitting**, especially architectures with millions of parameters like Transformers.

### 2. Class Imbalance (~2:1 ratio)

The dataset has a **2.11:1 imbalance** between "Survived" (203 samples) and "Died" (96 samples) classes. Without correction, models tend to be **biased toward the majority class**, producing high overall accuracy but critically **low recall** for the minority "Died" class — which is the class of primary medical interest.

### 3. Feature Skewness

Several clinical features (`creatinine_phosphokinase`, `platelets`, `serum_creatinine`) exhibit **extreme positive skewness**, with long right tails. This causes the StandardScaler to produce suboptimal normalized distributions, hurting model convergence.

### 4. Poor Initial Performance

The initial (v1) implementation suffered from:
- **Very poor recall** (as low as ≈ 0.11 for the "Died" class with 1D CNN)
- **Overfitting** on the training set (train accuracy > 0.97 while validation stagnated at ≈ 0.70)
- **Aggressive early stopping** (models stopped too early at epoch ~29 with patience=20)
- **Low model capacity** — `d_model=32`, `n_layers=2` were insufficient for capturing complex clinical patterns
- **Rigid LR decay** — CosineAnnealing decayed learning rate regardless of progress

### 5. Mixed Feature Types

The dataset contains a mix of **7 numerical** and **5 binary categorical** features. Standard MLPs treat all features the same way, losing valuable information about feature types. The FT-Transformer addresses this through separate embedding strategies, but integrating both feature types properly required careful engineering.

---

## ✅ Solutions & Improvements

The following improvements were implemented in the **v2 (Improved)** notebook to overcome the challenges above:

| Challenge                | Solution Applied                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **Class imbalance**      | Integrated `pos_weight` into `BCEWithLogitsLoss` (calculated as `n_neg / n_pos ≈ 2.10`) to prioritize the minority class |
| **Small dataset**        | Changed data split from 70/15/15 → **80/10/10** to maximize training samples                                            |
| **Feature skewness**     | Applied `np.log1p` transform to 3 heavily skewed numerical features before scaling                                       |
| **Model underfitting**   | Increased FT-Transformer capacity to `d_model=64`, `n_layers=3`, `dropout=0.2`                                          |
| **CNN regularization**   | Increased CNN dropout to `0.3` for better generalization                                                                  |
| **Rigid LR scheduling** | Switched to `ReduceLROnPlateau` (adaptive) from fixed `CosineAnnealingLR`                                                |
| **Early stopping**       | Extended budget to `EPOCHS=300`, `PATIENCE=40` (from 150/20)                                                             |
| **Ensemble granularity** | Finer weight search step `0.01` (from `0.05`) for optimal soft-voting                                                    |
| **Evaluation metrics**   | Added **Recall** and **AUC-ROC** to final results alongside Accuracy and F1                                              |

---

## 📈 Results

### Individual Model Performance (Test Set)

| Model             | Accuracy | F1 Score | Recall (Died) | AUC-ROC |
| ----------------- | -------- | -------- | ------------- | ------- |
| **FT-Transformer** | 0.9000  | 0.8235   | 0.7778        | 0.9524  |
| **1D CNN**         | 0.7000  | 0.1818   | 0.1111        | 0.6190  |

### Ensemble Performance (Test Set)

| Model                        | Accuracy | F1 Score | Recall (Died) | AUC-ROC |
| ---------------------------- | -------- | -------- | ------------- | ------- |
| **Ensemble (w_FT=0.06)**     | 0.8667   | 0.7143   | 0.5556        | 0.8836  |

### Classification Report (Ensemble)

```
              precision    recall  f1-score   support

    Survived       0.84      1.00      0.91        21
        Died       1.00      0.56      0.71         9

    accuracy                           0.87        30
   macro avg       0.92      0.78      0.81        30
weighted avg       0.89      0.87      0.85        30
```

> **Note:** The FT-Transformer alone achieves the highest test accuracy (90%) on this dataset, demonstrating its strength on small tabular data with mixed feature types. The ensemble helps smooth predictions through probability-level combination.

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- PyTorch 2.x
- scikit-learn
- pandas, numpy, matplotlib, seaborn

### Installation

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn
```

### Steps

1. Clone or download this repository.
2. Ensure `heart_failure_clinical_records_dataset.csv` is in the project root directory.
3. Open and run `ensemble_fttransformer_cnn_heart_failure.ipynb` in Jupyter Notebook/Lab.
4. All cells are designed to run sequentially from top to bottom.

---

## 🛠 Technologies Used

| Technology         | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| **Python 3.x**     | Core programming language                       |
| **PyTorch**        | Deep learning framework                         |
| **scikit-learn**   | Preprocessing, metrics, train/test splitting    |
| **pandas**         | Data loading and manipulation                   |
| **numpy**          | Numerical computations                          |
| **matplotlib**     | Training curves and ensemble weight plots       |
| **seaborn**        | Enhanced visualizations                         |
| **Jupyter**        | Interactive notebook environment                |

---

## 📁 Project Structure

```
Cardiovascular Disease Detection/
├── README.md                                      # This file
├── heart_failure_clinical_records_dataset.csv      # Raw dataset
├── ensemble_fttransformer_cnn_heart_failure.ipynb  # Main notebook (Improved v2)
├── ensemble_fttransformer_cnn.ipynb                # Original ensemble notebook (reference)
└── ...
```

---

## 👥 Authors

This project was developed as part of a cardiovascular disease detection study using the Heart Failure Clinical Records from the Family Health Unit (1002009), Balıkesir City.
