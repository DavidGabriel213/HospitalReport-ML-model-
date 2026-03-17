# 🏥 Nigerian Patient Disease Risk Prediction

A complete end-to-end Machine Learning project predicting cardiovascular 
disease risk in Nigerian patients.

---

## 📋 Project Overview

This project applies a full ML pipeline to 825 Nigerian patient records 
across 16 columns. Every column had serious quality issues. After thorough 
cleaning and feature engineering, three ML models were trained, compared 
and the best model fine-tuned using GridSearchCV with 5-fold cross validation.

---

## 🛠️ Tools & Libraries

- **Python**
- **Pandas** — data cleaning & analysis
- **NumPy** — numerical operations & IQR detection
- **Matplotlib** — data visualization
- **Scikit-learn** — ML models, evaluation & hyperparameter tuning

---

## 🧹 Data Cleaning Steps

| Column | Problem | Solution |
|---|---|---|
| Age | Outliers x10, negatives, nulls | np.abs(), IQR both bounds, median fill |
| Gender | 6 formats: M/F/male/Male/female/Female | Dictionary mapping |
| Systolic_BP | Some stored as '120/80' combined | String split on '/' — take left side |
| Diastolic_BP | Missing where BP combined, nulls | Fill from Systolic first, then split |
| BMI | '26.5kg/m2' strings, nulls | Strip 'kg/m2', median fill |
| Cholesterol | Outliers x10, nulls | IQR clipping, mean fill |
| BloodSugar | '95mg/dL' strings | Strip 'mg/dL' |
| Smoking/FamilyHistory | 1/0/yes/no/YES mixed | str.capitalize() + dictionary |
| Duplicates | 25 hidden duplicate rows | drop_duplicates() |

---

## 📐 Feature Engineering

| Feature | Formula | Clinical Meaning |
|---|---|---|
| PulsePressure | Systolic_BP - Diastolic_BP | Arterial wall stress |
| MAP | Diastolic + (PulsePressure / 3) | Mean blood flow pressure |
| BMI_Class | pd.cut(BMI, 5 bins) | Underweight → Severe Obesity |
| Age_Group | pd.cut(Age, 3 bins) | Young / Middle / Senior |

---

## 🤖 ML Models & Results

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **84.38%** 🏆 |
| Random Forest (tuned) | 79.37% |
| Random Forest (default) | 76.88% |
| Decision Tree | 70.00% |

> GridSearchCV tested **500 combinations** (100 params × 5 folds) to find optimal hyperparameters

---

## 🔍 Key Findings

- Logistic Regression outperformed Random Forest — simpler models win on well-structured data
- MAP and PulsePressure (engineered features) ranked highest in feature importance
- Tuning improved Random Forest accuracy by 2.5% automatically
- BP parsing technique handled 4 different blood pressure formats in real clinical data

---

## 📁 Files

| File | Description |
|---|---|
| `Code.py` | Full Python source code |
| `nigerian_health_messy.csv` | Raw messy dataset |
| `Health_Risk_Report.pdf` | Full analysis report |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python Code.py
Complete end-to-end ML project — cleaning, feature engineering,
model training, hyperparameter tuning and evaluation.
Part of my self-taught journey toward becoming an ML/AI Engineer.# HospitalReport-ML-model-
