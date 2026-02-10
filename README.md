# 2025AA05573_ML_assignment2

# Cardiovascular Disease Prediction

## Problem Statement

Predict cardiovascular disease using machine learning classification models.

---

## Dataset Description

The dataset contains 557 records and 13 features including:

- age
- gender
- height
- weight
- blood pressure
- cholesterol
- glucose
- smoking
- alcohol
- activity

Target variable:

cardio

Binary classification:

0 = No disease  
1 = Disease

---

## Models Used

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

---

## Evaluation Metrics

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC Score

---

## Observations

Logistic Regression  
Good baseline performance.

Decision Tree  
Interpretable but may overfit.

KNN  
Sensitive to scaling and distance.

Naive Bayes  
Fast but assumes feature independence.

Random Forest  
High accuracy and stable.

XGBoost  
Best performance due to boosting.

---

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
