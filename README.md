# Cardiovascular Disease Prediction using Machine Learning

---

# a. Problem Statement

Cardiovascular disease (CVD) is one of the leading causes of death globally. Early prediction of cardiovascular disease can help healthcare professionals provide timely treatment and reduce mortality risk.

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a patient has cardiovascular disease based on medical attributes such as age, blood pressure, cholesterol level, glucose level, smoking habits, alcohol consumption, and physical activity.

The project also includes deployment of an interactive Streamlit web application to allow users to upload data, select models, and view evaluation metrics and confusion matrix.

---

# b. Dataset Description

Dataset Name: Cardiovascular Disease Dataset

Source: Public dataset provided for Machine Learning Assignment

Dataset contains:

* Number of instances: 70,000
* Number of features: 12 input features
* Target variable: cardio

Target classes:

* 0 → No cardiovascular disease
* 1 → Cardiovascular disease present

Feature description:

| Feature     | Description              |
| ----------- | ------------------------ |
| age         | Age in days              |
| gender      | Gender                   |
| height      | Height (cm)              |
| weight      | Weight (kg)              |
| ap_hi       | Systolic blood pressure  |
| ap_lo       | Diastolic blood pressure |
| cholesterol | Cholesterol level        |
| gluc        | Glucose level            |
| smoke       | Smoking status           |
| alco        | Alcohol consumption      |
| active      | Physical activity        |
| cardio      | Target variable          |

This is a binary classification problem.

---

# c. Models Used and Comparison Table

The following machine learning models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

Evaluation metrics used:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

Metrics source:
[https://github.com/2025AA05573/2025AA05573_ML_assignment2/blob/main/metrics.csv](https://github.com/2025AA05573/2025AA05573_ML_assignment2/blob/main/metrics.csv)

---

## Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.7500   | 0.8378 | 0.7885    | 0.7069 | 0.7455 | 0.5042 |
| Decision Tree            | 0.6964   | 0.7337 | 0.8333    | 0.5172 | 0.6383 | 0.4345 |
| kNN                      | 0.6875   | 0.7979 | 0.7091    | 0.6724 | 0.6903 | 0.3759 |
| Naive Bayes              | 0.6696   | 0.7554 | 0.8182    | 0.4655 | 0.5934 | 0.3885 |
| Random Forest (Ensemble) | 0.7589   | 0.8263 | 0.7719    | 0.7586 | 0.7652 | 0.5176 |
| XGBoost (Ensemble)       | 0.6786   | 0.7698 | 0.6897    | 0.6897 | 0.6897 | 0.3563 |

---

# d. Observations on Model Performance

| ML Model Name            | Observation about model performance                                                                                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression provided strong performance with good balance between precision and recall. It achieved high AUC score (0.8378), indicating excellent ability to distinguish between classes. It works well because the dataset contains meaningful linear relationships. |
| Decision Tree            | Decision Tree achieved good precision but lower recall, meaning it missed some positive cases. It is easy to interpret but tends to overfit compared to ensemble models.                                                                                                      |
| kNN                      | kNN provided moderate performance with balanced precision and recall. It depends heavily on distance calculations and is sensitive to feature scaling.                                                                                                                        |
| Naive Bayes              | Naive Bayes achieved high precision but low recall, indicating it predicts positive cases conservatively. It is fast but assumes feature independence, which may reduce accuracy.                                                                                             |
| Random Forest (Ensemble) | Random Forest achieved the best overall performance with highest accuracy (0.7589), best F1 score (0.7652), and highest MCC (0.5176). Ensemble learning improves robustness and reduces overfitting.                                                                          |
| XGBoost (Ensemble)       | XGBoost achieved balanced precision and recall but slightly lower accuracy compared to Random Forest. It is a powerful boosting algorithm but may require hyperparameter tuning for optimal performance.                                                                      |

---

# Conclusion

Among all models, Random Forest performed the best overall with highest accuracy, F1 score, and MCC score. Logistic Regression also performed very well with the highest AUC score.

Ensemble models demonstrated superior performance due to their ability to combine multiple learners and capture complex feature relationships.

This project successfully demonstrates end-to-end machine learning workflow including model training, evaluation, comparison, and deployment using Streamlit.

---

# GitHub Repository

[https://github.com/2025AA05573/2025AA05573_ML_assignment2](https://github.com/2025AA05573/2025AA05573_ML_assignment2)

---

# Streamlit App Features

* Upload CSV dataset
* Select machine learning model
* View classification report
* View confusion matrix
* Interactive predictions
