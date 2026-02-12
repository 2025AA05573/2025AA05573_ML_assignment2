
import sys
import subprocess

# This will print to the Streamlit Cloud logs
print("--- DEBUGGING INSTALLED PACKAGES ---")
try:
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8')
    print(installed_packages)
except Exception as e:
    print(f"Could not list packages: {e}")
print("-------------------------------------")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Cardio Prediction App", layout="centered")

st.title("Cardiovascular Disease Prediction App")

# Model selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "KNN": "k-nearest_neighbors_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}



# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    model = joblib.load(f"model/{model_files[model_name]}")

    if "cardio" in df.columns:

        X = df.drop("cardio", axis=1)
        y = df["cardio"]

        y_pred = model.predict(X)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.imshow(np.zeros_like(cm), vmin=0, vmax=1)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=28,
                    fontweight="bold",
                    color="black"
                )

        ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=14, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Disease", "Disease"])
        ax.set_yticklabels(["No Disease", "Disease"])

        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.write("Uploaded file does not contain 'cardio' column.")
