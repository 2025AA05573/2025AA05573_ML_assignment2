import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.title("Cardiovascular Disease Prediction")

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

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    model = joblib.load(f"model/{model_name.replace(' ','_')}.pkl")

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    y_pred = model.predict(X)

    st.text("Classification Report")
    st.text(classification_report(y,y_pred))

    cm = confusion_matrix(y,y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
