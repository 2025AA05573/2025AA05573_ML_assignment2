import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# load dataset
df = pd.read_csv("cardio_train.csv")

df.drop("id", axis=1, inplace=True)

X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {

"Logistic Regression":
Pipeline([
("scaler", StandardScaler()),
("model", LogisticRegression(max_iter=500))
]),

"Decision Tree":
DecisionTreeClassifier(max_depth=5),

"KNN":
Pipeline([
("scaler", StandardScaler()),
("model", KNeighborsClassifier())
]),

"Naive Bayes":
GaussianNB(),

"Random Forest":
RandomForestClassifier(n_estimators=100),

"XGBoost":
XGBClassifier(eval_metric="logloss")

}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results.append({

        "Model": name,
        "Accuracy": accuracy_score(y_test,y_pred),
        "AUC": roc_auc_score(y_test,y_prob),
        "Precision": precision_score(y_test,y_pred),
        "Recall": recall_score(y_test,y_pred),
        "F1": f1_score(y_test,y_pred),
        "MCC": matthews_corrcoef(y_test,y_pred)

    })

    joblib.dump(model, f"model/{name.replace(' ','_')}.pkl")

pd.DataFrame(results).to_csv("metrics.csv", index=False)

print("Training complete")
