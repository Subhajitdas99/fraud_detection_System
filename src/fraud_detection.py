import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
classification_report,
precision_score,
recall_score,
f1_score,
roc_auc_score,
precision_recall_curve,
auc,
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================
# MLflow Setup
#==========================

mlflow.set_experiment("Fraud Detection XGBoost")

# ==========================
# Load Dataset
# ==========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

FEATURE_COLUMNS = X.columns.tolist()

print(f"Dataset Shape: {df.shape}")
print(f"Fraud Transactions: {y.sum()}")

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
stratify=y,
random_state=42,
)

# ==========================
# Handle Imbalance
# ==========================

print("Applying SMOTE...")

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
X_train,
y_train,
)

# ==========================
# Build Pipeline
# ==========================

pipeline = Pipeline(
[
("scaler", StandardScaler()),
(
"model",
XGBClassifier(
n_estimators=200,
max_depth=5,
learning_rate=0.1,
eval_metric="logloss",
random_state=42,
),
),
]
)

# ==========================
# MLflow Run
# ==========================

with mlflow.start_run():

    # --------------------------
    # Log Parameters
    # --------------------------

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("threshold", 0.5)
    mlflow.log_param("smote", True)

    # --------------------------
    # Train Model
    # --------------------------

    print("Training model...")

    pipeline.fit(
        X_train_resampled,
        y_train_resampled,
    )

    # --------------------------
    # Evaluation
    # --------------------------

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test,
    y_prob,
)

pr_auc = auc(pr_recall, pr_precision)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

# --------------------------
# Log Metrics
# --------------------------

mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)
mlflow.log_metric("pr_auc", pr_auc)

# --------------------------
# Save PR Curve
# --------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    pr_recall,
    pr_precision,
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

pr_curve_path = OUTPUT_DIR / "pr_curve.png"

plt.savefig(
    pr_curve_path,
    bbox_inches="tight",
)

plt.close()

mlflow.log_artifact(
    str(pr_curve_path)
)

# --------------------------
# Save Model Bundle
# --------------------------

model_path = MODEL_DIR / "fraud_model.pkl"

joblib.dump(
    {
        "model": pipeline,
        "features": FEATURE_COLUMNS,
        "threshold": threshold,
    },
    model_path,
)

# Log model artifact
mlflow.log_artifact(str(model_path))

print("\nModel saved successfully.")

print(
    f"MLflow Run ID: "
    f"{mlflow.active_run().info.run_id}"
)