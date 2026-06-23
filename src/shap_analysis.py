import joblib
import pandas as pd
import shap

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["model"]

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])

# Sample for speed
X_sample = X.sample(1000, random_state=42)

model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

import matplotlib.pyplot as plt

plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    bbox_inches="tight"
)

plt.close()

print("Saved shap_summary.png")