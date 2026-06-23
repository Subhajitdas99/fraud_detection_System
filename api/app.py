from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fraud_model.pkl"
)

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURE_COLUMNS = bundle["features"]


# ==========================
# API
# ==========================

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float

    threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )


@app.post("/predict")
def predict(transaction: Transaction):

    payload = transaction.dict()

    threshold = payload.pop("threshold")

    df = pd.DataFrame([payload])

    df = df[FEATURE_COLUMNS]

    probability = float(
        model.predict_proba(df)[0][1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "fraud_probability": probability,
        "threshold_used": threshold,
    }