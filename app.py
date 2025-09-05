from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
from fastapi.responses import JSONResponse

# Load trained model
model = joblib.load("fraud_model.pkl")

# -------------------------
# 1. Define transaction schema
# -------------------------
class Transaction(BaseModel):
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
    Time: float
    threshold: Optional[float] = Field(0.5, description="Fraud probability threshold, default=0.5")

# -------------------------
# 2. Initialize FastAPI app
# -------------------------
app = FastAPI(title="Fraud Detection API (Full Features)")

# -------------------------
# 3. /predict endpoint
# -------------------------
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Convert input to array in the correct order
        features = [
            transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
            transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
            transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
            transaction.V26, transaction.V27, transaction.V28, transaction.Amount, transaction.Time
        ]

        data = np.array([features])

        # Predict probability
        proba = model.predict_proba(data)[0][1]

        # Apply threshold
        prediction = int(proba >= transaction.threshold)

        return {
            "fraud_prediction": prediction,
            "fraud_probability": float(proba),
            "threshold_used": transaction.threshold
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
