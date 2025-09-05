from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")

# Define input schema
class Transaction(BaseModel):
    amount: float
    time: float
    location: float
    device: float
    # ➝ Add other features as per dataset

# Initialize app
app = FastAPI(title="Fraud Detection API")

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Convert input to numpy array
    data = np.array([[transaction.amount, transaction.time,
                      transaction.location, transaction.device]])
    
    # Make prediction
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]  # fraud probability

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(proba)
    }

    

    