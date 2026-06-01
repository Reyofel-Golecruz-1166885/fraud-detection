from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Load model and scaler once when API starts
model = joblib.load('../models/fraud_model_tuned.joblib')
scaler = joblib.load('../models/scaler.joblib')

app = FastAPI(
    title="Fraud Detection API",
    description="Detects fraudulent credit card transactions using XGBoost",
    version="1.0.0"
)

# Define what the input JSON should look like
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

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert to dataframe
    data = pd.DataFrame([transaction.dict()])

    # Scale Amount and Time
    data[['Amount', 'Time']] = scaler.transform(data[['Amount', 'Time']])

    # Reorder columns to match training data
    feature_cols = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
                    'V10','V11','V12','V13','V14','V15','V16','V17','V18',
                    'V19','V20','V21','V22','V23','V24','V25','V26','V27',
                    'V28','Amount']
    data = data[feature_cols]

    # Predict
    probability = model.predict_proba(data)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "prediction": "FRAUD" if prediction == 1 else "LEGIT",
        "fraud_probability": round(float(probability), 4),
        "confidence": "HIGH" if probability > 0.8 or probability < 0.2 else "MEDIUM"
    }