from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

app = FastAPI(title="Fraud Detection API", description="API for predicting fraudulent transactions")

# Custom Prometheus Metrics
FRAUD_PREDICTIONS = Counter("fraud_predictions_total", "Total count of fraud predictions", ["prediction_type"])
PREDICTION_CONFIDENCE = Histogram("prediction_confidence", "Histogram of prediction confidence scores", buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Initialize Prometheus Instrumentator
Instrumentator().instrument(app).expose(app)

# Load the best model (Cost-Sensitive XGBoost)
try:
    with open('models/xgb_cost.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class Transaction(BaseModel):
    # Defining a few key features for the payload based on our synthetic data
    TransactionAmt: float
    ProductCD: int
    card1: float
    card2: float
    addr1: float
    P_emaildomain: int
    DeviceType: int
    DeviceInfo: int
    # Assuming the other features are handled or defaulted
    
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "xgboost_cost_sensitive"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Convert input to DataFrame
        feature_dict = transaction.dict()
        
        full_features = {
            'TransactionAmt': feature_dict['TransactionAmt'],
            'ProductCD': feature_dict['ProductCD'],
            'card1': feature_dict['card1'],
            'card2': feature_dict['card2'],
            'addr1': feature_dict['addr1'],
            'P_emaildomain': feature_dict['P_emaildomain'],
            'id_01': 0.0,
            'id_02': 0.0,
            'DeviceType': feature_dict['DeviceType'],
            'DeviceInfo': feature_dict['DeviceInfo'],
            'Amt_to_card1_ratio': feature_dict['TransactionAmt'] / (feature_dict['card1'] + 1e-5),
            'card1_card2_interact': feature_dict['card1'] * feature_dict['card2']
        }
        
        df = pd.DataFrame([full_features])
        
        # Predict probability
        proba = float(model.predict_proba(df)[0][1])
        prediction = int(proba > 0.5)
        
        # Record Metrics
        PREDICTION_CONFIDENCE.observe(proba)
        type_str = "fraud" if prediction == 1 else "legit"
        FRAUD_PREDICTIONS.labels(prediction_type=type_str).inc()
        
        return {
            "fraud_probability": proba,
            "prediction": prediction,
            "risk_level": "High" if proba > 0.75 else "Medium" if proba > 0.4 else "Low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
