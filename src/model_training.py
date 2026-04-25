import pandas as pd
import numpy as np
import os
import mlflow
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import pickle

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def train_models():
    print("Running Model Training (Tasks 3 & 4)...")
    
    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    
    X_train_smote = pd.read_csv('data/processed/X_train_smote.csv')
    y_train_smote = pd.read_csv('data/processed/y_train_smote.csv').values.ravel()
    
    # We will save trained models
    os.makedirs('models', exist_ok=True)
    
    # 1. XGBoost (Standard vs Cost-Sensitive)
    print("Training XGBoost (Standard)")
    xgb_std = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_std.fit(X_train_smote, y_train_smote)  # Using SMOTE data
    
    print("Training XGBoost (Cost-Sensitive)")
    # cost_sensitive: scale_pos_weight = sum(negative instances) / sum(positive instances)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    spw = neg_count / pos_count if pos_count > 0 else 1
    
    xgb_cost = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=spw, random_state=42)
    xgb_cost.fit(X_train, y_train) # Applied to standard data for comparison
    
    # 2. LightGBM
    print("Training LightGBM")
    lgb_model = lgb.LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42)
    lgb_model.fit(X_train, y_train)
    
    # 3. Hybrid Model (RF + Feature Selection)
    print("Training Hybrid Model (RF + Feature Selection)")
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
    hybrid_pipeline = Pipeline([
        ('feature_selection', SelectFromModel(rf_selector, max_features=10)),
        ('classifier', xgb.XGBClassifier(n_estimators=50, random_state=42))
    ])
    hybrid_pipeline.fit(X_train_smote, y_train_smote)
    
    # Save models locally for evaluation script
    with open('models/xgb_std.pkl', 'wb') as f: pickle.dump(xgb_std, f)
    with open('models/xgb_cost.pkl', 'wb') as f: pickle.dump(xgb_cost, f)
    with open('models/lgb.pkl', 'wb') as f: pickle.dump(lgb_model, f)
    with open('models/hybrid.pkl', 'wb') as f: pickle.dump(hybrid_pipeline, f)
    
    # Log everything to MLFlow as artifacts (bypasses Model Registry 404 errors)
    mlflow.log_artifacts("models", artifact_path="trained_models")
    
    mlflow.log_param("xgb_scale_pos_weight", spw)
    print("Model training completed.")

if __name__ == "__main__":
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    if mlflow.active_run():
        train_models()
    else:
        with mlflow.start_run(run_name="Model Training (Manual)"):
            train_models()
