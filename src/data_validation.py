import pandas as pd
import numpy as np
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def run_validation():
    print("Running Data Validation...")
    
    # Load ingested data
    df = pd.read_csv('data/processed/merged_data.csv')
    
    # Check for target variable
    if 'isFraud' not in df.columns:
        raise ValueError("Target variable 'isFraud' is missing from the dataset.")
        
    # Calculate missing values
    missing_pct = df.isnull().mean() * 100
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    
    # Log metrics to MLFlow
    mlflow.log_metric("total_columns", len(df.columns))
    mlflow.log_metric("cols_with_missing_gt_50pct", len(high_missing_cols))
    
    # Calculate cardinality of categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    high_cardinality = [col for col in cat_cols if df[col].nunique() > 100]
    
    mlflow.log_metric("high_cardinality_cols", len(high_cardinality))
    mlflow.log_param("high_cardinality_features", ",".join(high_cardinality))
    
    print(f"Validation successful. Found {len(high_missing_cols)} columns with >50% missing data.")
    print(f"Found {len(high_cardinality)} high cardinality categorical features.")
    
    # We could drop high missing cols here, but we'll let preprocessing handle it
    mlflow.log_param("validation_status", "SUCCESS")

if __name__ == "__main__":
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    if mlflow.active_run():
        run_validation()
    else:
        with mlflow.start_run(run_name="Data Validation (Manual)"):
            run_validation()
