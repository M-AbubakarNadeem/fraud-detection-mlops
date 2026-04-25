import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import category_encoders as ce

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def run_preprocessing():
    print("Running Data Preprocessing...")
    
    df = pd.read_csv('data/processed/merged_data.csv')
    
    # Separate features and target
    X = df.drop(columns=['isFraud', 'TransactionID'])
    y = df['isFraud']
    
    # Train test split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Advanced Missing Value Strategy:
    # Numeric -> Median
    # Categorical -> 'Unknown'
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    for col in num_cols:
        median_val = X_train[col].median()
        X_train.fillna({col: median_val}, inplace=True)
        X_test.fillna({col: median_val}, inplace=True)
        
    for col in cat_cols:
        X_train.fillna({col: 'Unknown'}, inplace=True)
        X_test.fillna({col: 'Unknown'}, inplace=True)
        
    # Feature Encoding: Target Encoding for Categoricals
    print("Applying Target Encoding to categoricals...")
    target_encoder = ce.TargetEncoder(cols=cat_cols.tolist())
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)
    
    mlflow.log_param("missing_value_strategy", "Median/Unknown")
    mlflow.log_param("encoding_strategy", "TargetEncoding")
    
    # Save the standard encoded sets
    os.makedirs('data/processed', exist_ok=True)
    X_train_encoded.to_csv('data/processed/X_train.csv', index=False)
    X_test_encoded.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Compare 2 Imbalance Handling Strategies (SMOTE vs Undersampling)
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train)
    
    print("Applying Random Undersampling...")
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_encoded, y_train)
    
    # Save resampled sets
    X_train_smote.to_csv('data/processed/X_train_smote.csv', index=False)
    y_train_smote.to_csv('data/processed/y_train_smote.csv', index=False)
    
    X_train_rus.to_csv('data/processed/X_train_rus.csv', index=False)
    y_train_rus.to_csv('data/processed/y_train_rus.csv', index=False)
    
    mlflow.log_metric("train_size_standard", len(y_train))
    mlflow.log_metric("train_size_smote", len(y_train_smote))
    mlflow.log_metric("train_size_rus", len(y_train_rus))
    
    print("Data preprocessing completed.")

if __name__ == "__main__":
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    if mlflow.active_run():
        run_preprocessing()
    else:
        with mlflow.start_run(run_name="Data Preprocessing (Manual)"):
            run_preprocessing()
