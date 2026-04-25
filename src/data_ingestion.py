import pandas as pd
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def ingest_data():
    print("Ingesting data...")
    # Read the data
    try:
        transaction = pd.read_csv('data/train_transaction.csv')
        identity = pd.read_csv('data/train_identity.csv')
    except Exception as e:
        print(f"Error reading data: {e}")
        raise
    
    # Merge the datasets on TransactionID
    df = pd.merge(transaction, identity, on='TransactionID', how='left')
    
    # Save the merged dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/merged_data.csv', index=False)
    
    # Log metrics
    mlflow.log_metric("num_transactions", len(transaction))
    mlflow.log_metric("num_identities", len(identity))
    mlflow.log_metric("merged_rows", len(df))
    mlflow.log_metric("merged_cols", len(df.columns))
    print(f"Merged Data Shape: {df.shape}")

if __name__ == "__main__":
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    # For nested runs managed by main_pipeline
    if mlflow.active_run():
        ingest_data()
    else:
        with mlflow.start_run(run_name="Data Ingestion (Manual)"):
            ingest_data()
