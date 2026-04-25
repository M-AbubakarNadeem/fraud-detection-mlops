import pandas as pd
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def run_feature_eng():
    print("Running Feature Engineering...")
    
    datasets = ['X_train', 'X_test', 'X_train_smote', 'X_train_rus']
    
    for ds in datasets:
        file_path = f'data/processed/{ds}.csv'
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        
        # Example Feature Engineering for IEEE CIS Fraud data
        if 'TransactionAmt' in df.columns and 'card1' in df.columns:
            # Transaction amount to card1 ratio (proxy for unusual spending on a card)
            # Add a small epsilon to avoid division by zero
            df['Amt_to_card1_ratio'] = df['TransactionAmt'] / (df['card1'] + 1e-5)
            
        if 'card1' in df.columns and 'card2' in df.columns:
            # Interaction term between card1 and card2
            df['card1_card2_interact'] = df['card1'] * df['card2']
            
        # Log the number of new features only once
        if ds == 'X_train':
            mlflow.log_param("num_new_features_engineered", 2)
            mlflow.log_metric("final_feature_count", len(df.columns))
            
        df.to_csv(file_path, index=False)
        
    print("Feature engineering completed.")

if __name__ == "__main__":
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    if mlflow.active_run():
        run_feature_eng()
    else:
        with mlflow.start_run(run_name="Feature Engineering (Manual)"):
            run_feature_eng()
