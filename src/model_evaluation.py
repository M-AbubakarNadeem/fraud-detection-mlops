import pandas as pd
import numpy as np
import os
import mlflow
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def evaluate_models():
    print("Running Model Evaluation...")
    
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    models = {
        'xgb_std': 'models/xgb_std.pkl',
        'xgb_cost': 'models/xgb_cost.pkl',
        'lightgbm': 'models/lgb.pkl',
        'hybrid': 'models/hybrid.pkl'
    }
    
    os.makedirs('plots', exist_ok=True)
    best_recall = 0
    
    for name, path in models.items():
        if not os.path.exists(path):
            continue
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        # Log metrics with prefix
        mlflow.log_metric(f"{name}_precision", precision)
        mlflow.log_metric(f"{name}_recall", recall)
        mlflow.log_metric(f"{name}_f1", f1)
        mlflow.log_metric(f"{name}_auc_roc", auc)
        
        # We need a generic "test_recall" for conditional deployment (usually best model)
        if name == 'xgb_cost':  # Cost-sensitive usually has best recall
            best_recall = recall
            mlflow.log_metric("test_recall", best_recall)
            
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plot_path = f'plots/cm_{name}.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Log plot to MLFlow
        mlflow.log_artifact(plot_path, "confusion_matrices")
        
        print(f"[{name}] Recall: {recall:.4f} | AUC: {auc:.4f}")

    print("Model evaluation completed.")

if __name__ == "__main__":
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    if mlflow.active_run():
        evaluate_models()
    else:
        with mlflow.start_run(run_name="Model Evaluation (Manual)"):
            evaluate_models()
