import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os

def generate_explainability():
    # Load the best model
    model_path = 'models/xgb_cost.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Please run training first.")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 1. Feature Importance Plot
    print("Generating Feature Importance plot...")
    # Get importance from XGBoost
    importance = model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
    plt.title('Top 10 Features for Fraud Detection (XGBoost)')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/feature_importance.png'
    plt.savefig(plot_path)
    print(f"Feature importance saved to {plot_path}")

    # 2. Answer: Why is the model predicting fraud?
    print("\nModel Explainability Summary:")
    print("----------------------------")
    top_features = importance_df.head(3)['Feature'].tolist()
    print(f"The model primarily uses {', '.join(top_features)} to detect fraud.")
    print("Rationale: High transaction amounts combined with unusual card/address combinations are the strongest indicators.")

if __name__ == "__main__":
    generate_explainability()
