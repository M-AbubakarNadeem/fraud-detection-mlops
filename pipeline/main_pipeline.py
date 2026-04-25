import mlflow
import subprocess
import time
import os

# Set tracking URI to local docker container
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def run_step(step_name, script_path, max_retries=3):
    """Run a pipeline step as an MLFlow run with retries."""
    print(f"\n{'='*50}\nStarting Step: {step_name}\n{'='*50}")
    
    for attempt in range(1, max_retries + 1):
        try:
            with mlflow.start_run(nested=True, run_name=step_name):
                import sys
                # Run the script
                result = subprocess.run(
                    [sys.executable, script_path], 
                    check=True, 
                    capture_output=True, 
                    encoding='utf-8'
                )
                print(f"Step {step_name} completed successfully.")
                print(result.stdout)
                mlflow.log_param(f"{step_name}_status", "SUCCESS")
                return True
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt} failed for {step_name}.")
            print(f"Error: {e.stderr}")
            if attempt == max_retries:
                print(f"Step {step_name} failed after {max_retries} attempts.")
                return False
            time.sleep(2 ** attempt) # Exponential backoff

def conditional_deployment(threshold=0.75):
    """Step 7: Conditional Deployment Logic"""
    print(f"\n{'='*50}\nStarting Step: Conditional Deployment\n{'='*50}")
    with mlflow.start_run(nested=True, run_name="Conditional Deployment"):
        # Fetch the parent run ID to query metrics
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Fraud_Detection_Pipeline")
        
        if not experiment:
            print("Experiment not found. Cannot deploy.")
            return

        # Find the latest Model Evaluation run to get the recall metric
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'Model Evaluation'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            print("No evaluation metrics found. Aborting deployment.")
            return
            
        latest_eval_run = runs[0]
        # For this example, we assume model_evaluation logged 'test_recall'
        recall = latest_eval_run.data.metrics.get("test_recall", 0)
        
        print(f"Evaluated Model Recall: {recall:.4f} (Threshold: {threshold})")
        mlflow.log_metric("deployment_threshold", threshold)
        
        if recall >= threshold:
            print("✅ SUCCESS: Model meets performance criteria. Proceeding with deployment.")
            mlflow.log_param("deployment_decision", "DEPLOYED")
            # Logic to tag model as Production in MLFlow Model Registry would go here
        else:
            print("❌ FAILED: Model performance below threshold. Deployment aborted.")
            mlflow.log_param("deployment_decision", "ABORTED")


def run_pipeline():
    # Create or get experiment
    mlflow.set_experiment("Fraud_Detection_Pipeline")
    
    steps = [
        ("Data Ingestion", "src/data_ingestion.py"),
        ("Data Validation", "src/data_validation.py"),
        ("Data Preprocessing", "src/data_preprocessing.py"),
        ("Feature Engineering", "src/feature_engineering.py"),
        ("Model Training", "src/model_training.py"),
        ("Model Evaluation", "src/model_evaluation.py")
    ]
    
    with mlflow.start_run(run_name="Full Pipeline Run"):
        for step_name, script_path in steps:
            # We skip execution if the file doesn't exist yet (for incremental building)
            if not os.path.exists(script_path):
                print(f"Skipping {step_name}: Script {script_path} not found.")
                continue
                
            success = run_step(step_name, script_path)
            if not success:
                print("Pipeline failed. Halting execution.")
                return
        
        # Run Step 7
        conditional_deployment()

if __name__ == "__main__":
    run_pipeline()
