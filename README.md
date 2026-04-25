# MLOps Fraud Detection System

An end-to-end MLOps pipeline for detecting fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset.

## 🚀 Features
- **Automated Pipeline**: 7-stage orchestration (Ingestion -> Deployment) using MLFlow.
- **Advanced Data Handling**: Target encoding, SMOTE, and Random Undersampling.
- **Multi-Model Complexity**: Comparative analysis of XGBoost, LightGBM, and Hybrid (RF+FS) models.
- **Cost-Sensitive Learning**: Business-impact optimization by penalizing False Negatives.
- **CI/CD**: Production-grade GitHub Actions for linting, building Docker images, and automated deployment.
- **Monitoring & Observability**: Real-time metrics with Prometheus and Grafana dashboards.
- **Explainability**: SHAP/Feature Importance analysis for model transparency.

## 📁 Project Structure
- `/api`: FastAPI inference service and Dockerfile.
- `/src`: Modular Python scripts for each ML stage.
- `/pipeline`: Main orchestration logic.
- `/monitoring`: Prometheus alert rules and configuration.
- `/plots`: Explainability and evaluation visualizations.
- `docker-compose.yml`: Multi-container setup (MLFlow, API, Prometheus, Grafana).

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/M-AbubakarNadeem/fraud-detection-mlops.git
   ```
2. **Run the Infrastructure:**
   ```bash
   docker-compose up -d --build
   ```
3. **Execute the Pipeline:**
   ```bash
   python pipeline/main_pipeline.py
   ```

## 📊 Monitoring
- **MLFlow UI**: `http://localhost:5000`
- **Inference API**: `http://localhost:8000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

## ⚖️ License
Academic Project - Semester 8 MLOps Assignment.
