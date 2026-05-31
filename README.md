# Credit Risk Scoring — End-to-End MLOps on Amazon SageMaker

A production-grade machine learning pipeline that predicts loan default risk in real time,
built entirely on Amazon SageMaker.

## Architecture

```
Raw Data (S3)
     ↓
Feature Engineering (SageMaker Processing Jobs)
     ↓
Model Training (XGBoost + SageMaker Experiments)
     ↓
Hyperparameter Tuning (Bayesian HPO — 20 jobs)
     ↓
Model Registry (SageMaker Model Registry)
     ↓
Explainability (SageMaker Clarify — SHAP + Bias)
     ↓
Real-Time Endpoint → Lambda → API Gateway (REST API)
```

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- 307,511 loan applications
- 122 features (financial, demographic, credit history)
- Binary target: loan default (1) vs no default (0)
- Class imbalance: 92% / 8%

## Results

| Metric | Value |
|---|---|
| Baseline AUC-ROC | ~0.73 |
| After HPO AUC-ROC | ~0.76 |

## Project Structure

```
credit-risk-project/
├── notebooks/
│   ├── 01_setup_and_eda.ipynb           Phase 1: Data ingestion & EDA
│   ├── 02_feature_engineering.ipynb     Phase 2: SageMaker Processing Job
│   ├── 03_Model_training.ipynb          Phase 3: XgBoost Model Training  
│   ├── 04_HyperParameters_tuning.ipynb  Phase 4: Bayesian HPO
│   ├── 05_registry_and_clarify.ipynb    Phase 5: Model Registry + SHAP
│   └── 06_deployment.ipynb              Phase 6: Endpoint + REST API
├── src/
│   ├── preprocessing.py               Runs inside SageMaker Processing Job
│   └── train.py                       Runs inside SageMaker Training Job
├── requirements.txt
└── images/
```

## SageMaker Services Used

| Service | Purpose |
|---|---|
| Processing Jobs | Feature engineering at scale |
| Training Jobs | XGBoost model training |
| Automatic Model Tuning | Bayesian hyperparameter optimization |
| Experiments | Run tracking and comparison |
| Model Registry | Versioning and approval workflow |
| Clarify | SHAP explainability + bias detection |
| Real-Time Endpoints | Sub-100ms inference |
| Pipelines | Orchestrated retraining |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/prashanthreddy31/credit-risk-sagemaker.git
cd credit-risk-sagemaker

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure AWS credentials
aws configure

# 5. Create .env
cp .env
# Add your AWS_ROLE_ARN, S3_BUCKET, AWS_REGION

# 6. Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
```

## API Usage

```bash
# Score a loan application
curl -X POST https://<your-api-id>.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [<comma-separated feature values>]}'

# Response
{
  "default_probability": 0.0823,
  "risk_label": "LOW",
  "threshold": 0.5,
  "model_endpoint": "credit-risk-realtime"
}
```

## Cleanup

```python
# Delete endpoint (most important — stops billing)
sm_client.delete_endpoint()

## Author
- [Prashanth S V](https://github.com/prashanthreddy31)

Built as a portfolio project demonstrating end-to-end MLOps on AWS SageMaker.
