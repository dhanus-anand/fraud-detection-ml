# Fraud Detection ML System

A production-grade fraud detection system demonstrating end-to-end ML engineering: data pipeline, class imbalance handling, cost-sensitive learning, SHAP explainability, A/B testing, and drift monitoring.

**"Detect fraudulent transactions in real-time while optimizing business outcomes — fraud losses vs customer friction."**

> **Portfolio context:** This is Project 3 of 3. After building a platform product (Project 1) and distributed infrastructure (Project 2), this project explores production ML engineering — not just model training, but the full system around it. The goal was to understand what an ML Engineer actually builds at a company: feature stores, model registries, drift monitoring, explainability APIs, and A/B testing frameworks. The fraud domain was chosen specifically because it has a real business cost function that most ML tutorials ignore.

---

## What It Does

This is not just a model training exercise. It's a full production ML system:

1. **Data pipeline** — Kaggle Credit Card Fraud dataset (~284k transactions, 0.172% fraud)
2. **Class imbalance** — Cost-sensitive learning (not just SMOTE or undersampling)
3. **XGBoost model** — Custom objective function minimizing expected business cost
4. **Threshold optimization** — Finds the threshold minimizing `($10k × missed_frauds) + ($50 × false_declines)`
5. **SHAP explainability** — Per-transaction explanation of why a transaction was flagged
6. **FastAPI inference** — Real-time predictions with < 100ms p95 latency
7. **Feature store** — Redis (online) + PostgreSQL (offline) for sub-10ms feature lookups
8. **MLflow registry** — Model versioning, experiment tracking, artifact management
9. **A/B testing** — Champion/challenger framework with statistical significance testing
10. **Drift monitoring** — Feature and model performance drift detection with alerts

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Core ML | XGBoost 2.0, scikit-learn 1.5 |
| Explainability | SHAP 0.45 |
| Imbalanced Data | imbalanced-learn 0.12 |
| API | FastAPI 0.111, Uvicorn |
| Model Registry | MLflow 2.14 |
| Feature Store | Redis (online) + PostgreSQL (offline) |
| Monitoring | Prometheus client, custom metrics |
| Deployment | Railway or Fly.io (~$5/month) |
| CI/CD | GitHub Actions |
| Dataset | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Key Design Decisions

### Why Cost-Sensitive Learning?

The dataset has a 577:1 class imbalance (0.172% fraud). Standard approaches:

| Approach | What's Optimized | Business Aligned? |
|----------|-----------------|-------------------|
| Default threshold (0.5) | Raw probability | ❌ No |
| Accuracy | # correct / total | ❌ No (99.8% by predicting "all legitimate") |
| ROC-AUC | Rank ordering | Partially |
| SMOTE | Synthetic samples | ❌ Introduces noise |
| **Cost-sensitive learning** | Expected dollar cost | ✅ Yes |

**Business cost function:**
```
Expected cost = (missed_frauds × $10,000) + (false_declines × $50)
```

We optimize the model to minimize this cost — not raw accuracy or even F1.

### Why SHAP?

Fraud decisions must be explainable:
- Regulatory compliance (explainable AI requirements)
- Analyst trust: "Why was this flagged?"
- Model debugging: "What's driving false positives?"

SHAP provides both global feature importance and per-transaction explanations.

---

## API Endpoints

```
POST /predict              Real-time fraud prediction (< 100ms)
POST /predict/batch        Batch predictions
GET  /models               List available model versions
GET  /models/{version}/metrics  Model performance stats
POST /feedback             Report actual fraud outcome (for monitoring)
GET  /health               Health check
GET  /metrics              Prometheus metrics
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn-123",
    "amount": 450.00,
    "v1": -1.359807134,
    "v2": -0.072781173,
    ...
    "v28": -0.021053
  }'
```

**Response:**
```json
{
  "transaction_id": "txn-123",
  "fraud_probability": 0.87,
  "is_fraud": true,
  "threshold": 0.32,
  "model_version": "v1.2.0",
  "latency_ms": 12,
  "explanation": {
    "top_features": [
      {"name": "V14", "contribution": 0.42, "value": -15.3},
      {"name": "V10", "contribution": 0.28, "value": -8.7},
      {"name": "Amount", "contribution": 0.18, "value": 450.00}
    ],
    "shap_base_value": 0.0017
  }
}
```

---

## Project Structure

```
project-3-fraud-detection-ml/
├── src/
│   ├── data/           # Data loading, preprocessing, feature engineering
│   ├── features/       # Feature store (Redis + PostgreSQL)
│   ├── models/         # Training, evaluation, threshold optimization
│   ├── api/            # FastAPI application
│   └── monitoring/     # Drift detection, metrics, A/B testing
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline.ipynb       # Baseline + imbalance comparison
│   └── 03_optimization.ipynb  # XGBoost + threshold + SHAP
├── mlruns/             # MLflow artifacts (local)
├── docker-compose.yml  # PostgreSQL + Redis + MLflow + API
└── requirements.txt    # All Python dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Kaggle account (for dataset)

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd project-3-fraud-detection-ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your values

# 3. Start infrastructure
docker-compose up postgres redis mlflow -d

# 4. Download dataset
pip install kaggle
kaggle datasets download mlg-ulb/creditcardfraud -p data/raw --unzip

# 5. Run training pipeline
python src/models/train.py

# 6. Start inference API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Service URLs:**
- Inference API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full ML pipeline, feature store, deployment |
| [CLASS_IMBALANCE.md](CLASS_IMBALANCE.md) | Imbalance strategies compared, cost-sensitive approach |
| [THRESHOLD_TUNING.md](THRESHOLD_TUNING.md) | Business cost optimization, threshold calculator |
| [EXPLAINABILITY.md](EXPLAINABILITY.md) | SHAP implementation, per-transaction explanations |
| [MONITORING.md](MONITORING.md) | Drift detection, retraining triggers, alerting |
| [AB_TESTING.md](AB_TESTING.md) | Champion/challenger framework, statistical significance |
| [OPUS_AGENT_INSTRUCTIONS.md](OPUS_AGENT_INSTRUCTIONS.md) | Build instructions |

---

## Model Performance

Targets (to be updated with actual results):

| Metric | Target |
|--------|--------|
| ROC-AUC | > 0.95 |
| PR-AUC | > 0.75 |
| Recall at optimal threshold | > 80% |
| Precision at optimal threshold | > 70% |
| Expected cost reduction vs no-model | > 40% |
| Inference latency (p95) | < 100ms |

---

## Cost

| Service | Cost |
|---------|------|
| Railway API deployment | ~$5/month |
| Neon PostgreSQL | Free (0.5GB) |
| Upstash Redis | Free (10k commands/day) |
| **Total** | **~$5/month** |
