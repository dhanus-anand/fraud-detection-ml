# Architecture — Fraud Detection ML System

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                                                             │
│  Kaggle Dataset (CSV)                                       │
│       │                                                     │
│       ▼                                                     │
│  Data Loading & Validation                                  │
│  (src/data/loader.py)                                       │
│       │                                                     │
│       ▼                                                     │
│  Feature Engineering                                        │
│  (src/features/engineering.py)                              │
│  ─ Amount log-transform                                     │
│  ─ Time → hour-of-day, day-of-week                         │
│  ─ Amount statistics (rolling windows)                      │
│       │                                                     │
│       ▼                                                     │
│  Train/Val/Test Split (temporal)                            │
│       │                                                     │
│       ▼                                                     │
│  Baseline: Logistic Regression                              │
│  Primary:  XGBoost (cost-sensitive)                         │
│  (src/models/train.py)                                      │
│       │                                                     │
│       ▼                                                     │
│  Threshold Optimization                                     │
│  (src/models/threshold.py)                                  │
│  → minimize: (FN × $10k) + (FP × $50)                      │
│       │                                                     │
│       ▼                                                     │
│  SHAP Explainability                                        │
│  (src/models/explainer.py)                                  │
│       │                                                     │
│       ▼                                                     │
│  MLflow Registry                                            │
│  (model binary + preprocessing + metadata)                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ Model artifact
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
│                                                             │
│  HTTP Request (transaction features)                        │
│       │                                                     │
│       ▼                                                     │
│  FastAPI Endpoint                                           │
│  (src/api/main.py)                                          │
│       │                                                     │
│       ▼                                                     │
│  Feature Lookup (Feature Store)                             │
│  ─ Online: Redis (< 10ms)                                   │
│  ─ Offline: PostgreSQL (batch pre-computed)                 │
│       │                                                     │
│       ▼                                                     │
│  Preprocessing Pipeline                                     │
│  (StandardScaler from training)                             │
│       │                                                     │
│       ▼                                                     │
│  XGBoost Prediction                                         │
│  → fraud_probability                                        │
│       │                                                     │
│       ▼                                                     │
│  Threshold Application                                      │
│  → is_fraud = (probability >= optimal_threshold)            │
│       │                                                     │
│       ▼                                                     │
│  SHAP Explanation (if requested)                            │
│  → top 5 contributing features                              │
│       │                                                     │
│       ▼                                                     │
│  A/B Traffic Routing                                        │
│  (90% champion, 10% challenger model)                       │
│       │                                                     │
│       ▼                                                     │
│  Response + Monitoring Metrics                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Detail

### Training Data

```
Source: Kaggle Credit Card Fraud Detection dataset
  - creditcard.csv (143MB)
  - 284,807 transactions
  - 492 fraud cases (0.172%)
  - Features: V1-V28 (PCA-transformed), Amount, Time, Class

Temporal split (NOT random — preserves time ordering):
  Train:      80% (first 227,846 transactions)
  Validation: 10% (next 28,480 transactions)
  Test:       10% (last 28,481 transactions)

Why temporal split?
  Random split causes data leakage: future patterns leak into training set.
  Real fraud detection systems always train on past, predict on future.
```

### Feature Engineering

```python
# Features added on top of V1-V28:
df['Amount_log'] = np.log1p(df['Amount'])          # Log-transform skewed amount
df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
df['Hour'] = (df['Time'] // 3600) % 24              # Hour of day (0-23)
df['Time_scaled'] = StandardScaler().fit_transform(df[['Time']])
```

### Model Registry Versions

```
v1.0.0 — Logistic Regression baseline
v1.1.0 — XGBoost with scale_pos_weight
v1.2.0 — XGBoost with cost-sensitive custom objective  ← production champion
v2.0.0 — XGBoost + enhanced features + tuned threshold
```

---

## Feature Store Design

### Purpose

The feature store separates feature computation from model training/inference:
- Features are computed once and reused across model versions
- Online features serve real-time inference with low latency
- Offline features enable reproducible training

### Online Feature Store (Redis)

For real-time inference, frequently-needed features are pre-computed and cached:

```python
# Feature cache key format: features:{entity_type}:{entity_id}:{feature_name}
# Example: features:card:4111111111111111:txn_count_1h

class OnlineFeatureStore:
    def __init__(self, redis_client, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def get(self, entity_id: str, feature_names: list[str]) -> dict:
        pipe = self.redis.pipeline()
        for name in feature_names:
            key = f"features:{entity_id}:{name}"
            pipe.get(key)
        results = pipe.execute()
        return {name: val for name, val in zip(feature_names, results)}
    
    def set(self, entity_id: str, features: dict):
        pipe = self.redis.pipeline()
        for name, value in features.items():
            key = f"features:{entity_id}:{name}"
            pipe.setex(key, self.ttl, str(value))
        pipe.execute()
```

**Online features (pre-computed, cached in Redis):**
- `txn_count_1h` — transactions by this card in last 1 hour
- `txn_count_24h` — transactions in last 24 hours
- `avg_amount_7d` — average transaction amount in last 7 days
- `merchant_first_seen` — is this a new merchant for this card?

### Offline Feature Store (PostgreSQL)

For batch/training features, PostgreSQL stores pre-computed historical aggregates:

```sql
CREATE TABLE offline_features (
  entity_id   VARCHAR(255) NOT NULL,
  feature_name VARCHAR(100) NOT NULL,
  feature_value DOUBLE PRECISION,
  computed_at  TIMESTAMP NOT NULL DEFAULT NOW(),
  valid_from   TIMESTAMP NOT NULL,
  valid_until  TIMESTAMP,
  PRIMARY KEY (entity_id, feature_name, valid_from)
);
```

---

## API Design

### FastAPI Application Structure

```python
# src/api/main.py

app = FastAPI(title="Fraud Detection API")

@app.post("/predict")
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    # 1. Preprocess transaction features
    features = preprocessor.transform(transaction.model_dump())
    
    # 2. A/B routing: 90% to champion, 10% to challenger
    model = ab_router.get_model(transaction.transaction_id)
    
    # 3. Predict
    probability = model.predict_proba([features])[0][1]
    is_fraud = probability >= model.threshold
    
    # 4. Explain (top 5 SHAP features)
    explanation = explainer.explain_single(features, model)
    
    # 5. Record for monitoring
    monitoring.record_prediction(model.version, probability, is_fraud)
    
    return PredictionResponse(
        fraud_probability=probability,
        is_fraud=is_fraud,
        threshold=model.threshold,
        model_version=model.version,
        explanation=explanation
    )
```

---

## Deployment Architecture

### Current (Railway, ~$5/month)

```
Railway Service: fraud-detection-api
  - FastAPI + Uvicorn
  - Environment variables: DATABASE_URL, REDIS_URL, MLFLOW_TRACKING_URI
  - Model artifacts loaded from MLflow at startup

External Services:
  - Neon PostgreSQL (free tier) — feature store
  - Upstash Redis (free tier) — online feature cache
  - MLflow on Railway (or filesystem) — model registry
```

### Model Loading at Startup

```python
# Load champion model from MLflow at API startup
@app.on_event("startup")
async def load_model():
    client = mlflow.tracking.MlflowClient()
    
    champion = client.get_latest_versions("fraud_detector", stages=["Production"])[0]
    app.state.champion_model = mlflow.xgboost.load_model(champion.source)
    app.state.champion_threshold = champion.tags.get("optimal_threshold", 0.5)
    
    challenger = client.get_latest_versions("fraud_detector", stages=["Staging"])
    if challenger:
        app.state.challenger_model = mlflow.xgboost.load_model(challenger[0].source)
```

### Production Architecture (AWS, ~$50/month)

```
ECS Fargate: 2 tasks (FastAPI)
ElastiCache: Redis (online features)
RDS: PostgreSQL (offline features)
S3: MLflow artifact storage
SageMaker: Optional — model hosting (eliminates custom deployment)
CloudFront: API caching for GET endpoints
```
