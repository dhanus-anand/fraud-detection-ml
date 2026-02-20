# Monitoring — Fraud Detection System

Real-time metrics, feature drift detection, model performance monitoring, and the incident response playbook.

---

## Monitoring Stack

```
FastAPI Application
       │
       ▼ (Prometheus metrics)
Prometheus (scrapes /metrics every 15s)
       │
       ▼
Grafana (dashboards, alerts)
       │
       ▼ (on alert)
PagerDuty / Slack (incident notifications)
```

---

## Real-Time Metrics

### API Metrics (Prometheus)

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
prediction_requests = Counter(
    'fraud_prediction_requests_total',
    'Total prediction requests',
    ['model_version', 'result']  # result: fraud|legitimate
)

prediction_latency = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

# Model metrics
fraud_score_distribution = Histogram(
    'fraud_score_distribution',
    'Distribution of fraud probability scores',
    ['model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Business metrics
flags_per_minute = Gauge(
    'fraud_flags_per_minute',
    'Rolling count of fraud flags (1-minute window)'
)

approval_rate = Gauge(
    'transaction_approval_rate',
    'Fraction of transactions approved (1 - flag rate)'
)

# Confirmed fraud metrics (from POST /feedback)
confirmed_fraud_total = Counter(
    'confirmed_fraud_total',
    'Total confirmed fraud cases (from feedback)'
)

false_positive_total = Counter(
    'false_positive_total',
    'Confirmed legitimate transactions that were flagged'
)
```

### Grafana Dashboard Panels

1. **Request throughput** — predictions/second (rate)
2. **p50/p95/p99 latency** — must stay under 100ms for p95
3. **Flag rate** — % of transactions flagged as fraud (normal: ~0.3-0.5%)
4. **Score distribution** — histogram of fraud probabilities (should be stable)
5. **Approval rate** — % of transactions approved
6. **Confirmed precision** (from feedback) — FP / (TP + FP)
7. **Model version traffic split** — champion vs challenger

---

## Feature Drift Detection

Feature drift occurs when the distribution of input features shifts compared to the training distribution. This can degrade model performance without any visible accuracy drop (because ground truth labels arrive days/weeks later).

### Implementation

```python
class FeatureDriftDetector:
    """
    Detects statistical drift in input features using Population Stability Index (PSI)
    and Kolmogorov-Smirnov test.
    """
    
    def __init__(self, reference_stats: dict, alert_threshold: float = 0.2):
        """
        reference_stats: Statistics computed on training data.
        alert_threshold: PSI > 0.2 is considered significant drift.
        """
        self.reference = reference_stats
        self.threshold = alert_threshold
    
    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index (PSI):
          < 0.1: No significant drift
          0.1-0.2: Some drift, investigate
          > 0.2: Significant drift, retraining needed
        """
        # Compute bin edges from expected distribution
        _, bin_edges = np.histogram(expected, bins=bins)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Compute frequencies
        expected_freq, _ = np.histogram(expected, bins=bin_edges)
        actual_freq, _ = np.histogram(actual, bins=bin_edges)
        
        expected_pct = expected_freq / len(expected)
        actual_pct = actual_freq / len(actual)
        
        # Avoid log(0)
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi
    
    def check_drift(self, current_features: pd.DataFrame) -> DriftReport:
        """Check all features for drift and return report."""
        drift_scores = {}
        alerts = []
        
        for feature in current_features.columns:
            if feature not in self.reference:
                continue
            
            psi = self.compute_psi(
                self.reference[feature]['values'],
                current_features[feature].values
            )
            drift_scores[feature] = psi
            
            if psi > self.threshold:
                alerts.append(DriftAlert(
                    feature=feature,
                    psi=psi,
                    severity='high' if psi > 0.4 else 'medium',
                    message=f"Feature '{feature}' has PSI={psi:.3f} (threshold: {self.threshold})"
                ))
        
        return DriftReport(
            drift_scores=drift_scores,
            alerts=alerts,
            max_psi=max(drift_scores.values()) if drift_scores else 0,
            drifted_features=[k for k, v in drift_scores.items() if v > self.threshold]
        )
```

### Drift Monitoring Schedule

```python
# Background task: Check for drift every hour using last 1000 predictions
@app.on_event("startup")
async def start_drift_monitor():
    asyncio.create_task(drift_monitoring_loop())

async def drift_monitoring_loop():
    while True:
        await asyncio.sleep(3600)  # Run every hour
        
        # Get last 1000 feature vectors from database
        recent_features = await get_recent_features(limit=1000)
        
        if len(recent_features) < 100:
            continue  # Not enough data
        
        report = drift_detector.check_drift(recent_features)
        
        # Publish drift metrics
        for feature, psi in report.drift_scores.items():
            feature_drift_gauge.labels(feature=feature).set(psi)
        
        # Alert on high drift
        for alert in report.alerts:
            if alert.severity == 'high':
                await send_alert(f"HIGH DRIFT: {alert.message}")
                retraining_triggers.inc()
```

---

## Model Performance Monitoring

### Precision/Recall on Confirmed Fraud

Ground truth labels arrive via the `POST /feedback` endpoint. This endpoint is called by the fraud team when they confirm or deny a fraud case.

```python
@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest):
    """
    Record the actual outcome of a transaction.
    Called by fraud analysts or chargeback system.
    """
    prediction = await db.get_prediction(feedback.transaction_id)
    
    # Update monitoring counters
    if feedback.actual_label == 1 and prediction.is_fraud:
        true_positives.inc()  # Correctly caught fraud
    elif feedback.actual_label == 0 and prediction.is_fraud:
        false_positives.inc()  # False alarm (legitimate flagged)
    elif feedback.actual_label == 1 and not prediction.is_fraud:
        false_negatives.inc()  # Missed fraud
    else:
        true_negatives.inc()   # Correctly approved legitimate
    
    # Store for drift detection and retraining
    await db.store_feedback(feedback)
    
    return {"status": "recorded"}
```

### Performance Degradation Alerts

```python
# Alert if rolling precision drops below 60% (over last 500 confirmed cases)
precision_degradation_alert:
  condition: confirmed_precision_rolling_500 < 0.60
  severity: high
  action: Page on-call ML engineer

# Alert if estimated recall drops (fraud rate in feedback >> model flag rate)
recall_degradation_alert:
  condition: (confirmed_fraud_rate - model_flag_rate) > 0.05
  severity: high
  action: Immediate review + potential model rollback
```

---

## Automatic Retraining Triggers

Retraining is triggered when:

| Trigger | Condition | Action |
|---------|-----------|--------|
| Feature drift | Any feature PSI > 0.25 | Queue retraining |
| Performance drop | Precision < 60% over 500 confirmed | Immediate retraining |
| Score distribution shift | KL divergence of score histogram > 0.1 | Investigate + queue |
| Time-based | Weekly (Sundays 2am) | Scheduled retraining |
| Manual | On-call engineer judgment | Immediate retraining |

```python
class RetrainingTrigger:
    def __init__(self, training_pipeline, alert_service):
        self.pipeline = training_pipeline
        self.alerts = alert_service
    
    async def check_and_trigger(self, drift_report: DriftReport, perf_metrics: PerformanceMetrics):
        should_retrain = False
        reasons = []
        
        if drift_report.max_psi > 0.25:
            should_retrain = True
            reasons.append(f"Feature drift: PSI={drift_report.max_psi:.3f}")
        
        if perf_metrics.rolling_precision < 0.60:
            should_retrain = True
            reasons.append(f"Precision degradation: {perf_metrics.rolling_precision:.1%}")
        
        if should_retrain:
            await self.alerts.send(
                f"Retraining triggered: {', '.join(reasons)}",
                severity='warning'
            )
            await self.pipeline.queue_retraining(reasons=reasons)
```

---

## Business Metrics

Beyond technical metrics, track business impact:

```python
# Estimated fraud losses prevented (requires feedback data)
def calculate_business_impact(feedback_df: pd.DataFrame) -> dict:
    """
    Given actual fraud labels and model predictions, calculate business impact.
    """
    tp = ((feedback_df['actual'] == 1) & (feedback_df['predicted'] == 1)).sum()
    fp = ((feedback_df['actual'] == 0) & (feedback_df['predicted'] == 1)).sum()
    fn = ((feedback_df['actual'] == 1) & (feedback_df['predicted'] == 0)).sum()
    
    fraud_prevented = tp * 10_000      # Each caught fraud saves $10k
    decline_cost = fp * 50              # Each false decline costs $50
    missed_fraud_cost = fn * 10_000    # Each miss costs $10k
    
    net_value = fraud_prevented - decline_cost
    
    return {
        'fraud_prevented_count': int(tp),
        'fraud_prevented_value': fraud_prevented,
        'false_decline_cost': decline_cost,
        'missed_fraud_cost': missed_fraud_cost,
        'net_value_created': net_value,
        'approval_rate': 1 - (tp + fp) / len(feedback_df),
    }
```

---

## Incident Response Playbook

### Incident: Flag Rate Spikes (>5% of transactions flagged)

```
1. CHECK: Is it a model issue or an actual fraud wave?
   - Look at fraud feedback: Are flagged transactions confirmed fraud?
   - If yes: Actual fraud wave. Keep model, alert fraud team.
   - If no: Model is misfiring. Investigate.

2. IF MODEL ISSUE:
   - Check feature drift dashboard
   - If major drift: Rollback to previous model version
   - Notify on-call ML engineer
```

### Incident: API Latency > 200ms p95

```
1. CHECK: Is it inference latency or SHAP computation?
   - Disable SHAP for all requests (env var EXPLAIN_ENABLED=false)
   - Measure latency without SHAP

2. IF INFERENCE SLOW:
   - Scale up API replicas
   - Check Redis feature store latency
   - Consider model quantization

3. IF SHAP SLOW:
   - Enable async SHAP computation
   - Cache SHAP values for common feature vectors
```

### Incident: Model Precision < 50%

```
1. IMMEDIATE: Page on-call ML engineer
2. CHECK: Is this a data issue or model issue?
   - Review recent feedback data for anomalies
   - Check feature drift report

3. IF MODEL ISSUE:
   - Rollback to previous champion model:
     mlflow models transition --model-name fraud_detector \
       --version <prev_version> --stage Production

4. ROOT CAUSE:
   - Analyze what changed (new fraud patterns? data pipeline issue?)
   - Retrain with recent data
   - A/B test new model before promoting
```
