# Explainability — SHAP for Fraud Detection

Why explainability matters, how SHAP is implemented, and how to use explanations in the manual review workflow.

---

## Why Explainability Matters

### Regulatory Compliance

Financial services are subject to explainability requirements:
- **GDPR Article 22**: Right to explanation for automated decisions affecting individuals
- **Equal Credit Opportunity Act (ECOA)**: Must provide specific reasons for adverse actions
- **FCRA**: Must disclose factors that adversely affected a credit decision

A black-box model that rejects transactions without explanation is a compliance risk.

### Analyst Trust

Fraud analysts reviewing flagged transactions need to understand why the model flagged a transaction:

```
Without explanation:
  "This transaction has a 0.87 fraud probability"
  → Analyst: "...why? Should I trust this?"

With SHAP explanation:
  "This transaction has a 0.87 fraud probability because:
   - V14 is unusually low (-15.3) — contributes +0.42 to fraud score
   - Transaction amount ($450) is 3x this customer's typical amount — +0.18
   - V10 is outside normal range — +0.28"
  → Analyst: "OK, the unusual amount pattern makes sense. I'll investigate."
```

### Model Debugging

SHAP global explanations help diagnose model issues:
- If a non-informative feature has high SHAP importance: potential data leakage
- If expected predictive features rank low: model may have found spurious correlations

---

## SHAP Overview

SHAP (SHapley Additive exPlanations) is based on game theory's Shapley values.

**Core idea:** For a specific prediction, how much did each feature contribute to the difference between the prediction and the average prediction?

```
prediction = base_value + sum(SHAP_value_i for each feature i)

Where:
  base_value = mean(model_output) across training data
               = expected fraud probability (≈ 0.0017 for 0.172% fraud rate)
  
  SHAP_value_i = feature i's contribution to THIS prediction
               = positive → pushes toward fraud
               = negative → pushes toward legitimate
```

**Example:**
```
base_value = 0.0017
SHAP contributions:
  V14 = -15.3:        +0.42 (pushes toward fraud)
  V10 = -8.7:         +0.28
  Amount = $450:      +0.18
  V12 = +3.2:         -0.10 (pushes toward legitimate)
  Hour = 3am:         +0.15

prediction = 0.0017 + 0.42 + 0.28 + 0.18 - 0.10 + 0.15 + (others) ≈ 0.87
```

---

## Implementation

### TreeExplainer (Fast for XGBoost)

```python
import shap

class FraudExplainer:
    def __init__(self, model: xgb.XGBClassifier, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
    
    def explain_single(
        self,
        features: np.ndarray,
        top_n: int = 5,
    ) -> dict:
        """
        Generate SHAP explanation for a single transaction.
        
        Returns top N features contributing to the fraud decision.
        """
        shap_values = self.explainer.shap_values(features.reshape(1, -1))
        
        # shap_values shape: (1, n_features) for binary classification
        contributions = dict(zip(self.feature_names, shap_values[0]))
        
        # Sort by absolute contribution
        top_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        return {
            "base_value": float(self.explainer.expected_value),
            "top_features": [
                {
                    "name": name,
                    "contribution": float(contrib),
                    "value": float(features[self.feature_names.index(name)]),
                    "direction": "fraud" if contrib > 0 else "legitimate",
                }
                for name, contrib in top_features
            ]
        }
    
    def explain_batch(self, features: np.ndarray) -> list[dict]:
        """Batch explanation — more efficient than calling explain_single N times."""
        shap_values = self.explainer.shap_values(features)
        
        results = []
        for i in range(len(features)):
            contributions = dict(zip(self.feature_names, shap_values[i]))
            top_features = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            results.append({
                "base_value": float(self.explainer.expected_value),
                "top_features": [
                    {"name": n, "contribution": float(c)}
                    for n, c in top_features
                ]
            })
        return results
```

### Performance Optimization

TreeExplainer is fast for XGBoost (uses polynomial-time exact computation):
- Single prediction: ~5-15ms
- Batch of 100: ~50-100ms

For production at high throughput, pre-compute SHAP values in batches:

```python
# Cache SHAP explanations for batch predictions
@lru_cache(maxsize=10_000)
def cached_explain(features_tuple: tuple) -> dict:
    features = np.array(features_tuple)
    return explainer.explain_single(features)
```

---

## Global Feature Importance

SHAP summary plot shows the most important features across all test predictions:

```python
import shap
import matplotlib.pyplot as plt

# Compute SHAP values for test set (or a representative sample)
shap_values = explainer.explainer.shap_values(X_test_sample)

# Plot 1: Feature importance (mean absolute SHAP value)
shap.summary_plot(
    shap_values, X_test_sample,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig("global_feature_importance.png", dpi=150)

# Plot 2: Beeswarm plot (shows direction of effect + magnitude)
shap.summary_plot(
    shap_values, X_test_sample,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.savefig("shap_beeswarm.png", dpi=150)
```

**Interpretation of beeswarm plot:**
- Each dot = one transaction
- X-axis = SHAP value (impact on fraud probability)
- Color = feature value (red=high, blue=low)
- Features sorted by mean |SHAP value|

---

## Per-Transaction Explanation Examples

### Example 1: Caught Fraud

```
Transaction: Amount=$2,450 at 3am
Fraud probability: 0.92 → FRAUD

Base value: 0.002 (average fraud rate)

Top features pushing toward FRAUD:
  V14 = -18.5:  +0.51  (critical fraud indicator — very low V14)
  V10 = -12.1:  +0.29  (secondary fraud indicator)
  Amount:       +0.18  (unusually high amount)
  Hour = 3:     +0.12  (late-night transaction)
  V12 = -8.4:   +0.09

Features pushing toward LEGITIMATE:
  V2 = +1.2:    -0.05
  V4 = +2.1:    -0.04

Analyst notes: High-value transaction at 3am with anomalous V14/V10 values.
               Classic fraud pattern. Decline recommended.
```

### Example 2: Caught False Positive (Manual Review)

```
Transaction: Amount=$5,000 (international wire)
Fraud probability: 0.45 → below threshold 0.32 → LEGITIMATE

Explanation (still generated for borderline cases):
  Amount:      +0.20  (very high amount)
  V14 = -8.1:  +0.15  (somewhat unusual)
  V6 = +3.5:   -0.12  (pushes legitimate)
  Hour = 14:   -0.08  (normal business hours)

Analyst notes: High amount but during business hours. Customer profile shows
               international wire history. Approved.
```

---

## API Response Format

```json
{
  "transaction_id": "txn-123",
  "fraud_probability": 0.92,
  "is_fraud": true,
  "threshold": 0.32,
  "explanation": {
    "base_value": 0.0017,
    "top_features": [
      {
        "name": "V14",
        "contribution": 0.51,
        "value": -18.5,
        "direction": "fraud"
      },
      {
        "name": "V10",
        "contribution": 0.29,
        "value": -12.1,
        "direction": "fraud"
      },
      {
        "name": "Amount",
        "contribution": 0.18,
        "value": 2450.00,
        "direction": "fraud"
      }
    ],
    "explanation_text": "This transaction was flagged primarily due to unusual values in key fraud indicators (V14, V10) combined with a higher-than-typical transaction amount."
  }
}
```

---

## Manual Review Workflow

```
High fraud probability (>0.7):
  → Auto-decline + SHAP explanation logged
  → No analyst review required

Medium probability (threshold - 0.2 to threshold + 0.2):
  → Queue for manual review
  → SHAP explanation shown to analyst
  → Analyst approves or declines
  → Decision fed back via POST /feedback

Low fraud probability (<threshold - 0.2):
  → Auto-approve
  → Sampled 0.1% for quality review
```

---

## Explainability Performance Budget

Per-transaction SHAP computation takes 5-15ms. At p95 < 100ms latency target:
- Feature preprocessing: ~5ms
- XGBoost inference: ~3ms  
- SHAP computation: ~10ms
- Response serialization: ~2ms
- **Total: ~20ms** (well within 100ms target)

For high-throughput batch processing, disable SHAP by default and enable only for flagged transactions or sampled legitimate transactions.
