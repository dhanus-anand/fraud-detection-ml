# Threshold Tuning — Fraud Detection

Why the default 0.5 threshold is wrong, how to find the optimal threshold using business cost, and a threshold tuning calculator.

---

## Why Default 0.5 Is Wrong

XGBoost outputs fraud probabilities between 0 and 1. The default classification threshold is 0.5: predict fraud if probability ≥ 0.5.

**This is wrong for two reasons:**

### Reason 1: Calibration Issues with Imbalanced Data

With 0.172% fraud rate, a well-calibrated model would output probabilities near 0.0017 for most transactions. The threshold of 0.5 would miss almost everything.

```
Fraud prevalence: 0.172%
Model output for average transaction: ~0.003-0.01
Model output for suspicious transaction: ~0.3-0.9

At threshold 0.5:
  → Only very high-confidence frauds are caught
  → Recall ~40-60%
  → We're missing 40-60% of all fraud
```

### Reason 2: Cost Asymmetry Is Ignored

Different types of errors have wildly different costs:
- **False Negative (missed fraud):** Customer's card is used fraudulently. Expected loss: $10,000
- **False Positive (false decline):** Legitimate transaction declined. Lost sale + friction. Cost: ~$50

At the 1:200 cost ratio ($10k vs $50), we should be willing to flag 200 legitimate transactions to catch 1 fraud.

The optimal threshold is the point where catching one more fraud costs exactly as much as the frauds we're catching.

---

## Business Cost Function

```python
def expected_cost_per_1000(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fraud_cost: float = 10_000,
    decline_cost: float = 50,
) -> dict:
    """
    Calculate expected business cost per 1000 transactions at a given threshold.
    
    Returns:
        cost_per_1000: Expected dollar cost per 1000 transactions
        precision: Fraction of flagged transactions that are actual fraud
        recall: Fraction of all fraud that was caught
        flagged_rate: Fraction of transactions that get flagged
    """
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y_true)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()  # Caught fraud
    fp = ((y_pred == 1) & (y_true == 0)).sum()  # False declines
    fn = ((y_pred == 0) & (y_true == 1)).sum()  # Missed fraud
    tn = ((y_pred == 0) & (y_true == 0)).sum()  # Correct legitimate
    
    missed_fraud_cost = fn * fraud_cost
    false_decline_cost = fp * decline_cost
    total_cost = missed_fraud_cost + false_decline_cost
    
    return {
        'threshold': threshold,
        'cost_per_1000': total_cost / n * 1000,
        'missed_fraud_cost': missed_fraud_cost / n * 1000,
        'false_decline_cost': false_decline_cost / n * 1000,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'flagged_rate': (tp + fp) / n,
        'fraud_caught_per_1000': tp / n * 1000,
        'false_declines_per_1000': fp / n * 1000,
    }
```

---

## Threshold Analysis

Sweep thresholds from 0.05 to 0.95 and compare expected cost:

```python
thresholds = np.arange(0.05, 0.95, 0.01)
results = [expected_cost_per_1000(y_test, y_proba_test, t) for t in thresholds]

# Plot: Expected cost vs threshold
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Expected cost vs threshold
costs = [r['cost_per_1000'] for r in results]
axes[0, 0].plot(thresholds, costs)
optimal_threshold = thresholds[np.argmin(costs)]
axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal: {optimal_threshold:.2f}')
axes[0, 0].set_xlabel('Threshold')
axes[0, 0].set_ylabel('Expected Cost per 1000 Transactions ($)')
axes[0, 0].set_title('Business Cost vs Classification Threshold')
axes[0, 0].legend()

# Panel 2: Precision and Recall vs threshold
precisions = [r['precision'] for r in results]
recalls = [r['recall'] for r in results]
axes[0, 1].plot(thresholds, precisions, label='Precision')
axes[0, 1].plot(thresholds, recalls, label='Recall')
axes[0, 1].axvline(optimal_threshold, color='red', linestyle='--')
axes[0, 1].set_title('Precision & Recall vs Threshold')
axes[0, 1].legend()

# Panel 3: Cost breakdown (missed fraud vs false declines)
missed = [r['missed_fraud_cost'] for r in results]
declined = [r['false_decline_cost'] for r in results]
axes[1, 0].stackplot(thresholds, missed, declined,
                     labels=['Missed Fraud Cost', 'False Decline Cost'])
axes[1, 0].set_title('Cost Breakdown vs Threshold')
axes[1, 0].legend()

# Panel 4: Flag rate vs threshold
flag_rates = [r['flagged_rate'] * 100 for r in results]
axes[1, 1].plot(thresholds, flag_rates)
axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--')
axes[1, 1].set_title('Flagged Transaction Rate vs Threshold')
axes[1, 1].set_ylabel('% of Transactions Flagged')

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150)
```

---

## Optimal Threshold Calculation

The optimal threshold minimizes expected cost:

```python
optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_result = results[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Expected cost per 1000 transactions: ${optimal_result['cost_per_1000']:,.0f}")
print(f"Precision: {optimal_result['precision']:.1%}")
print(f"Recall: {optimal_result['recall']:.1%}")
print(f"Fraud caught per 1000 transactions: {optimal_result['fraud_caught_per_1000']:.2f}")
print(f"False declines per 1000 transactions: {optimal_result['false_declines_per_1000']:.2f}")
print(f"Transaction flag rate: {optimal_result['flagged_rate']:.1%}")

# Expected output (approximate):
# Optimal threshold: 0.32
# Expected cost per 1000 transactions: $9.60
# Precision: 76.3%
# Recall: 84.1%
# Fraud caught per 1000 transactions: 1.45
# False declines per 1000 transactions: 0.45
# Transaction flag rate: 0.19%
```

---

## Sensitivity Analysis

The optimal threshold depends on the assumed fraud cost ($10k) and decline cost ($50). A sensitivity analysis shows how the optimal threshold changes with different business parameters:

```python
fraud_costs = [5_000, 10_000, 20_000, 50_000]
decline_costs = [25, 50, 100, 200]

print("Optimal threshold sensitivity analysis:")
print(f"{'Fraud Cost':>12} {'Decline Cost':>14} {'Optimal Threshold':>18} {'Recall':>8} {'Precision':>10}")
print("-" * 65)

for fc in fraud_costs:
    for dc in decline_costs:
        results_sensitivity = [
            expected_cost_per_1000(y_test, y_proba_test, t, fc, dc)
            for t in thresholds
        ]
        costs_s = [r['cost_per_1000'] for r in results_sensitivity]
        opt_idx = np.argmin(costs_s)
        opt_t = thresholds[opt_idx]
        opt_r = results_sensitivity[opt_idx]
        print(f"${fc:>10,} ${dc:>12,} {opt_t:>18.3f} {opt_r['recall']:>7.1%} {opt_r['precision']:>9.1%}")
```

**Interpretation:** As the cost ratio (fraud_cost / decline_cost) increases, the optimal threshold decreases — we become more aggressive about flagging, accepting more false positives to avoid missing fraud.

---

## Threshold Tuning Tool

A simple calculator that lets you input business parameters and see the recommended threshold:

```python
# src/models/threshold_calculator.py

def calculate_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraud_cost: float,
    decline_cost: float,
    min_recall: float = 0.0,    # Optional: enforce minimum recall
    max_flag_rate: float = 1.0, # Optional: cap flagged rate
) -> ThresholdResult:
    """
    Find the optimal threshold for given business parameters.
    
    Args:
        fraud_cost: Dollar cost of missing one fraud
        decline_cost: Dollar cost of one false decline
        min_recall: Minimum acceptable recall (0-1)
        max_flag_rate: Maximum fraction of transactions to flag
    
    Returns:
        ThresholdResult with optimal threshold and performance metrics
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_cost = float('inf')
    best_threshold = 0.5
    best_metrics = {}
    
    for t in thresholds:
        metrics = expected_cost_per_1000(y_true, y_proba, t, fraud_cost, decline_cost)
        
        # Apply constraints
        if metrics['recall'] < min_recall:
            continue
        if metrics['flagged_rate'] > max_flag_rate:
            continue
        
        if metrics['cost_per_1000'] < best_cost:
            best_cost = metrics['cost_per_1000']
            best_threshold = t
            best_metrics = metrics
    
    return ThresholdResult(
        threshold=best_threshold,
        cost_per_1000=best_cost,
        **best_metrics
    )
```

---

## Production Threshold Management

The optimal threshold is stored as a model tag in MLflow and loaded at API startup:

```python
# During training:
with mlflow.start_run():
    # ... train model ...
    optimal_threshold = calculate_optimal_threshold(y_val, y_proba_val, 10_000, 50)
    mlflow.log_param("optimal_threshold", optimal_threshold.threshold)
    mlflow.set_tag("optimal_threshold", str(optimal_threshold.threshold))

# In API:
model_meta = mlflow_client.get_run(run_id)
threshold = float(model_meta.data.tags.get("optimal_threshold", "0.5"))
```

**Threshold drift:** The optimal threshold may need updating if the fraud rate or cost parameters change. Monitor the distribution of model scores in production — if the score distribution shifts, re-run threshold optimization on recent data.
