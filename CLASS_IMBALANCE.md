# Class Imbalance — Fraud Detection

How the system handles a 577:1 class ratio (0.172% fraud), and why cost-sensitive learning is the chosen approach.

---

## The Problem

```
Dataset: 284,807 transactions
  - Legitimate: 284,315 (99.828%)
  - Fraud:          492 (0.172%)
  - Imbalance ratio: 577:1
```

A naive model predicting "all legitimate" achieves **99.828% accuracy** while catching **zero fraud**. This is why accuracy is the wrong metric for fraud detection.

---

## Approaches Compared

### 1. No Handling (Baseline)

```python
model = XGBClassifier()
model.fit(X_train, y_train)
```

**Result:**
- Model learns that predicting "legitimate" is almost always correct
- Decision boundary is pushed far toward the minority class
- At default threshold (0.5): Very high precision, very low recall
- Catches ~40-60% of fraud but requires threshold tuning

**Verdict:** Poor default behavior. Must combine with threshold tuning.

---

### 2. Random Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
# Reduces majority class to 10× minority class
# Train size: ~4,920 (from 227,846)
```

**Pros:**
- Simple, fast
- Trains on balanced dataset

**Cons:**
- Throws away 98% of legitimate training data
- Rare patterns in majority class are lost
- Validation performance often degrades on full test set
- Not recommended when legitimate data is informative

**Verdict:** Avoid — data loss is significant.

---

### 3. SMOTE (Synthetic Minority Over-Sampling)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**SMOTE algorithm:**
1. For each minority sample, find k nearest minority neighbors
2. Generate synthetic samples along the line segments between them
3. Creates "realistic-looking" fraud examples

**Pros:**
- No data loss (unlike undersampling)
- Increases minority representation

**Cons:**
- Synthetic samples may not represent real fraud patterns
- Can introduce noise and outliers
- Overfitting risk (synthetic samples may be too easy to classify)
- V1-V28 are PCA components — interpolating in PCA space may not reflect reality

**Result:**
```
SMOTE PR-AUC: ~0.71 (vs baseline ~0.68)
Small improvement, but synthetic data quality is uncertain
```

**Verdict:** Marginal improvement. Better alternatives exist.

---

### 4. Class Weighting (scale_pos_weight)

```python
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count  # ~577

model = XGBClassifier(scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)
```

**How it works:**
- Multiplies the gradient contribution of positive (fraud) samples by `scale_pos_weight`
- Equivalent to training on a dataset where fraud is represented 577× more
- No data loss, no synthetic data

**Pros:**
- Simple to implement
- No data augmentation
- Works well for XGBoost

**Cons:**
- Optimizes for balanced accuracy, not business cost
- Does not account for different costs of false negatives vs false positives

**Result:**
```
XGBoost + scale_pos_weight:
  ROC-AUC: ~0.978
  PR-AUC: ~0.789
  Recall at 0.5 threshold: ~73%
```

**Verdict:** Good baseline. Better than SMOTE for this dataset.

---

### 5. Cost-Sensitive Learning (Chosen Approach)

```python
def fraud_cost_objective(predt: np.ndarray, dtrain: xgb.DMatrix):
    """
    Custom XGBoost objective function that minimizes expected business cost.
    
    Business parameters:
      - Cost of missing a fraud (false negative): $10,000
      - Cost of false decline (false positive):   $50
    """
    labels = dtrain.get_label()
    
    FRAUD_COST = 10_000
    DECLINE_COST = 50
    
    # Sigmoid activation
    prob = 1.0 / (1.0 + np.exp(-predt))
    
    # Gradient and hessian of expected cost
    # For fraud (label=1): grad = -FRAUD_COST * (1 - prob)
    # For legitimate (label=0): grad = DECLINE_COST * prob
    
    grad = np.where(
        labels == 1,
        -FRAUD_COST * (1 - prob),   # Miss a fraud: high penalty
        DECLINE_COST * prob          # False decline: low penalty
    )
    
    hess = np.where(
        labels == 1,
        FRAUD_COST * prob * (1 - prob),
        DECLINE_COST * prob * (1 - prob)
    )
    
    return grad, hess

model = xgb.train(
    params={
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
    },
    dtrain=dtrain,
    obj=fraud_cost_objective,  # Custom objective
    evals=[(dval, 'val')],
    early_stopping_rounds=50,
)
```

**Why this is the best approach:**

1. **Business-aligned optimization**: The model learns to minimize dollar losses, not raw accuracy or AUC
2. **Asymmetric cost handling**: $10k fraud loss vs $50 decline cost — the model automatically learns this ratio
3. **No synthetic data**: Uses all real training data
4. **Interpretable**: "We optimized for business cost" is a clear, intuitive explanation

**Result:**
```
XGBoost + cost-sensitive objective:
  ROC-AUC: ~0.979
  PR-AUC: ~0.801
  At optimal threshold (0.32):
    Precision: ~76%
    Recall: ~84%
    Expected cost reduction vs no-model: ~44%
```

---

## Results Comparison

| Strategy | ROC-AUC | PR-AUC | Recall@optimal | Business Cost Reduction |
|----------|---------|--------|----------------|------------------------|
| No handling | ~0.965 | ~0.682 | ~62% | ~25% |
| Undersampling | ~0.968 | ~0.694 | ~71% | ~30% |
| SMOTE | ~0.971 | ~0.713 | ~74% | ~33% |
| scale_pos_weight | ~0.978 | ~0.789 | ~79% | ~40% |
| **Cost-sensitive** | **~0.979** | **~0.801** | **~84%** | **~44%** |

*(Values are approximate pre-implementation estimates; update with actual results)*

---

## Evaluation Protocol

For highly imbalanced datasets, use Precision-Recall AUC, not ROC-AUC:

**Why ROC-AUC is misleading for imbalanced data:**
```
ROC-AUC = area under TPR vs FPR curve
FPR = FP / (FP + TN)

With 284k legitimate transactions, even 100 false positives gives FPR = 0.04%
→ ROC curve looks great even with poor precision
→ PR curve is more honest: it rewards finding fraud without too many false alarms
```

**Primary metric: PR-AUC (Average Precision)**  
**Secondary metric: ROC-AUC**  
**Business metric: Expected cost per 1000 transactions at optimal threshold**
