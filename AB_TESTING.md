# A/B Testing Framework — Fraud Detection

Champion/challenger model comparison with statistical rigor and safe rollout.

---

## Overview

The A/B testing framework allows testing a new (challenger) model against the current production (champion) model without risking all traffic on the new model.

```
Incoming request
       │
       ▼
A/B Router: hash(transaction_id) % 100
       │
    ┌──┴──┐
    │     │
   <10   >=10
    │     │
    ▼     ▼
Challenger  Champion
 (10%)      (90%)
    │     │
    └──┬──┘
       │
       ▼
Both predictions logged to database
       │
       ▼
Decision made by routed model
```

---

## Traffic Routing

### Deterministic Routing by Transaction ID

Using a hash ensures the same transaction always goes to the same model. This prevents a transaction from being evaluated by both models (which would skew feedback attribution):

```python
import hashlib

class ABRouter:
    def __init__(
        self,
        champion_model: FraudModel,
        challenger_model: FraudModel | None,
        challenger_traffic_pct: float = 0.10,
    ):
        self.champion = champion_model
        self.challenger = challenger_model
        self.challenger_pct = challenger_traffic_pct
    
    def get_model(self, transaction_id: str) -> tuple[FraudModel, str]:
        """
        Returns (model, variant) for a given transaction.
        Variant: 'champion' or 'challenger'
        """
        if self.challenger is None:
            return self.champion, 'champion'
        
        # Deterministic hash: same transaction always goes to same model
        hash_val = int(hashlib.md5(transaction_id.encode()).hexdigest(), 16) % 100
        
        if hash_val < (self.challenger_pct * 100):
            return self.challenger, 'challenger'
        return self.champion, 'champion'
    
    def update_challenger_pct(self, new_pct: float):
        """Gradually increase challenger traffic as confidence grows."""
        self.challenger_pct = new_pct
```

### Traffic Ramp Schedule

```
Day 1:   1% challenger  — Verify no major issues
Day 3:  10% challenger  — Collect initial data
Day 7:  25% challenger  — If stable, increase
Day 14: 50% challenger  — Nearing decision point
Day 21: Decision        — Promote or rollback
```

---

## Metrics Tracked

For each model version, track:

```python
@dataclass
class ModelMetrics:
    version: str
    variant: str  # 'champion' or 'challenger'
    
    # Volume
    total_predictions: int
    
    # Score distribution
    mean_score: float
    p50_score: float
    p95_score: float
    flag_rate: float  # % of transactions flagged
    
    # Performance (requires confirmed feedback)
    confirmed_tp: int   # Confirmed fraud caught
    confirmed_fp: int   # Confirmed legitimate flagged
    confirmed_fn: int   # Confirmed fraud missed
    
    # Derived metrics (once enough feedback)
    precision: float    # TP / (TP + FP)
    recall: float       # TP / (TP + FN)
    f1: float
    
    # Latency
    p50_latency_ms: float
    p95_latency_ms: float
    
    # Business metrics
    estimated_cost_per_1000: float
    approval_rate: float
```

---

## Statistical Significance Testing

### When to Decide

Do not make a decision until:
1. Minimum sample size reached: 10,000 predictions per variant
2. Minimum confirmed fraud cases: 50 confirmed fraud labels per variant
3. Experiment duration: at least 7 days (captures weekly patterns)

### Hypothesis Test

**Null hypothesis (H0):** Challenger has the same or worse business cost per 1000 transactions as champion.

**Alternative hypothesis (H1):** Challenger has significantly better (lower) business cost.

```python
from scipy import stats
import numpy as np

def test_significance(
    champion_costs: list[float],
    challenger_costs: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    One-tailed Welch's t-test:
    Tests if challenger cost < champion cost (one-tailed)
    
    Args:
        champion_costs: List of per-transaction costs for champion (0 or fraud_cost/decline_cost)
        challenger_costs: List of per-transaction costs for challenger
        alpha: Significance level (default 0.05)
    
    Returns:
        dict with test results and recommendation
    """
    n_champion = len(champion_costs)
    n_challenger = len(challenger_costs)
    
    mean_champion = np.mean(champion_costs)
    mean_challenger = np.mean(challenger_costs)
    
    # Two-sample Welch's t-test (one-tailed: challenger < champion)
    t_stat, p_value_two_tailed = stats.ttest_ind(
        challenger_costs,
        champion_costs,
        equal_var=False  # Welch's t-test (doesn't assume equal variance)
    )
    
    # One-tailed p-value (we care about challenger being BETTER, i.e., lower cost)
    p_value = p_value_two_tailed / 2 if t_stat < 0 else 1.0
    
    improvement_pct = (mean_champion - mean_challenger) / mean_champion * 100
    
    # Confidence interval for the difference
    se = np.sqrt(np.var(champion_costs) / n_champion + np.var(challenger_costs) / n_challenger)
    ci_lower = (mean_champion - mean_challenger) - 1.96 * se
    ci_upper = (mean_champion - mean_challenger) + 1.96 * se
    
    is_significant = p_value < alpha
    is_practically_significant = improvement_pct > 5  # Minimum 5% improvement
    
    return {
        'n_champion': n_champion,
        'n_challenger': n_challenger,
        'mean_champion': mean_champion,
        'mean_challenger': mean_challenger,
        'improvement_pct': improvement_pct,
        'p_value': p_value,
        'is_statistically_significant': is_significant,
        'is_practically_significant': is_practically_significant,
        'confidence_interval_95': (ci_lower, ci_upper),
        'recommendation': _get_recommendation(is_significant, is_practically_significant, improvement_pct),
    }

def _get_recommendation(is_stat_sig: bool, is_prac_sig: bool, improvement_pct: float) -> str:
    if is_stat_sig and is_prac_sig:
        return f"PROMOTE challenger — statistically and practically significant improvement ({improvement_pct:.1f}%)"
    elif is_stat_sig and not is_prac_sig:
        return f"INCONCLUSIVE — statistically significant but improvement ({improvement_pct:.1f}%) below practical threshold (5%)"
    elif improvement_pct < -5:
        return f"ROLLBACK challenger — significantly worse than champion ({-improvement_pct:.1f}% higher cost)"
    else:
        return f"KEEP CHAMPION — challenger not significantly better"
```

### Sample Size Calculation

Before starting the experiment, calculate how many samples are needed to detect a meaningful improvement:

```python
from statsmodels.stats.power import TTestIndPower

def calculate_required_sample_size(
    expected_improvement_pct: float = 0.10,  # 10% cost reduction
    current_cost_per_1000: float = 15.0,     # Current expected cost
    std_dev: float = 5.0,                     # Estimated cost std dev
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    effect_size_dollars = current_cost_per_1000 * expected_improvement_pct
    cohen_d = effect_size_dollars / std_dev
    
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=cohen_d,
        alpha=alpha,
        power=power,
        alternative='smaller'  # One-tailed
    )
    
    return int(np.ceil(n))

# With 10% improvement target and 80% power:
# n = ~3,000 samples per variant (for confirmed fraud feedback)
# At 0.172% fraud rate: need ~1.75M total predictions per variant
# → Time estimate: depends on transaction volume
```

---

## Decision Criteria

### Promote Challenger

Promote challenger to champion when ALL conditions are met:

| Condition | Requirement |
|-----------|-------------|
| Statistical significance | p-value < 0.05 (one-tailed) |
| Practical significance | ≥ 5% cost improvement |
| Minimum duration | ≥ 7 days |
| Minimum sample | ≥ 10,000 predictions per variant |
| Latency | Challenger p95 latency ≤ champion + 20ms |
| No regressions | Recall ≥ 80% |

### Keep Champion

Keep champion when:
- Challenger shows no improvement after 21 days
- Challenger is statistically worse
- Challenger has higher latency without compensating benefit

### Rollback Challenger

Immediately rollback challenger if:
- Recall drops below 70% (immediate)
- Latency > 300ms p95 (immediate)
- Flag rate is 3x normal (likely model bug)

---

## MLflow Integration

```python
# Tag models for A/B assignment
client = mlflow.MlflowClient()

# Promote challenger to staging (starts A/B test)
client.transition_model_version_stage(
    name="fraud_detector",
    version=challenger_version,
    stage="Staging"
)

# When test concludes — promote challenger to production
def promote_challenger(challenger_version: str, champion_version: str):
    client.transition_model_version_stage(
        name="fraud_detector",
        version=challenger_version,
        stage="Production"
    )
    client.transition_model_version_stage(
        name="fraud_detector",
        version=champion_version,
        stage="Archived"
    )
    client.set_model_version_tag(
        name="fraud_detector",
        version=challenger_version,
        key="promoted_at",
        value=datetime.now().isoformat()
    )
```

---

## Rollback Strategy

If a bad challenger is deployed:

```bash
# 1. Immediate: Set challenger traffic to 0%
# (In production: update env var CHALLENGER_TRAFFIC_PCT=0 + hot reload)

# 2. Via MLflow: Demote challenger from Staging
mlflow models transition \
  --model-name fraud_detector \
  --version <challenger_version> \
  --stage None

# 3. API auto-falls back to champion (no challenger in Staging)
# Next request: router returns champion for all traffic

# 4. Post-mortem: Analyze what went wrong with challenger
```

Total rollback time: < 1 minute (env var update, no redeployment needed).
