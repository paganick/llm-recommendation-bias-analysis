# Quick Start: Multi-Trial Bias Analysis

## Why Multi-Trial Analysis?

As you correctly pointed out, **we need to randomize the pool and run multiple trials** to get robust statistical estimates. A single trial can give unreliable results due to random sampling variation.

The `run_multi_trial_analysis.py` script addresses this by:
- Running N trials (e.g., 10) with different random pools
- Getting recommendations from each pool
- Aggregating results across all trials
- Computing mean, std, min, max for each bias metric

## Basic Usage

```bash
# Run 10 trials with moderate sample size
python run_multi_trial_analysis.py \
  --provider openai \
  --model gpt-4o-mini \
  --num-trials 10 \
  --dataset-size 2000 \
  --pool-size 50 \
  --k 10 \
  --prompt-style popular
```

## Parameters

- `--num-trials`: Number of trials to run (default: 10)
- `--dataset-size`: Number of tweets to load (default: 2000)
- `--pool-size`: Size of each random pool (default: 50)
- `--k`: Number of recommendations per trial (default: 10)
- `--prompt-style`: popular, engaging, informative, controversial

## Example: Small Test Run

```bash
# Quick test: 5 trials, small pools
python run_multi_trial_analysis.py \
  --num-trials 5 \
  --dataset-size 1000 \
  --pool-size 30 \
  --k 10 \
  --prompt-style popular
```

Estimated cost: ~$0.001 (5 trials × ~400 tokens × gpt-4o-mini rates)

## Example: Robust Analysis

```bash
# Robust analysis: 20 trials, larger pools
python run_multi_trial_analysis.py \
  --num-trials 20 \
  --dataset-size 5000 \
  --pool-size 100 \
  --k 20 \
  --prompt-style popular
```

Estimated cost: ~$0.006 (20 trials × ~1500 tokens)

## Output

The script creates a directory with:
```
outputs/multi_trial/
  └── openai_gpt-4o-mini_popular_10trials_20251201_123456/
      ├── trial_01/
      │   ├── recommendations.csv
      │   ├── pool.csv
      │   └── bias_results.json
      ├── trial_02/
      │   └── ...
      ├── ...
      ├── trial_10/
      │   └── ...
      ├── aggregated_results.json    # ← Main results
      └── experiment_config.json
```

## Interpreting Results

The aggregated results show:

### Demographic Bias
```json
{
  "demo_aa": {
    "mean_bias": -2.3,      // Average across trials
    "std_bias": 1.5,        // Variability across trials
    "min_bias": -4.1,       // Most negative trial
    "max_bias": -0.5,       // Least negative trial
    "all_scores": [-2.1, -3.5, -1.8, ...]
  }
}
```

**Interpretation:**
- `mean_bias`: Average bias across all trials
- `std_bias`: How much the bias varies between trials (lower = more consistent)
- If |mean_bias| > 2 × std_bias, the bias is likely real (not just noise)

### Sentiment Bias
```json
{
  "mean_bias": 0.45,
  "std_bias": 0.12,
  "min_bias": 0.28,
  "max_bias": 0.63
}
```

**Interpretation:**
- Positive mean = bias toward positive sentiment
- Small std = consistent across trials

## Advantages Over Single Trial

| Aspect | Single Trial | Multi-Trial |
|--------|-------------|-------------|
| Reliability | Low (may be lucky/unlucky sample) | High (averaged over many samples) |
| Statistical Power | Weak | Strong |
| Variance Estimates | None | Yes (std_bias) |
| Confidence | Unknown | Can compute confidence intervals |
| Cost | ~$0.0001/trial | $0.001-0.01 total |

## Recommended Configurations

### Quick Exploration
- num_trials: 5
- dataset_size: 1000
- pool_size: 30
- k: 10
- Cost: ~$0.001

### Standard Analysis
- num_trials: 10
- dataset_size: 2000
- pool_size: 50
- k: 10
- Cost: ~$0.002

### Publication-Quality
- num_trials: 50
- dataset_size: 10000
- pool_size: 100
- k: 20
- Cost: ~$0.01-0.02

## Batch Experiments

To run multi-trial analysis with different prompt styles:

```bash
# Create custom batch script
for style in popular engaging informative controversial; do
  python run_multi_trial_analysis.py \
    --num-trials 10 \
    --prompt-style $style \
    --output-dir outputs/multitrial_batch_$style
done
```

## Statistical Significance

With multiple trials, you can:
1. Compute confidence intervals: `mean ± 1.96 × std / sqrt(num_trials)`
2. Test if bias is significantly different from zero
3. Compare different prompt styles or models

Example:
```python
import numpy as np
from scipy import stats

# Load aggregated results
bias_scores = aggregated_results['demographic_bias']['demo_aa']['all_scores']

# Compute 95% confidence interval
mean = np.mean(bias_scores)
se = np.std(bias_scores) / np.sqrt(len(bias_scores))
ci = (mean - 1.96*se, mean + 1.96*se)

# Test if significantly different from zero
t_stat, p_value = stats.ttest_1samp(bias_scores, 0)
```

## Next Steps

After running multi-trial analysis:

1. **Compare prompt styles**: Run with different `--prompt-style` values
2. **Compare models**: Run with different `--model` (gpt-4o vs gpt-4o-mini)
3. **Analyze trends**: Plot bias scores across trials
4. **Statistical testing**: Compare distributions between conditions
