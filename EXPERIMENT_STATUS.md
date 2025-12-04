# Experiment Status

## Currently Running Experiments (100 trials each)

### Active Processes:
1. **Twitter** (PID: 144512) - `logs_twitter.txt`
2. **Reddit** (PID: 144943) - `logs_reddit.txt`
3. **Bluesky** (PID: 145119) - `logs_bluesky.txt`

All using:
- Model: OpenAI GPT-4o-mini
- Trials: 100 per prompt style
- Prompt styles: general, popular, engaging, informative, controversial, neutral
- Total trials: 600 per dataset (100 × 6 styles)

### Monitor Progress:
```bash
tail -f logs_twitter.txt
tail -f logs_reddit.txt
tail -f logs_bluesky.txt
```

### Check Running Processes:
```bash
ps aux | grep run_experiment.py | grep -v grep
```

## Regression Analysis Framework

### Purpose
Understand **confounding factors** in LLM selection bias:
- Does LLM prefer male authors? Or just tech topics (which men write more about)?
- Does LLM prefer formal writing? Or just older users (who write more formally)?
- Does LLM prefer positive sentiment? Or just certain topics (which tend to be positive)?

### How It Works
1. **Post-Level Data**: Track every post in every pool + whether it was selected
2. **Logistic Regression**: Model `P(selected | features, prompt_style)`
3. **Control for Confounders**: Include all features simultaneously to see independent effects
4. **Interpretation**: Coefficients show which features drive selection *after* controlling for others

### Files Created
- `run_experiment_with_tracking.py` - Enhanced experiment runner that saves post-level data
- `regression_analysis.py` - Regression analysis script with confounding analysis

### For Future Experiments
Use the tracking version to enable regression analysis:
```bash
python run_experiment_with_tracking.py \
  --dataset twitter \
  --provider openai \
  --model gpt-4o-mini \
  --n-trials 20
```

This saves:
- `post_level_data.pkl` - Full post-level data for regression
- `post_level_data.csv` - Human-readable format

Then analyze:
```bash
python regression_analysis.py \
  --results-dir outputs/experiments/twitter_openai_gpt-4o-mini \
  --dataset-name twitter
```

## Regression Analysis Features

### Content Features
- Sentiment (polarity, subjectivity, label)
- Topics (politics, sports, tech, etc.)
- Style (formality, emoji usage, word count)
- Polarization score

### Confounding Analysis
1. **Feature Correlations**: Which features are correlated (potential confounders)?
2. **Logistic Regression**: Which features independently predict selection?
3. **Model Comparison**:
   - Base model: Content features only
   - Full model: Content + prompt style
   - Test: Does prompt style matter after controlling for content?

### Outputs
- Coefficient estimates with p-values
- Odds ratios (multiplicative effect on selection odds)
- Confounding correlation matrix
- Model comparison statistics

## Example Research Questions

1. **Gender Bias**: If we see more male-authored content selected, is it because:
   - LLM directly prefers male authors? (gender bias)
   - Men write about topics LLM prefers? (topic confounding)
   - Men write in styles LLM prefers? (style confounding)

2. **Political Bias**: If we see more left/right content selected, is it because:
   - LLM prefers that political leaning? (political bias)
   - That leaning uses more positive/negative sentiment? (sentiment confounding)
   - That leaning discusses certain topics more? (topic confounding)

3. **Topic Bias**: If we see more tech content selected, is it because:
   - LLM prefers tech topics? (topic bias)
   - Tech content is more formal? (style confounding)
   - Tech content has less polarization? (polarization confounding)

## Next Steps

1. **Wait for experiments to complete** (~1-2 hours)
2. **Analyze current results**:
   ```bash
   python analyze_experiment.py --results-dir outputs/experiments/twitter_openai_gpt-4o-mini
   ```
3. **For future experiments with regression**:
   - Use `run_experiment_with_tracking.py`
   - Run `regression_analysis.py` on results
