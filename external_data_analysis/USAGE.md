# How to Run Complete Analysis on Survey Data

## Quick Start

Your collaborator needs to run just 3 commands to get all outputs:

### 1. Prepare Survey Data
```bash
cd external_data_analysis
python scripts/1_prepare_survey_data.py \
  --tweets path/to/tweet_data.csv \
  --users path/to/user_survey_data.csv \
  --output data/prepared_posts.csv
```

### 2. Run Recommendations
```bash
python scripts/2_run_recommendations.py \
  --input data/prepared_posts.csv \
  --provider gemini \
  --output outputs/experiments \
  --trials 100
```

*Supports multiple providers:* Run this command multiple times with different `--provider` (gemini, openai, anthropic) to compare models.

### 3. Run Complete Analysis
```bash
bash scripts/run_full_survey_analysis_simple.sh
```

**That's it!** All outputs will be in `outputs/analysis_full/`

## What Gets Generated

### Complete Output Structure
```
external_data_analysis/outputs/analysis_full/
├── visualizations/
│   ├── 1_distributions/          # One plot per feature (16 plots for core features + extras)
│   ├── 3_bias_heatmaps/          # 10 heatmaps
│   │   ├── disaggregated_prompt_*.png (6 heatmaps, one per prompt)
│   │   ├── aggregated_by_dataset.png
│   │   ├── aggregated_by_model.png
│   │   ├── aggregated_by_prompt.png
│   │   └── fully_aggregated.png
│   ├── 4_directional_bias/       # One plot per feature showing which categories favored
│   └── 5_feature_importance/     # 10 heatmaps (structure matches bias)
│       ├── disaggregated_prompt_*.png (6 heatmaps)
│       ├── aggregated_by_dataset.png
│       ├── aggregated_by_model.png
│       ├── aggregated_by_prompt.png
│       └── fully_aggregated.png
└── [CSV data files]
```

### Heatmap Details

#### 10 Bias Heatmaps
Show **magnitude** of bias for each feature:

1. **Disaggregated by prompt** (6 heatmaps):
   - One for each prompt style (general, popular, engaging, informative, controversial, neutral)
   - Rows = Features
   - Columns = Dataset × Model combinations
   - Values = Cramér's V (categorical) or |Cohen's d| (numerical)

2. **Aggregated by dataset** (1 heatmap):
   - Rows = Features
   - Columns = Datasets (e.g., if you have survey from multiple sources)
   - For single survey: will show 1 column

3. **Aggregated by model** (1 heatmap):
   - Rows = Features
   - Columns = Models/providers (gemini, openai, anthropic, etc.)
   - Shows which models have more bias per feature

4. **Aggregated by prompt** (1 heatmap):
   - Rows = Features
   - Columns = Prompt styles
   - Shows which prompts induce more bias

5. **Fully aggregated** (1 heatmap):
   - Single column showing overall bias per feature
   - Easiest way to see which features have most bias overall

#### 10 Feature Importance Heatmaps
Same structure as bias heatmaps, but showing **predictive importance** (SHAP values):
- Which features best predict whether a post gets recommended?
- Same 5 aggregation levels as bias heatmaps

## Multiple Models/Providers

The script automatically detects and analyzes ALL experiments in `outputs/experiments/survey_*/`:

```bash
# Run recommendations with different providers
python scripts/2_run_recommendations.py --provider gemini ...
python scripts/2_run_recommendations.py --provider openai ...
python scripts/2_run_recommendations.py --provider anthropic ...

# Analysis automatically processes all three
bash scripts/run_full_survey_analysis_simple.sh
```

The heatmaps will then show comparisons across all 3 models!

## Output Interpretation

### Bias Heatmaps (3_bias_heatmaps/)
**Color**: White (no bias) → Red (strong bias)

**Values**:
- Categorical features: Cramér's V (0-1)
- Numerical features: |Cohen's d|

**Interpretation**:
- < 0.10: Negligible bias
- 0.10-0.30: Small bias  
- 0.30-0.50: Medium bias
- > 0.50: Large bias

**Markers**:
- `*`: p < 0.05 and bias > 50th percentile
- `**`: p < 0.05 and bias > 60th percentile
- `***`: p < 0.05 and bias > 75th percentile

### Directional Bias (4_directional_bias/)
Shows WHICH categories are favored (not just magnitude)

**Colors**:
- Red: Over-represented in recommendations
- Blue: Under-represented in recommendations

**Example**: If gender bias heatmap shows 0.15 (small bias), look at directional plot to see if it's favoring males, females, or non-binary authors.

### Feature Importance (5_feature_importance/)
Shows which features drive recommendation decisions

**Values**: SHAP importance (normalized within each feature, 0-1)

**Interpretation**:
- High values = Feature strongly influences which posts get recommended
- Low values = Feature has little impact on recommendations

**Use case**: Focus analysis on top 5 most important features

## Comparing to Your Main Experiments

| Aspect | Your Main Pipeline | Survey Analysis |
|--------|-------------------|-----------------|
| Datasets | 3 (Twitter, Bluesky, Reddit) | 1 (Survey) - but can be multiple sources |
| Models | 3 (OpenAI, Anthropic, Gemini) | 1+ (whatever you run) |
| Features | 16 core features | Auto-detects ALL features in data |
| Heatmaps | 20 total (10 bias + 10 importance) | 20 total (same structure) |
| Disaggregation | Dataset × Model × Prompt | Model × Prompt (single dataset) |

The survey analysis produces **identical output structure** to your main experiments!

## Troubleshooting

### "No survey experiments found"
**Solution**: Run step 2 (recommendations) first to create experiment data

### Heatmaps show only 1 column
**Cause**: Only 1 model/provider analyzed
**Solution**: Run recommendations with multiple providers to enable cross-model comparison

### Analysis is slow
**Cause**: Feature importance computation (Random Forest + SHAP) is intensive
**Expected**: 2-5 minutes for typical survey data
**To speed up**: Results are cached - subsequent runs are instant

## Advanced: Custom Features

The analysis auto-detects features, but you can customize in `scripts/1_prepare_survey_data.py`:

```python
# Add custom feature extraction
df['custom_metric'] = your_function(df['text'])
FEATURE_TYPES['custom_metric'] = 'numerical'  # or 'categorical'
```

The heatmaps will automatically include your custom features!

---

**Questions?** See `COMPREHENSIVE_ANALYSIS.md` for detailed documentation.
