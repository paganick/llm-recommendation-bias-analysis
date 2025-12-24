# How to Run Complete Analysis on Survey Data

> **Note**: This repository includes fake/test data for demonstration purposes. Follow the instructions below to analyze your real survey data.

## Quick Start

Run just 3 commands to get complete analysis with 87 visualizations:

### 1. Prepare Survey Data
```bash
cd external_data_analysis
python scripts/1_prepare_survey_data.py \
  --tweets path/to/tweet_data.csv \
  --users path/to/user_survey_data.csv \
  --output data/prepared_posts.csv
```

**Replace paths** with your actual survey data files.

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
bash scripts/run_full_survey_analysis.sh
```

**That's it!** All outputs will be in `outputs/analysis_full/`

## What Gets Generated

### Complete Output Structure (87 Visualizations)
```
external_data_analysis/outputs/analysis_full/
├── visualizations/
│   ├── 1_distributions/          # Distribution plots (pool vs selected)
│   │                             # One plot per feature (34 in test data)
│   ├── 2_bias_heatmaps/          # 10 bias heatmaps
│   │   ├── disaggregated_prompt_*.png (6 heatmaps, one per prompt)
│   │   ├── aggregated_by_dataset.png
│   │   ├── aggregated_by_model.png
│   │   ├── aggregated_by_prompt.png
│   │   └── fully_aggregated.png
│   ├── 3_directional_bias/       # Directional bias heatmaps
│   │                             # One per feature showing which categories are over/under-represented
│   └── 4_feature_importance/     # 10 feature importance heatmaps
│       ├── disaggregated_prompt_*.png (6 heatmaps)
│       ├── aggregated_by_dataset.png
│       ├── aggregated_by_model.png
│       ├── aggregated_by_prompt.png
│       └── fully_aggregated.png
```

### Visualization Details

#### Distribution Plots (1_distributions/)
- **What**: Side-by-side comparison of feature distributions (Pool vs Selected)
- **Format**: Bar charts for categorical, histograms for numerical
- **Count**: One plot per feature auto-detected from your data

#### 10 Bias Heatmaps (2_bias_heatmaps/)
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

#### Directional Bias Heatmaps (3_directional_bias/)
- **What**: Shows WHICH categories/values are favored (not just magnitude)
- **Format**: Heatmaps with rows = categories, columns = prompt styles
- **Colors**:
  - Red/Positive = Over-represented in recommendations
  - Blue/Negative = Under-represented in recommendations
  - White/Zero = No directional bias
- **Example**: If gender shows bias of 0.15, directional plot reveals if it favors males, females, or non-binary

#### 10 Feature Importance Heatmaps (4_feature_importance/)
Same structure as bias heatmaps, but showing **predictive importance**:
- Which features best predict whether a post gets recommended?
- Same 5 aggregation levels as bias heatmaps
- Helps focus analysis on most influential features

## Multiple Models/Providers

The script automatically detects and analyzes ALL experiments in `outputs/experiments/survey_*/`:

```bash
# Run recommendations with different providers
python scripts/2_run_recommendations.py --provider gemini ...
python scripts/2_run_recommendations.py --provider openai ...
python scripts/2_run_recommendations.py --provider anthropic ...

# Analysis automatically processes all three
bash scripts/run_full_survey_analysis.sh
```

The heatmaps will then show comparisons across all 3 models!

## Output Interpretation

### Bias Heatmaps (2_bias_heatmaps/)
**Color**: White (no bias) → Red (strong bias)

**Values**:
- Categorical features: Cramér's V (0-1)
- Numerical features: |Cohen's d|

**Interpretation**:
- < 0.10: Negligible bias
- 0.10-0.30: Small bias
- 0.30-0.50: Medium bias
- > 0.50: Large bias

### Directional Bias (3_directional_bias/)
**Values**: Difference in proportions/means (Selected - Pool)

**Interpretation**:
- Positive (red): Category/value is over-represented in recommendations
- Negative (blue): Category/value is under-represented
- Zero (white): No directional bias

### Feature Importance (4_feature_importance/)
Shows which features drive recommendation decisions

**Values**: Statistical importance measures

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
| Visualizations | ~87 total | ~87 total (matching structure) |
| Disaggregation | Dataset × Model × Prompt | Model × Prompt (single dataset) |

The survey analysis produces **identical output structure** to your main experiments!

## Test/Fake Data

The repository includes test data for demonstration:
- **Location**: `outputs/experiments/survey_gemini_gemini-2.0-flash/`
- **Status**: ⚠️ **FAKE DATA** - generated for testing only
- **Purpose**: Demonstrates the pipeline and expected output structure
- **Usage**: Replace with your real survey data to get actual results

Test outputs in `outputs/analysis_full/` are also based on fake data.

## Troubleshooting

### "No survey experiments found"
**Solution**: Run step 2 (recommendations) first to create experiment data

### Heatmaps show only 1 column
**Cause**: Only 1 model/provider analyzed
**Solution**: Run recommendations with multiple providers to enable cross-model comparison

### Fewer features than expected
**Check**: Ensure your survey data has all expected columns
**Note**: Script auto-detects features - if columns are missing, they won't appear

### Analysis is slow
**Expected**: 2-5 minutes for typical survey data
**First run**: Computes all metrics from scratch
**Subsequent runs**: Previous directory is cleaned, so each run takes the same time

## Advanced: Custom Features

The analysis auto-detects features, but you can customize in `scripts/1_prepare_survey_data.py`:

```python
# Add custom feature extraction
df['custom_metric'] = your_function(df['text'])
FEATURE_TYPES['custom_metric'] = 'numerical'  # or 'categorical' or 'binary'
```

The analysis will automatically include your custom features in all visualizations!

---

**Questions?** See `COMPREHENSIVE_ANALYSIS.md` for detailed documentation.
