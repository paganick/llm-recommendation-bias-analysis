# Comprehensive Analysis Pipeline for Survey Data

This document describes the complete analysis pipeline that generates the same comprehensive outputs as the main project, adapted for external survey data.

## Overview

The pipeline automatically detects and analyzes all available features in your survey data, generating:
- **58 total visualizations** (from test data with 34 features)
- **3 comprehensive data files**
- **4 types of analysis** matching the main pipeline

## Key Differences from Main Pipeline

| Main Pipeline | Survey Pipeline |
|--------------|----------------|
| 3 datasets (Twitter, Bluesky, Reddit) | 1 dataset (Survey) |
| Fixed 16 features | **Auto-detects all features** |
| 54 conditions (3 datasets × 3 providers × 6 prompts) | 6 conditions (6 prompts) |
| ~50-60 plots | **Scalable** (34 features → 58 plots) |

## Running the Comprehensive Analysis

### Prerequisites
1. Run data preparation: `python scripts/1_prepare_survey_data.py`
2. Run recommendations: `python scripts/2_run_recommendations.py` (or use mock data)

### Execute Analysis
```bash
python scripts/run_comprehensive_survey_analysis.py
```

**Estimated runtime**: 2-5 minutes for 30-40 features

## Output Structure

```
external_data_analysis/outputs/analysis_comprehensive/
├── visualizations/
│   ├── 1_distributions/          # One plot per feature
│   │   ├── author_gender.png
│   │   ├── author_age.png
│   │   ├── text_length.png
│   │   └── ... (34 plots)
│   │
│   ├── 2_bias_heatmaps/          # Magnitude of bias
│   │   ├── bias_by_prompt.png    # Features × Prompt styles
│   │   └── overall_bias.png      # Overall bias ranking
│   │
│   ├── 3_directional_bias/       # Which categories are favored
│   │   ├── author_gender_directional.png
│   │   ├── author_race_directional.png
│   │   └── ... (21 plots for categorical/binary features)
│   │
│   └── 4_feature_importance/     # Importance for selection
│       └── overall_importance.png
│
├── bias_comprehensive.csv         # All bias metrics
├── directional_bias.csv          # Category-level bias
└── feature_importance.csv        # Feature rankings
```

## Analysis Types

### 1. Feature Distributions (1_distributions/)
**Purpose**: Compare pool vs recommended distributions for each feature

**Output**: One plot per feature
- **Categorical/Binary**: Side-by-side bar charts
- **Numerical**: Overlapping histograms with means

**Example insights**:
- "Female authors appear 15% more in recommended vs pool"
- "Recommended posts are 20 characters longer on average"

### 2. Bias Heatmaps (2_bias_heatmaps/)
**Purpose**: Visualize bias magnitude across features and conditions

**Outputs**:
- `bias_by_prompt.png`: Features × Prompt styles heatmap
- `overall_bias.png`: Features ranked by overall bias

**Metrics**:
- Cramér's V for categorical/binary features
- |Cohen's d| for numerical features

**Color coding**:
- Red (significant p<0.05) vs Blue (non-significant)
- Darker = stronger bias

**Example insights**:
- "Education shows highest bias (V=0.087)"
- "Neutral prompts show more gender bias than engaging prompts"

### 3. Directional Bias (3_directional_bias/)
**Purpose**: Show WHICH categories/values are over/under-represented

**Output**: One heatmap per categorical/binary feature
- Rows = Categories (e.g., male, female, non-binary)
- Columns = Prompt styles
- Values = Recommended % - Pool %

**Color scheme**:
- Red = Over-represented in recommendations
- Blue = Under-represented in recommendations

**Example insights**:
- "College-educated authors: +12% in recommendations"
- "Conservative ideology: -8% in informative prompts"

### 4. Feature Importance (4_feature_importance/)
**Purpose**: Rank features by association with selection

**Output**: Single bar chart ranking all features

**Method**:
- Categorical/Binary: Cramér's V with selection
- Numerical: Absolute correlation with selection

**Example insights**:
- "Engagement score is most predictive of selection"
- "Author age has minimal impact on recommendations"

## Auto-Detection of Features

The pipeline automatically categorizes features:

### Categorical Features
- Starts with `author_` and non-numerical (e.g., `author_race`, `author_education`)
- Specific columns: `primary_topic`, `controversy_level`, `sentiment_label`

### Binary Features (0/1 or True/False)
- Starts with `has_` (e.g., `has_emoji`, `has_url`)
- Starts with `is_` (e.g., `is_reply`, `is_retweet`)
- `user_verified`

### Numerical Features
- Continuous values: `text_length`, `sentiment_polarity`, `engagement_score`
- Count data: `user_followers_count`, `word_count`
- Age/time: `author_age`, `user_account_age_days`

**Flexibility**: The pipeline works with ANY features in your data - just ensure they follow these naming conventions or add them to the appropriate category in the script.

## Data Files

### bias_comprehensive.csv
Columns:
- `feature`: Feature name
- `prompt_style`: Prompt style (or "Overall")
- `bias`: Cramér's V or |Cohen's d|
- `p_value`: Statistical significance (where applicable)
- `metric`: Type of metric used
- `feature_type`: categorical/binary/numerical

### directional_bias.csv
Columns:
- `feature`: Feature name
- `prompt_style`: Prompt style
- `category`: Category value (e.g., "male", "college")
- `pool_prop`: Proportion in pool
- `rec_prop`: Proportion in recommendations
- `difference`: rec_prop - pool_prop

### feature_importance.csv
Columns:
- `feature`: Feature name
- `importance`: Association strength with selection
- `feature_type`: categorical/binary/numerical

## Interpretation Guide

### Bias Magnitude (Cramér's V / Cohen's d)
- **0.00-0.10**: Negligible
- **0.10-0.30**: Small
- **0.30-0.50**: Medium
- **0.50+**: Large

### Directional Bias (% difference)
- **-10% to +10%**: Minor over/under-representation
- **±10% to ±20%**: Moderate bias
- **±20%+**: Strong bias

### Statistical Significance
- **p < 0.05**: Statistically significant (marked with *)
- **p ≥ 0.05**: Not statistically significant

## Customization

### Adding Custom Features
Edit `run_comprehensive_survey_analysis.py`:

```python
# Add to categorical_features list
categorical_features.append('my_custom_category')

# Or add to numerical_features
numerical_features.append('my_custom_score')
```

### Changing Color Schemes
All heatmaps use:
- `cmap='Reds'` for magnitude (white to red)
- `cmap='RdBu_r'` for directional (blue to red, centered at 0)

Modify in the plotting functions if desired.

### Filtering Features
To exclude certain features, add them to the skip list:

```python
skip_features = ['tweet_id', 'user_id', 'text', 'persona']
for col in df.columns:
    if col in skip_features:
        continue
    # ... feature detection logic
```

## Troubleshooting

### Issue: "No recommended data" warning
**Cause**: Feature has no values in selected posts
**Solution**: This is normal for rare features (e.g., `is_retweet` if no retweets were selected)

### Issue: Empty directional bias plots
**Cause**: Feature has only 1 unique value
**Solution**: Feature will be skipped automatically

### Issue: Very small bias values
**Cause**: Small sample size or genuinely unbiased recommendations
**Solution**: Increase number of trials in recommendation step for more statistical power

## Example Workflow

1. **Prepare Data** (once):
   ```bash
   python scripts/1_prepare_survey_data.py \
     --tweets data/tweets.csv \
     --users data/users.csv \
     --output data/prepared.csv
   ```

2. **Run Recommendations** (per model/provider):
   ```bash
   python scripts/2_run_recommendations.py \
     --input data/prepared.csv \
     --provider gemini \
     --output outputs/experiments \
     --trials 100
   ```

3. **Comprehensive Analysis** (once per experiment):
   ```bash
   python scripts/run_comprehensive_survey_analysis.py
   ```

4. **Review Results**:
   - Browse `outputs/analysis_comprehensive/visualizations/`
   - Examine CSV files for quantitative analysis
   - Focus on features with highest bias in `2_bias_heatmaps/overall_bias.png`

## Comparison to Simple Analysis

The simple analysis (`scripts/run_survey_analysis.py`) provides:
- ✓ Quick bias summary
- ✓ 3 visualization plots
- ✓ Fast runtime (~10 seconds)

The comprehensive analysis adds:
- ✓ Per-feature distributions
- ✓ Directional bias (which categories favored)
- ✓ Bias variation by prompt style
- ✓ Feature importance ranking
- ✓ Publication-ready plots

**Recommendation**: Use simple analysis for quick checks, comprehensive analysis for final results and publications.

---

*Generated from test data: 20 posts, 34 features, 6 prompt styles*  
*Output: 58 plots + 3 CSV files*  
*Runtime: ~2 minutes*
