# LLM Recommendation Bias Analysis Pipeline

Comprehensive analysis pipeline for evaluating bias in LLM recommendation systems across 16 features, 3 datasets, 3 providers, and 6 prompt styles.

## Overview

This pipeline analyzes bias in LLM-based content recommendation systems by comparing feature distributions between the full post pool and recommended posts. It generates multiple types of visualizations and statistical analyses to identify and quantify bias patterns.

## Features Analyzed (16 Total)

**Author Features (3)**
- `author_gender`, `author_political_leaning`, `author_is_minority`

**Text Metrics (2)**
- `text_length`, `avg_word_length`

**Sentiment (2)**
- `sentiment_polarity`, `sentiment_subjectivity`

**Style Features (4)**
- `has_emoji`, `has_hashtag`, `has_mention`, `has_url`

**Content Features (3)**
- `polarization_score`, `controversy_level`, `primary_topic`

**Toxicity Features (2)**
- `toxicity`, `severe_toxicity`

## Experimental Design

- **Datasets**: Twitter, Bluesky, Reddit
- **LLM Providers**: OpenAI, Anthropic, Gemini
- **Prompt Styles**: general, popular, engaging, informative, controversial, neutral
- **Sample Size**: 10,000 posts per prompt style (1,000 selected, 9,000 non-selected)
- **Total Conditions**: 54 (3 datasets × 3 providers × 6 prompts)

## Output Structure

```
analysis_outputs/
├── feature_importance_data.csv          # Cached feature importance results
├── pool_vs_recommended_summary.csv      # Bias metrics for all conditions
└── visualizations/
    ├── 1_distributions/                 # Feature distributions (16 plots)
    ├── 2_bias_heatmaps/                 # Bias magnitude heatmaps (10 plots)
    ├── 3_directional_bias/              # Directional bias plots (16 plots)
    └── 4_feature_importance/            # Feature importance heatmaps (10 plots)
```

## Analysis Types

### 1. Feature Distributions (16 plots)
Shows the distribution of each feature in the pool data.

### 2. Bias Heatmaps (10 plots)
Normalized bias magnitude across features and conditions.
- **Metric**: Cramér's V (categorical), Cohen's d (continuous)
- **Normalization**: Within-feature min-max scaling [0, 1]
- **Significance**: `*` p<0.05 >50%, `**` >60%, `***` >75%

### 3. Directional Bias (16 plots)
Shows which categories/values are favored in recommendations.
- Categorical: proportion_recommended - proportion_pool
- Continuous: mean_recommended - mean_pool

### 4. Feature Importance (10 heatmaps)
Random Forest + SHAP analysis predicting recommendations.
- Mean AUROC: 0.870 (range: 0.81-0.95)
- Best: Twitter × Anthropic × Controversial (0.95)

## Usage

```bash
# Run full analysis
python3 run_comprehensive_analysis.py

# Runtime: First run ~30-40 min, subsequent ~2-3 min (uses cached data)

# Force recomputation:
rm analysis_outputs/feature_importance_data.csv
```

## Recent Updates (Dec 23, 2024)

- Fixed feature importance: Now includes all 16 features (was only 5)
- Fixed SHAP storage: Scalars instead of string arrays
- Added caching for feature importance results
- Updated all colormaps to 'Reds' (white-to-red)
- Reorganized output folders (1-4)
- Mean AUROC improved to 0.870

## Dependencies

`pandas numpy matplotlib seaborn scipy scikit-learn shap`

## Key Findings

### Bias Normalization
- Within-feature min-max scaling: values comparable within a feature, not across
- Consistent moderate bias → high normalized values
- Wide range with outliers → lower mean normalized values

### Directional vs Magnitude Bias  
- **Magnitude**: How strong is the association? (Cramér's V, Cohen's d)
- **Directional**: Which categories are favored?
- High magnitude doesn't always mean different directional patterns

See documentation in script header for detailed methodology.
