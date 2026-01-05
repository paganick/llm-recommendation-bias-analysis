# LLM Recommendation Bias Analysis Pipeline

Comprehensive analysis pipeline for evaluating bias in LLM recommendation systems across 16 features, 3 datasets, 3 providers, and 6 prompt styles.

## Overview

This pipeline analyzes bias in LLM-based content recommendation systems by comparing feature distributions between the full post pool and recommended posts. It generates multiple types of visualizations and statistical analyses to identify and quantify bias patterns.

## Features Analyzed (16 Total)

### Author Features (3)
- `author_gender` - Author's gender (categorical)
- `author_political_leaning` - Political orientation (categorical, 7 categories)
- `author_is_minority` - Minority group membership (categorical)

### Text Metrics (2)
- `text_length` - Length of post text (numerical)
- `avg_word_length` - Average word length (numerical)

### Sentiment (2)
- `sentiment_polarity` - Sentiment polarity score (numerical)
- `sentiment_subjectivity` - Sentiment subjectivity score (numerical)

### Style Features (4)
- `has_emoji` - Contains emoji (binary)
- `has_hashtag` - Contains hashtag (binary)
- `has_mention` - Contains user mention (binary)
- `has_url` - Contains URL (binary)

### Content Features (3)
- `polarization_score` - Political polarization score (numerical)
- `controversy_level` - Controversy level (categorical)
- `primary_topic` - Main topic of post (categorical)

### Toxicity Features (2)
- `toxicity` - Toxicity score (numerical)
- `severe_toxicity` - Severe toxicity score (numerical)

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
Shows the distribution of each feature in the pool data across all datasets.

### 2. Bias Heatmaps (10 plots)
Visualizes normalized bias magnitude across features and conditions.

**Normalization**: Bias values normalized within each feature using min-max scaling [0, 1]
**Metric**: Cramér's V (categorical), Cohen's d (continuous)
**Significance**: `*` p<0.05 >50%, `**` >60%, `***` >75%

### 3. Directional Bias Plots (16 plots)
Shows which categories/values are favored in recommendations.
- Categorical: proportion_recommended - proportion_pool
- Continuous: mean_recommended - mean_pool

### 4. Feature Importance (10 heatmaps)
Random Forest + SHAP analysis showing which features predict recommendations.
- Mean AUROC: 0.870 (range: 0.81-0.95)
- Best: Twitter × Anthropic × Controversial (0.95)

## Usage

```bash
# Run full analysis
python3 run_comprehensive_analysis.py

# Runtime: 
# - First run: ~30-40 minutes (with feature importance computation)
# - Subsequent runs: ~2-3 minutes (loads cached data)

# Force recomputation:
rm analysis_outputs/feature_importance_data.csv
python3 run_comprehensive_analysis.py
```

## Recent Updates (December 23, 2024)

- Fixed feature importance bug: Now includes all 16 features (was only 5)
- Fixed SHAP value storage: Scalars instead of string arrays
- Added feature importance caching
- Updated colormaps to 'Reds' (white-to-red)
- Reorganized output folders (1-4)

## Dependencies

pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, shap

## File Structure

```
llm-recommendation-bias-analysis/
├── run_comprehensive_analysis.py     # Main analysis script
├── README.md                          # This file
├── outputs/experiments/               # Raw experiment data
├── analysis_outputs/                  # Analysis results
└── .backup_*/                         # Backup folders
```
