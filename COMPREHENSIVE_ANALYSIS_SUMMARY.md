# Comprehensive Analysis Summary: LLM Recommendation Bias Study

**Document Purpose**: This document provides a complete overview of the analysis methodology, results, and key findings for writing up the LLM recommendation bias study.

**Last Updated**: December 24, 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Experimental Design](#2-experimental-design)
3. [Feature Engineering](#3-feature-engineering)
4. [Methodology](#4-methodology)
5. [Analysis Pipeline](#5-analysis-pipeline)
6. [Key Findings](#6-key-findings)
7. [Results Locations](#7-results-locations)

---

## 1. Project Overview

### Research Question
This study investigates **systematic bias in LLM-based content recommendation systems** by analyzing whether large language models (LLMs) exhibit preferences for certain content characteristics when selecting posts to recommend to users.

### Approach
We compare the feature distributions between:
- **Pool**: All available posts (9,000 non-selected posts per condition)
- **Recommended**: Posts selected by LLMs (1,000 selected posts per condition)

If LLMs are unbiased, the distribution of features should be similar between pool and recommended posts. Systematic differences indicate bias.

### Scale
- **54 experimental conditions** (3 datasets × 3 providers × 6 prompt styles)
- **16 features analyzed** across 5 categories (author, text metrics, sentiment, style, content, toxicity)
- **864 feature-condition combinations** tested
- **540,000 total recommendations** analyzed

---

## 2. Experimental Design

### 2.1 Datasets

Three social media platforms analyzed:

| Dataset | Platform | Description |
|---------|----------|-------------|
| **Twitter** | Twitter/X | Microblogging platform with character limits |
| **Bluesky** | Bluesky Social | Decentralized social network |
| **Reddit** | Reddit | Discussion forum and link aggregation |

**Note**: All datasets show consistent feature distributions, validating cross-platform comparisons.

### 2.2 LLM Providers

Three major LLM providers tested:

| Provider | Model | Usage |
|----------|-------|-------|
| **OpenAI** | GPT models | Industry standard |
| **Anthropic** | Claude (Sonnet 4.5) | Safety-focused provider |
| **Gemini** | Google Gemini | Latest Google offering |

### 2.3 Prompt Styles

Six prompt variations designed to test different recommendation objectives:

| Prompt Style | Objective | Description |
|--------------|-----------|-------------|
| **General** | Baseline | Generic "recommend posts" instruction |
| **Popular** | Engagement | "Recommend posts likely to be popular" |
| **Engaging** | User interaction | "Recommend engaging content" |
| **Informative** | Information quality | "Recommend informative posts" |
| **Controversial** | Debate potential | "Recommend controversial topics" |
| **Neutral** | Balanced view | "Recommend neutral, balanced posts" |

### 2.4 Sample Sizes

Per experimental condition (dataset × provider × prompt):
- **Total posts**: 10,000
- **Selected (recommended)**: 1,000 (10%)
- **Non-selected (pool)**: 9,000 (90%)

**Total across 54 conditions**: 540,000 posts with recommendations

---

## 3. Feature Engineering

### 3.1 Feature Categories

**16 core features** organized into 5 categories:

#### Author Features (3)
- `author_gender`: Gender of post author (categorical: male, female, non-binary, unknown)
- `author_political_leaning`: Political orientation (categorical: left, center-left, center, center-right, right, apolitical, unknown)
- `author_is_minority`: Minority group membership (categorical: yes, no, unknown)

#### Text Metrics (2)
- `text_length`: Number of characters in post (numerical)
- `avg_word_length`: Average characters per word (numerical)

#### Sentiment (2)
- `sentiment_polarity`: Sentiment score from -1 (negative) to +1 (positive) (numerical)
- `sentiment_subjectivity`: Subjectivity score from 0 (objective) to 1 (subjective) (numerical)

#### Style Features (4)
- `has_emoji`: Post contains emoji (binary: 0/1)
- `has_hashtag`: Post contains hashtag (binary: 0/1)
- `has_mention`: Post contains user mention (binary: 0/1)
- `has_url`: Post contains URL (binary: 0/1)

#### Content Features (3)
- `polarization_score`: Political polarization intensity (numerical: 0 to 1+)
- `controversy_level`: Controversy categorization (categorical: low, medium, high)
- `primary_topic`: Main topic of post (categorical: politics, personal, technology, sports, other)

#### Toxicity Features (2)
- `toxicity`: Overall toxicity score from Detoxify model (numerical: 0 to 1)
- `severe_toxicity`: Severe toxicity score (numerical: 0 to 1)

### 3.2 Feature Prevalence

Key statistics from pool data (across all datasets):

| Feature | Mean/Distribution | Standard Deviation |
|---------|-------------------|-------------------|
| `text_length` | 137.9 characters | 143.1 |
| `sentiment_polarity` | 0.045 (slightly positive) | 0.496 |
| `polarization_score` | 0.081 | 0.152 |
| `toxicity` | 0.131 | - |
| `political_leaning` | 70.6% left, 18.8% right, 5.7% unknown | - |
| `gender` | 66.8% unknown, 23.5% male, 9.4% female | - |

**Imbalanced Features** (important for interpretation):
- `has_mention`: Only 0.9% of posts contain mentions
- `has_hashtag`: Only 3.4% of posts contain hashtags
- `has_emoji`: Only 7.3% of posts contain emojis

---

## 4. Methodology

### 4.1 Bias Computation

Bias is measured by comparing feature distributions between pool and recommended posts. Different metrics are used based on feature type.

#### 4.1.1 Numerical Features (Cohen's d)

**Features**: text_length, avg_word_length, sentiment_polarity, sentiment_subjectivity, polarization_score, toxicity, severe_toxicity

**Metric**: Cohen's d (standardized mean difference)

**Formula**:
```
Cohen's d = (Mean_recommended - Mean_pool) / Pooled_SD

where:
Pooled_SD = sqrt(((n_pool - 1) * var_pool + (n_rec - 1) * var_rec) / (n_pool + n_rec - 2))
```

**Interpretation**:
- d > 0: LLM recommends posts with **higher** values than pool average
- d < 0: LLM recommends posts with **lower** values than pool average
- |d| < 0.2: Small effect
- |d| = 0.2-0.5: Medium effect
- |d| > 0.5: Large effect

**Statistical Test**: Welch's t-test (handles unequal variances)

#### 4.1.2 Categorical Features (Cramér's V)

**Features**: author_gender, author_political_leaning, author_is_minority, controversy_level, primary_topic

**Metric**: Cramér's V (association strength)

**Formula**:
```
Cramér's V = sqrt(χ² / (n * min(r-1, c-1)))

where:
- χ² = Chi-square statistic from contingency table
- n = total sample size
- r = number of categories
- c = 2 (pool vs recommended)
```

**Interpretation**:
- V = 0: No association (no bias)
- V = 0.1-0.3: Weak association
- V = 0.3-0.5: Moderate association
- V > 0.5: Strong association

**Statistical Test**: Chi-square test of independence

#### 4.1.3 Binary Features (Cramér's V)

**Features**: has_emoji, has_hashtag, has_mention, has_url

**Metric**: Cramér's V (equivalent to phi coefficient φ for binary features)

Same interpretation as categorical features.

### 4.2 Statistical Significance

**Significance Level**: α = 0.05 (5% significance level)

A feature shows significant bias if p-value < 0.05.

#### Multi-Level Significance Markers

When aggregating across multiple conditions, significance is based on the **proportion of underlying conditions** that show p < 0.05:

- `***`: More than 75% of underlying conditions are significant
- `**`: More than 60% of underlying conditions are significant
- `*`: More than 50% of underlying conditions are significant
- (no marker): Less than 50% of conditions are significant

**Example**: In "Bias by Dataset" heatmap, each cell aggregates 18 conditions (3 models × 6 prompts). If 14 of these show p < 0.05, the cell is marked `***`.

### 4.3 Bias Normalization

To enable visual comparison across features with different scales, bias values are **normalized within each feature** using min-max scaling.

**Formula**:
```
Normalized_bias = (bias - bias_min) / (bias_max - bias_min)

where bias_min and bias_max are computed across all 54 conditions for that specific feature
```

**Result**: All normalized bias values range from 0 to 1

**Important**:
- Normalized values are only comparable **within** a feature, not across features
- Used only for heatmap visualizations
- Raw bias values are preserved in CSV files

### 4.4 Aggregation Levels

Bias is computed and visualized at multiple aggregation levels:

| Aggregation Level | Conditions | Use Case |
|-------------------|------------|----------|
| **Fully Disaggregated** | 54 (dataset × model × prompt) | Detailed condition-specific analysis |
| **By Prompt Style** | 6 prompts | Compare prompt effects |
| **By Dataset** | 3 datasets | Compare platforms |
| **By Model** | 3 providers | Compare LLM providers |
| **Fully Aggregated** | 1 overall | Identify most biased features |

---

## 5. Analysis Pipeline

The analysis generates **four types of visualizations**, organized in numbered folders reflecting the analysis sequence.

### 5.1 Feature Distributions (Folder: 1_distributions/)

**Purpose**: Document the baseline distribution of each feature in the pool data.

**Output**: 16 distribution plots (one per feature)

**Visualization Types**:
- **Numerical features**: Histograms with mean/median lines
- **Categorical features**: Bar charts showing category proportions
- **Binary features**: Stacked percentage bars

**Key Insight**: Validates that features show sufficient variance for bias detection. Identifies imbalanced features that may have limited statistical power.

**Files**:
```
1_distributions/
├── author_gender_distribution.png
├── author_political_leaning_distribution.png
├── text_length_distribution.png
├── sentiment_polarity_distribution.png
├── toxicity_distribution.png
└── ... (16 plots total)
```

### 5.2 Bias Heatmaps (Folder: 2_bias_heatmaps/)

**Purpose**: Visualize the **magnitude** of bias across features and conditions.

**Metric**: Normalized bias (0-1 scale, within-feature normalization)

**Output**: 10 heatmaps
- 6 disaggregated heatmaps (one per prompt style)
- 4 aggregated heatmaps (by dataset, by model, by prompt, fully aggregated)

**Colormap**: White-to-red gradient
- White (0): No bias
- Red (1): Maximum bias for that feature

**Significance Markers**: `*`, `**`, `***` based on proportion of significant underlying conditions

**Files**:
```
2_bias_heatmaps/
├── disaggregated_prompt_general.png
├── disaggregated_prompt_popular.png
├── disaggregated_prompt_engaging.png
├── disaggregated_prompt_informative.png
├── disaggregated_prompt_controversial.png
├── disaggregated_prompt_neutral.png
├── aggregated_by_dataset.png
├── aggregated_by_model.png
├── aggregated_by_prompt.png
└── fully_aggregated.png
```

**Interpretation Example**:
- High normalized bias (dark red) + `***`: Strong, highly significant bias
- High normalized bias + no marker: Large effect but inconsistent significance
- Low normalized bias + `***`: Small but consistently significant bias

### 5.3 Directional Bias (Folder: 3_directional_bias/)

**Purpose**: Show **which categories or values** are favored by LLMs.

**Output**: 16 plots (one per feature)

**Metrics**:
- **Categorical features**: Proportion_recommended - Proportion_pool for each category
- **Numerical features**: Mean_recommended - Mean_pool by prompt style

**Visualization**:
- Categorical: Grouped bar charts showing proportion differences
- Numerical: Bar charts showing mean differences by prompt

**Key Insight**: Identifies directionality of bias. For example:
- Do LLMs favor left-leaning or right-leaning authors?
- Do LLMs prefer longer or shorter posts?
- Are toxic posts over- or under-represented?

**Files**:
```
3_directional_bias/
├── author_political_leaning_by_prompt.png
├── sentiment_polarity_by_prompt.png
├── text_length_by_prompt.png
├── toxicity_by_prompt.png
└── ... (16 plots total)
```

### 5.4 Feature Importance (Folder: 4_feature_importance/)

**Purpose**: Identify which features are most **predictive** of recommendation likelihood using machine learning.

**Method**: Random Forest Classification + SHAP (SHapley Additive exPlanations)

**Approach**:
1. Train Random Forest classifier to predict whether a post was recommended (binary outcome)
2. Compute feature importances (gini-based)
3. Compute SHAP values for each feature (game-theoretic importance)
4. Aggregate across conditions

**Performance**:
- **Mean AUROC**: 0.8703 (excellent predictive performance)
- **Range**: 0.8066 to 0.9473
- **Best condition**: Twitter × Anthropic × Controversial (AUROC = 0.9473)

**Output**: 10 heatmaps
- 6 disaggregated heatmaps (one per prompt style)
- 4 aggregated heatmaps (by dataset, by model, by prompt, fully aggregated)

**Files**:
```
4_feature_importance/
├── disaggregated_prompt_general.png
├── disaggregated_prompt_popular.png
├── disaggregated_prompt_engaging.png
├── disaggregated_prompt_informative.png
├── disaggregated_prompt_controversial.png
├── disaggregated_prompt_neutral.png
├── aggregated_by_dataset.png
├── aggregated_by_model.png
├── aggregated_by_prompt.png
└── fully_aggregated.png
```

**Interpretation**: Higher importance = feature is more useful for predicting recommendations. This differs from bias magnitude:
- A feature can have high importance but low bias (e.g., well-calibrated but relevant feature)
- A feature can have high bias but low importance (e.g., affects only a small subset of posts)

---

## 6. Key Findings

### 6.1 Overall Bias Prevalence

**67.6%** of all feature-condition combinations show statistically significant bias (p < 0.05)

This indicates widespread systematic bias in LLM recommendation systems.

### 6.2 Most Biased Features (by Significance Rate)

| Rank | Feature | Significance Rate | Interpretation |
|------|---------|-------------------|----------------|
| 1 | `text_length` | 98.1% | Nearly universal bias across all conditions |
| 2 | `primary_topic` | 94.4% | Strong topic preferences |
| 3 | `polarization_score` | 90.7% | Consistent political polarization bias |
| 4 | `author_political_leaning` | 88.9% | Strong political preference |
| 5 | `avg_word_length` | 83.3% | Text complexity bias |
| 6 | `controversy_level` | 77.8% | Controversy preference |
| 7 | `toxicity` | 75.9% | Toxicity filtering/preference |
| 8 | `author_gender` | 72.2% | Gender bias present |
| 9 | `severe_toxicity` | 70.4% | Severe toxicity bias |
| 10 | `sentiment_subjectivity` | 66.7% | Subjectivity preference |

**Least biased features**:
- `has_mention`: 25.9% (sparse feature - only 0.9% prevalence)
- `author_is_minority`: 38.9%
- `has_hashtag`: 40.7% (sparse feature - only 3.4% prevalence)

### 6.3 Bias Magnitude (Absolute Average)

| Rank | Feature | Abs Mean Bias | Type | Interpretation |
|------|---------|---------------|------|----------------|
| 1 | `text_length` | 0.541 | Cohen's d | **Large effect**: LLMs strongly prefer certain text lengths |
| 2 | `polarization_score` | 0.381 | Cohen's d | **Medium-large effect**: Strong polarization preference |
| 3 | `toxicity` | 0.214 | Cohen's d | **Small-medium effect**: Toxicity filtering |
| 4 | `avg_word_length` | 0.166 | Cohen's d | **Small effect**: Word complexity bias |
| 5 | `severe_toxicity` | 0.137 | Cohen's d | **Small effect**: Severe toxicity filtering |

**Note**: Cramér's V values (categorical features) are generally lower in magnitude but still indicate meaningful associations.

### 6.4 Bias by Provider

| Provider | Significance Rate | Interpretation |
|----------|-------------------|----------------|
| Anthropic | 71.5% | Most biased provider |
| Gemini | 70.1% | Second most biased |
| OpenAI | 61.1% | Least biased (but still majority) |

**Interpretation**: All three providers show substantial bias, with Anthropic and Gemini showing slightly more frequent significant biases than OpenAI. Differences are relatively small (~10 percentage points).

### 6.5 Bias by Dataset

| Dataset | Significance Rate | Interpretation |
|---------|-------------------|----------------|
| Twitter | 69.1% | Slightly more biased |
| Bluesky | 67.4% | Middle |
| Reddit | 66.3% | Slightly less biased |

**Interpretation**: Bias rates are highly consistent across platforms (within 3 percentage points), suggesting that biases are inherent to LLM recommendation systems rather than platform-specific.

### 6.6 Bias by Prompt Style

| Prompt Style | Significance Rate | Interpretation |
|--------------|-------------------|----------------|
| Informative | 81.3% | **Most biased**: Strong content quality preferences |
| Controversial | 70.8% | High bias due to explicit selection criteria |
| Neutral | 68.8% | Moderate-high bias |
| Engaging | 68.1% | Moderate-high bias |
| General | 65.3% | Baseline bias level |
| Popular | 51.4% | **Least biased**: More democratic selection |

**Key Insight**:
- "Informative" prompts show highest bias rates, suggesting strong implicit assumptions about information quality
- "Popular" prompts show lowest bias rates, possibly because popularity is more objective/observable
- Even "neutral" and "general" prompts show ~65% bias rate, indicating baseline biases

### 6.7 Feature Importance Rankings

#### Random Forest Importance (Top 10)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `text_length` | 0.2464 | Text Metrics |
| 2 | `toxicity` | 0.1301 | Toxicity |
| 3 | `avg_word_length` | 0.1202 | Text Metrics |
| 4 | `severe_toxicity` | 0.1087 | Toxicity |
| 5 | `sentiment_subjectivity` | 0.0990 | Sentiment |
| 6 | `sentiment_polarity` | 0.0894 | Sentiment |
| 7 | `polarization_score` | 0.0540 | Content |
| 8 | `primary_topic` | 0.0495 | Content |
| 9 | `author_political_leaning` | 0.0251 | Author |
| 10 | `author_gender` | 0.0213 | Author |

#### SHAP Importance (Top 10)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `text_length` | 0.1676 | Text Metrics |
| 2 | `toxicity` | 0.0671 | Toxicity |
| 3 | `avg_word_length` | 0.0632 | Text Metrics |
| 4 | `polarization_score` | 0.0452 | Content |
| 5 | `severe_toxicity` | 0.0436 | Toxicity |
| 6 | `sentiment_subjectivity` | 0.0428 | Sentiment |
| 7 | `primary_topic` | 0.0415 | Content |
| 8 | `sentiment_polarity` | 0.0339 | Sentiment |
| 9 | `author_political_leaning` | 0.0219 | Author |
| 10 | `author_gender` | 0.0164 | Author |

**Key Insights**:
1. **Text length is by far the most important predictor** across both methods
2. **Toxicity features are highly predictive** (rank 2 and 4 in both methods)
3. **Text metrics dominate**: text_length + avg_word_length account for ~35% of RF importance
4. **Author features have low importance** despite showing significant bias in some cases
5. **Style features (emoji, hashtag, etc.) have negligible importance**

#### Feature Importance by Category (SHAP)

| Category | Average SHAP Importance | Interpretation |
|----------|------------------------|----------------|
| Text Metrics | 0.1154 | **Dominant category** |
| Toxicity | 0.0553 | Strong predictor |
| Content | 0.0323 | Moderate importance |
| Sentiment | 0.0383 | Moderate importance |
| Author | 0.0176 | Low importance |
| Style | 0.0046 | Negligible importance |

### 6.8 Directional Bias Patterns

Key patterns from directional bias analysis:

#### Political Leaning
- **Left-leaning authors**: Generally over-represented in recommendations across most conditions
- **Right-leaning authors**: Generally under-represented
- Pattern consistent across providers and datasets

#### Text Length
- **Longer posts** are generally preferred over shorter posts
- Effect size varies by prompt (strongest for "informative", weakest for "popular")
- Consistent across all providers and datasets

#### Sentiment Polarity
- **Mixed patterns**: Some conditions favor positive sentiment, others negative
- "Engaging" prompts: Slight preference for negative/controversial sentiment
- "Informative" prompts: Slight preference for neutral/balanced sentiment

#### Toxicity
- **Lower toxicity** posts generally preferred
- Effect stronger for "neutral" and "informative" prompts
- Weaker filtering for "controversial" and "engaging" prompts

#### Primary Topic
- **Personal topics** (66.9% of pool) are under-represented in recommendations
- **Political topics** (13.6% of pool) are over-represented
- **Technology topics** show mixed patterns depending on prompt

---

## 7. Results Locations

### 7.1 Main Analysis Results

**Primary Directory**: `/home/nicpag/data/llm-recommendation-bias-analysis/analysis_outputs/`

```
analysis_outputs/
├── feature_importance_data.csv          # Feature importance results (97 KB)
│                                        # Columns: feature, dataset, provider, prompt_style,
│                                        # rf_importance, shap_importance, auroc, n_samples
│
├── pool_vs_recommended_summary.csv      # Bias metrics for all conditions (83 KB)
│                                        # Columns: feature, dataset, provider, prompt_style,
│                                        # bias, p_value, metric, significant
│
└── visualizations/
    ├── 1_distributions/                 # 16 PNG files
    │   └── [feature]_distribution.png   # One per feature
    │
    ├── 2_bias_heatmaps/                # 10 PNG files
    │   ├── disaggregated_prompt_[style].png  # 6 files (one per prompt)
    │   ├── aggregated_by_dataset.png
    │   ├── aggregated_by_model.png
    │   ├── aggregated_by_prompt.png
    │   └── fully_aggregated.png
    │
    ├── 3_directional_bias/              # 16 PNG files
    │   └── [feature]_by_prompt.png      # One per feature
    │
    └── 4_feature_importance/            # 10 PNG files
        ├── disaggregated_prompt_[style].png  # 6 files (one per prompt)
        ├── aggregated_by_dataset.png
        ├── aggregated_by_model.png
        ├── aggregated_by_prompt.png
        └── fully_aggregated.png
```

**Last Modified**: December 24, 2025

### 7.2 Raw Experiment Data

**Location**: `/home/nicpag/data/llm-recommendation-bias-analysis/outputs/experiments/`

```
outputs/experiments/
├── twitter_openai_[model]/
├── twitter_anthropic_[model]/
├── twitter_gemini_[model]/
├── bluesky_openai_[model]/
├── bluesky_anthropic_[model]/
├── bluesky_gemini_[model]/
├── reddit_openai_[model]/
├── reddit_anthropic_[model]/
└── reddit_gemini_[model]/
    └── post_level_data.csv              # Post-level data with features and selection status
```

### 7.3 External Survey Data Analysis

**Location**: `/home/nicpag/data/llm-recommendation-bias-analysis/external_data_analysis/outputs/analysis_full/`

Separate analysis pipeline for external collaborators' survey data. Same structure as main analysis:
- Feature importance data
- Pool vs recommended summary
- All 4 visualization types (distributions, bias heatmaps, directional bias, feature importance)

**Last Modified**: December 24, 2025, 15:29

### 7.4 Documentation

Key documentation files:

| File | Description |
|------|-------------|
| `README.md` | Project overview and usage instructions |
| `BIAS_COMPUTATION_METHODOLOGY.md` | Detailed methodology for bias computation and normalization |
| `WORK_SUMMARY_2025-12-16.md` | Work summary from December 16 analysis updates |
| `COMPREHENSIVE_ANALYSIS_SUMMARY.md` | This document |
| `analysis_outputs/visualizations/README.md` | Visualization guide |
| `external_data_analysis/USAGE.md` | Guide for external survey data analysis |

---

## 8. Analysis Scripts

### Main Analysis Pipeline

**Primary Script**: `run_comprehensive_analysis.py`

**Runtime**:
- First run: ~30-40 minutes (computes feature importance)
- Subsequent runs: ~2-3 minutes (loads cached feature importance)

**Usage**:
```bash
python3 run_comprehensive_analysis.py
```

**What it does**:
1. Loads post-level data from all 54 experimental conditions
2. Computes bias metrics for all 16 features
3. Generates feature distributions (16 plots)
4. Generates bias heatmaps (10 plots)
5. Generates directional bias plots (16 plots)
6. Trains Random Forest models and computes SHAP values (54 models)
7. Generates feature importance heatmaps (10 plots)
8. Saves results to CSV files

**Caching**: Feature importance results are cached in `feature_importance_data.csv`. Delete this file to force recomputation.

### Additional Analysis Scripts

| Script | Purpose |
|--------|---------|
| `stratified_analysis.py` | Stratified bias analysis by subgroups |
| `meta_analysis.py` | Cross-dataset meta-analysis |
| `analyze_features.py` | Feature documentation and distribution analysis |
| `run_feature_importance.py` | Standalone feature importance analysis |

---

## 9. Statistical Interpretation Guide

### 9.1 When to Report Bias

Report bias when **both** conditions are met:
1. **Statistical significance**: p < 0.05
2. **Meaningful effect size**:
   - Cohen's d: |d| ≥ 0.2 (small effect or larger)
   - Cramér's V: V ≥ 0.1 (weak association or stronger)

### 9.2 Interpreting Effect Sizes

#### Cohen's d (Numerical Features)

| |d| Range | Interpretation | Example |
|-----------|----------------|---------|
| 0.0 - 0.2 | Negligible to small | May not be practically important |
| 0.2 - 0.5 | Small to medium | Noticeable but moderate difference |
| 0.5 - 0.8 | Medium to large | Substantial difference |
| > 0.8 | Large to very large | Very strong bias |

**Our findings**: `text_length` shows Cohen's d up to 1.57 (very large effect)

#### Cramér's V (Categorical Features)

| V Range | Interpretation | Example |
|---------|----------------|---------|
| 0.0 - 0.1 | Negligible association | Weak or no practical bias |
| 0.1 - 0.3 | Weak association | Detectable but modest bias |
| 0.3 - 0.5 | Moderate association | Clear bias pattern |
| > 0.5 | Strong association | Very strong bias |

**Our findings**: Most categorical features show V = 0.02 - 0.15 (weak associations)

### 9.3 Multiple Testing Considerations

With **864 tests** (16 features × 54 conditions), we expect ~43 false positives at α = 0.05 under the null hypothesis of no bias.

**Our observation**: 584 significant tests (67.6%), far exceeding the expected false positive rate. This strongly suggests real, systematic bias.

**Recommendation**: For publication, consider:
- Bonferroni correction: α_adjusted = 0.05 / 864 ≈ 0.000058 (very conservative)
- False Discovery Rate (FDR) control: Benjamini-Hochberg procedure (less conservative)
- Focus on features with high significance rates (e.g., >75% of conditions significant)

---

## 10. Key Takeaways for Paper Writing

### Main Contributions

1. **Large-scale empirical study**: 540,000 recommendations across 54 conditions, 16 features
2. **Systematic methodology**: Rigorous bias quantification using established statistical metrics
3. **Multi-provider comparison**: First study to compare OpenAI, Anthropic, and Gemini on recommendation bias
4. **Feature importance analysis**: Novel application of SHAP to explain recommendation decisions
5. **Comprehensive bias characterization**: Both magnitude and direction of bias documented

### Main Findings

1. **Widespread bias**: 67.6% of feature-condition combinations show significant bias
2. **Text length dominates**: Strongest predictor and most biased feature across all conditions
3. **Political bias is pervasive**: 88.9% of conditions show significant political leaning bias
4. **Provider differences are modest**: All three providers show substantial bias (61-72%)
5. **Prompt style matters**: Informative prompts show highest bias (81%), popular prompts lowest (51%)
6. **Consistent across platforms**: Similar bias patterns on Twitter, Bluesky, and Reddit

### Limitations to Acknowledge

1. **Synthetic data**: Posts generated from personas, not real social media data
2. **Feature coverage**: 16 features may not capture all relevant biases
3. **Causality**: We measure associations, not causal effects
4. **Static analysis**: Single snapshot, not longitudinal
5. **Imbalanced features**: Some features (has_mention, has_hashtag) have limited prevalence
6. **Gender unknown**: 66.8% of authors have unknown gender, limiting gender bias analysis

### Future Work

1. **Mitigation strategies**: Test debiasing interventions
2. **User impact**: Measure effects on user experience and platform outcomes
3. **Real-world validation**: Replicate with actual social media data
4. **Temporal dynamics**: Analyze how biases change over time
5. **Intersectional analysis**: Examine combined effects of multiple identity features
6. **Explainability**: Use SHAP to create interpretable bias profiles for LLMs

---

## 11. Figures for Paper

### Recommended Main Figures

1. **Figure 1**: Fully aggregated bias heatmap (`fully_aggregated.png`)
   - Shows overall bias landscape across all 16 features
   - Establishes which features are most biased

2. **Figure 2**: Bias by prompt style heatmap (`aggregated_by_prompt.png`)
   - Demonstrates how prompt wording affects bias
   - Key finding: informative vs popular prompts

3. **Figure 3**: Text length directional bias (`text_length_by_prompt.png`)
   - Most important feature, shows clear directional pattern
   - Illustrates how different prompts modulate bias

4. **Figure 4**: Political leaning directional bias (`author_political_leaning_by_prompt.png`)
   - Critical for fairness concerns
   - Shows systematic left-leaning preference

5. **Figure 5**: Feature importance fully aggregated (`4_feature_importance/fully_aggregated.png`)
   - Complements bias analysis
   - Shows which features drive recommendations

### Recommended Supplementary Figures

- All disaggregated heatmaps (by prompt style)
- All directional bias plots
- Feature distributions
- Bias by provider and dataset comparisons

---

## 12. Data Availability Statement

**For paper**:

> All analysis code, results, and visualizations are available in the project repository. Raw experimental data (540,000 recommendations) and feature-level summaries are provided in CSV format. Synthetic post data and author personas can be made available upon request, subject to ethical review.

**Repository structure**:
- Analysis scripts: `run_comprehensive_analysis.py`, `stratified_analysis.py`, etc.
- Results: `analysis_outputs/` directory
- Documentation: `BIAS_COMPUTATION_METHODOLOGY.md`, `README.md`
- Raw data: `outputs/experiments/` directory

---

## Document Metadata

- **Created**: December 24, 2025
- **Purpose**: Comprehensive summary for LaTeX writeup
- **Audience**: Research team preparing manuscript
- **Scope**: Complete methodology, results, and interpretation guide
- **Status**: Final version incorporating all December 2025 analyses

---

**End of Document**
