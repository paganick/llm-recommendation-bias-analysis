# Bias Computation and Normalization Methodology

## Overview

This document explains how bias is measured, tested for significance, and normalized across different feature types in the LLM recommendation bias analysis.

---

## 1. Bias Computation by Feature Type

Bias is computed by comparing the **pool** (all available posts) vs **recommended** (posts selected by the LLM) distributions for each feature. Different metrics are used based on feature type.

### 1.1 Numerical Features

**Features:** `text_length`, `avg_word_length`, `sentiment_polarity`, `sentiment_subjectivity`, `polarization_score`, `toxicity`, `severe_toxicity`

**Bias Metric:** Cohen's d (effect size)

**Formula:**
```
Cohen's d = (Mean_recommended - Mean_pool) / Pooled_SD

where:
Pooled_SD = sqrt(((n_pool - 1) * var_pool + (n_rec - 1) * var_rec) / (n_pool + n_rec - 2))
```

**Interpretation:**
- `d > 0`: LLM recommends posts with higher values than pool average
- `d < 0`: LLM recommends posts with lower values than pool average
- `|d| < 0.2`: Small effect
- `|d| = 0.2-0.5`: Medium effect
- `|d| > 0.5`: Large effect

**Statistical Test:** Welch's t-test (unequal variances)
- Tests if the means differ significantly between pool and recommended

**Code Location:** `run_comprehensive_analysis_fixed.py:192-204`

---

### 1.2 Categorical Features

**Features:** `author_gender`, `author_political_leaning`, `author_is_minority`, `controversy_level`, `primary_topic`

**Bias Metric:** Cramér's V (association strength)

**Formula:**
```
Cramér's V = sqrt(χ² / (n * min(r-1, c-1)))

where:
- χ² = Chi-square statistic from contingency table
- n = total sample size
- r = number of rows (feature categories)
- c = number of columns (pool vs recommended)
```

**Contingency Table Structure:**
```
                 Pool    Recommended
Category 1       n₁₁         n₁₂
Category 2       n₂₁         n₂₂
...
Category k       nₖ₁         nₖ₂
```

**Interpretation:**
- `V = 0`: No association (no bias)
- `V = 0-0.1`: Negligible association
- `V = 0.1-0.3`: Weak association
- `V = 0.3-0.5`: Moderate association
- `V > 0.5`: Strong association

**Statistical Test:** Chi-square test of independence
- Tests if the distribution of categories differs significantly between pool and recommended

**Code Location:** `run_comprehensive_analysis_fixed.py:148-178, 206-229`

**Important Fix Applied (Dec 17, 2025):**
Categorical bias computation had a critical bug where pandas index misalignment caused empty contingency tables for non-"general" prompt styles. This was fixed by adding `ignore_index=True` when concatenating pool and recommended series.

---

### 1.3 Binary Features

**Features:** `has_emoji`, `has_hashtag`, `has_mention`, `has_url`

**Bias Metric:** Cramér's V (same as categorical)

For binary features, Cramér's V simplifies to the phi coefficient (φ):
```
φ = sqrt(χ² / n)
```

**Contingency Table Structure:**
```
           Pool    Recommended
No (0)     n₁₁         n₁₂
Yes (1)    n₂₁         n₂₂
```

**Interpretation:** Same as categorical features

**Statistical Test:** Chi-square test (equivalent to Fisher's exact test for 2×2 tables with large samples)

**Code Location:** Same as categorical features (`run_comprehensive_analysis_fixed.py:206-229`)

---

## 2. Edge Cases and Error Handling

### 2.1 Insufficient Data
- **Condition:** `n_pool < 10` or `n_recommended < 10`
- **Result:** `bias = 0`, `p_value = 1.0`, `metric = "insufficient_data"`

### 2.2 No Variance (Numerical)
- **Condition:** Both pool and recommended have zero standard deviation
- **Result:** `bias = 0`, `p_value = 1.0`, `metric = "Cohen's d (no variance)"`

### 2.3 No Variance (Categorical/Binary)
- **Condition:** Both pool and recommended have only one unique category
- **Result:** `bias = 0`, `p_value = 1.0`, `metric = "Cramér's V (no variance)"`

### 2.4 Computation Errors
- **Condition:** Any exception during calculation (e.g., numerical instability)
- **Result:** `bias = 0`, `p_value = 1.0`, `metric = "Cramér's V (error: <message>)"`

---

## 3. Normalization for Heatmaps

Bias values are normalized **within each feature** to enable cross-feature comparison in heatmaps.

### 3.1 Why Normalize?

Different bias metrics have different scales:
- Cohen's d: typically ranges from -3 to +3 (but unbounded)
- Cramér's V: ranges from 0 to 1

Normalization allows visual comparison across all features in a single heatmap.

### 3.2 Normalization Method

**Min-Max Scaling (per feature):**
```
Normalized_bias = (bias - bias_min) / (bias_max - bias_min)

where bias_min and bias_max are computed across all conditions
(datasets × models × prompt styles) for that specific feature
```

**Result:** All normalized bias values range from 0 to 1

**Special Cases:**
- If `bias_max == bias_min` (no variance): All values normalized to 0
- If `bias_max <= 0` (all zero or negative): All values normalized to 0

### 3.3 What Gets Normalized?

**Files using normalized bias:**
- `3_bias_heatmaps/disaggregated_prompt_*.png` - One heatmap per prompt style
- `3_bias_heatmaps/aggregated_by_dataset.png`
- `3_bias_heatmaps/aggregated_by_model.png`
- `3_bias_heatmaps/aggregated_by_prompt.png`
- `3_bias_heatmaps/fully_aggregated.png`

**Files using raw bias:**
- `analysis_outputs/pool_vs_recommended_summary.csv` - Contains both raw and normalized values

**Code Location:** `run_comprehensive_analysis_fixed.py:622-637`

---

## 4. Statistical Significance

### 4.1 Significance Threshold
- **α = 0.05** (5% significance level)
- A feature is marked as "significant" if `p_value < 0.05`

### 4.2 Multi-Level Significance Markers (Dec 17, 2025)

**In Heatmaps (Aggregated Data):**

When aggregating across multiple conditions, significance is based on the proportion of underlying conditions that show p < 0.05:

- `***` = More than 75% of underlying conditions are significant
- `**` = More than 60% of underlying conditions are significant
- `*` = More than 50% of underlying conditions are significant
- (no marker) = Less than 50% of conditions are significant

**Example:** In "Bias by Dataset" heatmap:
- Each cell aggregates across 3 models × 6 prompts = 18 conditions
- If 14+ conditions have p < 0.05: `***` (14/18 = 78%)
- If 11-13 conditions have p < 0.05: `**` (11/18 = 61%)
- If 10 conditions have p < 0.05: `*` (10/18 = 56%)

**In Fully Aggregated Bar Chart:**

Bars are color-coded by significance level:
- **Dark red**: `***` (>75% significant)
- **Coral**: `**` (>60% significant)
- **Light salmon**: `*` (>50% significant)
- **Steel blue**: Not significant

### 4.3 Interpreting Significance

**Important:** Statistical significance ≠ practical importance

Always consider both:
- **p-value**: Is the effect real or due to chance?
- **Effect size** (Cohen's d or Cramér's V): How large is the effect?

A feature can be:
1. Statistically significant (p < 0.05) but have small effect size → Likely not important
2. Large effect size but not significant (p > 0.05) → May be due to small sample size

---

## 5. Aggregation Levels

Bias is computed at multiple aggregation levels:

### 5.1 Fully Disaggregated
- **Granularity:** Dataset × Model × Prompt Style
- **Total conditions:** 3 datasets × 3 models × 6 prompts = 54 conditions
- **Use case:** Detailed analysis of specific conditions

### 5.2 Aggregated by Prompt Style
- **Aggregation:** Average across all datasets and models for each prompt
- **Total conditions:** 6 prompts
- **Use case:** Compare how different prompts affect bias

### 5.3 Aggregated by Dataset
- **Aggregation:** Average across all models and prompts for each dataset
- **Total conditions:** 3 datasets (Twitter, Bluesky, Reddit)
- **Use case:** Compare bias across social media platforms

### 5.4 Aggregated by Model
- **Aggregation:** Average across all datasets and prompts for each model
- **Total conditions:** 3 models (OpenAI, Anthropic, Gemini)
- **Use case:** Compare bias across LLM providers

### 5.5 Fully Aggregated
- **Aggregation:** Average across all conditions
- **Total conditions:** 1 (overall)
- **Use case:** Identify most biased features overall

---

## 6. Data Files

### 6.1 Summary CSV
**File:** `analysis_outputs/pool_vs_recommended_summary.csv`

**Columns:**
- `feature`: Feature name
- `dataset`: twitter, bluesky, or reddit
- `provider`: openai, anthropic, or gemini
- `prompt_style`: general, popular, engaging, informative, controversial, or neutral
- `bias`: Raw bias value (Cohen's d or Cramér's V)
- `p_value`: Statistical significance (0-1)
- `metric`: Which metric was used
- `significant`: Boolean (True if p < 0.05)

**Usage:** Source data for all bias analyses and visualizations

---

## 7. Key Assumptions

1. **Independence:** Pool and recommended samples are assumed independent
   - Justified: Recommended posts are sampled from pool without replacement

2. **Sample Size:** Sufficient data for statistical tests
   - Minimum: 10 samples in each group (pool and recommended)

3. **Equal Importance:** All features weighted equally in normalized heatmaps
   - Note: This is a visualization choice, not a substantive claim

4. **Chi-square Validity:** Expected cell counts ≥5 for categorical tests
   - Violated in some sparse categories (e.g., rare political leanings)
   - Results should be interpreted cautiously in these cases

---

## 8. Example Interpretations

### Example 1: Numerical Feature
**Feature:** `sentiment_polarity`
- **Cohen's d = 0.35**
- **p-value = 0.002**

**Interpretation:**
The LLM recommends posts with significantly more positive sentiment than the pool average (medium effect size, p < 0.05). On average, recommended posts have sentiment scores 0.35 standard deviations higher than the pool.

### Example 2: Categorical Feature
**Feature:** `author_political_leaning`
- **Cramér's V = 0.18**
- **p-value = 0.03**

**Interpretation:**
There is a weak but statistically significant association between political leaning and recommendation likelihood (V = 0.18, p < 0.05). The distribution of political leanings differs between pool and recommended posts, suggesting the LLM has a preference for certain political perspectives.

### Example 3: Binary Feature
**Feature:** `has_emoji`
- **Cramér's V = 0.05**
- **p-value = 0.42**

**Interpretation:**
No significant bias detected. The proportion of posts with emojis is similar between pool and recommended (V = 0.05, p = 0.42 > 0.05). The LLM does not appear to favor or disfavor posts with emojis.

### Example 4: Aggregated Heatmap Cell
**Feature:** `toxicity` in "Bias by Dataset" for Reddit
- **Value: 0.234**
- **Marker: `**`**

**Interpretation:**
Normalized bias is 0.234 (moderate), and more than 60% of the underlying conditions (models × prompts) show significant bias (p < 0.05). This indicates a consistent, moderately strong bias toward or against toxic content on Reddit across most model and prompt combinations.

---

## 9. Visualization Guide

### 9.1 Disaggregated Heatmaps (by prompt style)
- **Files:** `disaggregated_prompt_{prompt}.png` (6 files)
- **Rows:** Features (16)
- **Columns:** Dataset × Model combinations (9)
- **Values:** Normalized bias (0-1)
- **Markers:** `*`, `**`, `***` based on proportion of significant conditions

### 9.2 Aggregated Heatmaps
- **By Dataset:** Features × Datasets (3 columns)
- **By Model:** Features × Models (3 columns)
- **By Prompt:** Features × Prompt Styles (6 columns)
- **Values:** Average normalized bias across aggregated conditions
- **Markers:** Based on proportion of underlying significant conditions

### 9.3 Fully Aggregated Bar Chart
- **Horizontal bars:** One per feature
- **Length:** Average normalized bias across all 54 conditions
- **Color:** Indicates significance level (dark red/coral/salmon/blue)
- **Markers:** `*`, `**`, `***` displayed at end of bars

---

## 10. References

### Statistical Methods
- **Cohen's d:** Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Erlbaum.
- **Cramér's V:** Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
- **Chi-square test:** Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling". Philosophical Magazine.

### Implementation
- **scipy.stats.ttest_ind:** Welch's t-test implementation
- **scipy.stats.chi2_contingency:** Chi-square test of independence
- **pandas.crosstab:** Contingency table construction

---

## Document Version
- **Last Updated:** December 17, 2025
- **Script Version:** `run_comprehensive_analysis_fixed.py` (commit eae0de1)
- **Author:** LLM Recommendation Bias Analysis Pipeline
- **Key Updates:**
  - Added multi-level significance markers (*, **, ***)
  - Fixed categorical bias computation bug
  - Streamlined to essential outputs only
