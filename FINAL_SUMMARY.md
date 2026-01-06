# Directional Bias Visualization - Complete Regeneration
**Date**: 2026-01-06

## ✅ All Issues Resolved

### 1. Data Organization ✓
- **Restored 9 canonical experiments** from `outputs/experiments_backup/`
- **Removed test data** (survey_gemini with only 600 rows)
- **Removed older experiment** (claude-3-5-haiku, superseded by claude-sonnet-4-5)

### 2. Regenerated Directional Bias Data ✓
**Files**: `analysis_outputs/directional_bias_data.{csv,parquet}`

**Statistics**:
- **1,854 directional bias measurements** (up from 132 with test data)
- **540,000 total experiment rows** (3 datasets × 3 providers × 60,000 rows)
- **Balanced across**:
  - 3 datasets: twitter, bluesky, reddit
  - 3 providers: anthropic (Claude Sonnet 4.5), gemini (2.0-flash), openai (gpt-4o-mini)
  - 6 prompt styles: general, popular, engaging, informative, controversial, neutral
  - 16 features: all author, text, sentiment, style, content, toxicity features

### 3. Fixed All Plotting Bugs ✓

#### Bug 1: Missing Disaggregation (FIXED)
**Before**: Only showed dataset names (1 column/bar per dataset)
**After**: Shows all 9 dataset×model combinations

#### Bug 2: Wrong Plot Structure for by_dataset & by_model (FIXED)
**Before**: 
- 3 subplots (one per dataset/model)
- Aggregated across prompt styles

**After**:
- **6 subplots** (one per prompt style) ✓
- `by_dataset`: Shows 3 datasets (aggregated across models)
- `by_model`: Shows 3 providers (aggregated across datasets)

#### Bug 3: Unreadable Annotations (FIXED)
**Before**: Used `.3f` format (3 decimal places)
**After**: Uses `.2f` format (2 decimal places) for better readability

#### Bug 4: Continuous Features Missing Bars (FIXED)
**Before**: Only 1 bar (just the dataset name)
**After**: 
- `by_prompt`: 9 bars (all dataset×model combinations)
- `by_dataset`: 3 bars (3 datasets, averaged across models)
- `by_model`: 3 bars (3 providers, averaged across datasets)

### 4. Current Plot Structure

**48 total plots** (16 features × 3 plot types):

#### A. `*_by_prompt.png` (Fully Disaggregated)
- **6 subplots** (one per prompt style)
- **Categorical features**: Heatmaps with categories as rows, 9 dataset×model combinations as columns
- **Continuous features**: 9 horizontal bars (one per dataset×model combination)
- **Shows**: All granular combinations

#### B. `*_by_dataset.png` (Aggregated across Models)
- **6 subplots** (one per prompt style)
- **Categorical features**: Heatmaps with categories as rows, 3 datasets as columns
- **Continuous features**: 3 horizontal bars (one per dataset)
- **Aggregation**: Averaged across the 3 providers

#### C. `*_by_model.png` (Aggregated across Datasets)
- **6 subplots** (one per prompt style)
- **Categorical features**: Heatmaps with categories as rows, 3 providers as columns
- **Continuous features**: 3 horizontal bars (one per provider)
- **Aggregation**: Averaged across the 3 datasets

### 5. Visual Styling

**Colors** (PuOr diverging colormap):
- **Purple** (#998ec3): Positive directional bias (over-represented in recommendations)
- **Orange** (#f1a340): Negative directional bias (under-represented in recommendations)
- **White**: Neutral (no bias)

**Labels**:
- Clean feature names: "Author: Gender", "Text: Length (chars)", etc.
- Proper dataset labels: "Twitter/X", "Bluesky", "Reddit"
- Model names capitalized: "Anthropic", "Gemini", "Openai"

**Format**:
- Heatmap annotations: 2 decimal places (.2f)
- Range: -0.3 to +0.3 (centered at 0)

### 6. Gender Categories by Dataset

- **Twitter**: male, female, unknown (⚠️ no non-binary in source data)
- **Bluesky**: male, female, non-binary, unknown ✓
- **Reddit**: male, female, non-binary, unknown ✓

**Note**: Twitter's missing non-binary category is a data limitation, not a code issue.

### 7. Files Modified

1. **`regenerate_directional_bias.py`**
   - Loads all 9 experiments
   - Computes directional bias for all combinations
   - Saves to both parquet and CSV

2. **`regenerate_visualizations.py`**
   - Fixed `generate_directional_by_prompt()`: 9 dataset×model columns
   - Fixed `generate_directional_by_dataset()`: 6 prompt subplots, 3 dataset columns
   - Fixed `generate_directional_by_model()`: 6 prompt subplots, 3 provider columns
   - Changed annotation format from `.3f` to `.2f`

## 📂 Final Structure

```
outputs/experiments/                     (9 canonical experiments)
├── twitter_anthropic_claude-sonnet-4-5-20250929/
├── twitter_gemini_gemini-2.0-flash/
├── twitter_openai_gpt-4o-mini/
├── bluesky_anthropic_claude-sonnet-4-5-20250929/
├── bluesky_gemini_gemini-2.0-flash/
├── bluesky_openai_gpt-4o-mini/
├── reddit_anthropic_claude-sonnet-4-5-20250929/
├── reddit_gemini_gemini-2.0-flash/
└── reddit_openai_gpt-4o-mini/

analysis_outputs/
├── directional_bias_data.csv            (184K)
├── directional_bias_data.parquet        (125K)
└── visualizations/3_directional_bias/   (48 plots)
    ├── *_by_prompt.png                  (16 plots: 6 prompts × 9 dataset×model)
    ├── *_by_dataset.png                 (16 plots: 6 prompts × 3 datasets)
    └── *_by_model.png                   (16 plots: 6 prompts × 3 models)
```

## ✨ Summary

All bugs fixed, all data regenerated, all visualizations updated:
- ✅ Proper disaggregation (9 dataset×model combinations)
- ✅ Correct plot structure (6 prompt subplots)
- ✅ Readable annotations (2 decimal places)
- ✅ All bars showing correctly for continuous features
- ✅ Purple-orange diverging colormap centered at 0
- ✅ Clean labels throughout

**Ready for paper writeup!**
