"""
Enhance analysis results with normalization and feature categorization
Addresses Issues 2 & 3 from the analysis refinement plan

This script:
1. Adds normalized effect sizes (min-max scaling) to bias results
2. Adds normalized SHAP values to importance results
3. Categorizes features as stylistic vs substantive
4. Saves enhanced datasets for updated visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

print("="*80)
print("ENHANCING ANALYSIS WITH NORMALIZATION AND CATEGORIZATION")
print("="*80)

# ============================================================================
# FEATURE CATEGORIZATION (Issue 3)
# ============================================================================

# Stylistic features: formatting, length, style markers
STYLISTIC_FEATURES = [
    'text_length', 'word_count', 'avg_word_length',
    'has_emoji', 'has_hashtag', 'has_mention', 'has_url',
    'formality_score',
]

# Substantive features: content meaning, fairness, toxicity
SUBSTANTIVE_FEATURES = [
    # Content features
    'sentiment_polarity', 'sentiment_subjectivity',
    'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
    'sentiment_label',
    'primary_topic',
    'polarization_score',
    'has_polarizing_content',

    # Toxicity features
    'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack',

    # Fairness/demographics features (HIGH PRIORITY)
    'author_gender', 'author_political_leaning', 'author_is_minority'
]

print(f"\nFeature Categorization:")
print(f"  Stylistic features: {len(STYLISTIC_FEATURES)}")
print(f"  Substantive features: {len(SUBSTANTIVE_FEATURES)}")

# ============================================================================
# Load and Enhance Bias Results
# ============================================================================

print("\n" + "="*80)
print("ENHANCING BIAS RESULTS")
print("="*80)

bias_file = Path('analysis_outputs/bias_analysis/bias_results.parquet')
if not bias_file.exists():
    print(f"ERROR: {bias_file} not found!")
    exit(1)

bias_df = pd.read_parquet(bias_file)
print(f"\nLoaded {len(bias_df)} bias measurements")
print(f"Features analyzed: {bias_df['feature'].nunique()}")
print(f"Conditions: {len(bias_df['dataset'].unique())} datasets × {len(bias_df['model'].unique())} models × {len(bias_df['prompt_style'].unique())} prompts")

# Add feature category
def categorize_feature(feature_name):
    if feature_name in STYLISTIC_FEATURES:
        return 'stylistic'
    elif feature_name in SUBSTANTIVE_FEATURES or any(sub in feature_name for sub in ['author_', 'sentiment_', 'primary_topic', 'toxicity']):
        return 'substantive'
    else:
        # Default: substantive if unclear
        return 'substantive'

bias_df['feature_category'] = bias_df['feature'].apply(categorize_feature)

print(f"\nCategorization breakdown:")
print(bias_df['feature_category'].value_counts())

# Normalize Cohen's d using min-max scaling
print("\nNormalizing Cohen's d effect sizes...")
print(f"  Original Cohen's d range: [{bias_df['cohens_d'].min():.3f}, {bias_df['cohens_d'].max():.3f}]")

# Min-max scaling to [0, 1]
scaler = MinMaxScaler()
bias_df['cohens_d_normalized'] = scaler.fit_transform(bias_df[['cohens_d']])

# Also create an absolute normalized version (for visualization)
bias_df['cohens_d_abs'] = bias_df['cohens_d'].abs()
scaler_abs = MinMaxScaler()
bias_df['cohens_d_abs_normalized'] = scaler_abs.fit_transform(bias_df[['cohens_d_abs']])

print(f"  Normalized Cohen's d range: [{bias_df['cohens_d_normalized'].min():.3f}, {bias_df['cohens_d_normalized'].max():.3f}]")
print(f"  Normalized |Cohen's d| range: [{bias_df['cohens_d_abs_normalized'].min():.3f}, {bias_df['cohens_d_abs_normalized'].max():.3f}]")

# Save enhanced bias results
enhanced_bias_file = Path('analysis_outputs/bias_analysis/bias_results_enhanced.parquet')
bias_df.to_parquet(enhanced_bias_file)
print(f"\n✓ Saved enhanced bias results to: {enhanced_bias_file}")

# Also save substantive-only version
substantive_bias_df = bias_df[bias_df['feature_category'] == 'substantive'].copy()
substantive_file = Path('analysis_outputs/bias_analysis/bias_results_substantive_only.parquet')
substantive_bias_df.to_parquet(substantive_file)
print(f"✓ Saved substantive-only bias results to: {substantive_file}")
print(f"  ({len(substantive_bias_df)} measurements, {substantive_bias_df['feature'].nunique()} features)")

# ============================================================================
# Load and Enhance Importance Results
# ============================================================================

print("\n" + "="*80)
print("ENHANCING IMPORTANCE RESULTS")
print("="*80)

importance_file = Path('analysis_outputs/importance_analysis/importance_results.parquet')
if not importance_file.exists():
    print(f"ERROR: {importance_file} not found!")
    exit(1)

importance_df = pd.read_parquet(importance_file)
print(f"\nLoaded {len(importance_df)} models")
print(f"Conditions: {len(importance_df['dataset'].unique())} datasets × {len(importance_df['model'].unique())} models × {len(importance_df['prompt_style'].unique())} prompts")

# Find SHAP columns
shap_cols = [c for c in importance_df.columns if c.startswith('shap_') and c != 'shap_file']
print(f"\nFound {len(shap_cols)} SHAP value columns")

# Normalize SHAP values across all conditions
print("\nNormalizing SHAP importance values...")
for col in shap_cols:
    feature_name = col[5:]  # Remove 'shap_' prefix

    # Take absolute values for normalization
    abs_col = f'{col}_abs'
    importance_df[abs_col] = importance_df[col].abs()

    # Min-max normalize
    scaler = MinMaxScaler()
    normalized_col = f'{col}_normalized'
    importance_df[normalized_col] = scaler.fit_transform(importance_df[[abs_col]])

print(f"✓ Created {len(shap_cols)} normalized SHAP columns")

# Create summary of mean SHAP values (raw and normalized)
print("\nComputing feature importance summary...")
shap_summary = []
for col in shap_cols:
    if col == 'shap_file':
        continue

    feature_name = col[5:]
    abs_col = f'{col}_abs'
    norm_col = f'{col}_normalized'

    shap_summary.append({
        'feature': feature_name,
        'mean_shap_raw': importance_df[col].abs().mean(),
        'mean_shap_normalized': importance_df[norm_col].mean(),
        'feature_category': categorize_feature(feature_name)
    })

shap_summary_df = pd.DataFrame(shap_summary).sort_values('mean_shap_raw', ascending=False)
shap_summary_file = Path('analysis_outputs/importance_analysis/shap_summary_with_normalization.csv')
shap_summary_df.to_csv(shap_summary_file, index=False)
print(f"✓ Saved SHAP summary to: {shap_summary_file}")

# Save enhanced importance results
enhanced_importance_file = Path('analysis_outputs/importance_analysis/importance_results_enhanced.parquet')
importance_df.to_parquet(enhanced_importance_file)
print(f"✓ Saved enhanced importance results to: {enhanced_importance_file}")

# ============================================================================
# Generate Comparison Tables
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: RAW VS NORMALIZED RANKINGS")
print("="*80)

print("\nTop 15 features by RAW SHAP importance:")
print("-"*60)
for idx, row in shap_summary_df.head(15).iterrows():
    cat = "STYLISTIC" if row['feature_category'] == 'stylistic' else "substantive"
    print(f"{row['feature']:40s} {row['mean_shap_raw']:6.4f}  [{cat}]")

print("\nTop 15 features by NORMALIZED SHAP importance:")
print("-"*60)
for idx, row in shap_summary_df.sort_values('mean_shap_normalized', ascending=False).head(15).iterrows():
    cat = "STYLISTIC" if row['feature_category'] == 'stylistic' else "substantive"
    print(f"{row['feature']:40s} {row['mean_shap_normalized']:6.4f}  [{cat}]")

print("\nTop substantive features (raw SHAP):")
print("-"*60)
substantive_shap = shap_summary_df[shap_summary_df['feature_category'] == 'substantive']
for idx, row in substantive_shap.head(10).iterrows():
    print(f"{row['feature']:40s} {row['mean_shap_raw']:6.4f}")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"""
Files Created:
  ✓ {enhanced_bias_file}
  ✓ {substantive_file}
  ✓ {enhanced_importance_file}
  ✓ {shap_summary_file}

Bias Analysis:
  - Total measurements: {len(bias_df)}
  - Stylistic features: {len(bias_df[bias_df['feature_category'] == 'stylistic'])} measurements
  - Substantive features: {len(bias_df[bias_df['feature_category'] == 'substantive'])} measurements
  - New columns: cohens_d_normalized, cohens_d_abs_normalized, feature_category

Importance Analysis:
  - Total models: {len(importance_df)}
  - SHAP columns: {len(shap_cols)}
  - New columns: {len(shap_cols)} × 2 (abs and normalized versions)

Feature Categories:
  - Stylistic: {len(STYLISTIC_FEATURES)} base features
  - Substantive: {len(SUBSTANTIVE_FEATURES)} base features
""")

print("="*80)
print("✓ ENHANCEMENT COMPLETE")
print("="*80)
print("\nNext steps:")
print("  1. Update visualization scripts to use enhanced datasets")
print("  2. Generate normalized visualizations")
print("  3. Generate substantive-only analysis track")
print("  4. Focus on fairness features (Issue 4)")
