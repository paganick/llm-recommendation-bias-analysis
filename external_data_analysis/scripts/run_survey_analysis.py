#!/usr/bin/env python3
"""
Run bias analysis on survey data experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.stats import chi2_contingency

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from main analysis
from run_comprehensive_analysis import compute_cramers_v, compute_cohens_d

# Load experiment data
exp_file = 'external_data_analysis/outputs/experiments/survey_gemini_gemini-2.0-flash/post_level_data.csv'
df = pd.read_csv(exp_file)

print("="*80)
print("SURVEY DATA BIAS ANALYSIS")
print("="*80)
print(f"\nLoaded {len(df)} rows from experiment")
print(f"Selected posts: {df['selected'].sum()}")
print(f"Prompt styles: {df['prompt_style'].unique()}")

# Define features to analyze
categorical_features = [
    'author_gender', 'author_political_leaning', 'author_is_minority',
    'author_partisanship', 'author_race', 'author_education', 'author_marital_status'
]

numerical_features = [
    'author_age', 'text_length', 'sentiment_polarity', 'engagement_score'
]

# Compute bias for categorical features
print("\n" + "="*80)
print("CATEGORICAL FEATURE BIAS (Cramér's V)")
print("="*80)

bias_results = []

for feature in categorical_features:
    if feature not in df.columns:
        continue
    
    # Get pool and selected data
    pool_vals = df[feature].dropna()
    rec_vals = df[df['selected'] == 1][feature].dropna()
    
    if len(rec_vals) > 0 and len(pool_vals) > 0:
        cramers_v_val = compute_cramers_v(pool_vals, rec_vals)
        
        bias_results.append({
            'feature': feature,
            'cramers_v': cramers_v_val,
            'type': 'categorical'
        })
        print(f"{feature:30s}: Cramér's V = {cramers_v_val:.4f}")

print("\n" + "="*80)
print("NUMERICAL FEATURE BIAS (Cohen's d)")  
print("="*80)

for feature in numerical_features:
    if feature not in df.columns:
        continue
    
    # Cohen's d for numerical features
    selected = df[df['selected'] == 1][feature].dropna()
    not_selected = df[df['selected'] == 0][feature].dropna()
    
    if len(selected) > 0 and len(not_selected) > 0:
        cohens_d_val = compute_cohens_d(selected, not_selected)
        bias_results.append({
            'feature': feature,
            'cohens_d': cohens_d_val,
            'type': 'numerical'
        })
        print(f"{feature:30s}: Cohen's d = {cohens_d_val:.4f}")
        print(f"  Selected mean: {selected.mean():.2f}, Not selected mean: {not_selected.mean():.2f}")

# Save results
output_dir = Path('external_data_analysis/outputs/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(bias_results)
results_df.to_csv(output_dir / 'bias_summary.csv', index=False)

print("\n" + "="*80)
print(f"✓ Results saved to {output_dir / 'bias_summary.csv'}")
print("="*80)

# Analyze by prompt style
print("\n" + "="*80)
print("BIAS BY PROMPT STYLE (Gender example)")
print("="*80)

if 'author_gender' in df.columns:
    for prompt_style in df['prompt_style'].unique():
        subset = df[df['prompt_style'] == prompt_style]
        pool_vals = subset['author_gender'].dropna()
        rec_vals = subset[subset['selected'] == 1]['author_gender'].dropna()
        
        if len(rec_vals) > 0 and len(pool_vals) > 0:
            cv = compute_cramers_v(pool_vals, rec_vals)
            print(f"{prompt_style:20s}: Cramér's V = {cv:.4f}")

print("\n✓ Analysis complete!")
