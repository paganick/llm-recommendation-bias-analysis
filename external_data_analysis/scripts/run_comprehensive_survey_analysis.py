#!/usr/bin/env python3
"""
Comprehensive Bias Analysis for External Survey Data

Generates the same comprehensive outputs as the main pipeline:
1. Feature distributions (1_distributions/)
2. Bias heatmaps (2_bias_heatmaps/)
3. Directional bias plots (3_directional_bias/)
4. Feature importance analysis (4_feature_importance/)

Adapted for single dataset (survey) with flexible feature detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats
from scipy.stats.contingency import association

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utility functions from main analysis
from run_comprehensive_analysis import (
    compute_cramers_v, compute_cohens_d,
    standardize_categories
)

# Configuration
EXPERIMENT_DIR = Path('external_data_analysis/outputs/experiments/survey_gemini_gemini-2.0-flash')
OUTPUT_DIR = Path('external_data_analysis/outputs/analysis_comprehensive')
VIZ_DIR = OUTPUT_DIR / 'visualizations'

# Create output directories
for subdir in ['1_distributions', '2_bias_heatmaps', '3_directional_bias', '4_feature_importance']:
    (VIZ_DIR / subdir).mkdir(parents=True, exist_ok=True)

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")

# Load experiment data
print_section("LOADING SURVEY EXPERIMENT DATA")
df = pd.read_csv(EXPERIMENT_DIR / 'post_level_data.csv')
print(f"✓ Loaded {len(df)} rows")
print(f"  Selected posts: {df['selected'].sum()}")
print(f"  Prompt styles: {list(df['prompt_style'].unique())}")
print(f"  Total columns: {len(df.columns)}")

# Auto-detect features by type
print_section("AUTO-DETECTING FEATURES")

# Detect categorical features (starting with author_ and non-numerical)
categorical_features = []
numerical_features = []
binary_features = []

for col in df.columns:
    if col in ['selected', 'prompt_style', 'trial_id', 'pool_position', 'original_index', 
               'tweet_id', 'user_id', 'text', 'persona', 'username']:
        continue
    
    if col.startswith('author_') or col.startswith('user_') or col.startswith('has_') or col.startswith('is_'):
        # Check if it's numerical
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's actually binary (only 0/1 or True/False)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2:
                binary_features.append(col)
            # Check if it's count data with few unique values (likely categorical)
            elif len(unique_vals) <= 10 and col.startswith('author_'):
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        else:
            categorical_features.append(col)
    elif col in ['text_length', 'word_count', 'avg_word_length', 'sentiment_polarity', 
                 'sentiment_subjectivity', 'engagement_score', 'polarization_score',
                 'toxicity', 'severe_toxicity']:
        numerical_features.append(col)
    elif col in ['primary_topic', 'controversy_level', 'sentiment_label']:
        categorical_features.append(col)

print(f"Found {len(categorical_features)} categorical features:")
for f in categorical_features:
    print(f"  - {f}")

print(f"\nFound {len(binary_features)} binary features:")
for f in binary_features:
    print(f"  - {f}")

print(f"\nFound {len(numerical_features)} numerical features:")
for f in numerical_features:
    print(f"  - {f}")

# Organize features by category
FEATURES = {
    'categorical': categorical_features,
    'binary': binary_features,
    'numerical': numerical_features
}

all_features = categorical_features + binary_features + numerical_features
print(f"\n✓ Total features to analyze: {len(all_features)}")

# Feature types for analysis
FEATURE_TYPES = {}
for f in categorical_features:
    FEATURE_TYPES[f] = 'categorical'
for f in binary_features:
    FEATURE_TYPES[f] = 'binary'
for f in numerical_features:
    FEATURE_TYPES[f] = 'numerical'

print("\n" + "="*80)
print("STARTING COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# 1. FEATURE DISTRIBUTIONS
# ============================================================================
print_section("1. GENERATING FEATURE DISTRIBUTIONS")

def plot_feature_distribution(feature, feature_type):
    """Plot distribution for pool vs recommended"""
    
    pool_data = df[feature].dropna()
    rec_data = df[df['selected'] == 1][feature].dropna()
    
    if len(rec_data) == 0:
        print(f"  ⚠ Skipping {feature}: no recommended data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if feature_type == 'categorical':
        # Get value counts
        pool_counts = pool_data.value_counts(normalize=True).sort_index()
        rec_counts = rec_data.value_counts(normalize=True).sort_index()
        
        # Combine and fill missing
        all_cats = sorted(set(pool_counts.index) | set(rec_counts.index))
        pool_vals = [pool_counts.get(c, 0) for c in all_cats]
        rec_vals = [rec_counts.get(c, 0) for c in all_cats]
        
        x = np.arange(len(all_cats))
        width = 0.35
        
        ax.bar(x - width/2, pool_vals, width, label='Pool', alpha=0.7, color='skyblue')
        ax.bar(x + width/2, rec_vals, width, label='Recommended', alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, rotation=45, ha='right')
        ax.set_ylabel('Proportion')
        ax.legend()
        
    elif feature_type == 'binary':
        pool_mean = pool_data.mean()
        rec_mean = rec_data.mean()
        
        ax.bar([0, 1], [pool_mean, rec_mean], color=['skyblue', 'coral'], alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pool', 'Recommended'])
        ax.set_ylabel('Proportion = 1')
        ax.set_ylim([0, 1])
        
    else:  # numerical
        ax.hist(pool_data, bins=20, alpha=0.5, label='Pool', color='skyblue', density=True)
        ax.hist(rec_data, bins=20, alpha=0.5, label='Recommended', color='coral', density=True)
        ax.axvline(pool_data.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Pool mean: {pool_data.mean():.2f}')
        ax.axvline(rec_data.mean(), color='red', linestyle='--', alpha=0.7, label=f'Rec mean: {rec_data.mean():.2f}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
    
    ax.set_title(f'{feature}\n(Pool n={len(pool_data)}, Recommended n={len(rec_data)})', 
                 fontweight='bold')
    plt.tight_layout()
    
    output_file = VIZ_DIR / '1_distributions' / f'{feature}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {feature}")

for feature in all_features:
    if feature in df.columns:
        plot_feature_distribution(feature, FEATURE_TYPES[feature])

print(f"\n✓ Saved {len(all_features)} distribution plots to {VIZ_DIR / '1_distributions'}")

# ============================================================================
# 2. BIAS HEATMAPS
# ============================================================================
print_section("2. GENERATING BIAS HEATMAPS")

def compute_bias_with_significance(feature, feature_type):
    """Compute bias metric with significance test"""
    
    pool_vals = df[feature].dropna()
    rec_vals = df[df['selected'] == 1][feature].dropna()
    
    if len(rec_vals) < 5:
        return None
    
    if feature_type in ['categorical', 'binary']:
        # Cramér's V with chi-square test
        bias = compute_cramers_v(pool_vals, rec_vals)
        
        # Chi-square test for significance
        pool_counts = pool_vals.value_counts()
        rec_counts = rec_vals.value_counts()
        all_cats = sorted(set(pool_counts.index) | set(rec_counts.index))
        
        observed = np.array([[pool_counts.get(c, 0), rec_counts.get(c, 0)] for c in all_cats])
        
        if observed.sum() > 0 and observed.shape[0] > 1:
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            except:
                p_value = 1.0
        else:
            p_value = 1.0
            
        return {'bias': bias, 'p_value': p_value, 'metric': 'cramers_v'}
        
    else:  # numerical
        # Cohen's d with t-test
        pool_sample = pool_vals
        not_rec = df[df['selected'] == 0][feature].dropna()
        
        if len(not_rec) < 5:
            return None
        
        bias = compute_cohens_d(rec_vals, not_rec)
        
        try:
            t_stat, p_value = stats.ttest_ind(rec_vals, not_rec)
        except:
            p_value = 1.0
        
        return {'bias': abs(bias), 'p_value': p_value, 'metric': 'cohens_d'}

# Compute bias for all features across all prompt styles
bias_results = []

for feature in all_features:
    if feature not in df.columns:
        continue
    
    print(f"Computing bias for {feature}...")
    
    # Overall bias
    result = compute_bias_with_significance(feature, FEATURE_TYPES[feature])
    if result:
        bias_results.append({
            'feature': feature,
            'prompt_style': 'Overall',
            'bias': result['bias'],
            'p_value': result['p_value'],
            'metric': result['metric'],
            'feature_type': FEATURE_TYPES[feature]
        })
    
    # By prompt style
    for prompt in df['prompt_style'].unique():
        subset = df[df['prompt_style'] == prompt]
        
        pool_vals = subset[feature].dropna()
        rec_vals = subset[subset['selected'] == 1][feature].dropna()
        
        if len(rec_vals) < 3:
            continue
        
        if FEATURE_TYPES[feature] in ['categorical', 'binary']:
            bias = compute_cramers_v(pool_vals, rec_vals)
            bias_results.append({
                'feature': feature,
                'prompt_style': prompt,
                'bias': bias,
                'p_value': np.nan,
                'metric': 'cramers_v',
                'feature_type': FEATURE_TYPES[feature]
            })
        else:
            not_rec = subset[subset['selected'] == 0][feature].dropna()
            if len(not_rec) >= 3:
                bias = abs(compute_cohens_d(rec_vals, not_rec))
                bias_results.append({
                    'feature': feature,
                    'prompt_style': prompt,
                    'bias': bias,
                    'p_value': np.nan,
                    'metric': 'cohens_d',
                    'feature_type': FEATURE_TYPES[feature]
                })

bias_df = pd.DataFrame(bias_results)
bias_df.to_csv(OUTPUT_DIR / 'bias_comprehensive.csv', index=False)
print(f"\n✓ Saved bias results to {OUTPUT_DIR / 'bias_comprehensive.csv'}")

# Create bias heatmap by prompt style
print("\nGenerating bias heatmaps...")

# Heatmap: Features × Prompt Styles
pivot_data = bias_df[bias_df['prompt_style'] != 'Overall'].pivot_table(
    index='feature', columns='prompt_style', values='bias', aggfunc='mean'
)

if len(pivot_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_data) * 0.4)))
    
    # Normalize within each feature for better visualization
    pivot_norm = pivot_data.div(pivot_data.max(axis=1), axis=0).fillna(0)
    
    sns.heatmap(pivot_norm, annot=pivot_data.values, fmt='.3f', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Feature and Prompt Style\n(Normalized within feature, annotations show raw values)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Prompt Style', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '2_bias_heatmaps' / 'bias_by_prompt.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ bias_by_prompt.png")

# Overall bias summary
overall_bias = bias_df[bias_df['prompt_style'] == 'Overall'].sort_values('bias', ascending=False)

fig, ax = plt.subplots(figsize=(8, max(6, len(overall_bias) * 0.3)))
colors = ['coral' if p < 0.05 else 'skyblue' for p in overall_bias['p_value'].fillna(1.0)]
ax.barh(overall_bias['feature'], overall_bias['bias'], color=colors)
ax.set_xlabel('Bias (Cramér\'s V or |Cohen\'s d|)', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title('Overall Bias Across All Prompt Styles\n(Red = p<0.05)', fontweight='bold', fontsize=12)
ax.axvline(0.1, color='red', linestyle='--', alpha=0.3, label='Small effect')
ax.axvline(0.3, color='orange', linestyle='--', alpha=0.3, label='Medium effect')
ax.legend()
plt.tight_layout()
plt.savefig(VIZ_DIR / '2_bias_heatmaps' / 'overall_bias.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ overall_bias.png")

print(f"\n✓ Saved bias heatmaps to {VIZ_DIR / '2_bias_heatmaps'}")

# ============================================================================
# 3. DIRECTIONAL BIAS
# ============================================================================
print_section("3. GENERATING DIRECTIONAL BIAS PLOTS")

def compute_directional_bias_for_feature(feature, feature_type):
    """Compute which categories/values are over/under-represented"""
    
    if feature_type == 'numerical':
        # For numerical: show mean difference
        pool_mean = df[feature].mean()
        rec_mean = df[df['selected'] == 1][feature].mean()
        
        return pd.DataFrame([{
            'category': 'mean',
            'pool_prop': pool_mean,
            'rec_prop': rec_mean,
            'difference': rec_mean - pool_mean
        }])
    
    # For categorical/binary
    directional_data = []
    
    for prompt in df['prompt_style'].unique():
        subset = df[df['prompt_style'] == prompt]
        
        pool_counts = subset[feature].value_counts(normalize=True)
        rec_counts = subset[subset['selected'] == 1][feature].value_counts(normalize=True)
        
        for category in pool_counts.index:
            pool_prop = pool_counts.get(category, 0)
            rec_prop = rec_counts.get(category, 0)
            
            directional_data.append({
                'feature': feature,
                'prompt_style': prompt,
                'category': str(category),
                'pool_prop': pool_prop,
                'rec_prop': rec_prop,
                'difference': rec_prop - pool_prop
            })
    
    return pd.DataFrame(directional_data)

directional_results = []

for feature in categorical_features + binary_features:
    if feature not in df.columns:
        continue
    
    print(f"Computing directional bias for {feature}...")
    result_df = compute_directional_bias_for_feature(feature, FEATURE_TYPES[feature])
    directional_results.append(result_df)

if directional_results:
    directional_df = pd.concat(directional_results, ignore_index=True)
    directional_df.to_csv(OUTPUT_DIR / 'directional_bias.csv', index=False)
    print(f"\n✓ Saved directional bias to {OUTPUT_DIR / 'directional_bias.csv'}")
    
    # Plot directional bias for each categorical feature
    for feature in categorical_features + binary_features:
        if feature not in df.columns:
            continue
        
        feature_data = directional_df[directional_df['feature'] == feature]
        
        if len(feature_data) == 0:
            continue
        
        # Create pivot table
        pivot = feature_data.pivot_table(
            index='category', columns='prompt_style', values='difference', aggfunc='mean'
        )
        
        if len(pivot) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    vmin=-0.5, vmax=0.5, ax=ax, cbar_kws={'label': 'Rec % - Pool %'})
        ax.set_title(f'Directional Bias: {feature}\n(Positive = Over-represented in recommendations)',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('Prompt Style', fontsize=11)
        ax.set_ylabel('Category', fontsize=11)
        plt.tight_layout()
        
        plt.savefig(VIZ_DIR / '3_directional_bias' / f'{feature}_directional.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {feature}_directional.png")
    
    print(f"\n✓ Saved directional bias plots to {VIZ_DIR / '3_directional_bias'}")

# ============================================================================
# 4. FEATURE IMPORTANCE (Simple version without ML models)
# ============================================================================
print_section("4. GENERATING FEATURE IMPORTANCE ANALYSIS")

# Simple feature importance based on correlation with selection
importance_data = []

for feature in all_features:
    if feature not in df.columns:
        continue
    
    try:
        if FEATURE_TYPES[feature] in ['categorical', 'binary']:
            # Use Cramér's V as importance
            importance = compute_cramers_v(df[feature].dropna(), 
                                          df['selected'].dropna())
        else:
            # Use absolute correlation as importance
            correlation = df[[feature, 'selected']].corr().iloc[0, 1]
            importance = abs(correlation)
        
        importance_data.append({
            'feature': feature,
            'importance': importance,
            'feature_type': FEATURE_TYPES[feature]
        })
        
    except Exception as e:
        print(f"  ⚠ Error computing importance for {feature}: {e}")
        continue

importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

print(f"✓ Saved feature importance to {OUTPUT_DIR / 'feature_importance.csv'}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, max(8, len(importance_df) * 0.35)))

colors = ['steelblue' if ft in ['categorical', 'binary'] else 'coral' 
          for ft in importance_df['feature_type']]

ax.barh(importance_df['feature'], importance_df['importance'], color=colors, alpha=0.7)
ax.set_xlabel('Importance (Association with Selection)', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title('Feature Importance for Recommendation Selection\n(Blue=Categorical, Orange=Numerical)',
             fontweight='bold', fontsize=12)
ax.axvline(0.1, color='red', linestyle='--', alpha=0.3, label='Small effect')
ax.axvline(0.3, color='orange', linestyle='--', alpha=0.3, label='Medium effect')
ax.legend()
plt.tight_layout()

plt.savefig(VIZ_DIR / '4_feature_importance' / 'overall_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved feature importance plot to {VIZ_DIR / '4_feature_importance'}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print_section("ANALYSIS COMPLETE - SUMMARY")

print(f"📊 Generated Comprehensive Analysis for Survey Data:")
print(f"\n1. Feature Distributions: {len(all_features)} plots")
print(f"   → {VIZ_DIR / '1_distributions'}")

print(f"\n2. Bias Heatmaps: 2 plots")
print(f"   → {VIZ_DIR / '2_bias_heatmaps'}")
print(f"   - bias_by_prompt.png")
print(f"   - overall_bias.png")

print(f"\n3. Directional Bias: {len(categorical_features + binary_features)} plots")
print(f"   → {VIZ_DIR / '3_directional_bias'}")

print(f"\n4. Feature Importance: 1 plot")
print(f"   → {VIZ_DIR / '4_feature_importance'}")

print(f"\n📈 Data Files:")
print(f"   - {OUTPUT_DIR / 'bias_comprehensive.csv'}")
print(f"   - {OUTPUT_DIR / 'directional_bias.csv'}")
print(f"   - {OUTPUT_DIR / 'feature_importance.csv'}")

print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL ANALYSIS COMPLETE!{Colors.ENDC}")
print(f"{Colors.GREEN}Total plots generated: {len(all_features) + 2 + len(categorical_features + binary_features) + 1}{Colors.ENDC}\n")
