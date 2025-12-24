#!/usr/bin/env python3
"""
Comprehensive Survey Data Analysis - Generates ALL Heatmaps

Matches main pipeline output structure:
- 10 bias heatmaps (6 disaggregated by prompt + 4 aggregated)
- 10 feature importance heatmaps (6 disaggregated by prompt + 4 aggregated)

Auto-detects features and supports multiple models/providers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from main analysis
from run_comprehensive_analysis import (
    compute_cramers_v, compute_cohens_d,
    standardize_categories
)

print("="*80)
print("COMPREHENSIVE SURVEY DATA ANALYSIS")
print("="*80)

# Configuration
EXPERIMENTS_BASE = Path('external_data_analysis/outputs/experiments')
OUTPUT_DIR = Path('external_data_analysis/outputs/analysis_full')
VIZ_DIR = OUTPUT_DIR / 'visualizations'

# Clean and create directories
import shutil
if VIZ_DIR.exists():
    print("\n🧹 Cleaning old visualizations...")
    shutil.rmtree(VIZ_DIR)

for subdir in ['1_distributions', '2_bias_heatmaps', '3_directional_bias', '4_feature_importance']:
    (VIZ_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {subdir}/")

# Auto-detect experiments
exp_dirs = list(EXPERIMENTS_BASE.glob('survey_*'))
if len(exp_dirs) == 0:
    print("\nERROR: No survey experiments found!")
    sys.exit(1)

print(f"\nFound {len(exp_dirs)} experiment(s):")
experiments = []
for exp_dir in exp_dirs:
    data_file = exp_dir / 'post_level_data.csv'
    if data_file.exists():
        parts = exp_dir.name.split('_', 2)
        provider = parts[1] if len(parts) >= 2 else 'unknown'
        model = parts[2] if len(parts) >= 3 else exp_dir.name

        experiments.append({
            'provider': provider,
            'model': model,
            'file': data_file,
            'name': exp_dir.name
        })
        print(f"  ✓ {exp_dir.name}")

# Load all data
all_data = []
for exp in experiments:
    df_exp = pd.read_csv(exp['file'])
    df_exp['provider'] = exp['provider']
    df_exp['model'] = exp['model']
    df_exp['dataset'] = 'survey'  # Single dataset
    all_data.append(df_exp)

df = pd.concat(all_data, ignore_index=True)
print(f"\n✓ Combined: {len(df)} rows, {df['selected'].sum()} selected")
print(f"  Models: {df['model'].unique().tolist()}")
print(f"  Prompts: {df['prompt_style'].unique().tolist()}")

# Auto-detect features from the data
# Exclude system columns
SYSTEM_COLUMNS = {
    'tweet_id', 'user_id', 'text', 'persona', 'selected', 'pool_position',
    'original_index', 'prompt_style', 'trial_id', 'provider', 'model', 'dataset'
}

# Get all columns that aren't system columns
all_columns = set(df.columns) - SYSTEM_COLUMNS

# Categorize features based on their dtype and name patterns
categorical_features = []
numerical_features = []

for col in sorted(all_columns):
    # Binary/categorical indicators
    if col.startswith('has_') or col.startswith('is_') or col.startswith('user_verified'):
        categorical_features.append(col)
    # Categorical features
    elif any(keyword in col for keyword in ['author_gender', 'author_ideology', 'author_partisanship',
                                              'author_race', 'author_income', 'author_education',
                                              'author_marital_status', 'author_religiosity',
                                              'author_political_leaning', 'author_is_minority',
                                              'primary_topic', 'controversy_level']):
        categorical_features.append(col)
    # Numerical features
    elif df[col].dtype in ['int64', 'float64']:
        numerical_features.append(col)

available_features = categorical_features + numerical_features
print(f"\n✓ Detected {len(available_features)} features")
print(f"  - {len(numerical_features)} numerical: {', '.join(numerical_features[:5])}{'...' if len(numerical_features) > 5 else ''}")
print(f"  - {len(categorical_features)} categorical: {', '.join(categorical_features[:5])}{'...' if len(categorical_features) > 5 else ''}")

#==============================================================================
# BIAS ANALYSIS
#==============================================================================

print("\n" + "="*80)
print("COMPUTING BIAS METRICS")
print("="*80)

bias_results = []

# All combinations
datasets = df['dataset'].unique()
models = df['model'].unique()
prompts = df['prompt_style'].unique()

for dataset in datasets:
    for model in models:
        for prompt in prompts:
            subset = df[
                (df['dataset'] == dataset) &
                (df['model'] == model) &
                (df['prompt_style'] == prompt)
            ]

            if len(subset) < 10:
                continue

            selected = subset[subset['selected'] == True]
            not_selected = subset[subset['selected'] == False]

            if len(selected) < 5 or len(not_selected) < 5:
                continue

            print(f"\n{dataset} | {model} | {prompt}: {len(selected)}/{len(subset)}")

            for feature in available_features:
                if feature in categorical_features:
                    # Cramér's V
                    selected_vals = standardize_categories(selected[feature], feature)
                    not_selected_vals = standardize_categories(not_selected[feature], feature)

                    effect_size = compute_cramers_v(not_selected_vals, selected_vals)

                else:
                    # Cohen's d
                    sel_clean = selected[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    not_sel_clean = not_selected[feature].replace([np.inf, -np.inf], np.nan).dropna()

                    if len(sel_clean) < 5 or len(not_sel_clean) < 5:
                        continue

                    cohens_d = compute_cohens_d(sel_clean.values, not_sel_clean.values)
                    effect_size = abs(cohens_d)

                bias_results.append({
                    'dataset': dataset,
                    'model': model,
                    'prompt': prompt,
                    'feature': feature,
                    'effect_size': effect_size,
                    'is_categorical': feature in categorical_features
                })

df_bias = pd.DataFrame(bias_results)
print(f"\n✓ Computed {len(df_bias)} bias measurements")

#==============================================================================
# FEATURE IMPORTANCE (SHAP-like)
#==============================================================================

print("\n" + "="*80)
print("COMPUTING FEATURE IMPORTANCE")
print("="*80)

importance_results = []

for dataset in datasets:
    for model in models:
        for prompt in prompts:
            subset = df[
                (df['dataset'] == dataset) &
                (df['model'] == model) &
                (df['prompt_style'] == prompt)
            ]

            if len(subset) < 10:
                continue

            print(f"\n{dataset} | {model} | {prompt}")

            for feature in available_features:
                if feature in categorical_features:
                    try:
                        contingency = pd.crosstab(
                            standardize_categories(subset[feature], feature),
                            subset['selected']
                        )

                        # Skip if no variance
                        if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
                            continue

                        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                        importance = chi2 / len(subset)
                    except Exception as e:
                        # Skip features with errors (e.g., insufficient data)
                        continue

                else:
                    selected = subset[subset['selected'] == True][feature]
                    not_selected = subset[subset['selected'] == False][feature]

                    sel_clean = selected.replace([np.inf, -np.inf], np.nan).dropna()
                    not_sel_clean = not_selected.replace([np.inf, -np.inf], np.nan).dropna()

                    if len(sel_clean) < 5 or len(not_sel_clean) < 5:
                        continue

                    _, p_value = stats.mannwhitneyu(sel_clean, not_sel_clean, alternative='two-sided')

                    cohens_d = compute_cohens_d(sel_clean.values, not_sel_clean.values)
                    importance = abs(cohens_d)

                importance_results.append({
                    'dataset': dataset,
                    'model': model,
                    'prompt': prompt,
                    'feature': feature,
                    'importance': importance
                })

df_importance = pd.DataFrame(importance_results)
print(f"\n✓ Computed {len(df_importance)} importance measurements")

#==============================================================================
# GENERATE 20 HEATMAPS
#==============================================================================

print("\n" + "="*80)
print("GENERATING HEATMAPS")
print("="*80)

def create_heatmap(data, title, output_file, xlabel='', ylabel='Feature',
                   cmap='YlOrRd', vmin=0, vmax=None, fmt='.3f'):
    """Create a heatmap with consistent styling"""
    plt.figure(figsize=(max(8, len(data.columns) * 0.8), max(6, len(data) * 0.4)))

    if vmax is None:
        vmax = data.values.max()

    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                vmin=vmin, vmax=vmax, cbar_kws={'label': title})

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_file.name}")

#------------------------------------------------------------------------------
# 10 BIAS HEATMAPS
#------------------------------------------------------------------------------

print("\n📊 Generating 10 Bias Heatmaps...")
bias_dir = VIZ_DIR / '2_bias_heatmaps'

# 1-6. Disaggregated by prompt
for prompt in prompts:
    data = df_bias[df_bias['prompt'] == prompt].copy()

    pivot = data.pivot_table(
        values='effect_size',
        index='feature',
        columns=['dataset', 'model'],
        aggfunc='mean'
    )

    # Flatten multi-index columns
    pivot.columns = [f"{d}_{m}" for d, m in pivot.columns]
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)

    create_heatmap(
        pivot,
        f'Bias Effect Sizes - {prompt.title()} Prompt',
        bias_dir / f'disaggregated_prompt_{prompt}.png',
        xlabel='Dataset × Model'
    )

# 7. Aggregated by dataset
agg_dataset = df_bias.groupby(['dataset', 'feature'])['effect_size'].mean().unstack(level=0).T
agg_dataset = agg_dataset.reindex(columns=agg_dataset.mean().sort_values(ascending=False).index)

create_heatmap(
    agg_dataset,
    'Bias Effect Sizes - Aggregated by Dataset',
    bias_dir / 'aggregated_by_dataset.png',
    xlabel='Feature'
)

# 8. Aggregated by model
agg_model = df_bias.groupby(['model', 'feature'])['effect_size'].mean().unstack(level=0).T
agg_model = agg_model.reindex(columns=agg_model.mean().sort_values(ascending=False).index)

create_heatmap(
    agg_model,
    'Bias Effect Sizes - Aggregated by Model',
    bias_dir / 'aggregated_by_model.png',
    xlabel='Feature'
)

# 9. Aggregated by prompt
agg_prompt = df_bias.groupby(['prompt', 'feature'])['effect_size'].mean().unstack(level=0).T
agg_prompt = agg_prompt.reindex(columns=agg_prompt.mean().sort_values(ascending=False).index)

create_heatmap(
    agg_prompt,
    'Bias Effect Sizes - Aggregated by Prompt Style',
    bias_dir / 'aggregated_by_prompt.png',
    xlabel='Feature'
)

# 10. Fully aggregated
fully_agg = df_bias.groupby('feature')['effect_size'].mean().sort_values(ascending=False)
fully_agg_df = pd.DataFrame({'Overall': fully_agg})

create_heatmap(
    fully_agg_df,
    'Bias Effect Sizes - Fully Aggregated',
    bias_dir / 'fully_aggregated.png',
    xlabel=''
)

#------------------------------------------------------------------------------
# 10 FEATURE IMPORTANCE HEATMAPS
#------------------------------------------------------------------------------

print("\n📊 Generating 10 Feature Importance Heatmaps...")
imp_dir = VIZ_DIR / '4_feature_importance'

# 1-6. Disaggregated by prompt
for prompt in prompts:
    data = df_importance[df_importance['prompt'] == prompt].copy()

    pivot = data.pivot_table(
        values='importance',
        index='feature',
        columns=['dataset', 'model'],
        aggfunc='mean'
    )

    pivot.columns = [f"{d}_{m}" for d, m in pivot.columns]
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=False)

    create_heatmap(
        pivot,
        f'Feature Importance - {prompt.title()} Prompt',
        imp_dir / f'disaggregated_prompt_{prompt}.png',
        xlabel='Dataset × Model',
        cmap='viridis'
    )

# 7. Aggregated by dataset
agg_dataset_imp = df_importance.groupby(['dataset', 'feature'])['importance'].mean().unstack(level=0).T
agg_dataset_imp = agg_dataset_imp.reindex(columns=agg_dataset_imp.mean().sort_values(ascending=False).index)

create_heatmap(
    agg_dataset_imp,
    'Feature Importance - Aggregated by Dataset',
    imp_dir / 'aggregated_by_dataset.png',
    xlabel='Feature',
    cmap='viridis'
)

# 8. Aggregated by model
agg_model_imp = df_importance.groupby(['model', 'feature'])['importance'].mean().unstack(level=0).T
agg_model_imp = agg_model_imp.reindex(columns=agg_model_imp.mean().sort_values(ascending=False).index)

create_heatmap(
    agg_model_imp,
    'Feature Importance - Aggregated by Model',
    imp_dir / 'aggregated_by_model.png',
    xlabel='Feature',
    cmap='viridis'
)

# 9. Aggregated by prompt
agg_prompt_imp = df_importance.groupby(['prompt', 'feature'])['importance'].mean().unstack(level=0).T
agg_prompt_imp = agg_prompt_imp.reindex(columns=agg_prompt_imp.mean().sort_values(ascending=False).index)

create_heatmap(
    agg_prompt_imp,
    'Feature Importance - Aggregated by Prompt Style',
    imp_dir / 'aggregated_by_prompt.png',
    xlabel='Feature',
    cmap='viridis'
)

# 10. Fully aggregated
fully_agg_imp = df_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
fully_agg_imp_df = pd.DataFrame({'Overall': fully_agg_imp})

create_heatmap(
    fully_agg_imp_df,
    'Feature Importance - Fully Aggregated',
    imp_dir / 'fully_aggregated.png',
    xlabel='',
    cmap='viridis'
)

#==============================================================================
# DISTRIBUTION PLOTS
#==============================================================================

print("\n" + "="*80)
print("GENERATING DISTRIBUTION PLOTS")
print("="*80)

dist_dir = VIZ_DIR / '1_distributions'

for feature in available_features:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if feature in categorical_features:
            # Pool distribution
            pool_counts = df[feature].value_counts()
            axes[0].bar(range(len(pool_counts)), pool_counts.values)
            axes[0].set_xticks(range(len(pool_counts)))
            axes[0].set_xticklabels(pool_counts.index, rotation=45, ha='right')
            axes[0].set_title(f'{feature} - Pool Distribution')
            axes[0].set_ylabel('Count')

            # Selected distribution
            selected_df = df[df['selected'] == True]
            selected_counts = selected_df[feature].value_counts()
            axes[1].bar(range(len(selected_counts)), selected_counts.values, color='orange')
            axes[1].set_xticks(range(len(selected_counts)))
            axes[1].set_xticklabels(selected_counts.index, rotation=45, ha='right')
            axes[1].set_title(f'{feature} - Selected Distribution')
            axes[1].set_ylabel('Count')

        else:
            # Numerical feature - histograms
            axes[0].hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'{feature} - Pool Distribution')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Count')

            selected_vals = df[df['selected'] == True][feature].dropna()
            axes[1].hist(selected_vals, bins=30, edgecolor='black', alpha=0.7, color='orange')
            axes[1].set_title(f'{feature} - Selected Distribution')
            axes[1].set_xlabel(feature)
            axes[1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(dist_dir / f'{feature}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"  ⚠ Skipped {feature}: {str(e)}")
        continue

print(f"  ✓ Generated {len(list(dist_dir.glob('*.png')))} distribution plots")

#==============================================================================
# DIRECTIONAL BIAS HEATMAPS
#==============================================================================

print("\n" + "="*80)
print("COMPUTING DIRECTIONAL BIAS")
print("="*80)

directional_data = []

for dataset in datasets:
    for model in models:
        for prompt in prompts:
            subset = df[
                (df['dataset'] == dataset) &
                (df['model'] == model) &
                (df['prompt_style'] == prompt)
            ]

            if len(subset) < 10:
                continue

            pool = subset[subset['selected'] == False]
            selected = subset[subset['selected'] == True]

            if len(pool) < 5 or len(selected) < 5:
                continue

            for feature in available_features:
                if feature in categorical_features:
                    # For each category, compute prop_selected - prop_pool
                    categories = subset[feature].dropna().unique()
                    for category in categories:
                        prop_pool = (pool[feature] == category).sum() / len(pool) if len(pool) > 0 else 0
                        prop_selected = (selected[feature] == category).sum() / len(selected) if len(selected) > 0 else 0
                        dir_bias = prop_selected - prop_pool

                        directional_data.append({
                            'dataset': dataset,
                            'model': model,
                            'prompt': prompt,
                            'feature': feature,
                            'category': str(category),
                            'directional_bias': dir_bias,
                            'feature_type': 'categorical'
                        })
                else:
                    # For numerical, compute mean_selected - mean_pool
                    mean_pool = pool[feature].mean()
                    mean_selected = selected[feature].mean()
                    dir_bias = mean_selected - mean_pool

                    # Normalize by pool std for visualization
                    std_pool = pool[feature].std()
                    if std_pool > 0:
                        dir_bias_normalized = dir_bias / std_pool
                    else:
                        dir_bias_normalized = 0

                    directional_data.append({
                        'dataset': dataset,
                        'model': model,
                        'prompt': prompt,
                        'feature': feature,
                        'category': 'mean_difference',
                        'directional_bias': dir_bias_normalized,
                        'feature_type': 'numerical'
                    })

df_directional = pd.DataFrame(directional_data)
print(f"\n✓ Computed directional bias for {len(df_directional)} feature-category-condition combinations")

#==============================================================================
# DIRECTIONAL BIAS HEATMAPS
#==============================================================================

print("\n" + "="*80)
print("GENERATING DIRECTIONAL BIAS HEATMAPS")
print("="*80)

dir_bias_dir = VIZ_DIR / '3_directional_bias'

for feature in available_features:
    try:
        feature_data = df_directional[df_directional['feature'] == feature]
        if len(feature_data) == 0:
            continue

        # Create heatmap by prompt style
        pivot = feature_data.pivot_table(
            values='directional_bias',
            index='category',
            columns='prompt',
            aggfunc='mean'
        )

        plt.figure(figsize=(max(10, len(prompts) * 1.5), max(6, len(pivot) * 0.5)))

        # Use diverging colormap (red = under-represented, blue = over-represented)
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                   center=0, vmin=-0.3, vmax=0.3,
                   cbar_kws={'label': 'Directional Bias\n(Selected - Pool)'})

        plt.title(f'Directional Bias: {feature}\n(By Prompt Style)', fontsize=14, pad=20)
        plt.xlabel('Prompt Style', fontsize=11)
        plt.ylabel('Category' if feature in categorical_features else 'Metric', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(dir_bias_dir / f'{feature}_by_prompt.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"  ⚠ Skipped {feature}: {str(e)}")
        continue

print(f"  ✓ Generated {len(list(dir_bias_dir.glob('*.png')))} directional bias heatmaps")

#==============================================================================
# SUMMARY
#==============================================================================

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📊 Generated visualizations:")
print(f"  • {len(list(dist_dir.glob('*.png')))} distribution plots in: {dist_dir}")
print(f"  • 10 bias heatmaps in: {bias_dir}")
print(f"  • {len(list(dir_bias_dir.glob('*.png')))} directional bias plots in: {dir_bias_dir}")
print(f"  • 10 feature importance heatmaps in: {imp_dir}")
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print("\nAll visualizations match main pipeline structure!")
