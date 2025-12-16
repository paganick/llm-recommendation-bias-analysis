"""
Final fixes for visualizations based on user feedback:
1. Include author features in bias heatmaps
2. Fix color scaling (adjust ranges to actual data)
3. Add content features (polarity, sentiment, formality) to fairness deep dive
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'static'

print("="*80)
print("FINAL VISUALIZATION FIXES")
print("="*80)

# Load data
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results_enhanced.parquet')

# ============================================================================
# FIX 1: Bias Heatmaps with Author Features Included
# ============================================================================

print("\n1. Creating bias heatmaps with author + key content features...")

# Define key features to always show
KEY_FEATURES = [
    # Author/fairness features
    'author_gender', 'author_political_leaning', 'author_is_minority',
    # Content features
    'sentiment_polarity', 'sentiment_label', 'formality_score',
    'toxicity', 'polarization_score',
    # Other important
    'primary_topic',
    'text_length', 'word_count'  # For comparison
]

def create_fixed_heatmap(data, features_to_show, title, filename, use_normalized=True):
    """Create heatmap with specified features and proper color scaling"""

    # Filter for specified features
    plot_data = data[data['feature'].isin(features_to_show)].copy()

    if len(plot_data) == 0:
        print(f"  ⚠ No data for {filename}")
        return

    # Choose value column
    value_col = 'cohens_d_abs_normalized' if use_normalized else 'cohens_d'

    # Create pivot
    pivot = plot_data.pivot_table(
        index='feature',
        columns=['dataset', 'model'],
        values=value_col,
        aggfunc='mean'
    )

    # Sort by mean absolute value
    feature_means = pivot.abs().mean(axis=1).sort_values(ascending=False)
    pivot = pivot.reindex(feature_means.index)

    # Determine color scale based on actual data range
    if use_normalized:
        vmin, vmax = pivot.min().min(), pivot.max().max()
        center = (vmin + vmax) / 2
        cmap = 'RdBu_r'
        cbar_label = 'Normalized Effect Size'
    else:
        vmin, vmax = pivot.min().min(), pivot.max().max()
        center = 0
        cmap = 'RdBu_r'
        cbar_label = 'Cohen\'s d'

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.5)))

    sns.heatmap(
        pivot,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cbar_label},
        ax=ax,
        linewidths=0.5,
        annot=False
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset × Model', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)

    # Add note about values
    note = f"Data range: [{vmin:.3f}, {vmax:.3f}]"
    ax.text(0.02, -0.08, note, transform=ax.transAxes, fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ {filename}")
    print(f"    Features: {len(pivot)}, Range: [{vmin:.3f}, {vmax:.3f}]")

# Create comprehensive bias heatmap
available_key_features = [f for f in KEY_FEATURES if f in bias_df['feature'].values]

create_fixed_heatmap(
    bias_df,
    available_key_features,
    'Bias Heatmap: Key Features (Fairness + Content + Stylistic)\nNormalized for comparability',
    VIZ_DIR / 'normalized' / 'bias_heatmap_key_features.png',
    use_normalized=True
)

# Create author + content only (no stylistic)
AUTHOR_CONTENT_FEATURES = [f for f in KEY_FEATURES if f not in ['text_length', 'word_count']]
available_author_content = [f for f in AUTHOR_CONTENT_FEATURES if f in bias_df['feature'].values]

create_fixed_heatmap(
    bias_df,
    available_author_content,
    'Bias Heatmap: Author Demographics + Content Features Only\nNormalized for comparability',
    VIZ_DIR / 'substantive_only' / 'bias_heatmap_author_content.png',
    use_normalized=True
)

# ============================================================================
# FIX 2: Fairness Deep Dive with Content Features + Fixed Scaling
# ============================================================================

print("\n2. Fixing fairness deep dive with content features and proper scaling...")

# Extended fairness features to include content
EXTENDED_FAIRNESS = [
    'author_gender', 'author_political_leaning', 'author_is_minority',
    'sentiment_polarity', 'sentiment_label', 'formality_score',
    'toxicity', 'polarization_score'
]

fairness_extended = bias_df[bias_df['feature'].isin(EXTENDED_FAIRNESS)].copy()

print(f"  Features: {fairness_extended['feature'].unique()}")
print(f"  Measurements: {len(fairness_extended)}")

# Aggregate by key dimensions
def create_extended_fairness_heatmap(data, group_by, title, filename, value_col='cramers_v'):
    """Create fairness heatmap with extended features"""

    # For numerical features, use Cohen's d; for categorical, use Cramer's V
    # Determine which is available for each feature
    pivot_data = []

    for feature in data['feature'].unique():
        feat_data = data[data['feature'] == feature]

        # Choose appropriate metric
        if pd.notna(feat_data['cohens_d']).any():
            # Numerical feature - use normalized Cohen's d
            metric_col = 'cohens_d_abs_normalized'
        else:
            # Categorical feature - use Cramer's V (if available)
            if 'cramers_v' in feat_data.columns and pd.notna(feat_data['cramers_v']).any():
                metric_col = 'cramers_v'
            else:
                continue

        # Aggregate
        if isinstance(group_by, list):
            agg = feat_data.groupby(group_by + ['feature'])[metric_col].mean().reset_index()
        else:
            agg = feat_data.groupby([group_by, 'feature'])[metric_col].mean().reset_index()

        pivot_data.append(agg)

    if not pivot_data:
        print(f"  ⚠ No data for {filename}")
        return

    combined = pd.concat(pivot_data, ignore_index=True)

    # Create pivot
    if isinstance(group_by, list):
        pivot = combined.pivot_table(
            index='feature',
            columns=group_by,
            values=metric_col
        )
    else:
        pivot = combined.pivot(
            index='feature',
            columns=group_by,
            values=metric_col
        )

    # Determine color scale from actual data
    vmin = max(0, pivot.min().min())  # Start at 0 or slightly above
    vmax = pivot.max().max()

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8), max(6, len(pivot) * 0.6)))

    sns.heatmap(
        pivot,
        cmap='YlOrRd',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Effect Size'},
        ax=ax,
        linewidths=1,
        annot=True,
        fmt='.3f',
        annot_kws={'fontsize': 8}
    )

    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Feature', fontsize=10)

    # Add note
    note = f"Mixed metrics: Cramer's V (categorical) & normalized Cohen's d (numerical)"
    ax.text(0.02, -0.12, note, transform=ax.transAxes, fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ {filename}")
    print(f"    Range: [{vmin:.3f}, {vmax:.3f}]")

# By dataset
create_extended_fairness_heatmap(
    fairness_extended,
    'dataset',
    'Extended Fairness Analysis by Dataset\nAuthor Demographics + Content Features',
    VIZ_DIR / 'fairness_deep_dive' / 'extended_fairness_by_dataset.png'
)

# By model
create_extended_fairness_heatmap(
    fairness_extended,
    'model',
    'Extended Fairness Analysis by Model\nAuthor Demographics + Content Features',
    VIZ_DIR / 'fairness_deep_dive' / 'extended_fairness_by_model.png'
)

# By prompt
create_extended_fairness_heatmap(
    fairness_extended,
    'prompt_style',
    'Extended Fairness Analysis by Prompt Style\nAuthor Demographics + Content Features',
    VIZ_DIR / 'fairness_deep_dive' / 'extended_fairness_by_prompt.png'
)

# ============================================================================
# FIX 3: Better Aggregation Level Comparison
# ============================================================================

print("\n3. Creating better aggregation level visualization...")

# Focus on author features only for aggregation comparison
AUTHOR_FEATURES = ['author_gender', 'author_political_leaning', 'author_is_minority']
author_data = bias_df[bias_df['feature'].isin(AUTHOR_FEATURES)].copy()

# Compute aggregation levels using Cramer's V
levels_data = []

# L8: Overall
for feature in AUTHOR_FEATURES:
    feat_data = author_data[author_data['feature'] == feature]
    if len(feat_data) > 0 and 'cramers_v' in feat_data.columns:
        levels_data.append({
            'feature': feature,
            'level': 'L8: Overall',
            'mean': feat_data['cramers_v'].mean(),
            'std': feat_data['cramers_v'].std()
        })

# L6: By model
for feature in AUTHOR_FEATURES:
    for model in author_data['model'].unique():
        feat_model = author_data[(author_data['feature'] == feature) & (author_data['model'] == model)]
        if len(feat_model) > 0 and 'cramers_v' in feat_model.columns:
            levels_data.append({
                'feature': feature,
                'level': f'L6: {model[:15]}',
                'mean': feat_model['cramers_v'].mean(),
                'std': feat_model['cramers_v'].std()
            })

# L5: By dataset
for feature in AUTHOR_FEATURES:
    for dataset in author_data['dataset'].unique():
        feat_dataset = author_data[(author_data['feature'] == feature) & (author_data['dataset'] == dataset)]
        if len(feat_dataset) > 0 and 'cramers_v' in feat_dataset.columns:
            levels_data.append({
                'feature': feature,
                'level': f'L5: {dataset}',
                'mean': feat_dataset['cramers_v'].mean(),
                'std': feat_dataset['cramers_v'].std()
            })

if levels_data:
    levels_df = pd.DataFrame(levels_data)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, feature in enumerate(AUTHOR_FEATURES):
        feat_levels = levels_df[levels_df['feature'] == feature].sort_values('mean', ascending=False)

        if len(feat_levels) == 0:
            continue

        ax = axes[idx]

        y_pos = np.arange(len(feat_levels))
        ax.barh(y_pos, feat_levels['mean'], xerr=feat_levels['std'],
                capsize=4, alpha=0.7, color='steelblue')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_levels['level'], fontsize=9)
        ax.set_xlabel('Cramer\'s V', fontsize=10)
        ax.set_title(feature, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Author Demographics Bias Across Aggregation Levels\n' +
                 'L8=Overall, L6=By Model, L5=By Dataset',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = VIZ_DIR / 'fairness_deep_dive' / 'author_demographics_aggregation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF FIXES")
print("="*80)

print("""
Fixed Visualizations Created:

1. Bias Heatmaps (with author features):
   ✓ normalized/bias_heatmap_key_features.png
   ✓ substantive_only/bias_heatmap_author_content.png

2. Extended Fairness Deep Dive (author + content):
   ✓ fairness_deep_dive/extended_fairness_by_dataset.png
   ✓ fairness_deep_dive/extended_fairness_by_model.png
   ✓ fairness_deep_dive/extended_fairness_by_prompt.png

3. Author Demographics Aggregation:
   ✓ fairness_deep_dive/author_demographics_aggregation.png

Key Improvements:
- Author features (gender, political, minority) now visible in heatmaps
- Color scales adjusted to actual data ranges (no more all-red or all-yellow)
- Content features (polarity, sentiment, formality) added to fairness analysis
- Mixed metrics: Cramer's V for categorical, normalized Cohen's d for numerical
""")

print("="*80)
print("✓ ALL FIXES COMPLETE")
print("="*80)
