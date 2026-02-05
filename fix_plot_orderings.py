#!/usr/bin/env python3
"""
Fix Plot Orderings
==================

This script:
1. Regenerates the fully_aggregated_bar_plot with correct feature ordering (Author at top)
2. Ensures consistent model ordering across plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Configuration
FEATURES = {
    'author': ['author_gender', 'author_political_leaning', 'author_is_minority'],
    'text_metrics': ['text_length', 'avg_word_length'],
    'sentiment': ['sentiment_polarity', 'sentiment_subjectivity'],
    'style': ['has_emoji', 'has_hashtag', 'has_mention', 'has_url'],
    'content': ['polarization_score', 'controversy_level', 'primary_topic'],
    'toxicity': ['toxicity', 'severe_toxicity']
}

FEATURE_DISPLAY_NAMES = {
    'author_gender': 'Author: Gender',
    'author_political_leaning': 'Author: Political Leaning',
    'author_is_minority': 'Author: Is Minority',
    'text_length': 'Text: Length (chars)',
    'avg_word_length': 'Text: Avg Word Length',
    'polarization_score': 'Content: Polarization Score',
    'controversy_level': 'Content: Controversy Level',
    'primary_topic': 'Content: Primary Topic',
    'sentiment_polarity': 'Sentiment: Polarity',
    'sentiment_subjectivity': 'Sentiment: Subjectivity',
    'has_emoji': 'Style: Has Emoji',
    'has_hashtag': 'Style: Has Hashtag',
    'has_mention': 'Style: Has Mention',
    'has_url': 'Style: Has URL',
    'toxicity': 'Toxicity: Toxicity',
    'severe_toxicity': 'Toxicity: Severe Toxicity'
}

CATEGORY_COLORS = {
    'author': ['#8B4513', '#A0522D', '#CD853F'],
    'text_metrics': ['#1E90FF', '#4169E1'],
    'content': ['#32CD32', '#3CB371', '#2E8B57'],
    'sentiment': ['#FFD700', '#FFA500'],
    'style': ['#9370DB', '#8A2BE2', '#9400D3', '#9932CC'],
    'toxicity': ['#DC143C', '#B22222']
}

OUTPUT_DIR = Path('analysis_outputs')
HEATMAP_RAW_DIR = OUTPUT_DIR / 'visualizations' / '2_bias_heatmaps_raw'

def format_feature_name(feature):
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())

def sort_features_by_type(features):
    feature_order = sum(FEATURES.values(), [])
    return [f for f in feature_order if f in features]

def get_feature_category(feature):
    for category, feats in FEATURES.items():
        if feature in feats:
            return category
    return 'other'

def get_feature_color(feature, idx_within_category=0):
    category = get_feature_category(feature)
    colors = CATEGORY_COLORS.get(category, ['#888888'])
    return colors[idx_within_category % len(colors)]

def cohens_d_to_r_squared(d):
    return (d ** 2) / (d ** 2 + 4)

def cramers_v_to_r_squared(v):
    return v ** 2

def convert_to_r_squared(row):
    if pd.isna(row['bias']) or pd.isna(row['metric']):
        return np.nan
    abs_bias = abs(row['bias'])
    if row['metric'] == "Cohen's d":
        return cohens_d_to_r_squared(abs_bias)
    elif row['metric'] == "Cramér's V":
        return cramers_v_to_r_squared(abs_bias)
    else:
        return np.nan

def create_aggregated_bar_plot_fixed(comp_df):
    """
    Create bar plot with CORRECT ordering (Author features at TOP).
    """
    print("\nRegenerating fully aggregated bar plot with correct ordering...")

    # Calculate average R² per feature
    agg_full = comp_df.groupby('feature').agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    # Sort features by type
    agg_full = agg_full.set_index('feature')
    agg_full = agg_full.reindex(sort_features_by_type(agg_full.index.tolist()))

    # REVERSE the order so Author features appear at TOP in horizontal bar chart
    agg_full = agg_full[::-1]

    agg_full['feature_display'] = [format_feature_name(f) for f in agg_full.index]
    agg_full['category'] = [get_feature_category(f) for f in agg_full.index]

    # Assign colors based on category
    category_idx = {}
    colors = []
    for feature in agg_full.index:
        category = get_feature_category(feature)
        if category not in category_idx:
            category_idx[category] = 0
        colors.append(get_feature_color(feature, category_idx[category]))
        category_idx[category] += 1

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(agg_full['feature_display'], agg_full['r_squared'],
                   color=colors, edgecolor='black', alpha=0.8, linewidth=0.5)

    # Add significance markers
    for i, (idx, row) in enumerate(agg_full.iterrows()):
        if row['significant'] > 0.75:
            ax.text(row['r_squared'], i, ' ***', va='center', fontsize=10, fontweight='bold')
        elif row['significant'] > 0.60:
            ax.text(row['r_squared'], i, ' **', va='center', fontsize=10, fontweight='bold')
        elif row['significant'] > 0.50:
            ax.text(row['r_squared'], i, ' *', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Average R² (Variance Explained)\nAcross All Datasets, Models & Prompts', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title('Average Bias per Feature (R²)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)

    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = []
    for category, features in FEATURES.items():
        if features:
            color = get_feature_color(features[0], 0)
            legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                        label=category.replace('_', ' ').title()))
    ax.legend(handles=legend_elements, loc='lower right', title='Feature Category')

    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'fully_aggregated_bar_plot.png', bbox_inches='tight')
    plt.close()

    print("✓ Fixed bar plot saved!")

def main():
    print("\n" + "="*80)
    print("FIXING PLOT ORDERINGS")
    print("="*80)

    # Load data
    comp_data_file = OUTPUT_DIR / 'pool_vs_recommended_summary.csv'
    if not comp_data_file.exists():
        print(f"ERROR: Comparison data not found at {comp_data_file}")
        return

    comp_df = pd.read_csv(comp_data_file)
    comp_df['r_squared'] = comp_df.apply(convert_to_r_squared, axis=1)

    print(f"Loaded {len(comp_df)} comparisons\n")

    # Fix the bar plot
    create_aggregated_bar_plot_fixed(comp_df)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == '__main__':
    main()
