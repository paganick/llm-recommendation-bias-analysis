#!/usr/bin/env python3
"""
Regenerate Visualizations with Updated Styling

This script loads cached analysis data and regenerates all visualizations
with updated colors, labels, and formatting. It does NOT recompute the
analysis results, allowing fast iteration on visualization design.

Usage:
    python regenerate_visualizations.py

Dependencies:
    - Requires cached data files in analysis_outputs/:
      - comparison_data.parquet
      - directional_bias_data.parquet
      - feature_importance_data.csv (or importance_analysis/importance_results.parquet)

To recompute the analysis data, run:
    python run_comprehensive_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION (imported from main analysis)
# ============================================================================

# 16 Core Features (grouped by category)
FEATURES = {
    'author': ['author_gender', 'author_political_leaning', 'author_is_minority'],
    'text_metrics': ['text_length', 'avg_word_length'],
    'sentiment': ['sentiment_polarity', 'sentiment_subjectivity'],
    'style': ['has_emoji', 'has_hashtag', 'has_mention', 'has_url'],
    'content': ['polarization_score', 'controversy_level', 'primary_topic'],
    'toxicity': ['toxicity', 'severe_toxicity']
}

# Feature types
FEATURE_TYPES = {
    'author_gender': 'categorical',
    'author_political_leaning': 'categorical',
    'author_is_minority': 'categorical',
    'text_length': 'numerical',
    'avg_word_length': 'numerical',
    'sentiment_polarity': 'numerical',
    'sentiment_subjectivity': 'numerical',
    'has_emoji': 'binary',
    'has_hashtag': 'binary',
    'has_mention': 'binary',
    'has_url': 'binary',
    'polarization_score': 'numerical',
    'controversy_level': 'categorical',
    'primary_topic': 'categorical',
    'toxicity': 'numerical',
    'severe_toxicity': 'numerical'
}

# Datasets and models
DATASETS = ['twitter', 'bluesky', 'reddit']
PROVIDERS = ['openai', 'anthropic', 'gemini']
PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

# Output directories
OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations'
DIST_DIR = VIZ_DIR / '1_distributions'
HEATMAP_DIR = VIZ_DIR / '2_bias_heatmaps'
DIR_BIAS_DIR = VIZ_DIR / '3_directional_bias'
IMPORTANCE_DIR = VIZ_DIR / '4_feature_importance'

# Create all directories
for d in [VIZ_DIR, DIST_DIR, HEATMAP_DIR, DIR_BIAS_DIR, IMPORTANCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENHANCED VISUALIZATION SETTINGS
# ============================================================================

# Dataset color palette (consistent across all plots)
DATASET_COLORS = {
    'twitter': '#2F2F2F',      # Dark gray/blackish for Twitter/X
    'bluesky': '#4A90E2',      # Blue for Bluesky
    'reddit': '#FF6B35'        # Orange-red for Reddit
}

# Dataset display names
DATASET_LABELS = {
    'twitter': 'Twitter/X',
    'bluesky': 'Bluesky',
    'reddit': 'Reddit'
}

# Enhanced color schemes for different plot types
DIVERGING_CMAP = 'PuOr'       # Purple-White-Orange diverging (centered at 0, white in middle)
SEQUENTIAL_CMAP = 'YlOrRd'    # Yellow-Orange-Red for sequential/magnitude data
IMPORTANCE_CMAP = 'viridis'   # For feature importance (perceptually uniform)
CATEGORICAL_CMAP = 'tab20'    # For categorical features

# Directional bias colors (matching PuOr colormap - more interesting than red/green)
DIRECTIONAL_COLORS = {
    'negative': '#f1a340',  # Orange (matches PuOr low/negative values)
    'positive': '#998ec3',  # Purple (matches PuOr high/positive values)
    'neutral': '#f7f7f7'    # Very light gray (near white, neutral)
}

# Feature type order for heatmap rows (grouped by type)
FEATURE_TYPE_ORDER = (
    # Author features (demographics)
    FEATURES['author'] +
    # Text metrics
    FEATURES['text_metrics'] +
    # Content features
    FEATURES['content'] +
    # Sentiment features
    FEATURES['sentiment'] +
    # Style features (binary indicators)
    FEATURES['style'] +
    # Toxicity features
    FEATURES['toxicity']
)

# Feature display names (human-readable, no underscores)
FEATURE_DISPLAY_NAMES = {
    # Author features
    'author_gender': 'Author: Gender',
    'author_political_leaning': 'Author: Political Leaning',
    'author_is_minority': 'Author: Is Minority',
    # Text metrics
    'text_length': 'Text: Length (chars)',
    'avg_word_length': 'Text: Avg Word Length',
    # Content features
    'polarization_score': 'Content: Polarization Score',
    'controversy_level': 'Content: Controversy Level',
    'primary_topic': 'Content: Primary Topic',
    # Sentiment features
    'sentiment_polarity': 'Sentiment: Polarity',
    'sentiment_subjectivity': 'Sentiment: Subjectivity',
    # Style features (binary)
    'has_emoji': 'Style: Has Emoji',
    'has_hashtag': 'Style: Has Hashtag',
    'has_mention': 'Style: Has Mention',
    'has_url': 'Style: Has URL',
    # Toxicity features
    'toxicity': 'Toxicity: Score',
    'severe_toxicity': 'Toxicity: Severe Score'
}

def format_feature_name(feature_name):
    """Convert feature name to human-readable format."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace('_', ' ').title())

def get_feature_category(feature_name):
    """Get the category label for a feature."""
    for category, features in FEATURES.items():
        if feature_name in features:
            return category.replace('_', ' ').title()
    return 'Other'

def get_dataset_color(dataset_name):
    """Get consistent color for a dataset."""
    return DATASET_COLORS.get(dataset_name, '#666666')

def get_dataset_label(dataset_name):
    """Get display label for a dataset."""
    return DATASET_LABELS.get(dataset_name, dataset_name.title())

def get_directional_color(value):
    """Get color for directional bias value."""
    if value < -0.01:
        return DIRECTIONAL_COLORS['negative']
    elif value > 0.01:
        return DIRECTIONAL_COLORS['positive']
    else:
        return DIRECTIONAL_COLORS['neutral']

def sort_features_by_type(features_list):
    """Sort features by their type (author, text, content, etc.)."""
    return sorted(features_list, key=lambda x: (
        FEATURE_TYPE_ORDER.index(x) if x in FEATURE_TYPE_ORDER else 999,
        x
    ))

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_bias_heatmaps(comp_df):
    """
    Generate bias heatmaps with improved styling and feature type ordering.
    """
    print("\n" + "="*80)
    print("GENERATING BIAS HEATMAPS")
    print("="*80)

    all_features = sum(FEATURES.values(), [])

    # Create reverse mapping from feature to category
    feature_to_category = {}
    for category, features in FEATURES.items():
        for feature in features:
            feature_to_category[feature] = category

    # Normalize bias values within each feature (min-max scaling)
    comp_df_norm = comp_df.copy()

    for feature in all_features:
        feat_data = comp_df_norm[comp_df_norm['feature'] == feature]
        if len(feat_data) == 0:
            continue

        bias_vals = feat_data['bias'].values
        if bias_vals.max() != bias_vals.min() and bias_vals.max() > 0:
            normalized = (bias_vals - bias_vals.min()) / (bias_vals.max() - bias_vals.min())
        else:
            normalized = np.zeros_like(bias_vals)

        comp_df_norm.loc[comp_df_norm['feature'] == feature, 'bias_normalized'] = normalized

    # 1. Fully disaggregated: One heatmap per prompt style
    for prompt in PROMPT_STYLES:
        prompt_data = comp_df_norm[comp_df_norm['prompt_style'] == prompt]

        # Create pivot table: features × (dataset-model combinations)
        pivot = prompt_data.pivot_table(
            values='bias_normalized',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        # Create significance pivot
        pivot_sig = prompt_data.pivot_table(
            values='significant',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        # Sort features by type
        pivot = pivot.reindex(sort_features_by_type(pivot.index.tolist()))
        pivot_sig = pivot_sig.reindex(sort_features_by_type(pivot_sig.index.tolist()))

        # Rename index to display names
        pivot.index = [format_feature_name(f) for f in pivot.index]
        pivot_sig.index = [format_feature_name(f) for f in pivot_sig.index]

        # Rename columns to use dataset labels
        pivot.columns = [f'{get_dataset_label(d)}\n{p}' for d, p in pivot.columns]
        pivot_sig.columns = [f'{get_dataset_label(d)}\n{p}' for d, p in pivot_sig.columns]

        # Create custom annotations
        annot_array = np.empty_like(pivot, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
                if pd.isna(val):
                    annot_array[i, j] = ''
                elif sig > 0.75:
                    annot_array[i, j] = f'{val:.3f}***'
                elif sig > 0.60:
                    annot_array[i, j] = f'{val:.3f}**'
                elif sig > 0.50:
                    annot_array[i, j] = f'{val:.3f}*'
                else:
                    annot_array[i, j] = f'{val:.3f}'

        fig, ax = plt.subplots(figsize=(14, 10))
        # Use sequential colormap for magnitude (all values 0-1)
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap=SEQUENTIAL_CMAP,
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                    linewidths=0.5, linecolor='white')
        ax.set_title(f'Bias Heatmap: {prompt.title()} Prompt\n(* p<0.05 >50%, ** >60%, *** >75%)',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Dataset × Model', fontsize=11)
        ax.set_ylabel('Feature', fontsize=11)
        plt.tight_layout()
        plt.savefig(HEATMAP_DIR / f'disaggregated_prompt_{prompt}.png', bbox_inches='tight')
        plt.close()

    print("  ✓ Fully disaggregated heatmaps (by prompt style)")

    # 2. Aggregated by dataset
    agg_dataset = comp_df_norm.groupby(['feature', 'dataset']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_dataset = agg_dataset.pivot_table(
        values='bias_normalized',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    pivot_dataset_sig = agg_dataset.pivot_table(
        values='significant',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    # Sort and format
    pivot_dataset = pivot_dataset.reindex(sort_features_by_type(pivot_dataset.index.tolist()))
    pivot_dataset_sig = pivot_dataset_sig.reindex(sort_features_by_type(pivot_dataset_sig.index.tolist()))

    pivot_dataset.index = [format_feature_name(f) for f in pivot_dataset.index]
    pivot_dataset_sig.index = [format_feature_name(f) for f in pivot_dataset_sig.index]

    pivot_dataset.columns = [get_dataset_label(d) for d in pivot_dataset.columns]
    pivot_dataset_sig.columns = [get_dataset_label(d) for d in pivot_dataset_sig.columns]

    # Create annotations
    annot_dataset = np.empty_like(pivot_dataset, dtype=object)
    for i in range(pivot_dataset.shape[0]):
        for j in range(pivot_dataset.shape[1]):
            val = pivot_dataset.iloc[i, j]
            sig = pivot_dataset_sig.iloc[i, j] if not pd.isna(pivot_dataset_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_dataset[i, j] = ''
            elif sig > 0.75:
                annot_dataset[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_dataset[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_dataset[i, j] = f'{val:.3f}*'
            else:
                annot_dataset[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(8, 12))
    # Use sequential colormap
    sns.heatmap(pivot_dataset, annot=annot_dataset, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Dataset (Aggregated across Models & Prompts)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    # Color dataset labels
    for tick, dataset in zip(ax.get_xticklabels(), pivot_dataset.columns):
        dataset_key = [k for k, v in DATASET_LABELS.items() if v == dataset][0]
        tick.set_color(get_dataset_color(dataset_key))
        tick.set_weight('bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_dataset.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by dataset")

    # 3. Aggregated by model
    agg_model = comp_df_norm.groupby(['feature', 'provider']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_model = agg_model.pivot_table(
        values='bias_normalized',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    pivot_model_sig = agg_model.pivot_table(
        values='significant',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    # Sort and format
    pivot_model = pivot_model.reindex(sort_features_by_type(pivot_model.index.tolist()))
    pivot_model_sig = pivot_model_sig.reindex(sort_features_by_type(pivot_model_sig.index.tolist()))

    pivot_model.index = [format_feature_name(f) for f in pivot_model.index]
    pivot_model_sig.index = [format_feature_name(f) for f in pivot_model_sig.index]

    # Create annotations
    annot_model = np.empty_like(pivot_model, dtype=object)
    for i in range(pivot_model.shape[0]):
        for j in range(pivot_model.shape[1]):
            val = pivot_model.iloc[i, j]
            sig = pivot_model_sig.iloc[i, j] if not pd.isna(pivot_model_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_model[i, j] = ''
            elif sig > 0.75:
                annot_model[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_model[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_model[i, j] = f'{val:.3f}*'
            else:
                annot_model[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(8, 12))
    # Use sequential colormap
    sns.heatmap(pivot_model, annot=annot_model, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Model (Aggregated across Datasets & Prompts)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Model Provider', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_model.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by model")

    # 4. Aggregated by prompt style
    agg_prompt = comp_df_norm.groupby(['feature', 'prompt_style']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_prompt = agg_prompt.pivot_table(
        values='bias_normalized',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    pivot_prompt_sig = agg_prompt.pivot_table(
        values='significant',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    # Sort and format
    pivot_prompt = pivot_prompt.reindex(sort_features_by_type(pivot_prompt.index.tolist()))
    pivot_prompt_sig = pivot_prompt_sig.reindex(sort_features_by_type(pivot_prompt_sig.index.tolist()))

    pivot_prompt.index = [format_feature_name(f) for f in pivot_prompt.index]
    pivot_prompt_sig.index = [format_feature_name(f) for f in pivot_prompt_sig.index]

    pivot_prompt.columns = [p.title() for p in pivot_prompt.columns]
    pivot_prompt_sig.columns = [p.title() for p in pivot_prompt_sig.columns]

    # Create annotations
    annot_prompt = np.empty_like(pivot_prompt, dtype=object)
    for i in range(pivot_prompt.shape[0]):
        for j in range(pivot_prompt.shape[1]):
            val = pivot_prompt.iloc[i, j]
            sig = pivot_prompt_sig.iloc[i, j] if not pd.isna(pivot_prompt_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_prompt[i, j] = ''
            elif sig > 0.75:
                annot_prompt[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_prompt[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_prompt[i, j] = f'{val:.3f}*'
            else:
                annot_prompt[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(12, 12))
    # Use sequential colormap
    sns.heatmap(pivot_prompt, annot=annot_prompt, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Prompt Style (Aggregated across Datasets & Models)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Prompt Style', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_prompt.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by prompt style")

    # 5. Fully aggregated
    agg_full = comp_df_norm.groupby('feature').agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    # Sort features by type
    agg_full = agg_full.set_index('feature')
    agg_full = agg_full.reindex(sort_features_by_type(agg_full.index.tolist()))
    agg_full['feature_display'] = [format_feature_name(f) for f in agg_full.index]

    fig, ax = plt.subplots(figsize=(8, 12))
    colors = [get_directional_color(v) for v in agg_full['bias_normalized']]
    bars = ax.barh(agg_full['feature_display'], agg_full['bias_normalized'],
                   color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Normalized Bias (Averaged across all conditions)', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title('Fully Aggregated Bias (All Datasets, Models & Prompts)',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'fully_aggregated.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Fully aggregated")

    # ========================================================================
    # CATEGORY-AGGREGATED VERSIONS
    # ========================================================================
    print("\n" + "-"*80)
    print("GENERATING CATEGORY-AGGREGATED HEATMAPS")
    print("-"*80)

    # Add category column to normalized dataframe
    comp_df_norm['category'] = comp_df_norm['feature'].map(feature_to_category)

    # 1. Category-aggregated disaggregated: One heatmap per prompt style
    for prompt in PROMPT_STYLES:
        prompt_data = comp_df_norm[comp_df_norm['prompt_style'] == prompt]

        # Aggregate by category
        agg_cat = prompt_data.groupby(['category', 'dataset', 'provider']).agg({
            'bias_normalized': 'mean',
            'significant': 'mean'
        }).reset_index()

        # Create pivot table: categories × (dataset-model combinations)
        pivot = agg_cat.pivot_table(
            values='bias_normalized',
            index='category',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        pivot_sig = agg_cat.pivot_table(
            values='significant',
            index='category',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        # Rename columns to use dataset labels
        pivot.columns = [f'{get_dataset_label(d)}\n{p}' for d, p in pivot.columns]
        pivot_sig.columns = [f'{get_dataset_label(d)}\n{p}' for d, p in pivot_sig.columns]

        # Format category names
        pivot.index = [cat.replace('_', ' ').title() for cat in pivot.index]
        pivot_sig.index = [cat.replace('_', ' ').title() for cat in pivot_sig.index]

        # Create annotations with significance markers
        annot_array = np.empty_like(pivot, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
                if pd.isna(val):
                    annot_array[i, j] = ''
                elif sig > 0.75:
                    annot_array[i, j] = f'{val:.3f}***'
                elif sig > 0.60:
                    annot_array[i, j] = f'{val:.3f}**'
                elif sig > 0.50:
                    annot_array[i, j] = f'{val:.3f}*'
                else:
                    annot_array[i, j] = f'{val:.3f}'

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap=SEQUENTIAL_CMAP,
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                    linewidths=0.5, linecolor='white')
        ax.set_title(f'Bias by Feature Category: {prompt.title()} Prompt\n(* p<0.05 >50%, ** >60%, *** >75%)',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Dataset × Model', fontsize=11)
        ax.set_ylabel('Feature Category', fontsize=11)
        plt.tight_layout()
        plt.savefig(HEATMAP_DIR / f'disaggregated_prompt_{prompt}_by_category.png', bbox_inches='tight')
        plt.close()

    print("  ✓ Category-aggregated disaggregated heatmaps (by prompt style)")

    # 2. Category-aggregated by dataset
    agg_cat_dataset = comp_df_norm.groupby(['category', 'dataset']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_dataset = agg_cat_dataset.pivot_table(
        values='bias_normalized',
        index='category',
        columns='dataset',
        aggfunc='mean'
    )

    pivot_cat_dataset_sig = agg_cat_dataset.pivot_table(
        values='significant',
        index='category',
        columns='dataset',
        aggfunc='mean'
    )

    # Format names
    pivot_cat_dataset.index = [cat.replace('_', ' ').title() for cat in pivot_cat_dataset.index]
    pivot_cat_dataset_sig.index = [cat.replace('_', ' ').title() for cat in pivot_cat_dataset_sig.index]
    pivot_cat_dataset.columns = [get_dataset_label(d) for d in pivot_cat_dataset.columns]
    pivot_cat_dataset_sig.columns = [get_dataset_label(d) for d in pivot_cat_dataset_sig.columns]

    annot_cat_dataset = np.empty_like(pivot_cat_dataset, dtype=object)
    for i in range(pivot_cat_dataset.shape[0]):
        for j in range(pivot_cat_dataset.shape[1]):
            val = pivot_cat_dataset.iloc[i, j]
            sig = pivot_cat_dataset_sig.iloc[i, j] if not pd.isna(pivot_cat_dataset_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_cat_dataset[i, j] = ''
            elif sig > 0.75:
                annot_cat_dataset[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_cat_dataset[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_cat_dataset[i, j] = f'{val:.3f}*'
            else:
                annot_cat_dataset[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_cat_dataset, annot=annot_cat_dataset, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Feature Category × Dataset\n(Aggregated across Models & Prompts)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    # Color dataset labels
    for tick, dataset in zip(ax.get_xticklabels(), pivot_cat_dataset.columns):
        dataset_key = [k for k, v in DATASET_LABELS.items() if v == dataset][0]
        tick.set_color(get_dataset_color(dataset_key))
        tick.set_weight('bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_dataset_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated by dataset")

    # 3. Category-aggregated by model
    agg_cat_model = comp_df_norm.groupby(['category', 'provider']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_model = agg_cat_model.pivot_table(
        values='bias_normalized',
        index='category',
        columns='provider',
        aggfunc='mean'
    )

    pivot_cat_model_sig = agg_cat_model.pivot_table(
        values='significant',
        index='category',
        columns='provider',
        aggfunc='mean'
    )

    # Format names
    pivot_cat_model.index = [cat.replace('_', ' ').title() for cat in pivot_cat_model.index]
    pivot_cat_model_sig.index = [cat.replace('_', ' ').title() for cat in pivot_cat_model_sig.index]

    annot_cat_model = np.empty_like(pivot_cat_model, dtype=object)
    for i in range(pivot_cat_model.shape[0]):
        for j in range(pivot_cat_model.shape[1]):
            val = pivot_cat_model.iloc[i, j]
            sig = pivot_cat_model_sig.iloc[i, j] if not pd.isna(pivot_cat_model_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_cat_model[i, j] = ''
            elif sig > 0.75:
                annot_cat_model[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_cat_model[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_cat_model[i, j] = f'{val:.3f}*'
            else:
                annot_cat_model[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_cat_model, annot=annot_cat_model, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Feature Category × Model\n(Aggregated across Datasets & Prompts)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Model Provider', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_model_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated by model")

    # 4. Category-aggregated by prompt style
    agg_cat_prompt = comp_df_norm.groupby(['category', 'prompt_style']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_prompt = agg_cat_prompt.pivot_table(
        values='bias_normalized',
        index='category',
        columns='prompt_style',
        aggfunc='mean'
    )

    pivot_cat_prompt_sig = agg_cat_prompt.pivot_table(
        values='significant',
        index='category',
        columns='prompt_style',
        aggfunc='mean'
    )

    # Format names
    pivot_cat_prompt.index = [cat.replace('_', ' ').title() for cat in pivot_cat_prompt.index]
    pivot_cat_prompt_sig.index = [cat.replace('_', ' ').title() for cat in pivot_cat_prompt_sig.index]
    pivot_cat_prompt.columns = [p.title() for p in pivot_cat_prompt.columns]
    pivot_cat_prompt_sig.columns = [p.title() for p in pivot_cat_prompt_sig.columns]

    annot_cat_prompt = np.empty_like(pivot_cat_prompt, dtype=object)
    for i in range(pivot_cat_prompt.shape[0]):
        for j in range(pivot_cat_prompt.shape[1]):
            val = pivot_cat_prompt.iloc[i, j]
            sig = pivot_cat_prompt_sig.iloc[i, j] if not pd.isna(pivot_cat_prompt_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_cat_prompt[i, j] = ''
            elif sig > 0.75:
                annot_cat_prompt[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_cat_prompt[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_cat_prompt[i, j] = f'{val:.3f}*'
            else:
                annot_cat_prompt[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_cat_prompt, annot=annot_cat_prompt, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Feature Category × Prompt Style\n(Aggregated across Datasets & Models)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Prompt Style', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_prompt_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated by prompt style")

    # 5. Fully category-aggregated
    agg_cat_full = comp_df_norm.groupby('category').agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    # Format category names
    agg_cat_full['category_display'] = agg_cat_full['category'].apply(lambda x: x.replace('_', ' ').title())
    agg_cat_full_sorted = agg_cat_full.sort_values('bias_normalized', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [get_directional_color(v) for v in agg_cat_full_sorted['bias_normalized']]
    bars = ax.barh(agg_cat_full_sorted['category_display'], agg_cat_full_sorted['bias_normalized'],
                   color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Normalized Bias (Averaged across all conditions)', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    ax.set_title('Fully Aggregated Bias by Feature Category\n(All Datasets, Models & Prompts)',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'fully_aggregated_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Fully category-aggregated\n")


def generate_directional_bias_plots(df_directional):
    """
    Generate directional bias plots with improved colors and formatting.

    For each feature, generates 3 plots:
    1. By prompt style (6 subplots) - disaggregated by dataset×model
    2. By dataset (3 subplots) - aggregated across models and prompts
    3. By model (3 subplots) - aggregated across datasets and prompts
    """
    print("\n" + "="*80)
    print("GENERATING DIRECTIONAL BIAS PLOTS")
    print("="*80)

    all_features = sum(FEATURES.values(), [])

    for feature in all_features:
        print(f"  Generating: {format_feature_name(feature)}")

        feature_data = df_directional[df_directional['feature'] == feature].copy()
        if len(feature_data) == 0:
            print(f"    ⚠ No data for {feature}")
            continue

        feature_type = feature_data['feature_type'].iloc[0]

        # 1. Generate by-prompt plot (6 subplots) - disaggregated by dataset×model
        generate_directional_by_prompt(feature, feature_data, feature_type)

        # 2. Generate by-dataset plot (aggregated across models and prompts)
        generate_directional_by_dataset(feature, feature_data, feature_type)

        # 3. Generate by-model plot (aggregated across datasets and prompts)
        generate_directional_by_model(feature, feature_data, feature_type)

    print(f"\n✓ Saved directional bias plots to {DIR_BIAS_DIR}\n")


def generate_directional_by_prompt(feature, feature_data, feature_type):
    """Generate plot with 6 subplots (one per prompt style), disaggregated by dataset×model"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, prompt_style in enumerate(PROMPT_STYLES):
        ax = axes[idx]
        data = feature_data[feature_data['prompt_style'] == prompt_style].copy()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{prompt_style.title()}', fontweight='bold')
            continue

        # Create dataset×model labels
        data['dataset_model'] = data.apply(
            lambda row: f"{get_dataset_label(row['dataset'])}\n{row['provider'].title()}",
            axis=1
        )

        # Disaggregated by dataset×model (no aggregation)
        if feature_type == 'categorical':
            # Create pivot: categories × dataset×model
            pivot = data.groupby(['category', 'dataset_model'])['directional_bias'].mean().reset_index()
            pivot_wide = pivot.pivot(index='category', columns='dataset_model', values='directional_bias')

            # Sort categories by average bias
            pivot_wide['avg'] = pivot_wide.mean(axis=1)
            pivot_wide = pivot_wide.sort_values('avg', ascending=False).drop('avg', axis=1)

            # Create heatmap with reduced precision for readability
            sns.heatmap(pivot_wide, annot=True, fmt='.2f', cmap=DIVERGING_CMAP, center=0,
                       vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Directional Bias'},
                       linewidths=0.5, linecolor='gray')
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:  # continuous
            # Bar chart by dataset×model
            agg = data.groupby('dataset_model')['directional_bias'].mean().sort_values()

            # Use directional colors
            colors = [get_directional_color(x) for x in agg.values]
            agg.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Mean Difference (Rec - Pool)')
            ax.set_ylabel('')
            ax.grid(axis='x', alpha=0.3)

        ax.set_title(f'{prompt_style.title()}', fontweight='bold', fontsize=12)

    fig.suptitle(f'Directional Bias: {format_feature_name(feature)}\n(By Prompt Style, Disaggregated by Dataset×Model)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(DIR_BIAS_DIR / f'{feature}_by_prompt.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_directional_by_dataset(feature, feature_data, feature_type):
    """Generate plot with 6 subplots (one per prompt style), showing 3 datasets (aggregated across models)"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, prompt_style in enumerate(PROMPT_STYLES):
        ax = axes[idx]
        data = feature_data[feature_data['prompt_style'] == prompt_style].copy()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{prompt_style.title()}', fontweight='bold')
            continue

        # Add dataset labels
        data['dataset_label'] = data['dataset'].apply(get_dataset_label)

        # Aggregate across models (providers)
        if feature_type == 'categorical':
            # Create pivot: categories × datasets (averaged across providers)
            pivot = data.groupby(['category', 'dataset_label'])['directional_bias'].mean().reset_index()
            pivot_wide = pivot.pivot(index='category', columns='dataset_label', values='directional_bias')

            # Sort categories by average bias
            pivot_wide['avg'] = pivot_wide.mean(axis=1)
            pivot_wide = pivot_wide.sort_values('avg', ascending=False).drop('avg', axis=1)

            # Create heatmap with reduced precision for readability
            sns.heatmap(pivot_wide, annot=True, fmt='.2f', cmap=DIVERGING_CMAP, center=0,
                       vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Directional Bias'},
                       linewidths=0.5, linecolor='gray')
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:  # continuous
            # Bar chart by dataset (averaged across providers)
            agg = data.groupby('dataset_label')['directional_bias'].mean().sort_values()

            # Use directional colors
            colors = [get_directional_color(x) for x in agg.values]
            agg.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Mean Difference (Rec - Pool)')
            ax.set_ylabel('')
            ax.grid(axis='x', alpha=0.3)

        ax.set_title(f'{prompt_style.title()}', fontweight='bold', fontsize=12)

    fig.suptitle(f'Directional Bias: {format_feature_name(feature)}\n(By Prompt Style, Aggregated across Models)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(DIR_BIAS_DIR / f'{feature}_by_dataset.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_directional_by_model(feature, feature_data, feature_type):
    """Generate plot with 6 subplots (one per prompt style), showing 3 models (aggregated across datasets)"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, prompt_style in enumerate(PROMPT_STYLES):
        ax = axes[idx]
        data = feature_data[feature_data['prompt_style'] == prompt_style].copy()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{prompt_style.title()}', fontweight='bold')
            continue

        # Add provider labels
        data['provider_label'] = data['provider'].apply(lambda x: x.title())

        # Aggregate across datasets
        if feature_type == 'categorical':
            # Create pivot: categories × providers (averaged across datasets)
            pivot = data.groupby(['category', 'provider_label'])['directional_bias'].mean().reset_index()
            pivot_wide = pivot.pivot(index='category', columns='provider_label', values='directional_bias')

            # Sort categories by average bias
            pivot_wide['avg'] = pivot_wide.mean(axis=1)
            pivot_wide = pivot_wide.sort_values('avg', ascending=False).drop('avg', axis=1)

            # Create heatmap with reduced precision for readability
            sns.heatmap(pivot_wide, annot=True, fmt='.2f', cmap=DIVERGING_CMAP, center=0,
                       vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Directional Bias'},
                       linewidths=0.5, linecolor='gray')
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:  # continuous
            # Bar chart by provider (averaged across datasets)
            agg = data.groupby('provider_label')['directional_bias'].mean().sort_values()

            # Use directional colors
            colors = [get_directional_color(x) for x in agg.values]
            agg.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Mean Difference (Rec - Pool)')
            ax.set_ylabel('')
            ax.grid(axis='x', alpha=0.3)

        ax.set_title(f'{prompt_style.title()}', fontweight='bold', fontsize=12)

    fig.suptitle(f'Directional Bias: {format_feature_name(feature)}\n(By Prompt Style, Aggregated across Datasets)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(DIR_BIAS_DIR / f'{feature}_by_model.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_feature_importance_plots(imp_df_long):
    """
    Generate feature importance HEATMAPS matching bias heatmap structure.
    Uses normalized SHAP values (0-1 per feature) for better comparability.
    """
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE HEATMAPS")
    print("="*80)

    # Normalize SHAP values within each feature (min-max scaling)
    # This makes features comparable by scaling each to 0-1 range
    all_features = sum(FEATURES.values(), [])
    df_norm = imp_df_long.copy()

    for feature in all_features:
        feat_data = df_norm[df_norm['feature'] == feature]
        if len(feat_data) == 0:
            continue

        shap_vals = feat_data['shap_importance'].values
        max_val = float(shap_vals.max())
        min_val = float(shap_vals.min())
        if max_val != min_val and max_val > 0:
            normalized = (shap_vals - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(shap_vals)

        df_norm.loc[df_norm['feature'] == feature, 'shap_normalized'] = normalized

    print("  ✓ Normalized SHAP values within each feature (0-1 scale)")

    # 1. DISAGGREGATED BY PROMPT: One heatmap per prompt (features × dataset×model)
    print("\n  Disaggregated by prompt style...")
    for prompt in PROMPT_STYLES:
        prompt_data = df_norm[df_norm['prompt_style'] == prompt]

        if len(prompt_data) == 0:
            continue

        # Create pivot for normalized values (for heatmap colors)
        pivot = prompt_data.pivot_table(
            values='shap_normalized',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        # Create pivot for raw values (for annotations)
        pivot_raw = prompt_data.pivot_table(
            values='shap_importance',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        # Sort features by type order
        pivot = pivot.reindex(sort_features_by_type(pivot.index.tolist()))
        pivot_raw = pivot_raw.reindex(sort_features_by_type(pivot_raw.index.tolist()))

        # Format feature names
        pivot.index = [format_feature_name(f) for f in pivot.index]
        pivot_raw.index = [format_feature_name(f) for f in pivot_raw.index]

        # Format column names
        pivot.columns = [f'{get_dataset_label(d)}\n{p}' for d, p in pivot.columns]
        pivot_raw.columns = [f'{get_dataset_label(d)}\n{p}' for d, p in pivot_raw.columns]

        # Create annotations with raw values
        annot_array = np.empty_like(pivot, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val_raw = pivot_raw.iloc[i, j]
                if pd.isna(val_raw):
                    annot_array[i, j] = ''
                else:
                    annot_array[i, j] = f'{val_raw:.3f}'

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap=SEQUENTIAL_CMAP,
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'},
                    linewidths=0.5, linecolor='white')
        ax.set_title(f'Feature Importance (SHAP): {prompt.title()} Prompt\n(Normalized within feature, annotations show raw values)',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Dataset × Model', fontsize=11)
        ax.set_ylabel('Feature', fontsize=11)
        plt.tight_layout()
        plt.savefig(IMPORTANCE_DIR / f'disaggregated_prompt_{prompt}.png', bbox_inches='tight', dpi=300)
        plt.close()

    print("  ✓ Disaggregated heatmaps (by prompt style)")

    # 2. AGGREGATED BY DATASET (features × dataset)
    print("\n  Aggregated by dataset...")
    agg_dataset = df_norm.groupby(['feature', 'dataset']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    pivot_dataset = agg_dataset.pivot_table(
        values='shap_normalized',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    pivot_dataset_raw = agg_dataset.pivot_table(
        values='shap_importance',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    # Sort and format
    pivot_dataset = pivot_dataset.reindex(sort_features_by_type(pivot_dataset.index.tolist()))
    pivot_dataset_raw = pivot_dataset_raw.reindex(sort_features_by_type(pivot_dataset_raw.index.tolist()))
    pivot_dataset.index = [format_feature_name(f) for f in pivot_dataset.index]
    pivot_dataset_raw.index = [format_feature_name(f) for f in pivot_dataset_raw.index]
    pivot_dataset.columns = [get_dataset_label(d) for d in pivot_dataset.columns]
    pivot_dataset_raw.columns = [get_dataset_label(d) for d in pivot_dataset_raw.columns]

    # Create annotations with raw values
    annot_dataset = np.empty_like(pivot_dataset, dtype=object)
    for i in range(pivot_dataset.shape[0]):
        for j in range(pivot_dataset.shape[1]):
            val_raw = pivot_dataset_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_dataset[i, j] = ''
            else:
                annot_dataset[i, j] = f'{val_raw:.3f}'

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(pivot_dataset, annot=annot_dataset, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Feature Importance by Dataset\n(Aggregated across Models & Prompts)\n(Normalized within feature, annotations show raw values)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    # Color dataset labels
    for tick, dataset in zip(ax.get_xticklabels(), pivot_dataset.columns):
        dataset_key = [k for k, v in DATASET_LABELS.items() if v == dataset][0]
        tick.set_color(get_dataset_color(dataset_key))
        tick.set_weight('bold')
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'aggregated_by_dataset.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Aggregated by dataset")

    # 3. AGGREGATED BY MODEL (features × model)
    print("\n  Aggregated by model...")
    agg_model = df_norm.groupby(['feature', 'provider']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    pivot_model = agg_model.pivot_table(
        values='shap_normalized',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    pivot_model_raw = agg_model.pivot_table(
        values='shap_importance',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    # Sort and format
    pivot_model = pivot_model.reindex(sort_features_by_type(pivot_model.index.tolist()))
    pivot_model_raw = pivot_model_raw.reindex(sort_features_by_type(pivot_model_raw.index.tolist()))
    pivot_model.index = [format_feature_name(f) for f in pivot_model.index]
    pivot_model_raw.index = [format_feature_name(f) for f in pivot_model_raw.index]

    # Create annotations with raw values
    annot_model = np.empty_like(pivot_model, dtype=object)
    for i in range(pivot_model.shape[0]):
        for j in range(pivot_model.shape[1]):
            val_raw = pivot_model_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_model[i, j] = ''
            else:
                annot_model[i, j] = f'{val_raw:.3f}'

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(pivot_model, annot=annot_model, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Feature Importance by Model\n(Aggregated across Datasets & Prompts)\n(Normalized within feature, annotations show raw values)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Model Provider', fontsize=11)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'aggregated_by_model.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Aggregated by model")

    # 4. AGGREGATED BY PROMPT (features × prompt)
    print("\n  Aggregated by prompt style...")
    agg_prompt = df_norm.groupby(['feature', 'prompt_style']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    pivot_prompt = agg_prompt.pivot_table(
        values='shap_normalized',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    pivot_prompt_raw = agg_prompt.pivot_table(
        values='shap_importance',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    # Sort and format
    pivot_prompt = pivot_prompt.reindex(sort_features_by_type(pivot_prompt.index.tolist()))
    pivot_prompt_raw = pivot_prompt_raw.reindex(sort_features_by_type(pivot_prompt_raw.index.tolist()))
    pivot_prompt.index = [format_feature_name(f) for f in pivot_prompt.index]
    pivot_prompt_raw.index = [format_feature_name(f) for f in pivot_prompt_raw.index]
    pivot_prompt.columns = [p.title() for p in pivot_prompt.columns]
    pivot_prompt_raw.columns = [p.title() for p in pivot_prompt_raw.columns]

    # Create annotations with raw values
    annot_prompt = np.empty_like(pivot_prompt, dtype=object)
    for i in range(pivot_prompt.shape[0]):
        for j in range(pivot_prompt.shape[1]):
            val_raw = pivot_prompt_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_prompt[i, j] = ''
            else:
                annot_prompt[i, j] = f'{val_raw:.3f}'

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(pivot_prompt, annot=annot_prompt, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Feature Importance by Prompt Style\n(Aggregated across Datasets & Models)\n(Normalized within feature, annotations show raw values)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Prompt Style', fontsize=11)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'aggregated_by_prompt.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Aggregated by prompt style")

    # 5. FULLY AGGREGATED (bar chart)
    print("\n  Fully aggregated...")
    agg_full = df_norm.groupby('feature').agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    # Sort features by type
    agg_full = agg_full.set_index('feature')
    agg_full = agg_full.reindex(sort_features_by_type(agg_full.index.tolist()))
    agg_full['feature_display'] = [format_feature_name(f) for f in agg_full.index]

    fig, ax = plt.subplots(figsize=(8, 12))
    colors = [get_directional_color(v) for v in agg_full['shap_normalized']]
    bars = ax.barh(agg_full['feature_display'], agg_full['shap_normalized'],
                   color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Normalized SHAP Importance (Averaged across all conditions)', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title('Fully Aggregated Feature Importance\n(All Datasets, Models & Prompts)',
                 fontweight='bold', fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'fully_aggregated.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Fully aggregated")

    # 6. GROUPED BAR CHART: Prompt comparison (absolute values)
    print("\n  Grouped bar chart by prompt (absolute values)...")

    # Aggregate across datasets and models, keep prompts separate
    agg_by_prompt = imp_df_long.groupby(['feature', 'prompt_style']).agg({
        'shap_importance': 'mean'
    }).reset_index()

    # Pivot to get prompts as columns
    pivot_grouped = agg_by_prompt.pivot_table(
        values='shap_importance',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    # Sort features by type
    pivot_grouped = pivot_grouped.reindex(sort_features_by_type(pivot_grouped.index.tolist()))

    # Reorder columns by prompt style
    pivot_grouped = pivot_grouped[PROMPT_STYLES]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 14))

    # Format feature names
    feature_labels = [format_feature_name(f) for f in pivot_grouped.index]

    # Define colors for each prompt style (using a color palette)
    prompt_colors = {
        'general': '#1f77b4',      # blue
        'popular': '#ff7f0e',      # orange
        'engaging': '#2ca02c',     # green
        'informative': '#d62728',  # red
        'controversial': '#9467bd', # purple
        'neutral': '#8c564b'       # brown
    }

    # Set up bar positions
    n_features = len(pivot_grouped)
    n_prompts = len(PROMPT_STYLES)
    bar_height = 0.12
    y_positions = np.arange(n_features)

    # Plot bars for each prompt style
    for i, prompt in enumerate(PROMPT_STYLES):
        offset = (i - n_prompts/2 + 0.5) * bar_height
        values = pivot_grouped[prompt].values
        bars = ax.barh(y_positions + offset, values, bar_height,
                      label=prompt.title(), color=prompt_colors[prompt],
                      alpha=0.85, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel('Mean SHAP Importance (Raw Values)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance by Prompt Style\n(Aggregated across Datasets & Models - Absolute Values)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Prompt Style', loc='lower right', framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'prompt_comparison_absolute.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Grouped bar chart (prompt comparison with absolute values)")

    # 7. REVERSED GROUPED BAR CHART: Feature comparison by prompt (full version)
    print("\n  Grouped bar chart by feature - full version...")

    # Define colors for features (use a color cycle)
    feature_colors = plt.cm.tab20(np.linspace(0, 1, len(pivot_grouped)))

    # Create grouped bar chart with prompts on Y-axis
    fig, ax = plt.subplots(figsize=(14, 8))

    n_prompts = len(PROMPT_STYLES)
    n_features = len(pivot_grouped)
    bar_height = 0.04
    y_positions = np.arange(n_prompts)

    # Plot bars for each feature
    for i, (feature, color) in enumerate(zip(pivot_grouped.index, feature_colors)):
        offset = (i - n_features/2 + 0.5) * bar_height
        values = [pivot_grouped.loc[feature, prompt] for prompt in PROMPT_STYLES]
        bars = ax.barh(y_positions + offset, values, bar_height,
                      label=format_feature_name(feature), color=color,
                      alpha=0.85, edgecolor='black', linewidth=0.3)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p.title() for p in PROMPT_STYLES])
    ax.set_xlabel('Mean SHAP Importance (Raw Values)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Style', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison by Prompt Style\n(All Features - Aggregated across Datasets & Models)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left',
              framealpha=0.95, fontsize=8, ncol=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'feature_comparison_by_prompt_full.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Grouped bar chart (feature comparison - all 16 features)")

    # 8. REVERSED GROUPED BAR CHART: Top 5 features only
    print("\n  Grouped bar chart by feature - top 5 version...")

    # Calculate overall mean importance for each feature
    overall_importance = pivot_grouped.mean(axis=1).sort_values(ascending=False)
    top5_features = overall_importance.head(5).index.tolist()

    # Filter to top 5
    pivot_top5 = pivot_grouped.loc[top5_features]

    # Define colors for top 5 features
    top5_colors = plt.cm.Set2(np.linspace(0, 1, 5))

    # Create grouped bar chart with prompts on Y-axis (top 5 only)
    fig, ax = plt.subplots(figsize=(12, 8))

    n_prompts = len(PROMPT_STYLES)
    n_features_top5 = len(top5_features)
    bar_height = 0.12
    y_positions = np.arange(n_prompts)

    # Plot bars for each top 5 feature
    for i, (feature, color) in enumerate(zip(top5_features, top5_colors)):
        offset = (i - n_features_top5/2 + 0.5) * bar_height
        values = [pivot_top5.loc[feature, prompt] for prompt in PROMPT_STYLES]
        bars = ax.barh(y_positions + offset, values, bar_height,
                      label=format_feature_name(feature), color=color,
                      alpha=0.85, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p.title() for p in PROMPT_STYLES])
    ax.set_xlabel('Mean SHAP Importance (Raw Values)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Style', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Most Important Features by Prompt Style\n(Aggregated across Datasets & Models)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Feature (Top 5)', loc='lower right', framealpha=0.95, fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'feature_comparison_by_prompt_top5.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Grouped bar chart (feature comparison - top 5 features)")

    # 9. CATEGORY-AGGREGATED: Prompt comparison by feature category
    print("\n  Grouped bar chart by category (prompt comparison)...")

    # Create reverse mapping from feature to category
    feature_to_category = {}
    for category, features in FEATURES.items():
        for feature in features:
            feature_to_category[feature] = category

    # Add category column
    agg_by_prompt['category'] = agg_by_prompt['feature'].map(feature_to_category)

    # Aggregate by category and prompt
    agg_by_category_prompt = agg_by_prompt.groupby(['category', 'prompt_style']).agg({
        'shap_importance': 'mean'
    }).reset_index()

    # Pivot to get prompts as columns
    pivot_category = agg_by_category_prompt.pivot_table(
        values='shap_importance',
        index='category',
        columns='prompt_style',
        aggfunc='mean'
    )

    # Order categories
    category_order = ['author', 'text_metrics', 'sentiment', 'style', 'content', 'toxicity']
    pivot_category = pivot_category.reindex(category_order)

    # Reorder columns by prompt style
    pivot_category = pivot_category[PROMPT_STYLES]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Format category names
    category_labels = [cat.replace('_', ' ').title() for cat in pivot_category.index]

    # Set up bar positions
    n_categories = len(pivot_category)
    n_prompts = len(PROMPT_STYLES)
    bar_height = 0.12
    y_positions = np.arange(n_categories)

    # Plot bars for each prompt style
    for i, prompt in enumerate(PROMPT_STYLES):
        offset = (i - n_prompts/2 + 0.5) * bar_height
        values = pivot_category[prompt].values
        bars = ax.barh(y_positions + offset, values, bar_height,
                      label=prompt.title(), color=prompt_colors[prompt],
                      alpha=0.85, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(category_labels)
    ax.set_xlabel('Mean SHAP Importance (Raw Values)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Category', fontsize=12, fontweight='bold')
    ax.set_title('Feature Category Importance by Prompt Style\n(Aggregated across Datasets, Models & Features)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Prompt Style', loc='lower right', framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'prompt_comparison_by_category.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Grouped bar chart (prompt comparison by category)")

    # 10. CATEGORY-AGGREGATED: Feature category comparison by prompt
    print("\n  Grouped bar chart by category (feature comparison)...")

    # Define colors for feature categories
    category_colors = {
        'author': '#e377c2',        # pink
        'text_metrics': '#7f7f7f',  # gray
        'sentiment': '#bcbd22',     # olive
        'style': '#17becf',         # cyan
        'content': '#ff7f0e',       # orange
        'toxicity': '#d62728'       # red
    }

    # Create grouped bar chart with prompts on Y-axis
    fig, ax = plt.subplots(figsize=(12, 8))

    n_prompts = len(PROMPT_STYLES)
    n_categories = len(category_order)
    bar_height = 0.12
    y_positions = np.arange(n_prompts)

    # Plot bars for each category
    for i, category in enumerate(category_order):
        offset = (i - n_categories/2 + 0.5) * bar_height
        values = [pivot_category.loc[category, prompt] for prompt in PROMPT_STYLES]
        bars = ax.barh(y_positions + offset, values, bar_height,
                      label=category.replace('_', ' ').title(),
                      color=category_colors[category],
                      alpha=0.85, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p.title() for p in PROMPT_STYLES])
    ax.set_xlabel('Mean SHAP Importance (Raw Values)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Style', fontsize=12, fontweight='bold')
    ax.set_title('Feature Category Importance by Prompt Style\n(Aggregated across Datasets, Models & Features)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Feature Category', loc='lower right', framealpha=0.95, fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'feature_comparison_by_category.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Grouped bar chart (feature comparison by category)")

    print(f"\n✓ Saved feature importance heatmaps to {IMPORTANCE_DIR}\n")


def generate_importance_vs_bias_plot(imp_df_long, comp_df):
    """
    Generate scatter plot comparing SHAP importance vs. Bias magnitude.
    Shows the relationship between what the model uses vs. what bias emerges.
    """
    print("\n" + "="*80)
    print("GENERATING SHAP IMPORTANCE VS. BIAS COMPARISON")
    print("="*80)

    # Calculate mean SHAP importance per feature (across all conditions)
    mean_shap = imp_df_long.groupby('feature')['shap_importance'].mean().reset_index()
    mean_shap.columns = ['feature', 'mean_shap_importance']

    # Calculate mean bias per feature (across all conditions)
    # Need to normalize bias within each feature first (same as in bias heatmaps)
    all_features = sum(FEATURES.values(), [])
    comp_df_norm = comp_df.copy()

    for feature in all_features:
        feat_data = comp_df_norm[comp_df_norm['feature'] == feature]
        if len(feat_data) == 0:
            continue

        bias_vals = feat_data['bias'].values
        max_val = float(bias_vals.max())
        min_val = float(bias_vals.min())
        if max_val != min_val:
            normalized = (bias_vals - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(bias_vals)

        comp_df_norm.loc[comp_df_norm['feature'] == feature, 'bias_normalized'] = normalized

    mean_bias = comp_df_norm.groupby('feature')['bias_normalized'].mean().reset_index()
    mean_bias.columns = ['feature', 'mean_bias_normalized']

    # Merge
    comparison_df = mean_shap.merge(mean_bias, on='feature')

    # Add category information
    feature_to_category = {}
    for category, features in FEATURES.items():
        for feature in features:
            feature_to_category[feature] = category
    comparison_df['category'] = comparison_df['feature'].map(feature_to_category)

    # Sort by FEATURE_TYPE_ORDER
    comparison_df['sort_order'] = comparison_df['feature'].apply(
        lambda x: FEATURE_TYPE_ORDER.index(x) if x in FEATURE_TYPE_ORDER else 999
    )
    comparison_df = comparison_df.sort_values('sort_order')

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors by category
    category_colors = {
        'author': '#e377c2',
        'text_metrics': '#7f7f7f',
        'sentiment': '#bcbd22',
        'style': '#17becf',
        'content': '#ff7f0e',
        'toxicity': '#d62728'
    }

    # Plot points by category
    for category in ['author', 'text_metrics', 'sentiment', 'style', 'content', 'toxicity']:
        cat_data = comparison_df[comparison_df['category'] == category]
        ax.scatter(cat_data['mean_shap_importance'],
                  cat_data['mean_bias_normalized'],
                  s=150, alpha=0.7,
                  color=category_colors[category],
                  edgecolors='black', linewidth=1.5,
                  label=category.replace('_', ' ').title(),
                  zorder=3)

    # Calculate medians for quadrant lines
    median_shap = comparison_df['mean_shap_importance'].median()
    median_bias = comparison_df['mean_bias_normalized'].median()

    # Add labels with adjustText
    from adjustText import adjust_text
    texts = []
    for _, row in comparison_df.iterrows():
        text = ax.annotate(format_feature_name(row['feature']),
                          (row['mean_shap_importance'], row['mean_bias_normalized']),
                          fontsize=8, alpha=0.95,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='gray', alpha=0.85, linewidth=0.5))
        texts.append(text)

    # Adjust text positions to avoid overlaps
    adjust_text(texts, ax=ax,
               arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6),
               expand_points=(1.2, 1.2),
               force_text=(0.3, 0.5),
               force_points=(0.1, 0.3))

    # Add quadrant lines at median values
    ax.axvline(median_shap, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
    ax.axhline(median_bias, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)

    # Add quadrant labels (positioned to avoid data)
    ax.text(0.98, 0.98, 'High Importance\nHigh Bias',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='right', va='top', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))
    ax.text(0.02, 0.98, 'Low Importance\nHigh Bias\n(Indirect)',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='left', va='top', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))
    ax.text(0.98, 0.02, 'High Importance\nLow Bias',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='right', va='bottom', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))
    ax.text(0.02, 0.02, 'Low Importance\nLow Bias',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='left', va='bottom', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))

    # Customize plot
    ax.set_xlabel('Mean SHAP Importance\n(How much the model uses this feature)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Normalized Bias\n(How much the recommended set differs from pool)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance vs. Bias: Direct vs. Indirect Effects\n(Averaged across all datasets, models, and prompts)',
                fontsize=14, fontweight='bold', pad=20)

    # Move legend outside plot area to avoid overlapping data
    ax.legend(title='Feature Category', bbox_to_anchor=(1.02, 1), loc='upper left',
             framealpha=0.95, fontsize=9, borderaxespad=0)
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'importance_vs_bias_comparison_all_labels.png',
               bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ SHAP Importance vs. Bias scatter plot (all labels)")

    # ========================================================================
    # VERSION 2: TOP FEATURES ONLY
    # ========================================================================

    # Create second plot with only top features labeled
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot points by category (same as before)
    for category in ['author', 'text_metrics', 'sentiment', 'style', 'content', 'toxicity']:
        cat_data = comparison_df[comparison_df['category'] == category]
        ax.scatter(cat_data['mean_shap_importance'],
                  cat_data['mean_bias_normalized'],
                  s=150, alpha=0.7,
                  color=category_colors[category],
                  edgecolors='black', linewidth=1.5,
                  label=category.replace('_', ' ').title(),
                  zorder=3)

    # Add labels only for top features (top 40% by importance OR bias)
    threshold_shap = comparison_df['mean_shap_importance'].quantile(0.6)
    threshold_bias = comparison_df['mean_bias_normalized'].quantile(0.6)

    texts = []
    for _, row in comparison_df.iterrows():
        # Only label if high importance OR high bias
        if row['mean_shap_importance'] >= threshold_shap or row['mean_bias_normalized'] >= threshold_bias:
            text = ax.annotate(format_feature_name(row['feature']),
                              (row['mean_shap_importance'], row['mean_bias_normalized']),
                              fontsize=9, alpha=0.95,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='gray', alpha=0.85, linewidth=0.5))
            texts.append(text)

    # Adjust text positions to avoid overlaps
    adjust_text(texts, ax=ax,
               arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6),
               expand_points=(1.3, 1.3),
               force_text=(0.4, 0.6),
               force_points=(0.2, 0.4))

    # Add quadrant lines
    ax.axvline(median_shap, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
    ax.axhline(median_bias, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)

    # Add quadrant labels
    ax.text(0.98, 0.98, 'High Importance\nHigh Bias',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='right', va='top', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))
    ax.text(0.02, 0.98, 'Low Importance\nHigh Bias\n(Indirect)',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='left', va='top', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))
    ax.text(0.98, 0.02, 'High Importance\nLow Bias',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='right', va='bottom', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))
    ax.text(0.02, 0.02, 'Low Importance\nLow Bias',
           transform=ax.transAxes, fontsize=9, alpha=0.5,
           ha='left', va='bottom', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='none', alpha=0.6))

    # Customize plot
    ax.set_xlabel('Mean SHAP Importance\n(How much the model uses this feature)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Normalized Bias\n(How much the recommended set differs from pool)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance vs. Bias: Direct vs. Indirect Effects\n(Averaged across all datasets, models, and prompts)',
                fontsize=14, fontweight='bold', pad=20)

    # Move legend outside plot area
    ax.legend(title='Feature Category', bbox_to_anchor=(1.02, 1), loc='upper left',
             framealpha=0.95, fontsize=9, borderaxespad=0)
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'importance_vs_bias_comparison_top_labels.png',
               bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ SHAP Importance vs. Bias scatter plot (top labels only)")

    # Generate summary statistics for paper
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS (for paper)")
    print("-"*80)

    # Calculate correlation
    correlation = comparison_df['mean_shap_importance'].corr(comparison_df['mean_bias_normalized'])
    print(f"\nPearson correlation between SHAP importance and bias: {correlation:.3f}")

    # Identify top features by each metric
    top5_shap = comparison_df.nlargest(5, 'mean_shap_importance')[['feature', 'mean_shap_importance', 'mean_bias_normalized']]
    top5_bias = comparison_df.nlargest(5, 'mean_bias_normalized')[['feature', 'mean_shap_importance', 'mean_bias_normalized']]

    print("\nTop 5 features by SHAP importance:")
    for _, row in top5_shap.iterrows():
        print(f"  - {format_feature_name(row['feature'])}: SHAP={row['mean_shap_importance']:.4f}, Bias={row['mean_bias_normalized']:.3f}")

    print("\nTop 5 features by Bias:")
    for _, row in top5_bias.iterrows():
        print(f"  - {format_feature_name(row['feature'])}: SHAP={row['mean_shap_importance']:.4f}, Bias={row['mean_bias_normalized']:.3f}")

    # Category-level analysis
    print("\nCategory-level comparison:")
    cat_comparison = comparison_df.groupby('category').agg({
        'mean_shap_importance': 'mean',
        'mean_bias_normalized': 'mean'
    }).round(4)
    cat_comparison = cat_comparison.sort_values('mean_shap_importance', ascending=False)
    for category, row in cat_comparison.iterrows():
        print(f"  - {category.replace('_', ' ').title()}: SHAP={row['mean_shap_importance']:.4f}, Bias={row['mean_bias_normalized']:.3f}")

    print("\n" + "="*80 + "\n")


# ============================================================================
# DISTRIBUTION PLOTS
# ============================================================================

def load_pool_data(dataset):
    """Load unique pool posts for a dataset"""
    # Try multiple possible locations and providers
    possible_paths = [
        f"outputs/experiments_backup/{dataset}_anthropic_claude-sonnet-4-5-20250929",
        f"outputs/experiments/{dataset}_anthropic_claude-sonnet-4-5-20250929",
        f"outputs/experiments_backup/{dataset}_openai_gpt-4o-mini",
        f"outputs/experiments/{dataset}_openai_gpt-4o-mini",
    ]

    for exp_path in possible_paths:
        try:
            df = pd.read_csv(f"{exp_path}/post_level_data.csv")
            pool = df[df['selected'] == 0].drop_duplicates(subset='original_index').copy()
            return pool
        except:
            continue

    return None

CATEGORY_ORDERS = {
    'author_political_leaning': ['left', 'center-left', 'center', 'center-right', 'right', 'apolitical', 'unknown'],
    'author_gender': ['male', 'female', 'non-binary', 'unknown'],
    'author_is_minority': ['no', 'yes', 'unknown'],
    'controversy_level': ['low', 'medium', 'high'],
}

def generate_distribution_plots():
    """Generate distribution plots with improved styling."""
    print("\n" + "="*80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("="*80)

    all_features = sum(FEATURES.values(), [])

    # Load all pool data
    pools = {}
    for dataset in DATASETS:
        pool = load_pool_data(dataset)
        if pool is not None:
            pools[dataset] = pool

    if len(pools) == 0:
        print("ERROR: No pool data found! Cannot generate distribution plots.")
        print("  Expected data in: outputs/experiments/[dataset]_*/post_level_data.csv")
        return False

    print(f"✓ Loaded pool data for {len(pools)} datasets\n")

    for feature in all_features:
        feat_type = FEATURE_TYPES.get(feature, 'numerical')

        # Check if feature exists
        if feature not in list(pools.values())[0].columns:
            print(f"  ⚠ {feature} not found in data, skipping...")
            continue

        try:
            if feat_type == 'numerical':
                plot_numerical_distribution(pools, feature)
            elif feat_type == 'binary':
                plot_binary_distribution(pools, feature)
            else:  # categorical
                plot_categorical_distribution(pools, feature)

            print(f"  ✓ {format_feature_name(feature)}")
        except Exception as e:
            print(f"  ✗ Error plotting {feature}: {e}")

    print(f"\n✓ Saved {len(all_features)} distribution plots to {DIST_DIR}\n")
    return True

def plot_numerical_distribution(pools, feature):
    """Plot numerical feature with improved styling"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(format_feature_name(feature), fontsize=14, fontweight='bold')

    for idx, (dataset, pool) in enumerate(pools.items()):
        ax = axes[idx]
        values = pd.to_numeric(pool[feature], errors='coerce').dropna()

        if len(values) > 0 and values.std() > 0:
            # Use dataset-specific colors
            color = get_dataset_color(dataset)
            ax.hist(values, bins=50, alpha=0.75, color=color, edgecolor='black', linewidth=0.5)

            # Add mean and median lines
            ax.axvline(values.mean(), color='#E74C3C', linestyle='--', linewidth=2,
                      label=f'Mean: {values.mean():.3f}')
            ax.axvline(values.median(), color='#F39C12', linestyle='--', linewidth=2,
                      label=f'Median: {values.median():.3f}')

            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{get_dataset_label(dataset)} (n={len(values):,})',
                        fontsize=11, color=get_dataset_color(dataset), fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        elif len(values) > 0:
            ax.text(0.5, 0.5, f'No variation\n(all values = {values.iloc[0]:.3f})',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{get_dataset_label(dataset)} (n={len(values):,})',
                        color=get_dataset_color(dataset), fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{get_dataset_label(dataset)}',
                        color=get_dataset_color(dataset), fontweight='bold')

    plt.tight_layout()
    plt.savefig(DIST_DIR / f'{feature}_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_categorical_distribution(pools, feature):
    """Plot categorical feature with improved styling"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(format_feature_name(feature), fontsize=14, fontweight='bold')

    # Get ordering if specified
    order = CATEGORY_ORDERS.get(feature, None)

    for idx, (dataset, pool) in enumerate(pools.items()):
        ax = axes[idx]
        values = pool[feature].dropna().astype(str)

        if len(values) > 0:
            value_counts = values.value_counts()

            # Apply ordering if specified
            if order:
                order_str = [str(x).lower() for x in order]
                value_counts.index = value_counts.index.str.lower()
                value_counts = value_counts.reindex(order_str, fill_value=0)

            # Use dataset color with varying opacity
            base_color = get_dataset_color(dataset)
            from matplotlib.colors import to_rgba
            rgba = to_rgba(base_color)
            colors = [(*rgba[:3], 0.5 + 0.5 * i / max(1, len(value_counts)-1))
                     for i in range(len(value_counts))]

            value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Category', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{get_dataset_label(dataset)} (n={len(values):,})',
                        fontsize=11, color=get_dataset_color(dataset), fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{get_dataset_label(dataset)}',
                        color=get_dataset_color(dataset), fontweight='bold')

    plt.tight_layout()
    plt.savefig(DIST_DIR / f'{feature}_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_binary_distribution(pools, feature):
    """Plot binary feature with stacked bar chart"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data_to_plot = []
    labels = []
    colors_list = []

    for dataset, pool in pools.items():
        values = pd.to_numeric(pool[feature], errors='coerce').dropna()
        if len(values) > 0:
            prop_yes = (values == 1).sum() / len(values) * 100
            prop_no = (values == 0).sum() / len(values) * 100
            data_to_plot.append([prop_no, prop_yes])
            labels.append(f'{get_dataset_label(dataset)}\n(n={len(values):,})')
            colors_list.append(get_dataset_color(dataset))
        else:
            data_to_plot.append([0, 0])
            labels.append(f'{get_dataset_label(dataset)}\n(n=0)')
            colors_list.append(get_dataset_color(dataset))

    if len(data_to_plot) > 0:
        x = np.arange(len(labels))
        width = 0.6

        data_array = np.array(data_to_plot)

        # Use lighter shades for the stacked bars
        for i, dataset in enumerate(pools.keys()):
            base_color = get_dataset_color(dataset)
            rgba = to_rgba(base_color)
            color_no = (*rgba[:3], 0.4)
            color_yes = (*rgba[:3], 0.8)

            ax.bar(i, data_array[i, 0], width, label='No (0)' if i == 0 else '',
                  color=color_no, edgecolor='black', linewidth=0.5)
            ax.bar(i, data_array[i, 1], width, bottom=data_array[i, 0],
                  label='Yes (1)' if i == 0 else '', color=color_yes,
                  edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(format_feature_name(feature), fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        # Color x-axis labels
        for i, (tick, dataset) in enumerate(zip(ax.get_xticklabels(), pools.keys())):
            tick.set_color(get_dataset_color(dataset))
            tick.set_weight('bold')

        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        for i, (no, yes) in enumerate(data_array):
            if no > 5:
                ax.text(i, no/2, f'{no:.1f}%', ha='center', va='center',
                       fontweight='bold', fontsize=9)
            if yes > 5:
                ax.text(i, no + yes/2, f'{yes:.1f}%', ha='center', va='center',
                       fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(DIST_DIR / f'{feature}_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("REGENERATING VISUALIZATIONS WITH UPDATED STYLING")
    print("="*80)
    print("\nThis script loads cached analysis data and regenerates visualizations.")
    print("It does NOT recompute analysis results.")
    print("="*80 + "\n")

    # 1. Load comparison data for bias heatmaps
    # Try CSV first (doesn't require pyarrow)
    comp_data_file = OUTPUT_DIR / 'pool_vs_recommended_summary.csv'
    if not comp_data_file.exists():
        comp_data_file = OUTPUT_DIR / 'comparison_data.parquet'
        if not comp_data_file.exists():
            print(f"ERROR: Comparison data not found!")
            print(f"  Expected: {OUTPUT_DIR / 'pool_vs_recommended_summary.csv'}")
            print(f"  Run 'python run_comprehensive_analysis.py' first to generate data.")
            return
        try:
            comp_df = pd.read_parquet(comp_data_file)
        except ImportError:
            print(f"ERROR: Parquet file found but pyarrow not installed.")
            print(f"  Please install pyarrow or ensure CSV files exist.")
            return
    else:
        comp_df = pd.read_csv(comp_data_file)

    print(f"✓ Loaded {len(comp_df)} comparisons\n")

    # 2. Load directional bias data (optional - if not found, skip those plots)
    dir_bias_file = OUTPUT_DIR / 'directional_bias_data.parquet'
    dir_bias_csv = OUTPUT_DIR / 'directional_bias_data.csv'

    df_directional = None
    if dir_bias_csv.exists():
        try:
            df_directional = pd.read_csv(dir_bias_csv)
            if len(df_directional) == 0:
                df_directional = None
            else:
                print(f"✓ Loaded {len(df_directional)} directional bias measurements from CSV\n")
        except (pd.errors.EmptyDataError, Exception) as e:
            df_directional = None
    elif dir_bias_file.exists():
        try:
            df_directional = pd.read_parquet(dir_bias_file)
            print(f"✓ Loaded {len(df_directional)} directional bias measurements from Parquet\n")
        except ImportError:
            print(f"WARNING: Directional bias parquet found but pyarrow not installed")
            df_directional = None

    if df_directional is None:
        print(f"WARNING: Directional bias data not found, skipping directional bias plots")
        print(f"  To generate: Run 'python run_comprehensive_analysis.py' first\n")

    # 3. Load feature importance data (try CSV first)
    importance_csv = OUTPUT_DIR / 'feature_importance_data.csv'
    importance_parquet = OUTPUT_DIR / 'importance_analysis' / 'importance_results.parquet'

    importance_df = None
    if importance_csv.exists():
        importance_df = pd.read_csv(importance_csv)
        print(f"✓ Loaded feature importance data from CSV")
    elif importance_parquet.exists():
        try:
            importance_df = pd.read_parquet(importance_parquet)
            print(f"✓ Loaded feature importance data from Parquet")
        except ImportError:
            print(f"WARNING: Importance parquet found but pyarrow not installed")
            importance_df = None

    if importance_df is None:
        print(f"WARNING: Feature importance data not found, skipping importance plots\n")

    if importance_df is not None:
        # Check if data is already in long format (has 'feature' column)
        if 'feature' in importance_df.columns and 'shap_importance' in importance_df.columns:
            # Data is already in long format, use it directly
            imp_df_long = importance_df[['feature', 'dataset', 'provider', 'prompt_style', 'shap_importance']].copy()
            print(f"✓ Loaded {len(imp_df_long)} feature importance measurements (already in long format)\n")
        else:
            # Convert from wide format to long format
            shap_cols = [c for c in importance_df.columns if c.startswith('shap_') and c != 'shap_file']
            feature_names = [c.replace('shap_', '') for c in shap_cols]

            importance_long = []
            for _, row in importance_df.iterrows():
                for feat in feature_names:
                    shap_col = f'shap_{feat}'
                    if shap_col in importance_df.columns:
                        importance_long.append({
                            'dataset': row['dataset'],
                            'provider': row['provider'],
                            'prompt_style': row['prompt_style'],
                            'feature': feat,
                            'shap_importance': row[shap_col]
                        })

            imp_df_long = pd.DataFrame(importance_long)
            print(f"✓ Loaded {len(imp_df_long)} feature importance measurements (converted from wide format)\n")
    else:
        imp_df_long = None

    # Generate visualizations
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    # 0. Distribution plots (requires experiment data)
    dist_success = generate_distribution_plots()

    # 1. Bias heatmaps
    generate_bias_heatmaps(comp_df)

    # 2. Directional bias plots
    if df_directional is not None:
        generate_directional_bias_plots(df_directional)
    else:
        print("Skipping directional bias plots (no data)\n")

    # 3. Feature importance plots
    if imp_df_long is not None:
        generate_feature_importance_plots(imp_df_long)
        # Generate importance vs bias comparison
        generate_importance_vs_bias_plot(imp_df_long, comp_df)
    else:
        print("Skipping feature importance plots (no data)\n")

    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION REGENERATION COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {VIZ_DIR}")
    if dist_success:
        print(f"  - Distributions: {DIST_DIR}")
    print(f"  - Bias heatmaps: {HEATMAP_DIR}")
    print(f"  - Directional bias: {DIR_BIAS_DIR}")
    print(f"  - Feature importance: {IMPORTANCE_DIR}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
