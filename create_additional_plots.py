#!/usr/bin/env python3
"""
Create Additional Plots for Analysis
====================================

This script generates:
1. Non-normalized bias heatmaps (2_bias_heatmaps_raw folder)
2. R² conversion for comparing Cohen's d and Cramér's V
3. Aggregated bar plot showing average bias per feature (non-normalized)
4. Three statistical significance plots (comparing models, datasets, prompts)

Usage:
    python create_additional_plots.py
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

# ============================================================================
# CONFIGURATION
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

# Feature display names
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

# Dataset labels
DATASET_LABELS = {
    'twitter': 'Twitter/X',
    'bluesky': 'Bluesky',
    'reddit': 'Reddit'
}

# Dataset colors
DATASET_COLORS = {
    'twitter': '#2F2F2F',
    'bluesky': '#4A90E2',
    'reddit': '#FF6B35'
}

# Datasets and models
DATASETS = ['twitter', 'bluesky', 'reddit']
PROVIDERS = ['openai', 'anthropic', 'gemini']
PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

# Output directories
OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations'
HEATMAP_RAW_DIR = VIZ_DIR / '2_bias_heatmaps_raw'

# Create directories
HEATMAP_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Color schemes
SEQUENTIAL_CMAP = 'YlOrRd'  # Yellow-Orange-Red for sequential/magnitude data

# Feature category colors (for significance plots)
CATEGORY_COLORS = {
    'author': ['#8B4513', '#A0522D', '#CD853F'],  # Browns - demographic features
    'text_metrics': ['#1E90FF', '#4169E1'],  # Blues
    'content': ['#32CD32', '#3CB371', '#2E8B57'],  # Greens
    'sentiment': ['#FFD700', '#FFA500'],  # Gold/Orange
    'style': ['#9370DB', '#8A2BE2', '#9400D3', '#9932CC'],  # Purples
    'toxicity': ['#DC143C', '#B22222']  # Reds - toxicity features
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_feature_name(feature):
    """Format feature name for display."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())

def get_dataset_label(dataset):
    """Get display label for dataset."""
    return DATASET_LABELS.get(dataset, dataset.title())

def get_dataset_color(dataset):
    """Get color for dataset."""
    return DATASET_COLORS.get(dataset, '#666666')

def sort_features_by_type(features):
    """Sort features by category order."""
    feature_order = sum(FEATURES.values(), [])
    return [f for f in feature_order if f in features]

def get_feature_category(feature):
    """Get category for a feature."""
    for category, feats in FEATURES.items():
        if feature in feats:
            return category
    return 'other'

def cohens_d_to_r_squared(d):
    """Convert Cohen's d to R² (variance explained)."""
    return (d ** 2) / (d ** 2 + 4)

def cramers_v_to_r_squared(v):
    """Convert Cramér's V to R² (variance explained)."""
    return v ** 2

def convert_to_r_squared(row):
    """Convert bias metric to R² based on metric type."""
    if pd.isna(row['bias']) or pd.isna(row['metric']):
        return np.nan

    abs_bias = abs(row['bias'])
    if row['metric'] == "Cohen's d":
        return cohens_d_to_r_squared(abs_bias)
    elif row['metric'] == "Cramér's V":
        return cramers_v_to_r_squared(abs_bias)
    else:
        return np.nan

def get_feature_color(feature, idx_within_category=0):
    """Get color for a feature based on its category."""
    category = get_feature_category(feature)
    colors = CATEGORY_COLORS.get(category, ['#888888'])
    return colors[idx_within_category % len(colors)]

# ============================================================================
# 1. NON-NORMALIZED HEATMAPS
# ============================================================================

def generate_raw_bias_heatmaps(comp_df):
    """
    Generate bias heatmaps with NON-NORMALIZED (raw) bias values.
    Similar structure to normalized heatmaps but using actual bias values.
    Also converts to R² for comparable scale across metrics.
    """
    print("\n" + "="*80)
    print("GENERATING RAW (NON-NORMALIZED) BIAS HEATMAPS")
    print("="*80)

    # Add R² column
    comp_df['r_squared'] = comp_df.apply(convert_to_r_squared, axis=1)

    all_features = sum(FEATURES.values(), [])

    # Create reverse mapping from feature to category
    feature_to_category = {}
    for category, features in FEATURES.items():
        for feature in features:
            feature_to_category[feature] = category

    # 1. Fully disaggregated: One heatmap per prompt style (using R²)
    print("\n  Creating disaggregated heatmaps (by prompt)...")
    for prompt in PROMPT_STYLES:
        prompt_data = comp_df[comp_df['prompt_style'] == prompt].copy()

        # Create pivot table: features × (dataset-model combinations)
        pivot = prompt_data.pivot_table(
            values='r_squared',
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
        # Use sequential colormap
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap=SEQUENTIAL_CMAP,
                    vmin=0, vmax=pivot.max().max(), ax=ax,
                    cbar_kws={'label': 'R² (Variance Explained)'},
                    linewidths=0.5, linecolor='white')
        ax.set_title(f'Bias Heatmap (R²): {prompt.title()} Prompt\n(* p<0.05 >50%, ** >60%, *** >75%)',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Dataset × Model', fontsize=11)
        ax.set_ylabel('Feature', fontsize=11)
        plt.tight_layout()
        plt.savefig(HEATMAP_RAW_DIR / f'disaggregated_prompt_{prompt}.png', bbox_inches='tight')
        plt.close()

    print("  ✓ Disaggregated heatmaps (by prompt)")

    # 2. Aggregated by dataset
    print("  Creating aggregated heatmaps...")
    agg_dataset = comp_df.groupby(['feature', 'dataset']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_dataset = agg_dataset.pivot_table(
        values='r_squared',
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
    sns.heatmap(pivot_dataset, annot=annot_dataset, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=pivot_dataset.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Dataset (R²) - Aggregated across Models & Prompts\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    # Color dataset labels
    for tick, dataset in zip(ax.get_xticklabels(), pivot_dataset.columns):
        dataset_key = [k for k, v in DATASET_LABELS.items() if v == dataset][0]
        tick.set_color(get_dataset_color(dataset_key))
        tick.set_weight('bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'aggregated_by_dataset.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by dataset")

    # 3. Aggregated by model
    agg_model = comp_df.groupby(['feature', 'provider']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_model = agg_model.pivot_table(
        values='r_squared',
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
    sns.heatmap(pivot_model, annot=annot_model, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=pivot_model.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Model (R²) - Aggregated across Datasets & Prompts\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Model Provider', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'aggregated_by_model.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by model")

    # 4. Aggregated by prompt style
    agg_prompt = comp_df.groupby(['feature', 'prompt_style']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_prompt = agg_prompt.pivot_table(
        values='r_squared',
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
    sns.heatmap(pivot_prompt, annot=annot_prompt, fmt='', cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=pivot_prompt.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Prompt Style (R²) - Aggregated across Datasets & Models\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_xlabel('Prompt Style', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'aggregated_by_prompt.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by prompt style")

    # 5. Category-aggregated versions
    print("\n  Creating category-aggregated heatmaps...")
    comp_df['category'] = comp_df['feature'].apply(get_feature_category)

    # Category-aggregated by dataset
    agg_cat_dataset = comp_df.groupby(['category', 'dataset']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_dataset = agg_cat_dataset.pivot_table(
        values='r_squared',
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
                vmin=0, vmax=pivot_cat_dataset.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Category × Dataset (R²)\n(Aggregated across Models & Prompts)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    # Color dataset labels
    for tick, dataset in zip(ax.get_xticklabels(), pivot_cat_dataset.columns):
        dataset_key = [k for k, v in DATASET_LABELS.items() if v == dataset][0]
        tick.set_color(get_dataset_color(dataset_key))
        tick.set_weight('bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'aggregated_by_dataset_by_category.png', bbox_inches='tight')
    plt.close()

    # Category-aggregated by model
    agg_cat_model = comp_df.groupby(['category', 'provider']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_model = agg_cat_model.pivot_table(
        values='r_squared',
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
                vmin=0, vmax=pivot_cat_model.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Category × Model (R²)\n(Aggregated across Datasets & Prompts)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Model Provider', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'aggregated_by_model_by_category.png', bbox_inches='tight')
    plt.close()

    # Category-aggregated by prompt
    agg_cat_prompt = comp_df.groupby(['category', 'prompt_style']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_prompt = agg_cat_prompt.pivot_table(
        values='r_squared',
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
                vmin=0, vmax=pivot_cat_prompt.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Bias by Category × Prompt (R²)\n(Aggregated across Datasets & Models)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Prompt Style', fontsize=11)
    ax.set_ylabel('Feature Category', fontsize=11)
    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'aggregated_by_prompt_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated heatmaps")
    print("\n✓ All raw bias heatmaps generated!")

# ============================================================================
# 2. AGGREGATED BAR PLOT (NON-NORMALIZED)
# ============================================================================

def create_aggregated_bar_plot(comp_df):
    """
    Create a single bar plot showing average R² for each feature
    (aggregated across all conditions).
    """
    print("\n" + "="*80)
    print("GENERATING AGGREGATED BAR PLOT (R²)")
    print("="*80)

    # Calculate average R² per feature
    agg_full = comp_df.groupby('feature').agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    # Sort features by type
    agg_full = agg_full.set_index('feature')
    agg_full = agg_full.reindex(sort_features_by_type(agg_full.index.tolist()))
    agg_full['feature_display'] = [format_feature_name(f) for f in agg_full.index]
    agg_full['category'] = [get_feature_category(f) for f in agg_full.index]

    # Assign colors based on category
    category_idx = {}  # Track index within each category
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
        if features:  # Only add if category has features
            color = get_feature_color(features[0], 0)
            legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                        label=category.replace('_', ' ').title()))
    ax.legend(handles=legend_elements, loc='lower right', title='Feature Category')

    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'fully_aggregated_bar_plot.png', bbox_inches='tight')
    plt.close()

    print("✓ Aggregated bar plot created!")

# ============================================================================
# 3. STATISTICAL SIGNIFICANCE PLOTS
# ============================================================================

def create_significance_plots(comp_df):
    """
    Create 3 plots showing statistically significant features:
    1. Comparing models (across datasets & prompts)
    2. Comparing datasets (across models & prompts)
    3. Comparing prompts (across datasets & models)

    Each plot shows cumulative bars with different colors for each significant feature.
    """
    print("\n" + "="*80)
    print("GENERATING STATISTICAL SIGNIFICANCE PLOTS")
    print("="*80)

    # Only consider significant results
    sig_df = comp_df[comp_df['significant'] == True].copy()

    # Assign category and color to each feature
    sig_df['category'] = sig_df['feature'].apply(get_feature_category)

    # Create color mapping for each feature
    feature_colors = {}
    category_idx = {}
    for feature in sig_df['feature'].unique():
        category = get_feature_category(feature)
        if category not in category_idx:
            category_idx[category] = 0
        feature_colors[feature] = get_feature_color(feature, category_idx[category])
        category_idx[category] += 1

    # 1. COMPARING MODELS
    print("\n  Creating model comparison plot...")
    model_sig = sig_df.groupby(['provider', 'feature']).size().reset_index(name='count')

    # Pivot for stacked bar
    model_pivot = model_sig.pivot(index='provider', columns='feature', values='count').fillna(0)

    # Sort features by category
    model_pivot = model_pivot[sort_features_by_type(model_pivot.columns.tolist())]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar chart
    bottom = np.zeros(len(model_pivot))
    for feature in model_pivot.columns:
        values = model_pivot[feature].values
        color = feature_colors[feature]
        ax.bar(model_pivot.index, values, bottom=bottom,
               label=format_feature_name(feature), color=color,
               edgecolor='black', linewidth=0.5, alpha=0.9)
        bottom += values

    ax.set_xlabel('Model Provider', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of Statistically Significant Results', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Significance by Model\n(Aggregated across Datasets & Prompts, p < 0.05)',
                 fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'significance_by_model.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Model comparison plot")

    # 2. COMPARING DATASETS
    print("  Creating dataset comparison plot...")
    dataset_sig = sig_df.groupby(['dataset', 'feature']).size().reset_index(name='count')

    # Pivot for stacked bar
    dataset_pivot = dataset_sig.pivot(index='dataset', columns='feature', values='count').fillna(0)

    # Sort features by category
    dataset_pivot = dataset_pivot[sort_features_by_type(dataset_pivot.columns.tolist())]

    # Reorder datasets to match DATASETS order
    dataset_pivot = dataset_pivot.reindex([d for d in DATASETS if d in dataset_pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar chart
    bottom = np.zeros(len(dataset_pivot))
    for feature in dataset_pivot.columns:
        values = dataset_pivot[feature].values
        color = feature_colors[feature]
        ax.bar(dataset_pivot.index, values, bottom=bottom,
               label=format_feature_name(feature), color=color,
               edgecolor='black', linewidth=0.5, alpha=0.9)
        bottom += values

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of Statistically Significant Results', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Significance by Dataset\n(Aggregated across Models & Prompts, p < 0.05)',
                 fontweight='bold', fontsize=14)

    # Color x-axis labels by dataset
    for tick, dataset in zip(ax.get_xticklabels(), dataset_pivot.index):
        tick.set_color(get_dataset_color(dataset))
        tick.set_weight('bold')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'significance_by_dataset.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Dataset comparison plot")

    # 3. COMPARING PROMPTS
    print("  Creating prompt comparison plot...")
    prompt_sig = sig_df.groupby(['prompt_style', 'feature']).size().reset_index(name='count')

    # Pivot for stacked bar
    prompt_pivot = prompt_sig.pivot(index='prompt_style', columns='feature', values='count').fillna(0)

    # Sort features by category
    prompt_pivot = prompt_pivot[sort_features_by_type(prompt_pivot.columns.tolist())]

    # Reorder prompts to match PROMPT_STYLES order
    prompt_pivot = prompt_pivot.reindex([p for p in PROMPT_STYLES if p in prompt_pivot.index])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create stacked bar chart
    bottom = np.zeros(len(prompt_pivot))
    for feature in prompt_pivot.columns:
        values = prompt_pivot[feature].values
        color = feature_colors[feature]
        ax.bar(prompt_pivot.index, values, bottom=bottom,
               label=format_feature_name(feature), color=color,
               edgecolor='black', linewidth=0.5, alpha=0.9)
        bottom += values

    ax.set_xlabel('Prompt Style', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of Statistically Significant Results', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Significance by Prompt Style\n(Aggregated across Datasets & Models, p < 0.05)',
                 fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(axis='y', alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(HEATMAP_RAW_DIR / 'significance_by_prompt.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Prompt comparison plot")
    print("\n✓ All statistical significance plots created!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CREATING ADDITIONAL PLOTS FOR ANALYSIS")
    print("="*80)
    print("\nThis script generates:")
    print("  1. Non-normalized bias heatmaps (using R²)")
    print("  2. Aggregated bar plot showing average bias per feature")
    print("  3. Three statistical significance comparison plots")
    print("="*80 + "\n")

    # Load data
    comp_data_file = OUTPUT_DIR / 'pool_vs_recommended_summary.csv'
    if not comp_data_file.exists():
        print(f"ERROR: Comparison data not found!")
        print(f"  Expected: {comp_data_file}")
        print(f"  Run 'python run_comprehensive_analysis.py' first to generate data.")
        return

    comp_df = pd.read_csv(comp_data_file)
    print(f"✓ Loaded {len(comp_df)} comparisons from {comp_data_file.name}\n")

    # Add R² column for all analyses
    comp_df['r_squared'] = comp_df.apply(convert_to_r_squared, axis=1)
    print(f"✓ Converted metrics to R² (variance explained)\n")
    print(f"  - Cohen's d → R² = d² / (d² + 4)")
    print(f"  - Cramér's V → R² = V²\n")

    # Generate all plots
    generate_raw_bias_heatmaps(comp_df)
    create_aggregated_bar_plot(comp_df)
    create_significance_plots(comp_df)

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {HEATMAP_RAW_DIR}")
    print("\nGenerated files:")
    print(f"  - {len(PROMPT_STYLES)} disaggregated heatmaps (by prompt)")
    print(f"  - 3 aggregated heatmaps (by dataset, model, prompt)")
    print(f"  - 3 category-aggregated heatmaps")
    print(f"  - 1 fully aggregated bar plot")
    print(f"  - 3 statistical significance plots (model, dataset, prompt)")
    print(f"\nTotal: ~{len(PROMPT_STYLES) + 10} plot files\n")

if __name__ == '__main__':
    main()
