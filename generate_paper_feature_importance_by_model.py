#!/usr/bin/env python3
"""
Generate Feature Importance by Model Plot for Paper
====================================================

Creates horizontal heatmap with:
- Models on y-axis, features on x-axis
- White-to-red colormap
- Increased font sizes for publication quality
- Consistent styling with other paper plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12  # Increased from default

# Configuration
OUTPUT_DIR = Path('analysis_outputs')
PAPER_PLOTS_DIR = OUTPUT_DIR / 'visualizations' / 'paper_plots' / 'rq4'
PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature display names and ordering
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

# Feature ordering (same as in other plots)
FEATURE_ORDER = [
    'author_gender', 'author_political_leaning', 'author_is_minority',
    'text_length', 'avg_word_length',
    'sentiment_polarity', 'sentiment_subjectivity',
    'has_emoji', 'has_hashtag', 'has_mention', 'has_url',
    'polarization_score', 'controversy_level', 'primary_topic',
    'toxicity', 'severe_toxicity'
]

# Provider display names and ordering
PROVIDER_LABELS = {
    'anthropic': 'Claude Sonnet 4.5',
    'openai': 'GPT-4o-mini',
    'gemini': 'Gemini 2.0 Flash'
}
PROVIDER_ORDER = ['anthropic', 'openai', 'gemini']

def format_feature_name(feature):
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())

def generate_paper_plot():
    """
    Generate horizontal heatmap for paper.
    """
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE BY MODEL PLOT FOR PAPER")
    print("="*80)

    # Load feature importance data
    importance_csv = OUTPUT_DIR / 'feature_importance_data.csv'
    if not importance_csv.exists():
        print(f"ERROR: Feature importance data not found at {importance_csv}")
        return

    df = pd.read_csv(importance_csv)
    print(f"✓ Loaded {len(df)} feature importance measurements\n")

    # Check if data is in long format
    if 'feature' in df.columns and 'shap_importance' in df.columns:
        df_long = df[['feature', 'provider', 'shap_importance']].copy()
    else:
        # Convert from wide to long format
        shap_cols = [c for c in df.columns if c.startswith('shap_') and c != 'shap_file']
        feature_names = [c.replace('shap_', '') for c in shap_cols]

        data_long = []
        for _, row in df.iterrows():
            for feat in feature_names:
                shap_col = f'shap_{feat}'
                if shap_col in df.columns:
                    data_long.append({
                        'provider': row['provider'],
                        'feature': feat,
                        'shap_importance': row[shap_col]
                    })
        df_long = pd.DataFrame(data_long)

    # Normalize within features (0-1 scale per feature)
    df_norm = df_long.copy()
    for feature in df_long['feature'].unique():
        mask = df_norm['feature'] == feature
        values = df_norm.loc[mask, 'shap_importance']
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            df_norm.loc[mask, 'shap_normalized'] = (values - min_val) / (max_val - min_val)
        else:
            df_norm.loc[mask, 'shap_normalized'] = 0.5

    # Aggregate by model
    agg_model = df_norm.groupby(['feature', 'provider']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    # Pivot for heatmap: features on x-axis (columns), providers on y-axis (rows)
    # This is the TRANSPOSE of the original plot
    pivot_model = agg_model.pivot_table(
        values='shap_normalized',
        index='provider',  # SWAPPED: was 'feature'
        columns='feature',  # SWAPPED: was 'provider'
        aggfunc='mean'
    )

    pivot_model_raw = agg_model.pivot_table(
        values='shap_importance',
        index='provider',  # SWAPPED
        columns='feature',  # SWAPPED
        aggfunc='mean'
    )

    # Reorder rows (providers)
    pivot_model = pivot_model.reindex(PROVIDER_ORDER)
    pivot_model_raw = pivot_model_raw.reindex(PROVIDER_ORDER)

    # Add average row across providers
    avg_row_norm = pivot_model.mean(axis=0)
    avg_row_raw = pivot_model_raw.mean(axis=0)

    avg_series_norm = pd.Series(avg_row_norm, name='Average\n(across models)')
    avg_series_raw = pd.Series(avg_row_raw, name='Average\n(across models)')

    pivot_model = pd.concat([pivot_model, avg_series_norm.to_frame().T])
    pivot_model_raw = pd.concat([pivot_model_raw, avg_series_raw.to_frame().T])

    # Sort columns by decreasing average values
    avg_importance = pivot_model_raw.loc['Average\n(across models)'].sort_values(ascending=False)
    sorted_features = avg_importance.index.tolist()

    pivot_model = pivot_model[sorted_features]
    pivot_model_raw = pivot_model_raw[sorted_features]

    # Rename rows (providers) with display names (but keep average row name)
    index_labels = []
    for idx in pivot_model.index:
        if 'Average' in str(idx):
            index_labels.append(idx)
        else:
            index_labels.append(PROVIDER_LABELS.get(idx, idx))
    pivot_model.index = index_labels
    pivot_model_raw.index = index_labels

    # Rename columns (features) with display names
    pivot_model.columns = [format_feature_name(f) for f in pivot_model.columns]
    pivot_model_raw.columns = [format_feature_name(f) for f in pivot_model_raw.columns]

    # Create annotations with raw values
    annot_model = np.empty_like(pivot_model, dtype=object)
    for i in range(pivot_model.shape[0]):
        for j in range(pivot_model.shape[1]):
            val_raw = pivot_model_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_model[i, j] = ''
            else:
                annot_model[i, j] = f'{val_raw:.3f}'

    # Create white-to-red colormap
    colors_white_red = ['#FFFFFF', '#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272',
                        '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    cmap_white_red = LinearSegmentedColormap.from_list('white_red', colors_white_red, N=256)

    # Create figure (wider due to horizontal orientation, taller for 4 rows)
    fig, ax = plt.subplots(figsize=(16, 6))

    sns.heatmap(pivot_model, annot=annot_model, fmt='', cmap=cmap_white_red,
                vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'Normalized SHAP Importance'},
                linewidths=0.5, linecolor='lightgray',
                annot_kws={'fontsize': 10})

    # Add horizontal line to separate average row
    n_rows = len(pivot_model)
    ax.axhline(y=n_rows-1, color='black', linewidth=2.5)

    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title('Feature Importance by Model\n(Normalized within features, aggregated across datasets & prompts)',
                fontweight='bold', fontsize=16, pad=15)
    ax.set_xlabel('Feature', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model Provider', fontsize=14, fontweight='bold')

    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()

    output_path = PAPER_PLOTS_DIR / 'feature_importance_by_model_paper.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Paper plot saved: {output_path}")

def generate_paper_plot_absolute():
    """
    Generate horizontal heatmap for paper with absolute (non-normalized) values.
    """
    print("\n" + "="*80)
    print("GENERATING ABSOLUTE FEATURE IMPORTANCE BY MODEL PLOT FOR PAPER")
    print("="*80)

    # Load feature importance data
    importance_csv = OUTPUT_DIR / 'feature_importance_data.csv'
    if not importance_csv.exists():
        print(f"ERROR: Feature importance data not found at {importance_csv}")
        return

    df = pd.read_csv(importance_csv)
    print(f"✓ Loaded {len(df)} feature importance measurements\n")

    # Check if data is in long format
    if 'feature' in df.columns and 'shap_importance' in df.columns:
        df_long = df[['feature', 'provider', 'shap_importance']].copy()
    else:
        # Convert from wide to long format
        shap_cols = [c for c in df.columns if c.startswith('shap_') and c != 'shap_file']
        feature_names = [c.replace('shap_', '') for c in shap_cols]

        data_long = []
        for _, row in df.iterrows():
            for feat in feature_names:
                shap_col = f'shap_{feat}'
                if shap_col in df.columns:
                    data_long.append({
                        'provider': row['provider'],
                        'feature': feat,
                        'shap_importance': row[shap_col]
                    })
        df_long = pd.DataFrame(data_long)

    # Aggregate by model (NO normalization - use raw values)
    agg_model = df_long.groupby(['feature', 'provider']).agg({
        'shap_importance': 'mean'
    }).reset_index()

    # Pivot for heatmap: features on x-axis (columns), providers on y-axis (rows)
    pivot_model_raw = agg_model.pivot_table(
        values='shap_importance',
        index='provider',
        columns='feature',
        aggfunc='mean'
    )

    # Reorder rows (providers)
    pivot_model_raw = pivot_model_raw.reindex(PROVIDER_ORDER)

    # Add average row across providers
    avg_row_raw = pivot_model_raw.mean(axis=0)
    avg_series_raw = pd.Series(avg_row_raw, name='Average\n(across models)')
    pivot_model_raw = pd.concat([pivot_model_raw, avg_series_raw.to_frame().T])

    # Sort columns by decreasing average values
    avg_importance = pivot_model_raw.loc['Average\n(across models)'].sort_values(ascending=False)
    sorted_features = avg_importance.index.tolist()
    pivot_model_raw = pivot_model_raw[sorted_features]

    # Rename rows (providers) with display names (but keep average row name)
    index_labels = []
    for idx in pivot_model_raw.index:
        if 'Average' in str(idx):
            index_labels.append(idx)
        else:
            index_labels.append(PROVIDER_LABELS.get(idx, idx))
    pivot_model_raw.index = index_labels

    # Rename columns (features) with display names
    pivot_model_raw.columns = [format_feature_name(f) for f in pivot_model_raw.columns]

    # Create annotations with raw values
    annot_model = np.empty_like(pivot_model_raw, dtype=object)
    for i in range(pivot_model_raw.shape[0]):
        for j in range(pivot_model_raw.shape[1]):
            val_raw = pivot_model_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_model[i, j] = ''
            else:
                annot_model[i, j] = f'{val_raw:.3f}'

    # Create white-to-red colormap
    colors_white_red = ['#FFFFFF', '#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272',
                        '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    cmap_white_red = LinearSegmentedColormap.from_list('white_red', colors_white_red, N=256)

    # Create figure (wider due to horizontal orientation, taller for 4 rows)
    fig, ax = plt.subplots(figsize=(16, 6))

    # Use raw values for colorscale (vmin=0, vmax determined by data)
    vmax = pivot_model_raw.max().max()

    sns.heatmap(pivot_model_raw, annot=annot_model, fmt='', cmap=cmap_white_red,
                vmin=0, vmax=vmax, ax=ax,
                cbar_kws={'label': 'SHAP Importance'},
                linewidths=0.5, linecolor='lightgray',
                annot_kws={'fontsize': 10})

    # Add horizontal line to separate average row
    n_rows = len(pivot_model_raw)
    ax.axhline(y=n_rows-1, color='black', linewidth=2.5)

    # Increase colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title('Feature Importance by Model\n(Aggregated across datasets & prompts)',
                fontweight='bold', fontsize=16, pad=15)
    ax.set_xlabel('Feature', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model Provider', fontsize=14, fontweight='bold')

    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()

    output_path = PAPER_PLOTS_DIR / 'feature_importance_by_model_absolute_paper.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Absolute values paper plot saved: {output_path}")

def main():
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE BY MODEL PLOTS FOR PAPER")
    print("="*80)

    # Generate normalized version
    generate_paper_plot()

    # Generate absolute (non-normalized) version
    generate_paper_plot_absolute()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nPlots saved to: {PAPER_PLOTS_DIR}")

if __name__ == '__main__':
    main()
