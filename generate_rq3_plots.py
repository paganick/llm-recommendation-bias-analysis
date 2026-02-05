#!/usr/bin/env python3
"""
Generate RQ3 Plots: Model-Specific Content and Safety Biases
==============================================================

This script generates publication-quality plots for RQ3, examining how different
LLM providers (Anthropic, OpenAI, Google) differ in their handling of:
- Content polarization
- Sentiment polarity
- Toxicity

For each metric, we generate:
1. A heatmap showing directional bias across models and prompts
2. A bar plot with 6 subplots (one per prompt) comparing models
3. CSV export of the data

Usage:
    python generate_rq3_plots.py
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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory
OUTPUT_DIR = Path('analysis_outputs/visualizations/paper_plots/rq3')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Provider ordering and display names (FIXED ORDER)
PROVIDER_ORDER = ['anthropic', 'openai', 'gemini']
PROVIDER_LABELS = {
    'anthropic': 'Anthropic Claude Sonnet 4.5',
    'openai': 'OpenAI GPT-4o-mini',
    'gemini': 'Google Gemini 2.0 Flash'
}

# Dataset display names and colors
DATASET_LABELS = {
    'bluesky': 'Bluesky',
    'reddit': 'Reddit',
    'twitter': 'Twitter/X'
}
DATASET_COLORS = {
    'bluesky': '#2166AC',
    'reddit': '#D6604D',
    'twitter': '#333333'
}

# Prompt ordering
PROMPT_ORDER = ['neutral', 'general', 'popular', 'engaging', 'informative', 'controversial']
PROMPT_LABELS = {
    'neutral': 'Neutral',
    'general': 'General',
    'popular': 'Popular',
    'engaging': 'Engaging',
    'informative': 'Informative',
    'controversial': 'Controversial'
}

# Metrics to analyze
METRICS = {
    'polarization_score': {
        'display_name': 'Content Polarization Score',
        'short_name': 'Polarization',
        'ylabel': 'Polarization Bias\n(Recommended - Pool)',
        'description': 'Higher values indicate preference for more polarized content'
    },
    'sentiment_polarity': {
        'display_name': 'Sentiment Polarity',
        'short_name': 'Sentiment',
        'ylabel': 'Sentiment Polarity Bias\n(Recommended - Pool)',
        'description': 'Higher values indicate preference for more positive sentiment'
    },
    'toxicity': {
        'display_name': 'Toxicity',
        'short_name': 'Toxicity',
        'ylabel': 'Toxicity Bias\n(Recommended - Pool)',
        'description': 'Negative values indicate toxicity aversion; positive values indicate toxicity tolerance'
    }
}

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_metric_data(feature_name):
    """
    Load and prepare data for a specific metric.

    Returns DataFrame with columns:
    - provider, dataset, prompt_style, directional_bias
    """
    # Load directional bias data
    dir_bias_path = Path('analysis_outputs/directional_bias_data.csv')
    if not dir_bias_path.exists():
        raise FileNotFoundError(f"Could not find {dir_bias_path}")

    df = pd.read_csv(dir_bias_path)

    # Filter for the specific feature and continuous type
    feature_data = df[
        (df['feature'] == feature_name) &
        (df['feature_type'] == 'continuous')
    ].copy()

    if len(feature_data) == 0:
        raise ValueError(f"No data found for feature: {feature_name}")

    # Select relevant columns
    feature_data = feature_data[[
        'provider', 'dataset', 'prompt_style', 'directional_bias'
    ]].copy()

    return feature_data

# ============================================================================
# HEATMAP GENERATION
# ============================================================================

def create_heatmap(feature_name, metric_info):
    """
    Create a single heatmap showing directional bias for all models and prompts.

    Rows: Models (Anthropic, OpenAI, Gemini) - FIXED ORDER
    Columns: Prompts (neutral, general, popular, engaging, informative, controversial)
    Values: Average directional bias across all three datasets
    """
    print(f"\n{'='*80}")
    print(f"GENERATING HEATMAP: {metric_info['display_name']}")
    print('='*80)

    # Load data
    data = load_metric_data(feature_name)

    # Aggregate across datasets (average bias)
    agg_data = data.groupby(['provider', 'prompt_style'])['directional_bias'].mean().reset_index()

    # Pivot for heatmap: rows=providers, columns=prompts
    pivot_data = agg_data.pivot(
        index='provider',
        columns='prompt_style',
        values='directional_bias'
    )

    # Reorder rows and columns
    pivot_data = pivot_data.reindex(index=PROVIDER_ORDER, columns=PROMPT_ORDER)

    # Replace index/columns with display names
    pivot_data.index = [PROVIDER_LABELS[p] for p in pivot_data.index]
    pivot_data.columns = [PROMPT_LABELS[p] for p in pivot_data.columns]

    # Create annotations
    annot = np.empty_like(pivot_data, dtype=object)
    for i in range(pivot_data.shape[0]):
        for j in range(pivot_data.shape[1]):
            val = pivot_data.iloc[i, j]
            if pd.isna(val):
                annot[i, j] = ''
            else:
                annot[i, j] = f'{val:.3f}'

    # Determine symmetric color scale
    max_abs = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))

    # Create diverging colormap (blue-white-red)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                   '#FFFFFF',  # pure white center
                   '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#DE2D26']
    cmap = LinearSegmentedColormap.from_list('diverging', colors_list, N=256)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot heatmap
    sns.heatmap(pivot_data, annot=annot, fmt='', cmap=cmap,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar_kws={'label': metric_info['ylabel']},
                linewidths=0.5, linecolor='gray')

    # Set title and labels
    title = f'{metric_info["display_name"]} Directional Bias by Model and Prompt Style\n'
    title += '(Averaged across Bluesky, Reddit, Twitter/X)'
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Prompt Style', fontsize=12, fontweight='bold')
    ax.set_ylabel('LLM Provider', fontsize=12, fontweight='bold')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()

    # Save plot
    output_path = OUTPUT_DIR / f'{feature_name}_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Heatmap saved: {output_path}")

    # Save data to CSV
    csv_path = OUTPUT_DIR / f'{feature_name}_heatmap_data.csv'

    # Create output dataframe with original provider names for clarity
    output_df = agg_data.copy()
    output_df['provider_display'] = output_df['provider'].map(PROVIDER_LABELS)
    output_df['prompt_display'] = output_df['prompt_style'].map(PROMPT_LABELS)

    # Pivot for CSV (more readable format)
    csv_pivot = output_df.pivot(
        index='provider',
        columns='prompt_style',
        values='directional_bias'
    )
    csv_pivot = csv_pivot.reindex(index=PROVIDER_ORDER, columns=PROMPT_ORDER)

    # Add provider display names as a column
    csv_pivot.insert(0, 'provider_display', [PROVIDER_LABELS[p] for p in csv_pivot.index])

    csv_pivot.to_csv(csv_path)
    print(f"✓ Data saved: {csv_path}")

    return pivot_data

# ============================================================================
# COMBINED HEATMAP GENERATION
# ============================================================================

def create_combined_heatmap():
    """
    Create a combined figure with three heatmaps side by side.

    Each heatmap shows one metric (polarization, sentiment, toxicity).
    Layout:
    - Columns: Models (Anthropic, OpenAI, Gemini)
    - Rows: Prompts (neutral, general, popular, engaging, informative, controversial) + Average
    - Values: Average directional bias across all three datasets
    - Each subplot has its own colorbar
    """
    print(f"\n{'='*80}")
    print(f"GENERATING COMBINED HEATMAP: All Metrics")
    print('='*80)

    # Create diverging colormap (blue-white-red)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                   '#FFFFFF',  # pure white center
                   '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#DE2D26']
    cmap = LinearSegmentedColormap.from_list('diverging', colors_list, N=256)

    # Create figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Process each metric
    for idx, (feature_name, metric_info) in enumerate(METRICS.items()):
        ax = axes[idx]

        # Load data
        data = load_metric_data(feature_name)

        # Aggregate across datasets (average bias + std)
        agg_data_mean = data.groupby(['provider', 'prompt_style'])['directional_bias'].mean().reset_index()
        agg_data_std = data.groupby(['provider', 'prompt_style'])['directional_bias'].std().reset_index()

        # Pivot for heatmap: rows=prompts, columns=providers (SWAPPED!)
        pivot_mean = agg_data_mean.pivot(
            index='prompt_style',
            columns='provider',
            values='directional_bias'
        )
        pivot_std = agg_data_std.pivot(
            index='prompt_style',
            columns='provider',
            values='directional_bias'
        )

        # Reorder rows and columns
        pivot_mean = pivot_mean.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)
        pivot_std = pivot_std.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)

        # Add average row across prompts with std
        avg_row_mean = pivot_mean.mean(axis=0)
        std_row = pivot_mean.std(axis=0)

        avg_series_mean = pd.Series(avg_row_mean, name='Average')
        std_series = pd.Series(std_row, name='Average')

        pivot_mean = pd.concat([pivot_mean, avg_series_mean.to_frame().T])
        pivot_std = pd.concat([pivot_std, std_series.to_frame().T])

        # Replace index/columns with display names
        pivot_mean.columns = [PROVIDER_LABELS[p].replace('Anthropic ', '').replace('OpenAI ', '').replace('Google ', '')
                              for p in pivot_mean.columns]
        prompt_labels = [PROMPT_LABELS[p] for p in PROMPT_ORDER] + ['Average\n(across prompts)']
        pivot_mean.index = prompt_labels

        # Create annotations with std for average row
        annot = np.empty_like(pivot_mean, dtype=object)
        for i in range(pivot_mean.shape[0]):
            row_name = pivot_mean.index[i]
            for j in range(pivot_mean.shape[1]):
                val = pivot_mean.iloc[i, j]
                if pd.isna(val):
                    annot[i, j] = ''
                elif 'Average' in row_name:
                    # Show mean ± std for average row
                    std_val = pivot_std.iloc[i, j]
                    annot[i, j] = f'{val:.3f}\n±{std_val:.3f}'
                else:
                    annot[i, j] = f'{val:.3f}'

        # Determine symmetric color scale for this metric
        max_abs = max(abs(pivot_mean.min().min()), abs(pivot_mean.max().max()))

        # Plot heatmap with colorbar
        im = sns.heatmap(pivot_mean, annot=annot, fmt='', cmap=cmap,
                    center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                    cbar=True,
                    cbar_kws={'label': metric_info['ylabel'].replace('\n', ' ')},
                    linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 11})

        # Increase colorbar label font size
        cbar = im.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(14)

        # Add horizontal line to separate average row
        n_rows = len(pivot_mean)
        ax.axhline(y=n_rows-1, color='black', linewidth=2.5)

        # Set title
        ax.set_title(metric_info['short_name'], fontweight='bold', fontsize=16, pad=12)

        # Set labels
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Prompt Style', fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('')

        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Only show y-tick labels on leftmost subplot
        if idx > 0:
            ax.set_yticklabels([])

    # Overall title
    fig.suptitle('Content and Safety Directional Bias by Model and Prompt Style\n' +
                 '(Averaged across Bluesky, Reddit, Twitter/X)',
                 fontweight='bold', fontsize=18, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    output_path = OUTPUT_DIR / 'combined_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Combined heatmap saved: {output_path}")

    return output_path

# ============================================================================
# BAR PLOT GENERATION
# ============================================================================

def create_bar_plots(feature_name, metric_info):
    """
    Create bar plots with 6 subplots (one per prompt) comparing models.

    Each subplot shows:
    - X-axis: Models (Anthropic, OpenAI, Gemini)
    - Y-axis: Directional bias
    - Bars grouped by dataset (Bluesky, Reddit, Twitter/X)

    All subplots share the same y-axis scale for comparability.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING BAR PLOTS: {metric_info['display_name']}")
    print('='*80)

    # Load data
    data = load_metric_data(feature_name)

    # Determine global y-axis limits
    y_min = data['directional_bias'].min()
    y_max = data['directional_bias'].max()
    y_range = y_max - y_min
    y_min_plot = y_min - 0.1 * y_range
    y_max_plot = y_max + 0.1 * y_range

    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Overall title
    fig.suptitle(f'{metric_info["display_name"]} Directional Bias by Model and Dataset\n' +
                 '(Grouped by Prompt Style)',
                 fontweight='bold', fontsize=16, y=0.98)

    # Bar width and positions
    datasets = ['bluesky', 'reddit', 'twitter']
    n_datasets = len(datasets)
    bar_width = 0.25

    # Plot each prompt
    for idx, prompt in enumerate(PROMPT_ORDER):
        ax = axes[idx]

        # Filter data for this prompt
        prompt_data = data[data['prompt_style'] == prompt].copy()

        # Create positions for each provider
        x_positions = np.arange(len(PROVIDER_ORDER))

        # Plot bars for each dataset
        for ds_idx, dataset in enumerate(datasets):
            dataset_values = []

            for provider in PROVIDER_ORDER:
                # Get value for this provider-dataset combination
                value = prompt_data[
                    (prompt_data['provider'] == provider) &
                    (prompt_data['dataset'] == dataset)
                ]['directional_bias']

                if len(value) > 0:
                    dataset_values.append(value.values[0])
                else:
                    dataset_values.append(0)

            # Calculate offset for this dataset's bars
            offset = (ds_idx - 1) * bar_width

            # Plot bars
            ax.bar(x_positions + offset, dataset_values,
                   bar_width, label=DATASET_LABELS[dataset],
                   color=DATASET_COLORS[dataset], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        # Set labels and title
        ax.set_title(PROMPT_LABELS[prompt], fontweight='bold', fontsize=13)
        ax.set_ylabel(metric_info['ylabel'], fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([PROVIDER_LABELS[p].split()[0] for p in PROVIDER_ORDER],
                           fontsize=10, rotation=0)

        # Set y-axis limits (shared across all subplots)
        ax.set_ylim(y_min_plot, y_max_plot)

        # Grid
        ax.grid(axis='y', alpha=0.3)

        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    output_path = OUTPUT_DIR / f'{feature_name}_by_model.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Bar plots saved: {output_path}")

    # Save detailed data to CSV
    csv_path = OUTPUT_DIR / f'{feature_name}_by_model_data.csv'

    # Add display names
    output_df = data.copy()
    output_df['provider_display'] = output_df['provider'].map(PROVIDER_LABELS)
    output_df['dataset_display'] = output_df['dataset'].map(DATASET_LABELS)
    output_df['prompt_display'] = output_df['prompt_style'].map(PROMPT_LABELS)

    # Reorder columns for readability
    output_df = output_df[[
        'provider', 'provider_display',
        'dataset', 'dataset_display',
        'prompt_style', 'prompt_display',
        'directional_bias'
    ]]

    # Sort for easier reading
    output_df = output_df.sort_values(['prompt_style', 'provider', 'dataset'])

    output_df.to_csv(csv_path, index=False)
    print(f"✓ Data saved: {csv_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all RQ3 plots."""
    print("="*80)
    print("GENERATING RQ3 PLOTS: MODEL-SPECIFIC CONTENT AND SAFETY BIASES")
    print("="*80)

    # Generate combined heatmap first
    try:
        create_combined_heatmap()
    except Exception as e:
        print(f"ERROR generating combined heatmap: {e}")
        import traceback
        traceback.print_exc()

    # Generate plots for each metric
    for feature_name, metric_info in METRICS.items():
        try:
            # Generate individual heatmap
            create_heatmap(feature_name, metric_info)

            # Generate bar plots
            create_bar_plots(feature_name, metric_info)

        except Exception as e:
            print(f"ERROR processing {feature_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("ALL RQ3 PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for feature_name, metric_info in METRICS.items():
        print(f"\n{metric_info['display_name']}:")
        data = load_metric_data(feature_name)

        # Overall statistics
        print(f"  Overall range: [{data['directional_bias'].min():.4f}, {data['directional_bias'].max():.4f}]")
        print(f"  Mean: {data['directional_bias'].mean():.4f}")
        print(f"  Std: {data['directional_bias'].std():.4f}")

        # By provider
        print("\n  By provider:")
        for provider in PROVIDER_ORDER:
            provider_data = data[data['provider'] == provider]['directional_bias']
            print(f"    {PROVIDER_LABELS[provider]:30s}: mean={provider_data.mean():.4f}, std={provider_data.std():.4f}")

        # By prompt
        print("\n  By prompt:")
        for prompt in PROMPT_ORDER:
            prompt_data = data[data['prompt_style'] == prompt]['directional_bias']
            print(f"    {PROMPT_LABELS[prompt]:15s}: mean={prompt_data.mean():.4f}, std={prompt_data.std():.4f}")

if __name__ == '__main__':
    main()
