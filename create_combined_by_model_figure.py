#!/usr/bin/env python3
"""
Create a combined figure with all three metrics (_by_model) for the appendix.

Layout: 3 rows (one per metric) x 6 columns (one per prompt)
Each subplot shows bars grouped by dataset within each model.
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

# Configuration
OUTPUT_DIR = Path('analysis_outputs/visualizations/paper_plots/rq3')

PROVIDER_ORDER = ['anthropic', 'openai', 'gemini']
PROVIDER_LABELS = {
    'anthropic': 'Anthropic',
    'openai': 'OpenAI',
    'gemini': 'Google'
}

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

PROMPT_ORDER = ['neutral', 'general', 'popular', 'engaging', 'informative', 'controversial']
PROMPT_LABELS = {
    'neutral': 'Neutral',
    'general': 'General',
    'popular': 'Popular',
    'engaging': 'Engaging',
    'informative': 'Informative',
    'controversial': 'Controversial'
}

METRICS = {
    'polarization_score': {
        'display_name': 'Polarization',
        'ylabel': 'Polarization Bias'
    },
    'sentiment_polarity': {
        'display_name': 'Sentiment',
        'ylabel': 'Sentiment Bias'
    },
    'toxicity': {
        'display_name': 'Toxicity',
        'ylabel': 'Toxicity Bias'
    }
}

def load_metric_data(feature_name):
    """Load and prepare data for a specific metric."""
    dir_bias_path = Path('analysis_outputs/directional_bias_data.csv')
    df = pd.read_csv(dir_bias_path)

    feature_data = df[
        (df['feature'] == feature_name) &
        (df['feature_type'] == 'continuous')
    ].copy()

    feature_data = feature_data[[
        'provider', 'dataset', 'prompt_style', 'directional_bias'
    ]].copy()

    return feature_data

def create_combined_by_model_figure():
    """
    Create a combined figure with all three metrics.

    Layout: 3 rows (metrics) x 6 columns (prompts)
    """
    print(f"\n{'='*80}")
    print(f"GENERATING COMBINED BY-MODEL FIGURE")
    print('='*80)

    # Create figure with 3 rows, 6 columns
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))

    # Bar width and dataset configuration
    datasets = ['bluesky', 'reddit', 'twitter']
    n_datasets = len(datasets)
    bar_width = 0.25

    # Process each metric (row)
    for metric_idx, (feature_name, metric_info) in enumerate(METRICS.items()):
        print(f"Processing {metric_info['display_name']}...")

        # Load data
        data = load_metric_data(feature_name)

        # Determine y-axis limits for this metric
        y_min = data['directional_bias'].min()
        y_max = data['directional_bias'].max()
        y_range = y_max - y_min
        y_min_plot = y_min - 0.1 * y_range
        y_max_plot = y_max + 0.1 * y_range

        # Plot each prompt (column)
        for prompt_idx, prompt in enumerate(PROMPT_ORDER):
            ax = axes[metric_idx, prompt_idx]

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
                       color=DATASET_COLORS[dataset], alpha=0.8,
                       edgecolor='black', linewidth=0.5)

            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

            # Set title (only on top row)
            if metric_idx == 0:
                ax.set_title(PROMPT_LABELS[prompt], fontweight='bold', fontsize=14, pad=10)

            # Set y-label (only on leftmost column)
            if prompt_idx == 0:
                ax.set_ylabel(f'{metric_info["display_name"]}\n{metric_info["ylabel"]}',
                             fontsize=12, fontweight='bold')

            # Set x-ticks and labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels([PROVIDER_LABELS[p] for p in PROVIDER_ORDER],
                              fontsize=10, rotation=0)

            # Set y-axis limits (shared within metric)
            ax.set_ylim(y_min_plot, y_max_plot)

            # Grid
            ax.grid(axis='y', alpha=0.3)

            # Legend (only on first subplot)
            if metric_idx == 0 and prompt_idx == 0:
                ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Overall title
    fig.suptitle('Content and Safety Directional Bias by Model, Dataset, and Prompt Style',
                 fontweight='bold', fontsize=20, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save plot
    output_path = OUTPUT_DIR / 'combined_by_model.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Combined by-model figure saved: {output_path}")

    return output_path

if __name__ == '__main__':
    create_combined_by_model_figure()
