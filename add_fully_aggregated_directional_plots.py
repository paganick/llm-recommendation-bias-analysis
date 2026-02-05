#!/usr/bin/env python3
"""
Add Fully Aggregated Directional Bias Plots

For each feature, create a single plot showing directional bias aggregated
across ALL prompts, models, and datasets.

This provides a simple, clean visualization showing which categories/values
are systematically favored overall.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
OUTPUT_DIR = Path('analysis_outputs')
DIR_BIAS_DIR = OUTPUT_DIR / 'visualizations' / '3_directional_bias'
DIR_BIAS_DIR.mkdir(parents=True, exist_ok=True)

# Feature categories
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
    'toxicity': 'Toxicity: Score',
    'severe_toxicity': 'Toxicity: Severe Score'
}

def format_feature_name(feature_name):
    """Convert feature name to human-readable format."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace('_', ' ').title())


def load_directional_bias_data():
    """Load directional bias data"""
    data_file = OUTPUT_DIR / 'directional_bias_data.csv'
    if not data_file.exists():
        # Try parquet
        data_file = OUTPUT_DIR / 'directional_bias_data.parquet'

    if data_file.exists():
        if data_file.suffix == '.parquet':
            return pd.read_parquet(data_file)
        else:
            return pd.read_csv(data_file)
    else:
        raise FileNotFoundError(f"Directional bias data not found in {OUTPUT_DIR}")


def generate_fully_aggregated_plot(feature, feature_data):
    """
    Generate fully aggregated directional bias plot.

    Aggregates across ALL prompts, models, and datasets.

    For categorical: Bar chart with one bar per category
    For continuous: Single bar or just text summary
    """
    feature_type = feature_data['feature_type'].iloc[0]

    if feature_type == 'categorical':
        # Aggregate across all conditions
        agg_data = feature_data.groupby('category')['directional_bias'].agg(['mean', 'std', 'count']).reset_index()
        agg_data = agg_data.sort_values('mean', ascending=True)  # Sort for better visualization

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(agg_data) * 0.4)))

        # Color bars by sign
        colors = ['#f1a340' if x < 0 else '#998ec3' for x in agg_data['mean']]

        bars = ax.barh(agg_data['category'], agg_data['mean'], color=colors, edgecolor='black', alpha=0.8)

        # Add value labels
        for i, (idx, row) in enumerate(agg_data.iterrows()):
            value = row['mean']
            label = f'{value:.3f}'
            # Position label
            if abs(value) > 0.01:
                x_pos = value + (0.002 if value > 0 else -0.002)
                ha = 'left' if value > 0 else 'right'
            else:
                x_pos = 0.002
                ha = 'left'

            ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9, fontweight='bold')

        # Add vertical line at 0
        ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

        # Formatting
        ax.set_xlabel('Mean Directional Bias\n(Proportion Recommended - Proportion Pool)', fontsize=11)
        ax.set_ylabel('Category', fontsize=11)
        ax.set_title(f'{format_feature_name(feature)}\nDirectional Bias (Fully Aggregated across All Conditions)',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#998ec3', edgecolor='black', label='Over-represented in recommendations'),
            Patch(facecolor='#f1a340', edgecolor='black', label='Under-represented in recommendations')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True)

        # Add summary stats in text box
        max_bias = agg_data.loc[agg_data['mean'].abs().idxmax()]
        textstr = f"Largest bias: {max_bias['category']} ({max_bias['mean']:.3f})\n"
        textstr += f"Conditions: {int(agg_data['count'].iloc[0])} (aggregated)"
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    else:  # continuous
        # Aggregate across all conditions
        mean_bias = feature_data['directional_bias'].mean()
        std_bias = feature_data['directional_bias'].std()
        n_conditions = len(feature_data)

        # Create simple bar or text visualization
        fig, ax = plt.subplots(figsize=(10, 4))

        # Single bar
        color = '#f1a340' if mean_bias < 0 else '#998ec3'
        bar = ax.barh(['Mean Difference'], [mean_bias], color=color, edgecolor='black', alpha=0.8)

        # Add value label
        label = f'{mean_bias:.3f} ± {std_bias:.3f}'
        x_pos = mean_bias + (0.01 if mean_bias > 0 else -0.01)
        ha = 'left' if mean_bias > 0 else 'right'
        ax.text(x_pos, 0, label, va='center', ha=ha, fontsize=12, fontweight='bold')

        # Add vertical line at 0
        ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

        # Formatting
        ax.set_xlabel('Mean Directional Bias\n(Mean Recommended - Mean Pool)', fontsize=11)
        ax.set_title(f'{format_feature_name(feature)}\nDirectional Bias (Fully Aggregated across All Conditions)',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_ylim(-0.5, 0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#998ec3', edgecolor='black', label='Higher in recommendations'),
            Patch(facecolor='#f1a340', edgecolor='black', label='Lower in recommendations')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True)

        # Add summary stats
        textstr = f"Mean bias: {mean_bias:.3f}\n"
        textstr += f"Std dev: {std_bias:.3f}\n"
        textstr += f"Conditions: {n_conditions}"
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(DIR_BIAS_DIR / f'{feature}_fully_aggregated.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    print("="*80)
    print("GENERATING FULLY AGGREGATED DIRECTIONAL BIAS PLOTS")
    print("="*80)
    print()

    # Load data
    print("Loading directional bias data...")
    df_directional = load_directional_bias_data()
    print(f"✓ Loaded {len(df_directional)} rows")
    print()

    # Get all features
    all_features = sum(FEATURES.values(), [])

    # Generate plot for each feature
    print("Generating fully aggregated plots...")
    for feature in all_features:
        feature_data = df_directional[df_directional['feature'] == feature].copy()

        if len(feature_data) == 0:
            print(f"  ⚠ No data for {feature}")
            continue

        try:
            generate_fully_aggregated_plot(feature, feature_data)
            print(f"  ✓ {feature}")
        except Exception as e:
            print(f"  ✗ Error plotting {feature}: {e}")

    print()
    print("="*80)
    print(f"✓ Saved fully aggregated directional bias plots to {DIR_BIAS_DIR}")
    print("="*80)
    print()
    print("Files created:")
    print("  - <feature>_fully_aggregated.png for each of 16 features")
    print()
    print("These plots show directional bias aggregated across:")
    print("  - All 3 datasets (Twitter/X, Bluesky, Reddit)")
    print("  - All 3 models (OpenAI, Anthropic, Gemini)")
    print("  - All 6 prompt styles (general, popular, engaging, informative, controversial, neutral)")
    print()
    print("Interpretation:")
    print("  - Purple bars: Over-represented in recommendations")
    print("  - Orange bars: Under-represented in recommendations")
    print("  - Values: Raw proportion differences (categorical) or mean differences (continuous)")


if __name__ == '__main__':
    main()
