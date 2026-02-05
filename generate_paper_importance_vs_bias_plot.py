#!/usr/bin/env python3
"""
Generate Importance vs Bias Plot for Paper
===========================================

Creates the importance_vs_bias plot for the paper with:
- Matching color scheme from aggregated_bar_plot_ordered.png
- No quadrant annotations (they overlap with feature labels)
- Increased font sizes for publication quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12  # Increased from 10

# Configuration
FEATURES = {
    'author': ['author_gender', 'author_political_leaning', 'author_is_minority'],
    'text_metrics': ['text_length', 'avg_word_length'],
    'sentiment': ['sentiment_polarity', 'sentiment_subjectivity'],
    'style': ['has_emoji', 'has_hashtag', 'has_mention', 'has_url'],
    'content': ['polarization_score', 'controversy_level', 'primary_topic'],
    'toxicity': ['toxicity', 'severe_toxicity']
}

FEATURE_TYPE_ORDER = sum(FEATURES.values(), [])

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

OUTPUT_DIR = Path('analysis_outputs')
PAPER_PLOTS_DIR = OUTPUT_DIR / 'visualizations' / 'paper_plots' / 'rq4'
PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def format_feature_name(feature):
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())

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

def generate_paper_plot(imp_df_long, comp_df):
    """
    Generate scatter plot for paper with matching color scheme.
    """
    print("\n" + "="*80)
    print("GENERATING IMPORTANCE VS BIAS PLOT FOR PAPER")
    print("="*80)

    # Calculate mean SHAP importance per feature
    mean_shap = imp_df_long.groupby('feature')['shap_importance'].mean().reset_index()
    mean_shap.columns = ['feature', 'mean_shap_importance']

    # Calculate R² for each observation
    comp_df['r_squared'] = comp_df.apply(convert_to_r_squared, axis=1)

    # Calculate mean R² per feature
    mean_r2 = comp_df.groupby('feature')['r_squared'].mean().reset_index()
    mean_r2.columns = ['feature', 'mean_r_squared']

    # Merge
    comparison_df = mean_shap.merge(mean_r2, on='feature')

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

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))

    # Define colors matching aggregated_bar_plot_ordered.png
    # Based on the bar plot: Author=Brown, Text=Blue, Sentiment=Yellow, Style=Purple, Content=Green, Toxicity=Red
    category_colors = {
        'author': '#8B6F47',      # Brown
        'text_metrics': '#6BA3D4', # Blue
        'sentiment': '#F4D03F',    # Yellow/Gold
        'style': '#9B59B6',        # Purple
        'content': '#52B788',      # Green
        'toxicity': '#E74C3C'      # Red
    }

    # Plot points by category
    for category in ['author', 'text_metrics', 'sentiment', 'style', 'content', 'toxicity']:
        cat_data = comparison_df[comparison_df['category'] == category]
        ax.scatter(cat_data['mean_shap_importance'],
                  cat_data['mean_r_squared'],
                  s=180, alpha=0.75,
                  color=category_colors[category],
                  edgecolors='black', linewidth=1.5,
                  label=category.replace('_', ' ').title(),
                  zorder=3)

    # Calculate medians for quadrant lines
    median_shap = comparison_df['mean_shap_importance'].median()
    median_r2 = comparison_df['mean_r_squared'].median()

    # Add labels with adjustText
    texts = []
    for _, row in comparison_df.iterrows():
        text = ax.annotate(format_feature_name(row['feature']),
                          (row['mean_shap_importance'], row['mean_r_squared']),
                          fontsize=10, alpha=0.95,
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
    ax.axhline(median_r2, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)

    # NO QUADRANT LABELS (removed as requested)

    # Customize plot with larger fonts
    ax.set_xlabel('Mean SHAP Importance\n(How much the model uses this feature)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean R² (Variance Explained)\n(How much the recommended set differs from pool)',
                 fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance vs. Bias: Direct vs. Indirect Effects\n(Averaged across all datasets, models, and prompts)',
                fontsize=16, fontweight='bold', pad=20)

    # Move legend outside plot area
    ax.legend(title='Feature Category', bbox_to_anchor=(1.02, 1), loc='upper left',
             framealpha=0.95, fontsize=11, title_fontsize=12, borderaxespad=0)
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()

    output_path = PAPER_PLOTS_DIR / 'importance_vs_bias_paper.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Paper plot saved: {output_path}")

def main():
    print("\n" + "="*80)
    print("GENERATING IMPORTANCE VS BIAS PLOT FOR PAPER")
    print("="*80)

    # Load comparison data
    comp_data_file = OUTPUT_DIR / 'pool_vs_recommended_summary.csv'
    if not comp_data_file.exists():
        print(f"ERROR: Comparison data not found at {comp_data_file}")
        return

    comp_df = pd.read_csv(comp_data_file)
    print(f"✓ Loaded {len(comp_df)} comparisons\n")

    # Load feature importance data
    importance_csv = OUTPUT_DIR / 'feature_importance_data.csv'
    if not importance_csv.exists():
        print(f"ERROR: Feature importance data not found at {importance_csv}")
        return

    importance_df = pd.read_csv(importance_csv)

    # Check if data is already in long format
    if 'feature' in importance_df.columns and 'shap_importance' in importance_df.columns:
        imp_df_long = importance_df[['feature', 'dataset', 'provider', 'prompt_style', 'shap_importance']].copy()
        print(f"✓ Loaded {len(imp_df_long)} feature importance measurements (long format)\n")
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
        print(f"✓ Loaded {len(imp_df_long)} feature importance measurements (converted from wide)\n")

    # Generate the plot
    generate_paper_plot(imp_df_long, comp_df)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nPlot saved to: {PAPER_PLOTS_DIR}")

if __name__ == '__main__':
    main()
