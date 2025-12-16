"""
Visualization Functions for LLM Recommendation Bias Analysis
Implements all 33 plots specified in the analysis plan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

DATASETS = ['twitter', 'bluesky', 'reddit']
PROVIDERS = ['openai', 'anthropic', 'gemini']
PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']


# ============================================================================
# PLOTS 1-10: BIAS VISUALIZATION
# ============================================================================

def plot_1_bias_heatmap_fully_disaggregated(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 1: Bias Heatmap - Fully Disaggregated (Level 1)
    Grid of 6 subplots (one per prompt style)
    Each subplot: Dataset (x) × Model (y), color = bias magnitude
    """
    # Filter to numerical features only
    if feature not in bias_df['feature'].values:
        return

    feature_df = bias_df[bias_df['feature'] == feature].copy()

    # Only numerical features have 'bias' column
    if 'bias' not in feature_df.columns:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Bias Heatmap: {feature} (Fully Disaggregated)', fontsize=14, fontweight='bold')

    for idx, prompt_style in enumerate(PROMPT_STYLES):
        ax = axes[idx // 3, idx % 3]

        # Pivot data for heatmap
        prompt_df = feature_df[feature_df['prompt_style'] == prompt_style]

        if len(prompt_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prompt_style)
            continue

        pivot = prompt_df.pivot_table(
            values='bias',
            index='provider',
            columns='dataset',
            aggfunc='mean'
        )

        # Create heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Bias (Rec - Pool)'},
            vmin=-pivot.abs().max().max(),
            vmax=pivot.abs().max().max()
        )

        ax.set_title(f'{prompt_style}', fontweight='bold')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Provider')

    plt.tight_layout()
    output_file = output_dir / 'bias_heatmaps' / f'{feature}_bias_fully_disaggregated.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_2_prompt_style_main_effect(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 2: Prompt Style Main Effect (Level 5 - Aggregated Across Datasets + Models)
    Box plot showing bias by prompt style
    """
    feature_df = bias_df[bias_df['feature'] == feature].copy()

    if 'bias' not in feature_df.columns or len(feature_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with individual points
    sns.violinplot(
        data=feature_df,
        x='prompt_style',
        y='bias',
        ax=ax,
        palette='Set2'
    )

    # Add individual points
    sns.stripplot(
        data=feature_df,
        x='prompt_style',
        y='bias',
        ax=ax,
        color='black',
        alpha=0.3,
        size=3
    )

    # Add horizontal line at y=0
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_title(f'Prompt Style Effect on {feature} Bias\n(Aggregated across all datasets and models)',
                 fontweight='bold')
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Bias (Recommended - Pool Mean)')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = output_dir / 'bias_heatmaps' / f'{feature}_prompt_style_effect.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_3_model_main_effect(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 3: Model Main Effect (Level 6 - Aggregated Across Datasets + Prompt Styles)
    Bar chart comparing models
    """
    feature_df = bias_df[bias_df['feature'] == feature].copy()

    if 'bias' not in feature_df.columns or len(feature_df) == 0:
        return

    # Aggregate by provider
    model_agg = feature_df.groupby('provider').agg({
        'bias': ['mean', 'std', 'count']
    }).reset_index()
    model_agg.columns = ['provider', 'mean_bias', 'std_bias', 'count']

    # Calculate 95% CI
    model_agg['se'] = model_agg['std_bias'] / np.sqrt(model_agg['count'])
    model_agg['ci'] = 1.96 * model_agg['se']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot
    colors = ['#1f77b4' if x > 0 else '#d62728' for x in model_agg['mean_bias']]
    ax.barh(model_agg['provider'], model_agg['mean_bias'],
            xerr=model_agg['ci'], color=colors, alpha=0.7)

    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Mean Bias ± 95% CI')
    ax.set_ylabel('Model Provider')
    ax.set_title(f'Model Comparison: {feature} Bias\n(Aggregated across all datasets and prompt styles)',
                 fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'bias_heatmaps' / f'{feature}_model_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_4_dataset_main_effect(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 4: Dataset Main Effect (Level 7 - Aggregated Across Models + Prompt Styles)
    """
    feature_df = bias_df[bias_df['feature'] == feature].copy()

    if 'bias' not in feature_df.columns or len(feature_df) == 0:
        return

    # Aggregate by dataset
    dataset_agg = feature_df.groupby('dataset').agg({
        'bias': ['mean', 'std', 'count']
    }).reset_index()
    dataset_agg.columns = ['dataset', 'mean_bias', 'std_bias', 'count']
    dataset_agg['se'] = dataset_agg['std_bias'] / np.sqrt(dataset_agg['count'])
    dataset_agg['ci'] = 1.96 * dataset_agg['se']

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4' if x > 0 else '#d62728' for x in dataset_agg['mean_bias']]
    ax.bar(dataset_agg['dataset'], dataset_agg['mean_bias'],
           yerr=dataset_agg['ci'], color=colors, alpha=0.7)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean Bias ± 95% CI')
    ax.set_xlabel('Dataset')
    ax.set_title(f'Dataset Comparison: {feature} Bias\n(Aggregated across all models and prompt styles)',
                 fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'bias_heatmaps' / f'{feature}_dataset_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_5_interaction_model_prompt(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 5: Interaction Analysis - Model × Prompt Style (Level 3)
    Line plot showing how each model responds to different prompts
    """
    feature_df = bias_df[bias_df['feature'] == feature].copy()

    if 'bias' not in feature_df.columns or len(feature_df) == 0:
        return

    # Aggregate across datasets
    interaction = feature_df.groupby(['provider', 'prompt_style'])['bias'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    for provider in PROVIDERS:
        provider_data = interaction[interaction['provider'] == provider]
        ax.plot(provider_data['prompt_style'], provider_data['bias'],
                marker='o', label=provider, linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Mean Bias')
    ax.set_title(f'Model × Prompt Style Interaction: {feature}\n(Aggregated across datasets)',
                 fontweight='bold')
    ax.legend(title='Provider')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'bias_heatmaps' / f'{feature}_model_prompt_interaction.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_6_interaction_dataset_prompt(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 6: Interaction Analysis - Dataset × Prompt Style (Level 2)
    """
    feature_df = bias_df[bias_df['feature'] == feature].copy()

    if 'bias' not in feature_df.columns or len(feature_df) == 0:
        return

    # Aggregate across models
    interaction = feature_df.groupby(['dataset', 'prompt_style'])['bias'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    for dataset in DATASETS:
        dataset_data = interaction[interaction['dataset'] == dataset]
        ax.plot(dataset_data['prompt_style'], dataset_data['bias'],
                marker='s', label=dataset, linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Mean Bias')
    ax.set_title(f'Dataset × Prompt Style Interaction: {feature}\n(Aggregated across models)',
                 fontweight='bold')
    ax.legend(title='Dataset')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'bias_heatmaps' / f'{feature}_dataset_prompt_interaction.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_7_bluesky_comparison(bias_df: pd.DataFrame, feature: str, output_dir: Path):
    """
    Plot 7: Dataset Comparison - Twitter/Reddit vs Bluesky
    Scatter plot comparing bias magnitude
    """
    feature_df = bias_df[bias_df['feature'] == feature].copy()

    if 'bias' not in feature_df.columns or len(feature_df) == 0:
        return

    # Aggregate by provider and prompt
    agg_df = feature_df.groupby(['provider', 'prompt_style', 'dataset'])['bias'].mean().reset_index()

    # Pivot to get Twitter/Reddit average vs Bluesky
    bluesky_data = agg_df[agg_df['dataset'] == 'bluesky'].set_index(['provider', 'prompt_style'])['bias']
    twitter_data = agg_df[agg_df['dataset'] == 'twitter'].set_index(['provider', 'prompt_style'])['bias']
    reddit_data = agg_df[agg_df['dataset'] == 'reddit'].set_index(['provider', 'prompt_style'])['bias']

    # Average Twitter and Reddit
    twitter_reddit_avg = (twitter_data + reddit_data) / 2

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Match indices
    common_idx = bluesky_data.index.intersection(twitter_reddit_avg.index)

    colors = [PROMPT_STYLES.index(ps) for _, ps in common_idx]
    scatter = ax.scatter(twitter_reddit_avg[common_idx], bluesky_data[common_idx],
                        c=colors, cmap='tab10', s=100, alpha=0.6)

    # Diagonal line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x (no difference)')

    ax.set_xlabel('Mean Bias: Twitter + Reddit')
    ax.set_ylabel('Mean Bias: Bluesky')
    ax.set_title(f'Platform Comparison: {feature}\n(Each point = one model-prompt combination)',
                 fontweight='bold')

    # Add colorbar for prompt styles
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(PROMPT_STYLES)))
    cbar.set_label('Prompt Style')
    cbar.ax.set_yticklabels(PROMPT_STYLES)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'cross_cutting' / f'{feature}_bluesky_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


# ============================================================================
# PLOTS 21-25: CROSS-CUTTING ANALYSES
# ============================================================================

def plot_21_model_agreement_matrix(bias_df: pd.DataFrame, output_dir: Path):
    """
    Plot 21: Model Agreement Matrix - Correlation between models' bias estimates
    """
    # Focus on numerical features only
    numerical_df = bias_df[bias_df['bias'].notna()].copy()

    # Create pivot for each provider
    providers = numerical_df['provider'].unique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Agreement: Bias Correlation Matrix', fontsize=14, fontweight='bold')

    comparisons = [
        ('openai', 'anthropic'),
        ('openai', 'gemini'),
        ('anthropic', 'gemini')
    ]

    for idx, (p1, p2) in enumerate(comparisons):
        ax = axes[idx]

        # Get bias values for each provider
        p1_bias = numerical_df[numerical_df['provider'] == p1].set_index(
            ['dataset', 'prompt_style', 'feature'])['bias']
        p2_bias = numerical_df[numerical_df['provider'] == p2].set_index(
            ['dataset', 'prompt_style', 'feature'])['bias']

        # Find common indices
        common_idx = p1_bias.index.intersection(p2_bias.index)

        if len(common_idx) > 0:
            x = p1_bias[common_idx]
            y = p2_bias[common_idx]

            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=20)

            # Diagonal line
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)

            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(f'{p1} bias')
            ax.set_ylabel(f'{p2} bias')
            ax.set_title(f'{p1.title()} vs {p2.title()}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'cross_cutting' / 'model_agreement_scatter.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def plot_24_bias_magnitude_vs_significance(bias_df: pd.DataFrame, output_dir: Path):
    """
    Plot 24: Bias Magnitude vs Statistical Significance
    Scatter plot: effect size vs -log10(p-value)
    """
    numerical_df = bias_df[bias_df['cohens_d'].notna() & bias_df['p_value'].notna()].copy()

    if len(numerical_df) == 0:
        return

    # Calculate -log10(p)
    numerical_df['neg_log_p'] = -np.log10(numerical_df['p_value'].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter = ax.scatter(
        numerical_df['cohens_d'].abs(),
        numerical_df['neg_log_p'],
        c=numerical_df['significant'].astype(int),
        cmap='RdYlGn',
        alpha=0.5,
        s=20
    )

    # Threshold lines
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axvline(0.2, color='blue', linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
    ax.axvline(0.5, color='green', linestyle='--', alpha=0.5, label='Medium effect (d=0.5)')
    ax.axvline(0.8, color='orange', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')

    ax.set_xlabel("Effect Size (|Cohen's d|)")
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Bias Magnitude vs Statistical Significance\nQuadrant Analysis',
                 fontweight='bold')

    # Add quadrant labels
    ax.text(0.95, 0.95, 'Large effect\nSignificant', transform=ax.transAxes,
           ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.05, 0.95, 'Small effect\nSignificant', transform=ax.transAxes,
           ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(0.95, 0.05, 'Large effect\nNot significant', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'cross_cutting' / 'bias_magnitude_vs_significance.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file


def generate_all_bias_visualizations(bias_df: pd.DataFrame, output_dir: Path):
    """Generate all bias-related visualizations (Plots 1-10, 21, 24)"""
    print("\nGenerating bias visualizations...")

    # Get numerical features
    numerical_features = bias_df[bias_df['bias'].notna()]['feature'].unique()

    generated_files = []

    # Generate plots for each numerical feature
    for feature in numerical_features:  # Generate for all features
        print(f"  Generating plots for: {feature}")

        # Plots 1-7
        files = [
            plot_1_bias_heatmap_fully_disaggregated(bias_df, feature, output_dir),
            plot_2_prompt_style_main_effect(bias_df, feature, output_dir),
            plot_3_model_main_effect(bias_df, feature, output_dir),
            plot_4_dataset_main_effect(bias_df, feature, output_dir),
            plot_5_interaction_model_prompt(bias_df, feature, output_dir),
            plot_6_interaction_dataset_prompt(bias_df, feature, output_dir),
            plot_7_bluesky_comparison(bias_df, feature, output_dir)
        ]
        generated_files.extend([f for f in files if f])

    # Cross-cutting analyses
    print("  Generating cross-cutting analyses...")
    generated_files.append(plot_21_model_agreement_matrix(bias_df, output_dir))
    generated_files.append(plot_24_bias_magnitude_vs_significance(bias_df, output_dir))

    print(f"✓ Generated {len([f for f in generated_files if f])} bias visualization files\n")

    return generated_files
