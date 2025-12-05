"""
Cross-Dataset and Cross-Model Comparison

Compares experiment results across:
1. Different datasets (Reddit vs Twitter vs Bluesky)
2. Different models (GPT vs Claude vs Gemini)
3. Different prompt styles

Usage:
    python compare_experiments.py \
        --results-dirs outputs/experiments/reddit_openai_gpt-4o-mini \
                      outputs/experiments/twitter_openai_gpt-4o-mini \
                      outputs/experiments/reddit_anthropic_claude-3-5-sonnet-20241022

    # Or compare all experiments in outputs/experiments/
    python compare_experiments.py --compare-all
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, List, Any, Tuple
import json
import argparse
import glob

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_experiment_results(results_dir: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load experiment configuration and summary statistics."""

    config_path = results_dir / 'config.pkl'
    summary_path = results_dir / 'comprehensive_bias_analysis.csv'

    # Try comprehensive analysis first, fallback to basic
    if not summary_path.exists():
        summary_path = results_dir / 'summary_statistics.csv'

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    else:
        print(f"Warning: No summary statistics found in {results_dir}")
        summary_df = pd.DataFrame()

    # Add metadata from config
    config['results_dir'] = str(results_dir)
    config['experiment_name'] = results_dir.name

    return config, summary_df


def create_experiment_label(config: Dict) -> str:
    """Create readable label for experiment."""

    dataset = config.get('dataset', '?')
    provider = config.get('provider', '?')
    model = config.get('model', '?')

    # Shorten model name
    model_short = model.split('/')[-1][:20]  # Take last part if path, limit length

    return f"{dataset}/{provider}/{model_short}"


def compare_across_datasets(experiments: List[Tuple[Dict, pd.DataFrame]],
                             output_dir: Path):
    """Compare bias patterns across different datasets."""

    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)

    # Group experiments by dataset
    by_dataset = {}
    for config, summary_df in experiments:
        dataset = config.get('dataset', 'unknown')
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append((config, summary_df))

    if len(by_dataset) < 2:
        print("Need at least 2 different datasets to compare")
        return

    print(f"\nDatasets found: {list(by_dataset.keys())}")

    # Compare bias patterns for a specific attribute across datasets
    # For simplicity, we'll compare using the first experiment from each dataset

    comparison_data = []

    for dataset, exp_list in by_dataset.items():
        config, summary_df = exp_list[0]  # Take first experiment

        if len(summary_df) == 0:
            continue

        # Get average bias across all prompt styles
        if 'difference' in summary_df.columns:
            avg_bias = summary_df.groupby(['attribute', 'value'])['difference'].mean().reset_index()
            avg_bias['dataset'] = dataset

            comparison_data.append(avg_bias)

    if len(comparison_data) == 0:
        print("No comparable data found")
        return

    # Combine
    combined_df = pd.concat(comparison_data, ignore_index=True)

    # Plot top biases per dataset
    fig, axes = plt.subplots(1, len(by_dataset), figsize=(6 * len(by_dataset), 6), squeeze=False)
    axes = axes.flatten()

    for idx, dataset in enumerate(sorted(by_dataset.keys())):
        ax = axes[idx]

        dataset_df = combined_df[combined_df['dataset'] == dataset]
        top_biases = dataset_df.nlargest(10, 'difference')

        top_biases['label'] = top_biases['attribute'] + '=' + top_biases['value'].astype(str)

        top_biases.plot(x='label', y='difference', kind='barh', ax=ax, legend=False)

        ax.set_title(f'{dataset.title()} Dataset')
        ax.set_xlabel('Bias (pp)')
        ax.set_ylabel('')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    filename = output_dir / 'cross_dataset_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()


def compare_across_models(experiments: List[Tuple[Dict, pd.DataFrame]],
                           output_dir: Path):
    """Compare bias patterns across different models."""

    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON")
    print("="*80)

    # Group experiments by model
    by_model = {}
    for config, summary_df in experiments:
        model_key = f"{config.get('provider', '?')}:{config.get('model', '?')}"
        if model_key not in by_model:
            by_model[model_key] = []
        by_model[model_key].append((config, summary_df))

    if len(by_model) < 2:
        print("Need at least 2 different models to compare")
        return

    print(f"\nModels found: {list(by_model.keys())}")

    # Compare specific attributes across models
    comparison_data = []

    for model_key, exp_list in by_model.items():
        config, summary_df = exp_list[0]  # Take first experiment

        if len(summary_df) == 0:
            continue

        # Get average bias
        if 'difference' in summary_df.columns:
            avg_bias = summary_df.groupby(['attribute', 'value'])['difference'].mean().reset_index()
            avg_bias['model'] = model_key

            comparison_data.append(avg_bias)

    if len(comparison_data) == 0:
        print("No comparable data found")
        return

    # Combine
    combined_df = pd.concat(comparison_data, ignore_index=True)

    # Create comparison for common attributes
    # Find attributes present in all models
    attrs_per_model = combined_df.groupby('model')[['attribute', 'value']].apply(
        lambda x: set(zip(x['attribute'], x['value']))
    )

    common_attrs = set.intersection(*attrs_per_model.values)

    if len(common_attrs) > 0:
        print(f"\nFound {len(common_attrs)} common attribute-value pairs")

        # Filter to common
        combined_df['attr_value'] = list(zip(combined_df['attribute'], combined_df['value']))
        common_df = combined_df[combined_df['attr_value'].isin(common_attrs)]

        # Pivot for heatmap
        common_df['label'] = common_df['attribute'] + '=' + common_df['value'].astype(str)
        pivot_df = common_df.pivot_table(index='label', columns='model',
                                          values='difference', fill_value=0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_df) * 0.4)))

        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdBu_r',
                    center=0, cbar_kws={'label': 'Bias (pp)'},
                    ax=ax, vmin=-20, vmax=20)

        ax.set_title('Cross-Model Bias Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Attribute = Value')

        plt.tight_layout()

        filename = output_dir / 'cross_model_comparison.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {filename}")
        plt.close()
    else:
        print("No common attributes found across models")


def create_summary_table(experiments: List[Tuple[Dict, pd.DataFrame]],
                          output_dir: Path):
    """Create summary table of all experiments."""

    print("\n" + "="*80)
    print("CREATING SUMMARY TABLE")
    print("="*80)

    summary_data = []

    for config, summary_df in experiments:
        exp_name = create_experiment_label(config)

        # Count significant biases
        if 'difference' in summary_df.columns and len(summary_df) > 0:
            n_significant = (abs(summary_df['difference']) > 5).sum()
            max_bias = summary_df['difference'].abs().max()
            mean_abs_bias = summary_df['difference'].abs().mean()
        else:
            n_significant = 0
            max_bias = 0
            mean_abs_bias = 0

        summary_data.append({
            'Experiment': exp_name,
            'Dataset': config.get('dataset', '?'),
            'Provider': config.get('provider', '?'),
            'Model': config.get('model', '?'),
            'Prompt Styles': len(config.get('prompt_styles', [])),
            'Trials': config.get('n_trials', 0),
            'Significant Biases (>5pp)': n_significant,
            'Max Bias (pp)': f"{max_bias:.1f}",
            'Mean Abs Bias (pp)': f"{mean_abs_bias:.1f}"
        })

    summary_df = pd.DataFrame(summary_data)

    # Save
    csv_path = output_dir / 'experiments_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # Print
    print("\n" + summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--results-dirs', type=str, nargs='+',
                       help='List of results directories to compare')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all experiments in outputs/experiments/')
    parser.add_argument('--output-dir', type=str, default='outputs/comparisons',
                       help='Directory to save comparison results')

    args = parser.parse_args()

    # Determine which experiments to compare
    if args.compare_all:
        # Find all experiment directories
        results_dirs = list(Path('outputs/experiments').glob('*'))
        results_dirs = [d for d in results_dirs if d.is_dir() and (d / 'config.pkl').exists()]
    elif args.results_dirs:
        results_dirs = [Path(d) for d in args.results_dirs]
    else:
        print("Error: Specify --results-dirs or --compare-all")
        return

    if len(results_dirs) < 2:
        print("Error: Need at least 2 experiments to compare")
        return

    print('='*80)
    print('CROSS-EXPERIMENT COMPARISON')
    print('='*80)

    print(f"\nFound {len(results_dirs)} experiments:")
    for d in results_dirs:
        print(f"  - {d.name}")

    # Load all experiments
    print("\nLoading experiments...")
    experiments = []
    for results_dir in results_dirs:
        try:
            config, summary_df = load_experiment_results(results_dir)
            experiments.append((config, summary_df))
            print(f"  ✓ {results_dir.name}")
        except Exception as e:
            print(f"  ✗ {results_dir.name}: {e}")

    if len(experiments) < 2:
        print("\nError: Failed to load enough experiments")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create summary table
    create_summary_table(experiments, output_dir)

    # Cross-dataset comparison
    compare_across_datasets(experiments, output_dir)

    # Cross-model comparison
    compare_across_models(experiments, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
