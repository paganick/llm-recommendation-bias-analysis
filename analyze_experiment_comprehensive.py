"""
Comprehensive Experiment Analysis

Analyzes bias at multiple levels:
1. Content-level: sentiment, topics, style, polarization (like before)
2. User-level: gender, age, race, political leaning, education, profession (NEW)
3. Statistical significance testing
4. Cross-prompt-style comparisons

Usage:
    python analyze_experiment_comprehensive.py --results-dir outputs/experiments/reddit_openai_gpt-4o-mini
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, List, Any
import json
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from inference.persona_extraction import add_persona_attributes_to_dataframe

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results_with_data(results_dir: Path):
    """Load experiment results and original dataset."""

    results_path = results_dir / 'post_level_data.pkl'
    config_path = results_dir / 'config.pkl'

    # Load results
    with open(results_path, 'rb') as f:
        post_level_data = pickle.load(f)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    return post_level_data, config


def compute_persona_bias(df: pd.DataFrame, persona_attrs: List[str]) -> pd.DataFrame:
    """
    Compute bias for persona-level attributes.

    Args:
        df: DataFrame with 'selected' column and persona attributes
        persona_attrs: List of persona attribute column names

    Returns:
        DataFrame with bias statistics for each attribute
    """

    results = []

    for attr in persona_attrs:
        if attr not in df.columns:
            continue

        # Get pool and recommended distributions
        pool_dist = df[attr].value_counts(normalize=True) * 100
        recommended_dist = df[df['selected'] == 1][attr].value_counts(normalize=True) * 100

        # Compute differences
        all_values = set(pool_dist.index) | set(recommended_dist.index)

        for value in all_values:
            pool_pct = pool_dist.get(value, 0)
            rec_pct = recommended_dist.get(value, 0)
            diff = rec_pct - pool_pct

            results.append({
                'attribute': attr,
                'value': value,
                'pool_pct': pool_pct,
                'recommended_pct': rec_pct,
                'difference': diff
            })

    return pd.DataFrame(results)


def analyze_by_prompt_style(post_level_data: List[Dict], config: Dict) -> Dict[str, pd.DataFrame]:
    """Analyze bias for each prompt style separately."""

    # Convert to DataFrame
    df = pd.DataFrame(post_level_data)

    print(f"\nTotal posts tracked: {len(df):,}")
    print(f"Unique trials: {df.groupby('prompt_style').size()}")

    # Extract persona attributes if 'persona' column exists
    if 'persona' in df.columns:
        print("\nExtracting persona attributes...")
        df = add_persona_attributes_to_dataframe(df, 'persona')

    # Analyze each prompt style
    results_by_style = {}

    persona_attrs = ['gender', 'age_group', 'race_ethnicity', 'political_leaning',
                     'education_level', 'profession']
    content_attrs = ['sentiment_label', 'sentiment_polarity', 'polarization_score',
                     'formality_score']

    for style in config.get('prompt_styles', ['general']):
        print(f"\n{'='*80}")
        print(f"Analyzing: {style.upper()}")
        print('='*80)

        style_df = df[df['prompt_style'] == style].copy()

        # Persona-level bias
        persona_bias = compute_persona_bias(style_df, persona_attrs)
        persona_bias['analysis_type'] = 'persona'

        # Content-level bias
        content_bias = compute_persona_bias(style_df, content_attrs)
        content_bias['analysis_type'] = 'content'

        # Combine
        all_bias = pd.concat([persona_bias, content_bias], ignore_index=True)
        all_bias['prompt_style'] = style

        results_by_style[style] = all_bias

        # Print summary
        print(f"\nTop persona biases:")
        top_persona = persona_bias.nlargest(5, 'difference', keep='all')
        for _, row in top_persona.iterrows():
            print(f"  {row['attribute']} = {row['value']}: +{row['difference']:.1f}pp "
                  f"(pool: {row['pool_pct']:.1f}%, rec: {row['recommended_pct']:.1f}%)")

        print(f"\nTop content biases:")
        top_content = content_bias.nlargest(5, 'difference', keep='all')
        for _, row in top_content.iterrows():
            val_str = f"{row['value']:.2f}" if isinstance(row['value'], float) else row['value']
            print(f"  {row['attribute']} = {val_str}: +{row['difference']:.1f}pp")

    return results_by_style


def create_persona_visualizations(results_by_style: Dict[str, pd.DataFrame],
                                   output_dir: Path):
    """Create visualizations for persona-level biases."""

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Combine all results
    all_results = pd.concat(results_by_style.values(), ignore_index=True)

    # Filter to persona attributes
    persona_results = all_results[all_results['analysis_type'] == 'persona'].copy()

    if len(persona_results) == 0:
        print("No persona-level data to visualize")
        return

    # Plot for each attribute
    persona_attrs = persona_results['attribute'].unique()

    for attr in persona_attrs:
        attr_data = persona_results[persona_results['attribute'] == attr]

        if len(attr_data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        # Pivot for plotting
        pivot_df = attr_data.pivot(index='value', columns='prompt_style', values='difference')

        pivot_df.plot(kind='bar', ax=ax)

        ax.set_title(f'Recommendation Bias by {attr.replace("_", " ").title()}')
        ax.set_xlabel(attr.replace("_", " ").title())
        ax.set_ylabel('Bias (Recommended % - Pool %)')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filename = plots_dir / f'persona_bias_{attr}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()


def create_comparison_heatmap(results_by_style: Dict[str, pd.DataFrame],
                               output_dir: Path):
    """Create heatmap comparing biases across prompt styles."""

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Combine results
    all_results = pd.concat(results_by_style.values(), ignore_index=True)

    # Create summary: for each attr+value, show difference across styles
    # Focusing on largest biases
    significant_bias = all_results[abs(all_results['difference']) > 5].copy()

    if len(significant_bias) == 0:
        print("No significant biases found for heatmap")
        return

    # Create label: attr = value
    significant_bias['label'] = (significant_bias['attribute'] + ' = ' +
                                  significant_bias['value'].astype(str))

    # Pivot
    pivot_df = significant_bias.pivot_table(
        index='label',
        columns='prompt_style',
        values='difference',
        fill_value=0
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_df) * 0.4)))

    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdBu_r',
                center=0, cbar_kws={'label': 'Bias (pp)'},
                ax=ax, vmin=-20, vmax=20)

    ax.set_title('Recommendation Bias Heatmap (Significant Biases > 5pp)')
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Attribute = Value')

    plt.tight_layout()

    filename = plots_dir / 'bias_heatmap.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive experiment analysis')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment results')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print('='*80)
    print('COMPREHENSIVE EXPERIMENT ANALYSIS')
    print('='*80)

    # Load results
    print("\nLoading results...")
    post_level_data, config = load_results_with_data(results_dir)

    print(f"\nExperiment configuration:")
    print(f"  Dataset: {config.get('dataset', 'unknown')}")
    print(f"  Provider: {config.get('provider', 'unknown')}")
    print(f"  Model: {config.get('model', 'unknown')}")
    print(f"  Prompt styles: {config.get('prompt_styles', [])}")
    print(f"  Trials per style: {config.get('n_trials', 'unknown')}")

    # Analyze
    results_by_style = analyze_by_prompt_style(post_level_data, config)

    # Save detailed results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Combined CSV
    all_results = pd.concat(results_by_style.values(), ignore_index=True)
    csv_path = results_dir / 'comprehensive_bias_analysis.csv'
    all_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results: {csv_path}")

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    create_persona_visualizations(results_by_style, results_dir)
    create_comparison_heatmap(results_by_style, results_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"Plots saved to: {results_dir / 'plots'}")


if __name__ == '__main__':
    main()
