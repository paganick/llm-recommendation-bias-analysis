"""
Analyze and visualize experiment results

Compares how different prompt styles affect content selection bias.
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

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_dir: Path):
    """Load experiment results and config."""

    results_path = results_dir / 'prompt_style_results.pkl'
    config_path = results_dir / 'config.pkl'

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    return results, config


def group_by_style(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by prompt style."""

    grouped = {}
    for r in results:
        style = r['prompt_style']
        if style not in grouped:
            grouped[style] = []
        grouped[style].append(r)

    return grouped


def compute_style_statistics(results: List[Dict], attr_name: str,
                             attr_type: str = 'categorical') -> Dict[str, Any]:
    """
    Compute statistics for an attribute across trials of a single prompt style.
    """

    if attr_type == 'categorical':
        # Collect all trials' differences
        all_diffs = {}
        for r in results:
            diff_key = f'diff_{attr_name}'

            if diff_key in r:
                for val, diff in r[diff_key].items():
                    if val not in all_diffs:
                        all_diffs[val] = []
                    all_diffs[val].append(diff)

        # Compute mean and std for each value
        stats = {
            'type': 'categorical',
            'attribute': attr_name,
            'n_trials': len(results),
            'mean_diff': {},
            'std_diff': {}
        }

        for val, diffs in all_diffs.items():
            stats['mean_diff'][val] = np.mean(diffs)
            stats['std_diff'][val] = np.std(diffs, ddof=1) if len(diffs) > 1 else 0

    elif attr_type == 'numeric':
        # Collect all trials' differences
        diffs = []
        for r in results:
            diff_key = f'diff_{attr_name}'
            if diff_key in r:
                diffs.append(r[diff_key])

        stats = {
            'type': 'numeric',
            'attribute': attr_name,
            'n_trials': len(results),
            'mean_diff': np.mean(diffs) if diffs else 0,
            'std_diff': np.std(diffs, ddof=1) if len(diffs) > 1 else 0
        }

    elif attr_type == 'binary':
        # Boolean attributes
        diffs = []
        for r in results:
            diff_key = f'diff_{attr_name}_pct'
            if diff_key in r:
                diffs.append(r[diff_key])

        stats = {
            'type': 'binary',
            'attribute': attr_name,
            'n_trials': len(results),
            'mean_diff': np.mean(diffs) if diffs else 0,
            'std_diff': np.std(diffs, ddof=1) if len(diffs) > 1 else 0
        }

    return stats


def create_sentiment_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare sentiment distribution across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect sentiment differences for each style
    sentiment_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'sentiment_label', 'categorical')

        for sentiment, mean_diff in stats['mean_diff'].items():
            sentiment_data.append({
                'Prompt Style': style,
                'Sentiment': sentiment,
                'Difference (pp)': mean_diff
            })

    if not sentiment_data:
        print("  (No sentiment data available)")
        return

    df = pd.DataFrame(sentiment_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Sentiment', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Sentiment Bias Across Prompt Styles\n(Positive = Over-represented, Negative = Under-represented)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / 'sentiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_numeric_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path,
                              attr_name: str, title: str, ylabel: str):
    """Compare numeric attribute across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect differences
    data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, attr_name, 'numeric')

        data.append({
            'Prompt Style': style,
            'Mean Difference': stats['mean_diff'],
            'Std Error': stats['std_diff'] / np.sqrt(stats['n_trials']) if stats['n_trials'] > 0 else 0
        })

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    ax.bar(x, df['Mean Difference'], yerr=df['Std Error'], capsize=5, color='steelblue', alpha=0.7)

    ax.set_xlabel('Prompt Style', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Prompt Style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save with sanitized filename
    filename = attr_name.replace('_', '-') + '-comparison.png'
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_topic_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare topic distribution across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect topic differences
    topic_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'primary_topic', 'categorical')

        for topic, mean_diff in stats['mean_diff'].items():
            topic_data.append({
                'Prompt Style': style,
                'Topic': topic,
                'Difference (pp)': mean_diff
            })

    if not topic_data:
        print("  (No topic data available)")
        return

    df = pd.DataFrame(topic_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Topic', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Topic Bias Across Prompt Styles', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'topic_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(grouped_results: Dict[str, List[Dict]], output_dir: Path):
    """Create summary statistics table."""

    print('Creating summary statistics table...')

    styles = list(grouped_results.keys())

    summary = []

    attributes = [
        ('sentiment_polarity', 'numeric', 'Sentiment Polarity'),
        ('polarization_score', 'numeric', 'Polarization Score'),
        ('formality_score', 'numeric', 'Formality Score'),
        ('has_emoji', 'binary', 'Emoji Usage (%)'),
    ]

    for attr, attr_type, label in attributes:
        row = {'Attribute': label}

        for style in styles:
            results = grouped_results[style]
            stats = compute_style_statistics(results, attr, attr_type)
            row[style] = f"{stats['mean_diff']:.3f}"

        summary.append(row)

    df = pd.DataFrame(summary)

    # Save to CSV
    csv_path = output_dir / 'summary_statistics.csv'
    df.to_csv(csv_path, index=False)

    print(f'✓ Summary table saved to {csv_path}')
    print()

    return df


def main():
    """Main analysis pipeline."""

    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # Load results
    print('='*80)
    print('EXPERIMENT RESULTS ANALYSIS')
    print('='*80)
    print()

    results, config = load_results(results_dir)

    print(f'Loaded {len(results)} trial results')
    print()
    print('Experiment configuration:')
    for key, val in config.items():
        print(f'  {key}: {val}')
    print()

    # Group by style
    grouped = group_by_style(results)

    print(f'Prompt styles: {list(grouped.keys())}')
    print(f'Trials per style: {len(grouped[list(grouped.keys())[0]])}')
    print()

    # Create plots directory
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create summary table
    summary_df = create_summary_table(grouped, results_dir)

    print('Summary Statistics:')
    print(summary_df.to_string(index=False))
    print()

    # Create comparison plots
    print('='*80)
    print('CREATING VISUALIZATIONS')
    print('='*80)
    print()

    print('  - Sentiment comparison')
    create_sentiment_comparison(grouped, plots_dir)

    print('  - Sentiment polarity comparison')
    create_numeric_comparison(
        grouped, plots_dir,
        'sentiment_polarity',
        'Sentiment Polarity Bias Across Prompt Styles\n(Positive = More positive sentiment)',
        'Sentiment Polarity Difference'
    )

    print('  - Polarization comparison')
    create_numeric_comparison(
        grouped, plots_dir,
        'polarization_score',
        'Polarization Bias Across Prompt Styles\n(Positive = More polarizing content)',
        'Polarization Score Difference'
    )

    print('  - Formality comparison')
    create_numeric_comparison(
        grouped, plots_dir,
        'formality_score',
        'Formality Bias Across Prompt Styles\n(Positive = More formal content)',
        'Formality Score Difference'
    )

    print('  - Topic comparison')
    create_topic_comparison(grouped, plots_dir)

    print()
    print('='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print()
    print(f'Results saved to: {results_dir}')
    print(f'Plots saved to: {plots_dir}')
    print()
    print('Key findings:')
    print('  - Check sentiment_comparison.png for positive vs. negative content bias')
    print('  - Check sentiment-polarity-comparison.png for continuous sentiment scores')
    print('  - Check polarization-score-comparison.png for controversial content bias')
    print('  - Check topic_comparison.png for topic preferences')


if __name__ == '__main__':
    main()
