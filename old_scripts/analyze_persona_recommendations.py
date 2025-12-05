"""
Analyze bias in persona recommendation experiment

Compares pool vs. recommended distributions across:
- User-level attributes: gender, political leaning, race, age, education, profession
- Tweet-level attributes: sentiment, topics, style, polarization

Statistical tests and visualizations.
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

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_path: str = './outputs/persona_recommendations/recommendation_results.pkl'):
    """Load experiment results."""

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    return results


def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results across trials into a DataFrame."""

    # Separate by version
    anonymous_results = [r for r in results if r['version'] == 'anonymous']
    with_author_results = [r for r in results if r['version'] == 'with_author']

    print(f"Anonymous trials: {len(anonymous_results)}")
    print(f"With-author trials: {len(with_author_results)}")

    return anonymous_results, with_author_results


def compute_bias_statistics(results: List[Dict], attr_name: str,
                            attr_type: str = 'categorical') -> Dict[str, Any]:
    """
    Compute bias statistics for a single attribute across all trials.

    Args:
        results: List of trial results
        attr_name: Name of attribute (e.g., 'gender_value', 'sentiment_label')
        attr_type: 'categorical', 'binary', or 'numeric'

    Returns:
        Dict with statistical analysis
    """

    # Extract pool and recommended distributions for each trial
    pool_key = f'pool_{attr_name}'
    rec_key = f'recommended_{attr_name}'
    diff_key = f'diff_{attr_name}'

    # Handle different formats
    if attr_type == 'categorical':
        # For categorical attributes, we have distributions
        all_values = set()
        for r in results:
            if pool_key in r:
                all_values.update(r[pool_key].keys())
            if rec_key in r:
                all_values.update(r[rec_key].keys())

        # Compute mean differences across trials
        mean_diffs = {}
        std_diffs = {}

        for val in all_values:
            diffs = []
            for r in results:
                if diff_key in r and val in r[diff_key]:
                    diffs.append(r[diff_key][val])

            if diffs:
                mean_diffs[val] = np.mean(diffs)
                std_diffs[val] = np.std(diffs)

        # Compute mean pool and recommended distributions
        mean_pool = {}
        mean_rec = {}

        for val in all_values:
            pool_vals = [r[pool_key].get(val, 0) * 100 for r in results if pool_key in r]
            rec_vals = [r[rec_key].get(val, 0) * 100 for r in results if rec_key in r]

            if pool_vals:
                mean_pool[val] = np.mean(pool_vals)
            if rec_vals:
                mean_rec[val] = np.mean(rec_vals)

        return {
            'type': 'categorical',
            'values': sorted(all_values),
            'mean_pool': mean_pool,
            'mean_recommended': mean_rec,
            'mean_diff': mean_diffs,
            'std_diff': std_diffs,
            'n_trials': len(results)
        }

    elif attr_type == 'binary':
        # For binary attributes (has_emoji, etc.)
        pool_pct_key = f'pool_{attr_name}_pct'
        rec_pct_key = f'recommended_{attr_name}_pct'
        diff_pct_key = f'diff_{attr_name}_pct'

        pool_pcts = [r[pool_pct_key] for r in results if pool_pct_key in r]
        rec_pcts = [r[rec_pct_key] for r in results if rec_pct_key in r]
        diffs = [r[diff_pct_key] for r in results if diff_pct_key in r]

        return {
            'type': 'binary',
            'mean_pool': np.mean(pool_pcts) if pool_pcts else 0,
            'mean_recommended': np.mean(rec_pcts) if rec_pcts else 0,
            'mean_diff': np.mean(diffs) if diffs else 0,
            'std_diff': np.std(diffs) if diffs else 0,
            'n_trials': len(diffs)
        }

    elif attr_type == 'numeric':
        # For numeric attributes (polarization_score, etc.)
        pool_mean_key = f'pool_{attr_name}_mean'
        rec_mean_key = f'recommended_{attr_name}_mean'
        diff_key_num = f'diff_{attr_name}'

        pool_means = [r[pool_mean_key] for r in results if pool_mean_key in r]
        rec_means = [r[rec_mean_key] for r in results if rec_mean_key in r]
        diffs = [r[diff_key_num] for r in results if diff_key_num in r]

        return {
            'type': 'numeric',
            'mean_pool': np.mean(pool_means) if pool_means else 0,
            'mean_recommended': np.mean(rec_means) if rec_means else 0,
            'mean_diff': np.mean(diffs) if diffs else 0,
            'std_diff': np.std(diffs) if diffs else 0,
            'n_trials': len(diffs)
        }


def perform_significance_tests(stats: Dict, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform statistical significance tests.

    For categorical: one-sample t-test on differences (H0: diff = 0)
    For numeric/binary: same approach
    """

    results = {}

    if stats['type'] == 'categorical':
        for val, mean_diff in stats['mean_diff'].items():
            std_diff = stats['std_diff'].get(val, 0)
            n = stats['n_trials']

            if n > 1 and std_diff > 0:
                # One-sample t-test: is mean_diff significantly different from 0?
                t_stat = (mean_diff / std_diff) * np.sqrt(n)
                p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 1))

                results[val] = {
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha
                }

    elif stats['type'] in ['binary', 'numeric']:
        mean_diff = stats['mean_diff']
        std_diff = stats['std_diff']
        n = stats['n_trials']

        if n > 1 and std_diff > 0:
            t_stat = (mean_diff / std_diff) * np.sqrt(n)
            p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 1))

            results['overall'] = {
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }

    return results


def create_comparison_plot(stats: Dict, attr_name: str, title: str,
                           output_path: Path, version: str):
    """Create bar plot comparing pool vs. recommended distributions."""

    if stats['type'] == 'categorical':
        values = stats['values']
        pool_pcts = [stats['mean_pool'].get(v, 0) for v in values]
        rec_pcts = [stats['mean_recommended'].get(v, 0) for v in values]
        diffs = [stats['mean_diff'].get(v, 0) for v in values]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Pool vs Recommended
        ax1 = axes[0]
        x = np.arange(len(values))
        width = 0.35

        bars1 = ax1.bar(x - width/2, pool_pcts, width, label='Pool',
                        color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, rec_pcts, width, label='Recommended',
                        color='coral', alpha=0.8, edgecolor='black')

        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{title}\nPool vs Recommended ({version})',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(values, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        # Plot 2: Differences (Recommended - Pool)
        ax2 = axes[1]
        colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in diffs]
        bars = ax2.bar(values, diffs, color=colors, alpha=0.7, edgecolor='black')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Difference (pp)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Bias: Recommended - Pool\n({version})',
                     fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, diff in zip(bars, diffs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{diff:+.1f}pp', ha='center',
                    va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    elif stats['type'] in ['binary', 'numeric']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        pool_val = stats['mean_pool']
        rec_val = stats['mean_recommended']
        diff_val = stats['mean_diff']

        bars = ax.bar(['Pool', 'Recommended'], [pool_val, rec_val],
                     color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')

        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n({version})', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add values
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add difference annotation
        ax.text(0.5, max(pool_val, rec_val) * 1.1,
               f'Diff: {diff_val:+.2f}',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def analyze_version(results: List[Dict], version_name: str, output_dir: Path):
    """Analyze bias for one version (anonymous or with-author)."""

    print(f"\n{'='*80}")
    print(f"ANALYZING: {version_name.upper()}")
    print(f"{'='*80}\n")

    version_dir = output_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Define attributes to analyze
    user_level_categorical = [
        ('gender_value', 'Gender'),
        ('political_leaning_value', 'Political Leaning'),
        ('race_ethnicity_value', 'Race/Ethnicity'),
        ('age_generation_value', 'Age/Generation'),
        ('education_level_value', 'Education Level')
    ]

    tweet_level_categorical = [
        ('sentiment_label', 'Sentiment'),
        ('primary_topic', 'Primary Topic'),
        ('controversy_level', 'Controversy Level')
    ]

    binary_attributes = [
        'has_emoji',
        'has_hashtag',
        'has_mention'
    ]

    numeric_attributes = [
        'polarization_score',
        'formality_score'
    ]

    all_stats = {}
    all_sig_tests = {}

    # Analyze categorical attributes
    for attr_name, attr_title in user_level_categorical + tweet_level_categorical:
        print(f"Analyzing: {attr_title}")
        stats = compute_bias_statistics(results, attr_name, 'categorical')
        sig_tests = perform_significance_tests(stats)

        all_stats[attr_name] = stats
        all_sig_tests[attr_name] = sig_tests

        # Create plot
        plot_path = version_dir / f'{attr_name}_comparison.png'
        create_comparison_plot(stats, attr_name, attr_title, plot_path, version_name)

        # Print summary
        print(f"  Significant biases:")
        for val, test in sig_tests.items():
            if test['significant']:
                print(f"    {val}: {test['mean_diff']:+.2f}pp (p={test['p_value']:.4f})")
        print()

    # Analyze binary attributes
    for attr_name in binary_attributes:
        print(f"Analyzing: {attr_name}")
        stats = compute_bias_statistics(results, attr_name, 'binary')
        sig_tests = perform_significance_tests(stats)

        all_stats[attr_name] = stats
        all_sig_tests[attr_name] = sig_tests

        # Create plot
        plot_path = version_dir / f'{attr_name}_comparison.png'
        create_comparison_plot(stats, attr_name, attr_name.replace('_', ' ').title(),
                             plot_path, version_name)

        if 'overall' in sig_tests and sig_tests['overall']['significant']:
            print(f"  Significant bias: {sig_tests['overall']['mean_diff']:+.2f}pp " +
                  f"(p={sig_tests['overall']['p_value']:.4f})")
        print()

    # Analyze numeric attributes
    for attr_name in numeric_attributes:
        print(f"Analyzing: {attr_name}")
        stats = compute_bias_statistics(results, attr_name, 'numeric')
        sig_tests = perform_significance_tests(stats)

        all_stats[attr_name] = stats
        all_sig_tests[attr_name] = sig_tests

        # Create plot
        plot_path = version_dir / f'{attr_name}_comparison.png'
        create_comparison_plot(stats, attr_name, attr_name.replace('_', ' ').title(),
                             plot_path, version_name)

        if 'overall' in sig_tests and sig_tests['overall']['significant']:
            print(f"  Significant bias: {sig_tests['overall']['mean_diff']:+.2f} " +
                  f"(p={sig_tests['overall']['p_value']:.4f})")
        print()

    # Save statistics
    stats_path = version_dir / 'bias_statistics.json'
    with open(stats_path, 'w') as f:
        # Convert to JSON-serializable format
        json_stats = {}
        for key, val in all_stats.items():
            json_stats[key] = {k: v for k, v in val.items() if k != 'values'}
            if 'values' in val:
                json_stats[key]['values'] = list(val['values'])
        json.dump(json_stats, f, indent=2)

    print(f"✓ Statistics saved to {stats_path}")
    print(f"✓ Plots saved to {version_dir}")

    return all_stats, all_sig_tests


def create_summary_dashboard(anon_stats, author_stats, output_dir: Path):
    """Create summary dashboard comparing both versions."""

    print(f"\n{'='*80}")
    print("CREATING SUMMARY DASHBOARD")
    print(f"{'='*80}\n")

    # Create comparison visualizations
    # TODO: Implement cross-version comparison plots

    print("✓ Summary dashboard created")


def main():
    """Main analysis pipeline."""

    print('='*80)
    print('PERSONA RECOMMENDATION BIAS ANALYSIS')
    print('='*80)
    print()

    # Load results
    results_path = Path('./outputs/persona_recommendations/recommendation_results.pkl')
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Please run the experiment first.")
        return

    results = load_results(results_path)
    print(f"Loaded {len(results)} trial results")
    print()

    # Separate by version
    anon_results, author_results = aggregate_results(results)

    # Create output directory
    output_dir = Path('./outputs/persona_recommendations/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each version
    anon_stats, anon_sig = analyze_version(anon_results, 'anonymous', output_dir)
    author_stats, author_sig = analyze_version(author_results, 'with_author', output_dir)

    # Create summary dashboard
    create_summary_dashboard(anon_stats, author_stats, output_dir)

    print()
    print('='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print(f'Results saved to: {output_dir}')
    print()
    print('Key files:')
    print(f'  - {output_dir}/anonymous/bias_statistics.json')
    print(f'  - {output_dir}/with_author/bias_statistics.json')
    print(f'  - Visualization plots in each subdirectory')


if __name__ == '__main__':
    main()
