"""
Analyze and visualize prompt style comparison experiment

Compares how different prompt styles affect content selection bias,
particularly focusing on sentiment (positive vs. negative).
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


def load_results(results_path: str = './outputs/prompt_style_comparison/prompt_style_results.pkl'):
    """Load experiment results."""

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    return results


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

            # Try to get pre-computed diffs first
            if diff_key in r:
                for val, diff in r[diff_key].items():
                    if val not in all_diffs:
                        all_diffs[val] = []
                    all_diffs[val].append(diff)
            else:
                # Compute diffs from pool and recommended distributions
                pool_key = f'pool_{attr_name}'
                rec_key = f'recommended_{attr_name}'

                if pool_key in r and rec_key in r:
                    pool_dist = r[pool_key]
                    rec_dist = r[rec_key]

                    # Compute differences for all values
                    all_values = set(list(pool_dist.keys()) + list(rec_dist.keys()))
                    for val in all_values:
                        pool_pct = pool_dist.get(val, 0) * 100
                        rec_pct = rec_dist.get(val, 0) * 100
                        diff = rec_pct - pool_pct

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
        # Boolean attributes (e.g., has_emoji)
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


def create_comparison_plots(grouped_results: Dict[str, List[Dict]], output_dir: Path):
    """Create comparison visualizations across prompt styles."""

    print('Creating comparison visualizations...')
    print()

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sentiment comparison (KEY PLOT for the user's question)
    print('  - Sentiment comparison')
    create_sentiment_comparison(grouped_results, plots_dir)

    # 2. Sentiment polarity (continuous) comparison
    print('  - Sentiment polarity comparison')
    create_sentiment_polarity_comparison(grouped_results, plots_dir)

    # 3. Polarization comparison
    print('  - Polarization comparison')
    create_polarization_comparison(grouped_results, plots_dir)

    # 4. Formality comparison
    print('  - Formality comparison')
    create_formality_comparison(grouped_results, plots_dir)

    # 5. Emoji usage comparison
    print('  - Emoji usage comparison')
    create_emoji_comparison(grouped_results, plots_dir)

    # 6. Topic comparison
    print('  - Topic comparison')
    create_topic_comparison(grouped_results, plots_dir)

    # 7. Political leaning comparison
    print('  - Political leaning comparison')
    create_political_comparison(grouped_results, plots_dir)

    # 8. Gender comparison
    print('  - Gender comparison')
    create_gender_comparison(grouped_results, plots_dir)

    # 9. Race/ethnicity comparison
    print('  - Race/ethnicity comparison')
    create_race_comparison(grouped_results, plots_dir)

    # 10. Age/generation comparison
    print('  - Age/generation comparison')
    create_age_comparison(grouped_results, plots_dir)

    # 11. Education comparison
    print('  - Education comparison')
    create_education_comparison(grouped_results, plots_dir)

    print()
    print(f'✓ Plots saved to {plots_dir}')


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


def create_sentiment_polarity_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare sentiment polarity (continuous score) across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect polarity differences
    polarity_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'sentiment_polarity', 'numeric')

        polarity_data.append({
            'Prompt Style': style,
            'Mean Difference': stats['mean_diff'],
            'Std Error': stats['std_diff'] / np.sqrt(stats['n_trials']) if stats['n_trials'] > 0 else 0
        })

    df = pd.DataFrame(polarity_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    ax.bar(x, df['Mean Difference'], yerr=df['Std Error'], capsize=5, color='steelblue', alpha=0.7)

    ax.set_xlabel('Prompt Style', fontsize=12)
    ax.set_ylabel('Sentiment Polarity Difference', fontsize=12)
    ax.set_title('Sentiment Polarity Bias Across Prompt Styles\n(Positive = More positive sentiment, Negative = More negative sentiment)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Prompt Style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / 'sentiment_polarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_polarization_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare polarization score across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect polarization differences
    polar_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'polarization_score', 'numeric')

        polar_data.append({
            'Prompt Style': style,
            'Mean Difference': stats['mean_diff'],
            'Std Error': stats['std_diff'] / np.sqrt(stats['n_trials']) if stats['n_trials'] > 0 else 0
        })

    df = pd.DataFrame(polar_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    ax.bar(x, df['Mean Difference'], yerr=df['Std Error'], capsize=5, color='coral', alpha=0.7)

    ax.set_xlabel('Prompt Style', fontsize=12)
    ax.set_ylabel('Polarization Score Difference', fontsize=12)
    ax.set_title('Polarization Bias Across Prompt Styles\n(Positive = More polarizing content)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Prompt Style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / 'polarization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_formality_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare formality score across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect formality differences
    formal_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'formality_score', 'numeric')

        formal_data.append({
            'Prompt Style': style,
            'Mean Difference': stats['mean_diff'],
            'Std Error': stats['std_diff'] / np.sqrt(stats['n_trials']) if stats['n_trials'] > 0 else 0
        })

    df = pd.DataFrame(formal_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    ax.bar(x, df['Mean Difference'], yerr=df['Std Error'], capsize=5, color='mediumseagreen', alpha=0.7)

    ax.set_xlabel('Prompt Style', fontsize=12)
    ax.set_ylabel('Formality Score Difference', fontsize=12)
    ax.set_title('Formality Bias Across Prompt Styles\n(Positive = More formal content)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Prompt Style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / 'formality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_emoji_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare emoji usage across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect emoji differences
    emoji_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'has_emoji', 'binary')

        emoji_data.append({
            'Prompt Style': style,
            'Mean Difference': stats['mean_diff'],
            'Std Error': stats['std_diff'] / np.sqrt(stats['n_trials']) if stats['n_trials'] > 0 else 0
        })

    df = pd.DataFrame(emoji_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    ax.bar(x, df['Mean Difference'], yerr=df['Std Error'], capsize=5, color='gold', alpha=0.7)

    ax.set_xlabel('Prompt Style', fontsize=12)
    ax.set_ylabel('Emoji Usage Difference (%)', fontsize=12)
    ax.set_title('Emoji Usage Bias Across Prompt Styles\n(Positive = More tweets with emojis)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Prompt Style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / 'emoji_comparison.png', dpi=300, bbox_inches='tight')
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


def create_political_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare political leaning across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect political differences
    political_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'political_leaning_value', 'categorical')

        for leaning, mean_diff in stats['mean_diff'].items():
            political_data.append({
                'Prompt Style': style,
                'Political Leaning': leaning,
                'Difference (pp)': mean_diff
            })

    df = pd.DataFrame(political_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Political Leaning', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Political Leaning', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Political Leaning Bias Across Prompt Styles', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'political_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_gender_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare gender distribution across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect gender differences
    gender_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'gender_value', 'categorical')

        for gender, mean_diff in stats['mean_diff'].items():
            gender_data.append({
                'Prompt Style': style,
                'Gender': gender,
                'Difference (pp)': mean_diff
            })

    df = pd.DataFrame(gender_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Gender', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Gender Bias Across Prompt Styles\n(Positive = Over-represented)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / 'gender_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_race_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare race/ethnicity distribution across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect race differences
    race_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'race_ethnicity_value', 'categorical')

        for race, mean_diff in stats['mean_diff'].items():
            race_data.append({
                'Prompt Style': style,
                'Race/Ethnicity': race,
                'Difference (pp)': mean_diff
            })

    df = pd.DataFrame(race_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Race/Ethnicity', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Race/Ethnicity', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Race/Ethnicity Bias Across Prompt Styles\n(Positive = Over-represented)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'race_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_age_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare age/generation distribution across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect age differences
    age_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'age_generation_value', 'categorical')

        for age, mean_diff in stats['mean_diff'].items():
            age_data.append({
                'Prompt Style': style,
                'Age/Generation': age,
                'Difference (pp)': mean_diff
            })

    df = pd.DataFrame(age_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Age/Generation', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Age/Generation', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Age/Generation Bias Across Prompt Styles\n(Positive = Over-represented)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'age_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_education_comparison(grouped_results: Dict[str, List[Dict]], plots_dir: Path):
    """Compare education level distribution across prompt styles."""

    styles = list(grouped_results.keys())

    # Collect education differences
    education_data = []

    for style in styles:
        results = grouped_results[style]
        stats = compute_style_statistics(results, 'education_level_value', 'categorical')

        for education, mean_diff in stats['mean_diff'].items():
            education_data.append({
                'Prompt Style': style,
                'Education Level': education,
                'Difference (pp)': mean_diff
            })

    df = pd.DataFrame(education_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot for grouped bar chart
    pivot = df.pivot(index='Education Level', columns='Prompt Style', values='Difference (pp)')

    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Education Level', fontsize=12)
    ax.set_ylabel('Difference from Pool (%)', fontsize=12)
    ax.set_title('Education Level Bias Across Prompt Styles\n(Positive = Over-represented)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(title='Prompt Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'education_comparison.png', dpi=300, bbox_inches='tight')
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

    # Load results
    print('='*80)
    print('PROMPT STYLE COMPARISON ANALYSIS')
    print('='*80)
    print()

    results = load_results()

    print(f'Loaded {len(results)} trial results')

    # Group by style
    grouped = group_by_style(results)

    print(f'Prompt styles: {list(grouped.keys())}')
    print(f'Trials per style: {len(grouped[list(grouped.keys())[0]])}')
    print()

    # Output directory
    output_dir = Path('./outputs/prompt_style_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary table
    summary_df = create_summary_table(grouped, output_dir)

    print('Summary Statistics:')
    print(summary_df.to_string(index=False))
    print()

    # Create comparison plots
    print('='*80)
    print('CREATING VISUALIZATIONS')
    print('='*80)
    print()

    create_comparison_plots(grouped, output_dir)

    print()
    print('='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print()
    print(f'Results saved to: {output_dir}')
    print()
    print('Key findings:')
    print('  - Check sentiment_comparison.png for positive vs. negative content bias')
    print('  - Check sentiment_polarity_comparison.png for continuous sentiment scores')
    print('  - Check polarization_comparison.png for controversial content bias')
    print('  - Check emoji_comparison.png for stylistic differences')


if __name__ == '__main__':
    main()
