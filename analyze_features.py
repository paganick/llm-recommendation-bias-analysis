#!/usr/bin/env python3
"""
Comprehensive feature analysis and visualization for LLM recommendation bias study.
Generates:
1. Feature documentation table with ranges, types, computation methods
2. Distribution plots for all features across three datasets (pool only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Define output directories
OUTPUT_DIR = Path("analysis_outputs/feature_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment paths
EXPERIMENTS = {
    'bluesky': 'outputs/experiments/bluesky_anthropic_claude-sonnet-4-5-20250929',
    'reddit': 'outputs/experiments/reddit_anthropic_claude-sonnet-4-5-20250929',
    'twitter': 'outputs/experiments/twitter_anthropic_claude-sonnet-4-5-20250929'
}

def load_pool_data(experiment_path):
    """Load unique posts from experiment (to avoid duplicates across trials)"""
    df = pd.read_csv(f"{experiment_path}/post_level_data.csv")
    # Get unique posts only (by original_index) to avoid counting same post multiple times
    unique_posts = df.drop_duplicates(subset='original_index').copy()
    return unique_posts

def get_feature_metadata():
    """Define metadata for all features"""

    # Author-level features
    author_features = {
        'author_gender': {
            'level': 'Author',
            'type': 'Categorical',
            'range': 'female, male, unknown',
            'computation': 'Keyword matching in persona text (she/her/woman → female; he/him/man → male)',
            'order': ['female', 'male', 'non-binary', 'unknown']
        },
        'author_political_leaning': {
            'level': 'Author',
            'type': 'Categorical',
            'range': 'left, center-left, center, center-right, right, apolitical, unknown',
            'computation': 'Keyword matching in persona text (liberal/progressive → left; conservative/republican → right; etc.)',
            'order': ['left', 'center-left', 'center', 'center-right', 'right', 'apolitical', 'unknown']
        },
        'author_is_minority': {
            'level': 'Author',
            'type': 'Categorical',
            'range': 'no, yes, unknown',
            'computation': 'Detected from persona text mentioning minority status',
            'order': ['no', 'yes', 'unknown']
        },
    }

    # Tweet/Post-level features
    tweet_features = {
        # Text metrics
        'text_length': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0 to ~1000+ characters',
            'computation': 'len(message)'
        },
        'word_count': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0 to ~200+ words',
            'computation': 'Number of whitespace-separated tokens'
        },
        'avg_word_length': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to ~15.0 characters',
            'computation': 'Mean length of all words in message'
        },

        # Sentiment features
        'sentiment_polarity': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '-1.0 (negative) to +1.0 (positive)',
            'computation': 'TextBlob sentiment analysis on message text'
        },
        'sentiment_subjectivity': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 (objective) to 1.0 (subjective)',
            'computation': 'TextBlob subjectivity score'
        },
        'sentiment_label': {
            'level': 'Tweet',
            'type': 'Categorical',
            'range': 'negative, neutral, positive',
            'computation': 'Discretized from sentiment_polarity (>0.1 → positive, <-0.1 → negative, else neutral)',
            'order': ['negative', 'neutral', 'positive']
        },
        'sentiment_positive': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'VADER sentiment analyzer positive score'
        },
        'sentiment_negative': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'VADER sentiment analyzer negative score'
        },
        'sentiment_neutral': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'VADER sentiment analyzer neutral score'
        },

        # Style features
        'formality_score': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 (informal) to 1.0 (formal)',
            'computation': 'Based on avg_word_length/8, penalized for emojis, caps, excessive punctuation'
        },
        'has_emoji': {
            'level': 'Tweet',
            'type': 'Binary',
            'range': '0 (no), 1 (yes)',
            'computation': 'Regex detection of emoji characters'
        },
        'has_hashtag': {
            'level': 'Tweet',
            'type': 'Binary',
            'range': '0 (no), 1 (yes)',
            'computation': 'Regex detection of # symbols'
        },
        'has_mention': {
            'level': 'Tweet',
            'type': 'Binary',
            'range': '0 (no), 1 (yes)',
            'computation': 'Regex detection of @ mentions'
        },
        'has_url': {
            'level': 'Tweet',
            'type': 'Binary',
            'range': '0 (no), 1 (yes)',
            'computation': 'Regex detection of http/https URLs'
        },

        # Content features
        'polarization_score': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 (neutral) to 1.0 (polarizing)',
            'computation': 'Keyword-based detection of divisive/controversial topics and strong opinions'
        },
        'has_polarizing_content': {
            'level': 'Tweet',
            'type': 'Binary',
            'range': '0 (no), 1 (yes)',
            'computation': 'polarization_score > threshold'
        },
        'controversy_level': {
            'level': 'Tweet',
            'type': 'Categorical',
            'range': 'low, medium, high',
            'computation': 'Categorical level based on controversial keywords and patterns',
            'order': ['low', 'medium', 'high']
        },

        # Topic features
        'primary_topic': {
            'level': 'Tweet',
            'type': 'Categorical',
            'range': 'politics, sports, entertainment, technology, health, personal, other',
            'computation': 'Zero-shot classification or keyword-based topic detection',
            'order': ['politics', 'sports', 'entertainment', 'technology', 'health', 'personal', 'other']
        },
        'primary_topic_score': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Confidence score for primary_topic classification'
        },

        # Toxicity features (Detoxify library)
        'toxicity': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Detoxify transformer model - overall toxicity score'
        },
        'severe_toxicity': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Detoxify - severe toxicity score'
        },
        'obscene': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Detoxify - obscenity score'
        },
        'threat': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Detoxify - threat score'
        },
        'insult': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Detoxify - insult score'
        },
        'identity_attack': {
            'level': 'Tweet',
            'type': 'Numerical',
            'range': '0.0 to 1.0',
            'computation': 'Detoxify - identity-based attack score'
        },
    }

    return {**author_features, **tweet_features}

def analyze_feature_distributions():
    """Analyze actual distributions across all datasets"""

    print("Loading pool data from all datasets...")
    pools = {}
    for dataset, path in EXPERIMENTS.items():
        print(f"  Loading {dataset}...")
        pools[dataset] = load_pool_data(path)

    print(f"\nDataset sizes (pool only):")
    for dataset, pool in pools.items():
        print(f"  {dataset}: {len(pool):,} posts")

    # Get feature metadata
    metadata = get_feature_metadata()

    # Collect actual statistics
    feature_stats = []

    for feature, meta in metadata.items():
        if feature not in pools['twitter'].columns:
            print(f"Warning: {feature} not found in data, skipping...")
            continue

        stats = {
            'Feature': feature,
            'Level': meta['level'],
            'Type': meta['type'],
            'Range (Specification)': meta['range'],
            'Computation': meta['computation']
        }

        # Compute actual ranges across datasets
        if meta['type'] == 'Numerical':
            all_values = []
            for pool in pools.values():
                # Convert to numeric, coercing errors
                numeric_vals = pd.to_numeric(pool[feature], errors='coerce').dropna().tolist()
                all_values.extend(numeric_vals)

            if len(all_values) > 0:
                all_values = np.array(all_values, dtype=float)
                stats['Actual Min'] = f"{np.min(all_values):.3f}"
                stats['Actual Max'] = f"{np.max(all_values):.3f}"
                stats['Actual Mean'] = f"{np.mean(all_values):.3f}"
                stats['Actual Std'] = f"{np.std(all_values):.3f}"
            else:
                stats['Actual Min'] = 'N/A'
                stats['Actual Max'] = 'N/A'
                stats['Actual Mean'] = 'N/A'
                stats['Actual Std'] = 'N/A'

        elif meta['type'] in ['Categorical', 'Binary']:
            all_values = []
            for pool in pools.values():
                all_values.extend(pool[feature].dropna().astype(str).tolist())

            if len(all_values) > 0:
                value_counts = pd.Series(all_values).value_counts()
                stats['Actual Values'] = ', '.join([f"{val} ({count})" for val, count in value_counts.head(5).items()])
            else:
                stats['Actual Values'] = 'N/A'

        feature_stats.append(stats)

    # Create DataFrame
    stats_df = pd.DataFrame(feature_stats)

    # Save to CSV
    output_file = OUTPUT_DIR / "feature_summary.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"\nFeature summary saved to: {output_file}")

    # Also save as formatted table
    print("\n" + "="*100)
    print("FEATURE SUMMARY TABLE")
    print("="*100)
    print(stats_df.to_string(index=False))
    print("="*100)

    return pools, metadata, stats_df

def plot_numerical_distributions(pools, feature, meta, output_dir):
    """Plot distribution of a numerical feature across datasets"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{feature}\n({meta["range"]})', fontsize=14, fontweight='bold')

    for idx, (dataset, pool) in enumerate(pools.items()):
        ax = axes[idx]
        values = pd.to_numeric(pool[feature], errors='coerce').dropna()

        if len(values) > 0:
            ax.hist(values, bins=50, alpha=0.7, color=f'C{idx}', edgecolor='black')
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
            ax.axvline(values.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {values.median():.3f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{dataset.capitalize()} (n={len(values):,})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / f'{feature}_distribution.png', bbox_inches='tight')
    plt.close()

def plot_categorical_distributions(pools, feature, meta, output_dir):
    """Plot distribution of a categorical feature across datasets"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{feature}\n({meta["range"]})', fontsize=14, fontweight='bold')

    # Get ordering if specified
    order = meta.get('order', None)

    for idx, (dataset, pool) in enumerate(pools.items()):
        ax = axes[idx]
        values = pool[feature].dropna().astype(str)

        if len(values) > 0:
            value_counts = values.value_counts()

            # Apply ordering if specified
            if order:
                # Reindex to order, filling missing categories with 0
                value_counts = value_counts.reindex(order, fill_value=0)

            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.set_title(f'{dataset.capitalize()} (n={len(values):,})')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / f'{feature}_distribution.png', bbox_inches='tight')
    plt.close()

def plot_binary_distributions(pools, feature, meta, output_dir):
    """Plot distribution of a binary feature across datasets"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data_to_plot = []
    labels = []

    for dataset, pool in pools.items():
        values = pd.to_numeric(pool[feature], errors='coerce').dropna()
        if len(values) > 0:
            # Calculate proportions
            prop_yes = (values == 1).sum() / len(values) * 100
            prop_no = (values == 0).sum() / len(values) * 100
            data_to_plot.append([prop_no, prop_yes])
            labels.append(f'{dataset.capitalize()}\n(n={len(values):,})')

    if len(data_to_plot) > 0:
        x = np.arange(len(labels))
        width = 0.35

        data_array = np.array(data_to_plot)
        ax.bar(x, data_array[:, 0], width, label='No (0)', color='lightcoral', edgecolor='black')
        ax.bar(x, data_array[:, 1], width, bottom=data_array[:, 0], label='Yes (1)', color='lightgreen', edgecolor='black')

        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{feature}\n({meta["range"]})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        for i, (no, yes) in enumerate(data_array):
            ax.text(i, no/2, f'{no:.1f}%', ha='center', va='center', fontweight='bold')
            ax.text(i, no + yes/2, f'{yes:.1f}%', ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / f'{feature}_distribution.png', bbox_inches='tight')
    plt.close()

def generate_all_plots(pools, metadata):
    """Generate distribution plots for all features"""

    print("\nGenerating distribution plots...")

    plot_dir = OUTPUT_DIR / "distributions"
    plot_dir.mkdir(exist_ok=True)

    for feature, meta in metadata.items():
        if feature not in pools['twitter'].columns:
            continue

        print(f"  Plotting {feature} ({meta['type']})...")

        try:
            if meta['type'] == 'Numerical':
                plot_numerical_distributions(pools, feature, meta, plot_dir)
            elif meta['type'] == 'Binary':
                plot_binary_distributions(pools, feature, meta, plot_dir)
            elif meta['type'] == 'Categorical':
                plot_categorical_distributions(pools, feature, meta, plot_dir)
        except Exception as e:
            print(f"    Error plotting {feature}: {e}")

    print(f"\nAll plots saved to: {plot_dir}")

def main():
    """Main analysis pipeline"""

    print("="*80)
    print("LLM RECOMMENDATION BIAS - FEATURE ANALYSIS")
    print("="*80)

    # Analyze feature distributions
    pools, metadata, stats_df = analyze_feature_distributions()

    # Generate all distribution plots
    generate_all_plots(pools, metadata)

    # Generate summary report
    report_file = OUTPUT_DIR / "FEATURE_ANALYSIS_REPORT.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("Overview\n")
        f.write("-"*80 + "\n")
        f.write(f"This report analyzes {len(metadata)} features across 3 datasets (Twitter, Bluesky, Reddit).\n")
        f.write(f"All statistics are computed on the POOL SET ONLY (unselected posts).\n\n")

        f.write("AUTHOR-LEVEL FEATURES (3 features)\n")
        f.write("="*80 + "\n")
        author_features = stats_df[stats_df['Level'] == 'Author']
        f.write(author_features.to_string(index=False))
        f.write("\n\n")

        f.write("TWEET-LEVEL FEATURES (25 features)\n")
        f.write("="*80 + "\n")
        tweet_features = stats_df[stats_df['Level'] == 'Tweet']
        f.write(tweet_features.to_string(index=False))
        f.write("\n\n")

        f.write("VISUALIZATION FILES\n")
        f.write("="*80 + "\n")
        f.write("All distribution plots are saved in distributions/ directory:\n")
        for feature in sorted(metadata.keys()):
            if feature in pools['twitter'].columns:
                f.write(f"  - {feature}_distribution.png\n")

    print(f"\nReport saved to: {report_file}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
