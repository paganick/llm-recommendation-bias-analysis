"""
Clean Descriptive Analysis of Tweet Dataset

Load 10k tweets, ensure no duplicates, and generate correct statistics.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from inference.metadata_inference import infer_tweet_metadata

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_clean_dataset(sample_size=10000, force_reload=False):
    """Load dataset with metadata, ensuring no duplicates."""

    output_dir = Path('./outputs/descriptive_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f'dataset_{sample_size}_with_metadata.csv'

    if cache_path.exists() and not force_reload:
        print(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} tweets with {len(df.columns)} features")

        # Verify no duplicates
        n_duplicates = len(df) - df['tweet_id'].nunique()
        if n_duplicates > 0:
            print(f"WARNING: Found {n_duplicates} duplicates, removing...")
            df = df.drop_duplicates(subset='tweet_id')
            df.to_csv(cache_path, index=False)
            print(f"Cleaned to {len(df)} unique tweets")

        return df

    print(f"Loading fresh dataset of {sample_size} tweets and inferring metadata...")
    print("This may take several minutes...")
    print()

    # Load dataset
    tweets_df = load_dataset(
        'twitteraae',
        version='all_aa',  # Pre-filtered for AA English (1.1M tweets available)
        sample_size=sample_size
    )

    print(f"Loaded {len(tweets_df)} tweets")

    # Check for duplicates in loaded data
    n_duplicates = len(tweets_df) - tweets_df['tweet_id'].nunique()
    if n_duplicates > 0:
        print(f"Removing {n_duplicates} duplicate tweet_ids from loaded data...")
        tweets_df = tweets_df.drop_duplicates(subset='tweet_id')
        print(f"After deduplication: {len(tweets_df)} unique tweets")

    print(f"Inferring metadata for {len(tweets_df)} tweets...")

    # Infer metadata
    tweets_df = infer_tweet_metadata(
        tweets_df,
        text_column='text',
        sentiment_method='vader',
        topic_method='keyword',
        gender_method='keyword',
        political_method='keyword',
        include_gender=True,
        include_political=True
    )

    print(f"Metadata inference complete. Now have {len(tweets_df.columns)} features")

    # Final duplicate check
    n_duplicates = len(tweets_df) - tweets_df['tweet_id'].nunique()
    if n_duplicates > 0:
        print(f"WARNING: Found {n_duplicates} duplicates after metadata inference, removing...")
        tweets_df = tweets_df.drop_duplicates(subset='tweet_id')
        print(f"Final size: {len(tweets_df)} unique tweets")

    # Save cache
    tweets_df.to_csv(cache_path, index=False)
    print(f"Saved dataset to {cache_path}")

    return tweets_df


def generate_statistics_json(df: pd.DataFrame, output_path: Path):
    """Generate and save descriptive statistics."""

    stats = {
        'basic': {
            'total_tweets': int(len(df)),
            'unique_tweets': int(df['tweet_id'].nunique()),
            'total_features': int(len(df.columns)),
            'date_generated': datetime.now().isoformat()
        },
        'demographics': {},
        'sentiment': {},
        'topics': {},
        'style': {},
        'polarization': {},
        'gender': {},
        'political': {}
    }

    # Demographics
    for col in ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']:
        stats['demographics'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }

    # Sentiment
    stats['sentiment'] = {
        'polarity_mean': float(df['sentiment_polarity'].mean()),
        'polarity_std': float(df['sentiment_polarity'].std()),
        'polarity_median': float(df['sentiment_polarity'].median()),
        'label_distribution': df['sentiment_label'].value_counts().to_dict(),
        'label_percentages': {
            k: float(v / len(df) * 100)
            for k, v in df['sentiment_label'].value_counts().items()
        }
    }

    # Topics
    topic_counts = df['primary_topic'].value_counts()
    stats['topics'] = {
        'distribution': topic_counts.to_dict(),
        'percentages': {k: float(v / len(df) * 100) for k, v in topic_counts.items()},
        'most_common': df['primary_topic'].mode()[0] if len(df['primary_topic'].mode()) > 0 else 'unknown'
    }

    # Style
    stats['style'] = {
        'has_emoji_pct': float(df['has_emoji'].mean() * 100),
        'has_hashtag_pct': float(df['has_hashtag'].mean() * 100),
        'has_mention_pct': float(df['has_mention'].mean() * 100),
        'has_url_pct': float(df['has_url'].mean() * 100),
        'avg_word_count': float(df['word_count'].mean()),
        'avg_formality': float(df['formality_score'].mean())
    }

    # Polarization
    stats['polarization'] = {
        'mean_score': float(df['polarization_score'].mean()),
        'std_score': float(df['polarization_score'].std()),
        'controversy_distribution': df['controversy_level'].value_counts().to_dict(),
        'controversy_percentages': {
            k: float(v / len(df) * 100)
            for k, v in df['controversy_level'].value_counts().items()
        }
    }

    # Gender
    gender_counts = df['gender_prediction'].value_counts()
    stats['gender'] = {
        'distribution': gender_counts.to_dict(),
        'percentages': {k: float(v / len(df) * 100) for k, v in gender_counts.items()},
        'total_count': int(gender_counts.sum())
    }

    # Political
    political_counts = df['political_leaning'].value_counts()
    stats['political'] = {
        'distribution': political_counts.to_dict(),
        'percentages': {k: float(v / len(df) * 100) for k, v in political_counts.items()},
        'is_political_pct': float(df['is_political'].mean() * 100),
        'total_count': int(political_counts.sum())
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def print_summary(stats: dict):
    """Print formatted summary."""

    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("="*80)

    print(f"\nDataset: {stats['basic']['total_tweets']} tweets ({stats['basic']['unique_tweets']} unique)")
    print(f"Features: {stats['basic']['total_features']}")

    print("\n--- DEMOGRAPHICS (from original dataset) ---")
    for demo, vals in stats['demographics'].items():
        print(f"{demo:15s}: Mean={vals['mean']:.3f}, Median={vals['median']:.3f}, Std={vals['std']:.3f}")

    print("\n--- SENTIMENT ---")
    print(f"Polarity: Mean={stats['sentiment']['polarity_mean']:.3f}, Median={stats['sentiment']['polarity_median']:.3f}")
    print("Labels:")
    for label, pct in stats['sentiment']['label_percentages'].items():
        count = stats['sentiment']['label_distribution'][label]
        print(f"  {label:10s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- TOPICS ---")
    print(f"Most Common: {stats['topics']['most_common']}")
    sorted_topics = sorted(stats['topics']['percentages'].items(), key=lambda x: x[1], reverse=True)[:5]
    for topic, pct in sorted_topics:
        count = stats['topics']['distribution'][topic]
        print(f"  {topic:15s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- WRITING STYLE ---")
    for key in ['has_emoji_pct', 'has_hashtag_pct', 'has_mention_pct', 'has_url_pct']:
        print(f"{key.replace('_pct', ''):20s}: {stats['style'][key]:5.1f}%")
    print(f"{'Avg word count':20s}: {stats['style']['avg_word_count']:5.1f}")
    print(f"{'Avg formality':20s}: {stats['style']['avg_formality']:5.3f}")

    print("\n--- POLARIZATION ---")
    print(f"Mean Score: {stats['polarization']['mean_score']:.3f}")
    for level, pct in stats['polarization']['controversy_percentages'].items():
        count = stats['polarization']['controversy_distribution'][level]
        print(f"  {level:10s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- GENDER (inferred from text) ---")
    print(f"Total analyzed: {stats['gender']['total_count']}")
    for gender, pct in stats['gender']['percentages'].items():
        count = stats['gender']['distribution'][gender]
        print(f"  {gender:10s}: {count:5d} ({pct:5.1f}%)")
    total_pct = sum(stats['gender']['percentages'].values())
    print(f"  {'TOTAL':10s}: {stats['gender']['total_count']:5d} ({total_pct:5.1f}%)")

    print("\n--- POLITICAL LEANING (inferred from text) ---")
    print(f"Total analyzed: {stats['political']['total_count']}")
    print(f"Is political: {stats['political']['is_political_pct']:.1f}%")
    for leaning, pct in stats['political']['percentages'].items():
        count = stats['political']['distribution'][leaning]
        print(f"  {leaning:10s}: {count:5d} ({pct:5.1f}%)")
    total_pct = sum(stats['political']['percentages'].values())
    print(f"  {'TOTAL':10s}: {stats['political']['total_count']:5d} ({total_pct:5.1f}%)")

    print("\n" + "="*80)


def main():
    """Run clean descriptive analysis."""

    print("="*80)
    print("CLEAN DATASET ANALYSIS (10K TWEETS)")
    print("="*80)
    print()

    # Load clean dataset
    df = load_clean_dataset(sample_size=10000, force_reload=False)

    print()
    print(f"Final dataset: {len(df)} tweets, {len(df.columns)} features")
    print(f"Unique tweet_ids: {df['tweet_id'].nunique()}")
    print()

    # Generate statistics
    output_dir = Path('./outputs/descriptive_analysis')
    stats_path = output_dir / 'descriptive_statistics_clean.json'

    print("Generating statistics...")
    stats = generate_statistics_json(df, stats_path)
    print(f"Saved to {stats_path}")

    # Print summary
    print_summary(stats)

    print(f"\n✓ Analysis complete!")
    print(f"  Dataset: {output_dir / 'dataset_10000_with_metadata.csv'}")
    print(f"  Statistics: {stats_path}")


if __name__ == '__main__':
    main()
