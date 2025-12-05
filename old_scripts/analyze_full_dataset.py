"""
Descriptive Analysis of Full Tweet Dataset

Generate comprehensive statistics and visualizations for all 5000 tweets
including all inferred metadata.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from inference.metadata_inference import infer_tweet_metadata

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_or_create_dataset_with_metadata(output_dir: Path, force_reload: bool = False):
    """Load dataset with metadata, or create if doesn't exist."""

    cache_path = output_dir / 'full_dataset_with_metadata.csv'

    if cache_path.exists() and not force_reload:
        print(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} tweets with {len(df.columns)} features")
        return df

    print("Loading fresh dataset and inferring metadata...")
    print("This may take several minutes for 5000 tweets...")
    print()

    # Load dataset
    tweets_df = load_dataset(
        'twitteraae',
        version='all_aa',
        sample_size=5000
    )

    print(f"Loaded {len(tweets_df)} tweets")
    print(f"Inferring metadata for all tweets...")

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

    # Save cache
    tweets_df.to_csv(cache_path, index=False)
    print(f"Saved dataset to {cache_path}")

    return tweets_df


def generate_descriptive_statistics(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive descriptive statistics."""

    stats = {}

    # Basic info
    stats['basic'] = {
        'total_tweets': len(df),
        'total_features': len(df.columns),
        'date_generated': datetime.now().isoformat()
    }

    # Demographics
    stats['demographics'] = {
        'demo_aa': {
            'mean': float(df['demo_aa'].mean()),
            'std': float(df['demo_aa'].std()),
            'min': float(df['demo_aa'].min()),
            'max': float(df['demo_aa'].max()),
            'median': float(df['demo_aa'].median())
        },
        'demo_white': {
            'mean': float(df['demo_white'].mean()),
            'std': float(df['demo_white'].std()),
            'median': float(df['demo_white'].median())
        },
        'demo_hispanic': {
            'mean': float(df['demo_hispanic'].mean()),
            'std': float(df['demo_hispanic'].std()),
            'median': float(df['demo_hispanic'].median())
        },
        'demo_other': {
            'mean': float(df['demo_other'].mean()),
            'std': float(df['demo_other'].std()),
            'median': float(df['demo_other'].median())
        }
    }

    # Sentiment
    stats['sentiment'] = {
        'polarity_mean': float(df['sentiment_polarity'].mean()),
        'polarity_std': float(df['sentiment_polarity'].std()),
        'polarity_median': float(df['sentiment_polarity'].median()),
        'label_distribution': df['sentiment_label'].value_counts().to_dict()
    }

    # Topics
    stats['topics'] = {
        'distribution': df['primary_topic'].value_counts().to_dict(),
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
        'high_polarization_pct': float((df['controversy_level'] == 'high').mean() * 100)
    }

    # Gender (if available)
    if 'gender_prediction' in df.columns:
        stats['gender'] = {
            'distribution': df['gender_prediction'].value_counts().to_dict(),
            'unknown_pct': float((df['gender_prediction'] == 'unknown').mean() * 100)
        }

    # Political (if available)
    if 'political_leaning' in df.columns:
        stats['political'] = {
            'distribution': df['political_leaning'].value_counts().to_dict(),
            'is_political_pct': float(df['is_political'].mean() * 100)
        }

    # Save statistics
    stats_path = output_dir / 'descriptive_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved descriptive statistics to {stats_path}")

    return stats


def create_comprehensive_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualization plots."""

    viz_dir = output_dir / 'descriptive_plots'
    viz_dir.mkdir(exist_ok=True)

    # 1. Demographics Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    demo_cols = ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']
    demo_labels = ['African American', 'White', 'Hispanic', 'Other']

    for idx, (col, label) in enumerate(zip(demo_cols, demo_labels)):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{label} Demographic Distribution', fontsize=12, fontweight='bold')
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'demographics_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: demographics_distribution.png")
    plt.close()

    # 2. Sentiment Analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Polarity distribution
    axes[0].hist(df['sentiment_polarity'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(df['sentiment_polarity'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {df['sentiment_polarity'].mean():.3f}")
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Sentiment Polarity', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Sentiment Polarity Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sentiment labels
    sentiment_counts = df['sentiment_label'].value_counts()
    axes[1].bar(sentiment_counts.index, sentiment_counts.values, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Sentiment Label', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Sentiment Label Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add percentages
    for i, (label, count) in enumerate(sentiment_counts.items()):
        pct = count / len(df) * 100
        axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    # Subjectivity distribution
    axes[2].hist(df['sentiment_subjectivity'], bins=50, alpha=0.7, edgecolor='black', color='coral')
    axes[2].axvline(df['sentiment_subjectivity'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {df['sentiment_subjectivity'].mean():.3f}")
    axes[2].set_xlabel('Sentiment Subjectivity', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Sentiment Subjectivity Distribution', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'sentiment_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: sentiment_analysis.png")
    plt.close()

    # 3. Topic Distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    topic_counts = df['primary_topic'].value_counts()
    colors = sns.color_palette("husl", len(topic_counts))
    bars = ax.barh(topic_counts.index, topic_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Count', fontsize=11)
    ax.set_ylabel('Topic', fontsize=11)
    ax.set_title('Topic Distribution (All 5000 Tweets)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add percentages
    for i, (topic, count) in enumerate(topic_counts.items()):
        pct = count / len(df) * 100
        ax.text(count, i, f' {pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / 'topic_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: topic_distribution.png")
    plt.close()

    # 4. Writing Style Features
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Boolean features
    bool_features = ['has_emoji', 'has_hashtag', 'has_mention', 'has_url']
    for idx, feature in enumerate(bool_features):
        ax = axes[idx // 3, idx % 3]
        counts = df[feature].value_counts()
        ax.bar(['No', 'Yes'], [counts.get(False, 0), counts.get(True, 0)],
               alpha=0.7, edgecolor='black', color=['coral', 'steelblue'])
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{feature.replace("has_", "Has ").replace("_", " ").title()}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentages
        pct = counts.get(True, 0) / len(df) * 100
        ax.text(1, counts.get(True, 0), f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    # Word count distribution
    ax = axes[1, 1]
    ax.hist(df['word_count'], bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {df['word_count'].mean():.1f}")
    ax.set_xlabel('Word Count', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Formality score distribution
    ax = axes[1, 2]
    ax.hist(df['formality_score'], bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax.axvline(df['formality_score'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {df['formality_score'].mean():.3f}")
    ax.set_xlabel('Formality Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Formality Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'writing_style_features.png', dpi=300, bbox_inches='tight')
    print(f"Saved: writing_style_features.png")
    plt.close()

    # 5. Polarization Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Polarization score distribution
    axes[0].hist(df['polarization_score'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0].axvline(df['polarization_score'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {df['polarization_score'].mean():.3f}")
    axes[0].set_xlabel('Polarization Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Polarization Score Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Controversy level distribution
    controversy_counts = df['controversy_level'].value_counts()
    axes[1].bar(controversy_counts.index, controversy_counts.values,
                alpha=0.7, edgecolor='black', color=['green', 'orange', 'red'])
    axes[1].set_xlabel('Controversy Level', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Controversy Level Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add percentages
    for i, (level, count) in enumerate(controversy_counts.items()):
        pct = count / len(df) * 100
        axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(viz_dir / 'polarization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: polarization_analysis.png")
    plt.close()

    # 6. Gender and Political (if available)
    if 'gender_prediction' in df.columns and 'political_leaning' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gender distribution
        gender_counts = df['gender_prediction'].value_counts()
        axes[0].bar(gender_counts.index, gender_counts.values, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Gender Prediction', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Gender Prediction Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        for i, (gender, count) in enumerate(gender_counts.items()):
            pct = count / len(df) * 100
            axes[0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

        # Political leaning distribution
        political_counts = df['political_leaning'].value_counts()
        axes[1].bar(political_counts.index, political_counts.values, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Political Leaning', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Political Leaning Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, (leaning, count) in enumerate(political_counts.items()):
            pct = count / len(df) * 100
            axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(viz_dir / 'gender_political_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: gender_political_analysis.png")
        plt.close()

    # 7. Comprehensive Overview Dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Demographics summary
    ax1 = fig.add_subplot(gs[0, 0])
    demo_means = [df[col].mean() for col in demo_cols]
    ax1.bar(demo_labels, demo_means, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Probability', fontsize=10)
    ax1.set_title('Demographics', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Sentiment summary
    ax2 = fig.add_subplot(gs[0, 1])
    sentiment_props = df['sentiment_label'].value_counts(normalize=True)
    ax2.bar(sentiment_props.index, sentiment_props.values, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Proportion', fontsize=10)
    ax2.set_title('Sentiment', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Top topics
    ax3 = fig.add_subplot(gs[0, 2])
    top_topics = df['primary_topic'].value_counts().head(5)
    ax3.barh(top_topics.index, top_topics.values, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Count', fontsize=10)
    ax3.set_title('Top 5 Topics', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # Style features
    ax4 = fig.add_subplot(gs[1, :])
    style_props = [df[f].mean() * 100 for f in bool_features]
    style_labels = [f.replace('has_', '').replace('_', ' ').title() for f in bool_features]
    ax4.bar(style_labels, style_props, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Percentage (%)', fontsize=10)
    ax4.set_title('Writing Style Features', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Polarization
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(df['polarization_score'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax5.set_xlabel('Score', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Polarization', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Word count
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['word_count'], bins=30, alpha=0.7, edgecolor='black', color='green')
    ax6.set_xlabel('Word Count', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Tweet Length', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Formality
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(df['formality_score'], bins=30, alpha=0.7, edgecolor='black', color='purple')
    ax7.set_xlabel('Score', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('Formality', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    fig.suptitle('Full Dataset Overview (5000 Tweets)', fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(viz_dir / 'comprehensive_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved: comprehensive_overview.png")
    plt.close()

    print(f"\nAll visualizations saved to {viz_dir}")


def print_summary(stats: dict):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("="*80)

    print(f"\nDataset: {stats['basic']['total_tweets']} tweets with {stats['basic']['total_features']} features")

    print("\n--- DEMOGRAPHICS ---")
    for demo, values in stats['demographics'].items():
        print(f"{demo:15s}: Mean={values['mean']:.3f}, Median={values['median']:.3f}, Std={values['std']:.3f}")

    print("\n--- SENTIMENT ---")
    print(f"Polarity Mean: {stats['sentiment']['polarity_mean']:.3f}")
    print(f"Polarity Median: {stats['sentiment']['polarity_median']:.3f}")
    print("Label Distribution:")
    for label, count in stats['sentiment']['label_distribution'].items():
        pct = count / stats['basic']['total_tweets'] * 100
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")

    print("\n--- TOPICS ---")
    print(f"Most Common: {stats['topics']['most_common']}")
    print("Top 5 Topics:")
    sorted_topics = sorted(stats['topics']['distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
    for topic, count in sorted_topics:
        pct = count / stats['basic']['total_tweets'] * 100
        print(f"  {topic:15s}: {count:4d} ({pct:5.1f}%)")

    print("\n--- WRITING STYLE ---")
    print(f"Has Emoji:    {stats['style']['has_emoji_pct']:.1f}%")
    print(f"Has Hashtag:  {stats['style']['has_hashtag_pct']:.1f}%")
    print(f"Has Mention:  {stats['style']['has_mention_pct']:.1f}%")
    print(f"Has URL:      {stats['style']['has_url_pct']:.1f}%")
    print(f"Avg Words:    {stats['style']['avg_word_count']:.1f}")
    print(f"Avg Formality: {stats['style']['avg_formality']:.3f}")

    print("\n--- POLARIZATION ---")
    print(f"Mean Score: {stats['polarization']['mean_score']:.3f}")
    print(f"High Controversy: {stats['polarization']['high_polarization_pct']:.1f}%")

    if 'gender' in stats:
        print("\n--- GENDER ---")
        for gender, count in stats['gender']['distribution'].items():
            pct = count / stats['basic']['total_tweets'] * 100
            print(f"  {gender:10s}: {count:4d} ({pct:5.1f}%)")

    if 'political' in stats:
        print("\n--- POLITICAL LEANING ---")
        print(f"Is Political: {stats['political']['is_political_pct']:.1f}%")
        for leaning, count in stats['political']['distribution'].items():
            pct = count / stats['basic']['total_tweets'] * 100
            print(f"  {leaning:10s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "="*80)


def main():
    """Run complete descriptive analysis."""

    # Create output directory
    output_dir = Path('./outputs/descriptive_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FULL DATASET DESCRIPTIVE ANALYSIS")
    print("="*80)
    print()

    # Load or create dataset with metadata
    df = load_or_create_dataset_with_metadata(output_dir, force_reload=False)

    print()

    # Generate statistics
    print("Generating descriptive statistics...")
    stats = generate_descriptive_statistics(df, output_dir)

    print()

    # Create visualizations
    print("Creating visualizations...")
    create_comprehensive_visualizations(df, output_dir)

    # Print summary
    print_summary(stats)

    print(f"\n✓ Analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  - descriptive_statistics.json")
    print(f"  - full_dataset_with_metadata.csv")
    print(f"  - descriptive_plots/*.png")


if __name__ == '__main__':
    main()
