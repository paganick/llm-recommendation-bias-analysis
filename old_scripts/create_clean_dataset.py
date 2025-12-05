"""
Create clean 10k dataset with proper metadata - debugging version
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from inference.metadata_inference import infer_tweet_metadata

def main():
    output_dir = Path('./outputs/descriptive_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base dataset
    print("Step 1: Loading base dataset...")
    tweets_df = load_dataset('twitteraae', version='all_aa', sample_size=10000)
    print(f"  Loaded: {len(tweets_df)} tweets")
    print(f"  Unique IDs: {tweets_df['tweet_id'].nunique()}")
    print(f"  Columns: {len(tweets_df.columns)}")

    # Step 2: Check for duplicates
    print("\nStep 2: Checking for duplicates...")
    dups_before = len(tweets_df) - tweets_df['tweet_id'].nunique()
    if dups_before > 0:
        print(f"  WARNING: Found {dups_before} duplicates, removing...")
        tweets_df = tweets_df.drop_duplicates(subset='tweet_id')
        print(f"  After dedup: {len(tweets_df)} tweets")
    else:
        print(f"  No duplicates found")

    # Step 3: Infer metadata
    print("\nStep 3: Inferring metadata...")
    print(f"  Starting with {len(tweets_df)} tweets")

    tweets_with_meta = infer_tweet_metadata(
        tweets_df,
        text_column='text',
        sentiment_method='vader',
        topic_method='keyword',
        gender_method='keyword',
        political_method='keyword',
        include_gender=True,
        include_political=True
    )

    print(f"  After metadata: {len(tweets_with_meta)} tweets")
    print(f"  Unique IDs: {tweets_with_meta['tweet_id'].nunique()}")
    print(f"  Columns: {len(tweets_with_meta.columns)}")

    # Step 4: Final duplicate check
    print("\nStep 4: Final duplicate check...")
    dups_after = len(tweets_with_meta) - tweets_with_meta['tweet_id'].nunique()
    if dups_after > 0:
        print(f"  WARNING: Found {dups_after} duplicates after metadata inference!")
        print(f"  Removing duplicates...")
        tweets_with_meta = tweets_with_meta.drop_duplicates(subset='tweet_id', keep='first')
        print(f"  After final dedup: {len(tweets_with_meta)} tweets")
    else:
        print(f"  No duplicates found")

    # Step 5: Verify data quality
    print("\nStep 5: Verifying data quality...")
    print(f"  Final size: {len(tweets_with_meta)} tweets")
    print(f"  Final unique IDs: {tweets_with_meta['tweet_id'].nunique()}")
    print(f"  Total columns: {len(tweets_with_meta.columns)}")

    # Check metadata columns
    meta_cols = ['sentiment_polarity', 'primary_topic', 'gender_prediction', 'has_emoji']
    print("\n  Metadata completeness:")
    for col in meta_cols:
        non_null = tweets_with_meta[col].notna().sum()
        pct = non_null / len(tweets_with_meta) * 100
        print(f"    {col:25s}: {non_null:5d} ({pct:5.1f}%)")

    # Step 6: Save
    output_path = output_dir / 'clean_dataset_10k.csv'
    print(f"\nStep 6: Saving to {output_path}...")
    tweets_with_meta.to_csv(output_path, index=False)
    print(f"  Saved!")

    # Step 7: Quick stats
    print("\nStep 7: Quick statistics:")
    print(f"  Demo AA mean: {tweets_with_meta['demo_aa'].mean():.3f}")
    print(f"  Sentiment labels:")
    sent_counts = tweets_with_meta['sentiment_label'].value_counts()
    for label, count in sent_counts.items():
        pct = count / len(tweets_with_meta) * 100
        print(f"    {label:10s}: {count:5d} ({pct:5.1f}%)")

    print(f"  Gender distribution:")
    gender_counts = tweets_with_meta['gender_prediction'].value_counts()
    for gender, count in gender_counts.items():
        pct = count / len(tweets_with_meta) * 100
        print(f"    {gender:10s}: {count:5d} ({pct:5.1f}%)")

    print("\n✓ Done!")

if __name__ == '__main__':
    main()
