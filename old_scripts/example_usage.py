"""
Example Usage: LLM Recommendation Bias Analysis

This script demonstrates how to:
1. Load the TwitterAAE dataset
2. Get LLM recommendations
3. Analyze demographic bias in recommendations
"""

import pandas as pd
import os

# Load modules
from data.loaders import TwitterAAELoader
from utils.llm_client import get_llm_client
from recommender.llm_recommender import OneShotRecommender


def main():
    """Run a simple bias analysis example."""

    # 1. Load TwitterAAE dataset
    print("="*60)
    print("STEP 1: Loading TwitterAAE Dataset")
    print("="*60)

    loader = TwitterAAELoader('/data/nicpag/AI_recsys_project/TwitterAAE-full-v1.zip')

    # Load the full version with tweet text (sample 1000 tweets)
    tweets_df = loader.load(version='all_aa', sample_size=1000)

    print(f"\nLoaded {len(tweets_df)} tweets")
    print(f"Columns: {tweets_df.columns.tolist()}")
    print(f"\nSample tweet:")
    print(tweets_df.iloc[0]['text'])
    print(f"Demographic probabilities: AA={tweets_df.iloc[0]['demo_aa']:.3f}, "
          f"White={tweets_df.iloc[0]['demo_white']:.3f}")

    # 2. Initialize LLM client
    print("\n" + "="*60)
    print("STEP 2: Initialize LLM Client")
    print("="*60)

    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("\nWARNING: ANTHROPIC_API_KEY not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("\nSkipping LLM recommendations...")
        return

    try:
        llm = get_llm_client(
            provider='anthropic',
            model='claude-3-5-sonnet-20241022',
            temperature=0.7
        )
        print("LLM client initialized successfully")
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        return

    # 3. Get recommendations
    print("\n" + "="*60)
    print("STEP 3: Get LLM Recommendations")
    print("="*60)

    recommender = OneShotRecommender(llm, k=10)

    # Sample a pool of 50 tweets
    tweet_pool = tweets_df.sample(n=50, random_state=42)

    print(f"\nGetting recommendations from pool of {len(tweet_pool)} tweets...")

    try:
        recommended = recommender.recommend(
            tweet_pool,
            prompt_style='popular',
            max_pool_size=50
        )

        print(f"\nTop {len(recommended)} recommended tweets:")
        for _, row in recommended.iterrows():
            print(f"\n[Rank {row['rank']}]")
            print(f"  Text: {row['text'][:100]}...")
            print(f"  Demographics: AA={row['demo_aa']:.3f}, White={row['demo_white']:.3f}")

    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return

    # 4. Analyze bias
    print("\n" + "="*60)
    print("STEP 4: Analyze Demographic Bias")
    print("="*60)

    # Compare demographics in recommendations vs. pool
    pool_aa_mean = tweet_pool['demo_aa'].mean()
    pool_white_mean = tweet_pool['demo_white'].mean()

    rec_aa_mean = recommended['demo_aa'].mean()
    rec_white_mean = recommended['demo_white'].mean()

    print(f"\nDemographic Analysis:")
    print(f"  Pool (baseline):")
    print(f"    Mean P(AA): {pool_aa_mean:.3f}")
    print(f"    Mean P(White): {pool_white_mean:.3f}")
    print(f"\n  Recommended tweets:")
    print(f"    Mean P(AA): {rec_aa_mean:.3f}")
    print(f"    Mean P(White): {rec_white_mean:.3f}")
    print(f"\n  Difference:")
    print(f"    P(AA) difference: {rec_aa_mean - pool_aa_mean:+.3f}")
    print(f"    P(White) difference: {rec_white_mean - pool_white_mean:+.3f}")

    if rec_aa_mean < pool_aa_mean:
        print("\n  ⚠ Potential bias: LLM may be under-recommending AA content")
    elif rec_white_mean < pool_white_mean:
        print("\n  ⚠ Potential bias: LLM may be under-recommending White content")
    else:
        print("\n  ✓ No strong demographic bias detected in this sample")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == '__main__':
    main()
