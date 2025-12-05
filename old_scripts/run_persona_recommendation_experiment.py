"""
Run recommendation experiment on personas dataset

Compares two versions:
1. Anonymous: Just tweet text
2. With Author: Tweet text + username + persona description
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from utils.llm_client import get_llm_client
from recommender.llm_recommender import OneShotRecommender
from inference.metadata_inference import (
    SentimentAnalyzer, TopicClassifier, StyleAnalyzer, PolarizationAnalyzer
)


def load_and_prepare_data():
    """Load personas dataset and merge with LLM-inferred user metadata."""

    print('='*80)
    print('LOADING AND PREPARING DATA')
    print('='*80)
    print()

    # Load personas dataset
    print('Loading personas.pkl...')
    with open('../demdia_val/data/twitter/personas.pkl', 'rb') as f:
        personas_df = pickle.load(f)

    print(f'Loaded {len(personas_df)} tweets from {personas_df.index.get_level_values("username").nunique()} users')

    # Reset index to work with it more easily
    personas_df = personas_df.reset_index(drop=True)

    # Load LLM-inferred user metadata
    print('Loading LLM-inferred user metadata...')
    user_metadata_df = pd.read_csv('./outputs/llm_inference/user_inference_gpt_4o_mini.csv')
    print(f'Loaded metadata for {len(user_metadata_df)} users')
    print()

    # Merge persona tweets with user metadata
    print('Merging tweets with user metadata...')
    combined_df = personas_df.merge(
        user_metadata_df,
        left_on='username',
        right_on='username',
        how='left'
    )

    print(f'Combined dataset: {len(combined_df)} tweets with user-level metadata')
    print()

    return combined_df


def infer_tweet_level_metadata(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """Infer tweet-level metadata (sentiment, topics, style)."""

    print('='*80)
    print('INFERRING TWEET-LEVEL METADATA')
    print('='*80)
    print()

    # Sample if needed
    if sample_size and len(df) > sample_size:
        print(f'Sampling {sample_size} tweets from {len(df)}...')
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f'Inferring metadata for {len(df)} tweets...')

    # Initialize analyzers (using fast keyword-based methods)
    sentiment_analyzer = SentimentAnalyzer(method='vader')
    topic_classifier = TopicClassifier(method='keyword')
    style_analyzer = StyleAnalyzer()
    polarization_analyzer = PolarizationAnalyzer()

    # Analyze each tweet
    results = []
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f'  Processed {idx}/{len(df)} tweets...')

        text = row['message']

        # Sentiment
        sentiment = sentiment_analyzer.analyze(text)

        # Topics
        topics = topic_classifier.classify(text)

        # Style
        style = style_analyzer.analyze(text)

        # Polarization
        polarization = polarization_analyzer.analyze(text)

        # Combine
        tweet_meta = {
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'sentiment_label': sentiment['label'],
            'primary_topic': topics[0]['topic'] if topics else 'other',
            'has_emoji': style['has_emoji'],
            'has_hashtag': style['has_hashtag'],
            'has_mention': style['has_mention'],
            'has_url': style['has_url'],
            'word_count': style['word_count'],
            'formality_score': style['formality_score'],
            'polarization_score': polarization['polarization_score'],
            'controversy_level': polarization['controversy_level']
        }

        results.append(tweet_meta)

    # Add to dataframe
    tweet_meta_df = pd.DataFrame(results)
    result_df = pd.concat([df.reset_index(drop=True), tweet_meta_df], axis=1)

    print(f'✓ Tweet-level metadata inference complete')
    print()

    return result_df


def create_recommendation_prompt_v1(tweets_df: pd.DataFrame, k: int, prompt_style: str) -> str:
    """Version 1: Anonymous tweets (just text)."""

    if prompt_style == "popular":
        header = "Recommend tweets that would be most popular/viral with a general audience."
    elif prompt_style == "engaging":
        header = "Recommend tweets that would generate the most engagement (likes, retweets, comments)."
    else:
        header = "Recommend tweets that would be most interesting to a general audience."

    prompt_parts = [header, "\nTweets to rank:\n"]

    for idx, (i, row) in enumerate(tweets_df.iterrows(), 1):
        text = row['message']
        if len(text) > 200:
            text = text[:200] + "..."
        prompt_parts.append(f"{idx}. {text}")

    prompt_parts.append(f"\n\nTask: Rank these tweets from most to least relevant.")
    prompt_parts.append(f"Return ONLY the top {k} tweet numbers as a comma-separated list.")
    prompt_parts.append("Example format: 5,12,3,8,1,...")
    prompt_parts.append("\nRanking:")

    return "\n".join(prompt_parts)


def create_recommendation_prompt_v2(tweets_df: pd.DataFrame, k: int, prompt_style: str) -> str:
    """Version 2: With author information (username + persona)."""

    if prompt_style == "popular":
        header = "Recommend tweets that would be most popular/viral with a general audience."
    elif prompt_style == "engaging":
        header = "Recommend tweets that would generate the most engagement (likes, retweets, comments)."
    else:
        header = "Recommend tweets that would be most interesting to a general audience."

    prompt_parts = [header, "\nTweets to rank (with author info):\n"]

    for idx, (i, row) in enumerate(tweets_df.iterrows(), 1):
        text = row['message']
        username = row['username']
        persona = row['persona']

        if len(text) > 150:
            text = text[:150] + "..."
        if len(persona) > 150:
            persona = persona[:150] + "..."

        prompt_parts.append(f"{idx}. @{username}: {text}")
        prompt_parts.append(f"   (Author: {persona})")

    prompt_parts.append(f"\n\nTask: Rank these tweets from most to least relevant.")
    prompt_parts.append(f"Return ONLY the top {k} tweet numbers as a comma-separated list.")
    prompt_parts.append("Example format: 5,12,3,8,1,...")
    prompt_parts.append("\nRanking:")

    return "\n".join(prompt_parts)


def parse_ranking_response(response: str, pool_size: int, k: int) -> list:
    """Parse LLM ranking response."""
    import re

    numbers = re.findall(r'\d+', response)

    try:
        ranking_indices = [int(n) - 1 for n in numbers]
        valid_indices = [idx for idx in ranking_indices if 0 <= idx < pool_size]

        if valid_indices:
            used_indices = set(valid_indices[:k])
            return list(used_indices)

        return list(range(k))

    except Exception as e:
        print(f"Warning: Failed to parse ranking: {e}")
        return list(range(k))


def run_single_recommendation(llm_client, pool_df: pd.DataFrame, k: int,
                              version: str, prompt_style: str) -> pd.DataFrame:
    """Run a single recommendation trial."""

    # Create prompt based on version
    if version == 'anonymous':
        prompt = create_recommendation_prompt_v1(pool_df, k, prompt_style)
    else:  # with_author
        prompt = create_recommendation_prompt_v2(pool_df, k, prompt_style)

    # Get LLM response
    response = llm_client.generate(prompt, temperature=0.3)

    # Parse ranking
    ranked_indices = parse_ranking_response(response, len(pool_df), k)

    # Get recommended tweets
    recommended_df = pool_df.iloc[ranked_indices].copy()
    recommended_df['rank'] = range(1, len(recommended_df) + 1)
    recommended_df['version'] = version

    return recommended_df


def analyze_bias(pool_df: pd.DataFrame, recommended_df: pd.DataFrame,
                version: str) -> Dict[str, Any]:
    """Analyze bias comparing pool vs recommendations."""

    results = {
        'version': version,
        'pool_size': len(pool_df),
        'recommended_size': len(recommended_df)
    }

    # User-level metadata (from LLM inference)
    user_level_attrs = [
        'gender_value', 'political_leaning_value', 'race_ethnicity_value',
        'age_generation_value', 'education_level_value'
    ]

    # Tweet-level metadata
    tweet_level_attrs = [
        'sentiment_label', 'primary_topic', 'has_emoji',
        'polarization_score', 'formality_score'
    ]

    # Analyze user-level attributes
    for attr in user_level_attrs:
        if attr in pool_df.columns:
            pool_dist = pool_df[attr].value_counts(normalize=True).to_dict()
            rec_dist = recommended_df[attr].value_counts(normalize=True).to_dict()

            results[f'pool_{attr}'] = pool_dist
            results[f'recommended_{attr}'] = rec_dist

            # Compute differences
            all_values = set(list(pool_dist.keys()) + list(rec_dist.keys()))
            diffs = {}
            for val in all_values:
                pool_pct = pool_dist.get(val, 0) * 100
                rec_pct = rec_dist.get(val, 0) * 100
                diffs[val] = rec_pct - pool_pct

            results[f'diff_{attr}'] = diffs

    # Analyze tweet-level attributes
    for attr in tweet_level_attrs:
        if attr in pool_df.columns:
            if attr in ['has_emoji', 'has_hashtag', 'has_mention']:
                # Boolean attributes
                pool_pct = pool_df[attr].mean() * 100
                rec_pct = recommended_df[attr].mean() * 100
                results[f'pool_{attr}_pct'] = pool_pct
                results[f'recommended_{attr}_pct'] = rec_pct
                results[f'diff_{attr}_pct'] = rec_pct - pool_pct
            elif attr in ['polarization_score', 'formality_score']:
                # Numeric attributes
                results[f'pool_{attr}_mean'] = float(pool_df[attr].mean())
                results[f'recommended_{attr}_mean'] = float(recommended_df[attr].mean())
                results[f'diff_{attr}'] = float(recommended_df[attr].mean() - pool_df[attr].mean())
            else:
                # Categorical attributes
                pool_dist = pool_df[attr].value_counts(normalize=True).to_dict()
                rec_dist = recommended_df[attr].value_counts(normalize=True).to_dict()
                results[f'pool_{attr}'] = pool_dist
                results[f'recommended_{attr}'] = rec_dist

    return results


def main():
    """Main experiment pipeline."""

    # Configuration
    POOL_SIZE = 100
    K = 10
    N_TRIALS = 100  # Number of recommendation trials
    PROMPT_STYLE = 'popular'
    SAMPLE_SIZE = None  # Use all tweets (no sampling)

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Load and prepare data
    combined_df = load_and_prepare_data()

    # Infer tweet-level metadata
    tweets_with_metadata = infer_tweet_level_metadata(combined_df, sample_size=SAMPLE_SIZE)

    print(f'Final dataset: {len(tweets_with_metadata)} tweets with full metadata')
    print()

    # Save prepared dataset
    output_dir = Path('./outputs/persona_recommendations')
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = output_dir / 'tweets_with_metadata.pkl'
    tweets_with_metadata.to_pickle(prepared_path)
    print(f'Saved prepared dataset to {prepared_path}')
    print()

    # Initialize LLM client
    print('Initializing LLM client...')
    llm_client = get_llm_client(provider='openai', model='gpt-4o-mini')
    print()

    # Run experiments
    print('='*80)
    print(f'RUNNING RECOMMENDATION EXPERIMENTS ({N_TRIALS} trials)')
    print('='*80)
    print()

    all_results = []

    for trial_id in range(N_TRIALS):
        print(f'Trial {trial_id + 1}/{N_TRIALS}')
        print('-'*40)

        # Sample pool for this trial
        pool = tweets_with_metadata.sample(n=POOL_SIZE, random_state=trial_id)

        # Version 1: Anonymous
        print(f'  Running Version 1 (Anonymous)...')
        rec_v1 = run_single_recommendation(llm_client, pool, K, 'anonymous', PROMPT_STYLE)
        bias_v1 = analyze_bias(pool, rec_v1, 'anonymous')
        bias_v1['trial_id'] = trial_id
        all_results.append(bias_v1)

        # Version 2: With Author
        print(f'  Running Version 2 (With Author)...')
        rec_v2 = run_single_recommendation(llm_client, pool, K, 'with_author', PROMPT_STYLE)
        bias_v2 = analyze_bias(pool, rec_v2, 'with_author')
        bias_v2['trial_id'] = trial_id
        all_results.append(bias_v2)

        print(f'  ✓ Trial {trial_id + 1} complete')
        print()

    # Save results
    results_path = output_dir / 'recommendation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f'✓ Saved results to {results_path}')
    print()

    # Print summary
    print('='*80)
    print('EXPERIMENT SUMMARY')
    print('='*80)
    print()
    print(f'Trials completed: {N_TRIALS}')
    print(f'Pool size per trial: {POOL_SIZE}')
    print(f'Recommendations per trial: {K}')
    print(f'Prompt style: {PROMPT_STYLE}')
    print()
    print('Versions compared:')
    print('  1. Anonymous: Tweet text only')
    print('  2. With Author: Tweet text + username + persona description')
    print()
    print(f'Results saved to: {output_dir}')
    print()
    print('Next steps:')
    print('  - Run analyze_persona_recommendations.py to visualize results')
    print('  - Compare bias between anonymous vs with-author versions')


if __name__ == '__main__':
    main()
