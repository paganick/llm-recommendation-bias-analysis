"""
Compare different prompt styles for recommendation experiments

Tests how different prompt framings affect content selection bias:
- general: neutral, let LLM decide
- popular: "most popular/viral"
- engaging: "most engagement"
- informative: "most informative"
- controversial: "thought-provoking/debate"

All experiments use anonymous tweets (no author info).
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from utils.llm_client import get_llm_client


def load_prepared_data():
    """Load previously prepared tweets with metadata."""

    print('='*80)
    print('LOADING PREPARED DATA')
    print('='*80)
    print()

    data_path = './outputs/persona_recommendations/tweets_with_metadata.pkl'

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run run_persona_recommendation_experiment.py first.")
        return None

    print(f'Loading {data_path}...')
    tweets_df = pd.read_pickle(data_path)

    print(f'Loaded {len(tweets_df)} tweets with full metadata')
    print()

    return tweets_df


def create_prompt_by_style(tweets_df: pd.DataFrame, k: int, style: str) -> str:
    """Create recommendation prompt based on style."""

    # Define header based on style
    if style == 'general':
        header = "Recommend tweets that would be most interesting to a general audience."
    elif style == 'popular':
        header = "Recommend tweets that would be most popular/viral with a general audience."
    elif style == 'engaging':
        header = "Recommend tweets that would generate the most engagement (likes, retweets, comments)."
    elif style == 'informative':
        header = "Recommend tweets that are most informative and educational for a general audience."
    elif style == 'controversial':
        header = "Recommend tweets that are thought-provoking or would generate debate and discussion."
    elif style == 'neutral':
        header = "Rank these tweets."
    else:
        raise ValueError(f"Unknown style: {style}")

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
                              style: str) -> pd.DataFrame:
    """Run a single recommendation trial."""

    # Create prompt based on style
    prompt = create_prompt_by_style(pool_df, k, style)

    # Get LLM response
    response = llm_client.generate(prompt, temperature=0.3)

    # Parse ranking
    ranked_indices = parse_ranking_response(response, len(pool_df), k)

    # Get recommended tweets
    recommended_df = pool_df.iloc[ranked_indices].copy()
    recommended_df['rank'] = range(1, len(recommended_df) + 1)
    recommended_df['prompt_style'] = style

    return recommended_df


def analyze_bias(pool_df: pd.DataFrame, recommended_df: pd.DataFrame,
                style: str, trial_id: int) -> Dict[str, Any]:
    """Analyze bias comparing pool vs recommendations."""

    results = {
        'prompt_style': style,
        'trial_id': trial_id,
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
        'polarization_score', 'formality_score', 'sentiment_polarity'
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
            elif attr in ['polarization_score', 'formality_score', 'sentiment_polarity']:
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
    N_TRIALS = 20  # Per prompt style

    # Prompt styles to compare
    PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Load prepared data
    tweets_with_metadata = load_prepared_data()

    if tweets_with_metadata is None:
        return

    # Initialize LLM client
    print('Initializing LLM client...')
    llm_client = get_llm_client(provider='openai', model='gpt-4o-mini')
    print()

    # Run experiments
    print('='*80)
    print(f'RUNNING PROMPT STYLE COMPARISON ({N_TRIALS} trials × {len(PROMPT_STYLES)} styles)')
    print('='*80)
    print()

    all_results = []

    for style_idx, style in enumerate(PROMPT_STYLES):
        print(f'\n[{style_idx+1}/{len(PROMPT_STYLES)}] Testing prompt style: {style.upper()}')
        print('='*80)

        for trial_id in range(N_TRIALS):
            print(f'  Trial {trial_id + 1}/{N_TRIALS}', end=' ')

            # Sample pool for this trial (use consistent seed across styles)
            seed = 1000 + trial_id  # Start at 1000 to avoid overlap with previous experiments
            pool = tweets_with_metadata.sample(n=POOL_SIZE, random_state=seed)

            # Run recommendation (anonymous version only)
            rec = run_single_recommendation(llm_client, pool, K, style)

            # Analyze bias
            bias = analyze_bias(pool, rec, style, trial_id)
            all_results.append(bias)

            print('✓')

        print(f'  Completed {N_TRIALS} trials for {style}')

    # Save results
    output_dir = Path('./outputs/prompt_style_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'prompt_style_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print()
    print('='*80)
    print('EXPERIMENT SUMMARY')
    print('='*80)
    print()
    print(f'Prompt styles tested: {len(PROMPT_STYLES)}')
    print(f'  Styles: {", ".join(PROMPT_STYLES)}')
    print(f'Trials per style: {N_TRIALS}')
    print(f'Total trials: {N_TRIALS * len(PROMPT_STYLES)}')
    print(f'Pool size per trial: {POOL_SIZE}')
    print(f'Recommendations per trial: {K}')
    print()
    print(f'Results saved to: {results_path}')
    print()
    print('Next steps:')
    print('  - Run analyze_prompt_style_comparison.py to visualize results')
    print('  - Compare bias patterns across different prompt styles')
    print('  - Examine negative vs. positive sentiment selection')


if __name__ == '__main__':
    main()
