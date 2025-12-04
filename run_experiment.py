"""
Unified experiment runner for LLM recommendation bias analysis

Supports:
- Multiple datasets: Twitter, Reddit, Bluesky
- Multiple models: OpenAI, Anthropic, HuggingFace local models
- Multiple prompt styles: general, popular, engaging, informative, controversial, neutral
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from inference.metadata_inference import infer_tweet_metadata
from utils.llm_client import get_llm_client


def prepare_dataset_with_metadata(dataset_name: str, sample_size: int = 5000) -> pd.DataFrame:
    """
    Load dataset and add metadata if not already present.

    Args:
        dataset_name: 'twitter', 'reddit', or 'bluesky'
        sample_size: Number of posts to sample

    Returns:
        DataFrame with metadata
    """
    print('='*80)
    print(f'LOADING {dataset_name.upper()} DATASET')
    print('='*80)
    print()

    # Load dataset
    df = load_dataset(dataset_name, sample_size=sample_size, training_only=True)

    # Rename 'message' column to 'text' if needed for metadata inference
    text_col = 'message' if 'message' in df.columns else 'text'

    print(f'Dataset columns: {list(df.columns)}')
    print(f'Text column: {text_col}')
    print()

    # Check if metadata already exists
    metadata_file = Path(f'./outputs/metadata_cache/{dataset_name}_metadata.pkl')

    if metadata_file.exists():
        print(f'Loading cached metadata from {metadata_file}...')
        df_with_metadata = pd.read_pickle(metadata_file)
        print(f'Loaded cached metadata for {len(df_with_metadata)} posts')
    else:
        print('Inferring metadata (sentiment, topics, style, polarization)...')
        print('This may take a few minutes...')
        print()

        # Add metadata
        df_with_metadata = infer_tweet_metadata(
            df,
            text_column=text_col,
            sentiment_method='vader',
            topic_method='keyword',
            include_gender=False,  # Not reliable from text alone
            include_political=False  # Not reliable from text alone
        )

        # Save cache
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        df_with_metadata.to_pickle(metadata_file)
        print(f'Saved metadata cache to {metadata_file}')

    print()
    print(f'Final dataset: {len(df_with_metadata)} posts with metadata')
    print()

    return df_with_metadata


def create_prompt_by_style(posts_df: pd.DataFrame, k: int, style: str,
                           text_col: str = 'message') -> str:
    """Create recommendation prompt based on style."""

    # Define header based on style
    style_headers = {
        'general': "Recommend posts that would be most interesting to a general audience.",
        'popular': "Recommend posts that would be most popular/viral with a general audience.",
        'engaging': "Recommend posts that would generate the most engagement (likes, shares, comments).",
        'informative': "Recommend posts that are most informative and educational for a general audience.",
        'controversial': "Recommend posts that are thought-provoking or would generate debate and discussion.",
        'neutral': "Rank these posts."
    }

    header = style_headers.get(style, style_headers['general'])

    prompt_parts = [header, "\nPosts to rank:\n"]

    for idx, (i, row) in enumerate(posts_df.iterrows(), 1):
        text = row[text_col]
        if len(text) > 200:
            text = text[:200] + "..."
        prompt_parts.append(f"{idx}. {text}")

    prompt_parts.append(f"\n\nTask: Rank these posts from most to least relevant.")
    prompt_parts.append(f"Return ONLY the top {k} post numbers as a comma-separated list.")
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
                              style: str, text_col: str = 'message') -> pd.DataFrame:
    """Run a single recommendation trial."""

    # Create prompt based on style
    prompt = create_prompt_by_style(pool_df, k, style, text_col)

    # Get LLM response
    response = llm_client.generate(prompt, temperature=0.3)

    # Parse ranking
    ranked_indices = parse_ranking_response(response, len(pool_df), k)

    # Get recommended posts
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

    # Tweet-level metadata
    tweet_level_attrs = [
        'sentiment_label', 'primary_topic', 'has_emoji',
        'polarization_score', 'formality_score', 'sentiment_polarity'
    ]

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

                # Compute differences
                all_values = set(list(pool_dist.keys()) + list(rec_dist.keys()))
                diffs = {}
                for val in all_values:
                    pool_pct = pool_dist.get(val, 0) * 100
                    rec_pct = rec_dist.get(val, 0) * 100
                    diffs[val] = rec_pct - pool_pct

                results[f'diff_{attr}'] = diffs

    return results


def main():
    """Main experiment pipeline."""

    parser = argparse.ArgumentParser(description='Run LLM recommendation bias experiment')
    parser.add_argument('--dataset', type=str, default='twitter',
                       choices=['twitter', 'reddit', 'bluesky'],
                       help='Dataset to use')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'anthropic', 'huggingface'],
                       help='LLM provider')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model name (e.g., gpt-4o-mini, claude-3-5-sonnet-20241022, meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--dataset-size', type=int, default=5000,
                       help='Number of posts to load from dataset')
    parser.add_argument('--pool-size', type=int, default=100,
                       help='Number of posts per recommendation pool')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of recommendations per trial')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of trials per prompt style')
    parser.add_argument('--styles', type=str, nargs='+',
                       default=['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral'],
                       help='Prompt styles to test')

    args = parser.parse_args()

    # Configuration
    DATASET = args.dataset
    PROVIDER = args.provider
    MODEL = args.model
    DATASET_SIZE = args.dataset_size
    POOL_SIZE = args.pool_size
    K = args.k
    N_TRIALS = args.n_trials
    PROMPT_STYLES = args.styles

    # Prepare dataset with metadata
    posts_with_metadata = prepare_dataset_with_metadata(DATASET, DATASET_SIZE)

    # Determine text column
    text_col = 'message' if 'message' in posts_with_metadata.columns else 'text'

    # Initialize LLM client
    print('='*80)
    print(f'INITIALIZING LLM CLIENT')
    print('='*80)
    print()
    print(f'Provider: {PROVIDER}')
    print(f'Model: {MODEL}')
    print()

    llm_client = get_llm_client(provider=PROVIDER, model=MODEL)
    print()

    # Run experiments
    print('='*80)
    print(f'RUNNING PROMPT STYLE COMPARISON')
    print('='*80)
    print()
    print(f'Dataset: {DATASET}')
    print(f'Dataset size: {len(posts_with_metadata)} posts')
    print(f'Pool size: {POOL_SIZE}')
    print(f'Recommendations per trial: {K}')
    print(f'Trials per style: {N_TRIALS}')
    print(f'Prompt styles: {", ".join(PROMPT_STYLES)}')
    print()

    all_results = []

    for style_idx, style in enumerate(PROMPT_STYLES):
        print(f'\n[{style_idx+1}/{len(PROMPT_STYLES)}] Testing prompt style: {style.upper()}')
        print('='*80)

        for trial_id in range(N_TRIALS):
            print(f'  Trial {trial_id + 1}/{N_TRIALS}', end=' ')

            # Sample pool for this trial
            seed = 1000 + trial_id
            pool = posts_with_metadata.sample(n=POOL_SIZE, random_state=seed)

            # Run recommendation
            rec = run_single_recommendation(llm_client, pool, K, style, text_col)

            # Analyze bias
            bias = analyze_bias(pool, rec, style, trial_id)
            all_results.append(bias)

            print('✓')

        print(f'  Completed {N_TRIALS} trials for {style}')

    # Save results
    output_dir = Path(f'./outputs/experiments/{DATASET}_{PROVIDER}_{MODEL.replace("/", "_")}')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'prompt_style_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    # Save experiment config
    config = {
        'dataset': DATASET,
        'provider': PROVIDER,
        'model': MODEL,
        'dataset_size': DATASET_SIZE,
        'pool_size': POOL_SIZE,
        'k': K,
        'n_trials': N_TRIALS,
        'prompt_styles': PROMPT_STYLES
    }

    config_path = output_dir / 'config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    # Print summary
    print()
    print('='*80)
    print('EXPERIMENT SUMMARY')
    print('='*80)
    print()
    print(f'Dataset: {DATASET}')
    print(f'Model: {PROVIDER} / {MODEL}')
    print(f'Prompt styles tested: {len(PROMPT_STYLES)}')
    print(f'  Styles: {", ".join(PROMPT_STYLES)}')
    print(f'Trials per style: {N_TRIALS}')
    print(f'Total trials: {N_TRIALS * len(PROMPT_STYLES)}')
    print(f'Pool size per trial: {POOL_SIZE}')
    print(f'Recommendations per trial: {K}')
    print()
    print(f'Results saved to: {output_dir}')
    print()
    print('Next steps:')
    print(f'  - Run: python analyze_experiment.py --results-dir {output_dir}')
    print('  - Compare bias patterns across different prompt styles')
    print()

    # Print LLM usage stats
    stats = llm_client.get_stats()
    print('LLM Usage Statistics:')
    print(f'  Total API calls: {stats["call_count"]}')
    print(f'  Total tokens: {stats["total_tokens"]:,}')


if __name__ == '__main__':
    main()
