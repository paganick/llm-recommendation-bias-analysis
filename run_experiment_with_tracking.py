"""
Enhanced experiment runner with post-level tracking for regression analysis

This version saves:
1. All results (as before)
2. Post-level data: which posts were in each pool and which were selected
3. This enables regression analysis to understand confounding factors
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


# Import functions from run_experiment.py
from run_experiment import (
    prepare_dataset_with_metadata,
    create_prompt_by_style,
    parse_ranking_response,
    analyze_bias
)


def run_single_recommendation_with_tracking(llm_client, pool_df: pd.DataFrame,
                                           k: int, style: str,
                                           text_col: str = 'message') -> tuple:
    """
    Run a single recommendation trial and return both results and post-level data.

    Returns:
        - recommended_df: DataFrame of recommended posts
        - post_level_data: List of dicts with selection status for each post in pool
    """

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

    # Create post-level data (FOR REGRESSION ANALYSIS)
    post_level_data = []
    for idx, (original_idx, row) in enumerate(pool_df.iterrows()):
        post_data = {
            'pool_position': idx,
            'original_index': original_idx,  # Index in full dataset
            'selected': 1 if idx in ranked_indices else 0,
            'prompt_style': style
        }

        # Add all metadata
        for col in pool_df.columns:
            if col not in ['rank', 'prompt_style']:
                post_data[col] = row[col]

        post_level_data.append(post_data)

    return recommended_df, post_level_data


def main():
    """Main experiment pipeline with tracking."""

    parser = argparse.ArgumentParser(description='Run LLM recommendation bias experiment with tracking')
    parser.add_argument('--dataset', type=str, default='twitter',
                       choices=['twitter', 'reddit', 'bluesky'],
                       help='Dataset to use')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'anthropic', 'huggingface'],
                       help='LLM provider')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model name')
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
    print(f'RUNNING PROMPT STYLE COMPARISON WITH TRACKING')
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
    all_post_level_data = []  # NEW: Store post-level data

    for style_idx, style in enumerate(PROMPT_STYLES):
        print(f'\n[{style_idx+1}/{len(PROMPT_STYLES)}] Testing prompt style: {style.upper()}')
        print('='*80)

        for trial_id in range(N_TRIALS):
            print(f'  Trial {trial_id + 1}/{N_TRIALS}', end=' ')

            # Sample pool for this trial
            seed = 1000 + trial_id
            pool = posts_with_metadata.sample(n=POOL_SIZE, random_state=seed)

            # Run recommendation WITH TRACKING
            rec, post_data = run_single_recommendation_with_tracking(
                llm_client, pool, K, style, text_col
            )

            # Add trial ID to each post
            for post in post_data:
                post['trial_id'] = trial_id

            all_post_level_data.extend(post_data)

            # Analyze bias (aggregate level)
            bias = analyze_bias(pool, rec, style, trial_id)
            all_results.append(bias)

            print('✓')

        print(f'  Completed {N_TRIALS} trials for {style}')

    # Save results
    output_dir = Path(f'./outputs/experiments/{DATASET}_{PROVIDER}_{MODEL.replace("/", "_")}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregate results (as before)
    results_path = output_dir / 'prompt_style_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    # Save post-level data (NEW - for regression)
    post_level_path = output_dir / 'post_level_data.pkl'
    with open(post_level_path, 'wb') as f:
        pickle.dump(all_post_level_data, f)

    # Save as CSV too for easier inspection
    post_level_csv = output_dir / 'post_level_data.csv'
    pd.DataFrame(all_post_level_data).to_csv(post_level_csv, index=False)

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
    print(f'  - Aggregate results: {results_path}')
    print(f'  - Post-level data: {post_level_path}')
    print(f'  - Post-level CSV: {post_level_csv}')
    print()
    print('Next steps:')
    print(f'  - Analyze: python analyze_experiment.py --results-dir {output_dir}')
    print(f'  - Regression: python regression_analysis.py --results-dir {output_dir} --dataset-name {DATASET}')
    print()

    # Print LLM usage stats
    stats = llm_client.get_stats()
    print('LLM Usage Statistics:')
    print(f'  Total API calls: {stats["call_count"]}')
    print(f'  Total tokens: {stats["total_tokens"]:,}')


if __name__ == '__main__':
    main()
