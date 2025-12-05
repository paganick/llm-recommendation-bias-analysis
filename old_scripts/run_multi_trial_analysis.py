"""
Multi-Trial LLM Recommendation Bias Analysis

This script runs MULTIPLE trials with different randomized pools to get
robust statistical estimates of bias. This is important because a single
trial may give unreliable results due to random sampling variation.

Usage:
    python run_multi_trial_analysis.py --provider openai --model gpt-4o-mini --num-trials 10
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from utils.llm_client import get_llm_client
from inference.metadata_inference import infer_tweet_metadata
from recommender.llm_recommender import OneShotRecommender
from analysis.bias_analysis import BiasAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-trial LLM bias analysis')

    # LLM configuration
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['anthropic', 'openai'],
                       help='LLM provider')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='LLM model name')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='twitteraae',
                       help='Dataset to use')
    parser.add_argument('--dataset-version', type=str, default='all_aa',
                       help='Dataset version')
    parser.add_argument('--dataset-size', type=int, default=2000,
                       help='Number of tweets to load from dataset')

    # Experiment configuration
    parser.add_argument('--num-trials', type=int, default=10,
                       help='Number of trials (different random pools)')
    parser.add_argument('--pool-size', type=int, default=50,
                       help='Number of tweets in each recommendation pool')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of recommendations per trial')
    parser.add_argument('--prompt-style', type=str, default='popular',
                       choices=['general', 'popular', 'engaging', 'informative', 'controversial'],
                       help='Recommendation prompt style')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./outputs/multi_trial',
                       help='Output directory')

    return parser.parse_args()


def run_single_trial(llm, tweets_df: pd.DataFrame, pool_size: int, k: int,
                     prompt_style: str, trial_num: int) -> Dict[str, Any]:
    """Run a single trial with a random pool."""
    print(f"  Trial {trial_num}: Sampling pool of {pool_size} tweets...")

    # Sample random pool (different seed each time)
    pool_df = tweets_df.sample(n=pool_size, random_state=42 + trial_num)

    # Get recommendations
    recommender = OneShotRecommender(llm, k=k)
    recommended_df = recommender.recommend(pool_df, prompt_style=prompt_style)

    # Analyze bias
    analyzer = BiasAnalyzer(alpha=0.05)
    demographic_cols = [col for col in ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']
                       if col in pool_df.columns]

    bias_results = analyzer.comprehensive_bias_analysis(
        recommended_df,
        pool_df,
        demographic_cols=demographic_cols
    )

    print(f"  Trial {trial_num}: Complete ✓")

    return {
        'trial_num': trial_num,
        'pool_df': pool_df,
        'recommended_df': recommended_df,
        'bias_results': bias_results
    }


def aggregate_results(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across multiple trials."""
    print("\nAggregating results across trials...")

    aggregated = {
        'num_trials': len(trial_results),
        'demographic_bias': {},
        'sentiment_bias': {},
        'gender_bias': {},
        'topic_bias': {}
    }

    # Aggregate demographic bias
    demo_biases = []
    for trial in trial_results:
        if 'demographic_bias' in trial['bias_results']:
            demo_bias = trial['bias_results']['demographic_bias']
            demo_biases.append(demo_bias['bias_scores'])

    if demo_biases:
        # Calculate mean and std across trials
        all_demos = set()
        for db in demo_biases:
            all_demos.update(db.keys())

        for demo in all_demos:
            scores = [db.get(demo, 0) for db in demo_biases]
            aggregated['demographic_bias'][demo] = {
                'mean_bias': np.mean(scores),
                'std_bias': np.std(scores),
                'min_bias': np.min(scores),
                'max_bias': np.max(scores),
                'all_scores': scores
            }

    # Aggregate sentiment bias
    sent_biases = []
    for trial in trial_results:
        if 'sentiment_bias' in trial['bias_results']:
            sent_biases.append(trial['bias_results']['sentiment_bias']['bias_score'])

    if sent_biases:
        aggregated['sentiment_bias'] = {
            'mean_bias': np.mean(sent_biases),
            'std_bias': np.std(sent_biases),
            'min_bias': np.min(sent_biases),
            'max_bias': np.max(sent_biases),
            'all_scores': sent_biases
        }

    # Aggregate gender distribution
    all_pools = pd.concat([t['pool_df'] for t in trial_results])
    all_recs = pd.concat([t['recommended_df'] for t in trial_results])

    if 'gender_prediction' in all_pools.columns:
        pool_gender_dist = all_pools['gender_prediction'].value_counts(normalize=True).to_dict()
        rec_gender_dist = all_recs['gender_prediction'].value_counts(normalize=True).to_dict()

        aggregated['gender_bias'] = {
            'pool_distribution': pool_gender_dist,
            'recommended_distribution': rec_gender_dist,
            'bias_scores': {
                gender: (rec_gender_dist.get(gender, 0) - pool_gender_dist.get(gender, 0)) * 100
                for gender in set(list(pool_gender_dist.keys()) + list(rec_gender_dist.keys()))
            }
        }

    return aggregated


def main():
    """Run multi-trial analysis."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.provider}_{args.model.split('/')[-1]}_{args.prompt_style}_{args.num_trials}trials_{timestamp}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    print("="*80)
    print("MULTI-TRIAL LLM RECOMMENDATION BIAS ANALYSIS")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Pool size per trial: {args.pool_size}")
    print(f"Recommendations per trial: {args.k}")
    print(f"Output directory: {experiment_dir}")
    print()

    # Step 1: Load and prepare dataset
    print("="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)

    tweets_df = load_dataset(
        args.dataset,
        version=args.dataset_version,
        sample_size=args.dataset_size
    )

    print(f"Loaded {len(tweets_df):,} tweets")

    # Filter out empty tweets
    initial_count = len(tweets_df)
    tweets_df = tweets_df[tweets_df['text'].notna() & (tweets_df['text'] != '')]
    tweets_df = tweets_df[tweets_df['text'].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 10)]
    print(f"After filtering: {len(tweets_df):,} tweets ({initial_count - len(tweets_df)} removed)")
    print()

    # Step 2: Infer metadata
    print("="*80)
    print("STEP 2: Inferring Metadata (once for all trials)")
    print("="*80)

    tweets_df = infer_tweet_metadata(
        tweets_df,
        sentiment_method='vader',
        include_gender=True,
        include_political=True
    )

    print(f"Metadata columns: {len(tweets_df.columns)}")
    print()

    # Step 3: Initialize LLM
    print("="*80)
    print("STEP 3: Initializing LLM Client")
    print("="*80)

    api_key_env = f"{args.provider.upper()}_API_KEY"
    if not os.environ.get(api_key_env):
        print(f"ERROR: {api_key_env} environment variable not set!")
        return 1

    llm = get_llm_client(
        provider=args.provider,
        model=args.model,
        temperature=0.7
    )
    print(f"✓ LLM client initialized: {args.provider}/{args.model}")
    print()

    # Step 4: Run multiple trials
    print("="*80)
    print(f"STEP 4: Running {args.num_trials} Trials")
    print("="*80)

    trial_results = []
    for trial_num in range(1, args.num_trials + 1):
        try:
            result = run_single_trial(
                llm, tweets_df, args.pool_size, args.k,
                args.prompt_style, trial_num
            )
            trial_results.append(result)

            # Save individual trial
            trial_dir = experiment_dir / f'trial_{trial_num:02d}'
            trial_dir.mkdir(exist_ok=True)
            result['recommended_df'].to_csv(trial_dir / 'recommendations.csv', index=False)
            result['pool_df'].to_csv(trial_dir / 'pool.csv', index=False)
            with open(trial_dir / 'bias_results.json', 'w') as f:
                json.dump(result['bias_results'], f, indent=2, default=str)

        except Exception as e:
            print(f"  Trial {trial_num}: ERROR - {e}")
            continue

    print(f"\n✓ Completed {len(trial_results)}/{args.num_trials} trials successfully")
    print()

    # Step 5: Aggregate results
    print("="*80)
    print("STEP 5: Aggregating Results")
    print("="*80)

    aggregated = aggregate_results(trial_results)

    # Save aggregated results
    with open(experiment_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)

    if 'demographic_bias' in aggregated and aggregated['demographic_bias']:
        print("\nDemographic Bias (across all trials):")
        for demo, stats in aggregated['demographic_bias'].items():
            print(f"  {demo}:")
            print(f"    Mean: {stats['mean_bias']:+.2f}pp (±{stats['std_bias']:.2f})")
            print(f"    Range: [{stats['min_bias']:+.2f}, {stats['max_bias']:+.2f}]")

    if 'sentiment_bias' in aggregated and aggregated['sentiment_bias']:
        print("\nSentiment Bias (across all trials):")
        stats = aggregated['sentiment_bias']
        print(f"  Mean: {stats['mean_bias']:+.3f} (±{stats['std_bias']:.3f})")
        print(f"  Range: [{stats['min_bias']:+.3f}, {stats['max_bias']:+.3f}]")

    if 'gender_bias' in aggregated and aggregated['gender_bias']:
        print("\nGender Distribution (aggregated across all trials):")
        for gender, bias in aggregated['gender_bias']['bias_scores'].items():
            pool_pct = aggregated['gender_bias']['pool_distribution'].get(gender, 0) * 100
            rec_pct = aggregated['gender_bias']['recommended_distribution'].get(gender, 0) * 100
            print(f"  {gender}: Pool={pool_pct:.1f}%, Rec={rec_pct:.1f}%, Diff={bias:+.1f}pp")

    # Save configuration
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'llm_provider': args.provider,
        'llm_model': args.model,
        'dataset': args.dataset,
        'dataset_version': args.dataset_version,
        'dataset_size': args.dataset_size,
        'num_trials': args.num_trials,
        'pool_size': args.pool_size,
        'k': args.k,
        'prompt_style': args.prompt_style,
        'successful_trials': len(trial_results),
        'llm_stats': llm.get_stats()
    }

    with open(experiment_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {experiment_dir}")
    print(f"\nLLM Usage:")
    print(f"  API calls: {llm.get_stats()['call_count']}")
    print(f"  Total tokens: {llm.get_stats()['total_tokens']:,}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
