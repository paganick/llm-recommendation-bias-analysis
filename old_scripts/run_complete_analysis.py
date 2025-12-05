"""
Complete End-to-End Bias Analysis Script

This script demonstrates the complete workflow:
1. Load social media dataset (TwitterAAE)
2. Infer metadata (sentiment, topics, gender, political leaning, style, etc.)
3. Get LLM recommendations
4. Analyze bias in recommendations
5. Generate visualizations

Usage:
    python run_complete_analysis.py --provider anthropic --model claude-3-5-sonnet-20241022
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from utils.llm_client import get_llm_client
from inference.metadata_inference import infer_tweet_metadata
from recommender.llm_recommender import OneShotRecommender
from analysis.bias_analysis import BiasAnalyzer
from analysis.visualization import create_bias_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run complete LLM recommendation bias analysis')

    # LLM configuration
    parser.add_argument('--provider', type=str, default='anthropic',
                       choices=['anthropic', 'openai'],
                       help='LLM provider')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022',
                       help='LLM model name')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='twitteraae',
                       choices=['twitteraae', 'dadit'],
                       help='Dataset to use')
    parser.add_argument('--dataset-version', type=str, default='all_aa',
                       help='Dataset version')
    parser.add_argument('--dataset-size', type=int, default=5000,
                       help='Number of tweets to load')

    # Experiment configuration
    parser.add_argument('--pool-size', type=int, default=50,
                       help='Number of tweets in recommendation pool')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--prompt-style', type=str, default='popular',
                       choices=['general', 'popular', 'engaging', 'informative', 'controversial'],
                       help='Recommendation prompt style')

    # Inference configuration
    parser.add_argument('--include-gender', action='store_true', default=True,
                       help='Include gender inference')
    parser.add_argument('--include-political', action='store_true', default=True,
                       help='Include political leaning inference')
    parser.add_argument('--skip-metadata', action='store_true',
                       help='Skip metadata inference (if already cached)')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./outputs/complete_analysis',
                       help='Output directory')
    parser.add_argument('--metadata-cache', type=str, default=None,
                       help='Path to metadata cache file')

    return parser.parse_args()


def main():
    """Run complete bias analysis."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.provider}_{args.model.split('/')[-1]}_{args.prompt_style}_{timestamp}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    print("="*80)
    print("LLM RECOMMENDATION BIAS ANALYSIS")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {experiment_dir}")
    print()

    # Step 1: Load dataset
    print("="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)

    tweets_df = load_dataset(
        args.dataset,
        version=args.dataset_version,
        sample_size=args.dataset_size
    )

    print(f"Loaded {len(tweets_df):,} tweets")
    print(f"Columns: {tweets_df.columns.tolist()}")
    print()

    # Save dataset sample
    sample_tweets_path = experiment_dir / 'dataset_sample.csv'
    tweets_df.head(100).to_csv(sample_tweets_path, index=False)
    print(f"Saved dataset sample to {sample_tweets_path}")
    print()

    # Step 2: Infer metadata
    if not args.skip_metadata:
        print("="*80)
        print("STEP 2: Inferring Metadata")
        print("="*80)
        print("This may take a while for large datasets...")
        print()

        # Check if we have cached metadata
        if args.metadata_cache and Path(args.metadata_cache).exists():
            print(f"Loading metadata from cache: {args.metadata_cache}")
            cached_metadata = pd.read_csv(args.metadata_cache)

            # Merge with tweets_df
            merge_cols = [col for col in cached_metadata.columns if col in tweets_df.columns]
            if merge_cols:
                tweets_df = pd.merge(
                    tweets_df,
                    cached_metadata,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_meta')
                )
        else:
            # Infer metadata
            tweets_df = infer_tweet_metadata(
                tweets_df,
                text_column='text',
                sentiment_method='vader',
                topic_method='keyword',
                gender_method='keyword',
                political_method='keyword',
                include_gender=args.include_gender,
                include_political=args.include_political
            )

            # Cache metadata
            if args.metadata_cache:
                tweets_df.to_csv(args.metadata_cache, index=False)
                print(f"Saved metadata cache to {args.metadata_cache}")

        print(f"Metadata columns added: {len(tweets_df.columns)} total")
        print()

        # Save metadata summary
        metadata_cols = [col for col in tweets_df.columns if col not in ['tweet_id', 'timestamp', 'user_id', 'text']]
        metadata_summary = tweets_df[metadata_cols].describe()
        metadata_summary.to_csv(experiment_dir / 'metadata_summary.csv')

    # Step 3: Initialize LLM client
    print("="*80)
    print("STEP 3: Initializing LLM Client")
    print("="*80)

    # Check for API key
    api_key_env = f"{args.provider.upper()}_API_KEY"
    if not os.environ.get(api_key_env):
        print(f"ERROR: {api_key_env} environment variable not set!")
        print(f"Set it with: export {api_key_env}='your-key-here'")
        return 1

    try:
        llm = get_llm_client(
            provider=args.provider,
            model=args.model,
            temperature=0.7
        )
        print(f"✓ LLM client initialized: {args.provider}/{args.model}")
    except Exception as e:
        print(f"ERROR initializing LLM client: {e}")
        return 1

    print()

    # Step 4: Sample tweet pool and get recommendations
    print("="*80)
    print("STEP 4: Getting LLM Recommendations")
    print("="*80)

    # Sample tweet pool
    pool_df = tweets_df.sample(n=args.pool_size, random_state=42)
    print(f"Sampled pool of {len(pool_df)} tweets")

    # Get recommendations
    recommender = OneShotRecommender(llm, k=args.k)

    print(f"Requesting {args.k} recommendations (prompt style: {args.prompt_style})...")
    try:
        recommended_df = recommender.recommend(
            pool_df,
            prompt_style=args.prompt_style,
            max_pool_size=args.pool_size
        )
        print(f"✓ Received {len(recommended_df)} recommendations")
    except Exception as e:
        print(f"ERROR getting recommendations: {e}")
        return 1

    # Save recommendations
    recommended_df.to_csv(experiment_dir / 'recommended_tweets.csv', index=False)
    pool_df.to_csv(experiment_dir / 'pool_tweets.csv', index=False)

    print()
    print("Top 5 recommended tweets:")
    for _, row in recommended_df.head(5).iterrows():
        print(f"\n[Rank {row['rank']}]")
        text = str(row['text']) if not isinstance(row['text'], str) else row['text']
        print(f"  {text[:100]}...")
        if 'demo_aa' in row:
            print(f"  Demographics: AA={row['demo_aa']:.3f}, White={row['demo_white']:.3f}")
        if 'sentiment_label' in row:
            print(f"  Sentiment: {row['sentiment_label']} (polarity={row['sentiment_polarity']:.3f})")
        if 'gender_prediction' in row:
            print(f"  Gender: {row['gender_prediction']} (conf={row['gender_confidence']:.2f})")
        if 'political_leaning' in row:
            print(f"  Political: {row['political_leaning']} (conf={row['political_confidence']:.2f})")

    print()

    # Step 5: Analyze bias
    print("="*80)
    print("STEP 5: Analyzing Bias")
    print("="*80)

    analyzer = BiasAnalyzer(alpha=0.05)

    demographic_cols = ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']
    # Filter to only available columns
    demographic_cols = [col for col in demographic_cols if col in pool_df.columns]

    bias_results = analyzer.comprehensive_bias_analysis(
        recommended_df,
        pool_df,
        demographic_cols=demographic_cols
    )

    # Save bias results
    with open(experiment_dir / 'bias_analysis.json', 'w') as f:
        json.dump(bias_results, f, indent=2, default=str)

    # Print summary
    summary = bias_results.get('summary', {})
    print(f"\nBias Analysis Summary:")
    print(f"  Total significant biases detected: {summary.get('total_significant_biases', 0)}")
    print(f"  Has significant bias: {summary.get('has_significant_bias', False)}")

    if summary.get('significant_biases'):
        print(f"\n  Significant biases found:")
        for bias in summary['significant_biases'][:5]:
            print(f"    - {bias.get('type')}: {bias.get('attribute', 'N/A')} (p={bias.get('p_value', 'N/A'):.4f})")

    print()

    # Print demographic bias (AA vs non-AA)
    if 'demographic_bias' in bias_results:
        print("\nDemographic Bias (Author Demographics):")
        demo_bias = bias_results['demographic_bias']

        # African American vs non-AA comparison
        if 'demo_aa' in demo_bias['bias_scores']:
            aa_pool = demo_bias['pool_stats']['demo_aa']['mean']
            aa_rec = demo_bias['recommended_stats']['demo_aa']['mean']
            aa_diff = demo_bias['bias_scores']['demo_aa']
            aa_sig = demo_bias['statistical_tests']['demo_aa']['significant']

            marker = "***" if aa_sig else ""
            print(f"  African American authors:")
            print(f"    Pool: {aa_pool*100:.1f}% | Recommended: {aa_rec*100:.1f}% | Diff: {aa_diff:+.1f}pp {marker}")

        # Show other demographics for context
        print(f"\n  Other demographics (for reference):")
        for demo in ['demo_white', 'demo_hispanic', 'demo_other']:
            if demo in demo_bias['bias_scores']:
                pool_mean = demo_bias['pool_stats'][demo]['mean']
                rec_mean = demo_bias['recommended_stats'][demo]['mean']
                diff = demo_bias['bias_scores'][demo]
                sig = demo_bias['statistical_tests'][demo]['significant']
                marker = "***" if sig else ""
                demo_label = demo.replace('demo_', '').capitalize()
                print(f"    {demo_label}: Pool={pool_mean*100:.1f}%, Rec={rec_mean*100:.1f}%, Diff={diff:+.1f}pp {marker}")

    # Print sentiment bias
    if 'sentiment_bias' in bias_results:
        print("\nSentiment Bias:")
        sent_bias = bias_results['sentiment_bias']
        print(f"  Bias score: {sent_bias['bias_score']:+.3f}")
        print(f"  Favors: {sent_bias['favors']}")
        print(f"  Significant: {sent_bias['significant']}")

    # Print gender bias (if available)
    if 'gender_prediction' in pool_df.columns:
        print("\nGender Distribution:")
        pool_gender = pool_df['gender_prediction'].value_counts(normalize=True)
        rec_gender = recommended_df['gender_prediction'].value_counts(normalize=True)
        for gender in ['male', 'female', 'unknown']:
            pool_pct = pool_gender.get(gender, 0) * 100
            rec_pct = rec_gender.get(gender, 0) * 100
            diff = rec_pct - pool_pct
            print(f"  {gender}: Pool={pool_pct:.1f}%, Rec={rec_pct:.1f}%, Diff={diff:+.1f}%")

    # Print political bias (if available)
    if 'political_leaning' in pool_df.columns:
        print("\nPolitical Leaning Distribution:")
        pool_pol = pool_df['political_leaning'].value_counts(normalize=True)
        rec_pol = recommended_df['political_leaning'].value_counts(normalize=True)
        for leaning in ['left', 'right', 'center', 'unknown']:
            pool_pct = pool_pol.get(leaning, 0) * 100
            rec_pct = rec_pol.get(leaning, 0) * 100
            diff = rec_pct - pool_pct
            print(f"  {leaning}: Pool={pool_pct:.1f}%, Rec={rec_pct:.1f}%, Diff={diff:+.1f}%")

    print()

    # Step 6: Generate visualizations
    print("="*80)
    print("STEP 6: Generating Visualizations")
    print("="*80)

    viz_dir = experiment_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    try:
        create_bias_report(
            bias_results,
            output_dir=viz_dir,
            system_name=f"{args.provider}_{args.model.split('/')[-1]}"
        )
        print(f"✓ Visualizations saved to {viz_dir}")
    except Exception as e:
        print(f"Warning: Error generating visualizations: {e}")

    print()

    # Save experiment configuration
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'llm_provider': args.provider,
        'llm_model': args.model,
        'dataset': args.dataset,
        'dataset_version': args.dataset_version,
        'dataset_size': args.dataset_size,
        'pool_size': args.pool_size,
        'k': args.k,
        'prompt_style': args.prompt_style,
        'include_gender': args.include_gender,
        'include_political': args.include_political,
        'llm_stats': llm.get_stats()
    }

    with open(experiment_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Final summary
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {experiment_dir}")
    print(f"\nFiles generated:")
    print(f"  - experiment_config.json: Experiment configuration")
    print(f"  - bias_analysis.json: Complete bias analysis results")
    print(f"  - recommended_tweets.csv: Recommended tweets with metadata")
    print(f"  - pool_tweets.csv: Full pool of tweets")
    print(f"  - dataset_sample.csv: Sample of original dataset")
    print(f"  - visualizations/: Bias visualizations")
    print()
    print(f"LLM Usage:")
    print(f"  API calls: {llm.get_stats()['call_count']}")
    print(f"  Total tokens: {llm.get_stats()['total_tokens']:,}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
