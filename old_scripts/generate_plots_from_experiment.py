"""
Generate bias analysis and visualizations from existing experiment data.

This script loads completed experiment CSVs and generates:
1. Bias analysis results (JSON)
2. All visualizations (PNG files)
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.bias_analysis import BiasAnalyzer
from analysis.visualization import create_bias_report


def generate_analysis_from_experiment(experiment_dir: str):
    """
    Generate bias analysis and visualizations from experiment directory.

    Args:
        experiment_dir: Path to experiment directory containing CSVs
    """
    experiment_path = Path(experiment_dir)

    if not experiment_path.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return 1

    print("="*80)
    print(f"GENERATING ANALYSIS FOR: {experiment_path.name}")
    print("="*80)
    print()

    # Load data
    print("Loading experiment data...")
    pool_path = experiment_path / "pool_tweets.csv"
    rec_path = experiment_path / "recommended_tweets.csv"

    if not pool_path.exists() or not rec_path.exists():
        print(f"Error: Missing required CSV files in {experiment_dir}")
        print(f"  Expected: pool_tweets.csv and recommended_tweets.csv")
        return 1

    pool_df = pd.read_csv(pool_path)
    recommended_df = pd.read_csv(rec_path)

    print(f"✓ Loaded pool: {len(pool_df)} tweets")
    print(f"✓ Loaded recommendations: {len(recommended_df)} tweets")
    print()

    # Analyze bias
    print("Running bias analysis...")
    analyzer = BiasAnalyzer(alpha=0.05)

    # Identify available demographic columns
    demographic_cols = ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']
    demographic_cols = [col for col in demographic_cols if col in pool_df.columns]

    if not demographic_cols:
        print("Warning: No demographic columns found in data")
        demographic_cols = None

    bias_results = analyzer.comprehensive_bias_analysis(
        recommended_df,
        pool_df,
        demographic_cols=demographic_cols
    )

    # Save bias results
    bias_path = experiment_path / 'bias_analysis.json'
    with open(bias_path, 'w') as f:
        json.dump(bias_results, f, indent=2, default=str)
    print(f"✓ Saved bias analysis to: {bias_path}")
    print()

    # Print summary
    summary = bias_results.get('summary', {})
    print("BIAS ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total significant biases detected: {summary.get('total_significant_biases', 0)}")
    print(f"Has significant bias: {summary.get('has_significant_bias', False)}")

    if summary.get('significant_biases'):
        print(f"\nSignificant biases found:")
        for bias in summary['significant_biases']:
            print(f"  - {bias.get('type')}: {bias.get('attribute', 'N/A')} (p={bias.get('p_value', 'N/A'):.4f})")

    # Demographic details
    if 'demographic_bias' in bias_results:
        print("\nDEMOGRAPHIC BIAS DETAILS")
        print("="*80)
        demo_bias = bias_results['demographic_bias']

        for demo in demographic_cols:
            if demo in demo_bias['bias_scores']:
                pool_mean = demo_bias['pool_stats'][demo]['mean']
                rec_mean = demo_bias['recommended_stats'][demo]['mean']
                diff = demo_bias['bias_scores'][demo]
                sig = demo_bias['statistical_tests'][demo]['significant']
                p_val = demo_bias['statistical_tests'][demo]['p_value']
                marker = "***" if sig else ""
                demo_label = demo.replace('demo_', '').upper()
                print(f"{demo_label:15s}: Pool={pool_mean*100:5.1f}%, Rec={rec_mean*100:5.1f}%, "
                      f"Diff={diff:+6.1f}pp, p={p_val:.4f} {marker}")

    # Sentiment details
    if 'sentiment_bias' in bias_results:
        print("\nSENTIMENT BIAS DETAILS")
        print("="*80)
        sent_bias = bias_results['sentiment_bias']
        print(f"Pool mean sentiment:        {sent_bias['pool_mean_sentiment']:+.3f}")
        print(f"Recommended mean sentiment: {sent_bias['recommended_mean_sentiment']:+.3f}")
        print(f"Bias score:                 {sent_bias['bias_score']:+.3f}")
        print(f"Favors:                     {sent_bias['favors']}")
        print(f"Significant:                {sent_bias['significant']} (p={sent_bias.get('p_value', 'N/A'):.4f})")

    print()

    # Generate visualizations
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    viz_dir = experiment_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    try:
        create_bias_report(
            bias_results,
            output_dir=viz_dir,
            system_name=experiment_path.name
        )
        print(f"✓ Visualizations saved to: {viz_dir}")
        print()

        # List generated files
        viz_files = list(viz_dir.glob("*.png"))
        print(f"Generated {len(viz_files)} visualization files:")
        for f in sorted(viz_files):
            print(f"  - {f.name}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_plots_from_experiment.py <experiment_dir>")
        print("\nExample:")
        print("  python generate_plots_from_experiment.py outputs/complete_analysis/openai_gpt-4o-mini_popular_20251201_225610")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    sys.exit(generate_analysis_from_experiment(experiment_dir))
