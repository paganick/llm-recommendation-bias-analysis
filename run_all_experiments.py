"""
Master script to run comprehensive experiments across datasets and models.

This script:
1. Runs experiments with tracking for all combinations of datasets and models
2. Runs comprehensive analysis (including persona-level bias)
3. Creates cross-dataset and cross-model comparisons

Usage:
    # Run all experiments (default)
    python run_all_experiments.py

    # Run specific datasets
    python run_all_experiments.py --datasets reddit twitter

    # Run specific models
    python run_all_experiments.py --providers openai gemini

    # Quick test mode (fewer trials)
    python run_all_experiments.py --quick

    # Skip experiments, only run analysis
    python run_all_experiments.py --skip-experiments
"""

import subprocess
import argparse
from pathlib import Path
import sys
import time


def run_command(cmd: list, description: str):
    """Run a command and report success/failure."""

    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, text=True,
                                capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✓ Success ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed ({elapsed:.1f}s)")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['reddit', 'twitter', 'bluesky'],
                       help='Datasets to test')
    parser.add_argument('--providers', type=str, nargs='+',
                       default=['openai', 'gemini', 'anthropic'],
                       help='Model providers to test')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific models (overrides providers)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer trials)')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip experiments, only run analysis')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis, only run experiments')

    args = parser.parse_args()

    # Define model configurations
    if args.models:
        # Use specific models provided
        model_configs = []
        for model in args.models:
            # Try to infer provider from model name
            if 'gpt' in model.lower():
                provider = 'openai'
            elif 'claude' in model.lower():
                provider = 'anthropic'
            elif 'gemini' in model.lower():
                provider = 'gemini'
            else:
                print(f"Warning: Can't infer provider for {model}, defaulting to openai")
                provider = 'openai'

            model_configs.append((provider, model))
    else:
        # Use default models for each provider
        default_models = {
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-5-haiku-20241022',  # Cheaper model
            'gemini': 'gemini-2.0-flash'
        }

        model_configs = [(p, default_models[p]) for p in args.providers
                         if p in default_models]

    # Trial settings
    if args.quick:
        n_trials = 10
        dataset_size = 1000
    else:
        n_trials = 100
        dataset_size = 5000

    print('='*80)
    print('RUNNING COMPREHENSIVE EXPERIMENTS')
    print('='*80)
    print(f"\nDatasets: {args.datasets}")
    print(f"Models: {[f'{p}/{m}' for p, m in model_configs]}")
    print(f"Trials per style: {n_trials}")
    print(f"Dataset size: {dataset_size}")
    print()

    # Track results
    successful_experiments = []
    failed_experiments = []

    # Run experiments
    if not args.skip_experiments:
        print("\n" + "="*80)
        print("PHASE 1: RUNNING EXPERIMENTS")
        print("="*80)

        for dataset in args.datasets:
            for provider, model in model_configs:
                exp_name = f"{dataset}_{provider}_{model.replace('/', '_')}"

                cmd = [
                    'python', 'run_experiment_with_tracking.py',
                    '--dataset', dataset,
                    '--provider', provider,
                    '--model', model,
                    '--dataset-size', str(dataset_size),
                    '--n-trials', str(n_trials)
                ]

                success = run_command(cmd, f"Experiment: {exp_name}")

                if success:
                    successful_experiments.append(exp_name)
                else:
                    failed_experiments.append(exp_name)

    # Run comprehensive analysis
    if not args.skip_analysis:
        print("\n" + "="*80)
        print("PHASE 2: COMPREHENSIVE ANALYSIS")
        print("="*80)

        # Find all experiment directories
        experiments_dir = Path('outputs/experiments')
        if experiments_dir.exists():
            exp_dirs = [d for d in experiments_dir.glob('*')
                        if d.is_dir() and (d / 'post_level_data.pkl').exists()]

            print(f"\nFound {len(exp_dirs)} experiments to analyze")

            for exp_dir in exp_dirs:
                cmd = [
                    'python', 'analyze_experiment_comprehensive.py',
                    '--results-dir', str(exp_dir)
                ]

                run_command(cmd, f"Analyzing: {exp_dir.name}")

        # Cross-experiment comparison
        print("\n" + "="*80)
        print("PHASE 3: CROSS-EXPERIMENT COMPARISONS")
        print("="*80)

        cmd = ['python', 'compare_experiments.py', '--compare-all']
        run_command(cmd, "Cross-dataset and cross-model comparison")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not args.skip_experiments:
        print(f"\nExperiments run: {len(successful_experiments) + len(failed_experiments)}")
        print(f"  ✓ Successful: {len(successful_experiments)}")
        print(f"  ✗ Failed: {len(failed_experiments)}")

        if failed_experiments:
            print("\nFailed experiments:")
            for exp in failed_experiments:
                print(f"  - {exp}")

    print("\nResults are saved in:")
    print("  - outputs/experiments/{dataset}_{provider}_{model}/")
    print("  - outputs/comparisons/")

    print("\nKey outputs:")
    print("  - comprehensive_bias_analysis.csv: Detailed bias analysis")
    print("  - plots/persona_bias_*.png: Persona-level bias visualizations")
    print("  - plots/bias_heatmap.png: Cross-style comparison")
    print("  - ../comparisons/cross_dataset_comparison.png")
    print("  - ../comparisons/cross_model_comparison.png")


if __name__ == '__main__':
    main()
