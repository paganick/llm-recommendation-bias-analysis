"""
Master Analysis Pipeline for LLM Recommendation Bias Experiments

This script orchestrates the full analysis workflow:
1. Run stratified analysis on all completed experiments
2. Generate summary reports
3. Create cross-experiment comparisons

Usage:
    # Analyze all experiments
    python run_full_analysis_pipeline.py --analyze-all

    # Analyze specific experiments
    python run_full_analysis_pipeline.py --experiments bluesky_anthropic_claude-3-5-haiku-20241022

    # Skip already analyzed experiments
    python run_full_analysis_pipeline.py --analyze-all --skip-existing
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# Import our analysis modules
from stratified_analysis import analyze_experiment


def find_all_experiments(experiments_dir: Path) -> list:
    """
    Find all experiment directories that have post_level_data.csv

    Args:
        experiments_dir: Path to outputs/experiments directory

    Returns:
        List of experiment directory paths
    """
    experiments = []

    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            data_file = exp_dir / "post_level_data.csv"
            if data_file.exists():
                experiments.append(exp_dir)

    return sorted(experiments)


def is_experiment_analyzed(exp_dir: Path) -> bool:
    """
    Check if experiment has already been analyzed

    Args:
        exp_dir: Experiment directory path

    Returns:
        True if stratified_analysis directory exists with output files
    """
    analysis_dir = exp_dir / "stratified_analysis"
    if not analysis_dir.exists():
        return False

    # Check if key output files exist
    required_files = [
        analysis_dir / "by_style" / "general_regression.csv",
        analysis_dir / "comparison" / "coefficient_comparison.csv",
        analysis_dir / "comparison" / "bias_by_style.csv"
    ]

    return all(f.exists() for f in required_files)


def get_experiment_metadata(exp_dir: Path) -> dict:
    """
    Extract metadata from experiment directory name

    Example: bluesky_anthropic_claude-3-5-haiku-20241022
    Returns: {dataset: 'bluesky', provider: 'anthropic', model: 'claude-3-5-haiku-20241022'}
    """
    parts = exp_dir.name.split('_')

    if len(parts) >= 3:
        dataset = parts[0]
        provider = parts[1]
        model = '_'.join(parts[2:])
        return {
            'experiment_name': exp_dir.name,
            'dataset': dataset,
            'provider': provider,
            'model': model
        }
    else:
        return {
            'experiment_name': exp_dir.name,
            'dataset': 'unknown',
            'provider': 'unknown',
            'model': 'unknown'
        }


def generate_summary_report(experiments_dir: Path, output_file: Path):
    """
    Generate summary report of all experiments and their analysis status

    Args:
        experiments_dir: Path to outputs/experiments directory
        output_file: Where to save the summary report
    """
    experiments = find_all_experiments(experiments_dir)

    report_data = []

    for exp_dir in experiments:
        metadata = get_experiment_metadata(exp_dir)
        analyzed = is_experiment_analyzed(exp_dir)

        # Get data file info
        data_file = exp_dir / "post_level_data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            n_records = len(df)
            n_styles = df['prompt_style'].nunique()
            n_selected = df['selected'].sum()
        else:
            n_records = n_styles = n_selected = 0

        report_data.append({
            'experiment': metadata['experiment_name'],
            'dataset': metadata['dataset'],
            'provider': metadata['provider'],
            'model': metadata['model'],
            'n_records': n_records,
            'n_styles': n_styles,
            'n_selected': n_selected,
            'analyzed': 'Yes' if analyzed else 'No'
        })

    # Create DataFrame and save
    report_df = pd.DataFrame(report_data)

    # Add summary statistics
    summary_lines = [
        "="*80,
        "LLM Recommendation Bias Analysis - Summary Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        "",
        f"Total experiments: {len(experiments)}",
        f"Analyzed: {sum(1 for r in report_data if r['analyzed'] == 'Yes')}",
        f"Not analyzed: {sum(1 for r in report_data if r['analyzed'] == 'No')}",
        "",
        "="*80,
        "Experiments by dataset:",
        ""
    ]

    for dataset in report_df['dataset'].unique():
        dataset_exps = report_df[report_df['dataset'] == dataset]
        summary_lines.append(f"  {dataset}: {len(dataset_exps)} experiments")

    summary_lines.extend([
        "",
        "="*80,
        "Experiments by provider:",
        ""
    ])

    for provider in report_df['provider'].unique():
        provider_exps = report_df[report_df['provider'] == provider]
        summary_lines.append(f"  {provider}: {len(provider_exps)} experiments")

    summary_lines.extend([
        "",
        "="*80,
        "Detailed experiment list:",
        "="*80,
        ""
    ])

    # Save summary
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))
        f.write('\n')
        f.write(report_df.to_string(index=False))
        f.write('\n\n')

    print(f"\nSummary report saved to: {output_file}")
    print('\n'.join(summary_lines[:20]))  # Print first 20 lines


def main():
    parser = argparse.ArgumentParser(
        description='Run full analysis pipeline on LLM recommendation experiments'
    )

    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='outputs/experiments',
        help='Directory containing experiment outputs (default: outputs/experiments)'
    )

    parser.add_argument(
        '--analyze-all',
        action='store_true',
        help='Analyze all experiments in the experiments directory'
    )

    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        help='Specific experiment names to analyze (e.g., bluesky_anthropic_claude-3-5-haiku-20241022)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip experiments that have already been analyzed'
    )

    parser.add_argument(
        '--output-summary',
        type=str,
        default='outputs/analysis_summary_report.txt',
        help='Where to save summary report (default: outputs/analysis_summary_report.txt)'
    )

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)

    if not experiments_dir.exists():
        print(f"ERROR: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    print("\n" + "="*80)
    print("LLM Recommendation Bias - Full Analysis Pipeline")
    print("="*80 + "\n")

    # Determine which experiments to analyze
    if args.analyze_all:
        experiments_to_analyze = find_all_experiments(experiments_dir)
        print(f"Found {len(experiments_to_analyze)} experiments in {experiments_dir}")
    elif args.experiments:
        experiments_to_analyze = []
        for exp_name in args.experiments:
            exp_path = experiments_dir / exp_name
            if exp_path.exists() and (exp_path / "post_level_data.csv").exists():
                experiments_to_analyze.append(exp_path)
            else:
                print(f"WARNING: Experiment not found or incomplete: {exp_name}")
    else:
        print("ERROR: Must specify either --analyze-all or --experiments")
        sys.exit(1)

    if len(experiments_to_analyze) == 0:
        print("No experiments to analyze!")
        sys.exit(0)

    # Filter out already analyzed if requested
    if args.skip_existing:
        original_count = len(experiments_to_analyze)
        experiments_to_analyze = [
            e for e in experiments_to_analyze if not is_experiment_analyzed(e)
        ]
        skipped = original_count - len(experiments_to_analyze)
        if skipped > 0:
            print(f"Skipping {skipped} already-analyzed experiments")

    if len(experiments_to_analyze) == 0:
        print("All experiments already analyzed!")
        sys.exit(0)

    print(f"\nWill analyze {len(experiments_to_analyze)} experiments:\n")
    for i, exp in enumerate(experiments_to_analyze, 1):
        metadata = get_experiment_metadata(exp)
        print(f"  {i}. {metadata['dataset']:10} | {metadata['provider']:10} | {metadata['model']}")

    print("\n" + "="*80)
    print("Starting analysis...")
    print("="*80 + "\n")

    # Run analysis on each experiment
    success_count = 0
    fail_count = 0

    for i, exp_dir in enumerate(experiments_to_analyze, 1):
        metadata = get_experiment_metadata(exp_dir)

        print(f"\n[{i}/{len(experiments_to_analyze)}] Analyzing: {metadata['experiment_name']}")
        print("-" * 80)

        try:
            analyze_experiment(exp_dir)
            success_count += 1
            print(f"✓ Success: {metadata['experiment_name']}")
        except Exception as e:
            fail_count += 1
            print(f"✗ Failed: {metadata['experiment_name']}")
            print(f"  Error: {e}")

    # Generate summary report
    print("\n" + "="*80)
    print("Generating summary report...")
    print("="*80 + "\n")

    output_file = Path(args.output_summary)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    generate_summary_report(experiments_dir, output_file)

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nAnalyzed: {success_count} experiments")
    print(f"Failed: {fail_count} experiments")
    print(f"\nSummary report: {output_file}")

    # Run meta-analysis (Phase 3)
    print("\n" + "="*80)
    print("Running Meta-Analysis (Phase 3)...")
    print("="*80 + "\n")

    try:
        import subprocess
        meta_result = subprocess.run([
            sys.executable, 'meta_analysis.py',
            '--experiments-dir', str(experiments_dir),
            '--output-dir', 'outputs/meta_analysis'
        ], capture_output=True, text=True)

        if meta_result.returncode == 0:
            print("✓ Meta-analysis completed successfully")
            print(meta_result.stdout)
        else:
            print("✗ Meta-analysis failed")
            print(meta_result.stderr)
    except Exception as e:
        print(f"✗ Error running meta-analysis: {e}")

    # Create dashboards (Phase 4)
    print("\n" + "="*80)
    print("Creating Interactive Dashboards (Phase 4)...")
    print("="*80 + "\n")

    try:
        dashboard_result = subprocess.run([
            sys.executable, 'create_dashboards.py',
            '--all',
            '--experiments-dir', str(experiments_dir),
            '--meta-dir', 'outputs/meta_analysis',
            '--output-dir', 'outputs/dashboards'
        ], capture_output=True, text=True)

        if dashboard_result.returncode == 0:
            print("✓ Dashboards created successfully")
            print(dashboard_result.stdout)
        else:
            print("✗ Dashboard creation failed")
            print(dashboard_result.stderr)
    except Exception as e:
        print(f"✗ Error creating dashboards: {e}")

    print("\n" + "="*80)
    print("COMPLETE ANALYSIS PIPELINE FINISHED!")
    print("="*80)
    print(f"\nPhase 1 - Stratified Analysis: {success_count}/{success_count + fail_count} experiments")
    print("Phase 2 - Meta-Analysis: outputs/meta_analysis/")
    print("Phase 3 - Interactive Dashboards: outputs/dashboards/")
    print(f"\nSummary report: {output_file}")
    print("\nView dashboards:")
    print("  - Individual experiments: outputs/dashboards/dashboard_<experiment_name>.html")
    print("  - Cross-experiment: outputs/dashboards/dashboard_cross_experiment.html")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
