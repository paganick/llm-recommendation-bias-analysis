"""
Cross-Experiment Meta-Analysis for LLM Recommendation Bias

This script performs meta-analysis across all experiments to identify:
1. Universal bias patterns vs. dataset-specific patterns
2. Model differences (Anthropic vs. OpenAI vs. Google)
3. Dataset differences (Bluesky vs. Reddit vs. Twitter)
4. Prompt style effectiveness across all conditions

Key Functions:
1. aggregate_by_dataset() - Compare datasets (Twitter vs Reddit vs Bluesky)
2. aggregate_by_model() - Compare models (OpenAI vs Anthropic vs Gemini)
3. compute_meta_effect_sizes() - Meta-analytic pooling with heterogeneity

Usage:
    python meta_analysis.py --experiments-dir outputs/experiments
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_experiment_name(exp_name):
    """Parse experiment name to extract dataset, provider, model."""
    parts = exp_name.split('_')
    if len(parts) >= 3:
        dataset = parts[0]
        provider = parts[1]
        model = '_'.join(parts[2:])
        return dataset, provider, model
    return None, None, None


def load_all_experiments(experiments_dir):
    """Load stratified analysis results from all completed experiments."""
    exp_dir = Path(experiments_dir)
    all_results = []

    for exp_path in sorted(exp_dir.iterdir()):
        if not exp_path.is_dir():
            continue

        # Check if stratified analysis exists
        strat_dir = exp_path / "stratified_analysis"
        if not strat_dir.exists():
            print(f"Skipping {exp_path.name} (no stratified_analysis)")
            continue

        # Parse experiment metadata
        dataset, provider, model = parse_experiment_name(exp_path.name)
        if not dataset:
            print(f"Warning: Could not parse {exp_path.name}")
            continue

        # Load by_style regressions
        by_style_dir = strat_dir / "by_style"
        if not by_style_dir.exists():
            print(f"Skipping {exp_path.name} (no by_style directory)")
            continue

        for style_file in by_style_dir.glob("*_regression.csv"):
            prompt_style = style_file.stem.replace("_regression", "")
            df = pd.read_csv(style_file)

            # Add metadata
            df['experiment'] = exp_path.name
            df['dataset'] = dataset
            df['provider'] = provider
            df['model'] = model
            df['prompt_style'] = prompt_style

            all_results.append(df)

    if not all_results:
        raise ValueError(f"No experiments found in {experiments_dir}")

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nLoaded {len(all_results)} style-level results from {len(combined['experiment'].unique())} experiments")
    print(f"  Datasets: {sorted(combined['dataset'].unique())}")
    print(f"  Providers: {sorted(combined['provider'].unique())}")
    print(f"  Prompt styles: {sorted(combined['prompt_style'].unique())}")

    return combined


def aggregate_by_dataset(df_all, output_dir):
    """
    Compare datasets (Twitter vs Reddit vs Bluesky) for each prompt style and feature.
    Aggregate coefficients across models within each dataset.
    """
    print("\n" + "="*60)
    print("Aggregating by Dataset (Twitter vs Reddit vs Bluesky)")
    print("="*60)

    output_dir = Path(output_dir) / "by_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # For each prompt_style and feature, compare across datasets
    for prompt_style in df_all['prompt_style'].unique():
        style_df = df_all[df_all['prompt_style'] == prompt_style]

        for feature in style_df['feature'].unique():
            feature_df = style_df[style_df['feature'] == feature]

            # Aggregate by dataset (mean across models)
            dataset_stats = {}
            for dataset in feature_df['dataset'].unique():
                dataset_subset = feature_df[feature_df['dataset'] == dataset]

                if len(dataset_subset) > 0:
                    dataset_stats[dataset] = {
                        'n_models': len(dataset_subset),
                        'mean_coef': dataset_subset['coefficient'].mean(),
                        'std_coef': dataset_subset['coefficient'].std(),
                        'mean_se': dataset_subset['std_error'].mean(),
                        'mean_pval': dataset_subset['p_value'].mean(),
                    }

            # Perform ANOVA if we have all 3 datasets
            if len(dataset_stats) >= 2:
                groups = []
                dataset_names = []
                for dataset, stats_dict in dataset_stats.items():
                    dataset_subset = feature_df[feature_df['dataset'] == dataset]
                    groups.append(dataset_subset['coefficient'].values)
                    dataset_names.append(dataset)

                # One-way ANOVA
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                else:
                    f_stat, p_value = np.nan, np.nan

                # Record results
                result = {
                    'prompt_style': prompt_style,
                    'feature': feature,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                }

                # Add dataset-specific means
                for dataset, stats_dict in dataset_stats.items():
                    result[f'{dataset}_mean_coef'] = stats_dict['mean_coef']
                    result[f'{dataset}_std_coef'] = stats_dict['std_coef']
                    result[f'{dataset}_n_models'] = stats_dict['n_models']

                results.append(result)

    df_results = pd.DataFrame(results)

    # Sort by significance
    df_results = df_results.sort_values('p_value')

    # Save
    output_file = output_dir / "dataset_comparison.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved dataset comparison to: {output_file}")

    # Summary stats
    n_significant = (df_results['p_value'] < 0.05).sum()
    print(f"  Total comparisons: {len(df_results)}")
    print(f"  Significant differences (p<0.05): {n_significant} ({n_significant/len(df_results)*100:.1f}%)")

    # Show top differences
    print(f"\nTop 10 most significant dataset differences:")
    top_diffs = df_results.nsmallest(10, 'p_value')
    for _, row in top_diffs.iterrows():
        print(f"  {row['prompt_style']:15s} {row['feature']:30s} (F={row['f_statistic']:.2f}, p={row['p_value']:.4f})")

    return df_results


def aggregate_by_model(df_all, output_dir):
    """
    Compare models (OpenAI vs Anthropic vs Gemini) for each prompt style and feature.
    Aggregate coefficients across datasets within each model.
    """
    print("\n" + "="*60)
    print("Aggregating by Model (OpenAI vs Anthropic vs Gemini)")
    print("="*60)

    output_dir = Path(output_dir) / "by_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # For each prompt_style and feature, compare across models
    for prompt_style in df_all['prompt_style'].unique():
        style_df = df_all[df_all['prompt_style'] == prompt_style]

        for feature in style_df['feature'].unique():
            feature_df = style_df[style_df['feature'] == feature]

            # Aggregate by provider (mean across datasets)
            provider_stats = {}
            for provider in feature_df['provider'].unique():
                provider_subset = feature_df[feature_df['provider'] == provider]

                if len(provider_subset) > 0:
                    provider_stats[provider] = {
                        'n_datasets': len(provider_subset),
                        'mean_coef': provider_subset['coefficient'].mean(),
                        'std_coef': provider_subset['coefficient'].std(),
                        'mean_se': provider_subset['std_error'].mean(),
                        'mean_pval': provider_subset['p_value'].mean(),
                    }

            # Perform ANOVA if we have multiple providers
            if len(provider_stats) >= 2:
                groups = []
                provider_names = []
                for provider, stats_dict in provider_stats.items():
                    provider_subset = feature_df[feature_df['provider'] == provider]
                    groups.append(provider_subset['coefficient'].values)
                    provider_names.append(provider)

                # One-way ANOVA
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                else:
                    f_stat, p_value = np.nan, np.nan

                # Record results
                result = {
                    'prompt_style': prompt_style,
                    'feature': feature,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                }

                # Add provider-specific means
                for provider, stats_dict in provider_stats.items():
                    result[f'{provider}_mean_coef'] = stats_dict['mean_coef']
                    result[f'{provider}_std_coef'] = stats_dict['std_coef']
                    result[f'{provider}_n_datasets'] = stats_dict['n_datasets']

                results.append(result)

    df_results = pd.DataFrame(results)

    # Sort by significance
    df_results = df_results.sort_values('p_value')

    # Save
    output_file = output_dir / "model_comparison.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved model comparison to: {output_file}")

    # Summary stats
    n_significant = (df_results['p_value'] < 0.05).sum()
    print(f"  Total comparisons: {len(df_results)}")
    print(f"  Significant differences (p<0.05): {n_significant} ({n_significant/len(df_results)*100:.1f}%)")

    # Show top differences
    print(f"\nTop 10 most significant model differences:")
    top_diffs = df_results.nsmallest(10, 'p_value')
    for _, row in top_diffs.iterrows():
        print(f"  {row['prompt_style']:15s} {row['feature']:30s} (F={row['f_statistic']:.2f}, p={row['p_value']:.4f})")

    return df_results


def compute_meta_effect_sizes(df_all, output_dir):
    """
    Compute meta-analytic pooled effect sizes across all experiments.
    Calculate heterogeneity statistics (Q-test, I²).
    """
    print("\n" + "="*60)
    print("Computing Meta-Analytic Effect Sizes")
    print("="*60)

    output_dir = Path(output_dir) / "meta_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # For each prompt_style and feature, pool across all experiments
    for prompt_style in df_all['prompt_style'].unique():
        style_df = df_all[df_all['prompt_style'] == prompt_style]

        for feature in style_df['feature'].unique():
            feature_df = style_df[style_df['feature'] == feature]

            if len(feature_df) < 2:
                continue

            # Extract coefficients and standard errors
            coefs = feature_df['coefficient'].values
            ses = feature_df['std_error'].values

            # Fixed-effects meta-analysis (inverse variance weighting)
            weights = 1 / (ses ** 2)
            pooled_coef = np.sum(weights * coefs) / np.sum(weights)
            pooled_se = np.sqrt(1 / np.sum(weights))

            # Z-test for pooled effect
            z_score = pooled_coef / pooled_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            # Heterogeneity: Cochran's Q test
            Q = np.sum(weights * (coefs - pooled_coef) ** 2)
            df_Q = len(coefs) - 1
            p_heterogeneity = 1 - stats.chi2.cdf(Q, df_Q) if df_Q > 0 else np.nan

            # I² statistic (proportion of variation due to heterogeneity)
            I_squared = max(0, (Q - df_Q) / Q * 100) if Q > 0 else 0

            results.append({
                'prompt_style': prompt_style,
                'feature': feature,
                'n_experiments': len(feature_df),
                'pooled_coefficient': pooled_coef,
                'pooled_se': pooled_se,
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'Q_statistic': Q,
                'Q_df': df_Q,
                'Q_p_value': p_heterogeneity,
                'I_squared': I_squared,
                'mean_coef': coefs.mean(),
                'std_coef': coefs.std(),
                'min_coef': coefs.min(),
                'max_coef': coefs.max(),
            })

    df_results = pd.DataFrame(results)

    # Sort by absolute pooled coefficient
    df_results = df_results.sort_values('pooled_coefficient', key=abs, ascending=False)

    # Save
    output_file = output_dir / "meta_effect_sizes.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved meta-analysis results to: {output_file}")

    # Summary stats
    n_significant = (df_results['p_value'] < 0.05).sum()
    n_heterogeneous = (df_results['Q_p_value'] < 0.05).sum()

    print(f"  Total meta-analyses: {len(df_results)}")
    print(f"  Significant pooled effects (p<0.05): {n_significant} ({n_significant/len(df_results)*100:.1f}%)")
    print(f"  Heterogeneous effects (Q p<0.05): {n_heterogeneous} ({n_heterogeneous/len(df_results)*100:.1f}%)")
    print(f"  Mean I² (heterogeneity): {df_results['I_squared'].mean():.1f}%")

    # Show top effects by prompt style
    print(f"\nTop 5 largest pooled effects by prompt style:")
    for prompt_style in df_results['prompt_style'].unique():
        style_df = df_results[df_results['prompt_style'] == prompt_style]
        top_effects = style_df.nlargest(5, 'pooled_coefficient', keep='all')
        print(f"\n  {prompt_style}:")
        for _, row in top_effects.head(5).iterrows():
            sig = "*" if row['significant'] else ""
            het = f"(I²={row['I_squared']:.0f}%)" if row['I_squared'] > 50 else ""
            print(f"    {row['feature']:30s}: {row['pooled_coefficient']:+.3f} {sig} {het}")

    return df_results


def create_summary_report(df_all, df_dataset, df_model, df_meta, output_dir):
    """Create a comprehensive summary report."""
    output_file = Path(output_dir) / "meta_analysis_summary.txt"

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-EXPERIMENT META-ANALYSIS SUMMARY\n")
        f.write("LLM Recommendation Bias Analysis\n")
        f.write("="*80 + "\n\n")

        # Overview
        f.write("OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total experiments analyzed: {df_all['experiment'].nunique()}\n")
        f.write(f"Datasets: {', '.join(sorted(df_all['dataset'].unique()))}\n")
        f.write(f"Providers: {', '.join(sorted(df_all['provider'].unique()))}\n")
        f.write(f"Prompt styles: {', '.join(sorted(df_all['prompt_style'].unique()))}\n")
        f.write(f"Features analyzed: {df_all['feature'].nunique()}\n\n")

        # Dataset comparison
        f.write("DATASET COMPARISON (Twitter vs Reddit vs Bluesky)\n")
        f.write("-"*80 + "\n")
        n_sig_dataset = (df_dataset['p_value'] < 0.05).sum()
        f.write(f"Total comparisons: {len(df_dataset)}\n")
        f.write(f"Significant differences (p<0.05): {n_sig_dataset} ({n_sig_dataset/len(df_dataset)*100:.1f}%)\n\n")
        f.write("Top 10 most significant differences:\n")
        for i, row in df_dataset.nsmallest(10, 'p_value').iterrows():
            f.write(f"  {i+1}. [{row['prompt_style']}] {row['feature']}\n")
            f.write(f"     F={row['f_statistic']:.2f}, p={row['p_value']:.4f}\n")
        f.write("\n")

        # Model comparison
        f.write("MODEL COMPARISON (OpenAI vs Anthropic vs Gemini)\n")
        f.write("-"*80 + "\n")
        n_sig_model = (df_model['p_value'] < 0.05).sum()
        f.write(f"Total comparisons: {len(df_model)}\n")
        f.write(f"Significant differences (p<0.05): {n_sig_model} ({n_sig_model/len(df_model)*100:.1f}%)\n\n")
        f.write("Top 10 most significant differences:\n")
        for i, row in df_model.nsmallest(10, 'p_value').iterrows():
            f.write(f"  {i+1}. [{row['prompt_style']}] {row['feature']}\n")
            f.write(f"     F={row['f_statistic']:.2f}, p={row['p_value']:.4f}\n")
        f.write("\n")

        # Meta-analysis
        f.write("META-ANALYTIC POOLED EFFECTS\n")
        f.write("-"*80 + "\n")
        n_sig_meta = (df_meta['p_value'] < 0.05).sum()
        n_het = (df_meta['Q_p_value'] < 0.05).sum()
        f.write(f"Total meta-analyses: {len(df_meta)}\n")
        f.write(f"Significant pooled effects (p<0.05): {n_sig_meta} ({n_sig_meta/len(df_meta)*100:.1f}%)\n")
        f.write(f"Heterogeneous effects (Q p<0.05): {n_het} ({n_het/len(df_meta)*100:.1f}%)\n")
        f.write(f"Mean I² (heterogeneity): {df_meta['I_squared'].mean():.1f}%\n\n")

        # Top universal effects (consistent across all)
        f.write("Top 15 universal effects (pooled across all experiments):\n")
        top_universal = df_meta[df_meta['significant']].nlargest(15, 'pooled_coefficient', keep='all')
        for i, row in top_universal.head(15).iterrows():
            het_flag = "⚠" if row['I_squared'] > 50 else ""
            f.write(f"  {i+1}. [{row['prompt_style']}] {row['feature']}\n")
            f.write(f"     Pooled coef: {row['pooled_coefficient']:+.3f} (SE={row['pooled_se']:.3f}, p={row['p_value']:.4e})\n")
            f.write(f"     I²={row['I_squared']:.0f}% {het_flag}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\nSummary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-experiment meta-analysis")
    parser.add_argument('--experiments-dir', type=str, default='outputs/experiments',
                       help='Directory containing all experiments')
    parser.add_argument('--output-dir', type=str, default='outputs/comparisons',
                       help='Output directory for meta-analysis results')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CROSS-EXPERIMENT META-ANALYSIS")
    print("="*60)

    # Load all experiments
    df_all = load_all_experiments(args.experiments_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    df_dataset = aggregate_by_dataset(df_all, output_dir)
    df_model = aggregate_by_model(df_all, output_dir)
    df_meta = compute_meta_effect_sizes(df_all, output_dir)

    # Create summary report
    create_summary_report(df_all, df_dataset, df_model, df_meta, output_dir)

    print("\n" + "="*60)
    print("✓ Meta-analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
