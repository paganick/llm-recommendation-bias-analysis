"""
Comprehensive Bias Analysis Framework

This module implements the complete bias analysis pipeline from the research plan:
1. Bias calculation for all feature types (numerical, categorical, binary)
2. Effect size computation (Cohen's d, Cramér's V, relative risk)
3. Statistical testing with multiple comparison correction
4. 8-level aggregation hierarchy
5. Publication-ready output tables

Usage:
    python comprehensive_bias_analysis.py --experiment outputs/experiments/twitter_anthropic_claude-sonnet-4-5-20250929
    python comprehensive_bias_analysis.py --all-experiments
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FEATURE TYPE DEFINITIONS
# ============================================================================

NUMERICAL_FEATURES = [
    'text_length',
    'word_count',
    'avg_word_length',
    'sentiment_polarity',
    'sentiment_subjectivity',
    'sentiment_positive',
    'sentiment_negative',
    'sentiment_neutral',
    'formality_score',
    'polarization_score',
    'toxicity',
    'severe_toxicity',
    'obscene',
    'threat',
    'insult',
    'identity_attack',
    'primary_topic_score'
]

CATEGORICAL_FEATURES = [
    'sentiment_label',
    'primary_topic',
    'controversy_level',
    'author_gender',
    'author_politics'
]

BINARY_FEATURES = [
    'has_emoji',
    'has_hashtag',
    'has_mention',
    'has_url',
    'has_polarizing_content'
]


# ============================================================================
# BIAS CALCULATION FUNCTIONS
# ============================================================================

def calculate_numerical_bias(pool_data: pd.Series, recommended_data: pd.Series) -> Dict:
    """
    Calculate bias metrics for numerical features.

    Returns:
        - mean_pool: Mean in pool
        - mean_recommended: Mean in recommended set
        - bias: Difference (recommended - pool)
        - cohens_d: Effect size
        - t_statistic: T-test statistic
        - p_value: T-test p-value
        - ci_lower, ci_upper: 95% confidence interval for bias
    """
    # Remove NaN values
    pool_clean = pool_data.dropna()
    rec_clean = recommended_data.dropna()

    if len(pool_clean) == 0 or len(rec_clean) == 0:
        return {
            'mean_pool': np.nan,
            'mean_recommended': np.nan,
            'bias': np.nan,
            'cohens_d': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }

    # Basic statistics
    mean_pool = pool_clean.mean()
    mean_rec = rec_clean.mean()
    bias = mean_rec - mean_pool

    # Cohen's d effect size
    pooled_std = np.sqrt(((len(pool_clean) - 1) * pool_clean.std()**2 +
                          (len(rec_clean) - 1) * rec_clean.std()**2) /
                         (len(pool_clean) + len(rec_clean) - 2))

    cohens_d = bias / pooled_std if pooled_std > 0 else 0

    # T-test (two-sample, two-tailed)
    t_stat, p_val = stats.ttest_ind(rec_clean, pool_clean, equal_var=False)

    # 95% Confidence interval for the mean difference
    se_diff = np.sqrt(pool_clean.var()/len(pool_clean) + rec_clean.var()/len(rec_clean))
    ci_lower = bias - 1.96 * se_diff
    ci_upper = bias + 1.96 * se_diff

    return {
        'mean_pool': mean_pool,
        'mean_recommended': mean_rec,
        'bias': bias,
        'cohens_d': cohens_d,
        't_statistic': t_stat,
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def calculate_categorical_bias(pool_data: pd.Series, recommended_data: pd.Series) -> Dict:
    """
    Calculate bias metrics for categorical features.

    Returns:
        - distribution_pool: Category proportions in pool (dict)
        - distribution_recommended: Category proportions in recommended (dict)
        - cramers_v: Effect size
        - chi2_statistic: Chi-square statistic
        - p_value: Chi-square p-value
        - max_bias_category: Category with largest absolute bias
        - max_bias_value: Largest absolute bias value
    """
    # Remove NaN values
    pool_clean = pool_data.dropna()
    rec_clean = recommended_data.dropna()

    if len(pool_clean) == 0 or len(rec_clean) == 0:
        return {
            'distribution_pool': {},
            'distribution_recommended': {},
            'cramers_v': np.nan,
            'chi2_statistic': np.nan,
            'p_value': np.nan,
            'max_bias_category': None,
            'max_bias_value': np.nan
        }

    # Get distributions
    pool_dist = pool_clean.value_counts(normalize=True).to_dict()
    rec_dist = rec_clean.value_counts(normalize=True).to_dict()

    # Get all categories
    all_categories = set(pool_dist.keys()) | set(rec_dist.keys())

    # Create contingency table
    pool_counts = pool_clean.value_counts()
    rec_counts = rec_clean.value_counts()

    # Align categories
    categories = sorted(all_categories)
    observed_rec = [rec_counts.get(cat, 0) for cat in categories]
    observed_pool = [pool_counts.get(cat, 0) for cat in categories]

    # Chi-square test
    if len(categories) > 1 and sum(observed_rec) > 0 and sum(observed_pool) > 0:
        contingency_table = np.array([observed_pool, observed_rec])
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)

        # Cramér's V
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    else:
        chi2, p_val, cramers_v = np.nan, np.nan, np.nan

    # Find category with maximum bias
    biases = {}
    for cat in categories:
        pool_prop = pool_dist.get(cat, 0)
        rec_prop = rec_dist.get(cat, 0)
        biases[cat] = rec_prop - pool_prop

    if biases:
        max_bias_cat = max(biases.items(), key=lambda x: abs(x[1]))
        max_bias_category = max_bias_cat[0]
        max_bias_value = max_bias_cat[1]
    else:
        max_bias_category = None
        max_bias_value = np.nan

    return {
        'distribution_pool': pool_dist,
        'distribution_recommended': rec_dist,
        'cramers_v': cramers_v,
        'chi2_statistic': chi2,
        'p_value': p_val,
        'max_bias_category': max_bias_category,
        'max_bias_value': max_bias_value,
        'all_category_biases': biases
    }


def calculate_binary_bias(pool_data: pd.Series, recommended_data: pd.Series) -> Dict:
    """
    Calculate bias metrics for binary features.

    Returns:
        - proportion_pool: Proportion of True in pool
        - proportion_recommended: Proportion of True in recommended
        - bias: Difference (recommended - pool)
        - relative_risk: RR of being recommended given True
        - odds_ratio: OR of being recommended given True
        - p_value: Fisher's exact test p-value
    """
    # Remove NaN and convert to boolean
    pool_clean = pool_data.dropna().astype(bool)
    rec_clean = recommended_data.dropna().astype(bool)

    if len(pool_clean) == 0 or len(rec_clean) == 0:
        return {
            'proportion_pool': np.nan,
            'proportion_recommended': np.nan,
            'bias': np.nan,
            'relative_risk': np.nan,
            'odds_ratio': np.nan,
            'p_value': np.nan
        }

    # Proportions
    prop_pool = pool_clean.sum() / len(pool_clean)
    prop_rec = rec_clean.sum() / len(rec_clean)
    bias = prop_rec - prop_pool

    # Contingency table: [pool_false, pool_true], [rec_false, rec_true]
    pool_true = pool_clean.sum()
    pool_false = len(pool_clean) - pool_true
    rec_true = rec_clean.sum()
    rec_false = len(rec_clean) - rec_true

    # Relative risk
    if prop_pool > 0:
        relative_risk = prop_rec / prop_pool
    else:
        relative_risk = np.nan

    # Odds ratio and Fisher's exact test
    if pool_false > 0 and rec_false > 0 and pool_true > 0 and rec_true > 0:
        odds_ratio = (rec_true * pool_false) / (rec_false * pool_true)
        _, p_val = fisher_exact([[pool_false, pool_true], [rec_false, rec_true]])
    else:
        odds_ratio = np.nan
        p_val = np.nan

    return {
        'proportion_pool': prop_pool,
        'proportion_recommended': prop_rec,
        'bias': bias,
        'relative_risk': relative_risk,
        'odds_ratio': odds_ratio,
        'p_value': p_val
    }


# ============================================================================
# EXPERIMENT-LEVEL BIAS ANALYSIS
# ============================================================================

def analyze_experiment_bias(data_file: Path, output_dir: Path) -> pd.DataFrame:
    """
    Perform comprehensive bias analysis on a single experiment.

    Args:
        data_file: Path to post_level_data.csv or .parquet
        output_dir: Where to save analysis results

    Returns:
        DataFrame with bias metrics for all features × prompt styles
    """
    # Load data
    if data_file.suffix == '.parquet':
        df = pd.read_parquet(data_file)
    else:
        df = pd.read_csv(data_file)

    print(f"Loaded {len(df):,} records from {data_file.name}")

    # Get available features
    available_numerical = [f for f in NUMERICAL_FEATURES if f in df.columns]
    available_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    available_binary = [f for f in BINARY_FEATURES if f in df.columns]

    print(f"  Numerical features: {len(available_numerical)}")
    print(f"  Categorical features: {len(available_categorical)}")
    print(f"  Binary features: {len(available_binary)}")

    # Analyze by prompt style
    results = []

    for prompt_style in df['prompt_style'].unique():
        style_df = df[df['prompt_style'] == prompt_style]

        # Split into pool and recommended
        pool_df = style_df
        recommended_df = style_df[style_df['selected'] == 1]

        print(f"\n  Analyzing {prompt_style}: {len(pool_df)} pool, {len(recommended_df)} recommended")

        # Numerical features
        for feature in available_numerical:
            bias_metrics = calculate_numerical_bias(
                pool_df[feature],
                recommended_df[feature]
            )

            results.append({
                'prompt_style': prompt_style,
                'feature': feature,
                'feature_type': 'numerical',
                **bias_metrics
            })

        # Categorical features
        for feature in available_categorical:
            bias_metrics = calculate_categorical_bias(
                pool_df[feature],
                recommended_df[feature]
            )

            # Flatten the result
            results.append({
                'prompt_style': prompt_style,
                'feature': feature,
                'feature_type': 'categorical',
                'cramers_v': bias_metrics['cramers_v'],
                'chi2_statistic': bias_metrics['chi2_statistic'],
                'p_value': bias_metrics['p_value'],
                'max_bias_category': bias_metrics['max_bias_category'],
                'max_bias_value': bias_metrics['max_bias_value'],
                'distribution_pool': str(bias_metrics['distribution_pool']),
                'distribution_recommended': str(bias_metrics['distribution_recommended'])
            })

        # Binary features
        for feature in available_binary:
            bias_metrics = calculate_binary_bias(
                pool_df[feature],
                recommended_df[feature]
            )

            results.append({
                'prompt_style': prompt_style,
                'feature': feature,
                'feature_type': 'binary',
                **bias_metrics
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction for multiple testing
    if 'p_value' in results_df.columns:
        p_values = results_df['p_value'].fillna(1.0)
        reject, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        results_df['p_value_adjusted'] = p_adjusted
        results_df['significant_fdr'] = reject

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comprehensive_bias_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved comprehensive bias analysis to: {output_file}")

    # Summary statistics
    print(f"\n  Summary:")
    print(f"    Total comparisons: {len(results_df)}")
    if 'p_value_adjusted' in results_df.columns:
        n_significant = results_df['significant_fdr'].sum()
        print(f"    Significant after FDR correction: {n_significant} ({n_significant/len(results_df)*100:.1f}%)")

    return results_df


# ============================================================================
# AGGREGATION FRAMEWORK (8 LEVELS)
# ============================================================================

def aggregate_bias_results(experiment_results: Dict[str, pd.DataFrame],
                          output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Implement 8-level aggregation hierarchy from the plan.

    Levels:
        1. Fully disaggregated (dataset × model × prompt × feature)
        2. Aggregate across trials (dataset × model × prompt × feature)
        3. Aggregate across features (dataset × model × prompt)
        4. Aggregate across prompts (dataset × model)
        5. Aggregate across datasets + models (prompt only)
        6. Aggregate across datasets + prompts (model only)
        7. Aggregate across models + prompts (dataset only)
        8. Fully aggregated (overall)

    Args:
        experiment_results: Dict mapping experiment_name -> bias_df
        output_dir: Where to save aggregated results

    Returns:
        Dict mapping level_name -> aggregated_df
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse experiment metadata
    all_data = []
    for exp_name, bias_df in experiment_results.items():
        parts = exp_name.split('_')
        if len(parts) >= 3:
            dataset = parts[0]
            provider = parts[1]
            model = '_'.join(parts[2:])

            bias_df_copy = bias_df.copy()
            bias_df_copy['dataset'] = dataset
            bias_df_copy['provider'] = provider
            bias_df_copy['model'] = model
            bias_df_copy['experiment'] = exp_name

            all_data.append(bias_df_copy)

    if not all_data:
        print("No experiment results to aggregate")
        return {}

    # Combine all results
    combined_df = pd.concat(all_data, ignore_index=True)

    aggregated_results = {}

    # Level 1: Fully disaggregated (already have this)
    level1 = combined_df.copy()
    level1.to_csv(output_dir / "level1_fully_disaggregated.csv", index=False)
    aggregated_results['level1'] = level1
    print(f"✓ Level 1: {len(level1)} rows (fully disaggregated)")

    # Level 5: Aggregate across datasets + models (prompt style only)
    # Group by prompt_style and feature, average bias metrics
    level5 = combined_df.groupby(['prompt_style', 'feature', 'feature_type']).agg({
        'bias': 'mean',
        'cohens_d': 'mean',
        'cramers_v': 'mean',
        'relative_risk': 'mean',
        'p_value': lambda x: stats.combine_pvalues(x.dropna())[1] if len(x.dropna()) > 0 else np.nan
    }).reset_index()
    level5.to_csv(output_dir / "level5_by_prompt_style.csv", index=False)
    aggregated_results['level5'] = level5
    print(f"✓ Level 5: {len(level5)} rows (by prompt style)")

    # Level 6: Aggregate across datasets + prompts (model only)
    level6 = combined_df.groupby(['model', 'feature', 'feature_type']).agg({
        'bias': 'mean',
        'cohens_d': 'mean',
        'cramers_v': 'mean',
        'relative_risk': 'mean',
        'p_value': lambda x: stats.combine_pvalues(x.dropna())[1] if len(x.dropna()) > 0 else np.nan
    }).reset_index()
    level6.to_csv(output_dir / "level6_by_model.csv", index=False)
    aggregated_results['level6'] = level6
    print(f"✓ Level 6: {len(level6)} rows (by model)")

    # Level 7: Aggregate across models + prompts (dataset only)
    level7 = combined_df.groupby(['dataset', 'feature', 'feature_type']).agg({
        'bias': 'mean',
        'cohens_d': 'mean',
        'cramers_v': 'mean',
        'relative_risk': 'mean',
        'p_value': lambda x: stats.combine_pvalues(x.dropna())[1] if len(x.dropna()) > 0 else np.nan
    }).reset_index()
    level7.to_csv(output_dir / "level7_by_dataset.csv", index=False)
    aggregated_results['level7'] = level7
    print(f"✓ Level 7: {len(level7)} rows (by dataset)")

    # Level 8: Fully aggregated (overall)
    level8 = combined_df.groupby(['feature', 'feature_type']).agg({
        'bias': 'mean',
        'cohens_d': 'mean',
        'cramers_v': 'mean',
        'relative_risk': 'mean',
        'p_value': lambda x: stats.combine_pvalues(x.dropna())[1] if len(x.dropna()) > 0 else np.nan
    }).reset_index()
    level8.to_csv(output_dir / "level8_fully_aggregated.csv", index=False)
    aggregated_results['level8'] = level8
    print(f"✓ Level 8: {len(level8)} rows (fully aggregated)")

    return aggregated_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive bias analysis for LLM recommendation experiments'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Path to single experiment directory'
    )

    parser.add_argument(
        '--all-experiments',
        action='store_true',
        help='Analyze all experiments in outputs/experiments/'
    )

    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='outputs/experiments',
        help='Directory containing experiments'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/comprehensive_bias_analysis',
        help='Where to save analysis results'
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE BIAS ANALYSIS")
    print("="*80)

    experiment_results = {}

    if args.experiment:
        # Analyze single experiment
        exp_path = Path(args.experiment)
        exp_name = exp_path.name

        data_file = exp_path / "post_level_data.csv"
        if not data_file.exists():
            data_file = exp_path / "post_level_data.parquet"

        if not data_file.exists():
            print(f"ERROR: No data file found in {exp_path}")
            return

        output_dir = Path(args.output_dir) / exp_name
        bias_df = analyze_experiment_bias(data_file, output_dir)
        experiment_results[exp_name] = bias_df

    elif args.all_experiments:
        # Analyze all experiments
        experiments_dir = Path(args.experiments_dir)
        experiments = [d for d in experiments_dir.iterdir()
                      if d.is_dir() and (d / "post_level_data.csv").exists()
                      or (d / "post_level_data.parquet").exists()]

        print(f"\nFound {len(experiments)} experiments\n")

        for i, exp_path in enumerate(experiments, 1):
            exp_name = exp_path.name
            print(f"\n[{i}/{len(experiments)}] {exp_name}")
            print("-" * 80)

            data_file = exp_path / "post_level_data.csv"
            if not data_file.exists():
                data_file = exp_path / "post_level_data.parquet"

            output_dir = Path(args.output_dir) / exp_name

            try:
                bias_df = analyze_experiment_bias(data_file, output_dir)
                experiment_results[exp_name] = bias_df
            except Exception as e:
                print(f"  ✗ Error: {e}")

    else:
        print("ERROR: Must specify --experiment or --all-experiments")
        return

    # Aggregate results if multiple experiments
    if len(experiment_results) > 1:
        print("\n" + "="*80)
        print("AGGREGATING RESULTS ACROSS EXPERIMENTS")
        print("="*80 + "\n")

        agg_output_dir = Path(args.output_dir) / "aggregated"
        aggregated = aggregate_bias_results(experiment_results, agg_output_dir)

    print("\n" + "="*80)
    print("COMPREHENSIVE BIAS ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
