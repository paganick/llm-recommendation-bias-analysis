"""
Stratified Analysis Pipeline for LLM Recommendation Bias

This script performs stratified regression analysis where each prompt_style is analyzed separately,
then coefficients are compared across styles to understand how different prompts affect bias.

Key Functions:
1. run_stratified_regression() - Run logistic regression for one prompt style
2. compare_coefficients_pairwise() - Compare coefficients across styles using Wald test
3. compute_bias_per_style() - Compute pool vs recommended bias for each style
4. create_publication_regression_table() - Format results for publication
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import argparse
import json

# Statistical modeling
import statsmodels.api as sm
from statsmodels.formula.api import logit


# ============================================================================
# 16 CORE FEATURES (December 2025 refinement - removed 12 redundant features)
# ============================================================================

# Numerical Feature columns for regression (9 features)
FEATURE_COLUMNS = [
    # Text metrics (2)
    'text_length',
    'avg_word_length',

    # Sentiment (2)
    'sentiment_polarity',
    'sentiment_subjectivity',

    # Content (1)
    'polarization_score',

    # Toxicity (2)
    'toxicity',
    'severe_toxicity',

    # Style indicators (4) - Binary, treated as numerical (0/1) in regression
    'has_emoji',
    'has_hashtag',
    'has_mention',
    'has_url',
]

# Categorical feature columns (3 features)
# Note: These require dummy coding for regression
# controversy_level is handled as ordinal (low=0, medium=1, high=2)
CATEGORICAL_FEATURES = [
    'author_gender',            # female, male, non-binary, unknown
    'author_political_leaning', # left, center-left, center, center-right, right, apolitical, unknown
    'author_is_minority',       # yes, no, unknown
    'primary_topic',            # politics, sports, entertainment, technology, health, personal, other
    'controversy_level',        # low, medium, high (ordinal)
]


def run_stratified_regression(df: pd.DataFrame, prompt_style: str,
                              feature_cols: List[str] = None) -> Dict:
    """
    Run logistic regression for a single prompt style.

    Args:
        df: Full post-level DataFrame
        prompt_style: Which prompt style to analyze (e.g., 'general', 'popular')
        feature_cols: List of feature column names (default: FEATURE_COLUMNS)

    Returns:
        Dictionary with:
        - coefficients: Feature coefficients
        - std_errors: Standard errors
        - p_values: P-values
        - odds_ratios: Exp(coefficients)
        - conf_intervals: 95% confidence intervals for odds ratios
        - n_obs: Number of observations
        - n_selected: Number of selected posts
        - model: Fitted statsmodels result object
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Filter to this prompt style
    style_df = df[df['prompt_style'] == prompt_style].copy()

    # Prepare features
    X = style_df[feature_cols].copy()

    # Handle different data types
    for col in X.columns:
        # Handle boolean columns (convert to int)
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

        # Handle categorical/object columns
        elif X[col].dtype == 'object':
            # Try ordinal encoding for known ordinal variables
            if col == 'controversy_level':
                level_map = {'low': 0, 'medium': 1, 'high': 2}
                X[col] = X[col].map(level_map)
            else:
                # For other categorical, use dummy coding (drop first to avoid collinearity)
                print(f"  Warning: Dummy coding {col} (categorical)")
                # We'll skip this for now to keep it simple
                X = X.drop(columns=[col])
                continue

        # Handle missing values
        if X[col].isna().any():
            # Impute with median for numeric
            if X[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(0, inplace=True)

    # Add constant for intercept
    X = sm.add_constant(X)

    # Target variable
    y = style_df['selected']

    # Fit logistic regression
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception as e:
        print(f"  Warning: Regression failed for {prompt_style}: {e}")
        return None

    # Extract results
    results = {
        'prompt_style': prompt_style,
        'coefficients': model.params.to_dict(),
        'std_errors': model.bse.to_dict(),
        'p_values': model.pvalues.to_dict(),
        'odds_ratios': np.exp(model.params).to_dict(),
        'conf_intervals_lower': np.exp(model.conf_int()[0]).to_dict(),
        'conf_intervals_upper': np.exp(model.conf_int()[1]).to_dict(),
        'n_obs': len(style_df),
        'n_selected': style_df['selected'].sum(),
        'aic': model.aic,
        'bic': model.bic,
        'pseudo_r2': model.prsquared,
        'model': model  # Keep for further analysis
    }

    return results


def compare_coefficients_pairwise(style_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare coefficients across prompt styles using Wald test.

    Test: z = (β1 - β2) / sqrt(SE1² + SE2²)

    Args:
        style_results: Dict mapping prompt_style -> regression results

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - style1, style2: Prompt styles being compared
        - coef1, coef2: Coefficients for each style
        - diff: Difference in coefficients (style1 - style2)
        - z_stat: Z-statistic for difference
        - p_value: P-value for difference
        - significant: Boolean indicating significance at p<0.05
    """
    comparisons = []

    # Get all prompt styles
    styles = list(style_results.keys())

    # Get all features (excluding constant)
    features = [f for f in style_results[styles[0]]['coefficients'].keys()
                if f != 'const']

    # Pairwise comparisons
    for i, style1 in enumerate(styles):
        for style2 in styles[i+1:]:
            res1 = style_results[style1]
            res2 = style_results[style2]

            for feature in features:
                coef1 = res1['coefficients'][feature]
                coef2 = res2['coefficients'][feature]
                se1 = res1['std_errors'][feature]
                se2 = res2['std_errors'][feature]

                # Wald test for difference
                diff = coef1 - coef2
                se_diff = np.sqrt(se1**2 + se2**2)
                z_stat = diff / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                comparisons.append({
                    'feature': feature,
                    'style1': style1,
                    'style2': style2,
                    'coef1': coef1,
                    'coef2': coef2,
                    'diff': diff,
                    'z_stat': z_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

    return pd.DataFrame(comparisons)


def compute_bias_per_style(df: pd.DataFrame, prompt_style: str,
                           feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Compute bias (pool vs recommended) for each feature, for one prompt style.

    Args:
        df: Full post-level DataFrame
        prompt_style: Which prompt style to analyze
        feature_cols: Features to analyze (default: FEATURE_COLUMNS + CATEGORICAL_FEATURES)

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - pool_mean: Mean in pool
        - recommended_mean: Mean in recommended
        - difference: recommended - pool
        - chi2_stat: Chi-square statistic (for categorical) or t-statistic (for numeric)
        - p_value: P-value for test
        - significant: Boolean indicating significance
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES

    # Filter to this prompt style
    style_df = df[df['prompt_style'] == prompt_style].copy()

    bias_results = []

    for feature in feature_cols:
        pool_data = style_df[feature]
        recommended_data = style_df[style_df['selected'] == 1][feature]

        # Skip if feature is missing
        if pool_data.isna().all() or recommended_data.isna().all():
            continue

        # For categorical features, use mode and chi-square test
        if feature in CATEGORICAL_FEATURES or pool_data.dtype == 'object':
            # Get most common value
            pool_mode = pool_data.mode()[0] if len(pool_data.mode()) > 0 else None
            rec_mode = recommended_data.mode()[0] if len(recommended_data.mode()) > 0 else None

            # Chi-square test
            try:
                contingency = pd.crosstab(style_df['selected'], style_df[feature])
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                stat = chi2
            except:
                stat = np.nan
                p_value = np.nan

            bias_results.append({
                'feature': feature,
                'pool_value': str(pool_mode),
                'recommended_value': str(rec_mode),
                'difference': 'categorical',
                'test_stat': stat,
                'p_value': p_value,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            })

        # For numeric features, use mean and t-test
        else:
            pool_mean = pool_data.mean()
            rec_mean = recommended_data.mean()
            diff = rec_mean - pool_mean

            # T-test
            try:
                t_stat, p_value = stats.ttest_ind(
                    recommended_data.dropna(),
                    pool_data[style_df['selected'] == 0].dropna()
                )
            except:
                t_stat = np.nan
                p_value = np.nan

            bias_results.append({
                'feature': feature,
                'pool_mean': pool_mean,
                'recommended_mean': rec_mean,
                'difference': diff,
                'test_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            })

    return pd.DataFrame(bias_results)


def create_publication_regression_table(style_results: Dict[str, Dict],
                                        feature_order: List[str] = None) -> pd.DataFrame:
    """
    Create publication-ready regression table.

    Format: Feature | General | Popular | Engaging | Informative | Controversial | Neutral
    Each cell: "coef*** (SE)" with significance stars
    *** p<0.001, ** p<0.01, * p<0.05

    Args:
        style_results: Dict mapping prompt_style -> regression results
        feature_order: Order of features in table (default: alphabetical)

    Returns:
        Formatted DataFrame ready for export
    """
    # Get all features (excluding constant)
    all_features = set()
    for res in style_results.values():
        all_features.update([f for f in res['coefficients'].keys() if f != 'const'])

    if feature_order is None:
        feature_order = sorted(all_features)

    # Build table
    table_data = []

    for feature in feature_order:
        row = {'Feature': feature}

        for style, res in style_results.items():
            if feature in res['coefficients']:
                coef = res['coefficients'][feature]
                se = res['std_errors'][feature]
                p = res['p_values'][feature]

                # Significance stars
                if p < 0.001:
                    stars = '***'
                elif p < 0.01:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'
                else:
                    stars = ''

                # Format: "coef*** (SE)"
                cell = f"{coef:.4f}{stars} ({se:.4f})"
            else:
                cell = "—"

            row[style.capitalize()] = cell

        table_data.append(row)

    # Add model statistics at bottom
    for stat_name in ['n_obs', 'pseudo_r2', 'aic']:
        row = {'Feature': stat_name}
        for style, res in style_results.items():
            if stat_name == 'n_obs':
                row[style.capitalize()] = str(res['n_obs'])
            else:
                row[style.capitalize()] = f"{res[stat_name]:.4f}"
        table_data.append(row)

    return pd.DataFrame(table_data)


def analyze_experiment(experiment_dir: Path, output_dir: Path = None):
    """
    Run full stratified analysis pipeline for one experiment.

    Args:
        experiment_dir: Path to experiment directory (e.g., outputs/experiments/bluesky_anthropic_claude-3-5-haiku-20241022)
        output_dir: Where to save results (default: {experiment_dir}/stratified_analysis)
    """
    experiment_dir = Path(experiment_dir)

    if output_dir is None:
        output_dir = experiment_dir / "stratified_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "by_style").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Stratified Analysis: {experiment_dir.name}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading post-level data...")
    data_file = experiment_dir / "post_level_data.csv"
    if not data_file.exists():
        print(f"ERROR: {data_file} not found!")
        return

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} post-level records")
    print(f"Prompt styles: {df['prompt_style'].unique().tolist()}")
    print(f"Selected posts: {df['selected'].sum():,} ({df['selected'].mean()*100:.1f}%)")

    # Run regression for each prompt style
    print("\n" + "="*60)
    print("Running stratified regressions...")
    print("="*60 + "\n")

    style_results = {}

    for style in df['prompt_style'].unique():
        print(f"Analyzing prompt style: {style}")
        results = run_stratified_regression(df, style)

        if results is not None:
            style_results[style] = results

            # Save individual style results
            style_output = output_dir / "by_style" / f"{style}_regression.csv"

            # Create detailed results DataFrame
            style_df = pd.DataFrame({
                'feature': list(results['coefficients'].keys()),
                'coefficient': list(results['coefficients'].values()),
                'std_error': list(results['std_errors'].values()),
                'p_value': list(results['p_values'].values()),
                'odds_ratio': list(results['odds_ratios'].values()),
                'ci_lower': list(results['conf_intervals_lower'].values()),
                'ci_upper': list(results['conf_intervals_upper'].values())
            })
            style_df.to_csv(style_output, index=False)
            print(f"  Saved: {style_output}")
            print(f"  N={results['n_obs']:,}, Selected={results['n_selected']:,}, Pseudo-R²={results['pseudo_r2']:.4f}")
        else:
            print(f"  Failed to fit model for {style}")

    # Compare coefficients across styles
    print("\n" + "="*60)
    print("Comparing coefficients across styles...")
    print("="*60 + "\n")

    if len(style_results) < 2:
        print(f"ERROR: Need at least 2 successful regressions to compare, got {len(style_results)}")
        print("Skipping comparison and remaining analyses")
        return

    comparison_df = compare_coefficients_pairwise(style_results)
    comparison_output = output_dir / "comparison" / "coefficient_comparison.csv"
    comparison_df.to_csv(comparison_output, index=False)
    print(f"Saved: {comparison_output}")

    # Show significant differences
    sig_diffs = comparison_df[comparison_df['significant']]
    if len(sig_diffs) > 0:
        print(f"\nFound {len(sig_diffs)} significant differences (p<0.05):")
        print(sig_diffs[['feature', 'style1', 'style2', 'diff', 'p_value']].head(10))
    else:
        print("No significant differences found")

    # Compute bias per style
    print("\n" + "="*60)
    print("Computing bias (pool vs recommended) per style...")
    print("="*60 + "\n")

    all_bias_results = []

    for style in df['prompt_style'].unique():
        print(f"Computing bias for: {style}")
        bias_df = compute_bias_per_style(df, style)
        bias_df['prompt_style'] = style
        all_bias_results.append(bias_df)

        # Show significant biases
        sig_bias = bias_df[bias_df['significant']]
        if len(sig_bias) > 0:
            print(f"  {len(sig_bias)} features with significant bias")

    bias_output = output_dir / "comparison" / "bias_by_style.csv"
    pd.concat(all_bias_results, ignore_index=True).to_csv(bias_output, index=False)
    print(f"\nSaved: {bias_output}")

    # Create publication tables
    print("\n" + "="*60)
    print("Creating publication-ready tables...")
    print("="*60 + "\n")

    pub_table = create_publication_regression_table(style_results)
    pub_output = output_dir / "tables" / "regression_table_publication.csv"
    pub_table.to_csv(pub_output, index=False)
    print(f"Saved: {pub_output}")

    # Feature importance ranking
    print("\nRanking features by average |coefficient|...")
    feature_importance = []

    for feature in FEATURE_COLUMNS:
        coefs = [abs(res['coefficients'].get(feature, 0))
                for res in style_results.values()]
        avg_coef = np.mean(coefs)
        feature_importance.append({
            'feature': feature,
            'avg_abs_coefficient': avg_coef
        })

    importance_df = pd.DataFrame(feature_importance).sort_values(
        'avg_abs_coefficient', ascending=False
    )
    importance_output = output_dir / "comparison" / "feature_importance.csv"
    importance_df.to_csv(importance_output, index=False)
    print(f"Saved: {importance_output}")
    print("\nTop 10 most important features:")
    print(importance_df.head(10))

    print("\n" + "="*60)
    print(f"✓ Stratified analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run stratified analysis on LLM recommendation experiment'
    )
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        help='Path to experiment directory (e.g., outputs/experiments/bluesky_anthropic_claude-3-5-haiku-20241022)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Where to save results (default: {experiment-dir}/stratified_analysis)'
    )

    args = parser.parse_args()

    analyze_experiment(args.experiment_dir, args.output_dir)


if __name__ == '__main__':
    main()
