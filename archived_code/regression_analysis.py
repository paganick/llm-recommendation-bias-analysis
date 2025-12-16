"""
Regression Analysis for Understanding Bias Sources

This module performs regression analysis to disentangle direct biases from confounding factors.

Example confounders:
- Gender might correlate with topics (e.g., men talk more about sports/tech)
- Political leaning might correlate with sentiment
- Age might correlate with formality/emoji usage

Goal: Understand if LLM selection is driven by:
1. Direct demographic bias (e.g., prefers male authors)
2. Topic preferences (e.g., prefers tech content, which men write more)
3. Style preferences (e.g., prefers formal writing, which older users use)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# For regression
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn/statsmodels not installed. Install with: pip install scikit-learn statsmodels")


def load_results_with_pools(results_dir: Path) -> Tuple[List[Dict], Dict]:
    """Load experiment results and config."""

    results_path = results_dir / 'prompt_style_results.pkl'
    config_path = results_dir / 'config.pkl'

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    return results, config


def create_selection_dataset(results: List[Dict],
                             dataset_name: str,
                             metadata_cache_path: str = None) -> pd.DataFrame:
    """
    Create a dataset for regression analysis.

    Each row represents a post in a pool, with:
    - selected: 1 if recommended, 0 otherwise
    - prompt_style: which prompt style was used
    - All metadata features (sentiment, topic, style, etc.)

    This allows us to model: P(selected | features, prompt_style)
    """

    print('='*80)
    print('CREATING SELECTION DATASET FOR REGRESSION')
    print('='*80)
    print()

    # Load metadata cache to get full post data
    if metadata_cache_path is None:
        metadata_cache_path = f'./outputs/metadata_cache/{dataset_name}_metadata.pkl'

    metadata_cache = pd.read_pickle(metadata_cache_path)
    print(f'Loaded metadata cache: {len(metadata_cache)} posts')

    # Build dataset
    all_rows = []

    for trial_idx, result in enumerate(results):
        prompt_style = result['prompt_style']
        trial_id = result['trial_id']

        # Get pool indices (we need to reconstruct this from trial info)
        # Since we used consistent seeds in the experiment, we can reconstruct
        seed = 1000 + trial_id
        pool_size = result['pool_size']

        # Sample same pool
        pool_sample = metadata_cache.sample(n=pool_size, random_state=seed)
        pool_indices = pool_sample.index.tolist()

        # Determine which were selected (this is tricky - we don't have exact indices)
        # We'll approximate by matching based on result statistics
        # For now, mark top-k as selected based on pool characteristics

        # Actually, we need to track this differently. Let me restructure.
        # For each post in pool, mark if it was in recommendations

        # This is approximate - in production, we'd want to track exact indices
        for idx in pool_indices:
            row = {
                'trial_id': trial_id,
                'prompt_style': prompt_style,
                'selected': 0,  # Will mark selected ones below
                'pool_index': idx
            }

            # Add all metadata features
            post_data = metadata_cache.loc[idx]
            for col in metadata_cache.columns:
                if col not in ['text', 'message']:  # Skip text columns
                    row[col] = post_data[col]

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    print(f'Created dataset: {len(df)} observations')
    print(f'  Trials: {df["trial_id"].nunique()}')
    print(f'  Prompt styles: {df["prompt_style"].nunique()}')
    print()

    return df


def prepare_regression_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for regression.

    Returns:
        - DataFrame with encoded features
        - List of feature names
    """

    print('Preparing regression features...')

    # Numerical features
    numerical_features = [
        'sentiment_polarity',
        'sentiment_subjectivity',
        'polarization_score',
        'formality_score',
        'word_count',
        'avg_word_length'
    ]

    # Binary features
    binary_features = [
        'has_emoji',
        'has_hashtag',
        'has_mention',
        'has_url'
    ]

    # Categorical features (one-hot encode)
    categorical_features = {
        'sentiment_label': ['positive', 'negative', 'neutral'],
        'primary_topic': df['primary_topic'].unique().tolist() if 'primary_topic' in df.columns else []
    }

    # Create feature matrix
    X = pd.DataFrame()
    feature_names = []

    # Add numerical features
    for feat in numerical_features:
        if feat in df.columns:
            X[feat] = df[feat].fillna(0)
            feature_names.append(feat)

    # Add binary features
    for feat in binary_features:
        if feat in df.columns:
            X[feat] = df[feat].astype(int)
            feature_names.append(feat)

    # Add categorical features (one-hot encoded)
    for feat, values in categorical_features.items():
        if feat in df.columns:
            for val in values:
                col_name = f'{feat}_{val}'
                X[col_name] = (df[feat] == val).astype(int)
                feature_names.append(col_name)

    # Add prompt style dummies
    for style in df['prompt_style'].unique():
        col_name = f'prompt_{style}'
        X[col_name] = (df['prompt_style'] == style).astype(int)
        feature_names.append(col_name)

    print(f'  Created {len(feature_names)} features')
    print(f'  Numerical: {len(numerical_features)}')
    print(f'  Binary: {len(binary_features)}')
    print(f'  Categorical: {sum(len(v) for v in categorical_features.values())}')
    print(f'  Prompt styles: {len(df["prompt_style"].unique())}')
    print()

    return X, feature_names


def run_logistic_regression(X: pd.DataFrame, y: pd.Series,
                            feature_names: List[str]) -> Dict[str, Any]:
    """
    Run logistic regression to predict selection.

    Model: P(selected = 1 | features)

    This tells us which features drive LLM selection after controlling for confounders.
    """

    print('='*80)
    print('LOGISTIC REGRESSION ANALYSIS')
    print('='*80)
    print()

    if not HAS_SKLEARN:
        print('ERROR: sklearn/statsmodels not installed')
        return {}

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Fit logistic regression with statsmodels (gives p-values)
    X_with_const = sm.add_constant(X_scaled_df)

    try:
        model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=100)

        # Get results
        results = {
            'coefficients': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'pseudo_r2': model.prsquared,
            'aic': model.aic,
            'bic': model.bic
        }

        # Print summary
        print('Model Performance:')
        print(f'  Pseudo R²: {model.prsquared:.4f}')
        print(f'  AIC: {model.aic:.2f}')
        print(f'  BIC: {model.bic:.2f}')
        print()

        # Print significant coefficients
        print('Significant Features (p < 0.05):')
        print('='*80)
        print(f'{"Feature":<30} {"Coefficient":>12} {"Odds Ratio":>12} {"p-value":>10}')
        print('-'*80)

        sig_features = []
        for feat in feature_names:
            if feat in model.params.index:
                coef = model.params[feat]
                pval = model.pvalues[feat]

                if pval < 0.05:
                    odds_ratio = np.exp(coef)
                    print(f'{feat:<30} {coef:>12.4f} {odds_ratio:>12.4f} {pval:>10.4f}')
                    sig_features.append({
                        'feature': feat,
                        'coefficient': coef,
                        'odds_ratio': odds_ratio,
                        'pvalue': pval
                    })

        print()
        print(f'Found {len(sig_features)} significant features')
        print()

        results['significant_features'] = sig_features

        # Interpretation
        print('Interpretation:')
        print('-'*80)
        print('Positive coefficients → feature increases selection probability')
        print('Negative coefficients → feature decreases selection probability')
        print('Odds ratio > 1 → increases odds of selection')
        print('Odds ratio < 1 → decreases odds of selection')
        print()

        return results

    except Exception as e:
        print(f'Error fitting model: {e}')
        return {}


def analyze_confounding(X: pd.DataFrame, y: pd.Series,
                       feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze feature correlations to identify potential confounders.

    Example: If sentiment_polarity and primary_topic are highly correlated,
    then topic preferences might confound sentiment analysis.
    """

    print('='*80)
    print('CONFOUNDING ANALYSIS')
    print('='*80)
    print()

    # Compute correlation matrix
    corr_matrix = X.corr()

    # Find high correlations (|r| > 0.3)
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            feat1 = feature_names[i]
            feat2 = feature_names[j]

            if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                corr = corr_matrix.loc[feat1, feat2]

                if abs(corr) > 0.3:
                    high_corr.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'correlation': corr
                    })

    if high_corr:
        print('High Feature Correlations (potential confounders):')
        print('='*80)

        df_corr = pd.DataFrame(high_corr)
        df_corr = df_corr.sort_values('correlation', key=abs, ascending=False)

        for _, row in df_corr.head(20).iterrows():
            print(f'{row["feature1"]:<30} <-> {row["feature2"]:<30} r={row["correlation"]:>6.3f}')

        print()
        print(f'Found {len(high_corr)} high correlations')
        print()
    else:
        print('No high correlations found (|r| > 0.3)')
        print()

    return pd.DataFrame(high_corr) if high_corr else pd.DataFrame()


def compare_models(X: pd.DataFrame, y: pd.Series,
                  feature_names: List[str]) -> Dict[str, Any]:
    """
    Compare different models to understand feature importance.

    1. Base model: Only content features (sentiment, topic, style)
    2. Full model: Content + prompt style
    3. Interaction model: Content + prompt + interactions
    """

    print('='*80)
    print('MODEL COMPARISON')
    print('='*80)
    print()

    if not HAS_SKLEARN:
        return {}

    # Define feature groups
    content_features = [f for f in feature_names
                       if not f.startswith('prompt_')]
    prompt_features = [f for f in feature_names
                      if f.startswith('prompt_')]

    results = {}

    # Model 1: Content only
    print('Model 1: Content Features Only')
    X_content = X[content_features]
    X_content = sm.add_constant(X_content)
    try:
        model1 = sm.Logit(y, X_content).fit(disp=0, maxiter=100)
        results['content_only'] = {
            'pseudo_r2': model1.prsquared,
            'aic': model1.aic,
            'bic': model1.bic
        }
        print(f'  Pseudo R²: {model1.prsquared:.4f}')
        print(f'  AIC: {model1.aic:.2f}')
        print()
    except:
        print('  Failed to fit')
        print()

    # Model 2: Content + Prompt
    print('Model 2: Content + Prompt Style')
    X_full = X[content_features + prompt_features]
    X_full = sm.add_constant(X_full)
    try:
        model2 = sm.Logit(y, X_full).fit(disp=0, maxiter=100)
        results['content_and_prompt'] = {
            'pseudo_r2': model2.prsquared,
            'aic': model2.aic,
            'bic': model2.bic
        }
        print(f'  Pseudo R²: {model2.prsquared:.4f}')
        print(f'  AIC: {model2.aic:.2f}')
        print()

        # Likelihood ratio test
        if 'content_only' in results:
            lr_stat = -2 * (model1.llf - model2.llf)
            df = len(prompt_features)
            p_value = 1 - scipy_stats.chi2.cdf(lr_stat, df)
            print(f'Likelihood Ratio Test:')
            print(f'  LR statistic: {lr_stat:.2f}')
            print(f'  p-value: {p_value:.4f}')
            print(f'  → Prompt style {"significantly" if p_value < 0.05 else "not significantly"} improves model')
            print()
    except:
        print('  Failed to fit')
        print()

    return results


def main():
    """Main analysis pipeline."""

    parser = argparse.ArgumentParser(description='Regression analysis of experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Dataset name (twitter, reddit, bluesky)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_name = args.dataset_name

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    print()
    print('='*80)
    print('REGRESSION ANALYSIS: UNDERSTANDING BIAS SOURCES')
    print('='*80)
    print()
    print(f'Dataset: {dataset_name}')
    print(f'Results: {results_dir}')
    print()

    # Load results
    results, config = load_results_with_pools(results_dir)
    print(f'Loaded {len(results)} trial results')
    print()

    # Create selection dataset
    # Note: This is a simplified version. In production, we'd need to track
    # exact post indices during the experiment to know which were selected.
    # For now, we'll analyze aggregate statistics.

    print('Note: Full regression analysis requires tracking individual post selections.')
    print('      This will be implemented in future experiment runs.')
    print()
    print('For now, we can analyze aggregate bias patterns from the existing results.')
    print()

    # Save placeholder
    output_dir = results_dir
    analysis_path = output_dir / 'regression_analysis.txt'

    with open(analysis_path, 'w') as f:
        f.write('Regression Analysis\n')
        f.write('='*80 + '\n\n')
        f.write('This analysis requires tracking individual post selections during experiments.\n')
        f.write('Future experiments will include this tracking.\n\n')
        f.write('To enable regression analysis:\n')
        f.write('1. Modify run_experiment.py to save post indices\n')
        f.write('2. Re-run experiments with tracking enabled\n')
        f.write('3. Run this analysis again\n')

    print(f'Analysis notes saved to: {analysis_path}')
    print()


if __name__ == '__main__':
    main()
