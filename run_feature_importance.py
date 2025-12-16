"""
Feature Importance Analysis for LLM Recommendation Bias
Implements logistic regression and SHAP analysis for all conditions
Following: LLM_Recommendation_Bias_Analysis_Plan (1).md Section 11.5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXPERIMENT_DIR = Path('outputs/experiments')
OUTPUT_DIR = Path('analysis_outputs/importance_analysis')
SHAP_DIR = OUTPUT_DIR / 'shap_values'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configuration
DATASETS = ['twitter', 'bluesky', 'reddit']
PROVIDERS = ['openai', 'anthropic', 'gemini']
PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

# ============================================================================
# 16 CORE FEATURES (December 2025 refinement - removed 12 redundant features)
# ============================================================================
#
# REMOVED FEATURES (highly correlated/redundant):
#   - sentiment_neutral (perfect correlation with subjectivity)
#   - sentiment_label (discretized polarity)
#   - sentiment_negative, sentiment_positive (components of polarity)
#   - word_count (nearly identical to text_length, r=0.996)
#   - formality_score (derived from avg_word_length)
#   - primary_topic_score (confidence score, not needed)
#   - has_polarizing_content (binary threshold of polarization_score)
#   - obscene, insult, threat, identity_attack (highly correlated with toxicity)
#
# ============================================================================

# Numerical Features (9 features)
NUMERICAL_FEATURES = [
    # Text metrics (2)
    'text_length',
    'avg_word_length',

    # Sentiment (2)
    'sentiment_polarity',      # -1 to 1
    'sentiment_subjectivity',  # 0 to 1

    # Content (1)
    'polarization_score',      # 0 to 1

    # Toxicity (2)
    'toxicity',                # Overall toxicity score
    'severe_toxicity',         # Extreme cases only
]

# Categorical Features (7 features)
# Note: Binary features (has_emoji, etc.) are treated as categorical
#       for proper one-hot encoding in ML models
CATEGORICAL_FEATURES = [
    # Author demographics (3)
    'author_gender',           # female, male, non-binary, unknown
    'author_political_leaning', # left, center-left, center, center-right, right, apolitical, unknown
    'author_is_minority',      # yes, no, unknown

    # Content (2)
    'primary_topic',           # politics, sports, entertainment, technology, health, personal, other
    'controversy_level',       # low, medium, high

    # Style indicators (4) - Binary features treated as categorical
    'has_emoji',               # 0, 1
    'has_hashtag',             # 0, 1
    'has_mention',             # 0, 1
    'has_url',                 # 0, 1
]


def load_experiment_data(dataset: str, provider: str) -> tuple:
    """Load data for a single experiment"""
    # Find the experiment directory
    exp_dirs = list(EXPERIMENT_DIR.glob(f"{dataset}_{provider}_*"))
    if not exp_dirs:
        return None, None

    exp_dir = exp_dirs[0]
    data_file = exp_dir / 'post_level_data.csv'

    if not data_file.exists():
        return None, None

    # Extract model name from directory
    model_name = exp_dir.name.split('_', 2)[2]  # e.g., 'gpt-4o-mini' from 'twitter_openai_gpt-4o-mini'

    df = pd.read_csv(data_file)
    df['dataset'] = dataset
    df['provider'] = provider

    return df, model_name


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix and target variable"""
    # Select available features
    available_numerical = [f for f in NUMERICAL_FEATURES if f in df.columns]
    available_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    # Start with numerical features
    X = df[available_numerical].copy()

    # One-hot encode categorical features
    for cat_feat in available_categorical:
        if cat_feat in df.columns:
            dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    # Fill missing values with median/mode
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else X[col].mode()[0],
                         inplace=True)

    # Target variable
    y = df['selected'].astype(int)

    # Feature names
    feature_names = X.columns.tolist()

    return X, y, feature_names


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, feature_names: list,
                       dataset: str, provider: str, model: str, prompt_style: str) -> dict:
    """Train logistic regression and compute SHAP values"""

    # Check class balance
    pos_rate = y.mean()
    if pos_rate == 0 or pos_rate == 1:
        return None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression with balanced class weights
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    try:
        clf.fit(X_scaled, y)

        # Predictions
        y_pred_proba = clf.predict_proba(X_scaled)[:, 1]

        # Compute metrics
        auroc = roc_auc_score(y, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')

        # Feature importances (coefficients)
        coefficients = clf.coef_[0]

        # SHAP values
        explainer = shap.LinearExplainer(clf, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

        # Save SHAP values
        shap_file = SHAP_DIR / f"{dataset}_{provider}_{prompt_style}_shap.npy"
        np.save(shap_file, shap_values)

        # Aggregate SHAP importance
        shap_importance = np.abs(shap_values).mean(axis=0)

        # Create results dictionary
        result = {
            'dataset': dataset,
            'provider': provider,
            'model': model,
            'prompt_style': prompt_style,
            'auroc': auroc,
            'cv_auroc_mean': cv_scores.mean(),
            'cv_auroc_std': cv_scores.std(),
            'n_samples': len(y),
            'n_features': len(feature_names),
            'pos_rate': pos_rate,
            'shap_file': str(shap_file)
        }

        # Add feature coefficients and SHAP importance
        for i, feat in enumerate(feature_names):
            result[f'coef_{feat}'] = coefficients[i]
            result[f'shap_{feat}'] = shap_importance[i]

        return result

    except Exception as e:
        print(f"  ⚠ Error training model: {e}")
        return None


def main():
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print()

    print("Loading data and training models for all conditions...")
    print()

    results = []
    total_conditions = len(DATASETS) * len(PROVIDERS) * len(PROMPT_STYLES)
    completed = 0

    for dataset in DATASETS:
        for provider in PROVIDERS:
            # Load experiment data
            df, model = load_experiment_data(dataset, provider)

            if df is None:
                print(f"  ⚠ No data found for {dataset}_{provider}")
                completed += len(PROMPT_STYLES)
                continue

            print(f"\n{dataset.upper()} × {model}")
            print("-" * 80)

            for prompt_style in PROMPT_STYLES:
                # Filter to this prompt style
                prompt_df = df[df['prompt_style'] == prompt_style].copy()

                if len(prompt_df) == 0:
                    completed += 1
                    continue

                print(f"  Training: {prompt_style:15s} ", end='', flush=True)

                # Prepare features
                X, y, feature_names = prepare_features(prompt_df)

                # Train and evaluate
                result = train_and_evaluate(X, y, feature_names, dataset, provider,
                                           model, prompt_style)

                if result:
                    results.append(result)
                    print(f"✓ AUROC: {result['auroc']:.3f}")
                else:
                    print("✗ Failed")

                completed += 1

                # Progress update
                if completed % 10 == 0:
                    print(f"\n  Progress: {completed}/{total_conditions} ({100*completed/total_conditions:.1f}%)")

    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save full results
    output_file = OUTPUT_DIR / 'importance_results.parquet'
    results_df.to_parquet(output_file, index=False)
    print(f"✓ Saved full results: {output_file}")

    # Save summary statistics
    summary_file = OUTPUT_DIR / 'importance_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS - SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total models trained: {len(results_df)}\n")
        f.write(f"Mean AUROC: {results_df['auroc'].mean():.3f} ± {results_df['auroc'].std():.3f}\n")
        f.write(f"Median AUROC: {results_df['auroc'].median():.3f}\n")
        f.write(f"Min AUROC: {results_df['auroc'].min():.3f}\n")
        f.write(f"Max AUROC: {results_df['auroc'].max():.3f}\n")
        f.write(f"Models with AUROC > 0.6: {(results_df['auroc'] > 0.6).sum()}/{len(results_df)} "
                f"({100*(results_df['auroc'] > 0.6).mean():.1f}%)\n")
        f.write("\n")

        f.write("AUROC by Dataset:\n")
        f.write("-" * 80 + "\n")
        for dataset in DATASETS:
            dataset_df = results_df[results_df['dataset'] == dataset]
            if len(dataset_df) > 0:
                f.write(f"  {dataset:10s}: {dataset_df['auroc'].mean():.3f} ± {dataset_df['auroc'].std():.3f}\n")
        f.write("\n")

        f.write("AUROC by Model:\n")
        f.write("-" * 80 + "\n")
        for model in results_df['model'].unique():
            model_df = results_df[results_df['model'] == model]
            model_name = model.split('/')[-1]
            f.write(f"  {model_name:30s}: {model_df['auroc'].mean():.3f} ± {model_df['auroc'].std():.3f}\n")
        f.write("\n")

        f.write("AUROC by Prompt Style:\n")
        f.write("-" * 80 + "\n")
        for prompt in PROMPT_STYLES:
            prompt_df = results_df[results_df['prompt_style'] == prompt]
            if len(prompt_df) > 0:
                f.write(f"  {prompt:15s}: {prompt_df['auroc'].mean():.3f} ± {prompt_df['auroc'].std():.3f}\n")

    print(f"✓ Saved summary: {summary_file}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Full results: importance_results.parquet")
    print(f"  - Summary: importance_summary.txt")
    print(f"  - SHAP values: shap_values/*.npy ({len(results_df)} files)")
    print()
    print("Next steps:")
    print("  1. Generate importance visualizations")
    print("  2. Analyze feature rankings across conditions")
    print("  3. Identify consistent vs condition-specific predictors")


if __name__ == '__main__':
    main()
