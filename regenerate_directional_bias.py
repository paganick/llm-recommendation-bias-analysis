#!/usr/bin/env python3
"""
Regenerate Directional Bias Data

This script computes directional bias for each feature showing which categories/values are favored.

For categorical features: proportion_recommended - proportion_pool for each category
For continuous features: mean_recommended - mean_pool
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# 16 Core Features (grouped by category)
FEATURES = {
    'author': ['author_gender', 'author_political_leaning', 'author_is_minority'],
    'text_metrics': ['text_length', 'avg_word_length'],
    'sentiment': ['sentiment_polarity', 'sentiment_subjectivity'],
    'style': ['has_emoji', 'has_hashtag', 'has_mention', 'has_url'],
    'content': ['polarization_score', 'controversy_level', 'primary_topic'],
    'toxicity': ['toxicity', 'severe_toxicity']
}

# Feature types
FEATURE_TYPES = {
    # Author (categorical)
    'author_gender': 'categorical',
    'author_political_leaning': 'categorical',
    'author_is_minority': 'categorical',

    # Text metrics (numerical)
    'text_length': 'numerical',
    'avg_word_length': 'numerical',

    # Sentiment (numerical)
    'sentiment_polarity': 'numerical',
    'sentiment_subjectivity': 'numerical',

    # Style (binary)
    'has_emoji': 'binary',
    'has_hashtag': 'binary',
    'has_mention': 'binary',
    'has_url': 'binary',

    # Content (mixed)
    'polarization_score': 'numerical',
    'controversy_level': 'categorical',
    'primary_topic': 'categorical',

    # Toxicity (numerical)
    'toxicity': 'numerical',
    'severe_toxicity': 'numerical'
}

# Datasets and models
DATASETS = ['twitter', 'bluesky', 'reddit']
PROVIDERS = ['openai', 'anthropic', 'gemini']
PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

# Output directories
OUTPUT_DIR = Path('analysis_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_experiment_data(dataset, provider):
    """Load full experiment data"""
    # Find experiment directory - try both old and new naming patterns
    exp_dirs = list(Path('outputs/experiments').glob(f"{dataset}_{provider}_*"))
    if not exp_dirs:
        return None

    csv_path = exp_dirs[0] / 'post_level_data.csv'
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    return df

def load_all_experiment_data():
    """Load all available experiment data and infer dataset/provider from directory names"""
    all_data = []

    exp_base = Path('outputs/experiments')
    if not exp_base.exists():
        print("❌ No experiments directory found")
        return pd.DataFrame()

    # Find all experiment directories
    exp_dirs = [d for d in exp_base.iterdir() if d.is_dir()]

    for exp_dir in exp_dirs:
        csv_path = exp_dir / 'post_level_data.csv'
        if not csv_path.exists():
            continue

        # Parse directory name to extract dataset and provider
        dir_name = exp_dir.name
        parts = dir_name.split('_')

        if len(parts) >= 2:
            dataset = parts[0]  # e.g., 'survey', 'twitter', 'bluesky', 'reddit'
            provider = parts[1]  # e.g., 'gemini', 'openai', 'anthropic'

            df = pd.read_csv(csv_path)
            df['dataset'] = dataset
            df['provider'] = provider
            df['experiment_dir'] = dir_name

            all_data.append(df)
            print(f"  ✓ Loaded {len(df)} rows from {dir_name}")

    if not all_data:
        print("❌ No data found in any experiment directories")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total loaded: {len(combined_df)} rows from {len(all_data)} experiments")

    return combined_df

# ============================================================================
# DIRECTIONAL BIAS COMPUTATION
# ============================================================================

def compute_directional_bias():
    """
    Compute directional bias for each feature showing which categories/values are favored.

    For categorical features: proportion_recommended - proportion_pool for each category
    For continuous features: mean_recommended - mean_pool
    """
    print("\n" + "="*80)
    print("COMPUTING DIRECTIONAL BIAS")
    print("="*80)

    # Load all experiment data
    print("\nLoading experiment data...")
    df_all = load_all_experiment_data()

    if len(df_all) == 0:
        print("❌ No data loaded. Cannot compute directional bias.")
        return pd.DataFrame()

    all_directional_data = []
    all_features = sum(FEATURES.values(), [])

    # Get unique combinations of dataset, provider, prompt
    datasets = df_all['dataset'].unique()
    providers = df_all['provider'].unique()
    prompts = df_all['prompt_style'].unique()

    print(f"\nFound: {len(datasets)} datasets, {len(providers)} providers, {len(prompts)} prompt styles")
    print(f"Datasets: {datasets}")
    print(f"Providers: {providers}")
    print(f"Prompt styles: {prompts}")

    # Process each dataset × provider × prompt combination
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        for provider in providers:
            df_subset = df_all[(df_all['dataset'] == dataset) & (df_all['provider'] == provider)]

            if len(df_subset) == 0:
                continue

            print(f"  - {provider} ({len(df_subset)} rows)...")

            for prompt in prompts:
                prompt_df = df_subset[df_subset['prompt_style'] == prompt]

                # Split into pool and recommended
                pool = prompt_df[prompt_df['selected'] == 0].copy()
                recommended = prompt_df[prompt_df['selected'] == 1].copy()

                if len(pool) == 0 or len(recommended) == 0:
                    continue

                # Process each feature
                for feature in all_features:
                    if feature not in prompt_df.columns:
                        continue

                    feature_type = FEATURE_TYPES[feature]

                    if feature_type == 'categorical':
                        # Get all unique categories (excluding NaN for now)
                        all_categories = sorted(prompt_df[feature].dropna().unique())

                        # Check if there are any NaN values - if so, treat as "unknown"
                        has_nan = prompt_df[feature].isna().any()

                        for category in all_categories:
                            # Compute proportions
                            prop_pool = (pool[feature] == category).sum() / len(pool) if len(pool) > 0 else 0
                            prop_rec = (recommended[feature] == category).sum() / len(recommended) if len(recommended) > 0 else 0

                            # Directional bias: positive = over-represented, negative = under-represented
                            dir_bias = prop_rec - prop_pool

                            all_directional_data.append({
                                'feature': feature,
                                'category': str(category),
                                'dataset': dataset,
                                'provider': provider,
                                'prompt_style': prompt,
                                'directional_bias': dir_bias,
                                'prop_pool': prop_pool,
                                'prop_recommended': prop_rec,
                                'feature_type': 'categorical'
                            })

                        # Add "unknown" category for NaN values
                        if has_nan:
                            prop_pool_nan = pool[feature].isna().sum() / len(pool) if len(pool) > 0 else 0
                            prop_rec_nan = recommended[feature].isna().sum() / len(recommended) if len(recommended) > 0 else 0
                            dir_bias_nan = prop_rec_nan - prop_pool_nan

                            all_directional_data.append({
                                'feature': feature,
                                'category': 'unknown',
                                'dataset': dataset,
                                'provider': provider,
                                'prompt_style': prompt,
                                'directional_bias': dir_bias_nan,
                                'prop_pool': prop_pool_nan,
                                'prop_recommended': prop_rec_nan,
                                'feature_type': 'categorical'
                            })

                    else:  # continuous (numerical or binary)
                        # Compute mean difference
                        mean_pool = pool[feature].mean()
                        mean_rec = recommended[feature].mean()
                        dir_bias = mean_rec - mean_pool

                        # Also compute standardized difference for reference
                        std_pool = pool[feature].std()
                        std_diff = dir_bias / std_pool if std_pool > 0 else 0

                        all_directional_data.append({
                            'feature': feature,
                            'category': 'mean_difference',  # For continuous, single "category"
                            'dataset': dataset,
                            'provider': provider,
                            'prompt_style': prompt,
                            'directional_bias': dir_bias,
                            'mean_pool': mean_pool,
                            'mean_recommended': mean_rec,
                            'std_diff': std_diff,
                            'feature_type': 'continuous'
                        })

    df_directional = pd.DataFrame(all_directional_data)
    print(f"\n✓ Computed directional bias for {len(df_directional)} feature-category-condition combinations")

    return df_directional

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DIRECTIONAL BIAS DATA REGENERATION")
    print("="*80)

    # Compute directional bias
    df_directional = compute_directional_bias()

    if len(df_directional) == 0:
        print("\n❌ No directional bias data computed. Exiting.")
        exit(1)

    # Save to parquet (if possible)
    try:
        parquet_path = OUTPUT_DIR / 'directional_bias_data.parquet'
        df_directional.to_parquet(parquet_path, index=False)
        print(f"\n✓ Saved directional bias data to {parquet_path}")
    except ImportError as e:
        print(f"\n⚠ Could not save as parquet (missing pyarrow): {e}")
        print("  Skipping parquet format, saving as CSV only...")

    # Save to CSV for easy inspection
    csv_path = OUTPUT_DIR / 'directional_bias_data.csv'
    df_directional.to_csv(csv_path, index=False)
    print(f"✓ Saved directional bias data to {csv_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total rows: {len(df_directional):,}")
    print(f"\nBy feature type:")
    print(df_directional['feature_type'].value_counts())
    print(f"\nBy dataset:")
    print(df_directional['dataset'].value_counts())
    print(f"\nBy provider:")
    print(df_directional['provider'].value_counts())
    print(f"\nBy prompt style:")
    print(df_directional['prompt_style'].value_counts())

    print(f"\nUnique features: {df_directional['feature'].nunique()}")
    print(df_directional['feature'].unique())

    print("\n✓ Done!")
