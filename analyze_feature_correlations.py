#!/usr/bin/env python3
"""
Feature Correlation Analysis for Mixed-Type Features
Handles numerical, binary, and categorical features appropriately
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = Path("analysis_outputs/feature_analysis/correlations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment path (using one dataset as they're similar)
EXPERIMENT_PATH = 'outputs/experiments/bluesky_anthropic_claude-sonnet-4-5-20250929'

def load_data():
    """Load unique posts from one experiment"""
    df = pd.read_csv(f"{EXPERIMENT_PATH}/post_level_data.csv")
    unique_posts = df.drop_duplicates(subset='original_index').copy()
    return unique_posts

def cramers_v(x, y):
    """Compute Cramér's V for categorical-categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))

def correlation_ratio(categories, values):
    """
    Compute correlation ratio (eta) for categorical-numerical association
    Eta-squared measures how much of variance in numerical variable is explained by categorical
    """
    categories = pd.Series(categories).dropna()
    values = pd.Series(values).dropna()

    # Align indices
    common_idx = categories.index.intersection(values.index)
    categories = categories[common_idx]
    values = values[common_idx]

    if len(categories) == 0 or len(values) == 0:
        return 0

    # Overall mean
    overall_mean = values.mean()

    # Group means and sizes
    groups = values.groupby(categories)

    # Between-group variance
    ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for _, group in groups)

    # Total variance
    ss_total = sum((values - overall_mean)**2)

    if ss_total == 0:
        return 0

    # Eta-squared
    eta_squared = ss_between / ss_total

    # Return eta (square root)
    return np.sqrt(eta_squared)

def compute_correlation(df, feat1, feat2, type1, type2):
    """
    Compute appropriate correlation measure based on feature types
    Returns: correlation value between 0 and 1
    """
    # Get non-null values
    mask = df[feat1].notna() & df[feat2].notna()
    x = df.loc[mask, feat1]
    y = df.loc[mask, feat2]

    if len(x) < 10:  # Not enough data
        return 0

    try:
        # Numerical - Numerical: Pearson correlation (absolute value)
        if type1 == 'numerical' and type2 == 'numerical':
            x_num = pd.to_numeric(x, errors='coerce').dropna()
            y_num = pd.to_numeric(y, errors='coerce').dropna()
            common_idx = x_num.index.intersection(y_num.index)
            if len(common_idx) < 10:
                return 0
            corr, _ = stats.pearsonr(x_num[common_idx], y_num[common_idx])
            return abs(corr)

        # Binary - Binary: Phi coefficient (same as Pearson for binary)
        elif type1 == 'binary' and type2 == 'binary':
            x_bin = pd.to_numeric(x, errors='coerce').dropna()
            y_bin = pd.to_numeric(y, errors='coerce').dropna()
            common_idx = x_bin.index.intersection(y_bin.index)
            if len(common_idx) < 10:
                return 0
            corr, _ = stats.pearsonr(x_bin[common_idx], y_bin[common_idx])
            return abs(corr)

        # Numerical - Binary: Point-biserial correlation
        elif (type1 == 'numerical' and type2 == 'binary') or (type1 == 'binary' and type2 == 'numerical'):
            if type1 == 'binary':
                x, y = y, x  # Swap so y is binary
            x_num = pd.to_numeric(x, errors='coerce').dropna()
            y_bin = pd.to_numeric(y, errors='coerce').dropna()
            common_idx = x_num.index.intersection(y_bin.index)
            if len(common_idx) < 10:
                return 0
            corr, _ = stats.pearsonr(x_num[common_idx], y_bin[common_idx])
            return abs(corr)

        # Categorical - Categorical: Cramér's V
        elif type1 == 'categorical' and type2 == 'categorical':
            return cramers_v(x, y)

        # Numerical - Categorical: Correlation ratio
        elif (type1 == 'numerical' and type2 == 'categorical') or (type1 == 'categorical' and type2 == 'numerical'):
            if type1 == 'categorical':
                x, y = y, x  # Swap so x is numerical, y is categorical
            x_num = pd.to_numeric(x, errors='coerce')
            return correlation_ratio(y, x_num)

        # Binary - Categorical: Treat binary as categorical
        elif (type1 == 'binary' and type2 == 'categorical') or (type1 == 'categorical' and type2 == 'binary'):
            return cramers_v(x.astype(str), y.astype(str))

        else:
            return 0

    except Exception as e:
        print(f"Error computing correlation between {feat1} and {feat2}: {e}")
        return 0

def define_features():
    """
    Define the 16 core features to analyze and their types.

    Updated December 2025: Removed 12 redundant/highly-correlated features.

    Feature type categories:
    - 'numerical': Continuous or count variables
    - 'categorical': Discrete categories (including binary 0/1 variables)
    - 'binary': Special case of categorical with exactly 2 values

    Note: Binary features (has_emoji, etc.) are treated separately from
    multi-category categorical features in some analyses.
    """

    features = {
        # ====================================================================
        # AUTHOR DEMOGRAPHICS (3 features)
        # ====================================================================
        'author_gender': 'categorical',
        'author_political_leaning': 'categorical',
        'author_is_minority': 'categorical',  # yes, no, unknown (treated as categorical)

        # ====================================================================
        # TEXT METRICS (2 features)
        # ====================================================================
        'text_length': 'numerical',
        'avg_word_length': 'numerical',
        # REMOVED: word_count (r=0.996 with text_length)
        # REMOVED: formality_score (derived from avg_word_length)

        # ====================================================================
        # SENTIMENT (2 features)
        # ====================================================================
        'sentiment_polarity': 'numerical',      # -1 to 1
        'sentiment_subjectivity': 'numerical',  # 0 to 1
        # REMOVED: sentiment_label (discretized polarity)
        # REMOVED: sentiment_positive, sentiment_negative (components of polarity)
        # REMOVED: sentiment_neutral (perfect correlation with subjectivity)

        # ====================================================================
        # STYLE INDICATORS (4 binary features)
        # ====================================================================
        'has_emoji': 'binary',
        'has_hashtag': 'binary',
        'has_mention': 'binary',
        'has_url': 'binary',

        # ====================================================================
        # CONTENT (2 features)
        # ====================================================================
        'polarization_score': 'numerical',      # 0 to 1
        'controversy_level': 'categorical',     # low, medium, high
        # REMOVED: has_polarizing_content (binary threshold of polarization_score)

        # ====================================================================
        # TOPICS (1 feature)
        # ====================================================================
        'primary_topic': 'categorical',
        # REMOVED: primary_topic_score (confidence, not interpretable)

        # ====================================================================
        # TOXICITY (2 features)
        # ====================================================================
        'toxicity': 'numerical',
        'severe_toxicity': 'numerical',
        # REMOVED: obscene, insult, threat, identity_attack (all r>0.8 with toxicity)
    }

    return features

def compute_correlation_matrix(df, features):
    """Compute full correlation matrix with appropriate measures"""

    print("Computing correlation matrix...")
    feature_names = list(features.keys())
    n_features = len(feature_names)

    # Initialize matrix
    corr_matrix = np.zeros((n_features, n_features))

    # Compute correlations
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:  # Only compute upper triangle
                type1 = features[feat1]
                type2 = features[feat2]
                corr = compute_correlation(df, feat1, feat2, type1, type2)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetric

                if (i * n_features + j) % 100 == 0:
                    print(f"  Progress: {i*n_features + j}/{n_features*n_features}")

    return pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)

def identify_redundant_features(corr_matrix, threshold=0.8):
    """Identify highly correlated feature pairs"""

    redundant_pairs = []

    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr = corr_matrix.iloc[i, j]
            if corr >= threshold:
                feat1 = corr_matrix.index[i]
                feat2 = corr_matrix.columns[j]
                redundant_pairs.append((feat1, feat2, corr))

    # Sort by correlation
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)

    return redundant_pairs

def plot_correlation_heatmap(corr_matrix, output_path):
    """Plot correlation heatmap"""

    fig, ax = plt.subplots(figsize=(20, 18))

    # Create mask for upper triangle (optional, for cleaner look)
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=False,  # Don't annotate all cells (too many)
        fmt='.2f',
        cmap='RdYlGn_r',  # Red = high correlation, Green = low
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Strength'},
        ax=ax
    )

    ax.set_title('Feature Correlation Matrix\n(Mixed correlation measures: Pearson, Cramér\'s V, Eta)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}")

def plot_high_correlations(redundant_pairs, output_path, threshold=0.8):
    """Plot only highly correlated pairs for easier viewing"""

    if len(redundant_pairs) == 0:
        print(f"No feature pairs with correlation >= {threshold}")
        return

    # Create a smaller correlation matrix with only high correlations
    features = set()
    for feat1, feat2, _ in redundant_pairs:
        features.add(feat1)
        features.add(feat2)

    features = sorted(list(features))
    n = len(features)

    # Build subset matrix
    subset_corr = np.zeros((n, n))
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i == j:
                subset_corr[i, j] = 1.0
            else:
                # Find correlation from pairs
                for f1, f2, corr in redundant_pairs:
                    if (f1 == feat1 and f2 == feat2) or (f1 == feat2 and f2 == feat1):
                        subset_corr[i, j] = corr
                        break

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        subset_corr,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation'},
        xticklabels=features,
        yticklabels=features,
        ax=ax
    )

    ax.set_title(f'Highly Correlated Features (r >= {threshold})',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"High correlation plot saved to: {output_path}")

def main():
    """Main analysis pipeline"""

    print("="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df):,} unique posts")

    # Define features
    features = define_features()
    print(f"\nAnalyzing {len(features)} features")

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(df, features)

    # Save correlation matrix
    corr_matrix.to_csv(OUTPUT_DIR / 'correlation_matrix.csv')
    print(f"\nCorrelation matrix saved to: {OUTPUT_DIR / 'correlation_matrix.csv'}")

    # Plot full heatmap
    plot_correlation_heatmap(corr_matrix, OUTPUT_DIR / 'correlation_heatmap_full.png')

    # Identify redundant features
    print("\n" + "="*80)
    print("HIGHLY CORRELATED FEATURES (r >= 0.8)")
    print("="*80)

    redundant_pairs = identify_redundant_features(corr_matrix, threshold=0.8)

    if len(redundant_pairs) > 0:
        print(f"\nFound {len(redundant_pairs)} highly correlated pairs:\n")
        for feat1, feat2, corr in redundant_pairs:
            print(f"  {feat1:30s} <-> {feat2:30s} : {corr:.3f}")

        # Plot high correlations
        plot_high_correlations(redundant_pairs, OUTPUT_DIR / 'high_correlations.png', threshold=0.8)
    else:
        print("\nNo feature pairs with correlation >= 0.8")

    # Also check moderate correlations (0.6-0.8)
    print("\n" + "="*80)
    print("MODERATELY CORRELATED FEATURES (0.6 <= r < 0.8)")
    print("="*80)

    moderate_pairs = [p for p in identify_redundant_features(corr_matrix, threshold=0.6)
                      if p[2] < 0.8]

    if len(moderate_pairs) > 0:
        print(f"\nFound {len(moderate_pairs)} moderately correlated pairs:\n")
        for feat1, feat2, corr in moderate_pairs[:20]:  # Show top 20
            print(f"  {feat1:30s} <-> {feat2:30s} : {corr:.3f}")
        if len(moderate_pairs) > 20:
            print(f"  ... and {len(moderate_pairs) - 20} more")

    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FEATURE PRUNING")
    print("="*80)

    if len(redundant_pairs) > 0:
        print("\nHighly redundant features (consider removing one from each pair):")
        for feat1, feat2, corr in redundant_pairs:
            print(f"\n  {feat1} <-> {feat2} (r={corr:.3f})")
            print(f"    → Recommendation: Keep {feat1}, remove {feat2}")
            print(f"    → Rationale: [You should decide based on interpretability/importance]")
    else:
        print("\n✓ No highly redundant features found!")
        print("  All features have correlation < 0.8")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  - correlation_matrix.csv")
    print("  - correlation_heatmap_full.png")
    if len(redundant_pairs) > 0:
        print("  - high_correlations.png")

if __name__ == '__main__':
    main()
