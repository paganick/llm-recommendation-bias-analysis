"""
Fix Issues with Bias Visualizations and Re-run Feature Importance

Issues addressed:
1. Bias heatmaps: Normalize Cohen's d values so text_length doesn't dominate
2. Feature importance: Re-run COMPLETE analysis for substantive-only features
3. Proper comparison between full and substantive-only analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("FIXING BIAS HEATMAPS AND RE-RUNNING FEATURE IMPORTANCE")
print("="*80)

# Configuration
OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'static'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 1: FIX BIAS HEATMAPS WITH NORMALIZED COHEN'S D
# ============================================================================

print("\n" + "="*80)
print("PART 1: CREATING NORMALIZED BIAS HEATMAPS")
print("="*80)

# Load bias results
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results_enhanced.parquet')

print(f"\nLoaded {len(bias_df)} bias measurements")
print(f"Cohen's d range: [{bias_df['cohens_d'].min():.3f}, {bias_df['cohens_d'].max():.3f}]")

# Create normalized heatmaps using normalized Cohen's d
def create_normalized_heatmap(data, title, filename, top_n=15):
    """Create heatmap using normalized Cohen's d values"""

    # Get top features by absolute normalized Cohen's d
    feature_importance = data.groupby('feature')['cohens_d_abs_normalized'].mean().sort_values(ascending=False)
    top_features = feature_importance.head(top_n).index.tolist()

    # Filter data
    plot_data = data[data['feature'].isin(top_features)].copy()

    # Create pivot table with normalized values
    pivot = plot_data.pivot_table(
        index='feature',
        columns=['dataset', 'model'],
        values='cohens_d_normalized',  # Using normalized values!
        aggfunc='mean'
    )

    # Reorder by importance
    pivot = pivot.reindex(top_features)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.5)))

    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0.5,  # Normalized values are 0-1, center at 0.5
        cbar_kws={'label': 'Normalized Effect Size'},
        ax=ax,
        linewidths=0.5,
        annot=False
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset × Model', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ {filename}")

# Create normalized heatmaps
create_normalized_heatmap(
    bias_df,
    'Bias Heatmap: All Features (Normalized Cohen\'s d)\nAll features now visually comparable',
    VIZ_DIR / 'normalized' / 'bias_heatmap_all_normalized.png',
    top_n=20
)

create_normalized_heatmap(
    bias_df[bias_df['feature_category'] == 'substantive'],
    'Bias Heatmap: Substantive Features Only (Normalized Cohen\'s d)',
    VIZ_DIR / 'substantive_only' / 'bias_heatmap_substantive_normalized.png',
    top_n=15
)

# ============================================================================
# PART 2: RE-RUN FEATURE IMPORTANCE FOR SUBSTANTIVE-ONLY
# ============================================================================

print("\n" + "="*80)
print("PART 2: RE-RUNNING FEATURE IMPORTANCE FOR SUBSTANTIVE-ONLY")
print("="*80)

# Feature definitions
STYLISTIC_FEATURES = [
    'text_length', 'word_count', 'avg_word_length',
    'has_emoji', 'has_hashtag', 'has_mention', 'has_url',
    'formality_score',
]

NUMERICAL_FEATURES_FULL = [
    'text_length', 'word_count', 'avg_word_length',
    'sentiment_polarity', 'sentiment_subjectivity',
    'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
    'formality_score', 'polarization_score',
    'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'
]

NUMERICAL_FEATURES_SUBSTANTIVE = [f for f in NUMERICAL_FEATURES_FULL if f not in STYLISTIC_FEATURES]

CATEGORICAL_FEATURES = [
    'sentiment_label', 'primary_topic',
    'author_gender', 'author_political_leaning', 'author_is_minority',
    'has_emoji', 'has_hashtag', 'has_mention', 'has_url', 'has_polarizing_content'
]

CATEGORICAL_FEATURES_SUBSTANTIVE = [f for f in CATEGORICAL_FEATURES if f not in STYLISTIC_FEATURES]

print(f"\nFull analysis features:")
print(f"  Numerical: {len(NUMERICAL_FEATURES_FULL)}")
print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")

print(f"\nSubstantive-only features:")
print(f"  Numerical: {len(NUMERICAL_FEATURES_SUBSTANTIVE)}")
print(f"  Categorical: {len(CATEGORICAL_FEATURES_SUBSTANTIVE)}")

def run_importance_analysis(numerical_features, categorical_features, output_suffix):
    """Run complete feature importance analysis"""

    results = []

    # Find all experiment directories
    EXPERIMENT_DIR = Path('outputs/experiments')
    exp_dirs = list(EXPERIMENT_DIR.glob('*'))

    datasets = ['twitter', 'bluesky', 'reddit']
    providers = ['openai', 'anthropic', 'gemini']
    prompt_styles = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

    total_conditions = len(datasets) * len(providers) * len(prompt_styles)
    current = 0

    for dataset in datasets:
        for provider in providers:
            # Find experiment directory
            exp_pattern = list(EXPERIMENT_DIR.glob(f"{dataset}_{provider}_*"))
            if not exp_pattern:
                continue

            exp_dir = exp_pattern[0]
            data_file = exp_dir / 'post_level_data.csv'

            if not data_file.exists():
                continue

            # Load data
            df = pd.read_csv(data_file)
            model_name = exp_dir.name.split('_', 2)[2]

            for prompt_style in prompt_styles:
                current += 1
                print(f"\r  Progress: {current}/{total_conditions} - {dataset}/{provider}/{prompt_style}    ", end='')

                # Filter for this prompt style
                prompt_data = df[df['prompt_style'] == prompt_style].copy()

                if len(prompt_data) < 50:
                    continue

                # Prepare features
                available_numerical = [f for f in numerical_features if f in prompt_data.columns]
                available_categorical = [f for f in categorical_features if f in prompt_data.columns]

                X = prompt_data[available_numerical].copy()

                # One-hot encode categoricals
                for cat_feat in available_categorical:
                    if cat_feat in prompt_data.columns:
                        dummies = pd.get_dummies(prompt_data[cat_feat], prefix=cat_feat, drop_first=True)
                        X = pd.concat([X, dummies], axis=1)

                # Fill NAs
                for col in X.columns:
                    if X[col].isna().any():
                        X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else 0, inplace=True)

                y = prompt_data['selected'].astype(int)

                # Check class balance
                if y.mean() == 0 or y.mean() == 1:
                    continue

                # Scale features (CRITICAL FOR PROPER COMPARISON)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train model
                clf = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs'
                )
                clf.fit(X_scaled, y)

                # Evaluate
                try:
                    auroc = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])
                except:
                    auroc = np.nan

                # SHAP analysis
                try:
                    explainer = shap.LinearExplainer(clf, X_scaled, feature_perturbation="interventional")
                    shap_values = explainer.shap_values(X_scaled)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Get positive class

                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                except:
                    mean_abs_shap = np.zeros(X.shape[1])

                # Store results
                result = {
                    'dataset': dataset,
                    'provider': provider,
                    'model': model_name,
                    'prompt_style': prompt_style,
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'auroc': auroc
                }

                # Add feature-level results
                for i, feat in enumerate(X.columns):
                    result[f'coef_{feat}'] = clf.coef_[0][i]
                    result[f'shap_{feat}'] = mean_abs_shap[i]

                results.append(result)

    print(f"\n  Completed {len(results)} models")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / 'importance_analysis' / f'importance_results_{output_suffix}.parquet'
    results_df.to_parquet(output_file)
    print(f"  ✓ Saved to: {output_file}")

    # Compute summary
    shap_cols = [c for c in results_df.columns if c.startswith('shap_')]
    shap_summary = []

    for col in shap_cols:
        feat_name = col[5:]
        shap_summary.append({
            'feature': feat_name,
            'mean_shap': results_df[col].abs().mean(),
            'std_shap': results_df[col].abs().std()
        })

    summary_df = pd.DataFrame(shap_summary).sort_values('mean_shap', ascending=False)
    summary_file = OUTPUT_DIR / 'importance_analysis' / f'shap_summary_{output_suffix}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ Summary saved to: {summary_file}")

    return results_df, summary_df

# Run substantive-only analysis
print("\nRunning substantive-only analysis...")
results_substantive, summary_substantive = run_importance_analysis(
    NUMERICAL_FEATURES_SUBSTANTIVE,
    CATEGORICAL_FEATURES_SUBSTANTIVE,
    'substantive_only'
)

print(f"\nSubstantive-only analysis complete:")
print(f"  Models trained: {len(results_substantive)}")
print(f"  Mean AUROC: {results_substantive['auroc'].mean():.3f}")

# ============================================================================
# PART 3: CREATE COMPARISON VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("PART 3: CREATING COMPARISON VISUALIZATIONS")
print("="*80)

# Load original (full) results
results_full = pd.read_parquet('analysis_outputs/importance_analysis/importance_results.parquet')
summary_full_file = Path('analysis_outputs/importance_analysis/importance_summary.txt')

# Create SHAP summary for full if doesn't exist
shap_cols_full = [c for c in results_full.columns if c.startswith('shap_') and c != 'shap_file']
shap_summary_full = []
for col in shap_cols_full:
    feat_name = col[5:]
    try:
        shap_summary_full.append({
            'feature': feat_name,
            'mean_shap': results_full[col].abs().mean()
        })
    except:
        continue
summary_full = pd.DataFrame(shap_summary_full).sort_values('mean_shap', ascending=False)

# Compare top features
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Panel A: Full analysis
top_full = summary_full.head(15)
axes[0].barh(range(len(top_full)), top_full['mean_shap'], color='steelblue')
axes[0].set_yticks(range(len(top_full)))
axes[0].set_yticklabels(top_full['feature'])
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean Absolute SHAP Value')
axes[0].set_title('A) Full Analysis (All Features)\nIncludes stylistic features', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Panel B: Substantive only
top_substantive = summary_substantive.head(15)
axes[1].barh(range(len(top_substantive)), top_substantive['mean_shap'], color='darkgreen')
axes[1].set_yticks(range(len(top_substantive)))
axes[1].set_yticklabels(top_substantive['feature'])
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean Absolute SHAP Value')
axes[1].set_title('B) Substantive-Only Analysis\nStylistic features removed', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'comparison' / 'full_vs_substantive_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ {VIZ_DIR / 'comparison' / 'full_vs_substantive_importance.png'}")

# Create AUROC comparison
fig, ax = plt.subplots(figsize=(10, 6))

auroc_data = pd.DataFrame({
    'Full Analysis': results_full.groupby(['dataset', 'model', 'prompt_style'])['auroc'].mean(),
    'Substantive Only': results_substantive.groupby(['dataset', 'model', 'prompt_style'])['auroc'].mean()
})

auroc_data.plot(kind='box', ax=ax)
ax.set_ylabel('AUROC Score')
ax.set_title('Prediction Performance: Full vs Substantive-Only Features', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'comparison' / 'auroc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ {VIZ_DIR / 'comparison' / 'auroc_comparison.png'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Files Created:

Bias Heatmaps (Normalized):
  ✓ {VIZ_DIR / 'normalized' / 'bias_heatmap_all_normalized.png'}
  ✓ {VIZ_DIR / 'substantive_only' / 'bias_heatmap_substantive_normalized.png'}

Feature Importance (Substantive-Only):
  ✓ {OUTPUT_DIR / 'importance_analysis' / 'importance_results_substantive_only.parquet'}
  ✓ {OUTPUT_DIR / 'importance_analysis' / 'shap_summary_substantive_only.csv'}

Comparisons:
  ✓ {VIZ_DIR / 'comparison' / 'full_vs_substantive_importance.png'}
  ✓ {VIZ_DIR / 'comparison' / 'auroc_comparison.png'}

Results:

Full Analysis:
  - Models: {len(results_full)}
  - Mean AUROC: {results_full['auroc'].mean():.3f}
  - Top feature: {summary_full.iloc[0]['feature']} (SHAP: {summary_full.iloc[0]['mean_shap']:.3f})

Substantive-Only Analysis:
  - Models: {len(results_substantive)}
  - Mean AUROC: {results_substantive['auroc'].mean():.3f}
  - Top feature: {summary_substantive.iloc[0]['feature']} (SHAP: {summary_substantive.iloc[0]['mean_shap']:.3f})
""")

print("="*80)
print("✓ FIXES COMPLETE")
print("="*80)
