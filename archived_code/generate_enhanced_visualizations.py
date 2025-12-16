"""
Generate Enhanced Visualizations with Normalization and Substantive-Only Views
Addresses Issues 2 & 3 from the analysis refinement plan

Generates three parallel visualization tracks:
1. Raw effect sizes (original)
2. Normalized effect sizes (all features visible)
3. Substantive-only (excluding stylistic features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'static'

# Create directories
(VIZ_DIR / 'normalized').mkdir(parents=True, exist_ok=True)
(VIZ_DIR / 'substantive_only').mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING ENHANCED VISUALIZATIONS")
print("="*80)

# Load enhanced data
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results_enhanced.parquet')
substantive_bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results_substantive_only.parquet')
importance_df = pd.read_parquet('analysis_outputs/importance_analysis/importance_results_enhanced.parquet')
shap_summary_df = pd.read_csv('analysis_outputs/importance_analysis/shap_summary_with_normalization.csv')

print(f"\nLoaded data:")
print(f"  Bias measurements: {len(bias_df)}")
print(f"  Substantive bias measurements: {len(substantive_bias_df)}")
print(f"  Importance models: {len(importance_df)}")
print(f"  SHAP features: {len(shap_summary_df)}")

# ============================================================================
# 1. Feature Importance Heatmaps: Raw vs Normalized
# ============================================================================

print("\n" + "="*80)
print("1. FEATURE IMPORTANCE HEATMAPS")
print("="*80)

def create_importance_heatmap(data, value_col, title, filename, top_n=15):
    """Create feature importance heatmap"""
    # Get top features
    top_features = data.nlargest(top_n, value_col)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bar plot
    colors = ['#d62728' if cat == 'stylistic' else '#2ca02c'
              for cat in top_features['feature_category']]

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features[value_col], color=colors)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Stylistic'),
        Patch(facecolor='#2ca02c', label='Substantive')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")

# Raw importance
create_importance_heatmap(
    shap_summary_df,
    'mean_shap_raw',
    'Feature Importance Rankings (Raw SHAP Values)',
    VIZ_DIR / 'importance_rankings' / 'feature_importance_raw.png'
)

# Normalized importance
create_importance_heatmap(
    shap_summary_df.sort_values('mean_shap_normalized', ascending=False),
    'mean_shap_normalized',
    'Feature Importance Rankings (Normalized SHAP Values)',
    VIZ_DIR / 'normalized' / 'feature_importance_normalized.png'
)

# Substantive only
substantive_shap = shap_summary_df[shap_summary_df['feature_category'] == 'substantive']
create_importance_heatmap(
    substantive_shap,
    'mean_shap_raw',
    'Feature Importance Rankings (Substantive Features Only)',
    VIZ_DIR / 'substantive_only' / 'feature_importance_substantive.png'
)

# ============================================================================
# 2. Effect Size Comparison: Raw vs Normalized
# ============================================================================

print("\n" + "="*80)
print("2. EFFECT SIZE COMPARISON")
print("="*80)

# Create comparison plot showing top features by raw and normalized
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel A: Raw SHAP values
top_raw = shap_summary_df.nlargest(12, 'mean_shap_raw')
colors = ['#d62728' if cat == 'stylistic' else '#2ca02c' for cat in top_raw['feature_category']]
axes[0].barh(range(len(top_raw)), top_raw['mean_shap_raw'], color=colors)
axes[0].set_yticks(range(len(top_raw)))
axes[0].set_yticklabels(top_raw['feature'])
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean Absolute SHAP Value (Raw)')
axes[0].set_title('A) Raw SHAP Values\n(Stylistic features dominate)', fontsize=12, fontweight='bold')

# Panel B: Normalized SHAP values
top_norm = shap_summary_df.nlargest(12, 'mean_shap_normalized')
colors = ['#d62728' if cat == 'stylistic' else '#2ca02c' for cat in top_norm['feature_category']]
axes[1].barh(range(len(top_norm)), top_norm['mean_shap_normalized'], color=colors)
axes[1].set_yticks(range(len(top_norm)))
axes[1].set_yticklabels(top_norm['feature'])
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean Absolute SHAP Value (Normalized)')
axes[1].set_title('B) Normalized SHAP Values\n(Substantive features visible)', fontsize=12, fontweight='bold')

# Shared legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='Stylistic'),
    Patch(facecolor='#2ca02c', label='Substantive')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(VIZ_DIR / 'normalized' / 'raw_vs_normalized_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ {VIZ_DIR / 'normalized' / 'raw_vs_normalized_comparison.png'}")

# ============================================================================
# 3. Bias Heatmaps by Feature Category
# ============================================================================

print("\n" + "="*80)
print("3. BIAS HEATMAPS BY CATEGORY")
print("="*80)

def create_category_heatmap(data, category, value_col, title, filename):
    """Create heatmap for a feature category"""
    # Filter by category
    cat_data = data[data['feature_category'] == category].copy()

    if len(cat_data) == 0:
        print(f"  ⚠ No data for category: {category}")
        return

    # Aggregate by feature, dataset, model
    pivot_data = cat_data.groupby(['feature', 'dataset', 'model'])[value_col].mean().reset_index()
    pivot_table = pivot_data.pivot_table(
        index='feature',
        columns=['dataset', 'model'],
        values=value_col
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot_table) * 0.4)))

    sns.heatmap(
        pivot_table,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Effect Size'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset × Model', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")

# Stylistic features - raw
create_category_heatmap(
    bias_df,
    'stylistic',
    'cohens_d',
    'Bias Heatmap: Stylistic Features (Raw Cohen\'s d)',
    VIZ_DIR / 'cross_cutting' / 'bias_heatmap_stylistic_raw.png'
)

# Substantive features - raw
create_category_heatmap(
    bias_df,
    'substantive',
    'cohens_d',
    'Bias Heatmap: Substantive Features (Raw Cohen\'s d)',
    VIZ_DIR / 'cross_cutting' / 'bias_heatmap_substantive_raw.png'
)

# Substantive features - normalized
create_category_heatmap(
    bias_df,
    'substantive',
    'cohens_d_normalized',
    'Bias Heatmap: Substantive Features (Normalized Cohen\'s d)',
    VIZ_DIR / 'normalized' / 'bias_heatmap_substantive_normalized.png'
)

# ============================================================================
# 4. Fairness Feature Spotlight
# ============================================================================

print("\n" + "="*80)
print("4. FAIRNESS FEATURE SPOTLIGHT")
print("="*80)

# Identify fairness features
fairness_features = ['author_gender', 'author_political_leaning', 'author_is_minority']
fairness_bias = bias_df[bias_df['feature'].isin(fairness_features)].copy()

# Check which fairness features actually have data
available_fairness = [f for f in fairness_features if f in fairness_bias['feature'].values]

if len(available_fairness) > 0:
    # Create detailed fairness heatmap
    fig, axes = plt.subplots(len(available_fairness), 1, figsize=(12, 4 * len(available_fairness)))

    if len(available_fairness) == 1:
        axes = [axes]

    plot_idx = 0
    for feature in available_fairness:
        feat_data = fairness_bias[fairness_bias['feature'] == feature]

        if len(feat_data) == 0:
            continue

        # Pivot by dataset × model × prompt_style
        pivot = feat_data.pivot_table(
            index='prompt_style',
            columns=['dataset', 'model'],
            values='cohens_d',
            aggfunc='mean'
        )

        # Only plot if pivot has data
        if pivot.empty or pivot.shape[0] == 0:
            continue

        ax = axes[plot_idx] if len(available_fairness) > 1 else axes

        sns.heatmap(
            pivot,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Cohen\'s d'},
            ax=ax,
            linewidths=0.5,
            annot=True,
            fmt='.2f'
        )

        ax.set_title(f'Bias in {feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset × Model')
        ax.set_ylabel('Prompt Style')

        plot_idx += 1

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'substantive_only' / 'fairness_features_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {VIZ_DIR / 'substantive_only' / 'fairness_features_detailed.png'}")
else:
    print("  ⚠ No fairness feature data available")

# ============================================================================
# 5. Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Visualizations Generated:

Normalized Track ({VIZ_DIR / 'normalized'}):
  ✓ feature_importance_normalized.png
  ✓ raw_vs_normalized_comparison.png
  ✓ bias_heatmap_substantive_normalized.png

Substantive-Only Track ({VIZ_DIR / 'substantive_only'}):
  ✓ feature_importance_substantive.png
  ✓ fairness_features_detailed.png

Category Analysis ({VIZ_DIR / 'cross_cutting'}):
  ✓ bias_heatmap_stylistic_raw.png
  ✓ bias_heatmap_substantive_raw.png

Key Insights:
  - Normalization reveals {len(substantive_shap)} substantive features
  - Top raw feature: {shap_summary_df.iloc[0]['feature']} ({shap_summary_df.iloc[0]['feature_category']})
  - Top normalized feature: {shap_summary_df.sort_values('mean_shap_normalized', ascending=False).iloc[0]['feature']} ({shap_summary_df.sort_values('mean_shap_normalized', ascending=False).iloc[0]['feature_category']})
  - Fairness features analyzed: {len(fairness_bias) if len(fairness_bias) > 0 else 'None found'}
""")

print("="*80)
print("✓ VISUALIZATION GENERATION COMPLETE")
print("="*80)
