"""
Comprehensive Fairness Feature Analysis - 8-Level Aggregation Hierarchy
Issue 4 from the analysis refinement plan

For fairness-critical features (gender, political leaning, minorities),
this script computes bias at ALL aggregation levels to understand:
- What persists across all conditions (robust bias)
- What's context-specific (conditional bias)
- Which combinations are most problematic

8-Level Aggregation Hierarchy:
Level 1: dataset × model × prompt (54 conditions) - Fully disaggregated
Level 2: dataset × model (9 conditions) - Aggregate over prompts
Level 3: dataset × prompt (18 conditions) - Aggregate over models
Level 4: model × prompt (18 conditions) - Aggregate over datasets
Level 5: dataset (3 conditions) - Aggregate over models & prompts
Level 6: model (3 conditions) - Aggregate over datasets & prompts
Level 7: prompt (6 conditions) - Aggregate over datasets & models
Level 8: overall (1 condition) - Fully aggregated
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("FAIRNESS FEATURE DEEP DIVE - 8-LEVEL AGGREGATION ANALYSIS")
print("="*80)

# Configuration
OUTPUT_DIR = Path('analysis_outputs')
FAIRNESS_DIR = OUTPUT_DIR / 'fairness_analysis'
FAIRNESS_VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'static' / 'fairness_deep_dive'
FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
FAIRNESS_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Load enhanced bias data
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results_enhanced.parquet')

# ============================================================================
# Identify Fairness Features
# ============================================================================

print("\nIdentifying fairness-critical features...")

# Core fairness features
FAIRNESS_FEATURES = ['author_gender', 'author_political_leaning', 'author_is_minority']

# Filter for fairness features only
fairness_data = bias_df[bias_df['feature'].isin(FAIRNESS_FEATURES)].copy()

print(f"\nFairness features found: {fairness_data['feature'].unique()}")
print(f"Total measurements: {len(fairness_data)}")
print(f"Datasets: {sorted(fairness_data['dataset'].unique())}")
print(f"Models: {sorted(fairness_data['model'].unique())}")
print(f"Prompts: {sorted(fairness_data['prompt_style'].unique())}")

# ============================================================================
# 8-Level Aggregation Hierarchy
# ============================================================================

print("\n" + "="*80)
print("COMPUTING 8-LEVEL AGGREGATION HIERARCHY")
print("="*80)

aggregation_results = []

# Level 1: Fully disaggregated (dataset × model × prompt)
print("\nLevel 1: dataset × model × prompt (Fully disaggregated)")
level1 = fairness_data.groupby(['dataset', 'model', 'prompt_style', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level1.columns = ['_'.join(col).strip('_') for col in level1.columns.values]
level1['aggregation_level'] = 'L1_dataset_model_prompt'
level1['n_conditions'] = len(fairness_data.groupby(['dataset', 'model', 'prompt_style']))
print(f"  Conditions: {level1['aggregation_level'].value_counts().sum()}")
aggregation_results.append(level1)

# Level 2: dataset × model (aggregate over prompts)
print("\nLevel 2: dataset × model (Aggregate over prompts)")
level2 = fairness_data.groupby(['dataset', 'model', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level2.columns = ['_'.join(col).strip('_') for col in level2.columns.values]
level2['aggregation_level'] = 'L2_dataset_model'
level2['n_conditions'] = len(fairness_data.groupby(['dataset', 'model']))
print(f"  Conditions: {len(level2)}")
aggregation_results.append(level2)

# Level 3: dataset × prompt (aggregate over models)
print("\nLevel 3: dataset × prompt (Aggregate over models)")
level3 = fairness_data.groupby(['dataset', 'prompt_style', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level3.columns = ['_'.join(col).strip('_') for col in level3.columns.values]
level3['aggregation_level'] = 'L3_dataset_prompt'
level3['n_conditions'] = len(fairness_data.groupby(['dataset', 'prompt_style']))
print(f"  Conditions: {len(level3)}")
aggregation_results.append(level3)

# Level 4: model × prompt (aggregate over datasets)
print("\nLevel 4: model × prompt (Aggregate over datasets)")
level4 = fairness_data.groupby(['model', 'prompt_style', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level4.columns = ['_'.join(col).strip('_') for col in level4.columns.values]
level4['aggregation_level'] = 'L4_model_prompt'
level4['n_conditions'] = len(fairness_data.groupby(['model', 'prompt_style']))
print(f"  Conditions: {len(level4)}")
aggregation_results.append(level4)

# Level 5: dataset only (aggregate over models & prompts)
print("\nLevel 5: dataset (Aggregate over models & prompts)")
level5 = fairness_data.groupby(['dataset', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level5.columns = ['_'.join(col).strip('_') for col in level5.columns.values]
level5['aggregation_level'] = 'L5_dataset'
level5['n_conditions'] = len(fairness_data['dataset'].unique())
print(f"  Conditions: {len(level5)}")
aggregation_results.append(level5)

# Level 6: model only (aggregate over datasets & prompts)
print("\nLevel 6: model (Aggregate over datasets & prompts)")
level6 = fairness_data.groupby(['model', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level6.columns = ['_'.join(col).strip('_') for col in level6.columns.values]
level6['aggregation_level'] = 'L6_model'
level6['n_conditions'] = len(fairness_data['model'].unique())
print(f"  Conditions: {len(level6)}")
aggregation_results.append(level6)

# Level 7: prompt only (aggregate over datasets & models)
print("\nLevel 7: prompt (Aggregate over datasets & models)")
level7 = fairness_data.groupby(['prompt_style', 'feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level7.columns = ['_'.join(col).strip('_') for col in level7.columns.values]
level7['aggregation_level'] = 'L7_prompt'
level7['n_conditions'] = len(fairness_data['prompt_style'].unique())
print(f"  Conditions: {len(level7)}")
aggregation_results.append(level7)

# Level 8: overall (fully aggregated)
print("\nLevel 8: overall (Fully aggregated)")
level8 = fairness_data.groupby(['feature']).agg({
    'cohens_d': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level8.columns = ['_'.join(col).strip('_') for col in level8.columns.values]
level8['aggregation_level'] = 'L8_overall'
level8['n_conditions'] = 1
print(f"  Conditions: {len(level8)}")
aggregation_results.append(level8)

# Save aggregation results
all_levels = pd.concat(aggregation_results, ignore_index=True)
all_levels_file = FAIRNESS_DIR / 'fairness_8level_aggregation.parquet'
all_levels.to_parquet(all_levels_file)
print(f"\n✓ Saved all aggregation levels to: {all_levels_file}")

# ============================================================================
# Robustness Analysis
# ============================================================================

print("\n" + "="*80)
print("ROBUSTNESS ANALYSIS")
print("="*80)

print("\nLevel 8 (Overall) - Most robust estimates:")
print("-" * 60)
for _, row in level8.iterrows():
    sig_pct = row['significant_mean'] * 100
    print(f"{row['feature']:30s}: Cohen's d = {row['cohens_d_mean']:6.3f} ± {row['cohens_d_std']:5.3f}")
    print(f"{'':30s}  Significant in {sig_pct:.1f}% of conditions")

# ============================================================================
# Visualization 1: Aggregation Level Comparison
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

print("\n1. Aggregation Level Comparison")

# Extract mean effect sizes across all levels for each feature
fairness_features_present = level8['feature'].unique()

fig, axes = plt.subplots(len(fairness_features_present), 1,
                          figsize=(14, 5 * len(fairness_features_present)))

if len(fairness_features_present) == 1:
    axes = [axes]

for idx, feature in enumerate(fairness_features_present):
    ax = axes[idx]

    # Gather data for this feature across all levels
    plot_data = []
    for level_df, level_name in [
        (level1, 'L1: D×M×P'),
        (level2, 'L2: D×M'),
        (level3, 'L3: D×P'),
        (level4, 'L4: M×P'),
        (level5, 'L5: D'),
        (level6, 'L6: M'),
        (level7, 'L7: P'),
        (level8, 'L8: All')
    ]:
        feat_data = level_df[level_df['feature'] == feature]
        if len(feat_data) > 0:
            plot_data.append({
                'level': level_name,
                'mean': feat_data['cohens_d_mean'].mean(),
                'std': feat_data['cohens_d_std'].mean() if 'cohens_d_std' in feat_data.columns else 0,
                'n': len(feat_data)
            })

    plot_df = pd.DataFrame(plot_data)

    # Bar plot with error bars
    x_pos = np.arange(len(plot_df))
    ax.bar(x_pos, plot_df['mean'], yerr=plot_df['std'],
           capsize=5, alpha=0.7, color='steelblue')

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df['level'], rotation=45, ha='right')
    ax.set_ylabel('Cohen\'s d (Mean ± SD)')
    ax.set_title(f'Bias in {feature} Across Aggregation Levels',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add sample sizes as text
    for i, row in plot_df.iterrows():
        ax.text(i, row['mean'] + row['std'] + 0.02, f"n={row['n']}",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FAIRNESS_VIZ_DIR / 'aggregation_level_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ {FAIRNESS_VIZ_DIR / 'aggregation_level_comparison.png'}")

# ============================================================================
# Visualization 2: Heatmaps for Each Aggregation Level
# ============================================================================

print("\n2. Heatmaps for Key Aggregation Levels")

# Level 5: Dataset
if len(level5) > 0:
    pivot5 = level5.pivot(index='feature', columns='dataset', values='cohens_d_mean')

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot5) * 0.8)))
    sns.heatmap(pivot5, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Cohen\'s d'}, ax=ax, linewidths=1)
    ax.set_title('Fairness Bias by Dataset (Aggregated over models & prompts)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Fairness Feature')
    plt.tight_layout()
    plt.savefig(FAIRNESS_VIZ_DIR / 'fairness_by_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_dataset.png'}")

# Level 6: Model
if len(level6) > 0:
    pivot6 = level6.pivot(index='feature', columns='model', values='cohens_d_mean')

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot6) * 0.8)))
    sns.heatmap(pivot6, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Cohen\'s d'}, ax=ax, linewidths=1)
    ax.set_title('Fairness Bias by Model (Aggregated over datasets & prompts)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Fairness Feature')
    plt.tight_layout()
    plt.savefig(FAIRNESS_VIZ_DIR / 'fairness_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_model.png'}")

# Level 7: Prompt
if len(level7) > 0:
    pivot7 = level7.pivot(index='feature', columns='prompt_style', values='cohens_d_mean')

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot7) * 0.8)))
    sns.heatmap(pivot7, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Cohen\'s d'}, ax=ax, linewidths=1)
    ax.set_title('Fairness Bias by Prompt Style (Aggregated over datasets & models)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Fairness Feature')
    plt.tight_layout()
    plt.savefig(FAIRNESS_VIZ_DIR / 'fairness_by_prompt.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_prompt.png'}")

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

print(f"""
Files Created:
  ✓ {all_levels_file}
  ✓ {FAIRNESS_VIZ_DIR / 'aggregation_level_comparison.png'}
  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_dataset.png'}
  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_model.png'}
  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_prompt.png'}

Aggregation Levels Computed:
  Level 1 (D×M×P): {len(level1)} conditions
  Level 2 (D×M): {len(level2)} conditions
  Level 3 (D×P): {len(level3)} conditions
  Level 4 (M×P): {len(level4)} conditions
  Level 5 (D): {len(level5)} conditions
  Level 6 (M): {len(level6)} conditions
  Level 7 (P): {len(level7)} conditions
  Level 8 (Overall): {len(level8)} conditions

Fairness Features Analyzed:
""")

for _, row in level8.iterrows():
    sig_pct = row['significant_mean'] * 100
    print(f"  {row['feature']}:")
    print(f"    Overall Cohen's d: {row['cohens_d_mean']:.3f} ± {row['cohens_d_std']:.3f}")
    print(f"    Significant in {sig_pct:.1f}% of disaggregated conditions")
    print()

print("="*80)
print("✓ FAIRNESS DEEP DIVE COMPLETE")
print("="*80)
