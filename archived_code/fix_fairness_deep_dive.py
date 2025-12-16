"""
Fix Fairness Deep Dive - Use Cramer's V for Categorical Features

For fairness features (gender, political leaning, minority status),
we need to use Cramer's V as the effect size, not Cohen's d
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("FIXING FAIRNESS DEEP DIVE - USING CRAMER'S V")
print("="*80)

OUTPUT_DIR = Path('analysis_outputs')
FAIRNESS_DIR = OUTPUT_DIR / 'fairness_analysis'
FAIRNESS_VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'static' / 'fairness_deep_dive'
FAIRNESS_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Load bias data
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results_enhanced.parquet')

# Fairness features
FAIRNESS_FEATURES = ['author_gender', 'author_political_leaning', 'author_is_minority']

fairness_data = bias_df[bias_df['feature'].isin(FAIRNESS_FEATURES)].copy()

print(f"\nFairness features found: {fairness_data['feature'].unique()}")
print(f"Total measurements: {len(fairness_data)}")

# ============================================================================
# 8-Level Aggregation with Cramer's V
# ============================================================================

print("\n" + "="*80)
print("COMPUTING 8-LEVEL AGGREGATION HIERARCHY (CRAMER'S V)")
print("="*80)

aggregation_results = []

# Level 1: Fully disaggregated
print("\nLevel 1: dataset × model × prompt")
level1 = fairness_data.groupby(['dataset', 'model', 'prompt_style', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level1.columns = ['_'.join(col).strip('_') for col in level1.columns.values]
level1['aggregation_level'] = 'L1_dataset_model_prompt'
print(f"  Conditions: {len(level1)}")
aggregation_results.append(level1)

# Level 2: dataset × model
print("\nLevel 2: dataset × model")
level2 = fairness_data.groupby(['dataset', 'model', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level2.columns = ['_'.join(col).strip('_') for col in level2.columns.values]
level2['aggregation_level'] = 'L2_dataset_model'
print(f"  Conditions: {len(level2)}")
aggregation_results.append(level2)

# Level 3: dataset × prompt
print("\nLevel 3: dataset × prompt")
level3 = fairness_data.groupby(['dataset', 'prompt_style', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level3.columns = ['_'.join(col).strip('_') for col in level3.columns.values]
level3['aggregation_level'] = 'L3_dataset_prompt'
print(f"  Conditions: {len(level3)}")
aggregation_results.append(level3)

# Level 4: model × prompt
print("\nLevel 4: model × prompt")
level4 = fairness_data.groupby(['model', 'prompt_style', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level4.columns = ['_'.join(col).strip('_') for col in level4.columns.values]
level4['aggregation_level'] = 'L4_model_prompt'
print(f"  Conditions: {len(level4)}")
aggregation_results.append(level4)

# Level 5: dataset
print("\nLevel 5: dataset")
level5 = fairness_data.groupby(['dataset', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level5.columns = ['_'.join(col).strip('_') for col in level5.columns.values]
level5['aggregation_level'] = 'L5_dataset'
print(f"  Conditions: {len(level5)}")
aggregation_results.append(level5)

# Level 6: model
print("\nLevel 6: model")
level6 = fairness_data.groupby(['model', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level6.columns = ['_'.join(col).strip('_') for col in level6.columns.values]
level6['aggregation_level'] = 'L6_model'
print(f"  Conditions: {len(level6)}")
aggregation_results.append(level6)

# Level 7: prompt
print("\nLevel 7: prompt")
level7 = fairness_data.groupby(['prompt_style', 'feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level7.columns = ['_'.join(col).strip('_') for col in level7.columns.values]
level7['aggregation_level'] = 'L7_prompt'
print(f"  Conditions: {len(level7)}")
aggregation_results.append(level7)

# Level 8: overall
print("\nLevel 8: overall")
level8 = fairness_data.groupby(['feature']).agg({
    'cramers_v': ['mean', 'std', 'count'],
    'p_value': 'mean',
    'significant': 'mean'
}).reset_index()
level8.columns = ['_'.join(col).strip('_') for col in level8.columns.values]
level8['aggregation_level'] = 'L8_overall'
print(f"  Conditions: {len(level8)}")
aggregation_results.append(level8)

# Save
all_levels = pd.concat(aggregation_results, ignore_index=True)
all_levels_file = FAIRNESS_DIR / 'fairness_8level_aggregation_cramersv.parquet'
all_levels.to_parquet(all_levels_file)
print(f"\n✓ Saved to: {all_levels_file}")

# ============================================================================
# Robustness Report
# ============================================================================

print("\n" + "="*80)
print("ROBUSTNESS ANALYSIS")
print("="*80)

for _, row in level8.iterrows():
    sig_pct = row['significant_mean'] * 100
    print(f"\n{row['feature']}:")
    print(f"  Mean Cramer's V: {row['cramers_v_mean']:.4f} ± {row['cramers_v_std']:.4f}")
    print(f"  Significant in {sig_pct:.1f}% of conditions")
    print(f"  Mean p-value: {row['p_value_mean']:.6f}")

# ============================================================================
# Visualization 1: Aggregation Level Comparison
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fairness_features_present = level8['feature'].unique()

fig, axes = plt.subplots(len(fairness_features_present), 1,
                          figsize=(14, 5 * len(fairness_features_present)))

if len(fairness_features_present) == 1:
    axes = [axes]

for idx, feature in enumerate(fairness_features_present):
    ax = axes[idx]

    # Gather data across all levels
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
                'mean': feat_data['cramers_v_mean'].mean(),
                'std': feat_data['cramers_v_std'].mean() if 'cramers_v_std' in feat_data.columns else 0,
                'n': len(feat_data)
            })

    plot_df = pd.DataFrame(plot_data)

    # Bar plot
    x_pos = np.arange(len(plot_df))
    ax.bar(x_pos, plot_df['mean'], yerr=plot_df['std'],
           capsize=5, alpha=0.7, color='steelblue')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df['level'], rotation=45, ha='right')
    ax.set_ylabel('Cramer\'s V (Mean ± SD)')
    ax.set_title(f'Bias in {feature} Across Aggregation Levels',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add sample sizes
    for i, row in plot_df.iterrows():
        ax.text(i, row['mean'] + row['std'] + 0.01, f"n={row['n']}",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_file = FAIRNESS_VIZ_DIR / 'aggregation_level_comparison_cramersv.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {output_file}")

# ============================================================================
# Visualization 2: Heatmaps
# ============================================================================

# Level 5: Dataset
if len(level5) > 0:
    pivot5 = level5.pivot(index='feature', columns='dataset', values='cramers_v_mean')

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot5) * 0.8)))
    sns.heatmap(pivot5, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Cramer\'s V'}, ax=ax, linewidths=1,
                vmin=0, vmax=0.3)
    ax.set_title('Fairness Bias by Dataset (Cramer\'s V)\nAggregated over models & prompts',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Fairness Feature')
    plt.tight_layout()
    output_file = FAIRNESS_VIZ_DIR / 'fairness_by_dataset_cramersv.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_file}")

# Level 6: Model
if len(level6) > 0:
    pivot6 = level6.pivot(index='feature', columns='model', values='cramers_v_mean')

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot6) * 0.8)))
    sns.heatmap(pivot6, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Cramer\'s V'}, ax=ax, linewidths=1,
                vmin=0, vmax=0.3)
    ax.set_title('Fairness Bias by Model (Cramer\'s V)\nAggregated over datasets & prompts',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Fairness Feature')
    plt.tight_layout()
    output_file = FAIRNESS_VIZ_DIR / 'fairness_by_model_cramersv.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_file}")

# Level 7: Prompt
if len(level7) > 0:
    pivot7 = level7.pivot(index='feature', columns='prompt_style', values='cramers_v_mean')

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot7) * 0.8)))
    sns.heatmap(pivot7, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Cramer\'s V'}, ax=ax, linewidths=1,
                vmin=0, vmax=0.3)
    ax.set_title('Fairness Bias by Prompt Style (Cramer\'s V)\nAggregated over datasets & models',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Fairness Feature')
    plt.tight_layout()
    output_file = FAIRNESS_VIZ_DIR / 'fairness_by_prompt_cramersv.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Fairness Features Analyzed:
""")

for _, row in level8.iterrows():
    sig_pct = row['significant_mean'] * 100
    print(f"  {row['feature']}:")
    print(f"    Cramer's V: {row['cramers_v_mean']:.4f} ± {row['cramers_v_std']:.4f}")
    print(f"    Significant in {sig_pct:.1f}% of conditions")
    print()

print("Files Created:")
print(f"  ✓ {all_levels_file}")
print(f"  ✓ {FAIRNESS_VIZ_DIR / 'aggregation_level_comparison_cramersv.png'}")
print(f"  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_dataset_cramersv.png'}")
print(f"  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_model_cramersv.png'}")
print(f"  ✓ {FAIRNESS_VIZ_DIR / 'fairness_by_prompt_cramersv.png'}")

print("\n" + "="*80)
print("✓ FAIRNESS DEEP DIVE COMPLETE")
print("="*80)
