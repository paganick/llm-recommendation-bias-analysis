"""
Create comparison visualizations between full and substantive-only analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'static' / 'comparison'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("Creating comparison visualizations...")

# Load results
results_full = pd.read_parquet('analysis_outputs/importance_analysis/importance_results.parquet')
results_substantive = pd.read_parquet('analysis_outputs/importance_analysis/importance_results_substantive_only.parquet')

# Create SHAP summaries
shap_cols_full = [c for c in results_full.columns if c.startswith('shap_') and c != 'shap_file']
shap_summary_full = []
for col in shap_cols_full:
    feat_name = col[5:]
    try:
        mean_val = results_full[col].abs().mean()
        if not np.isnan(mean_val):
            shap_summary_full.append({'feature': feat_name, 'mean_shap': mean_val})
    except:
        continue
summary_full = pd.DataFrame(shap_summary_full).sort_values('mean_shap', ascending=False)

shap_cols_sub = [c for c in results_substantive.columns if c.startswith('shap_')]
shap_summary_sub = []
for col in shap_cols_sub:
    feat_name = col[5:]
    try:
        mean_val = results_substantive[col].abs().mean()
        if not np.isnan(mean_val):
            shap_summary_sub.append({'feature': feat_name, 'mean_shap': mean_val})
    except:
        continue
summary_substantive = pd.DataFrame(shap_summary_sub).sort_values('mean_shap', ascending=False)

# Comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Panel A: Full
top_full = summary_full.head(15)
axes[0].barh(range(len(top_full)), top_full['mean_shap'], color='steelblue')
axes[0].set_yticks(range(len(top_full)))
axes[0].set_yticklabels(top_full['feature'])
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean Absolute SHAP Value')
axes[0].set_title('A) Full Analysis (All Features)\nIncludes stylistic features', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Panel B: Substantive
top_sub = summary_substantive.head(15)
axes[1].barh(range(len(top_sub)), top_sub['mean_shap'], color='darkgreen')
axes[1].set_yticks(range(len(top_sub)))
axes[1].set_yticklabels(top_sub['feature'])
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean Absolute SHAP Value')
axes[1].set_title('B) Substantive-Only Analysis\nStylistic features removed', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
output_file = VIZ_DIR / 'full_vs_substantive_importance.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {output_file}")

# AUROC comparison
fig, ax = plt.subplots(figsize=(10, 6))

auroc_comparison = pd.DataFrame({
    'Full Analysis': [results_full['auroc'].mean()],
    'Substantive Only': [results_substantive['auroc'].mean()]
})

x = np.arange(len(auroc_comparison.columns))
means = [results_full['auroc'].mean(), results_substantive['auroc'].mean()]
stds = [results_full['auroc'].std(), results_substantive['auroc'].std()]

ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7, color=['steelblue', 'darkgreen'])
ax.set_xticks(x)
ax.set_xticklabels(auroc_comparison.columns)
ax.set_ylabel('AUROC Score')
ax.set_title('Prediction Performance: Full vs Substantive-Only Features', fontsize=14, fontweight='bold')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
ax.grid(axis='y', alpha=0.3)
ax.legend()
plt.tight_layout()
output_file = VIZ_DIR / 'auroc_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ {output_file}")

print(f"\nSummary:")
print(f"Full Analysis: AUROC = {results_full['auroc'].mean():.3f} ± {results_full['auroc'].std():.3f}")
print(f"Substantive Only: AUROC = {results_substantive['auroc'].mean():.3f} ± {results_substantive['auroc'].std():.3f}")
print(f"\nTop feature (Full): {summary_full.iloc[0]['feature']} (SHAP: {summary_full.iloc[0]['mean_shap']:.3f})")
print(f"Top feature (Substantive): {summary_substantive.iloc[0]['feature']} (SHAP: {summary_substantive.iloc[0]['mean_shap']:.3f})")
