#!/usr/bin/env python3
"""
Create visualizations for survey data bias analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
bias_file = 'external_data_analysis/outputs/analysis/bias_summary.csv'
bias_df = pd.read_csv(bias_file)

# Create output directory
viz_dir = Path('external_data_analysis/outputs/analysis/visualizations')
viz_dir.mkdir(parents=True, exist_ok=True)

# 1. Bar chart of Cramér's V for categorical features
cat_data = bias_df[bias_df['type'] == 'categorical'].sort_values('cramers_v', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(cat_data['feature'], cat_data['cramers_v'], color='steelblue')
plt.xlabel("Cramér's V (Effect Size)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Categorical Feature Bias in Survey Data Recommendations", fontsize=14, fontweight='bold')
plt.xlim(0, max(0.15, cat_data['cramers_v'].max() * 1.2))
plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Small effect (0.05)')
plt.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5, label='Medium effect (0.10)')
plt.legend()
plt.tight_layout()
plt.savefig(viz_dir / 'categorical_bias.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created categorical_bias.png")

# 2. Bar chart of Cohen's d for numerical features
num_data = bias_df[bias_df['type'] == 'numerical'].sort_values('cohens_d', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'coral' for x in num_data['cohens_d']]
plt.barh(num_data['feature'], num_data['cohens_d'], color=colors)
plt.xlabel("Cohen's d (Effect Size)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Numerical Feature Bias in Survey Data Recommendations", fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Small effect (±0.2)')
plt.axvline(x=-0.2, color='red', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(viz_dir / 'numerical_bias.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created numerical_bias.png")

# 3. Summary table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_text = "Survey Data Bias Analysis - Summary\n" + "="*60 + "\n\n"
summary_text += "Categorical Features (Cramér's V):\n"
for _, row in cat_data.iterrows():
    summary_text += f"  {row['feature']:30s}: {row['cramers_v']:.4f}\n"

summary_text += "\nNumerical Features (Cohen's d):\n"
for _, row in num_data.iterrows():
    summary_text += f"  {row['feature']:30s}: {row['cohens_d']:+.4f}\n"

summary_text += "\n" + "="*60 + "\n"
summary_text += f"Total features analyzed: {len(bias_df)}\n"
summary_text += f"  Categorical: {len(cat_data)}\n"
summary_text += f"  Numerical: {len(num_data)}\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig(viz_dir / 'summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created summary.png")

print(f"\n✓ All visualizations saved to {viz_dir}/")
