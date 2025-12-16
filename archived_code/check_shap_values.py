import pandas as pd
import numpy as np

# Load importance results
df = pd.read_parquet('analysis_outputs/importance_analysis/importance_results.parquet')

print("Columns with SHAP values:")
shap_cols = [c for c in df.columns if c.startswith('shap_')]
print(f"Total SHAP columns: {len(shap_cols)}")
print(f"First 10: {shap_cols[:10]}")

print("\n" + "="*80)
print("Mean absolute SHAP values (top 15):")
print("="*80)

shap_means = {}
for col in shap_cols:
    if col == 'shap_file':  # Skip file path column
        continue
    feature_name = col[5:]  # Remove 'shap_' prefix
    try:
        shap_means[feature_name] = df[col].abs().mean()
    except:
        print(f"Skipping {col} (non-numeric)")
        continue

for feat, val in sorted(shap_means.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"{feat:40s}: {val:.4f}")
