"""
Feature Importance Visualization Pipeline
Generates visualizations from importance analysis results
Following: NEXT_SESSION_GUIDE.md Step 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
INPUT_FILE = Path('analysis_outputs/importance_analysis/importance_results.parquet')
SHAP_DIR = Path('analysis_outputs/importance_analysis/shap_values')
OUTPUT_DIR = Path('analysis_outputs/visualizations/static/importance_rankings')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
sns.set_palette("husl")

print("=" * 80)
print("FEATURE IMPORTANCE VISUALIZATION PIPELINE")
print("=" * 80)

# Load data
print("\nLoading data...")
df = pd.read_parquet(INPUT_FILE)
print(f"✓ Loaded {len(df)} importance results")

# Extract coefficient and SHAP columns
coef_cols = [c for c in df.columns if c.startswith('coef_')]
shap_cols = [c for c in df.columns if c.startswith('shap_')]
feature_names = [c.replace('coef_', '') for c in coef_cols]

print(f"✓ Found {len(feature_names)} features")
print(f"✓ Features: {', '.join(feature_names[:5])}...")


def plot_importance_heatmap():
    """
    Plot 1.1: Feature Importance Ranking Heatmap
    Shows top N features for each condition
    """
    print("\n[1/6] Generating importance ranking heatmap...")

    # Prepare data: Get top 15 features by absolute coefficient for each condition
    top_n = 15
    importance_matrix = []
    condition_labels = []

    for _, row in df.iterrows():
        # Get absolute coefficients
        coefs = {feat: abs(row[f'coef_{feat}']) for feat in feature_names}
        # Sort and take top N
        top_features = sorted(coefs.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Create row for heatmap
        row_values = [coefs.get(feat, 0) for feat, _ in sorted(top_features, key=lambda x: x[1], reverse=True)]
        importance_matrix.append(row_values)

        # Label: dataset_model_prompt
        label = f"{row['dataset']}_{row['model'].split('-')[0]}_{row['prompt_style'][:4]}"
        condition_labels.append(label)

    # Get consistent top features across all conditions
    all_feature_importance = {feat: [] for feat in feature_names}
    for _, row in df.iterrows():
        for feat in feature_names:
            all_feature_importance[feat].append(abs(row[f'coef_{feat}']))

    # Calculate mean importance
    mean_importance = {feat: np.mean(vals) for feat, vals in all_feature_importance.items()}
    top_features_global = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_feature_names = [feat for feat, _ in top_features_global]

    # Create heatmap matrix with consistent features
    heatmap_data = np.zeros((len(df), top_n))
    for i, (_, row) in enumerate(df.iterrows()):
        for j, feat in enumerate(top_feature_names):
            heatmap_data[i, j] = abs(row[f'coef_{feat}'])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')

    # Set ticks
    ax.set_yticks(range(len(condition_labels)))
    ax.set_yticklabels(condition_labels, fontsize=7)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_feature_names, rotation=45, ha='right', fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Coefficient|', rotation=270, labelpad=15)

    # Labels
    ax.set_xlabel('Feature')
    ax.set_ylabel('Condition (Dataset_Model_Prompt)')
    ax.set_title('Feature Importance Rankings Across All Conditions\n(Top 15 Features by Mean Absolute Coefficient)',
                 fontsize=12, pad=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'importance_heatmap_all_conditions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    return output_file


def plot_shap_summary():
    """
    Plot 1.2: SHAP Summary Plot (Aggregated)
    Shows aggregate SHAP values across all conditions
    """
    print("\n[2/6] Generating aggregated SHAP summary plot...")

    # Calculate mean SHAP importance across all conditions
    mean_shap = {}
    std_shap = {}

    for feat in feature_names:
        shap_col = f'shap_{feat}'
        if shap_col in df.columns:
            mean_shap[feat] = df[shap_col].mean()
            std_shap[feat] = df[shap_col].std()

    # Sort by mean importance
    sorted_features = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)

    # Plot top 20
    top_n = 20
    features = [f for f, _ in sorted_features[:top_n]]
    means = [mean_shap[f] for f in features]
    stds = [std_shap[f] for f in features]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(features))

    ax.barh(y_pos, means, xerr=stds, capsize=3, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Mean SHAP Importance (± Std Dev)')
    ax.set_title('Aggregate Feature Importance Across All Conditions\n(SHAP Values)',
                 fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'shap_summary_aggregated.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    return output_file


def plot_feature_consistency():
    """
    Plot 1.3: Feature Consistency Plot
    Shows how consistent each feature's importance is across conditions
    """
    print("\n[3/6] Generating feature consistency plot...")

    # Calculate coefficient of variation for each feature
    cv_data = []
    for feat in feature_names:
        coef_col = f'coef_{feat}'
        values = df[coef_col].abs()
        mean_val = values.mean()
        std_val = values.std()
        cv = (std_val / mean_val) if mean_val > 0 else 0

        cv_data.append({
            'feature': feat,
            'mean_importance': mean_val,
            'std_importance': std_val,
            'cv': cv
        })

    cv_df = pd.DataFrame(cv_data).sort_values('mean_importance', ascending=False)

    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean vs Std
    ax1.scatter(cv_df['mean_importance'], cv_df['std_importance'],
                s=100, alpha=0.6, c=cv_df['cv'], cmap='viridis')

    # Add labels for top features
    for _, row in cv_df.head(10).iterrows():
        ax1.annotate(row['feature'],
                    (row['mean_importance'], row['std_importance']),
                    fontsize=7, alpha=0.7)

    ax1.set_xlabel('Mean Importance (|Coefficient|)')
    ax1.set_ylabel('Std Dev of Importance')
    ax1.set_title('Feature Importance: Magnitude vs Variability')
    ax1.grid(alpha=0.3)

    # Plot 2: Coefficient of Variation
    y_pos = np.arange(min(20, len(cv_df)))
    ax2.barh(y_pos, cv_df.head(20)['cv'], color='coral', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cv_df.head(20)['feature'], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Coefficient of Variation')
    ax2.set_title('Feature Consistency (Lower = More Consistent)')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='CV=1.0')
    ax2.legend()

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'feature_consistency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    return output_file


def plot_model_comparison():
    """
    Plot 1.4: Model Comparison (Importance)
    Compare feature importance across models
    """
    print("\n[4/6] Generating model comparison plots...")

    # Get unique models
    models = df['model'].unique()

    # Calculate mean importance by model
    model_importance = {}
    for model in models:
        model_df = df[df['model'] == model]
        mean_coefs = {}
        for feat in feature_names:
            mean_coefs[feat] = model_df[f'coef_{feat}'].abs().mean()
        model_importance[model] = mean_coefs

    # Get top 15 features overall
    overall_importance = {feat: np.mean([model_importance[m][feat] for m in models])
                         for feat in feature_names}
    top_features = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    top_feature_names = [f for f, _ in top_features]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, model in enumerate(sorted(models)):
        ax = axes[idx]
        values = [model_importance[model][feat] for feat in top_feature_names]
        y_pos = np.arange(len(top_feature_names))

        ax.barh(y_pos, values, alpha=0.7)
        if idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_feature_names, fontsize=9)
            ax.invert_yaxis()
        ax.set_xlabel('Mean |Coefficient|')
        ax.set_title(f'{model}', fontsize=11)
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Feature Importance Comparison Across Models\n(Top 15 Features)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    output_file = OUTPUT_DIR / 'model_comparison_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    return output_file


def plot_prompt_comparison():
    """
    Plot 1.5: Prompt Comparison (Importance)
    Compare feature importance across prompt styles
    """
    print("\n[5/6] Generating prompt comparison plots...")

    # Get unique prompts
    prompts = df['prompt_style'].unique()

    # Calculate mean importance by prompt
    prompt_importance = {}
    for prompt in prompts:
        prompt_df = df[df['prompt_style'] == prompt]
        mean_coefs = {}
        for feat in feature_names:
            mean_coefs[feat] = prompt_df[f'coef_{feat}'].abs().mean()
        prompt_importance[prompt] = mean_coefs

    # Get top 12 features overall
    overall_importance = {feat: np.mean([prompt_importance[p][feat] for p in prompts])
                         for feat in feature_names}
    top_features = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)[:12]
    top_feature_names = [f for f, _ in top_features]

    # Plot with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for idx, prompt in enumerate(sorted(prompts)):
        ax = axes[idx]
        values = [prompt_importance[prompt][feat] for feat in top_feature_names]
        y_pos = np.arange(len(top_feature_names))

        ax.barh(y_pos, values, alpha=0.7, color=f'C{idx}')
        if idx % 3 == 0:  # First column
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_feature_names, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel('Mean |Coefficient|', fontsize=9)
        ax.set_title(f'{prompt}', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Feature Importance Comparison Across Prompt Styles\n(Top 12 Features)',
                 fontsize=13, y=0.995)
    plt.tight_layout()

    output_file = OUTPUT_DIR / 'prompt_comparison_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    return output_file


def plot_shap_dependence():
    """
    Plot 1.6: SHAP Dependence Plots
    Show how feature values affect predictions for top features
    """
    print("\n[6/6] Generating SHAP dependence plots...")

    # Identify top 5 features by mean SHAP importance
    mean_shap = {}
    for feat in feature_names:
        shap_col = f'shap_{feat}'
        if shap_col in df.columns:
            mean_shap[feat] = df[shap_col].mean()

    top_features = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)[:5]
    top_feature_names = [f for f, _ in top_features]

    print(f"  Top 5 features: {', '.join(top_feature_names)}")

    # For each top feature, create a dependence-style plot
    # (Note: We don't have actual SHAP dependence data, so we'll create a proxy visualization)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, feat in enumerate(top_feature_names):
        ax = axes[idx]

        # Get SHAP importance for this feature across conditions
        shap_values = df[f'shap_{feat}'].values
        coef_values = df[f'coef_{feat}'].values

        # Create scatter plot showing relationship
        scatter = ax.scatter(coef_values, shap_values,
                           c=range(len(df)), cmap='viridis',
                           alpha=0.6, s=50)

        ax.set_xlabel('Coefficient Value', fontsize=9)
        ax.set_ylabel('SHAP Importance', fontsize=9)
        ax.set_title(feat, fontsize=10)
        ax.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(coef_values, shap_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(coef_values.min(), coef_values.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)

    fig.suptitle('Feature Coefficient vs SHAP Importance\n(Top 5 Features)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    output_file = OUTPUT_DIR / 'shap_dependence_top5.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_summary_report():
    """Create a text summary of importance visualizations"""
    print("\nGenerating summary report...")

    # Calculate key statistics
    mean_auroc = df['auroc'].mean()
    std_auroc = df['auroc'].std()

    # Best and worst models
    best_row = df.loc[df['auroc'].idxmax()]
    worst_row = df.loc[df['auroc'].idxmin()]

    # Top features overall
    mean_shap = {}
    for feat in feature_names:
        shap_col = f'shap_{feat}'
        if shap_col in df.columns:
            mean_shap[feat] = df[shap_col].mean()
    top_features = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)[:10]

    report = f"""
FEATURE IMPORTANCE VISUALIZATION SUMMARY
{'=' * 80}

Generated: {pd.Timestamp.now()}
Total Conditions: {len(df)}
Features Analyzed: {len(feature_names)}

PREDICTIVE PERFORMANCE
{'=' * 80}
Mean AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}
Min AUROC: {df['auroc'].min():.3f}
Max AUROC: {df['auroc'].max():.3f}
Conditions with AUROC > 0.7: {(df['auroc'] > 0.7).sum()} ({(df['auroc'] > 0.7).sum()/len(df)*100:.1f}%)

Best Performance:
  - Condition: {best_row['dataset']} × {best_row['model']} × {best_row['prompt_style']}
  - AUROC: {best_row['auroc']:.3f}

Worst Performance:
  - Condition: {worst_row['dataset']} × {worst_row['model']} × {worst_row['prompt_style']}
  - AUROC: {worst_row['auroc']:.3f}

TOP 10 FEATURES BY MEAN SHAP IMPORTANCE
{'=' * 80}
"""

    for rank, (feat, importance) in enumerate(top_features, 1):
        report += f"{rank:2d}. {feat:30s} {importance:.4f}\n"

    report += f"""
VISUALIZATIONS GENERATED
{'=' * 80}
1. importance_heatmap_all_conditions.png - Heatmap of top 15 features across all conditions
2. shap_summary_aggregated.png - Aggregate SHAP values with error bars
3. feature_consistency.png - Feature importance variability analysis
4. model_comparison_importance.png - Top features by model
5. prompt_comparison_importance.png - Top features by prompt style
6. shap_dependence_top5.png - Coefficient vs SHAP for top 5 features

All files saved to: {OUTPUT_DIR}
"""

    output_file = OUTPUT_DIR / 'importance_visualization_summary.txt'
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"  ✓ Saved: {output_file}")
    return report


# Main execution
if __name__ == '__main__':
    try:
        # Generate all plots
        plot_importance_heatmap()
        plot_shap_summary()
        plot_feature_consistency()
        plot_model_comparison()
        plot_prompt_comparison()
        plot_shap_dependence()

        # Create summary
        summary = create_summary_report()

        print("\n" + "=" * 80)
        print("VISUALIZATION PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nGenerated 6 visualization files in: {OUTPUT_DIR}")
        print(f"\nSummary saved to: {OUTPUT_DIR / 'importance_visualization_summary.txt'}")
        print("\n" + summary)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
