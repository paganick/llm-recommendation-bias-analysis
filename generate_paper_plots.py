#!/usr/bin/env python3
"""
Generate Paper-Ready Plots
===========================

This script generates publication-quality plots for the paper.
Each plot is carefully formatted with appropriate font sizes, ordering, and styling.

Usage:
    python generate_paper_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

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

# Feature display names
FEATURE_DISPLAY_NAMES = {
    'author_gender': 'Author: Gender',
    'author_political_leaning': 'Author: Political Leaning',
    'author_is_minority': 'Author: Is Minority',
    'text_length': 'Text: Length (chars)',
    'avg_word_length': 'Text: Avg Word Length',
    'polarization_score': 'Content: Polarization Score',
    'controversy_level': 'Content: Controversy Level',
    'primary_topic': 'Content: Primary Topic',
    'sentiment_polarity': 'Sentiment: Polarity',
    'sentiment_subjectivity': 'Sentiment: Subjectivity',
    'has_emoji': 'Style: Has Emoji',
    'has_hashtag': 'Style: Has Hashtag',
    'has_mention': 'Style: Has Mention',
    'has_url': 'Style: Has URL',
    'toxicity': 'Toxicity: Toxicity',
    'severe_toxicity': 'Toxicity: Severe Toxicity'
}

# Feature category colors
CATEGORY_COLORS = {
    'author': ['#8B4513', '#A0522D', '#CD853F'],  # Browns - demographic features
    'text_metrics': ['#1E90FF', '#4169E1'],  # Blues
    'content': ['#32CD32', '#3CB371', '#2E8B57'],  # Greens
    'sentiment': ['#FFD700', '#FFA500'],  # Gold/Orange
    'style': ['#9370DB', '#8A2BE2', '#9400D3', '#9932CC'],  # Purples
    'toxicity': ['#DC143C', '#B22222']  # Reds - toxicity features
}

# Output directory
OUTPUT_DIR = Path('analysis_outputs/visualizations/paper_plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_feature_name(feature):
    """Format feature name for display."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())

def get_feature_category(feature):
    """Get category for a feature."""
    for category, feats in FEATURES.items():
        if feature in feats:
            return category
    return 'other'

def get_feature_color(feature, idx_within_category=0):
    """Get color for a feature based on its category."""
    category = get_feature_category(feature)
    colors = CATEGORY_COLORS.get(category, ['#888888'])
    return colors[idx_within_category % len(colors)]

def cohens_d_to_r_squared(d):
    """Convert Cohen's d to R² (variance explained)."""
    return (d ** 2) / (d ** 2 + 4)

def cramers_v_to_r_squared(v):
    """Convert Cramér's V to R² (variance explained)."""
    return v ** 2

def convert_to_r_squared(row):
    """Convert bias metric to R² based on metric type."""
    if pd.isna(row['bias']) or pd.isna(row['metric']):
        return np.nan

    abs_bias = abs(row['bias'])
    if row['metric'] == "Cohen's d":
        return cohens_d_to_r_squared(abs_bias)
    elif row['metric'] == "Cramér's V":
        return cramers_v_to_r_squared(abs_bias)
    else:
        return np.nan

# ============================================================================
# PLOT 1: AGGREGATED BAR PLOT WITH IMPROVED ORDERING
# ============================================================================

def create_aggregated_bar_plot_ordered(comp_df):
    """
    Create bar plot showing average R² for each feature,
    ordered by R² magnitude in descending order.

    Improvements over original:
    - Larger fonts for better readability
    - Features ordered by R² magnitude (descending)
    - Exports raw data to CSV
    """
    print("\n" + "="*80)
    print("GENERATING AGGREGATED BAR PLOT (R²) - PAPER VERSION")
    print("="*80)

    # Calculate average R² per feature
    agg_full = comp_df.groupby('feature').agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    # Add category and display name
    agg_full['category'] = agg_full['feature'].apply(get_feature_category)
    agg_full['feature_display'] = agg_full['feature'].apply(format_feature_name)

    # Sort features by R² magnitude (ascending for horizontal bar plot - smallest at bottom)
    # This makes the largest values appear at the TOP of the plot
    agg_full = agg_full.sort_values('r_squared', ascending=True).reset_index(drop=True)

    # Assign colors based on category
    colors = []
    category_counts = {}
    for _, row in agg_full.iterrows():
        category = row['category']
        if category not in category_counts:
            category_counts[category] = 0
        colors.append(get_feature_color(row['feature'], category_counts[category]))
        category_counts[category] += 1

    # Create figure with larger fonts
    fig, ax = plt.subplots(figsize=(10, 14))

    bars = ax.barh(agg_full['feature_display'], agg_full['r_squared'],
                   color=colors, edgecolor='black', alpha=0.8, linewidth=0.5)

    # Add significance markers
    for i, (_, row) in enumerate(agg_full.iterrows()):
        if row['significant'] > 0.75:
            ax.text(row['r_squared'], i, ' ***', va='center',
                   fontsize=14, fontweight='bold')
        elif row['significant'] > 0.60:
            ax.text(row['r_squared'], i, ' **', va='center',
                   fontsize=14, fontweight='bold')
        elif row['significant'] > 0.50:
            ax.text(row['r_squared'], i, ' *', va='center',
                   fontsize=14, fontweight='bold')

    # Set labels with larger fonts
    ax.set_xlabel('Average R² (Variance Explained)\nAcross All Datasets, Models & Prompts',
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=16, fontweight='bold')
    ax.set_title('Average Bias per Feature (R²)\n(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=18)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.grid(axis='x', alpha=0.3)

    # Add legend for categories with larger font
    from matplotlib.patches import Patch
    legend_elements = []

    # Create legend in a consistent order
    category_order = ['author', 'text_metrics', 'sentiment', 'style', 'content', 'toxicity']
    for category in category_order:
        if category in FEATURES:
            features = FEATURES[category]
            if features:
                color = get_feature_color(features[0], 0)
                legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                            label=category.replace('_', ' ').title()))

    ax.legend(handles=legend_elements, loc='lower right',
             title='Feature Category', fontsize=14, title_fontsize=14)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'aggregated_bar_plot_ordered.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Paper-ready aggregated bar plot saved to: {output_path}")

    # Save raw data to CSV (reverse order for descending R² in CSV)
    data_output_path = OUTPUT_DIR / 'aggregated_bar_plot_data.csv'
    agg_full_csv = agg_full.sort_values('r_squared', ascending=False)
    agg_full_csv[['feature', 'feature_display', 'category', 'r_squared', 'significant']].to_csv(
        data_output_path, index=False
    )
    print(f"✓ Raw data saved to: {data_output_path}")

    # Print feature ordering for reference (from top to bottom in plot = largest to smallest)
    print("\nFeature ordering (top to bottom in plot):")
    for i in range(len(agg_full)-1, -1, -1):
        row = agg_full.iloc[i]
        sig_marker = ''
        if row['significant'] > 0.75:
            sig_marker = '***'
        elif row['significant'] > 0.60:
            sig_marker = '**'
        elif row['significant'] > 0.50:
            sig_marker = '*'
        print(f"  {len(agg_full)-i:2d}. [{row['category']:12s}] {row['feature_display']:35s} R²={row['r_squared']:.4f} {sig_marker}")

    # Return data sorted descending for easier stats access
    return agg_full_csv

# ============================================================================
# PLOT 2: BIAS BY PROMPT HEATMAP
# ============================================================================

def create_bias_by_prompt_heatmap(comp_df):
    """
    Create heatmap showing R² values for each feature across prompt styles.

    Improvements:
    - Color bar starts from white (0) to dark red (max)
    - Columns ordered: neutral, general, popular, engaging, informative, controversial
    - Additional "Average" column showing overall feature importance
    - Additional "Mean Across Features" row showing mean R² per prompt
    - Rows ordered by Average column (descending)
    - Larger fonts for readability
    """
    print("\n" + "="*80)
    print("GENERATING BIAS BY PROMPT HEATMAP - PAPER VERSION")
    print("="*80)

    # Aggregate by feature and prompt
    agg_prompt = comp_df.groupby(['feature', 'prompt_style']).agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()

    # Calculate overall average per feature
    agg_overall = comp_df.groupby('feature').agg({
        'r_squared': 'mean',
        'significant': 'mean'
    }).reset_index()
    agg_overall['prompt_style'] = 'Average'

    # Combine
    agg_combined = pd.concat([agg_prompt, agg_overall], ignore_index=True)

    # Pivot for heatmap
    pivot_r2 = agg_combined.pivot(index='feature', columns='prompt_style', values='r_squared')
    pivot_sig = agg_combined.pivot(index='feature', columns='prompt_style', values='significant')

    # Order columns: neutral, general, popular, engaging, informative, controversial, Average
    prompt_order = ['neutral', 'general', 'popular', 'engaging', 'informative', 'controversial', 'Average']
    pivot_r2 = pivot_r2[prompt_order]
    pivot_sig = pivot_sig[prompt_order]

    # Order rows by Average column (descending)
    pivot_r2 = pivot_r2.sort_values('Average', ascending=False)
    pivot_sig = pivot_sig.reindex(pivot_r2.index)

    # Add Mean Across Features row
    prompt_cols = ['neutral', 'general', 'popular', 'engaging', 'informative', 'controversial']
    mean_row = {}
    for col in prompt_cols + ['Average']:
        mean_row[col] = pivot_r2[col].mean()

    mean_row_series = pd.Series(mean_row, name='Mean Across Features')
    pivot_r2 = pd.concat([pivot_r2, mean_row_series.to_frame().T])

    mean_sig_row = pd.Series({col: np.nan for col in pivot_sig.columns}, name='Mean Across Features')
    pivot_sig = pd.concat([pivot_sig, mean_sig_row.to_frame().T])

    # Format feature names
    pivot_r2.index = [format_feature_name(f) if f != 'Mean Across Features' else f for f in pivot_r2.index]
    pivot_sig.index = [format_feature_name(f) if f != 'Mean Across Features' else f for f in pivot_sig.index]

    # Capitalize column names
    pivot_r2.columns = [col.title() for col in pivot_r2.columns]
    pivot_sig.columns = [col.title() for col in pivot_sig.columns]

    # Create annotations with significance markers
    annot = np.empty_like(pivot_r2, dtype=object)
    for i in range(pivot_r2.shape[0]):
        row_name = pivot_r2.index[i]
        for j in range(pivot_r2.shape[1]):
            val = pivot_r2.iloc[i, j]
            sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot[i, j] = ''
            elif row_name == 'Mean Across Features':
                # No significance markers for mean row
                annot[i, j] = f'{val:.3f}'
            elif sig > 0.75:
                annot[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot[i, j] = f'{val:.3f}*'
            else:
                annot[i, j] = f'{val:.3f}'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create custom colormap starting from white
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#FFFFFF', '#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('white_red', colors_list, N=n_bins)

    # Plot heatmap
    sns.heatmap(pivot_r2, annot=annot, fmt='', cmap=cmap,
                vmin=0, vmax=pivot_r2.max().max(), ax=ax,
                cbar_kws={'label': 'R² (Variance Explained)'},
                linewidths=0.5, linecolor='lightgray')

    # Highlight the Average column with a thicker border and shading
    average_col_idx = pivot_r2.columns.get_loc('Average')
    ax.axvline(x=average_col_idx, color='black', linewidth=3)
    ax.axvline(x=average_col_idx+1, color='black', linewidth=3)

    # Add subtle background to Average column
    for i in range(len(pivot_r2)):
        rect = plt.Rectangle((average_col_idx, i), 1, 1,
                            fill=True, facecolor='lightgray', alpha=0.2,
                            edgecolor='black', linewidth=3, zorder=0)
        ax.add_patch(rect)

    # Add horizontal line to separate Mean row from features
    mean_row_idx = len(pivot_r2) - 1
    ax.axhline(y=mean_row_idx, color='black', linewidth=3)

    # Set title and labels with larger fonts
    ax.set_title('Bias by Prompt Style ($R^2$) - Aggregated across Datasets & Models\n' +
                 '(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Prompt Style', fontsize=15, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=15, fontweight='bold')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'bias_by_prompt_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Paper-ready bias-by-prompt heatmap saved to: {output_path}")

    # Save raw data
    data_output_path = OUTPUT_DIR / 'bias_by_prompt_heatmap_data.csv'
    # Reshape for CSV (one row per feature-prompt combination)
    raw_feature_names = [f for f in comp_df['feature'].unique() if f in pivot_r2.index.map(
        lambda x: next((k for k, v in FEATURE_DISPLAY_NAMES.items() if v == x), None)
    )]

    # Create clean output dataframe
    output_rows = []
    for feature_display in pivot_r2.index:
        # Find original feature name
        feature = next((k for k, v in FEATURE_DISPLAY_NAMES.items() if v == feature_display), feature_display)
        row_data = {'feature': feature, 'feature_display': feature_display}
        for col in pivot_r2.columns:
            row_data[col.lower()] = pivot_r2.loc[feature_display, col]
        output_rows.append(row_data)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(data_output_path, index=False)
    print(f"✓ Raw data saved to: {data_output_path}")

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS BY PROMPT STYLE")
    print("="*80)

    print("\nTop 3 features with highest variance across prompts:")
    prompt_cols = ['Neutral', 'General', 'Popular', 'Engaging', 'Informative', 'Controversial']
    variances = pivot_r2[prompt_cols].var(axis=1).sort_values(ascending=False)
    for i, (feat, var) in enumerate(variances.head(3).items(), 1):
        avg_val = pivot_r2.loc[feat, 'Average']
        min_val = pivot_r2.loc[feat, prompt_cols].min()
        max_val = pivot_r2.loc[feat, prompt_cols].max()
        print(f"  {i}. {feat}: variance={var:.6f}, range=[{min_val:.4f}, {max_val:.4f}], avg={avg_val:.4f}")

    print("\nTop 3 features most consistent across prompts (lowest variance):")
    for i, (feat, var) in enumerate(variances.tail(3).items(), 1):
        avg_val = pivot_r2.loc[feat, 'Average']
        min_val = pivot_r2.loc[feat, prompt_cols].min()
        max_val = pivot_r2.loc[feat, prompt_cols].max()
        print(f"  {i}. {feat}: variance={var:.6f}, range=[{min_val:.4f}, {max_val:.4f}], avg={avg_val:.4f}")

    return pivot_r2

# ============================================================================
# PLOT 3: NORMALIZED BIAS BY PROMPT HEATMAP (for Appendix)
# ============================================================================

def create_normalized_bias_by_prompt_heatmap(comp_df):
    """
    Create heatmap showing normalized bias (Cohen's d or Cramér's V) for each feature
    across prompt styles. Normalization is within-feature (row-wise) to highlight
    relative differences across prompts.

    Improvements:
    - Uses original metrics (d/V) instead of R²
    - Row-wise z-score normalization to make features comparable
    - Diverging colormap (blue-white-red) for normalized values
    - Same column ordering and layout as R² version
    """
    print("\n" + "="*80)
    print("GENERATING NORMALIZED BIAS BY PROMPT HEATMAP - PAPER VERSION")
    print("="*80)

    # Aggregate by feature and prompt - use original bias metric
    agg_prompt = comp_df.groupby(['feature', 'prompt_style']).agg({
        'bias': 'mean',  # Use Cohen's d or Cramér's V
        'significant': 'mean'
    }).reset_index()

    # Calculate overall average per feature
    agg_overall = comp_df.groupby('feature').agg({
        'bias': 'mean',
        'significant': 'mean'
    }).reset_index()
    agg_overall['prompt_style'] = 'Average'

    # Combine
    agg_combined = pd.concat([agg_prompt, agg_overall], ignore_index=True)

    # Pivot for heatmap
    pivot_bias = agg_combined.pivot(index='feature', columns='prompt_style', values='bias')
    pivot_sig = agg_combined.pivot(index='feature', columns='prompt_style', values='significant')

    # Order columns: neutral, general, popular, engaging, informative, controversial, Average
    prompt_order = ['neutral', 'general', 'popular', 'engaging', 'informative', 'controversial', 'Average']
    pivot_bias = pivot_bias[prompt_order]
    pivot_sig = pivot_sig[prompt_order]

    # Normalize within each row (feature) - compute z-scores across the 6 prompts only
    prompt_cols = ['neutral', 'general', 'popular', 'engaging', 'informative', 'controversial']
    pivot_normalized = pivot_bias[prompt_cols].copy()

    for feature in pivot_normalized.index:
        values = pivot_bias.loc[feature, prompt_cols].values
        mean_val = values.mean()
        std_val = values.std()

        if std_val > 0:  # Avoid division by zero
            # Normalize the 6 prompt columns
            pivot_normalized.loc[feature, prompt_cols] = (values - mean_val) / std_val
        else:
            # If no variation, set to zero
            pivot_normalized.loc[feature, prompt_cols] = 0

    # Order rows by average R² (from comp_df)
    avg_r2_dict = {}
    for feature in pivot_normalized.index:
        avg_r2_dict[feature] = comp_df[comp_df['feature'] == feature]['r_squared'].mean()

    ordering = pd.Series(avg_r2_dict).sort_values(ascending=False).index
    pivot_normalized = pivot_normalized.reindex(ordering)
    pivot_sig = pivot_sig[prompt_cols].reindex(ordering)

    # Format feature names
    pivot_normalized.index = [format_feature_name(f) for f in pivot_normalized.index]
    pivot_sig.index = [format_feature_name(f) for f in pivot_sig.index]

    # Capitalize column names
    pivot_normalized.columns = [col.title() for col in pivot_normalized.columns]
    pivot_sig.columns = [col.title() for col in pivot_sig.columns]

    # Create annotations - show z-scores with significance markers
    annot = np.empty_like(pivot_normalized, dtype=object)
    for i in range(pivot_normalized.shape[0]):
        for j in range(pivot_normalized.shape[1]):
            val = pivot_normalized.iloc[i, j]
            sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0

            if pd.isna(val):
                annot[i, j] = ''
            else:
                # Show z-score with significance markers
                if sig > 0.75:
                    annot[i, j] = f'{val:.1f}***'
                elif sig > 0.60:
                    annot[i, j] = f'{val:.1f}**'
                elif sig > 0.50:
                    annot[i, j] = f'{val:.1f}*'
                else:
                    annot[i, j] = f'{val:.1f}'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create diverging colormap for normalized values (blue-white-red)
    # Ensure white is exactly at center (0)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    colors_list = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                   '#FFFFFF',  # pure white center
                   '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#DE2D26']
    cmap = LinearSegmentedColormap.from_list('diverging', colors_list, N=256)

    # Determine vmin/vmax for normalized columns (symmetric around 0)
    normalized_values = pivot_normalized.values.flatten()
    normalized_values = normalized_values[~np.isnan(normalized_values)]
    max_abs = max(abs(normalized_values.min()), abs(normalized_values.max()))

    # Plot heatmap
    sns.heatmap(pivot_normalized, annot=annot, fmt='', cmap=cmap,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar_kws={'label': 'Normalized Bias (z-score)\n← Reduced | Enhanced →'},
                linewidths=0.5, linecolor='lightgray')

    # Set title and labels with larger fonts
    ax.set_title('Normalized Bias by Prompt Style - Aggregated across Datasets & Models\n' +
                 '(Normalized within feature: red = enhanced, blue = reduced)\n' +
                 '(* p<0.05 >50%, ** >60%, *** >75%)',
                 fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Prompt Style', fontsize=15, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=15, fontweight='bold')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'bias_by_prompt_normalized_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Paper-ready normalized bias-by-prompt heatmap saved to: {output_path}")

    # Save raw data
    data_output_path = OUTPUT_DIR / 'bias_by_prompt_normalized_heatmap_data.csv'

    # Create clean output dataframe with normalized values
    output_rows = []
    for i, feature_display in enumerate(pivot_normalized.index):
        # Find original feature name
        feature = next((k for k, v in FEATURE_DISPLAY_NAMES.items() if v == feature_display), feature_display)
        row_data = {'feature': feature, 'feature_display': feature_display}

        # Add normalized values for prompts
        for col in ['Neutral', 'General', 'Popular', 'Engaging', 'Informative', 'Controversial']:
            row_data[f'{col.lower()}_normalized'] = pivot_normalized.loc[feature_display, col]

        output_rows.append(row_data)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(data_output_path, index=False)
    print(f"✓ Raw data saved to: {data_output_path}")

    # Print key findings about prompt sensitivity
    print("\n" + "="*80)
    print("KEY FINDINGS: PROMPT SENSITIVITY (Normalized)")
    print("="*80)

    print("\nFeatures with highest prompt sensitivity (largest z-score range):")
    ranges = pivot_normalized.max(axis=1) - pivot_normalized.min(axis=1)
    ranges = ranges.sort_values(ascending=False)

    for i, (feat, range_val) in enumerate(ranges.head(5).items(), 1):
        min_val = pivot_normalized.loc[feat].min()
        max_val = pivot_normalized.loc[feat].max()
        print(f"  {i}. {feat}: z-score range = {range_val:.2f} (from {min_val:.2f} to {max_val:.2f})")

    print("\nFeatures with lowest prompt sensitivity (smallest z-score range):")
    for i, (feat, range_val) in enumerate(ranges.tail(5).items(), 1):
        min_val = pivot_normalized.loc[feat].min()
        max_val = pivot_normalized.loc[feat].max()
        print(f"  {i}. {feat}: z-score range = {range_val:.2f} (from {min_val:.2f} to {max_val:.2f})")

    return pivot_normalized

# ============================================================================
# PLOT 4-6: SENSITIVE ATTRIBUTE DIRECTIONAL BIAS HEATMAPS
# ============================================================================

def create_sensitive_attribute_heatmaps():
    """
    Create directional bias heatmaps for sensitive attributes:
    - One plot per attribute (3 plots total)
    - Each plot has 3 horizontal subplots (one per dataset)
    - All subplots share the same color scale
    - Dataset labels are color-coded
    - X-labels include pool percentages

    Note: Directional biases are normalized to sum to zero for each combination
    of dataset, provider, and prompt style to ensure mathematical consistency.
    """
    print("\n" + "="*80)
    print("GENERATING SENSITIVE ATTRIBUTE DIRECTIONAL BIAS HEATMAPS")
    print("="*80)

    # Load directional bias data
    dir_bias_path = Path('analysis_outputs/directional_bias_data.csv')
    if not dir_bias_path.exists():
        print(f"ERROR: Could not find {dir_bias_path}")
        return

    dir_bias_df = pd.read_csv(dir_bias_path)

    # Extract and save pool distributions for sensitive attributes
    print("\nExtracting pool distributions for sensitive attributes...")
    categorical_features = ['author_gender', 'author_political_leaning', 'author_is_minority']

    pool_distributions = []
    for feature in categorical_features:
        feature_data = dir_bias_df[dir_bias_df['feature'] == feature].copy()
        # Average prop_pool across providers and prompt styles for each dataset-category
        pool_dist = feature_data.groupby(['dataset', 'category'])['prop_pool'].mean().reset_index()
        pool_dist['feature'] = feature
        pool_distributions.append(pool_dist)

    pool_dist_df = pd.concat(pool_distributions, ignore_index=True)
    pool_dist_path = OUTPUT_DIR / 'pool_distributions_sensitive_attributes.csv'
    pool_dist_df.to_csv(pool_dist_path, index=False)
    print(f"✓ Pool distributions saved to: {pool_dist_path}")

    # Normalize directional biases to sum to zero for categorical features
    # This ensures mathematical consistency (if some categories are over-represented,
    # others must be under-represented by the same total amount)
    print("\nNormalizing directional biases to sum to zero for categorical features...")

    categorical_features = ['author_gender', 'author_political_leaning', 'author_is_minority']
    normalized_rows = []
    corrections_applied = []

    for feature in categorical_features:
        feature_data = dir_bias_df[dir_bias_df['feature'] == feature].copy()

        # Group by dataset, provider, prompt_style and normalize within each group
        for (dataset, provider, prompt), group in feature_data.groupby(['dataset', 'provider', 'prompt_style']):
            bias_sum = group['directional_bias'].sum()

            if abs(bias_sum) > 1e-10:
                # Normalize: subtract equal amount from each category to make sum = 0
                correction = bias_sum / len(group)
                corrections_applied.append((feature, dataset, provider, prompt, bias_sum, correction))
            else:
                correction = 0

            group_normalized = group.copy()
            group_normalized['directional_bias'] = group['directional_bias'] - correction

            # Verify sum is now zero
            new_sum = group_normalized['directional_bias'].sum()
            if abs(new_sum) > 1e-10:
                print(f"  WARNING: {feature}, {dataset}, {provider}, {prompt}: sum = {new_sum:.10f}")

            normalized_rows.append(group_normalized)

    # Replace categorical features with normalized versions
    dir_bias_df = dir_bias_df[~dir_bias_df['feature'].isin(categorical_features)]
    dir_bias_df = pd.concat([dir_bias_df] + normalized_rows, ignore_index=True)

    print(f"  ✓ Normalization complete: {len(corrections_applied)} groups required correction")

    # Show largest corrections
    if corrections_applied:
        corrections_applied.sort(key=lambda x: abs(x[4]), reverse=True)
        print("\n  Largest corrections applied:")
        for i, (feat, ds, prov, prompt, orig_sum, correction) in enumerate(corrections_applied[:5], 1):
            print(f"    {i}. {feat:30s} ({ds}, {prov}, {prompt}): " +
                  f"sum={orig_sum:+.6f}, correction={correction:+.6f}")

    # Define features and their category orderings
    sensitive_features = {
        'author_political_leaning': {
            'display_name': 'Author Political Leaning',
            'categories': ['left', 'center-left', 'center', 'center-right', 'right', 'unknown'],
            'category_labels': ['Left', 'Center-Left', 'Center', 'Center-Right', 'Right', 'Unknown']
        },
        'author_gender': {
            'display_name': 'Author Gender',
            'categories': ['female', 'male', 'non-binary', 'unknown'],
            'category_labels': ['Female', 'Male', 'Non-Binary', 'Unknown']
        },
        'author_is_minority': {
            'display_name': 'Author Minority Status',
            'categories': ['yes', 'no', 'unknown'],
            'category_labels': ['Minority', 'Non-Minority', 'Unknown']
        }
    }

    datasets = ['bluesky', 'reddit', 'twitter']
    dataset_labels = {'twitter': 'Twitter/X', 'bluesky': 'Bluesky', 'reddit': 'Reddit'}
    dataset_colors = {'bluesky': '#2166AC', 'reddit': '#D6604D', 'twitter': '#333333'}
    providers = ['anthropic', 'openai', 'gemini']
    provider_labels = {'anthropic': 'Claude Sonnet 4.5', 'openai': 'GPT-4o-mini', 'gemini': 'Gemini 2.0 Flash'}

    # Create diverging colormap (blue-white-red)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                   '#FFFFFF',  # pure white center
                   '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#DE2D26']
    cmap = LinearSegmentedColormap.from_list('diverging', colors_list, N=256)

    for feature, feature_info in sensitive_features.items():
        print(f"\n{feature_info['display_name']}:")

        # Filter data for this feature
        feature_data = dir_bias_df[dir_bias_df['feature'] == feature].copy()

        # Compute global min/max for shared color scale
        all_data = []
        pivot_data_dict = {}

        for dataset in datasets:
            # Filter for this dataset
            dataset_data = feature_data[feature_data['dataset'] == dataset].copy()

            if len(dataset_data) == 0:
                continue

            # Aggregate over prompt styles (average directional bias with std)
            agg_data_mean = dataset_data.groupby(['provider', 'category'])['directional_bias'].mean().reset_index()
            agg_data_std = dataset_data.groupby(['provider', 'category'])['directional_bias'].std().reset_index()

            # Pivot: rows=providers, columns=categories
            pivot_data = agg_data_mean.pivot(index='provider', columns='category', values='directional_bias')
            pivot_std = agg_data_std.pivot(index='provider', columns='category', values='directional_bias')

            # Reorder columns according to specified category order
            available_cats = [cat for cat in feature_info['categories'] if cat in pivot_data.columns]
            pivot_data = pivot_data[available_cats]
            pivot_std = pivot_std[available_cats]

            # Reorder rows to match provider order
            pivot_data = pivot_data.reindex(providers)
            pivot_std = pivot_std.reindex(providers)

            # Add average row (mean across models with std)
            avg_row = pivot_data.mean(axis=0)
            std_row = pivot_data.std(axis=0)
            avg_series = pd.Series(avg_row, name='Average (across models)')
            std_series = pd.Series(std_row, name='Average (across models)')
            pivot_data = pd.concat([pivot_data, avg_series.to_frame().T])
            # For the average row, store the std in pivot_std
            pivot_std = pd.concat([pivot_std, std_series.to_frame().T])

            pivot_data_dict[dataset] = (pivot_data, pivot_std, agg_data_mean, available_cats)
            all_data.extend(pivot_data.values.flatten())

        # Determine global vmin/vmax (symmetric around 0)
        all_data = [x for x in all_data if not pd.isna(x)]
        max_abs = max(abs(min(all_data)), abs(max(all_data)))

        # Create figure with 3 horizontal subplots (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{feature_info["display_name"]} Directional Bias by Dataset and Model\n' +
                    '(Averaged over all prompt styles)',
                    fontweight='bold', fontsize=20, y=1.02)

        for idx, (dataset, ax) in enumerate(zip(datasets, axes)):
            if dataset not in pivot_data_dict:
                ax.axis('off')
                continue

            pivot_data, pivot_std, agg_data, available_cats = pivot_data_dict[dataset]

            # Get pool distributions for this feature and dataset
            pool_dist_for_feature = pool_dist_df[(pool_dist_df['feature'] == feature) &
                                                 (pool_dist_df['dataset'] == dataset)]

            # Replace column names with display labels + pool percentages
            col_mapping = dict(zip(feature_info['categories'], feature_info['category_labels']))
            pivot_data_display = pivot_data.copy()

            # Create new column labels with percentages
            new_col_labels = []
            for col in pivot_data_display.columns:
                display_label = col_mapping.get(col, col)
                # Get pool percentage for this category
                pool_pct = pool_dist_for_feature[pool_dist_for_feature['category'] == col]['prop_pool'].values
                if len(pool_pct) > 0:
                    pct = pool_pct[0] * 100
                    new_col_labels.append(f'{display_label}\n({pct:.1f}%)')
                else:
                    new_col_labels.append(display_label)

            pivot_data_display.columns = new_col_labels

            # Replace row names with display labels (keep 'Average (across models)' as is)
            new_index = []
            for p in pivot_data_display.index:
                if p == 'Average (across models)':
                    new_index.append(p)
                else:
                    new_index.append(provider_labels.get(p, p))
            pivot_data_display.index = new_index

            # Create annotations with std for average row
            annot = np.empty_like(pivot_data_display, dtype=object)
            for i in range(pivot_data_display.shape[0]):
                row_name = pivot_data_display.index[i]
                for j in range(pivot_data_display.shape[1]):
                    val = pivot_data_display.iloc[i, j]
                    if pd.isna(val):
                        annot[i, j] = ''
                    elif row_name == 'Average (across models)':
                        # Show mean ± std for average row
                        std_val = pivot_std.iloc[i, j]
                        annot[i, j] = f'{val:.3f}\n±{std_val:.3f}'
                    else:
                        annot[i, j] = f'{val:.3f}'

            # Plot heatmap (colorbar only on rightmost subplot)
            heatmap = sns.heatmap(pivot_data_display, annot=annot, fmt='', cmap=cmap,
                       center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                       cbar=(idx == 2),  # Only show colorbar on rightmost subplot
                       cbar_kws={'label': 'Directional Bias\n← Under-represented | Over-represented →'} if idx == 2 else None,
                       linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 11})

            # Increase colorbar label size if it exists
            if idx == 2 and heatmap.collections:
                cbar = ax.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.tick_params(labelsize=12)
                    cbar.set_label('Directional Bias\n← Under-represented | Over-represented →',
                                  fontsize=14, fontweight='bold')

            # Add horizontal line to separate average row from model rows
            n_rows = len(pivot_data_display)
            ax.axhline(y=n_rows-1, color='black', linewidth=2.5)

            # Set dataset label as title with color
            ax.set_title(dataset_labels[dataset], fontsize=18, fontweight='bold',
                        color=dataset_colors[dataset], pad=14)

            # Show x-label on all subplots
            ax.set_xlabel('Category', fontsize=15, fontweight='bold')

            # Only show y-label on leftmost subplot
            if idx == 0:
                ax.set_ylabel('Model', fontsize=15, fontweight='bold')
            else:
                ax.set_ylabel('')

            # Increase tick label sizes and rotate x labels if needed
            ax.tick_params(axis='both', which='major', labelsize=13)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0, ha='right')

            # Only show y-tick labels on leftmost subplot
            if idx > 0:
                ax.set_yticklabels([])

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save plot
        plot_filename = f'{feature}_by_dataset_heatmap.png'
        output_path = OUTPUT_DIR / plot_filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  ✓ Saved: {plot_filename}")

        # Save raw data for all datasets in one CSV
        csv_filename = f'{feature}_by_dataset_data.csv'
        csv_path = OUTPUT_DIR / csv_filename

        all_rows = []
        for dataset in datasets:
            if dataset not in pivot_data_dict:
                continue

            pivot_data, pivot_std, agg_data, available_cats = pivot_data_dict[dataset]

            # Add individual model rows
            for provider in providers:
                row_data = {'dataset': dataset, 'provider': provider}
                for cat in available_cats:
                    row_data[cat] = pivot_data.loc[provider, cat] if provider in pivot_data.index else np.nan
                    row_data[f'{cat}_std'] = pivot_std.loc[provider, cat] if provider in pivot_std.index else np.nan
                all_rows.append(row_data)

            # Add average row if it exists
            if 'Average (across models)' in pivot_data.index:
                row_data = {'dataset': dataset, 'provider': 'Average (across models)'}
                for cat in available_cats:
                    row_data[cat] = pivot_data.loc['Average (across models)', cat]
                    row_data[f'{cat}_std'] = pivot_std.loc['Average (across models)', cat]
                all_rows.append(row_data)

        output_df = pd.DataFrame(all_rows)
        output_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_filename}")

    print("\n✓ All sensitive attribute heatmaps generated!")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("GENERATING PAPER-READY PLOTS")
    print("="*80)

    # Load comprehensive comparison data
    comp_path = Path('analysis_outputs/pool_vs_recommended_summary.csv')
    if not comp_path.exists():
        print(f"ERROR: Could not find {comp_path}")
        print("Please run the comprehensive analysis first.")
        return

    print(f"\nLoading data from: {comp_path}")
    comp_df = pd.read_csv(comp_path)

    # Convert to R²
    print("Converting bias metrics to R²...")
    comp_df['r_squared'] = comp_df.apply(convert_to_r_squared, axis=1)
    comp_df = comp_df.dropna(subset=['r_squared'])

    print(f"Loaded {len(comp_df)} rows")

    # Generate plots
    agg_data = create_aggregated_bar_plot_ordered(comp_df)
    heatmap_data = create_bias_by_prompt_heatmap(comp_df)
    normalized_heatmap_data = create_normalized_bias_by_prompt_heatmap(comp_df)

    # Generate sensitive attribute heatmaps
    create_sensitive_attribute_heatmaps()

    print("\n" + "="*80)
    print("ALL PAPER PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Print statistics summary for figure caption
    print("\n" + "="*80)
    print("STATISTICS SUMMARY FOR FIGURE CAPTION")
    print("="*80)
    print("\nTop 5 features by R²:")
    for i, row in agg_data.head(5).iterrows():
        sig_pct = row['significant'] * 100
        print(f"  {row['feature_display']:35s} R²={row['r_squared']:.4f} ({sig_pct:.1f}% significant)")

    print("\nDemographic features:")
    demo_features = agg_data[agg_data['category'] == 'author']
    for i, row in demo_features.iterrows():
        sig_pct = row['significant'] * 100
        print(f"  {row['feature_display']:35s} R²={row['r_squared']:.4f} ({sig_pct:.1f}% significant)")

    print("\nStyle features:")
    style_features = agg_data[agg_data['category'] == 'style']
    for i, row in style_features.iterrows():
        sig_pct = row['significant'] * 100
        print(f"  {row['feature_display']:35s} R²={row['r_squared']:.4f} ({sig_pct:.1f}% significant)")

if __name__ == '__main__':
    main()
