"""
Interactive Dashboards for LLM Recommendation Bias Analysis

This script creates interactive Plotly HTML dashboards:
1. Stratified Analysis Dashboard (single experiment)
2. Cross-Experiment Dashboard (all experiments)

Usage:
    # Create dashboard for single experiment
    python create_dashboards.py --experiment-dir outputs/experiments/bluesky_anthropic_claude-sonnet-4-5-20250929 --output single_dashboard.html

    # Create cross-experiment dashboard
    python create_dashboards.py --experiments-dir outputs/experiments --meta-dir outputs/meta_analysis --output cross_dashboard.html
"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_stratified_dashboard(experiment_dir, output_file):
    """
    Create interactive dashboard for a single experiment with tabs:
    1. Bias by Style - Bar charts comparing pool vs recommended
    2. Feature Importance Heatmap - Features × Styles
    3. Coefficient Comparison - Coefficients across styles with error bars
    """
    exp_path = Path(experiment_dir)
    strat_dir = exp_path / "stratified_analysis"

    if not strat_dir.exists():
        raise ValueError(f"No stratified_analysis directory found in {experiment_dir}")

    print(f"\n{'='*60}")
    print(f"Creating Stratified Analysis Dashboard")
    print(f"Experiment: {exp_path.name}")
    print(f"{'='*60}")

    # Load data
    bias_file = strat_dir / "comparison" / "bias_by_style.csv"
    coef_file = strat_dir / "comparison" / "coefficient_comparison.csv"
    feature_imp_file = strat_dir / "comparison" / "feature_importance.csv"

    if not all([bias_file.exists(), coef_file.exists(), feature_imp_file.exists()]):
        raise ValueError(f"Missing required files in {strat_dir}")

    df_bias = pd.read_csv(bias_file)
    df_coef = pd.read_csv(coef_file)
    df_feat = pd.read_csv(feature_imp_file)

    # Create figure with subplots (tabs implemented via visibility)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Bias by Style", "Feature Importance (Top 15)", "Coefficient Comparison (Top 20 Significant)"),
        specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}]],
        vertical_spacing=0.15
    )

    # Tab 1: Bias by Style
    styles = sorted(df_bias['prompt_style'].unique())
    for style in styles:
        style_data = df_bias[df_bias['prompt_style'] == style].iloc[0]

        # Convert to float to ensure formatting works
        difference = float(style_data['difference'])
        test_stat = float(style_data['test_stat'])
        p_value = float(style_data['p_value'])

        fig.add_trace(
            go.Bar(
                name=style,
                x=['Pool Mean', 'Recommended Mean'],
                y=[style_data['pool_mean'], style_data['recommended_mean']],
                text=[f"{float(style_data['pool_mean']):.3f}", f"{float(style_data['recommended_mean']):.3f}"],
                textposition='auto',
                hovertemplate=f"<b>{style}</b><br>" +
                              "Value: %{y:.3f}<br>" +
                              f"Difference: {difference:.3f}<br>" +
                              f"Test Statistic: {test_stat:.3f}<br>" +
                              f"P-value: {p_value:.4f}<extra></extra>",
                visible=True
            ),
            row=1, col=1
        )

    # Tab 2: Feature Importance (overall across styles)
    df_feat_sorted = df_feat.sort_values('avg_abs_coefficient', ascending=False).head(15)

    fig.add_trace(
        go.Bar(
            x=df_feat_sorted['feature'],
            y=df_feat_sorted['avg_abs_coefficient'],
            marker_color='steelblue',
            hovertemplate="<b>%{x}</b><br>Avg |Coefficient|: %{y:.3f}<extra></extra>",
            visible=True,
            showlegend=False
        ),
        row=2, col=1
    )

    # Tab 3: Coefficient Comparison (show significant differences)
    df_coef_sig = df_coef[df_coef['p_value'] < 0.05].head(20)  # Top 20 significant

    if len(df_coef_sig) > 0:
        df_coef_sig['comparison'] = df_coef_sig['feature'] + '<br>(' + df_coef_sig['style1'] + ' vs ' + df_coef_sig['style2'] + ')'

        fig.add_trace(
            go.Bar(
                x=df_coef_sig['comparison'],
                y=df_coef_sig['diff'],
                marker_color=['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow'
                              for p in df_coef_sig['p_value']],
                hovertemplate="<b>%{x}</b><br>" +
                              "Coefficient Difference: %{y:.3f}<br>" +
                              "Z-statistic: " + df_coef_sig['z_stat'].apply(lambda x: f"{x:.3f}").values + "<br>" +
                              "P-value: " + df_coef_sig['p_value'].apply(lambda x: f"{x:.4f}").values + "<extra></extra>",
                visible=True,
                showlegend=False
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Stratified Analysis Dashboard</b><br><sub>{exp_path.name}</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=1800,
        showlegend=True,
        template='plotly_white',
        hovermode='closest'
    )

    fig.update_xaxes(title_text="Metric", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)

    fig.update_xaxes(title_text="Feature", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Avg |Coefficient|", row=2, col=1)

    fig.update_xaxes(title_text="Comparison", row=3, col=1, tickangle=45)
    fig.update_yaxes(title_text="Coefficient Difference", row=3, col=1)

    # Save
    output_path = Path(output_file)
    fig.write_html(str(output_path))
    print(f"\n✓ Dashboard saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return str(output_path)


def create_cross_experiment_dashboard(experiments_dir, meta_dir, output_file):
    """
    Create cross-experiment dashboard with tabs:
    1. Cross-Model Comparison - Compare OpenAI/Anthropic/Gemini
    2. Cross-Dataset Comparison - Compare Twitter/Reddit/Bluesky
    3. Meta-Analysis View - Forest plots of pooled effects
    """
    meta_path = Path(meta_dir)

    if not meta_path.exists():
        raise ValueError(f"Meta-analysis directory not found: {meta_dir}")

    print(f"\n{'='*60}")
    print(f"Creating Cross-Experiment Dashboard")
    print(f"{'='*60}")

    # Load meta-analysis results
    dataset_file = meta_path / "by_dataset" / "dataset_comparison.csv"
    model_file = meta_path / "by_model" / "model_comparison.csv"
    meta_file = meta_path / "meta_analysis" / "meta_effect_sizes.csv"

    if not all([dataset_file.exists(), model_file.exists(), meta_file.exists()]):
        raise ValueError(f"Missing meta-analysis files in {meta_dir}")

    df_dataset = pd.read_csv(dataset_file)
    df_model = pd.read_csv(model_file)
    df_meta = pd.read_csv(meta_file)

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Cross-Dataset Comparison (Twitter vs Reddit vs Bluesky)",
            "Cross-Model Comparison (OpenAI vs Anthropic vs Gemini)",
            "Meta-Analysis: Universal Predictors"
        ),
        specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "scatter"}]],
        vertical_spacing=0.12
    )

    # Tab 1: Cross-Dataset Comparison (top 15 significant)
    df_dataset_sig = df_dataset[df_dataset['p_value'] < 0.05].sort_values('f_statistic', ascending=False).head(15)

    if len(df_dataset_sig) > 0:
        df_dataset_sig['label'] = df_dataset_sig['prompt_style'] + '<br>' + df_dataset_sig['feature']

        colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow'
                  for p in df_dataset_sig['p_value']]

        fig.add_trace(
            go.Bar(
                x=df_dataset_sig['label'],
                y=df_dataset_sig['f_statistic'],
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>" +
                              "F-statistic: %{y:.2f}<br>" +
                              "P-value: " + df_dataset_sig['p_value'].apply(lambda x: f"{x:.6f}").values +
                              "<br>Datasets differ significantly<extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )

    # Tab 2: Cross-Model Comparison (show all comparisons, highlight significant)
    df_model_sorted = df_model.sort_values('f_statistic', ascending=False).head(20)

    if len(df_model_sorted) > 0:
        df_model_sorted['label'] = df_model_sorted['prompt_style'] + '<br>' + df_model_sorted['feature']

        colors = ['red' if p < 0.05 else 'lightgray' for p in df_model_sorted['p_value']]

        fig.add_trace(
            go.Bar(
                x=df_model_sorted['label'],
                y=df_model_sorted['f_statistic'],
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>" +
                              "F-statistic: %{y:.2f}<br>" +
                              "P-value: " + df_model_sorted['p_value'].apply(lambda x: f"{x:.6f}").values +
                              "<br>Models " + ["DIFFER" if p < 0.05 else "consistent"
                                              for p in df_model_sorted['p_value']] + "<extra></extra>",
                showlegend=False
            ),
            row=2, col=1
        )

    # Tab 3: Meta-Analysis Forest Plot (top 20 universal effects)
    df_meta_sig = df_meta[df_meta['p_value'] < 0.05].sort_values('pooled_coefficient', ascending=False).head(20)

    if len(df_meta_sig) > 0:
        df_meta_sig['label'] = df_meta_sig['prompt_style'] + ': ' + df_meta_sig['feature']
        df_meta_sig['ci_lower'] = df_meta_sig['pooled_coefficient'] - 1.96 * df_meta_sig['pooled_se']
        df_meta_sig['ci_upper'] = df_meta_sig['pooled_coefficient'] + 1.96 * df_meta_sig['pooled_se']

        # Color by heterogeneity
        colors = ['red' if i2 > 75 else 'orange' if i2 > 50 else 'green'
                  for i2 in df_meta_sig['I_squared']]

        fig.add_trace(
            go.Scatter(
                x=df_meta_sig['pooled_coefficient'],
                y=df_meta_sig['label'],
                mode='markers',
                marker=dict(size=10, color=colors),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=df_meta_sig['ci_upper'] - df_meta_sig['pooled_coefficient'],
                    arrayminus=df_meta_sig['pooled_coefficient'] - df_meta_sig['ci_lower']
                ),
                hovertemplate="<b>%{y}</b><br>" +
                              "Pooled Coefficient: %{x:.3f}<br>" +
                              "95% CI: [" + df_meta_sig['ci_lower'].apply(lambda x: f"{x:.3f}").values + ", " +
                              df_meta_sig['ci_upper'].apply(lambda x: f"{x:.3f}").values + "]<br>" +
                              "P-value: " + df_meta_sig['p_value'].apply(lambda x: f"{x:.6f}").values + "<br>" +
                              "I²: " + df_meta_sig['I_squared'].apply(lambda x: f"{x:.1f}%").values +
                              "<br>Heterogeneity: " +
                              ["High (>75%)" if i2 > 75 else "Moderate (50-75%)" if i2 > 50 else "Low (<50%)"
                               for i2 in df_meta_sig['I_squared']] + "<extra></extra>",
                showlegend=False
            ),
            row=3, col=1
        )

        # Add vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Cross-Experiment Dashboard</b><br><sub>Meta-Analysis Across All Experiments</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=1800,
        template='plotly_white',
        hovermode='closest'
    )

    fig.update_xaxes(title_text="Comparison", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="F-statistic", row=1, col=1)

    fig.update_xaxes(title_text="Comparison", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="F-statistic", row=2, col=1)

    fig.update_xaxes(title_text="Pooled Coefficient (95% CI)", row=3, col=1)
    fig.update_yaxes(title_text="Feature", row=3, col=1)

    # Save
    output_path = Path(output_file)
    fig.write_html(str(output_path))
    print(f"\n✓ Dashboard saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return str(output_path)


def create_all_dashboards(experiments_dir, meta_dir, output_dir):
    """
    Create dashboards for all experiments plus cross-experiment dashboard.
    """
    exp_path = Path(experiments_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Creating All Dashboards")
    print(f"{'='*60}")

    dashboards = []

    # Create dashboard for each experiment
    for exp_dir in sorted(exp_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        strat_dir = exp_dir / "stratified_analysis"
        if not strat_dir.exists():
            print(f"Skipping {exp_dir.name} (no stratified analysis)")
            continue

        try:
            output_file = output_path / f"dashboard_{exp_dir.name}.html"
            create_stratified_dashboard(exp_dir, output_file)
            dashboards.append(output_file)
        except Exception as e:
            print(f"Error creating dashboard for {exp_dir.name}: {e}")

    # Create cross-experiment dashboard
    if meta_dir:
        try:
            cross_output = output_path / "dashboard_cross_experiment.html"
            create_cross_experiment_dashboard(experiments_dir, meta_dir, cross_output)
            dashboards.append(cross_output)
        except Exception as e:
            print(f"Error creating cross-experiment dashboard: {e}")

    print(f"\n{'='*60}")
    print(f"✓ All dashboards created!")
    print(f"  Total dashboards: {len(dashboards)}")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}\n")

    return dashboards


def main():
    parser = argparse.ArgumentParser(description="Create interactive dashboards for LLM bias analysis")
    parser.add_argument('--experiment-dir', type=str, help='Single experiment directory')
    parser.add_argument('--experiments-dir', type=str, help='Directory containing all experiments')
    parser.add_argument('--meta-dir', type=str, help='Meta-analysis directory')
    parser.add_argument('--output', type=str, help='Output HTML file')
    parser.add_argument('--output-dir', type=str, default='outputs/dashboards',
                        help='Output directory for all dashboards')
    parser.add_argument('--all', action='store_true', help='Create dashboards for all experiments')

    args = parser.parse_args()

    if args.all:
        if not args.experiments_dir:
            raise ValueError("--experiments-dir required when using --all")
        create_all_dashboards(args.experiments_dir, args.meta_dir, args.output_dir)

    elif args.experiment_dir:
        if not args.output:
            raise ValueError("--output required when using --experiment-dir")
        create_stratified_dashboard(args.experiment_dir, args.output)

    elif args.experiments_dir and args.meta_dir:
        if not args.output:
            raise ValueError("--output required when creating cross-experiment dashboard")
        create_cross_experiment_dashboard(args.experiments_dir, args.meta_dir, args.output)

    else:
        parser.print_help()
        raise ValueError("Must specify either --all, --experiment-dir, or --experiments-dir with --meta-dir")


if __name__ == "__main__":
    main()
