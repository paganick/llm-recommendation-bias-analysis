"""
Generate Standalone Interactive HTML Visualizations
Creates self-contained HTML files using Plotly that can be opened in any browser
No server required - fully interactive with zoom, pan, hover tooltips, and filtering
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Configuration
OUTPUT_DIR = Path('analysis_outputs/interactive_html')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INTERACTIVE HTML VISUALIZATION GENERATOR")
print("=" * 80)

# Load data
print("\nLoading data...")
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results.parquet')
importance_df = pd.read_parquet('analysis_outputs/importance_analysis/importance_results.parquet')
print(f"✓ Loaded {len(bias_df)} bias metrics")
print(f"✓ Loaded {len(importance_df)} importance results")

# Extract unique values
datasets = sorted(bias_df['dataset'].unique())
models = sorted(bias_df['model'].unique())
prompts = sorted(bias_df['prompt_style'].unique())
features = sorted(bias_df['feature'].unique())

print(f"\n✓ {len(datasets)} datasets, {len(models)} models, {len(prompts)} prompts, {len(features)} features")


def create_bias_heatmap_interactive():
    """
    Interactive 1: Comprehensive Bias Heatmap
    Shows all features × conditions with filtering capabilities
    """
    print("\n[1/6] Creating interactive bias heatmap...")

    # Create pivot table for heatmap
    # Aggregate by feature and model
    pivot_data = bias_df.groupby(['feature', 'model'])['bias'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='feature', columns='model', values='bias')

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(pivot_table.values, 3),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Bias<br>(Effect Size)"),
        hoverongaps=False,
        hovertemplate='Model: %{x}<br>Feature: %{y}<br>Bias: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': 'Interactive Bias Heatmap: Average Bias by Feature and Model<br><sub>Click and drag to zoom | Double-click to reset | Hover for details</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Model',
        yaxis_title='Feature',
        height=1200,
        width=1400,
        font=dict(size=10),
        hovermode='closest'
    )

    output_file = OUTPUT_DIR / 'interactive_bias_heatmap.html'
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_feature_importance_interactive():
    """
    Interactive 2: Feature Importance Across All Conditions
    Bar chart with dropdown to select condition
    """
    print("\n[2/6] Creating interactive feature importance chart...")

    # Get coefficient columns
    coef_cols = [c for c in importance_df.columns if c.startswith('coef_')]
    feature_names = [c.replace('coef_', '') for c in coef_cols]

    # Create figure
    fig = go.Figure()

    # Add traces for each condition
    for idx, (_, row) in enumerate(importance_df.iterrows()):
        condition_name = f"{row['dataset']} × {row['model']} × {row['prompt_style']}"

        # Get coefficients and sort by absolute value
        coefs = {feat: row[f'coef_{feat}'] for feat in feature_names}
        sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

        features_list = [f for f, _ in sorted_coefs]
        values_list = [v for _, v in sorted_coefs]
        colors = ['green' if v > 0 else 'red' for v in values_list]

        fig.add_trace(go.Bar(
            name=condition_name,
            x=values_list,
            y=features_list,
            orientation='h',
            marker=dict(color=colors),
            visible=(idx == 0),  # Only first trace visible initially
            hovertemplate='Feature: %{y}<br>Coefficient: %{x:.4f}<extra></extra>'
        ))

    # Create dropdown menu
    buttons = []
    for idx, (_, row) in enumerate(importance_df.iterrows()):
        condition_name = f"{row['dataset']} × {row['model']} × {row['prompt_style']}"
        auroc = row['auroc']

        visible = [False] * len(importance_df)
        visible[idx] = True

        buttons.append(
            dict(
                label=f"{condition_name} (AUROC: {auroc:.3f})",
                method='update',
                args=[{'visible': visible},
                      {'title': f'Top 20 Features for: {condition_name}<br><sub>AUROC: {auroc:.3f} | Green=Positive, Red=Negative</sub>'}]
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title='Top 20 Features by Logistic Regression Coefficient<br><sub>Use dropdown to select condition</sub>',
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        height=800,
        width=1200,
        showlegend=False,
        hovermode='closest'
    )

    output_file = OUTPUT_DIR / 'interactive_feature_importance.html'
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_model_comparison_interactive():
    """
    Interactive 3: Model Comparison
    Box plots showing bias distribution across models
    """
    print("\n[3/6] Creating interactive model comparison...")

    # Create subplots for top 6 features
    top_features = bias_df.groupby('feature')['bias'].mean().abs().nlargest(6).index.tolist()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'<b>{feat}</b>' for feat in top_features],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, feature in enumerate(top_features):
        row = (idx // 3) + 1
        col = (idx % 3) + 1

        feature_df = bias_df[bias_df['feature'] == feature]

        for dataset in datasets:
            dataset_df = feature_df[feature_df['dataset'] == dataset]

            for model in models:
                model_df = dataset_df[dataset_df['model'] == model]

                if len(model_df) > 0:
                    fig.add_trace(
                        go.Box(
                            y=model_df['bias'],
                            name=f'{model}<br>{dataset}',
                            boxmean='sd',
                            hovertemplate='%{y:.3f}<extra></extra>',
                            legendgroup=f'{model}_{dataset}',
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )

        # Update axes
        fig.update_xaxes(title_text='', row=row, col=col, showticklabels=False)
        fig.update_yaxes(title_text='Bias', row=row, col=col)

    fig.update_layout(
        title={
            'text': 'Model Comparison: Bias Distribution for Top 6 Features<br><sub>Hover for values | Click legend to filter</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=900,
        width=1600,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    output_file = OUTPUT_DIR / 'interactive_model_comparison.html'
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_dataset_comparison_interactive():
    """
    Interactive 4: Dataset (Platform) Comparison
    Grouped bar chart showing bias by platform
    """
    print("\n[4/6] Creating interactive dataset comparison...")

    # Get top 10 features
    top_features = bias_df.groupby('feature')['bias'].mean().abs().nlargest(10).index.tolist()

    # Aggregate data
    plot_data = []
    for feature in top_features:
        for dataset in datasets:
            dataset_df = bias_df[(bias_df['feature'] == feature) & (bias_df['dataset'] == dataset)]
            mean_bias = dataset_df['bias'].mean()
            std_bias = dataset_df['bias'].std()
            n_sig = dataset_df['significant'].sum()
            n_total = len(dataset_df)

            plot_data.append({
                'feature': feature,
                'dataset': dataset,
                'mean_bias': mean_bias,
                'std_bias': std_bias,
                'pct_sig': (n_sig / n_total * 100) if n_total > 0 else 0
            })

    plot_df = pd.DataFrame(plot_data)

    # Create grouped bar chart
    fig = go.Figure()

    for dataset in datasets:
        dataset_data = plot_df[plot_df['dataset'] == dataset]

        fig.add_trace(go.Bar(
            name=dataset.capitalize(),
            x=dataset_data['feature'],
            y=dataset_data['mean_bias'],
            error_y=dict(type='data', array=dataset_data['std_bias']),
            hovertemplate='<b>%{x}</b><br>Mean Bias: %{y:.3f}<br>% Significant: %{customdata:.1f}%<extra></extra>',
            customdata=dataset_data['pct_sig']
        ))

    fig.update_layout(
        title='Platform Comparison: Mean Bias for Top 10 Features<br><sub>Error bars show standard deviation | Hover for details</sub>',
        xaxis_title='Feature',
        yaxis_title='Mean Bias (Effect Size)',
        barmode='group',
        height=600,
        width=1400,
        hovermode='closest',
        legend=dict(title='Platform')
    )

    output_file = OUTPUT_DIR / 'interactive_dataset_comparison.html'
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_prompt_comparison_interactive():
    """
    Interactive 5: Prompt Style Comparison
    Shows how different prompts affect bias
    """
    print("\n[5/6] Creating interactive prompt comparison...")

    # Get top 8 features
    top_features = bias_df.groupby('feature')['bias'].mean().abs().nlargest(8).index.tolist()

    # Create data for heatmap
    heatmap_data = []
    for feature in top_features:
        row_data = []
        for prompt in prompts:
            prompt_df = bias_df[(bias_df['feature'] == feature) & (bias_df['prompt_style'] == prompt)]
            mean_bias = prompt_df['bias'].mean()
            row_data.append(mean_bias)
        heatmap_data.append(row_data)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=prompts,
        y=top_features,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(heatmap_data, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Mean Bias"),
        hovertemplate='Prompt: %{x}<br>Feature: %{y}<br>Mean Bias: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Prompt Style Sensitivity: Mean Bias for Top 8 Features<br><sub>Shows how prompt formulation affects bias magnitude</sub>',
        xaxis_title='Prompt Style',
        yaxis_title='Feature',
        height=600,
        width=1200,
        hovermode='closest'
    )

    output_file = OUTPUT_DIR / 'interactive_prompt_comparison.html'
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_summary_dashboard():
    """
    Interactive 6: Summary Dashboard
    Multiple panels showing key insights
    """
    print("\n[6/6] Creating interactive summary dashboard...")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Bias Significance Rate by Model</b>',
            '<b>AUROC Distribution by Model</b>',
            '<b>Top 10 Features by Mean Bias</b>',
            '<b>Bias Magnitude Distribution</b>'
        ),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'bar'}, {'type': 'histogram'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # Panel 1: Significance rate by model
    sig_by_model = bias_df.groupby('model').agg({
        'significant': lambda x: (x.sum() / len(x) * 100)
    }).reset_index()
    sig_by_model.columns = ['model', 'pct_sig']

    fig.add_trace(
        go.Bar(
            x=sig_by_model['model'],
            y=sig_by_model['pct_sig'],
            marker_color='steelblue',
            hovertemplate='%{x}<br>% Significant: %{y:.1f}%<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text='Model', row=1, col=1)
    fig.update_yaxes(title_text='% Significant Tests', row=1, col=1)

    # Panel 2: AUROC by model
    for model in models:
        model_df = importance_df[importance_df['model'] == model]
        fig.add_trace(
            go.Box(
                y=model_df['auroc'],
                name=model,
                boxmean='sd',
                hovertemplate='AUROC: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    fig.update_xaxes(title_text='Model', row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text='AUROC', row=1, col=2)

    # Panel 3: Top features
    top_features_data = bias_df.groupby('feature')['bias'].mean().abs().nlargest(10).reset_index()
    top_features_data.columns = ['feature', 'mean_abs_bias']

    fig.add_trace(
        go.Bar(
            x=top_features_data['mean_abs_bias'],
            y=top_features_data['feature'],
            orientation='h',
            marker_color='coral',
            hovertemplate='%{y}<br>Mean |Bias|: %{x:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.update_xaxes(title_text='Mean Absolute Bias', row=2, col=1)
    fig.update_yaxes(title_text='Feature', row=2, col=1)

    # Panel 4: Bias distribution
    fig.add_trace(
        go.Histogram(
            x=bias_df['bias'],
            nbinsx=50,
            marker_color='lightgreen',
            hovertemplate='Bias: %{x:.2f}<br>Count: %{y}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text='Bias (Effect Size)', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=2)

    # Overall layout
    fig.update_layout(
        title={
            'text': 'LLM Recommendation Bias: Summary Dashboard<br><sub>Interactive overview of key findings</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=900,
        width=1600,
        hovermode='closest',
        showlegend=True
    )

    output_file = OUTPUT_DIR / 'interactive_summary_dashboard.html'
    fig.write_html(output_file)
    print(f"  ✓ Saved: {output_file}")
    return output_file


def create_index_html():
    """Create an index page linking to all visualizations"""
    print("\nCreating index page...")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Recommendation Bias - Interactive Visualizations</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .subtitle {
            margin-top: 10px;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .viz-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .viz-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .viz-card h3 {
            color: #667eea;
            margin-top: 0;
            font-size: 1.4em;
        }
        .viz-card p {
            color: #666;
            margin-bottom: 20px;
        }
        .viz-card a {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: opacity 0.3s ease;
        }
        .viz-card a:hover {
            opacity: 0.9;
        }
        .stats {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stats h2 {
            color: #667eea;
            margin-top: 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 LLM Recommendation Bias Analysis</h1>
        <div class="subtitle">Interactive Visualizations - Standalone HTML Files</div>
    </div>

    <div class="stats">
        <h2>📊 Analysis Summary</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-number">1,404</div>
                <div class="stat-label">Bias Metrics</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">73.9%</div>
                <div class="stat-label">Significant Tests</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">54</div>
                <div class="stat-label">Conditions Tested</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">0.717</div>
                <div class="stat-label">Mean AUROC</div>
            </div>
        </div>
    </div>

    <div class="viz-grid">
        <div class="viz-card">
            <h3>📈 Summary Dashboard</h3>
            <p>Multi-panel overview showing significance rates, AUROC distribution, top features, and bias distribution.</p>
            <a href="interactive_summary_dashboard.html" target="_blank">Open Dashboard →</a>
        </div>

        <div class="viz-card">
            <h3>🔥 Bias Heatmap</h3>
            <p>Comprehensive heatmap showing average bias across all features and models. Zoom and pan to explore patterns.</p>
            <a href="interactive_bias_heatmap.html" target="_blank">Open Heatmap →</a>
        </div>

        <div class="viz-card">
            <h3>⭐ Feature Importance</h3>
            <p>Top 20 features by logistic regression coefficient for each condition. Use dropdown to switch between conditions.</p>
            <a href="interactive_feature_importance.html" target="_blank">Open Importance →</a>
        </div>

        <div class="viz-card">
            <h3>🤖 Model Comparison</h3>
            <p>Box plots comparing bias distribution across models for top features. Click legend to filter.</p>
            <a href="interactive_model_comparison.html" target="_blank">Open Comparison →</a>
        </div>

        <div class="viz-card">
            <h3>🌐 Platform Analysis</h3>
            <p>Grouped bar chart showing how bias varies across platforms (Twitter, Reddit, Bluesky) for top features.</p>
            <a href="interactive_dataset_comparison.html" target="_blank">Open Platform →</a>
        </div>

        <div class="viz-card">
            <h3>💬 Prompt Sensitivity</h3>
            <p>Heatmap showing how different prompt styles affect bias magnitude across top features.</p>
            <a href="interactive_prompt_comparison.html" target="_blank">Open Prompts →</a>
        </div>
    </div>

    <div style="margin-top: 40px; padding: 20px; background: white; border-radius: 10px; text-align: center;">
        <h3 style="color: #667eea;">💡 How to Use</h3>
        <p style="color: #666;">
            All visualizations are fully interactive:<br>
            🖱️ <strong>Hover</strong> over elements for detailed tooltips<br>
            🔍 <strong>Click and drag</strong> to zoom into areas of interest<br>
            🔄 <strong>Double-click</strong> to reset zoom<br>
            👆 <strong>Click legend items</strong> to show/hide data series<br>
            📥 Use the camera icon (top-right of each viz) to download as PNG
        </p>
    </div>
</body>
</html>"""

    output_file = OUTPUT_DIR / 'index.html'
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"  ✓ Saved: {output_file}")
    return output_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\nGenerating interactive visualizations...\n")

    files_created = []

    try:
        # Generate all visualizations
        files_created.append(create_bias_heatmap_interactive())
        files_created.append(create_feature_importance_interactive())
        files_created.append(create_model_comparison_interactive())
        files_created.append(create_dataset_comparison_interactive())
        files_created.append(create_prompt_comparison_interactive())
        files_created.append(create_summary_dashboard())

        # Create index page
        index_file = create_index_html()

        print("\n" + "=" * 80)
        print("INTERACTIVE VISUALIZATIONS COMPLETE")
        print("=" * 80)
        print(f"\n✓ Created 6 interactive HTML visualizations")
        print(f"✓ All files saved to: {OUTPUT_DIR}")
        print(f"\n📂 Output Directory: {OUTPUT_DIR.absolute()}")
        print(f"\n🌐 Open this file to start: {index_file.absolute()}")
        print("\nAll visualizations are standalone HTML files that work in any browser.")
        print("No server required - just open the files directly!")

        # Summary
        print("\n" + "=" * 80)
        print("FILES CREATED:")
        print("=" * 80)
        print(f"\n  📑 index.html                          - Main landing page")
        print(f"  📈 interactive_summary_dashboard.html  - Multi-panel overview")
        print(f"  🔥 interactive_bias_heatmap.html       - Comprehensive bias heatmap")
        print(f"  ⭐ interactive_feature_importance.html - Feature importance by condition")
        print(f"  🤖 interactive_model_comparison.html   - Model-to-model comparison")
        print(f"  🌐 interactive_dataset_comparison.html - Platform analysis")
        print(f"  💬 interactive_prompt_comparison.html  - Prompt sensitivity analysis")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
