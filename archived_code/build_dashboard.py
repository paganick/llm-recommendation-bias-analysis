"""
Interactive Dashboard for LLM Recommendation Bias Analysis
Built with Plotly Dash

Following: NEXT_SESSION_GUIDE.md Step 2

To run: python build_dashboard.py
Then open: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
print("Loading data...")
bias_df = pd.read_parquet('analysis_outputs/bias_analysis/bias_results.parquet')
importance_df = pd.read_parquet('analysis_outputs/importance_analysis/importance_results.parquet')
print(f"✓ Loaded {len(bias_df)} bias metrics")
print(f"✓ Loaded {len(importance_df)} importance results")

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "LLM Recommendation Bias Analysis"

# Styling
COLORS = {
    'background': '#f8f9fa',
    'text': '#212529',
    'primary': '#0d6efd',
    'secondary': '#6c757d',
    'success': '#198754',
    'danger': '#dc3545',
    'warning': '#ffc107'
}

# Common styles
CARD_STYLE = {
    'background': 'white',
    'padding': '20px',
    'margin': '10px',
    'borderRadius': '5px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
}

TAB_STYLE = {
    'padding': '10px',
    'fontWeight': 'bold'
}

TAB_SELECTED_STYLE = {
    'padding': '10px',
    'fontWeight': 'bold',
    'backgroundColor': COLORS['primary'],
    'color': 'white'
}

# Extract unique values for filters
datasets = sorted(bias_df['dataset'].unique())
models = sorted(bias_df['model'].unique())
prompts = sorted(bias_df['prompt_style'].unique())
features = sorted(bias_df['feature'].unique())

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("LLM Recommendation Bias Analysis Dashboard",
                style={'textAlign': 'center', 'color': COLORS['primary'], 'marginBottom': '10px'}),
        html.P("Interactive exploration of bias patterns across models, datasets, and prompts",
               style={'textAlign': 'center', 'color': COLORS['secondary'], 'marginBottom': '20px'}),
    ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

    # Tabs
    dcc.Tabs(id='main-tabs', value='tab-bias', children=[
        dcc.Tab(label='Bias Explorer', value='tab-bias', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Feature Importance', value='tab-importance', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Model Comparison', value='tab-models', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Dataset Analysis', value='tab-datasets', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Statistical Tables', value='tab-tables', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
    ], style={'marginBottom': '20px'}),

    # Tab content
    html.Div(id='tab-content', style={'padding': '20px'})
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})


# ============================================================================
# TAB 1: BIAS EXPLORER
# ============================================================================

def render_bias_explorer():
    return html.Div([
        html.H2("Bias Explorer", style={'color': COLORS['primary']}),

        # Filters
        html.Div([
            html.Div([
                html.Label("Dataset:"),
                dcc.Dropdown(
                    id='bias-dataset-filter',
                    options=[{'label': 'All', 'value': 'all'}] + [{'label': d, 'value': d} for d in datasets],
                    value='all',
                    clearable=False
                )
            ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

            html.Div([
                html.Label("Model:"),
                dcc.Dropdown(
                    id='bias-model-filter',
                    options=[{'label': 'All', 'value': 'all'}] + [{'label': m, 'value': m} for m in models],
                    value='all',
                    clearable=False
                )
            ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

            html.Div([
                html.Label("Prompt Style:"),
                dcc.Dropdown(
                    id='bias-prompt-filter',
                    options=[{'label': 'All', 'value': 'all'}] + [{'label': p, 'value': p} for p in prompts],
                    value='all',
                    clearable=False
                )
            ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

            html.Div([
                html.Label("Feature:"),
                dcc.Dropdown(
                    id='bias-feature-filter',
                    options=[{'label': 'All', 'value': 'all'}] + [{'label': f, 'value': f} for f in features],
                    value='all',
                    clearable=False
                )
            ], style={'width': '24%', 'display': 'inline-block'}),
        ], style=CARD_STYLE),

        # Main visualization
        html.Div([
            dcc.Graph(id='bias-heatmap', style={'height': '600px'})
        ], style=CARD_STYLE),

        # Summary stats
        html.Div(id='bias-summary-stats', style=CARD_STYLE),

        # Data table
        html.Div([
            html.H4("Filtered Data (Top 100 rows)"),
            html.Button("Download CSV", id="bias-download-btn", n_clicks=0,
                       style={'marginBottom': '10px', 'backgroundColor': COLORS['primary'],
                              'color': 'white', 'border': 'none', 'padding': '10px 20px',
                              'borderRadius': '5px', 'cursor': 'pointer'}),
            dcc.Download(id="bias-download-dataframe-csv"),
            dash_table.DataTable(
                id='bias-data-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'column_id': 'significant', 'filter_query': '{significant} eq True'},
                     'backgroundColor': '#d4edda', 'color': '#155724'}
                ],
                page_size=20
            )
        ], style=CARD_STYLE),
    ])


# ============================================================================
# TAB 2: FEATURE IMPORTANCE
# ============================================================================

def render_importance_tab():
    # Create condition labels
    condition_options = []
    for _, row in importance_df.iterrows():
        label = f"{row['dataset']} × {row['model']} × {row['prompt_style']}"
        value = f"{row['dataset']}_{row['model']}_{row['prompt_style']}"
        condition_options.append({'label': label, 'value': value})

    return html.Div([
        html.H2("Feature Importance Analysis", style={'color': COLORS['primary']}),

        # Condition selector
        html.Div([
            html.Label("Select Condition:"),
            dcc.Dropdown(
                id='importance-condition-selector',
                options=condition_options,
                value=condition_options[0]['value'] if condition_options else None,
                clearable=False
            )
        ], style=CARD_STYLE),

        # Top features bar chart
        html.Div([
            html.H4("Top 15 Features by Importance"),
            dcc.Graph(id='importance-bar-chart', style={'height': '500px'})
        ], style=CARD_STYLE),

        # SHAP values
        html.Div([
            html.H4("SHAP Importance Values"),
            dcc.Graph(id='shap-bar-chart', style={'height': '500px'})
        ], style=CARD_STYLE),

        # Performance metrics
        html.Div(id='importance-metrics', style=CARD_STYLE),
    ])


# ============================================================================
# TAB 3: MODEL COMPARISON
# ============================================================================

def render_model_comparison():
    return html.Div([
        html.H2("Model Comparison", style={'color': COLORS['primary']}),

        # Feature selector
        html.Div([
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id='model-comp-feature',
                options=[{'label': f, 'value': f} for f in features],
                value=features[0] if features else None,
                clearable=False
            )
        ], style=CARD_STYLE),

        # Bias comparison across models
        html.Div([
            html.H4("Bias Comparison Across Models"),
            dcc.Graph(id='model-bias-comparison', style={'height': '500px'})
        ], style=CARD_STYLE),

        # AUROC comparison
        html.Div([
            html.H4("Model Predictive Performance (AUROC)"),
            dcc.Graph(id='model-auroc-comparison', style={'height': '400px'})
        ], style=CARD_STYLE),

        # Model agreement scatter
        html.Div([
            html.H4("Model Agreement on Bias Patterns"),
            dcc.Graph(id='model-agreement-scatter', style={'height': '500px'})
        ], style=CARD_STYLE),
    ])


# ============================================================================
# TAB 4: DATASET ANALYSIS
# ============================================================================

def render_dataset_analysis():
    return html.Div([
        html.H2("Dataset (Platform) Analysis", style={'color': COLORS['primary']}),

        # Feature selector
        html.Div([
            html.Label("Select Feature:"),
            dcc.Dropdown(
                id='dataset-comp-feature',
                options=[{'label': f, 'value': f} for f in features],
                value=features[0] if features else None,
                clearable=False
            )
        ], style=CARD_STYLE),

        # Bias comparison across datasets
        html.Div([
            html.H4("Bias Comparison Across Platforms"),
            dcc.Graph(id='dataset-bias-comparison', style={'height': '500px'})
        ], style=CARD_STYLE),

        # Dataset × Prompt interaction
        html.Div([
            html.H4("Platform × Prompt Interaction Effects"),
            dcc.Graph(id='dataset-prompt-interaction', style={'height': '500px'})
        ], style=CARD_STYLE),
    ])


# ============================================================================
# TAB 5: STATISTICAL TABLES
# ============================================================================

def render_statistical_tables():
    return html.Div([
        html.H2("Statistical Summary Tables", style={'color': COLORS['primary']}),

        # Top biases table
        html.Div([
            html.H4("Top 20 Strongest Biases"),
            html.Button("Download Top Biases CSV", id="top-biases-download-btn", n_clicks=0,
                       style={'marginBottom': '10px', 'backgroundColor': COLORS['success'],
                              'color': 'white', 'border': 'none', 'padding': '10px 20px',
                              'borderRadius': '5px', 'cursor': 'pointer'}),
            dcc.Download(id="top-biases-download"),
            dash_table.DataTable(
                id='top-biases-table',
                columns=[
                    {'name': 'Dataset', 'id': 'dataset'},
                    {'name': 'Model', 'id': 'model'},
                    {'name': 'Prompt', 'id': 'prompt_style'},
                    {'name': 'Feature', 'id': 'feature'},
                    {'name': 'Bias', 'id': 'bias'},
                    {'name': 'P-value', 'id': 'p_value'},
                    {'name': 'Significant', 'id': 'significant'},
                ],
                data=bias_df.nlargest(20, 'bias').to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'column_id': 'significant', 'filter_query': '{significant} eq True'},
                     'backgroundColor': '#d4edda', 'color': '#155724'}
                ],
            )
        ], style=CARD_STYLE),

        # Summary statistics
        html.Div([
            html.H4("Overall Statistics"),
            html.Div(id='overall-stats-table')
        ], style=CARD_STYLE),
    ])


# ============================================================================
# CALLBACKS
# ============================================================================

# Main tab rendering
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-bias':
        return render_bias_explorer()
    elif tab == 'tab-importance':
        return render_importance_tab()
    elif tab == 'tab-models':
        return render_model_comparison()
    elif tab == 'tab-datasets':
        return render_dataset_analysis()
    elif tab == 'tab-tables':
        return render_statistical_tables()
    return html.Div("Tab not found")


# Bias Explorer - Heatmap and table
@app.callback(
    [Output('bias-heatmap', 'figure'),
     Output('bias-summary-stats', 'children'),
     Output('bias-data-table', 'data'),
     Output('bias-data-table', 'columns')],
    [Input('bias-dataset-filter', 'value'),
     Input('bias-model-filter', 'value'),
     Input('bias-prompt-filter', 'value'),
     Input('bias-feature-filter', 'value')]
)
def update_bias_explorer(dataset, model, prompt, feature):
    # Filter data
    filtered_df = bias_df.copy()
    if dataset != 'all':
        filtered_df = filtered_df[filtered_df['dataset'] == dataset]
    if model != 'all':
        filtered_df = filtered_df[filtered_df['model'] == model]
    if prompt != 'all':
        filtered_df = filtered_df[filtered_df['prompt_style'] == prompt]
    if feature != 'all':
        filtered_df = filtered_df[filtered_df['feature'] == feature]

    # Create heatmap
    if feature == 'all':
        # Show heatmap by feature
        pivot_data = filtered_df.groupby(['feature', 'model'])['bias'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='feature', columns='model', values='bias')

        fig = px.imshow(pivot_table,
                       labels=dict(x="Model", y="Feature", color="Bias"),
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       title='Average Bias by Feature and Model')
    else:
        # Show heatmap by condition
        pivot_data = filtered_df.groupby(['dataset', 'model', 'prompt_style'])['bias'].mean().reset_index()
        pivot_data['condition'] = pivot_data['dataset'] + '_' + pivot_data['prompt_style']
        pivot_table = pivot_data.pivot(index='condition', columns='model', values='bias')

        fig = px.imshow(pivot_table,
                       labels=dict(x="Model", y="Condition", color="Bias"),
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       title=f'Bias for {feature}')

    fig.update_layout(height=600, margin=dict(l=150, r=50, t=100, b=50))

    # Summary stats
    n_total = len(filtered_df)
    n_sig = filtered_df['significant'].sum()
    pct_sig = (n_sig / n_total * 100) if n_total > 0 else 0
    mean_bias = filtered_df['bias'].mean()
    std_bias = filtered_df['bias'].std()

    summary = html.Div([
        html.H4("Summary Statistics"),
        html.Div([
            html.Div([
                html.H3(f"{n_total:,}", style={'color': COLORS['primary'], 'margin': '0'}),
                html.P("Total Tests", style={'margin': '0'})
            ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H3(f"{n_sig:,}", style={'color': COLORS['success'], 'margin': '0'}),
                html.P("Significant", style={'margin': '0'})
            ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H3(f"{pct_sig:.1f}%", style={'color': COLORS['warning'], 'margin': '0'}),
                html.P("% Significant", style={'margin': '0'})
            ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H3(f"{mean_bias:.3f}", style={'color': COLORS['danger'], 'margin': '0'}),
                html.P("Mean Bias ± Std", style={'margin': '0'}),
                html.P(f"± {std_bias:.3f}", style={'margin': '0', 'fontSize': '12px'})
            ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ])

    # Table data
    table_data = filtered_df.head(100).to_dict('records')
    table_columns = [{'name': col, 'id': col} for col in ['dataset', 'model', 'prompt_style', 'feature',
                                                           'bias', 'p_value', 'significant']]

    return fig, summary, table_data, table_columns


# Bias Explorer - Download
@app.callback(
    Output("bias-download-dataframe-csv", "data"),
    Input("bias-download-btn", "n_clicks"),
    [State('bias-dataset-filter', 'value'),
     State('bias-model-filter', 'value'),
     State('bias-prompt-filter', 'value'),
     State('bias-feature-filter', 'value')],
    prevent_initial_call=True,
)
def download_bias_data(n_clicks, dataset, model, prompt, feature):
    filtered_df = bias_df.copy()
    if dataset != 'all':
        filtered_df = filtered_df[filtered_df['dataset'] == dataset]
    if model != 'all':
        filtered_df = filtered_df[filtered_df['model'] == model]
    if prompt != 'all':
        filtered_df = filtered_df[filtered_df['prompt_style'] == prompt]
    if feature != 'all':
        filtered_df = filtered_df[filtered_df['feature'] == feature]

    return dcc.send_data_frame(filtered_df.to_csv, "bias_data_filtered.csv", index=False)


# Feature Importance - Bar charts and metrics
@app.callback(
    [Output('importance-bar-chart', 'figure'),
     Output('shap-bar-chart', 'figure'),
     Output('importance-metrics', 'children')],
    Input('importance-condition-selector', 'value')
)
def update_importance_tab(condition_value):
    if not condition_value:
        return {}, {}, html.Div("No condition selected")

    # Parse condition
    parts = condition_value.split('_')
    dataset = parts[0]
    prompt = parts[-1]
    model = '_'.join(parts[1:-1])

    # Get row
    row = importance_df[(importance_df['dataset'] == dataset) &
                        (importance_df['model'] == model) &
                        (importance_df['prompt_style'] == prompt)]

    if row.empty:
        return {}, {}, html.Div("Condition not found")

    row = row.iloc[0]

    # Extract coefficients
    coef_cols = [c for c in importance_df.columns if c.startswith('coef_')]
    coefs = {c.replace('coef_', ''): row[c] for c in coef_cols}
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

    # Coefficient bar chart
    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        y=[f for f, _ in sorted_coefs],
        x=[v for _, v in sorted_coefs],
        orientation='h',
        marker=dict(color=[COLORS['success'] if v > 0 else COLORS['danger'] for _, v in sorted_coefs])
    ))
    fig_coef.update_layout(
        title='Top 15 Features by Logistic Regression Coefficient',
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),
        height=500
    )

    # Extract SHAP values
    shap_cols = [c for c in importance_df.columns if c.startswith('shap_')]
    shaps = {c.replace('shap_', ''): row[c] for c in shap_cols}
    sorted_shaps = sorted(shaps.items(), key=lambda x: x[1], reverse=True)[:15]

    # SHAP bar chart
    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        y=[f for f, _ in sorted_shaps],
        x=[v for _, v in sorted_shaps],
        orientation='h',
        marker=dict(color=COLORS['primary'])
    ))
    fig_shap.update_layout(
        title='Top 15 Features by SHAP Importance',
        xaxis_title='SHAP Value',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),
        height=500
    )

    # Metrics
    metrics = html.Div([
        html.H4("Model Performance"),
        html.Div([
            html.Div([
                html.H3(f"{row['auroc']:.3f}", style={'color': COLORS['primary'], 'margin': '0'}),
                html.P("AUROC", style={'margin': '0'})
            ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H3(f"{row['n_samples']:,}", style={'color': COLORS['secondary'], 'margin': '0'}),
                html.P("Samples", style={'margin': '0'})
            ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H3(f"{row['n_features']}", style={'color': COLORS['success'], 'margin': '0'}),
                html.P("Features", style={'margin': '0'})
            ], style={'width': '33%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ])

    return fig_coef, fig_shap, metrics


# Model Comparison callbacks
@app.callback(
    [Output('model-bias-comparison', 'figure'),
     Output('model-auroc-comparison', 'figure'),
     Output('model-agreement-scatter', 'figure')],
    Input('model-comp-feature', 'value')
)
def update_model_comparison(feature):
    if not feature:
        return {}, {}, {}

    # Filter for selected feature
    feature_df = bias_df[bias_df['feature'] == feature]

    # Bias comparison
    fig_bias = px.box(feature_df, x='model', y='bias', color='dataset',
                     title=f'Bias Distribution for {feature} Across Models',
                     labels={'bias': 'Bias (Effect Size)', 'model': 'Model'})
    fig_bias.update_layout(height=500)

    # AUROC comparison
    fig_auroc = px.box(importance_df, x='model', y='auroc', color='dataset',
                      title='Model Predictive Performance (AUROC)',
                      labels={'auroc': 'AUROC', 'model': 'Model'})
    fig_auroc.add_hline(y=0.7, line_dash="dash", line_color="red",
                       annotation_text="AUROC = 0.7 (Good)")
    fig_auroc.update_layout(height=400)

    # Model agreement - compare bias values between models
    models_list = feature_df['model'].unique()
    if len(models_list) >= 2:
        model1, model2 = models_list[0], models_list[1]

        # Get common conditions
        df1 = feature_df[feature_df['model'] == model1].set_index(['dataset', 'prompt_style'])
        df2 = feature_df[feature_df['model'] == model2].set_index(['dataset', 'prompt_style'])

        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) > 0:
            bias1 = df1.loc[common_idx, 'bias']
            bias2 = df2.loc[common_idx, 'bias']

            fig_agreement = go.Figure()
            fig_agreement.add_trace(go.Scatter(
                x=bias1, y=bias2, mode='markers',
                text=[f"{d}_{p}" for d, p in common_idx],
                marker=dict(size=10, color=COLORS['primary'], opacity=0.6)
            ))
            fig_agreement.add_trace(go.Scatter(
                x=[bias1.min(), bias1.max()],
                y=[bias1.min(), bias1.max()],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Agreement'
            ))
            fig_agreement.update_layout(
                title=f'Model Agreement on {feature} Bias<br>{model1} vs {model2}',
                xaxis_title=f'{model1} Bias',
                yaxis_title=f'{model2} Bias',
                height=500
            )
        else:
            fig_agreement = {}
    else:
        fig_agreement = {}

    return fig_bias, fig_auroc, fig_agreement


# Dataset Analysis callbacks
@app.callback(
    [Output('dataset-bias-comparison', 'figure'),
     Output('dataset-prompt-interaction', 'figure')],
    Input('dataset-comp-feature', 'value')
)
def update_dataset_analysis(feature):
    if not feature:
        return {}, {}

    feature_df = bias_df[bias_df['feature'] == feature]

    # Dataset bias comparison
    fig_bias = px.box(feature_df, x='dataset', y='bias', color='model',
                     title=f'Bias Distribution for {feature} Across Platforms',
                     labels={'bias': 'Bias (Effect Size)', 'dataset': 'Platform'})
    fig_bias.update_layout(height=500)

    # Dataset × Prompt interaction
    interaction_df = feature_df.groupby(['dataset', 'prompt_style'])['bias'].mean().reset_index()
    fig_interaction = px.bar(interaction_df, x='dataset', y='bias', color='prompt_style',
                            barmode='group',
                            title=f'Platform × Prompt Interaction for {feature}',
                            labels={'bias': 'Mean Bias', 'dataset': 'Platform', 'prompt_style': 'Prompt Style'})
    fig_interaction.update_layout(height=500)

    return fig_bias, fig_interaction


# Statistical Tables - Download
@app.callback(
    Output("top-biases-download", "data"),
    Input("top-biases-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_top_biases(n_clicks):
    top_biases = bias_df.nlargest(100, 'bias')
    return dcc.send_data_frame(top_biases.to_csv, "top_biases.csv", index=False)


# Overall statistics
@app.callback(
    Output('overall-stats-table', 'children'),
    Input('main-tabs', 'value')
)
def update_overall_stats(tab):
    if tab != 'tab-tables':
        return html.Div()

    # Calculate statistics
    stats = {
        'Total Tests': len(bias_df),
        'Significant Tests': bias_df['significant'].sum(),
        '% Significant': f"{bias_df['significant'].sum() / len(bias_df) * 100:.1f}%",
        'Mean Bias': f"{bias_df['bias'].mean():.3f}",
        'Std Bias': f"{bias_df['bias'].std():.3f}",
        'Mean AUROC': f"{importance_df['auroc'].mean():.3f}",
        'Conditions > 0.7 AUROC': f"{(importance_df['auroc'] > 0.7).sum()} / {len(importance_df)}"
    }

    table = html.Table([
        html.Tbody([
            html.Tr([html.Td(k, style={'fontWeight': 'bold', 'padding': '10px'}),
                    html.Td(v, style={'padding': '10px'})])
            for k, v in stats.items()
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': f'1px solid {COLORS["secondary"]}'})

    return table


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("DASHBOARD STARTING")
    print("=" * 80)
    print(f"\n✓ Loaded {len(bias_df)} bias metrics")
    print(f"✓ Loaded {len(importance_df)} importance results")
    print(f"\nDashboard ready!")
    print(f"Open your browser to: http://localhost:8050")
    print(f"\nPress Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    app.run_server(debug=True, host='0.0.0.0', port=8050)
