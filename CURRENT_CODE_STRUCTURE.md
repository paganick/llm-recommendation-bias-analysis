# Current Code Structure

This document outlines the active codebase after cleanup (December 2025).

## Core Experiment Scripts

### Running Experiments
- **run_experiment.py** - Main experiment runner for a single dataset/model/prompt combination
- **run_all_experiments.py** - Batch runner for multiple experiments in parallel
- **run_experiment_with_tracking.py** - Experiment runner with progress tracking

## Analysis Pipeline

### Statistical Analysis
- **stratified_analysis.py** - Stratified logistic regression analysis per prompt style
  - Computes bias metrics, regression coefficients, statistical tests
  - Generates per-style analysis results

- **meta_analysis.py** - Cross-experiment meta-analysis
  - Aggregates results across datasets/models
  - Performs ANOVA for dataset and model comparisons
  - Computes pooled effect sizes with heterogeneity testing

- **run_feature_importance.py** - Feature importance analysis
  - Trains logistic regression models
  - Computes SHAP values
  - Generates feature importance rankings

### Visualization
- **generate_all_visualizations.py** - Master visualization pipeline
  - Generates bias heatmaps
  - Creates cross-cutting analyses
  - Produces all static plots

- **visualization_functions.py** - Plotting utility functions
  - Heatmap functions
  - Comparison plots
  - Effect size visualizations

- **analyze_features.py** - Feature documentation and distribution analysis
  - Documents all features with ranges, types, computation methods
  - Generates distribution plots for all features across datasets
  - Creates comprehensive feature summary

### Additional Utilities
- **create_dashboards.py** - Dashboard generation (if needed)
- **run_full_analysis_pipeline.py** - Complete analysis pipeline orchestration

## Infrastructure Modules

### Data Loading
- **data/loaders.py** - Dataset loading and preprocessing
- **data/__init__.py** - Package initialization

### LLM Clients
- **utils/llm_client.py** - Unified LLM API client
  - Supports OpenAI, Anthropic, Google Gemini
  - Handles rate limiting and errors
- **utils/__init__.py** - Package initialization

### Inference
- **inference/metadata_inference.py** - Extract metadata from posts
  - Sentiment analysis
  - Topic classification
  - Style features
- **inference/persona_extraction.py** - Process persona information
- **inference/__init__.py** - Package initialization

### Analysis Modules
- **analysis/bias_analysis.py** - Core bias computation functions
- **analysis/visualization.py** - Visualization utilities
- **analysis/__init__.py** - Package initialization

### Recommender
- **recommender/llm_recommender.py** - LLM-based recommender implementation
- **recommender/__init__.py** - Package initialization

### Experiments
- **experiments/experiment_runner.py** - Experiment orchestration
- **experiments/__init__.py** - Package initialization

## Testing
- **test_pipeline.py** - Integration tests for the full pipeline

## Documentation Files (Active)
- **README.md** - Main project documentation
- **CLEANUP_PLAN.txt** - Code cleanup plan
- **CURRENT_CODE_STRUCTURE.md** - This file
- **outputs/FEATURE_DEFINITIONS.md** - Feature definitions
- **analysis_outputs/feature_analysis/FEATURE_ANALYSIS_REPORT.txt** - Comprehensive feature analysis

## Archived Directories
- **old_scripts/** - Deprecated scripts from early development
- **archived_code/** - Recently archived one-off and superseded scripts

## Data Directories
- **datasets/** - Original dataset files (not committed)
- **outputs/experiments/** - Experiment results
- **outputs/metadata_cache/** - Cached metadata
- **analysis_outputs/** - Analysis results and visualizations

## Recommended Workflow

### 1. Run New Experiments
```bash
python run_experiment_with_tracking.py --dataset twitter --provider openai --model gpt-4o-mini
```

### 2. Analyze Experiments
```bash
# Stratified analysis
python stratified_analysis.py --experiment-dir outputs/experiments/twitter_openai_gpt-4o-mini

# Meta-analysis across experiments
python meta_analysis.py --experiments-dir outputs/experiments --output-dir outputs/meta_analysis

# Feature importance
python run_feature_importance.py
```

### 3. Generate Visualizations
```bash
python generate_all_visualizations.py
```

### 4. Analyze Features
```bash
python analyze_features.py
```

### 5. Complete Pipeline
```bash
python run_full_analysis_pipeline.py --analyze-all
```

## File Count Summary
- Active Python scripts: ~15
- Archived Python scripts: 25
- Old scripts: ~15
- Infrastructure modules: ~12
- **Total cleaned**: 40 files archived

