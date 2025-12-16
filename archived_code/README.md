# Archived Code

This directory contains Python scripts that were used during development but are no longer actively maintained or have been superseded by newer implementations.

## Superseded Analysis Scripts

- `analyze_experiment_comprehensive.py` - Replaced by `stratified_analysis.py`
- `compare_experiments.py` - Replaced by `meta_analysis.py`
- `comprehensive_bias_analysis.py` - Replaced by `stratified_analysis.py`
- `regression_analysis.py` - Replaced by `stratified_analysis.py`

## One-Time Enhancement Scripts

These scripts were used to add features or fix issues but are no longer needed:

- `enhance_analysis_with_normalization.py` - Added normalization to bias results
- `fix_bias_heatmaps_and_rerun_importance.py` - Fixed heatmap visualizations
- `fix_fairness_deep_dive.py` - Fixed fairness analysis metrics
- `fix_toxicity_extraction.py` - Fixed toxicity feature extraction
- `fix_visualizations_final.py` - Final visualization adjustments
- `create_comparison_visualizations.py` - Created comparison plots
- `generate_enhanced_visualizations.py` - Generated enhanced plots
- `generate_importance_visualizations.py` - Replaced by `generate_all_visualizations.py`
- `fairness_deep_dive.py` - One-time fairness deep dive analysis

## One-Time Feature Extraction Scripts

These scripts were used to extract and join features to the experiment data:

- `extract_additional_features.py` - Extracted additional features
- `extract_author_demographics.py` - Extracted author demographics
- `join_features_to_experiments.py` - Joined features to experiment data
- `join_toxicity_features.py` - Joined toxicity features
- `join_author_demographics.py` - Joined author demographics

## Utility and Test Scripts

- `check_shap_values.py` - Diagnostic utility for SHAP values
- `review_visualizations.py` - One-time visualization review
- `test_gemini.py` - One-time Gemini API test
- `test_claude.py` - One-time Claude API test

## Dashboard/Report Scripts

- `build_dashboard.py` - Dashboard builder (functionality moved elsewhere)
- `generate_report.py` - Report generator (functionality moved elsewhere)
- `generate_interactive_html.py` - Interactive HTML generator

## Note

These files are kept for reference but should not be used in the current workflow. See the main README for the current analysis pipeline.
