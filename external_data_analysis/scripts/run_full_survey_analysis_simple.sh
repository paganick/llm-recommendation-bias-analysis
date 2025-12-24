#!/usr/bin/bash
#
# Run full comprehensive analysis on survey data using main pipeline
#

set -e

echo "=============================================================================="
echo "SURVEY DATA COMPREHENSIVE ANALYSIS"
echo "=============================================================================="
echo ""

# Check for survey experiments
if [ ! -d "external_data_analysis/outputs/experiments" ]; then
    echo "ERROR: No experiments directory found!"
    exit 1
fi

# Count experiments
n_exp=$(ls -d external_data_analysis/outputs/experiments/survey_* 2>/dev/null | wc -l)
if [ "$n_exp" -eq 0 ]; then
    echo "ERROR: No survey experiments found!"
    echo "Expected: external_data_analysis/outputs/experiments/survey_*/"
    exit 1
fi

echo "Found $n_exp survey experiment(s)"
echo ""

# Copy survey experiments to main experiments dir temporarily
echo "Preparing data..."
mkdir -p outputs/experiments

for exp_dir in external_data_analysis/outputs/experiments/survey_*; do
    exp_name=$(basename "$exp_dir")
    echo "  - Copying $exp_name"
    rm -rf "outputs/experiments/$exp_name"
    cp -r "$exp_dir" "outputs/experiments/$exp_name"
done

echo ""
echo "=============================================================================="
echo "RUNNING MAIN ANALYSIS PIPELINE"
echo "=============================================================================="
echo ""

# Run main comprehensive analysis
python3 run_comprehensive_analysis.py

echo ""
echo "=============================================================================="
echo "MOVING OUTPUTS TO SURVEY DIRECTORY"
echo "=============================================================================="
echo ""

# Move outputs to survey-specific location
output_dir="external_data_analysis/outputs/analysis_full"
rm -rf "$output_dir"
mkdir -p "$output_dir"

echo "Copying analysis outputs..."
cp -r analysis_outputs/* "$output_dir/"

echo ""
echo "Cleaning up temporary files..."
for exp_dir in external_data_analysis/outputs/experiments/survey_*; do
    exp_name=$(basename "$exp_dir")
    rm -rf "outputs/experiments/$exp_name"
    echo "  - Removed outputs/experiments/$exp_name"
done

echo ""
echo "=============================================================================="
echo "ANALYSIS COMPLETE!"
echo "=============================================================================="
echo ""
echo "All outputs saved to: $output_dir"
echo ""
echo "Generated:"
echo "  - 1_distributions/ (one plot per feature)"
echo "  - 2_bias_heatmaps/ (10 heatmaps)"
echo "  - 3_directional_bias/ (per-feature plots)"  
echo "  - 4_feature_importance/ (10 heatmaps)"
echo ""

