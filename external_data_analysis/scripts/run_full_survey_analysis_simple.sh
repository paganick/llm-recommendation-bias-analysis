#!/usr/bin/bash
#
# Simple wrapper to run comprehensive survey data analysis
# Generates all 20 heatmaps matching main pipeline structure
#

echo "========================================================================"
echo "COMPREHENSIVE SURVEY DATA ANALYSIS"
echo "========================================================================"
echo ""
echo "This will generate:"
echo "  • 10 bias heatmaps (disaggregated + aggregated)"
echo "  • 10 feature importance heatmaps (disaggregated + aggregated)"
echo ""

# Run the fixed comprehensive analysis
python external_data_analysis/scripts/run_comprehensive_survey_analysis_fixed.py

echo ""
echo "========================================================================"
echo "✅ DONE!"
echo "========================================================================"
echo ""
echo "Output location: external_data_analysis/outputs/analysis_full/visualizations/"
echo ""
echo "Generated heatmaps:"
echo "  • 2_bias_heatmaps/ - 10 bias effect size heatmaps"
echo "  • 4_feature_importance/ - 10 feature importance heatmaps"
echo ""
