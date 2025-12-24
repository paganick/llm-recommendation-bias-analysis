#!/usr/bin/bash
#
# Wrapper to run comprehensive survey data analysis
# Generates all visualizations matching main pipeline structure
#

echo "========================================================================"
echo "COMPREHENSIVE SURVEY DATA ANALYSIS"
echo "========================================================================"
echo ""
echo "This will generate:"
echo "  • Distribution plots (pool vs selected)"
echo "  • 10 bias heatmaps (disaggregated + aggregated)"
echo "  • Directional bias heatmaps (over/under-representation)"
echo "  • 10 feature importance heatmaps (disaggregated + aggregated)"
echo ""

# Run the comprehensive analysis
python external_data_analysis/scripts/run_comprehensive_survey_analysis.py

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
