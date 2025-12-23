#!/usr/bin/env python3
"""
Step 3: Analyze Bias in Survey Data Recommendations

Uses the same analysis pipeline as the main project.

Usage:
    python 3_analyze_bias.py \
        --experiments outputs/experiments/*/post_level_data.csv \
        --output outputs/analysis/
"""

import sys
from pathlib import Path

# Add parent directory to path to import main analysis functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_comprehensive_analysis import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', required=True, 
                       help='Paths to experiment post_level_data.csv files')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    
    # Setup output directory
    global OUTPUT_DIR, HEATMAP_DIR, PLOT_DIR
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR = OUTPUT_DIR / 'visualizations' / '2_bias_heatmaps'
    PLOT_DIR = OUTPUT_DIR / 'visualizations'
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SURVEY DATA BIAS ANALYSIS")
    print("="*80)
    
    # Note: This assumes experiments are already run and saved
    # The main analysis functions from run_comprehensive_analysis.py
    # will be used to generate all visualizations
    
    print("\nRun the comprehensive analysis on your experiment outputs:")
    print(f"  python run_comprehensive_analysis.py")
    print("\nOr modify this script to point to your specific experiment structure.")

if __name__ == '__main__':
    main()
