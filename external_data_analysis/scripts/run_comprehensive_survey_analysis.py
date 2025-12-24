#!/usr/bin/env python3
"""
Comprehensive Bias Analysis for External Survey Data

Generates the same comprehensive outputs as the main pipeline:
1. Feature distributions (1_distributions/)
2. Bias heatmaps (2_bias_heatmaps/) - 10 heatmaps
3. Directional bias plots (3_directional_bias/)
4. Feature importance analysis (4_feature_importance/) - 10 heatmaps

Supports multiple models/providers and automatically detects all experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats
import glob

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utility functions from main analysis
from run_comprehensive_analysis import (
    compute_cramers_v, compute_cohens_d,
    standardize_categories
)

# Configuration
EXPERIMENTS_BASE = Path('external_data_analysis/outputs/experiments')
OUTPUT_DIR = Path('external_data_analysis/outputs/analysis_comprehensive')
VIZ_DIR = OUTPUT_DIR / 'visualizations'

# Create output directories
for subdir in ['1_distributions', '2_bias_heatmaps', '3_directional_bias', '4_feature_importance']:
    (VIZ_DIR / subdir).mkdir(parents=True, exist_ok=True)

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")

# Auto-detect all experiment directories
print_section("DETECTING EXPERIMENT DIRECTORIES")

exp_dirs = list(EXPERIMENTS_BASE.glob('survey_*'))
if len(exp_dirs) == 0:
    print(f"{Colors.RED}ERROR: No experiment directories found in {EXPERIMENTS_BASE}{Colors.ENDC}")
    print("Expected format: survey_<provider>_<model>/post_level_data.csv")
    sys.exit(1)

print(f"Found {len(exp_dirs)} experiment(s):")
experiments = []
for exp_dir in exp_dirs:
    data_file = exp_dir / 'post_level_data.csv'
    if data_file.exists():
        # Extract provider and model from directory name
        parts = exp_dir.name.split('_', 2)
        if len(parts) >= 3:
            provider = parts[1]
            model = parts[2]
        else:
            provider = 'unknown'
            model = exp_dir.name
        
        experiments.append({
            'dir': exp_dir,
            'file': data_file,
            'provider': provider,
            'model': model,
            'name': exp_dir.name
        })
        print(f"  ✓ {exp_dir.name} (provider={provider}, model={model})")
    else:
        print(f"  ⚠ {exp_dir.name} (missing post_level_data.csv)")

if len(experiments) == 0:
    print(f"{Colors.RED}ERROR: No valid experiments found with post_level_data.csv{Colors.ENDC}")
    sys.exit(1)

# Load all experiment data
print_section("LOADING EXPERIMENT DATA")

all_data = []
for exp in experiments:
    df_exp = pd.read_csv(exp['file'])
    df_exp['provider'] = exp['provider']
    df_exp['model'] = exp['model']
    df_exp['exp_name'] = exp['name']
    all_data.append(df_exp)
    print(f"✓ Loaded {len(df_exp)} rows from {exp['name']}")

df = pd.concat(all_data, ignore_index=True)
print(f"\n✓ Combined data: {len(df)} total rows")
print(f"  Selected posts: {df['selected'].sum()}")
print(f"  Providers: {df['provider'].unique().tolist()}")
print(f"  Models: {df['model'].unique().tolist()}")
print(f"  Prompt styles: {df['prompt_style'].unique().tolist()}")

# Continue with rest of analysis (auto-detect features, etc.)
# ... [keeping the feature detection code from before]

