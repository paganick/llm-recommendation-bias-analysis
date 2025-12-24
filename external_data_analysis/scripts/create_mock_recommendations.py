#!/usr/bin/env python3
"""
Create mock recommendation data for testing the analysis pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load prepared data
df = pd.read_csv('external_data_analysis/data/examples/prepared_example.csv')

print(f"Loaded {len(df)} posts")

# Simulate 6 prompt styles × 10 trials = 60 trials
# Pool size = 10
prompt_styles = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']
n_trials = 10
pool_size = 10

all_results = []

np.random.seed(42)  # Reproducible results

for prompt_style in prompt_styles:
    print(f"Simulating prompt style: {prompt_style}")
    
    for trial_id in range(n_trials):
        # Sample pool (with replacement since we only have 20 posts)
        pool = df.sample(n=pool_size, replace=True).reset_index(drop=True)
        
        # Randomly select one post (simulating LLM choice)
        selected_idx = np.random.randint(0, pool_size)
        
        # Mark selected
        pool['selected'] = 0
        pool.loc[selected_idx, 'selected'] = 1
        pool['prompt_style'] = prompt_style
        pool['trial_id'] = trial_id
        pool['pool_position'] = range(pool_size)
        
        all_results.append(pool)

# Combine all results
results_df = pd.concat(all_results, ignore_index=True)

# Create output directory
output_dir = Path('external_data_analysis/outputs/experiments/survey_gemini_gemini-2.0-flash')
output_dir.mkdir(parents=True, exist_ok=True)

# Save results
output_file = output_dir / 'post_level_data.csv'
results_df.to_csv(output_file, index=False)

print(f"\n✓ Created mock recommendations: {output_file}")
print(f"  Total rows: {len(results_df)}")
print(f"  Selected posts: {results_df['selected'].sum()}")
print(f"  Prompt styles: {prompt_styles}")
print(f"  Trials per style: {n_trials}")
