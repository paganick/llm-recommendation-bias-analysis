#!/usr/bin/env python3
"""
Adapt survey experiment data for main analysis pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load experiment data
input_file = 'external_data_analysis/outputs/experiments/survey_gemini_gemini-2.0-flash/post_level_data.csv'
df = pd.read_csv(input_file)

print(f"Loaded {len(df)} rows")
print(f"Current columns: {len(df.columns)}")

# Map author_ideology to author_political_leaning
if 'author_ideology' in df.columns:
    df['author_political_leaning'] = df['author_ideology']
    print("✓ Mapped author_ideology → author_political_leaning")

# Create author_is_minority from author_race
if 'author_race' in df.columns:
    # Consider 'white' as majority, all others as minority
    df['author_is_minority'] = df['author_race'].apply(
        lambda x: 'no' if pd.isna(x) or str(x).lower() == 'white' else 'yes'
    )
    print("✓ Created author_is_minority from author_race")

# Add missing features with default values (if not present)
default_features = {
    'polarization_score': 0.5,  # Neutral default
    'controversy_level': 'low',
    'primary_topic': 'general',
    'toxicity': 0.0,
    'severe_toxicity': 0.0
}

for feature, default_value in default_features.items():
    if feature not in df.columns:
        df[feature] = default_value
        print(f"✓ Added {feature} (default: {default_value})")

# Ensure required columns exist
required_cols = [
    'author_gender', 'author_political_leaning', 'author_is_minority',
    'text_length', 'avg_word_length',
    'sentiment_polarity', 'sentiment_subjectivity',
    'has_emoji', 'has_hashtag', 'has_mention', 'has_url',
    'polarization_score', 'controversy_level', 'primary_topic',
    'toxicity', 'severe_toxicity'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"\n⚠ Warning: Still missing columns: {missing}")
else:
    print(f"\n✓ All required features present!")

# Save adapted data
df.to_csv(input_file, index=False)
print(f"\n✓ Saved adapted data to {input_file}")
print(f"  Total columns: {len(df.columns)}")
