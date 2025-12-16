"""
Join Author Demographics to Experiment Files

Joins gender, political_leaning, is_minority, minority_groups from
outputs/features/*_author_demographics.json to experiment CSV files.
"""

import pandas as pd
import json
from pathlib import Path
import glob

def join_demographics_to_experiment(exp_dir: Path, dataset_name: str):
    """Join demographics to a single experiment."""

    print(f"\n{exp_dir.name}")

    # Load demographics
    demo_path = Path("outputs/features") / f"{dataset_name}_author_demographics.json"
    if not demo_path.exists():
        print(f"  ❌ Demographics file not found: {demo_path}")
        return False

    with open(demo_path) as f:
        demographics = json.load(f)
    print(f"  Loaded {len(demographics)} author demographics")

    # Load experiment data
    csv_path = exp_dir / "post_level_data.csv"
    parquet_path = exp_dir / "post_level_data.parquet"

    df = pd.read_csv(csv_path)
    print(f"  Experiment: {len(df):,} rows")

    # Check if already joined
    if 'author_gender' in df.columns:
        print(f"  ⚠️  Demographics already joined, skipping")
        return True

    # Create demographics DataFrame
    demo_df = pd.DataFrame([
        {
            'username': username,
            'author_gender': data['gender'],
            'author_political_leaning': data['political_leaning'],
            'author_is_minority': data['is_minority'],
            'author_minority_groups': data['minority_groups']
        }
        for username, data in demographics.items()
    ])

    # Join
    df = df.merge(demo_df, on='username', how='left')

    # Check coverage
    populated = df['author_gender'].notna().sum()
    pct = 100 * populated / len(df)
    print(f"  Demographics: {populated:,} / {len(df):,} ({pct:.1f}%)")

    # Save
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"  ✅ Saved")

    return True


def main():
    """Join demographics to all experiments."""

    print("="*70)
    print("JOIN AUTHOR DEMOGRAPHICS TO EXPERIMENTS")
    print("="*70)

    experiments_dir = Path("outputs/experiments")

    # Process each dataset
    for dataset_name in ['twitter', 'bluesky', 'reddit']:
        print(f"\n{'='*70}")
        print(f"{dataset_name.upper()}")
        print(f"{'='*70}")

        # Find all experiments for this dataset
        exp_dirs = sorted(experiments_dir.glob(f"{dataset_name}_*"))

        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            try:
                join_demographics_to_experiment(exp_dir, dataset_name)
            except Exception as e:
                print(f"  ❌ ERROR: {e}")

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
