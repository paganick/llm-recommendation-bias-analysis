"""
Join Toxicity Features to Experiment Data

This script joins the pre-extracted toxicity features from outputs/features/*.parquet
to the experiment post_level_data files. This is much faster than re-extracting toxicity.

Usage:
    python join_toxicity_features.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

def join_toxicity_to_experiment(experiment_dir: Path, dataset_name: str):
    """
    Join toxicity features to a single experiment's data.

    Args:
        experiment_dir: Path to experiment directory
        dataset_name: Name of dataset (twitter, bluesky, reddit)
    """
    print(f"\nProcessing: {experiment_dir.name}")

    # Load experiment data
    csv_path = experiment_dir / "post_level_data.csv"
    parquet_path = experiment_dir / "post_level_data.parquet"

    print(f"  Loading experiment data...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows")

    # Check if toxicity is already populated
    if df['toxicity'].notna().any():
        print(f"  ⚠️  Toxicity already populated, skipping...")
        return

    # Load toxicity features
    toxicity_path = Path("outputs/features") / f"{dataset_name}_toxicity.parquet"
    if not toxicity_path.exists():
        print(f"  ❌ ERROR: Toxicity file not found: {toxicity_path}")
        return

    print(f"  Loading toxicity features from {toxicity_path.name}...")
    toxicity_df = pd.read_parquet(toxicity_path)
    print(f"  Loaded toxicity for {len(toxicity_df):,} posts")

    # Verify original_index exists and is valid
    if 'original_index' not in df.columns:
        print(f"  ❌ ERROR: 'original_index' column not found in experiment data")
        return

    # Check for any missing indices
    missing_indices = df['original_index'].unique()
    missing_indices = [idx for idx in missing_indices if idx not in toxicity_df['original_index'].values]
    if missing_indices:
        print(f"  ⚠️  WARNING: {len(missing_indices)} indices in experiment not found in toxicity data")

    # Drop existing toxicity columns (they're all NaN anyway)
    toxicity_cols = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    existing_tox_cols = [col for col in toxicity_cols if col in df.columns]
    if existing_tox_cols:
        print(f"  Dropping existing NaN toxicity columns: {existing_tox_cols}")
        df = df.drop(columns=existing_tox_cols)

    # Join toxicity features
    print(f"  Joining toxicity features on 'original_index'...")
    df = df.merge(toxicity_df, on='original_index', how='left')

    # Verify join worked
    toxicity_populated = df['toxicity'].notna().sum()
    toxicity_missing = df['toxicity'].isna().sum()
    print(f"  ✅ Joined successfully:")
    print(f"     - {toxicity_populated:,} rows with toxicity ({100*toxicity_populated/len(df):.1f}%)")
    print(f"     - {toxicity_missing:,} rows missing toxicity ({100*toxicity_missing/len(df):.1f}%)")

    if toxicity_missing > 0:
        print(f"  ⚠️  Some rows missing toxicity - this might indicate index misalignment")

    # Save updated files
    print(f"  Saving updated data...")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"  ✅ Saved: {csv_path.name} and {parquet_path.name}")


def main():
    """Join toxicity features to all experiments."""
    experiments_dir = Path("outputs/experiments")

    if not experiments_dir.exists():
        print(f"ERROR: Experiments directory not found: {experiments_dir}")
        return

    # Map experiment directories to dataset names
    experiments = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Extract dataset name from directory name
        # Format: {dataset}_{provider}_{model}
        parts = exp_dir.name.split('_')
        if len(parts) >= 2:
            dataset_name = parts[0]  # twitter, bluesky, or reddit
            experiments.append((exp_dir, dataset_name))

    print(f"Found {len(experiments)} experiments to process")
    print("=" * 80)

    # Process each experiment
    for exp_dir, dataset_name in experiments:
        try:
            join_toxicity_to_experiment(exp_dir, dataset_name)
        except Exception as e:
            print(f"  ❌ ERROR processing {exp_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ Processing complete!")
    print("\nVerification:")
    print("  You can verify the join by checking a few experiments:")
    print("  python -c \"import pandas as pd; df = pd.read_csv('outputs/experiments/twitter_openai_gpt-4o-mini/post_level_data.csv'); print(f'Toxicity populated: {df.toxicity.notna().sum()} / {len(df)} rows')\"")


if __name__ == "__main__":
    main()
