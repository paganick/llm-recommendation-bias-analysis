"""
Join Additional Features to Experiment Data

This script takes the extracted features (toxicity, author demographics) and joins them
to all experiment post-level data using the 'original_index' key.

Usage:
    python join_features_to_experiments.py --all
    python join_features_to_experiments.py --experiments twitter_anthropic_claude-sonnet-4-5-20250929
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_feature_files(dataset_name: str, features_dir: Path) -> dict:
    """
    Load all available feature files for a dataset.

    Args:
        dataset_name: Name of dataset (twitter, bluesky, reddit)
        features_dir: Directory containing feature files

    Returns:
        Dictionary of feature DataFrames indexed by original_index
    """
    features = {}

    # Toxicity features (try parquet, then CSV)
    toxicity_file = features_dir / f"{dataset_name}_toxicity.parquet"
    toxicity_csv = features_dir / f"{dataset_name}_toxicity.csv"
    if toxicity_file.exists():
        print(f"  Loading toxicity features from {toxicity_file.name}")
        features['toxicity'] = pd.read_parquet(toxicity_file)
    elif toxicity_csv.exists():
        print(f"  Loading toxicity features from {toxicity_csv.name}")
        features['toxicity'] = pd.read_csv(toxicity_csv)
    else:
        print(f"  ⚠ Toxicity features not found")

    # Author gender (try parquet, then CSV)
    gender_file = features_dir / f"{dataset_name}_author_gender.parquet"
    gender_csv = features_dir / f"{dataset_name}_author_gender.csv"
    if gender_file.exists():
        print(f"  Loading gender features from {gender_file.name}")
        features['gender'] = pd.read_parquet(gender_file)
    elif gender_csv.exists():
        print(f"  Loading gender features from {gender_csv.name}")
        features['gender'] = pd.read_csv(gender_csv)
    else:
        print(f"  ⚠ Gender features not found")

    # Author politics (try parquet, then CSV)
    politics_file = features_dir / f"{dataset_name}_author_politics.parquet"
    politics_csv = features_dir / f"{dataset_name}_author_politics.csv"
    if politics_file.exists():
        print(f"  Loading politics features from {politics_file.name}")
        features['politics'] = pd.read_parquet(politics_file)
    elif politics_csv.exists():
        print(f"  Loading politics features from {politics_csv.name}")
        features['politics'] = pd.read_csv(politics_csv)
    else:
        print(f"  ⚠ Politics features not found")

    return features


def join_features_to_experiment(experiment_dir: Path, features_dir: Path) -> bool:
    """
    Join features to a single experiment's data.

    Args:
        experiment_dir: Path to experiment directory
        features_dir: Path to features directory

    Returns:
        True if successful, False otherwise
    """
    # Parse experiment metadata
    exp_name = experiment_dir.name
    parts = exp_name.split('_')
    if len(parts) < 3:
        print(f"  ⚠ Cannot parse experiment name: {exp_name}")
        return False

    dataset = parts[0]
    print(f"\n  Dataset: {dataset}")

    # Load post-level data
    data_file = experiment_dir / "post_level_data.csv"
    if not data_file.exists():
        print(f"  ⚠ Data file not found: {data_file}")
        return False

    print(f"  Loading experiment data...")
    df = pd.read_csv(data_file)
    original_shape = df.shape
    print(f"    Original shape: {original_shape}")

    # Load features for this dataset
    print(f"  Loading features for {dataset}...")
    features = load_feature_files(dataset, features_dir)

    if len(features) == 0:
        print(f"  ⚠ No features found for {dataset}")
        return False

    # Join each feature set
    for feature_name, feature_df in features.items():
        print(f"  Joining {feature_name} features...")

        # Get columns to add (exclude original_index and author which are keys)
        cols_to_add = [col for col in feature_df.columns
                      if col not in ['original_index', 'author'] and col not in df.columns]

        if len(cols_to_add) == 0:
            print(f"    ⚠ No new columns to add from {feature_name}")
            continue

        print(f"    Adding columns: {', '.join(cols_to_add)}")

        # Join on original_index
        df = df.merge(
            feature_df[['original_index'] + cols_to_add],
            on='original_index',
            how='left'
        )

        # Check for missing values
        for col in cols_to_add:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                print(f"    ⚠ {col}: {n_missing} missing values ({n_missing/len(df)*100:.1f}%)")

    print(f"  Final shape: {df.shape} (added {df.shape[1] - original_shape[1]} columns)")

    # Save updated data
    print(f"  Saving updated data...")
    df.to_csv(data_file, index=False)

    # Also save as parquet for faster loading
    parquet_file = experiment_dir / "post_level_data.parquet"
    df.to_parquet(parquet_file, index=False)

    print(f"  ✓ Saved to {data_file}")
    print(f"  ✓ Saved to {parquet_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Join extracted features to experiment data'
    )

    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='outputs/experiments',
        help='Directory containing experiments'
    )

    parser.add_argument(
        '--features-dir',
        type=str,
        default='outputs/features',
        help='Directory containing feature files'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all experiments'
    )

    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        help='Specific experiments to process'
    )

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    features_dir = Path(args.features_dir)

    if not experiments_dir.exists():
        print(f"ERROR: Experiments directory not found: {experiments_dir}")
        return

    if not features_dir.exists():
        print(f"ERROR: Features directory not found: {features_dir}")
        print(f"Run extract_additional_features.py first!")
        return

    print("="*80)
    print("JOIN FEATURES TO EXPERIMENTS")
    print("="*80)
    print(f"\nExperiments directory: {experiments_dir}")
    print(f"Features directory: {features_dir}")
    print()

    # Determine which experiments to process
    if args.all:
        experiments = [d for d in experiments_dir.iterdir()
                      if d.is_dir() and (d / "post_level_data.csv").exists()]
        print(f"Found {len(experiments)} experiments")
    elif args.experiments:
        experiments = []
        for exp_name in args.experiments:
            exp_path = experiments_dir / exp_name
            if exp_path.exists() and (exp_path / "post_level_data.csv").exists():
                experiments.append(exp_path)
            else:
                print(f"⚠ Experiment not found or incomplete: {exp_name}")
    else:
        print("ERROR: Must specify --all or --experiments")
        return

    if len(experiments) == 0:
        print("No experiments to process!")
        return

    print(f"\nProcessing {len(experiments)} experiments:\n")

    # Process each experiment
    success_count = 0
    fail_count = 0

    for i, exp_dir in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_dir.name}")
        print("-" * 80)

        try:
            if join_features_to_experiment(exp_dir, features_dir):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("FEATURE JOINING COMPLETE")
    print("="*80)
    print(f"\nSuccessful: {success_count}/{len(experiments)}")
    print(f"Failed: {fail_count}/{len(experiments)}")
    print("\nNext steps:")
    print("  1. Run comprehensive bias analysis with new features")
    print("  2. Run feature importance analysis (including new features)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
