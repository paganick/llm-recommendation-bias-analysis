"""
Fix Toxicity Extraction for Bluesky and Reddit

The issue: The experiments use original_index 0-4998 which are sequential indices
after sampling and resetting. But the toxicity extraction used the original
pandas indices from the full datasets, causing a mismatch.

Solution: Extract toxicity directly from the post messages in the experiment files,
then map back using original_index.

Usage:
    python fix_toxicity_extraction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    print("ERROR: detoxify not installed. Install with: pip install detoxify")
    DETOXIFY_AVAILABLE = False


def extract_toxicity_from_experiments(dataset_name: str, output_path: Path):
    """
    Extract toxicity by loading messages from experiment files.

    Args:
        dataset_name: 'twitter', 'bluesky', or 'reddit'
        output_path: Where to save the toxicity parquet file
    """
    print(f"\n{'=' * 70}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'=' * 70}")

    # Find experiment files for this dataset
    import glob
    exp_dirs = glob.glob(f'outputs/experiments/{dataset_name}_*/post_level_data.csv')
    if not exp_dirs:
        print(f"ERROR: No experiment files found for {dataset_name}")
        return False

    print(f"Found {len(exp_dirs)} experiment files for {dataset_name}")
    print(f"Using: {exp_dirs[0]}")

    # Load one experiment file to get the pool mapping
    df_exp = pd.read_csv(exp_dirs[0])
    print(f"Loaded {len(df_exp):,} rows from experiment")

    # Get unique original_index and message pairs
    # Use the first occurrence of each original_index (they all have same message)
    pool_mapping = df_exp.drop_duplicates('original_index')[['original_index', 'message']].copy()
    pool_mapping = pool_mapping.sort_values('original_index').reset_index(drop=True)

    print(f"Found {len(pool_mapping)} unique messages with original_index 0-{pool_mapping['original_index'].max()}")

    # Check if messages are valid
    pool_mapping['message'] = pool_mapping['message'].fillna("")
    empty_messages = (pool_mapping['message'] == "").sum()
    if empty_messages > 0:
        print(f"WARNING: {empty_messages} empty messages found")

    # Extract toxicity using Detoxify
    if not DETOXIFY_AVAILABLE:
        print("ERROR: Detoxify not available")
        return False

    print("\nInitializing Detoxify model...")
    toxicity_model = Detoxify('original', device='cpu')

    print(f"\nExtracting toxicity for {len(pool_mapping)} messages...")
    results = []
    batch_size = 32

    for i in tqdm(range(0, len(pool_mapping), batch_size), desc="Processing batches"):
        batch = pool_mapping.iloc[i:i+batch_size]
        texts = batch['message'].tolist()

        try:
            # Get predictions
            predictions = toxicity_model.predict(texts)

            # Store results
            for j, idx in enumerate(batch.index):
                result = {
                    'original_index': int(pool_mapping.loc[idx, 'original_index']),
                    'toxicity': predictions['toxicity'][j],
                    'severe_toxicity': predictions['severe_toxicity'][j],
                    'obscene': predictions['obscene'][j],
                    'threat': predictions['threat'][j],
                    'insult': predictions['insult'][j],
                    'identity_attack': predictions['identity_attack'][j]
                }
                results.append(result)
        except Exception as e:
            print(f"\nERROR in batch {i}-{i+batch_size}: {e}")
            # Add NaN results for failed batch
            for j, idx in enumerate(batch.index):
                result = {
                    'original_index': int(pool_mapping.loc[idx, 'original_index']),
                    'toxicity': np.nan,
                    'severe_toxicity': np.nan,
                    'obscene': np.nan,
                    'threat': np.nan,
                    'insult': np.nan,
                    'identity_attack': np.nan
                }
                results.append(result)

    # Create DataFrame
    toxicity_df = pd.DataFrame(results)

    print(f"\n✅ Extracted toxicity for {len(toxicity_df)} messages")
    print(f"   Original_index range: {toxicity_df['original_index'].min()} - {toxicity_df['original_index'].max()}")
    print(f"   Non-null toxicity: {toxicity_df['toxicity'].notna().sum()}")

    # Save
    toxicity_df.to_parquet(output_path, index=False)
    print(f"   Saved to: {output_path}")

    # Show sample statistics
    if toxicity_df['toxicity'].notna().any():
        print(f"\n   Toxicity statistics:")
        print(f"     Mean: {toxicity_df['toxicity'].mean():.4f}")
        print(f"     Median: {toxicity_df['toxicity'].median():.4f}")
        print(f"     Max: {toxicity_df['toxicity'].max():.4f}")
        print(f"     > 0.5 (toxic): {(toxicity_df['toxicity'] > 0.5).sum()} ({100*(toxicity_df['toxicity'] > 0.5).sum()/len(toxicity_df):.1f}%)")

    return True


def main():
    """Fix toxicity extraction for Bluesky and Reddit."""

    if not DETOXIFY_AVAILABLE:
        print("\nERROR: Detoxify library is required")
        print("Install with: pip install detoxify")
        return

    output_dir = Path("outputs/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    datasets_to_fix = ['bluesky', 'reddit']

    print("=" * 70)
    print("FIX TOXICITY EXTRACTION")
    print("=" * 70)
    print("\nThis will re-extract toxicity features using the exact messages")
    print("from the experiment files, ensuring proper index alignment.")
    print(f"\nDatasets to fix: {datasets_to_fix}")
    print(f"Output directory: {output_dir}")

    for dataset_name in datasets_to_fix:
        output_path = output_dir / f"{dataset_name}_toxicity.parquet"

        # Backup existing file if it exists
        if output_path.exists():
            backup_path = output_dir / f"{dataset_name}_toxicity_backup.parquet"
            print(f"\nBacking up existing file to {backup_path.name}")
            import shutil
            shutil.copy(output_path, backup_path)

        # Extract
        success = extract_toxicity_from_experiments(dataset_name, output_path)

        if not success:
            print(f"\n❌ Failed to process {dataset_name}")
        else:
            print(f"\n✅ Successfully processed {dataset_name}")

    print("\n" + "=" * 70)
    print("DONE! Now run join_toxicity_features.py to update experiment files")
    print("=" * 70)


if __name__ == "__main__":
    main()
