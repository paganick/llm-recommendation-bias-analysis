"""
Dataset loaders for TwitterAAE and other social media datasets.
"""

import pandas as pd
import zipfile
from typing import Optional, List, Dict
from pathlib import Path
import json


class TwitterAAELoader:
    """Loader for TwitterAAE (African American English Twitter Corpus)."""

    def __init__(self, zip_path: str):
        """
        Initialize loader.

        Args:
            zip_path: Path to TwitterAAE-full-v1.zip
        """
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Dataset not found: {zip_path}")

        self.available_versions = {
            'limited': 'twitteraae_limited',
            'limited_aa': 'twitteraae_limited_aa',
            'all': 'twitteraae_all',
            'all_aa': 'twitteraae_all_aa'
        }

    def load(self, version: str = 'limited_aa',
             sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load TwitterAAE dataset.

        Args:
            version: Dataset version (limited, limited_aa, all, all_aa)
            sample_size: Number of tweets to sample (None for all)

        Returns:
            DataFrame with columns:
            - tweet_id: Tweet ID
            - timestamp: Tweet timestamp
            - demo_aa: P(African American)
            - demo_hispanic: P(Hispanic)
            - demo_other: P(Other)
            - demo_white: P(White)
            - [if 'all' version] user_id, lon, lat, blockgroup, text
        """
        if version not in self.available_versions:
            raise ValueError(f"Unknown version: {version}. "
                           f"Available: {list(self.available_versions.keys())}")

        filename = self.available_versions[version]
        file_path = f"TwitterAAE-full-v1/{filename}"

        print(f"Loading TwitterAAE ({version})...")

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(file_path) as f:
                # Determine columns based on version
                if 'all' in version:
                    # Full version with text
                    columns = ['tweet_id', 'timestamp', 'user_id', 'lon', 'lat',
                             'blockgroup', 'text', 'demo_aa', 'demo_hispanic',
                             'demo_other', 'demo_white']
                else:
                    # Limited version (no text, location, etc.)
                    columns = ['tweet_id', 'timestamp', 'demo_aa', 'demo_hispanic',
                             'demo_other', 'demo_white']

                # Read tab-delimited file
                df = pd.read_csv(f, sep='\t', names=columns,
                               parse_dates=['timestamp'], quoting=3)

        print(f"Loaded {len(df):,} tweets")

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size:,} tweets")

        return df

    def get_dataset_info(self) -> Dict:
        """Get information about available dataset versions."""
        return {
            'path': str(self.zip_path),
            'versions': {
                'limited': 'Tweet IDs, timestamps, demographic inferences (59.2M tweets)',
                'limited_aa': 'Same as limited, filtered to high AA probability (1.1M tweets)',
                'all': 'Includes full tweet text, user IDs, locations (59.2M tweets)',
                'all_aa': 'Same as all, filtered to high AA probability (1.1M tweets)'
            }
        }


class DADITLoader:
    """Loader for DADIT dataset (mental health/user classification)."""

    def __init__(self, zip_path: str):
        """
        Initialize loader.

        Args:
            zip_path: Path to DADIT_data_for_publishing.zip
        """
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Dataset not found: {zip_path}")

    def load(self, version: str = 'test_anon',
             sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load DADIT dataset.

        Args:
            version: Dataset version (test_anon, train_anon, test, train, etc.)
            sample_size: Number of rows to sample (None for all)

        Returns:
            DataFrame with dataset contents
        """
        file_path = f"g100_work/IscrC_mental/data/user_classification/data_for_publishing/{version}.parquet"

        print(f"Loading DADIT ({version})...")

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(file_path) as f:
                df = pd.read_parquet(f)

        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size:,} rows")

        return df


def load_dataset(dataset_name: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to load datasets.

    Args:
        dataset_name: 'twitteraae' or 'dadit'
        **kwargs: Arguments passed to specific loader

    Returns:
        Loaded DataFrame
    """
    if dataset_name.lower() == 'twitteraae':
        loader = TwitterAAELoader(kwargs.get('zip_path',
                                            '/data/nicpag/AI_recsys_project/TwitterAAE-full-v1.zip'))
        return loader.load(version=kwargs.get('version', 'limited_aa'),
                          sample_size=kwargs.get('sample_size'))

    elif dataset_name.lower() == 'dadit':
        loader = DADITLoader(kwargs.get('zip_path',
                                       '/data/nicpag/AI_recsys_project/DADIT_data_for_publishing.zip'))
        return loader.load(version=kwargs.get('version', 'test_anon'),
                          sample_size=kwargs.get('sample_size'))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
