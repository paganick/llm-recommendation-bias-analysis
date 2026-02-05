"""
Dataset loaders for social media datasets (Twitter, Reddit, Bluesky, TwitterAAE).
"""

import pandas as pd
import zipfile
from typing import Optional, List, Dict
from pathlib import Path
import json


class PersonaDatasetLoader:
    """Loader for persona-based social media datasets (Twitter, Reddit, Bluesky)."""

    def __init__(self, dataset_path: str):
        """
        Initialize loader.

        Args:
            dataset_path: Path to personas.pkl file
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def load(self, sample_size: Optional[int] = None,
             training_only: bool = False) -> pd.DataFrame:
        """
        Load persona dataset.

        Args:
            sample_size: Number of posts to sample (None for all)
            training_only: If True, only load training data

        Returns:
            DataFrame with columns:
            - username (or user_id): User identifier
            - persona: User demographic/persona information
            - message: Post text
            - reply_to: Reply information (if applicable)
            - training: Train/test split indicator
        """
        print(f"Loading dataset from {self.dataset_path}...")

        df = pd.read_pickle(self.dataset_path)

        print(f"Loaded {len(df):,} posts")

        # Filter training data if requested
        if training_only and 'training' in df.columns:
            df = df[df['training'] == 1]
            print(f"Filtered to {len(df):,} training posts")

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size:,} posts")

        return df

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        df = pd.read_pickle(self.dataset_path)
        return {
            'path': str(self.dataset_path),
            'total_posts': len(df),
            'columns': list(df.columns),
            'has_training_split': 'training' in df.columns
        }


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
                    # Note: lon/lat are stored as a single column [lon, lat]
                    columns = ['tweet_id', 'timestamp', 'user_id', 'lon_lat',
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
        dataset_name: 'twitter', 'reddit', 'bluesky', 'survey_twitter', 'twitteraae', or 'dadit'
        **kwargs: Arguments passed to specific loader

    Returns:
        Loaded DataFrame
    """
    dataset_name = dataset_name.lower()

    # Persona-based datasets (including survey_twitter)
    if dataset_name in ['twitter', 'reddit', 'bluesky', 'survey_twitter']:
        # Try default path first, then alternative paths
        default_path = f'./datasets/{dataset_name}/personas.pkl'
        alt_paths = {
            'bluesky': '../demdia_val/data/bluesky/personas.pkl',
            'survey_twitter': './datasets/survey_twitter/personas.pkl'
        }

        # Check if default path exists, otherwise try alternative
        if Path(default_path).exists():
            dataset_path = default_path
        elif dataset_name in alt_paths and Path(alt_paths[dataset_name]).exists():
            dataset_path = alt_paths[dataset_name]
            print(f"Using alternative path: {dataset_path}")
        else:
            dataset_path = default_path  # Will raise error in PersonaDatasetLoader

        loader = PersonaDatasetLoader(kwargs.get('dataset_path', dataset_path))
        return loader.load(
            sample_size=kwargs.get('sample_size'),
            training_only=kwargs.get('training_only', False)
        )

    # TwitterAAE dataset
    elif dataset_name == 'twitteraae':
        loader = TwitterAAELoader(kwargs.get('zip_path',
                                            '/data/nicpag/AI_recsys_project/TwitterAAE-full-v1.zip'))
        return loader.load(version=kwargs.get('version', 'limited_aa'),
                          sample_size=kwargs.get('sample_size'))

    # DADIT dataset
    elif dataset_name == 'dadit':
        loader = DADITLoader(kwargs.get('zip_path',
                                       '/data/nicpag/AI_recsys_project/DADIT_data_for_publishing.zip'))
        return loader.load(version=kwargs.get('version', 'test_anon'),
                          sample_size=kwargs.get('sample_size'))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                       f"Available: twitter, reddit, bluesky, survey_twitter, twitteraae, dadit")
