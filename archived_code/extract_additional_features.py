"""
Feature Extraction Pipeline for LLM Recommendation Bias Analysis

This script extracts additional features needed for the comprehensive analysis:
1. Toxicity scores (using Detoxify)
2. Author-level demographics (gender, political leaning) using LLM inference

The features are computed on the original 5000-post pools and saved as lookup tables
that can be joined to experiment results using 'original_index'.

Usage:
    # Extract all features for all datasets
    python extract_additional_features.py --all

    # Extract specific features
    python extract_additional_features.py --datasets twitter bluesky --features toxicity

    # Use specific model for author inference
    python extract_additional_features.py --all --author-model claude-3-5-haiku-20241022
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For toxicity detection
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    print("Warning: detoxify not installed. Install with: pip install detoxify")
    DETOXIFY_AVAILABLE = False

# For LLM inference
from utils.llm_client import get_llm_client
from data.loaders import load_dataset


class FeatureExtractor:
    """Extract additional features for bias analysis."""

    def __init__(self, dataset_name: str, output_dir: str = "outputs/features"):
        """
        Initialize feature extractor.

        Args:
            dataset_name: Name of dataset (twitter, bluesky, reddit)
            output_dir: Where to save feature files
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load original dataset (with same sampling as experiments)
        print(f"\nLoading {dataset_name} dataset...")
        self.df = load_dataset(dataset_name, sample_size=5000, training_only=True)
        # Reset index to 0-4999 to match experiment original_index
        self.df = self.df.reset_index(drop=True)
        print(f"Loaded {len(self.df)} posts (index: 0-{len(self.df)-1})")

        # Determine text column
        self.text_col = 'message' if 'message' in self.df.columns else 'text'

        # Initialize toxicity model if available
        self.toxicity_model = None
        if DETOXIFY_AVAILABLE:
            print("Initializing Detoxify model...")
            self.toxicity_model = Detoxify('original', device='cpu')

    def extract_toxicity_scores(self) -> pd.DataFrame:
        """
        Extract toxicity scores for all posts using Detoxify.

        Returns:
            DataFrame with columns: index, toxicity, severe_toxicity, obscene,
                                   threat, insult, identity_attack
        """
        if not DETOXIFY_AVAILABLE or self.toxicity_model is None:
            print("ERROR: Detoxify not available")
            return None

        print(f"\nExtracting toxicity scores for {len(self.df)} posts...")

        results = []
        batch_size = 32

        for i in tqdm(range(0, len(self.df), batch_size), desc="Processing batches"):
            batch = self.df.iloc[i:i+batch_size]
            texts = batch[self.text_col].fillna("").tolist()

            # Get predictions
            predictions = self.toxicity_model.predict(texts)

            # Store results (index is already 0-4999 after reset_index)
            for j, idx in enumerate(batch.index):
                results.append({
                    'original_index': int(idx),
                    'toxicity': float(predictions['toxicity'][j]),
                    'severe_toxicity': float(predictions['severe_toxicity'][j]),
                    'obscene': float(predictions['obscene'][j]),
                    'threat': float(predictions['threat'][j]),
                    'insult': float(predictions['insult'][j]),
                    'identity_attack': float(predictions['identity_attack'][j])
                })

        toxicity_df = pd.DataFrame(results)

        # Save (try parquet first, fallback to CSV)
        try:
            output_file = self.output_dir / f"{self.dataset_name}_toxicity.parquet"
            toxicity_df.to_parquet(output_file, index=False)
            print(f"✓ Saved toxicity scores to: {output_file}")
        except ImportError:
            output_file = self.output_dir / f"{self.dataset_name}_toxicity.csv"
            toxicity_df.to_csv(output_file, index=False)
            print(f"✓ Saved toxicity scores to: {output_file} (CSV fallback)")

        # Print statistics
        print(f"\nToxicity Statistics:")
        print(f"  Mean toxicity: {toxicity_df['toxicity'].mean():.4f}")
        print(f"  Posts with toxicity > 0.5: {(toxicity_df['toxicity'] > 0.5).sum()} ({(toxicity_df['toxicity'] > 0.5).mean()*100:.1f}%)")
        print(f"  Posts with toxicity > 0.8: {(toxicity_df['toxicity'] > 0.8).sum()} ({(toxicity_df['toxicity'] > 0.8).mean()*100:.1f}%)")

        return toxicity_df

    def extract_author_demographics(self, llm_provider: str = 'anthropic',
                                   llm_model: str = 'claude-3-5-haiku-20241022',
                                   batch_size: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Extract author-level demographics using LLM inference.

        This analyzes all posts from each author to infer:
        - Gender: male, female, non-binary, organization, unknown
        - Political leaning: left, center-left, center, center-right, right, apolitical, unknown

        Args:
            llm_provider: LLM provider to use
            llm_model: Specific model to use
            batch_size: Number of authors to process before saving checkpoint

        Returns:
            Dict with 'gender' and 'politics' DataFrames
        """
        print(f"\nExtracting author demographics using {llm_provider}/{llm_model}...")

        # Check if username/author column exists
        author_col = None
        for col in ['username', 'user_id', 'author']:
            if col in self.df.columns:
                author_col = col
                break

        if author_col is None:
            print("ERROR: No author/username column found in dataset")
            return None

        # Group posts by author
        author_posts = self.df.groupby(author_col)[self.text_col].apply(list).to_dict()
        print(f"Found {len(author_posts)} unique authors")

        # Initialize LLM client
        llm_client = get_llm_client(provider=llm_provider, model=llm_model)

        # Check for existing checkpoint
        checkpoint_file = self.output_dir / f"{self.dataset_name}_author_demographics_checkpoint.pkl"
        if checkpoint_file.exists():
            print(f"Loading checkpoint from {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            gender_results = checkpoint.get('gender', {})
            politics_results = checkpoint.get('politics', {})
            processed_authors = set(gender_results.keys())
            print(f"Resuming from checkpoint: {len(processed_authors)} authors already processed")
        else:
            gender_results = {}
            politics_results = {}
            processed_authors = set()

        # Process each author
        authors_to_process = [a for a in author_posts.keys() if a not in processed_authors]

        if len(authors_to_process) == 0:
            print("All authors already processed!")
        else:
            print(f"Processing {len(authors_to_process)} remaining authors...")

            for i, author in enumerate(tqdm(authors_to_process, desc="Inferring demographics")):
                posts = author_posts[author]

                # Limit to first 50 posts to avoid context limits
                posts_sample = posts[:50]
                posts_text = "\n\n".join([f"Post {j+1}: {post}" for j, post in enumerate(posts_sample)])

                # Infer gender
                gender_prompt = f"""Based on the following social media posts from a single author, infer their likely gender identity or account type.

Consider self-references, language patterns, and explicit statements.

Posts:
{posts_text}

Respond with ONLY ONE of these exact labels:
- male
- female
- non-binary
- organization
- unknown

Your response (one word only):"""

                try:
                    gender_response = llm_client.generate(gender_prompt, temperature=0.0)
                    gender = gender_response.strip().lower()
                    # Validate response
                    valid_genders = ['male', 'female', 'non-binary', 'organization', 'unknown']
                    if gender not in valid_genders:
                        gender = 'unknown'
                    gender_results[author] = gender
                except Exception as e:
                    print(f"\nError inferring gender for {author}: {e}")
                    gender_results[author] = 'unknown'

                # Infer political leaning
                politics_prompt = f"""Based on the following social media posts from a single author, classify their political leaning.

Consider policy positions, language use, and topics discussed.

Posts:
{posts_text}

Respond with ONLY ONE of these exact labels:
- left
- center-left
- center
- center-right
- right
- apolitical
- unknown

Your response (one word only):"""

                try:
                    politics_response = llm_client.generate(politics_prompt, temperature=0.0)
                    politics = politics_response.strip().lower()
                    # Validate response
                    valid_politics = ['left', 'center-left', 'center', 'center-right', 'right', 'apolitical', 'unknown']
                    if politics not in valid_politics:
                        politics = 'unknown'
                    politics_results[author] = politics
                except Exception as e:
                    print(f"\nError inferring politics for {author}: {e}")
                    politics_results[author] = 'unknown'

                # Save checkpoint periodically
                if (i + 1) % batch_size == 0:
                    checkpoint = {
                        'gender': gender_results,
                        'politics': politics_results
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint, f)

        # Create lookup DataFrames
        # Map author to each post's original index
        author_gender_map = []
        author_politics_map = []

        for idx, row in self.df.iterrows():
            author = row[author_col]
            author_gender_map.append({
                'original_index': idx,
                'author': author,
                'author_gender': gender_results.get(author, 'unknown')
            })
            author_politics_map.append({
                'original_index': idx,
                'author': author,
                'author_politics': politics_results.get(author, 'unknown')
            })

        gender_df = pd.DataFrame(author_gender_map)
        politics_df = pd.DataFrame(author_politics_map)

        # Save final results (try parquet first, fallback to CSV)
        try:
            gender_file = self.output_dir / f"{self.dataset_name}_author_gender.parquet"
            politics_file = self.output_dir / f"{self.dataset_name}_author_politics.parquet"
            gender_df.to_parquet(gender_file, index=False)
            politics_df.to_parquet(politics_file, index=False)
        except ImportError:
            gender_file = self.output_dir / f"{self.dataset_name}_author_gender.csv"
            politics_file = self.output_dir / f"{self.dataset_name}_author_politics.csv"
            gender_df.to_csv(gender_file, index=False)
            politics_df.to_csv(politics_file, index=False)

        print(f"\n✓ Saved gender inferences to: {gender_file}")
        print(f"✓ Saved politics inferences to: {politics_file}")

        # Print statistics
        print(f"\nGender Distribution:")
        gender_counts = gender_df['author_gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count} ({count/len(gender_df)*100:.1f}%)")

        print(f"\nPolitical Leaning Distribution:")
        politics_counts = politics_df['author_politics'].value_counts()
        for politics, count in politics_counts.items():
            print(f"  {politics}: {count} ({count/len(politics_df)*100:.1f}%)")

        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n✓ Removed checkpoint file")

        return {
            'gender': gender_df,
            'politics': politics_df
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract additional features for bias analysis'
    )

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['twitter', 'bluesky', 'reddit'],
        choices=['twitter', 'bluesky', 'reddit'],
        help='Datasets to process'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets'
    )

    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=['toxicity', 'author'],
        choices=['toxicity', 'author', 'all'],
        help='Which features to extract'
    )

    parser.add_argument(
        '--author-provider',
        type=str,
        default='openai',
        help='LLM provider for author inference'
    )

    parser.add_argument(
        '--author-model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model for author inference'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/features',
        help='Output directory for feature files'
    )

    args = parser.parse_args()

    # Determine datasets to process
    if args.all:
        datasets = ['twitter', 'bluesky', 'reddit']
    else:
        datasets = args.datasets

    # Determine features to extract
    extract_toxicity = 'toxicity' in args.features or 'all' in args.features
    extract_author = 'author' in args.features or 'all' in args.features

    print("="*80)
    print("FEATURE EXTRACTION PIPELINE")
    print("="*80)
    print(f"\nDatasets: {', '.join(datasets)}")
    print(f"Features: {', '.join(args.features)}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Process each dataset
    for dataset in datasets:
        print("\n" + "="*80)
        print(f"PROCESSING: {dataset.upper()}")
        print("="*80)

        try:
            extractor = FeatureExtractor(dataset, args.output_dir)

            # Extract toxicity
            if extract_toxicity:
                if DETOXIFY_AVAILABLE:
                    extractor.extract_toxicity_scores()
                else:
                    print("\n⚠ Skipping toxicity: detoxify not installed")
                    print("  Install with: pip install detoxify")

            # Extract author demographics
            if extract_author:
                extractor.extract_author_demographics(
                    llm_provider=args.author_provider,
                    llm_model=args.author_model
                )

            print(f"\n✓ Completed feature extraction for {dataset}")

        except Exception as e:
            print(f"\n✗ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nFeatures saved to: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Run join_features_to_experiments.py to add features to experiment data")
    print("  2. Run comprehensive bias analysis")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
