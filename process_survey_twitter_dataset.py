#!/usr/bin/env python3
"""
Process Survey + Twitter Dataset for LLM Recommendation Bias Analysis

This script processes a dataset combining:
1. Twitter user-level data (account metadata, profile, activity stats)
2. Tweet-level data (content, engagement, replies, retweets, quotes)
3. Survey data (demographics: gender, ideology, partisanship, race, income, education, age)

Outputs:
1. Experiment-ready dataset (personas.pkl) for use with run_experiment.py
2. Analysis-ready dataset with pre-computed features for run_comprehensive_analysis.py

Usage:
    python process_survey_twitter_dataset.py \
        --users users.csv \
        --tweets tweets.csv \
        --survey survey.csv \
        --output-dir datasets/survey_twitter

Author: LLM Recommendation Bias Analysis Pipeline
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import re
from collections import Counter


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mapping from survey columns to analysis columns
DEMOGRAPHIC_MAPPINGS = {
    # Gender: map survey values to standard categories
    'gender': {
        'column': 'gender',  # Column name in survey data
        'output': 'author_gender',
        'mapping': {
            'male': 'male',
            'm': 'male',
            '1': 'male',
            'female': 'female',
            'f': 'female',
            '2': 'female',
            'non-binary': 'non-binary',
            'other': 'non-binary',
            '3': 'non-binary',
        },
        'default': 'unknown'
    },

    # Political leaning: map ideology/partisanship to categories
    'political': {
        'column': 'ideology',  # Can also use 'partisanship'
        'output': 'author_political_leaning',
        'mapping': {
            # Numeric scale (1=very liberal to 7=very conservative)
            '1': 'left',
            '2': 'center-left',
            '3': 'center-left',
            '4': 'center',
            '5': 'center-right',
            '6': 'center-right',
            '7': 'right',
            # Text labels
            'very liberal': 'left',
            'liberal': 'left',
            'slightly liberal': 'center-left',
            'moderate': 'center',
            'slightly conservative': 'center-right',
            'conservative': 'right',
            'very conservative': 'right',
            # Party affiliation
            'democrat': 'left',
            'democratic': 'left',
            'republican': 'right',
            'independent': 'center',
        },
        'default': 'unknown'
    },

    # Minority status: derived from race
    'minority': {
        'column': 'race',
        'output': 'author_is_minority',
        'mapping': {
            # Non-minority categories
            'white': 'no',
            'caucasian': 'no',
            '1': 'no',  # Often coded as white
            # Minority categories
            'black': 'yes',
            'african american': 'yes',
            'african-american': 'yes',
            'hispanic': 'yes',
            'latino': 'yes',
            'latina': 'yes',
            'asian': 'yes',
            'asian american': 'yes',
            'native american': 'yes',
            'american indian': 'yes',
            'pacific islander': 'yes',
            'middle eastern': 'yes',
            'mixed': 'yes',
            'multiracial': 'yes',
            'other': 'yes',
            '2': 'yes',  # Often coded as Black
            '3': 'yes',  # Often coded as Hispanic
            '4': 'yes',  # Often coded as Asian
            '5': 'yes',  # Often coded as Other
        },
        'default': 'unknown'
    }
}

# Features to compute from tweet text
TEXT_FEATURES = [
    'text_length',
    'avg_word_length',
    'has_emoji',
    'has_hashtag',
    'has_mention',
    'has_url',
]

# Sentiment features (computed using textblob/vader)
SENTIMENT_FEATURES = [
    'sentiment_polarity',
    'sentiment_subjectivity',
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_flexible(filepath: str, **kwargs) -> pd.DataFrame:
    """Load CSV with flexible encoding and separator detection."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            print(f"  Loaded {len(df):,} rows from {filepath}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if 'encoding' not in str(e).lower():
                raise

    raise ValueError(f"Could not load {filepath} with any encoding")


def load_users(filepath: str) -> pd.DataFrame:
    """Load user-level Twitter data."""
    print("\nLoading user data...")
    df = load_csv_flexible(filepath)

    # Standardize user_id column
    if 'id' in df.columns and 'user_id' not in df.columns:
        df['user_id'] = df['id']
    elif 'id_str' in df.columns and 'user_id' not in df.columns:
        df['user_id'] = df['id_str']

    # Ensure user_id is string for consistent merging
    df['user_id'] = df['user_id'].astype(str)

    print(f"  Columns: {list(df.columns)[:10]}...")
    print(f"  Unique users: {df['user_id'].nunique():,}")

    return df


def load_tweets(filepath: str) -> pd.DataFrame:
    """Load tweet-level data."""
    print("\nLoading tweet data...")
    df = load_csv_flexible(filepath)

    # Standardize columns
    if 'full_text' in df.columns and 'text' not in df.columns:
        df['text'] = df['full_text']

    # Ensure user_id is string
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(str)

    print(f"  Columns: {list(df.columns)[:10]}...")
    print(f"  Total tweets: {len(df):,}")
    print(f"  Unique users: {df['user_id'].nunique():,}")

    return df


def load_survey(filepath: str) -> pd.DataFrame:
    """Load survey demographic data."""
    print("\nLoading survey data...")
    df = load_csv_flexible(filepath)

    # Find and standardize user_id column
    user_id_candidates = ['user_id', 'twitter_id', 'twitter_user_id', 'id', 'respondent_id']
    user_id_col = None
    for col in user_id_candidates:
        if col in df.columns:
            user_id_col = col
            break

    if user_id_col is None:
        raise ValueError(f"Could not find user_id column. Available: {list(df.columns)}")

    if user_id_col != 'user_id':
        df['user_id'] = df[user_id_col]

    df['user_id'] = df['user_id'].astype(str)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Respondents: {len(df):,}")

    return df


# =============================================================================
# DATA MERGING
# =============================================================================

def merge_datasets(users_df: pd.DataFrame, tweets_df: pd.DataFrame,
                   survey_df: pd.DataFrame) -> pd.DataFrame:
    """Merge user, tweet, and survey data."""
    print("\n" + "="*60)
    print("MERGING DATASETS")
    print("="*60)

    # First merge survey with users
    print("\nMerging survey with user data...")
    user_survey = pd.merge(users_df, survey_df, on='user_id', how='inner', suffixes=('', '_survey'))
    print(f"  Matched users: {len(user_survey):,}")

    # Then merge with tweets
    print("\nMerging with tweet data...")
    merged = pd.merge(tweets_df, user_survey, on='user_id', how='inner', suffixes=('', '_user'))
    print(f"  Final dataset: {len(merged):,} tweets from {merged['user_id'].nunique():,} users")

    return merged


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def map_demographic(value: Any, mapping: Dict, default: str) -> str:
    """Map a demographic value using the provided mapping."""
    if pd.isna(value):
        return default

    value_str = str(value).lower().strip()

    if value_str in mapping:
        return mapping[value_str]

    # Try partial matching for text values
    for key, mapped_value in mapping.items():
        if key in value_str or value_str in key:
            return mapped_value

    return default


def extract_demographic_features(df: pd.DataFrame, survey_columns: Dict[str, str]) -> pd.DataFrame:
    """Extract and standardize demographic features from survey data."""
    print("\nExtracting demographic features...")

    result_df = df.copy()

    for feature_name, config in DEMOGRAPHIC_MAPPINGS.items():
        # Find the source column
        source_col = survey_columns.get(feature_name, config['column'])
        output_col = config['output']

        if source_col in df.columns:
            result_df[output_col] = df[source_col].apply(
                lambda x: map_demographic(x, config['mapping'], config['default'])
            )

            # Print distribution
            dist = result_df[output_col].value_counts()
            print(f"\n  {output_col}:")
            for val, count in dist.items():
                pct = count / len(result_df) * 100
                print(f"    {val}: {count:,} ({pct:.1f}%)")
        else:
            print(f"\n  Warning: Column '{source_col}' not found for {feature_name}")
            result_df[output_col] = config['default']

    return result_df


def extract_text_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Extract text-based features from tweets."""
    print("\nExtracting text features...")

    result_df = df.copy()

    # Text length
    result_df['text_length'] = result_df[text_col].fillna('').apply(len)

    # Average word length
    def avg_word_len(text):
        words = str(text).split()
        if not words:
            return 0
        return sum(len(w) for w in words) / len(words)

    result_df['avg_word_length'] = result_df[text_col].fillna('').apply(avg_word_len)

    # Style features (binary)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF]"
    )

    result_df['has_emoji'] = result_df[text_col].fillna('').apply(
        lambda x: 'yes' if emoji_pattern.search(str(x)) else 'no'
    )
    result_df['has_hashtag'] = result_df[text_col].fillna('').apply(
        lambda x: 'yes' if '#' in str(x) else 'no'
    )
    result_df['has_mention'] = result_df[text_col].fillna('').apply(
        lambda x: 'yes' if '@' in str(x) else 'no'
    )
    result_df['has_url'] = result_df[text_col].fillna('').apply(
        lambda x: 'yes' if any(p in str(x).lower() for p in ['http://', 'https://', 't.co/', 'www.']) else 'no'
    )

    # Print summary
    for col in ['text_length', 'avg_word_length']:
        print(f"  {col}: mean={result_df[col].mean():.1f}, std={result_df[col].std():.1f}")

    for col in ['has_emoji', 'has_hashtag', 'has_mention', 'has_url']:
        yes_pct = (result_df[col] == 'yes').mean() * 100
        print(f"  {col}: {yes_pct:.1f}% yes")

    return result_df


def extract_sentiment_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Extract sentiment features using TextBlob."""
    print("\nExtracting sentiment features...")

    result_df = df.copy()

    try:
        from textblob import TextBlob

        def get_sentiment(text):
            try:
                blob = TextBlob(str(text))
                return blob.sentiment.polarity, blob.sentiment.subjectivity
            except:
                return 0.0, 0.5

        sentiments = result_df[text_col].fillna('').apply(get_sentiment)
        result_df['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
        result_df['sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])

        print(f"  sentiment_polarity: mean={result_df['sentiment_polarity'].mean():.3f}")
        print(f"  sentiment_subjectivity: mean={result_df['sentiment_subjectivity'].mean():.3f}")

    except ImportError:
        print("  Warning: textblob not installed. Skipping sentiment features.")
        result_df['sentiment_polarity'] = 0.0
        result_df['sentiment_subjectivity'] = 0.5

    return result_df


def extract_content_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Extract content-based features (polarization, controversy, topic)."""
    print("\nExtracting content features...")

    result_df = df.copy()

    # Simple topic detection based on keywords
    topic_keywords = {
        'politics': ['trump', 'biden', 'democrat', 'republican', 'election', 'vote', 'congress', 'senate', 'president', 'government', 'policy'],
        'sports': ['game', 'team', 'win', 'player', 'score', 'nfl', 'nba', 'mlb', 'football', 'basketball', 'soccer'],
        'entertainment': ['movie', 'music', 'show', 'celebrity', 'star', 'album', 'concert', 'netflix', 'film', 'actor'],
        'technology': ['tech', 'ai', 'software', 'app', 'iphone', 'google', 'microsoft', 'apple', 'computer', 'digital'],
        'news': ['breaking', 'news', 'report', 'update', 'latest', 'today', 'announced', 'official'],
        'lifestyle': ['food', 'travel', 'health', 'fitness', 'recipe', 'workout', 'diet', 'vacation'],
    }

    def get_primary_topic(text):
        text_lower = str(text).lower()
        topic_counts = {}
        for topic, keywords in topic_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                topic_counts[topic] = count

        if topic_counts:
            return max(topic_counts, key=topic_counts.get)
        return 'other'

    result_df['primary_topic'] = result_df[text_col].fillna('').apply(get_primary_topic)

    # Polarization score (based on political keywords)
    polarizing_keywords = [
        'hate', 'love', 'worst', 'best', 'terrible', 'amazing', 'disgusting',
        'brilliant', 'stupid', 'genius', 'idiot', 'fascist', 'communist',
        'radical', 'extreme', 'destroy', 'save', 'evil', 'hero'
    ]

    def get_polarization(text):
        text_lower = str(text).lower()
        count = sum(1 for kw in polarizing_keywords if kw in text_lower)
        return min(count / 3.0, 1.0)  # Normalize to 0-1

    result_df['polarization_score'] = result_df[text_col].fillna('').apply(get_polarization)

    # Controversy level (categorical)
    def get_controversy(row):
        pol = row.get('polarization_score', 0)
        if pol > 0.5:
            return 'high'
        elif pol > 0.2:
            return 'medium'
        else:
            return 'low'

    result_df['controversy_level'] = result_df.apply(get_controversy, axis=1)

    # Print summary
    print(f"  primary_topic distribution:")
    for topic, count in result_df['primary_topic'].value_counts().head(5).items():
        print(f"    {topic}: {count:,} ({count/len(result_df)*100:.1f}%)")

    print(f"  polarization_score: mean={result_df['polarization_score'].mean():.3f}")
    print(f"  controversy_level distribution: {dict(result_df['controversy_level'].value_counts())}")

    return result_df


def extract_toxicity_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Extract toxicity features (placeholder - requires Perspective API or similar)."""
    print("\nExtracting toxicity features...")

    result_df = df.copy()

    # Simple keyword-based toxicity estimation (placeholder)
    toxic_keywords = [
        'hate', 'kill', 'die', 'stupid', 'idiot', 'dumb', 'shut up',
        'loser', 'moron', 'trash', 'garbage', 'pathetic', 'disgusting'
    ]

    def estimate_toxicity(text):
        text_lower = str(text).lower()
        count = sum(1 for kw in toxic_keywords if kw in text_lower)
        return min(count / 5.0, 1.0)  # Normalize to 0-1

    result_df['toxicity'] = result_df[text_col].fillna('').apply(estimate_toxicity)
    result_df['severe_toxicity'] = result_df['toxicity'].apply(lambda x: x * 0.5)  # Placeholder

    print(f"  toxicity: mean={result_df['toxicity'].mean():.3f}")
    print(f"  severe_toxicity: mean={result_df['severe_toxicity'].mean():.3f}")

    return result_df


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def create_experiment_dataset(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Create experiment-ready dataset (personas.pkl format)."""
    print("\nCreating experiment-ready dataset...")

    # Create persona description from demographics
    def create_persona(row):
        parts = []

        # Gender
        gender = row.get('author_gender', 'unknown')
        if gender != 'unknown':
            parts.append(f"a {gender}")

        # Political leaning
        political = row.get('author_political_leaning', 'unknown')
        if political != 'unknown':
            parts.append(f"politically {political}")

        # Race/minority status
        minority = row.get('author_is_minority', 'unknown')
        if minority == 'yes':
            parts.append("from a minority background")

        # Age if available
        age = row.get('age', row.get('Age', None))
        if age and not pd.isna(age):
            parts.append(f"{int(age)} years old")

        # Education if available
        education = row.get('education', row.get('Education', None))
        if education and not pd.isna(education):
            parts.append(f"with {education} education")

        if parts:
            return f"This user is {', '.join(parts)}."
        return "User demographics unknown."

    # Build output DataFrame
    output_df = pd.DataFrame({
        'user_id': df['user_id'],
        'username': df.get('screen_name', df.get('username', df['user_id'])),
        'message': df[text_col],
        'persona': df.apply(create_persona, axis=1),
        'training': 1,  # All data as training by default
    })

    # Add reply_to if available
    if 'in_reply_to_status_id_str' in df.columns:
        output_df['reply_to'] = df['in_reply_to_status_id_str']
    else:
        output_df['reply_to'] = None

    print(f"  Created dataset with {len(output_df):,} posts")

    return output_df


def create_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create analysis-ready dataset with all features."""
    print("\nCreating analysis-ready dataset...")

    # Required columns for analysis
    required_cols = [
        'user_id', 'text',
        # Demographics
        'author_gender', 'author_political_leaning', 'author_is_minority',
        # Text features
        'text_length', 'avg_word_length',
        'has_emoji', 'has_hashtag', 'has_mention', 'has_url',
        # Sentiment
        'sentiment_polarity', 'sentiment_subjectivity',
        # Content
        'polarization_score', 'controversy_level', 'primary_topic',
        # Toxicity
        'toxicity', 'severe_toxicity',
    ]

    # Select and verify columns
    available_cols = [col for col in required_cols if col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")

    output_df = df[available_cols].copy()

    # Add any additional useful columns
    optional_cols = [
        'tweet_id', 'created_at', 'favorite_count', 'retweet_count',
        'screen_name', 'followers_count', 'friends_count'
    ]
    for col in optional_cols:
        if col in df.columns:
            output_df[col] = df[col]

    print(f"  Created dataset with {len(output_df):,} rows and {len(output_df.columns)} columns")

    return output_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process Survey + Twitter dataset for LLM recommendation bias analysis'
    )
    parser.add_argument('--users', required=True, help='Path to users CSV file')
    parser.add_argument('--tweets', required=True, help='Path to tweets CSV file')
    parser.add_argument('--survey', required=True, help='Path to survey CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--text-col', default='text', help='Name of text column in tweets (default: text)')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size (default: all)')

    # Column mapping options
    parser.add_argument('--gender-col', default='gender', help='Gender column in survey')
    parser.add_argument('--political-col', default='ideology', help='Political ideology column in survey')
    parser.add_argument('--race-col', default='race', help='Race column in survey')

    args = parser.parse_args()

    print("="*60)
    print("SURVEY + TWITTER DATASET PROCESSOR")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    users_df = load_users(args.users)
    tweets_df = load_tweets(args.tweets)
    survey_df = load_survey(args.survey)

    # Merge datasets
    merged_df = merge_datasets(users_df, tweets_df, survey_df)

    # Sample if requested
    if args.sample_size and args.sample_size < len(merged_df):
        print(f"\nSampling {args.sample_size:,} tweets...")
        merged_df = merged_df.sample(n=args.sample_size, random_state=42)

    # Extract features
    survey_columns = {
        'gender': args.gender_col,
        'political': args.political_col,
        'minority': args.race_col,
    }

    df = extract_demographic_features(merged_df, survey_columns)
    df = extract_text_features(df, args.text_col)
    df = extract_sentiment_features(df, args.text_col)
    df = extract_content_features(df, args.text_col)
    df = extract_toxicity_features(df, args.text_col)

    # Create output datasets
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)

    # 1. Experiment-ready dataset (personas.pkl)
    experiment_df = create_experiment_dataset(df, args.text_col)
    experiment_path = output_dir / 'personas.pkl'
    experiment_df.to_pickle(experiment_path)
    print(f"\n  Saved experiment dataset: {experiment_path}")

    # 2. Analysis-ready dataset (parquet and CSV)
    analysis_df = create_analysis_dataset(df)
    analysis_parquet_path = output_dir / 'analysis_ready.parquet'
    analysis_csv_path = output_dir / 'analysis_ready.csv'
    analysis_df.to_parquet(analysis_parquet_path, index=False)
    analysis_df.to_csv(analysis_csv_path, index=False)
    print(f"  Saved analysis dataset: {analysis_parquet_path}")
    print(f"  Saved analysis dataset: {analysis_csv_path}")

    # 3. Full merged dataset (for reference)
    full_path = output_dir / 'full_merged_data.parquet'
    df.to_parquet(full_path, index=False)
    print(f"  Saved full merged dataset: {full_path}")

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nDataset: {args.output_dir}")
    print(f"  Total tweets: {len(df):,}")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"\nTo run experiments:")
    print(f"  python run_experiment.py --dataset survey_twitter --provider anthropic --prompt general")
    print(f"\nTo run analysis:")
    print(f"  python run_comprehensive_analysis.py")


if __name__ == '__main__':
    main()
