#!/usr/bin/env python3
"""
Step 1: Prepare Twitter Survey Data for LLM Recommendation Experiments

This script merges tweet-level and user-level survey data, extracts features,
and prepares the data in the format required for running LLM experiments.

Usage:
    python 1_prepare_survey_data.py \
        --tweets /path/to/tweet_data.csv \
        --users /path/to/user_survey_data.csv \
        --output data/prepared_posts.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re
import logging
from textblob import TextBlob

# ============================================================================
# FEATURE TYPE DEFINITIONS
# ============================================================================
# IMPORTANT: Modify these if your data uses different formats!

FEATURE_TYPES = {
    # Demographic features (from survey)
    'author_gender': 'categorical',
    'author_ideology': 'categorical',  # Change to 'numerical' if using numeric scale
    'author_partisanship': 'categorical',
    'author_race': 'categorical',
    'author_income': 'categorical',  # Change to 'numerical' if using dollar amounts
    'author_education': 'categorical',
    'author_marital_status': 'categorical',
    'author_religiosity': 'categorical',
    'author_age': 'numerical',
    
    # Tweet text metrics
    'text_length': 'numerical',
    'avg_word_length': 'numerical',
    'word_count': 'numerical',
    
    # Tweet style
    'has_url': 'binary',
    'has_hashtag': 'binary',
    'has_mention': 'binary',
    'has_emoji': 'binary',
    
    # Tweet type
    'is_reply': 'binary',
    'is_retweet': 'binary',
    'is_quote': 'binary',
    
    # Sentiment
    'sentiment_polarity': 'numerical',
    'sentiment_subjectivity': 'numerical',
    
    # User metadata
    'user_followers_count': 'numerical',
    'user_friends_count': 'numerical',
    'user_statuses_count': 'numerical',
    'user_account_age_days': 'numerical',
    'user_verified': 'binary',
    
    # Engagement
    'engagement_score': 'numerical',
}

# ============================================================================
# COLUMN NAME MAPPING
# ============================================================================
# Maps your column names to our standard names
# Modify if your columns have different names!

COLUMN_MAPPING = {
    # Survey columns
    'gender': 'author_gender',
    'ideology': 'author_ideology',
    'partisanship': 'author_partisanship',
    'race': 'author_race',
    'income': 'author_income',
    'education': 'author_education',
    'marital_status': 'author_marital_status',
    'religiosity': 'author_religiosity',
    'age': 'author_age',
    
    # Twitter user metadata columns
    'followers_count': 'user_followers_count',
    'friends_count': 'user_friends_count',
    'statuses_count': 'user_statuses_count',
    'verified': 'user_verified',
    'created_at': 'account_created_at',  # User account creation
    
    # Tweet columns (usually don't need mapping)
    'tweet_id': 'tweet_id',
    'user_id': 'user_id',
    'text': 'text',
    'full_text': 'text',  # Use full_text if available
}


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/preprocessing.log'),
            logging.StreamHandler()
        ]
    )


def load_and_merge_data(tweets_path, users_path):
    """Load and merge tweet and user data"""
    logging.info(f"Loading tweet data from {tweets_path}")
    tweets = pd.read_csv(tweets_path)
    logging.info(f"  Loaded {len(tweets)} tweets")
    
    logging.info(f"Loading user survey data from {users_path}")
    users = pd.read_csv(users_path)
    logging.info(f"  Loaded {len(users)} users")
    
    # Ensure user_id exists in both
    if 'user_id' not in tweets.columns:
        raise ValueError("tweets data must have 'user_id' column")
    if 'user_id' not in users.columns:
        raise ValueError("users data must have 'user_id' column")
    
    # Merge
    logging.info("Merging tweet and user data...")
    merged = tweets.merge(users, on='user_id', how='inner', suffixes=('_tweet', '_user'))
    logging.info(f"  Merged data: {len(merged)} rows")
    
    if len(merged) == 0:
        raise ValueError("Merge resulted in 0 rows! Check that user_id values match between files.")
    
    return merged


def extract_text_features(df):
    """Extract text-based features"""
    logging.info("Extracting text features...")
    
    # Get text column (prefer full_text if available)
    text_col = 'full_text' if 'full_text' in df.columns else 'text'
    if text_col not in df.columns:
        logging.warning(f"No text column found!")
        return df
    
    df['text'] = df[text_col].fillna('')
    
    # Text length
    df['text_length'] = df['text'].str.len()
    
    # Word count and average word length
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )
    
    # Style features
    df['has_url'] = df['text'].str.contains(r'http[s]?://\S+', regex=True).astype(int)
    df['has_hashtag'] = df['text'].str.contains('#', regex=False).astype(int)
    df['has_mention'] = df['text'].str.contains('@', regex=False).astype(int)
    
    # Emoji detection (simple)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    df['has_emoji'] = df['text'].apply(lambda x: 1 if emoji_pattern.search(x) else 0)
    
    logging.info(f"  Extracted text features for {len(df)} rows")
    return df


def extract_sentiment(df):
    """Extract sentiment using TextBlob"""
    logging.info("Computing sentiment (this may take a while)...")
    
    def get_sentiment(text):
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0.0, 0.0
    
    sentiments = df['text'].apply(get_sentiment)
    df['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
    df['sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])
    
    logging.info(f"  Computed sentiment for {len(df)} rows")
    return df


def extract_tweet_type(df):
    """Determine tweet type (reply, retweet, quote)"""
    logging.info("Extracting tweet type features...")
    
    # Reply
    df['is_reply'] = (df.get('in_reply_to_status_id_str', pd.Series()).notna()).astype(int)
    
    # Retweet
    df['is_retweet'] = (df.get('retweeted_status_id', pd.Series()).notna()).astype(int)
    
    # Quote
    if 'is_quote_status' in df.columns:
        df['is_quote'] = df['is_quote_status'].fillna(False).astype(int)
    else:
        df['is_quote'] = (df.get('quoted_status_id_str', pd.Series()).notna()).astype(int)
    
    return df


def extract_engagement(df):
    """Compute engagement score"""
    logging.info("Computing engagement score...")
    
    favorites = df.get('favorite_count', pd.Series(0)).fillna(0)
    retweets = df.get('retweet_count', pd.Series(0)).fillna(0)
    
    # Log-transformed engagement
    df['engagement_score'] = np.log1p(favorites + retweets)
    
    return df


def extract_user_metadata(df):
    """Extract user-level metadata"""
    logging.info("Extracting user metadata features...")
    
    # Account age
    if 'account_created_at' in df.columns or 'created_at_user' in df.columns:
        created_col = 'account_created_at' if 'account_created_at' in df.columns else 'created_at_user'
        try:
            df['user_account_age_days'] = (
                pd.Timestamp.now() - pd.to_datetime(df[created_col])
            ).dt.days
        except:
            logging.warning("Could not compute account age")
            df['user_account_age_days'] = np.nan
    
    # Verified status
    if 'verified' in df.columns or 'user_verified' in df.columns:
        verified_col = 'user_verified' if 'user_verified' in df.columns else 'verified'
        df['user_verified'] = df[verified_col].fillna(False).astype(int)
    
    return df


def map_survey_features(df):
    """Map survey column names to standard names"""
    logging.info("Mapping survey feature names...")
    
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
            logging.info(f"  Mapped '{old_name}' → '{new_name}'")
    
    return df


def create_persona_description(row):
    """Create persona description for each user"""
    parts = []
    
    # Demographics
    if pd.notna(row.get('author_gender')):
        parts.append(f"{row['author_gender']}")
    if pd.notna(row.get('author_age')):
        parts.append(f"{int(row['author_age'])} years old")
    if pd.notna(row.get('author_education')):
        parts.append(f"{row['author_education']} education")
    if pd.notna(row.get('author_partisanship')):
        parts.append(f"{row['author_partisanship']}")
    if pd.notna(row.get('author_ideology')):
        parts.append(f"ideology: {row['author_ideology']}")
    
    return ", ".join(parts) if parts else "No demographic information available"


def prepare_for_experiments(df, sample_size=None):
    """Prepare final dataset for experiments"""
    logging.info("Preparing data for experiments...")
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        logging.info(f"Sampling {sample_size} tweets from {len(df)}")
        df = df.sample(n=sample_size, random_state=42)
    
    # Create persona
    df['persona'] = df.apply(create_persona_description, axis=1)
    
    # Add required experiment columns
    df['selected'] = 0  # Will be updated during experiments
    df['pool_position'] = range(len(df))
    df['original_index'] = df.index
    
    # Select and order columns
    required_cols = ['tweet_id', 'user_id', 'text', 'persona', 'selected', 'pool_position', 'original_index']
    
    # Get all feature columns
    feature_cols = [col for col in df.columns if col in FEATURE_TYPES.keys()]
    
    # Final column order
    final_cols = required_cols + feature_cols
    final_cols = [c for c in final_cols if c in df.columns]
    
    df_final = df[final_cols].copy()
    
    logging.info(f"Final dataset: {len(df_final)} rows, {len(final_cols)} columns")
    logging.info(f"Features extracted: {len(feature_cols)}")
    logging.info(f"Feature list: {', '.join(feature_cols)}")
    
    return df_final


def main():
    parser = argparse.ArgumentParser(description='Prepare survey data for LLM experiments')
    parser.add_argument('--tweets', required=True, help='Path to tweet_data.csv')
    parser.add_argument('--users', required=True, help='Path to user_survey_data.csv')
    parser.add_argument('--output', required=True, help='Output path for prepared data')
    parser.add_argument('--sample_size', type=int, help='Optional: sample N tweets')
    args = parser.parse_args()
    
    # Setup
    Path('outputs').mkdir(exist_ok=True)
    setup_logging()
    
    logging.info("="*80)
    logging.info("PREPARING TWITTER SURVEY DATA FOR LLM EXPERIMENTS")
    logging.info("="*80)
    
    # Load and merge
    df = load_and_merge_data(args.tweets, args.users)
    
    # Extract features
    df = map_survey_features(df)
    df = extract_text_features(df)
    df = extract_sentiment(df)
    df = extract_tweet_type(df)
    df = extract_engagement(df)
    df = extract_user_metadata(df)
    
    # Prepare final dataset
    df_final = prepare_for_experiments(df, args.sample_size)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    logging.info(f"\n✓ Prepared data saved to: {output_path}")
    logging.info(f"✓ Ready for running LLM recommendation experiments!")
    
    # Save feature metadata
    feature_metadata = pd.DataFrame([
        {'feature': feat, 'type': FEATURE_TYPES[feat]}
        for feat in FEATURE_TYPES.keys()
        if feat in df_final.columns
    ])
    metadata_path = output_path.parent / 'feature_metadata.csv'
    feature_metadata.to_csv(metadata_path, index=False)
    logging.info(f"✓ Feature metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
