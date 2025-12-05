"""
Extract user-level metadata from personas.pkl dataset

This script parses the LLM-generated persona descriptions to extract
structured metadata about each user.
"""

import pickle
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any
import json


def extract_political_leaning(persona_text: str) -> Dict[str, Any]:
    """Extract political leaning from persona description."""

    text_lower = persona_text.lower()

    # Keywords for conservative/right
    conservative_keywords = [
        'conservative', 'maga', 'trump', 'republican', 'right-wing',
        'right wing', 'pro-trump', 'gop'
    ]

    # Keywords for liberal/left
    liberal_keywords = [
        'liberal', 'progressive', 'democrat', 'democratic', 'left-wing',
        'left wing', 'pro-biden', 'voteblue'
    ]

    # Count matches
    conservative_count = sum(1 for kw in conservative_keywords if kw in text_lower)
    liberal_count = sum(1 for kw in liberal_keywords if kw in text_lower)

    if conservative_count > liberal_count:
        return {
            'political_leaning': 'conservative',
            'is_political': True,
            'confidence': 'high' if conservative_count >= 2 else 'medium'
        }
    elif liberal_count > conservative_count:
        return {
            'political_leaning': 'liberal',
            'is_political': True,
            'confidence': 'high' if liberal_count >= 2 else 'medium'
        }
    elif conservative_count > 0 or liberal_count > 0:
        return {
            'political_leaning': 'center',
            'is_political': True,
            'confidence': 'low'
        }
    else:
        # Check for general political engagement
        political_keywords = ['politic', 'election', 'government', 'policy']
        is_political = any(kw in text_lower for kw in political_keywords)

        return {
            'political_leaning': 'unknown',
            'is_political': is_political,
            'confidence': 'none'
        }


def extract_topics(persona_text: str) -> list:
    """Extract main topics of interest."""

    topics = []
    text_lower = persona_text.lower()

    # Topic keywords
    topic_map = {
        'politics': ['politic', 'election', 'government', 'policy', 'conservative', 'liberal'],
        'sports': ['sports', 'football', 'baseball', 'basketball', 'soccer', 'athlete'],
        'technology': ['tech', 'crypto', 'blockchain', 'software', 'ai', 'tesla'],
        'healthcare': ['health', 'medical', 'nurse', 'doctor', 'surgeon', 'patient'],
        'business': ['business', 'entrepreneur', 'startup', 'finance', 'economy'],
        'social_issues': ['social', 'justice', 'equity', 'rights', 'activism'],
        'entertainment': ['entertainment', 'movie', 'tv', 'music', 'celebrity'],
        'religion': ['faith', 'religious', 'biblical', 'christian', 'spiritual'],
        'family': ['family', 'parenting', 'children', 'kids'],
        'food': ['food', 'cooking', 'recipe', 'restaurant']
    }

    for topic, keywords in topic_map.items():
        if any(kw in text_lower for kw in keywords):
            topics.append(topic)

    return topics if topics else ['general']


def extract_writing_style(persona_text: str) -> Dict[str, Any]:
    """Extract writing style characteristics."""

    text_lower = persona_text.lower()

    # Tone descriptors
    tones = []
    tone_keywords = {
        'casual': ['casual', 'informal'],
        'formal': ['formal', 'professional'],
        'sarcastic': ['sarcastic', 'irony', 'ironic'],
        'enthusiastic': ['enthusiastic', 'passionate', 'fervent'],
        'humorous': ['humor', 'humorous', 'witty', 'funny'],
        'analytical': ['analytical', 'critical', 'thoughtful'],
        'assertive': ['assertive', 'direct', 'confrontational'],
        'warm': ['warm', 'friendly', 'supportive']
    }

    for tone, keywords in tone_keywords.items():
        if any(kw in text_lower for kw in keywords):
            tones.append(tone)

    # Length descriptors
    length = 'medium'
    if 'concise' in text_lower or 'brief' in text_lower:
        length = 'short'
    elif 'detailed' in text_lower or 'lengthy' in text_lower:
        length = 'long'

    # Feature usage
    uses_emoji = 'emoji' in text_lower
    uses_slang = 'slang' in text_lower
    uses_hashtags = 'hashtag' in text_lower

    return {
        'tones': tones if tones else ['neutral'],
        'length': length,
        'uses_emoji': uses_emoji,
        'uses_slang': uses_slang,
        'uses_hashtags': uses_hashtags
    }


def extract_profession(persona_text: str) -> str:
    """Extract profession if mentioned."""

    text_lower = persona_text.lower()

    professions = [
        'journalist', 'writer', 'author', 'blogger',
        'doctor', 'nurse', 'surgeon', 'physician',
        'teacher', 'professor', 'educator',
        'lawyer', 'attorney',
        'engineer', 'developer', 'programmer',
        'entrepreneur', 'business owner',
        'analyst', 'consultant',
        'artist', 'designer',
        'scientist', 'researcher'
    ]

    for profession in professions:
        if profession in text_lower:
            return profession

    return 'unknown'


def extract_gender(persona_text: str, username: str) -> str:
    """Extract gender if mentioned or can be inferred."""

    text_lower = persona_text.lower()

    # Direct mentions
    if any(word in text_lower for word in ['he ', 'his ', 'him ', 'male', 'man']):
        return 'male'
    elif any(word in text_lower for word in ['she ', 'her ', 'hers ', 'female', 'woman']):
        return 'female'

    # Username hints (very rough heuristics)
    username_lower = username.lower()
    female_names = ['jenny', 'sara', 'sarah', 'jessica', 'emily', 'katie', 'amy', 'lisa']
    male_names = ['john', 'mike', 'james', 'david', 'robert', 'william', 'mark']

    for name in female_names:
        if name in username_lower:
            return 'female'
    for name in male_names:
        if name in username_lower:
            return 'male'

    return 'unknown'


def extract_sentiment_polarity(persona_text: str) -> str:
    """Extract overall sentiment/polarity of persona."""

    text_lower = persona_text.lower()

    # Positive indicators
    positive_words = ['passionate', 'enthusiastic', 'supportive', 'positive', 'warm',
                      'encouraging', 'optimistic', 'hopeful', 'friendly']

    # Negative indicators
    negative_words = ['critical', 'skeptical', 'confrontational', 'negative',
                      'sarcastic', 'cynical', 'critical']

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'


def extract_user_metadata(username: str, persona_text: str, tweet_count: int) -> Dict[str, Any]:
    """Extract all metadata for a user."""

    political = extract_political_leaning(persona_text)
    style = extract_writing_style(persona_text)

    metadata = {
        'username': username,
        'tweet_count': tweet_count,

        # Political
        'political_leaning': political['political_leaning'],
        'is_political': political['is_political'],
        'political_confidence': political['confidence'],

        # Topics
        'topics': extract_topics(persona_text),
        'primary_topic': extract_topics(persona_text)[0],

        # Writing style
        'tones': style['tones'],
        'primary_tone': style['tones'][0],
        'message_length': style['length'],
        'uses_emoji': style['uses_emoji'],
        'uses_slang': style['uses_slang'],
        'uses_hashtags': style['uses_hashtags'],

        # Demographics
        'profession': extract_profession(persona_text),
        'gender': extract_gender(persona_text, username),

        # Sentiment
        'sentiment_polarity': extract_sentiment_polarity(persona_text),

        # Original persona
        'persona_description': persona_text
    }

    return metadata


def main():
    """Main extraction pipeline."""

    print('='*80)
    print('PERSONA METADATA EXTRACTION')
    print('='*80)
    print()

    # Load data
    print('Loading personas.pkl...')
    with open('../demdia_val/data/twitter/personas.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f'Loaded {len(data)} tweets from {data.index.get_level_values("username").nunique()} users')
    print()

    # Get one row per user (persona is the same for all tweets from a user)
    print('Extracting unique users...')
    user_personas = []

    for username in data.index.get_level_values('username').unique():
        user_data = data.loc[username]
        persona_text = user_data.iloc[0]['persona']
        tweet_count = len(user_data)

        user_personas.append({
            'username': username,
            'persona_text': persona_text,
            'tweet_count': tweet_count
        })

    print(f'Found {len(user_personas)} unique users')
    print()

    # Extract metadata for each user
    print('Extracting metadata...')
    user_metadata_list = []

    for i, user_info in enumerate(user_personas, 1):
        if i % 50 == 0:
            print(f'  Processed {i}/{len(user_personas)} users...')

        metadata = extract_user_metadata(
            user_info['username'],
            user_info['persona_text'],
            user_info['tweet_count']
        )
        user_metadata_list.append(metadata)

    print(f'  Completed! Extracted metadata for {len(user_metadata_list)} users')
    print()

    # Convert to DataFrame
    user_metadata_df = pd.DataFrame(user_metadata_list)

    # Save results
    output_dir = Path('./outputs/persona_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_dir / 'user_metadata.csv'
    user_metadata_df.to_csv(csv_path, index=False)
    print(f'Saved metadata to {csv_path}')

    # Save as pickle (preserves list types)
    pkl_path = output_dir / 'user_metadata.pkl'
    user_metadata_df.to_pickle(pkl_path)
    print(f'Saved metadata to {pkl_path}')

    # Generate summary statistics
    print()
    print('='*80)
    print('SUMMARY STATISTICS')
    print('='*80)
    print()

    print(f'Total users: {len(user_metadata_df)}')
    print(f'Total tweets: {user_metadata_df["tweet_count"].sum()}')
    print(f'Avg tweets per user: {user_metadata_df["tweet_count"].mean():.1f}')
    print()

    print('Political Leaning:')
    for leaning, count in user_metadata_df['political_leaning'].value_counts().items():
        pct = count / len(user_metadata_df) * 100
        print(f'  {leaning:15s}: {count:3d} ({pct:5.1f}%)')
    print()

    print('Primary Topics:')
    for topic, count in user_metadata_df['primary_topic'].value_counts().head(10).items():
        pct = count / len(user_metadata_df) * 100
        print(f'  {topic:15s}: {count:3d} ({pct:5.1f}%)')
    print()

    print('Gender:')
    for gender, count in user_metadata_df['gender'].value_counts().items():
        pct = count / len(user_metadata_df) * 100
        print(f'  {gender:15s}: {count:3d} ({pct:5.1f}%)')
    print()

    print('Primary Tone:')
    for tone, count in user_metadata_df['primary_tone'].value_counts().items():
        pct = count / len(user_metadata_df) * 100
        print(f'  {tone:15s}: {count:3d} ({pct:5.1f}%)')
    print()

    print('Profession:')
    prof_counts = user_metadata_df['profession'].value_counts()
    for profession, count in prof_counts.head(10).items():
        pct = count / len(user_metadata_df) * 100
        print(f'  {profession:15s}: {count:3d} ({pct:5.1f}%)')

    # Save summary
    summary = {
        'total_users': int(len(user_metadata_df)),
        'total_tweets': int(user_metadata_df['tweet_count'].sum()),
        'avg_tweets_per_user': float(user_metadata_df['tweet_count'].mean()),
        'political_leaning': user_metadata_df['political_leaning'].value_counts().to_dict(),
        'primary_topics': user_metadata_df['primary_topic'].value_counts().to_dict(),
        'gender': user_metadata_df['gender'].value_counts().to_dict(),
        'primary_tone': user_metadata_df['primary_tone'].value_counts().to_dict(),
        'profession': user_metadata_df['profession'].value_counts().to_dict()
    }

    summary_path = output_dir / 'metadata_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print(f'Saved summary to {summary_path}')

    print()
    print('✓ Metadata extraction complete!')


if __name__ == '__main__':
    main()
