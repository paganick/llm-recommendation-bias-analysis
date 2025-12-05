"""
LLM-based User Metadata Inference

Use GPT to infer user-level attributes from a sample of their tweets.
"""

import pickle
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import openai
import os
from tqdm import tqdm

# Set up OpenAI client
openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


def sample_user_tweets(user_data: pd.DataFrame, n_samples: int = 75) -> List[str]:
    """Sample tweets from a user, stratified if possible."""

    if len(user_data) <= n_samples:
        # Use all tweets if user has fewer than n_samples
        return user_data['message'].tolist()
    else:
        # Sample uniformly across their timeline
        indices = np.linspace(0, len(user_data) - 1, n_samples, dtype=int)
        return user_data.iloc[indices]['message'].tolist()


def create_inference_prompt(username: str, tweets: List[str], persona_description: str = None) -> str:
    """Create the prompt for GPT to infer user metadata."""

    prompt = f"""You are analyzing Twitter/X user @{username} to infer demographic and behavioral characteristics.

Below are {len(tweets)} sample tweets from this user:

"""

    # Add tweets
    for i, tweet in enumerate(tweets[:50], 1):  # Limit to 50 for token efficiency
        prompt += f"{i}. {tweet}\n"

    if persona_description:
        prompt += f"\n\nAdditional context: {persona_description}\n"

    prompt += """

Based on these tweets, infer the following user-level attributes. Use your best judgment and cultural/linguistic knowledge. If you cannot confidently determine an attribute, mark it as "unknown" and set confidence to "low".

Provide your response as a JSON object with the following structure:

{
  "gender": {
    "value": "male" | "female" | "non-binary" | "unknown",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation"
  },
  "political_leaning": {
    "value": "left" | "center-left" | "center" | "center-right" | "right" | "unknown",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation"
  },
  "race_ethnicity": {
    "value": "african_american" | "white" | "hispanic_latino" | "asian" | "middle_eastern" | "mixed" | "unknown",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation based on AAVE, cultural references, etc."
  },
  "geographic_origin": {
    "value": "country or US region (e.g., 'US_South', 'UK', 'Canada')",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation based on slang, references, spelling"
  },
  "age_generation": {
    "value": "gen_z" | "millennial" | "gen_x" | "boomer" | "unknown",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation based on cultural references, language"
  },
  "profession": {
    "value": "specific profession or 'unknown'",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation"
  },
  "education_level": {
    "value": "high_school" | "some_college" | "college" | "graduate" | "unknown",
    "confidence": "high" | "medium" | "low",
    "reasoning": "brief explanation based on vocabulary, topics"
  },
  "primary_interests": ["list", "of", "3-5", "main", "interests"],
  "writing_style_notes": "brief description of distinctive writing style"
}

IMPORTANT:
- Be culturally sensitive and base inferences on linguistic patterns, not stereotypes
- For race/ethnicity, focus on AAVE usage, cultural references, self-identification cues
- Geographic origin can use spelling (colour vs color), slang, place mentions
- Mark as "unknown" if truly uncertain rather than guessing
- Confidence should reflect how clear the signals are

Return ONLY the JSON object, no other text.
"""

    return prompt


def query_gpt_for_inference(username: str, tweets: List[str],
                            persona_description: str = None,
                            model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Query GPT to infer user metadata."""

    prompt = create_inference_prompt(username, tweets, persona_description)

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing social media content to infer user demographics and characteristics. You provide structured, well-reasoned inferences based on linguistic and cultural patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent outputs
            max_tokens=800
        )

        # Parse JSON response
        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)

        # Add usage stats
        result['_usage'] = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }

        return result

    except Exception as e:
        print(f"Error querying GPT for user {username}: {e}")
        return {
            'error': str(e),
            'username': username
        }


def process_all_users(model: str = "gpt-4o-mini",
                     checkpoint_interval: int = 20,
                     start_from: int = 0):
    """Process all users with LLM inference."""

    print('='*80)
    print('LLM-BASED USER METADATA INFERENCE')
    print('='*80)
    print(f'Model: {model}')
    print()

    # Load data
    print('Loading personas.pkl...')
    with open('../demdia_val/data/twitter/personas.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f'Loaded {len(data)} tweets from {data.index.get_level_values("username").nunique()} users')
    print()

    # Get unique users
    usernames = data.index.get_level_values('username').unique().tolist()
    print(f'Processing {len(usernames)} users...')
    print()

    # Create output directory
    output_dir = Path('./outputs/llm_inference')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint
    checkpoint_path = output_dir / f'checkpoint_{model.replace("-", "_")}.json'
    results = []

    if checkpoint_path.exists() and start_from == 0:
        print(f'Found existing checkpoint: {checkpoint_path}')
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data['results']
            start_from = len(results)
        print(f'Resuming from user {start_from}')
        print()

    # Process users
    total_tokens = 0

    for i, username in enumerate(tqdm(usernames[start_from:],
                                      initial=start_from,
                                      total=len(usernames),
                                      desc="Processing users")):

        # Get user data
        user_data = data.loc[username]

        # Sample tweets
        tweets = sample_user_tweets(user_data, n_samples=75)

        # Get persona description (from first tweet, since it's the same for all)
        persona_description = user_data.iloc[0]['persona']

        # Query LLM
        inference_result = query_gpt_for_inference(username, tweets, persona_description, model)

        # Add metadata
        inference_result['username'] = username
        inference_result['tweet_count'] = len(user_data)
        inference_result['sampled_tweets'] = len(tweets)

        results.append(inference_result)

        # Track tokens
        if '_usage' in inference_result:
            total_tokens += inference_result['_usage']['total_tokens']

        # Save checkpoint
        if (i + 1 + start_from) % checkpoint_interval == 0:
            checkpoint_data = {
                'results': results,
                'processed': len(results),
                'total': len(usernames),
                'total_tokens': total_tokens
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

        # Rate limiting (OpenAI has limits)
        time.sleep(0.5)  # Be nice to the API

    print()
    print(f'✓ Completed processing {len(results)} users')
    print(f'Total tokens used: {total_tokens:,}')

    # Calculate cost
    if model == "gpt-4o-mini":
        input_cost = 0.150  # per 1M tokens
        output_cost = 0.600  # per 1M tokens
    elif model == "gpt-4o":
        input_cost = 2.50
        output_cost = 10.00
    else:
        input_cost = 0.150
        output_cost = 0.600

    # Rough estimate (assuming 90% input, 10% output)
    estimated_cost = (total_tokens * 0.9 * input_cost / 1_000_000) + \
                     (total_tokens * 0.1 * output_cost / 1_000_000)
    print(f'Estimated cost: ${estimated_cost:.2f}')
    print()

    # Save final results
    final_path = output_dir / f'user_inference_{model.replace("-", "_")}.json'
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {final_path}')

    # Convert to DataFrame and save
    df = flatten_inference_results(results)
    csv_path = output_dir / f'user_inference_{model.replace("-", "_")}.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved flattened results to {csv_path}')

    return results, df


def flatten_inference_results(results: List[Dict]) -> pd.DataFrame:
    """Flatten nested inference results into a DataFrame."""

    flattened = []

    for result in results:
        if 'error' in result:
            continue

        row = {
            'username': result['username'],
            'tweet_count': result['tweet_count'],
            'sampled_tweets': result['sampled_tweets'],
        }

        # Extract each attribute
        for attr in ['gender', 'political_leaning', 'race_ethnicity',
                     'geographic_origin', 'age_generation', 'profession',
                     'education_level']:
            if attr in result:
                row[f'{attr}_value'] = result[attr]['value']
                row[f'{attr}_confidence'] = result[attr]['confidence']
                row[f'{attr}_reasoning'] = result[attr]['reasoning']

        # Handle lists
        if 'primary_interests' in result:
            row['primary_interests'] = ', '.join(result['primary_interests'])

        if 'writing_style_notes' in result:
            row['writing_style_notes'] = result['writing_style_notes']

        # Add usage
        if '_usage' in result:
            row['tokens_used'] = result['_usage']['total_tokens']

        flattened.append(row)

    return pd.DataFrame(flattened)


if __name__ == '__main__':
    import numpy as np

    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    # Run inference
    results, df = process_all_users(model="gpt-4o-mini", checkpoint_interval=20)

    print()
    print('='*80)
    print('INFERENCE SUMMARY')
    print('='*80)
    print()
    print(f'Successfully processed: {len(df)} users')
    print()

    # Show distributions
    print('Gender distribution:')
    print(df['gender_value'].value_counts())
    print()

    print('Political leaning distribution:')
    print(df['political_leaning_value'].value_counts())
    print()

    print('Race/ethnicity distribution:')
    print(df['race_ethnicity_value'].value_counts())
    print()

    print('Age/generation distribution:')
    print(df['age_generation_value'].value_counts())
    print()
