"""
Extract Author Demographics using LLM inference

Extracts for each author:
- Gender (male, female, non-binary, organization, unknown)
- Political leaning (left, center-left, center, center-right, right, apolitical, unknown)
- Minority status (yes/no)
- Minority group (if applicable)

Uses: persona description + username + 20 message samples
One LLM call per author (~829 total)
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import time

from utils.llm_client import get_llm_client
from data.loaders import load_dataset


DEMOGRAPHICS_PROMPT = """Based on the following information about a social media user, infer their demographics:

**Username:** {username}

**Persona Description:** {persona}

**Sample Posts (20 recent):**
{messages}

Analyze the above and provide:
1. **Gender**: male, female, non-binary, organization, or unknown
2. **Political Leaning**: left, center-left, center, center-right, right, apolitical, or unknown
3. **Minority Status**: Does this person appear to be from a minority group? (yes/no)
4. **Minority Group**: If yes, which group(s)? (e.g., racial, ethnic, LGBTQ+, religious, disability, etc.) If no, write "N/A"

Respond ONLY with valid JSON in this exact format:
{{
  "gender": "...",
  "political_leaning": "...",
  "is_minority": "yes" or "no",
  "minority_groups": "..." or "N/A"
}}"""


def extract_author_demographics(dataset_name: str,
                                model_name: str = "gpt-4o-mini",
                                output_dir: str = "outputs/features"):
    """Extract author demographics for a dataset."""

    print(f"\n{'='*70}")
    print(f"Extracting Author Demographics: {dataset_name.upper()}")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_author_demographics.json"

    # Load existing if available
    if output_path.exists():
        print(f"Loading existing demographics from {output_path.name}")
        with open(output_path) as f:
            demographics = json.load(f)
        print(f"Already have {len(demographics)} authors")
    else:
        demographics = {}

    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    df = load_dataset(dataset_name, sample_size=5000, training_only=True)

    # Reset index completely to avoid conflicts
    df = df.reset_index(drop=True)

    # Get unique authors
    text_col = 'message' if 'message' in df.columns else 'text'
    author_col = 'username' if 'username' in df.columns else 'user_id'

    authors_df = df.groupby(author_col).agg({
        'persona': 'first',
        text_col: list
    }).reset_index()

    authors_df.columns = ['username', 'persona', 'messages']

    print(f"Found {len(authors_df)} unique authors")

    # Filter to authors not yet processed
    authors_to_process = authors_df[~authors_df['username'].isin(demographics.keys())]

    if len(authors_to_process) == 0:
        print("✅ All authors already processed!")
        return demographics

    print(f"Processing {len(authors_to_process)} new authors...")

    # Initialize LLM client
    llm_client = get_llm_client("openai", model_name)

    # Process each author
    errors = []
    for idx, row in tqdm(authors_to_process.iterrows(), total=len(authors_to_process), desc="Processing authors"):
        username = row['username']
        persona = row['persona'] if pd.notna(row['persona']) else "No persona available"
        messages = row['messages']

        # Sample up to 20 messages
        if len(messages) > 20:
            import random
            random.seed(42)
            messages = random.sample(messages, 20)

        # Format messages
        messages_text = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])

        # Create prompt
        prompt = DEMOGRAPHICS_PROMPT.format(
            username=username,
            persona=persona,
            messages=messages_text
        )

        try:
            # Call LLM
            response = llm_client.generate(prompt, max_tokens=200, temperature=0.3)

            # Parse JSON response
            # Remove markdown code blocks if present
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            result = json.loads(response_text)

            # Validate fields
            required_fields = ['gender', 'political_leaning', 'is_minority', 'minority_groups']
            if not all(field in result for field in required_fields):
                raise ValueError(f"Missing required fields in response: {result}")

            # Store result
            demographics[username] = result

            # Save periodically (every 50 authors)
            if len(demographics) % 50 == 0:
                with open(output_path, 'w') as f:
                    json.dump(demographics, f, indent=2)

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"\n  ❌ Error processing {username}: {e}")
            errors.append((username, str(e)))
            # Store placeholder
            demographics[username] = {
                'gender': 'unknown',
                'political_leaning': 'unknown',
                'is_minority': 'no',
                'minority_groups': 'N/A',
                'error': str(e)
            }

    # Final save
    with open(output_path, 'w') as f:
        json.dump(demographics, f, indent=2)

    print(f"\n✅ Saved demographics to {output_path}")
    print(f"   Total authors: {len(demographics)}")
    if errors:
        print(f"   Errors: {len(errors)}")

    # Print summary statistics
    print(f"\n📊 Summary:")

    # Gender distribution
    genders = [d['gender'] for d in demographics.values()]
    print(f"   Gender:")
    for gender in set(genders):
        count = genders.count(gender)
        print(f"     - {gender}: {count} ({100*count/len(genders):.1f}%)")

    # Political leaning
    politics = [d['political_leaning'] for d in demographics.values()]
    print(f"   Political Leaning:")
    for pol in set(politics):
        count = politics.count(pol)
        print(f"     - {pol}: {count} ({100*count/len(politics):.1f}%)")

    # Minority status
    minorities = [d['is_minority'] for d in demographics.values()]
    minority_count = minorities.count('yes')
    print(f"   Minority Status:")
    print(f"     - Yes: {minority_count} ({100*minority_count/len(minorities):.1f}%)")
    print(f"     - No: {len(minorities)-minority_count} ({100*(len(minorities)-minority_count)/len(minorities):.1f}%)")

    return demographics


def main():
    """Extract author demographics for all datasets."""

    print("="*70)
    print("AUTHOR DEMOGRAPHICS EXTRACTION")
    print("="*70)
    print("\nExtracting: gender, political leaning, minority status")
    print("Model: gpt-4o-mini")
    print("Input: persona + username + 20 message samples")

    datasets = ['twitter', 'bluesky', 'reddit']

    for dataset in datasets:
        try:
            demographics = extract_author_demographics(
                dataset,
                model_name="gpt-4o-mini"
            )
        except Exception as e:
            print(f"\n❌ ERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\nNext step: Join demographics to experiment files")
    print("Run: python join_author_demographics.py")


if __name__ == "__main__":
    main()
