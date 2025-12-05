"""
Extract demographic and persona attributes from LLM-generated persona descriptions.

Extracts:
- Gender (male, female, unknown)
- Age group (young, middle-aged, senior, unknown)
- Race/ethnicity (if mentioned)
- Political leaning (conservative, liberal, center, unknown)
- Topics of interest
- Writing style
"""

import re
from typing import Dict, Any, List
import pandas as pd


def extract_gender(persona_text: str) -> str:
    """Extract gender from persona description."""
    text_lower = persona_text.lower()

    # Explicit gender mentions
    if any(word in text_lower for word in ['she ', 'her ', 'herself', 'woman', 'female', 'mother', 'wife', 'daughter', 'sister']):
        return 'female'
    elif any(word in text_lower for word in ['he ', 'him ', 'himself', 'man', 'male', 'father', 'husband', 'son', 'brother']):
        return 'male'
    elif any(word in text_lower for word in ['they/them', 'non-binary', 'nonbinary', 'genderqueer']):
        return 'non-binary'
    else:
        return 'unknown'


def extract_age_group(persona_text: str) -> str:
    """Extract age group from persona description."""
    text_lower = persona_text.lower()

    # Direct age mentions
    age_match = re.search(r'\b(\d{1,2})[-\s]year[-\s]old\b', text_lower)
    if age_match:
        age = int(age_match.group(1))
        if age < 30:
            return 'young'
        elif age < 50:
            return 'middle-aged'
        else:
            return 'senior'

    # Indirect indicators
    if any(word in text_lower for word in ['young', 'gen z', 'millennial', 'college student', 'twenties']):
        return 'young'
    elif any(word in text_lower for word in ['middle-aged', 'gen x', 'forties', 'fifties']):
        return 'middle-aged'
    elif any(word in text_lower for word in ['senior', 'retiree', 'retired', 'elderly', 'boomer', 'baby boomer', 'sixties', 'seventies']):
        return 'senior'
    elif any(word in text_lower for word in ['parent', 'children', 'kids', 'family']):
        # Parents are typically middle-aged or older
        return 'middle-aged'
    else:
        return 'unknown'


def extract_race_ethnicity(persona_text: str) -> str:
    """Extract race/ethnicity if explicitly mentioned."""
    text_lower = persona_text.lower()

    # Explicit mentions
    if any(word in text_lower for word in ['african american', 'black american', 'black']):
        return 'black'
    elif any(word in text_lower for word in ['hispanic', 'latino', 'latina', 'latinx', 'mexican', 'puerto rican']):
        return 'hispanic'
    elif any(word in text_lower for word in ['asian', 'asian american', 'chinese', 'indian', 'korean', 'japanese']):
        return 'asian'
    elif any(word in text_lower for word in ['white', 'caucasian', 'european american']):
        return 'white'
    elif any(word in text_lower for word in ['native american', 'indigenous']):
        return 'native_american'
    else:
        return 'unknown'


def extract_political_leaning(persona_text: str) -> Dict[str, Any]:
    """Extract political leaning from persona description."""
    text_lower = persona_text.lower()

    # Conservative keywords
    conservative_keywords = [
        'conservative', 'maga', 'trump supporter', 'republican', 'right-wing',
        'right wing', 'pro-trump', 'gop', 'pro-life'
    ]

    # Liberal keywords
    liberal_keywords = [
        'liberal', 'progressive', 'democrat', 'democratic', 'left-wing',
        'left wing', 'pro-biden', 'voteblue', 'pro-choice', 'blm'
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
        political_keywords = ['politic', 'election', 'government', 'policy', 'vote']
        is_political = any(kw in text_lower for kw in political_keywords)

        return {
            'political_leaning': 'unknown',
            'is_political': is_political,
            'confidence': 'none'
        }


def extract_education_level(persona_text: str) -> str:
    """Extract education level if mentioned."""
    text_lower = persona_text.lower()

    if any(word in text_lower for word in ['phd', 'doctorate', 'professor', 'researcher']):
        return 'doctorate'
    elif any(word in text_lower for word in ['master', 'mba', 'graduate degree', 'grad school']):
        return 'masters'
    elif any(word in text_lower for word in ['college', 'university', 'bachelor', 'undergraduate', 'degree']):
        return 'bachelors'
    elif any(word in text_lower for word in ['high school', 'diploma']):
        return 'high_school'
    else:
        return 'unknown'


def extract_profession(persona_text: str) -> str:
    """Extract profession/occupation if mentioned."""
    text_lower = persona_text.lower()

    profession_keywords = {
        'healthcare': ['doctor', 'nurse', 'physician', 'surgeon', 'medical', 'healthcare'],
        'education': ['teacher', 'professor', 'educator', 'instructor'],
        'technology': ['software engineer', 'developer', 'programmer', 'data scientist', 'tech'],
        'business': ['entrepreneur', 'ceo', 'manager', 'business owner', 'executive'],
        'law': ['lawyer', 'attorney', 'legal'],
        'journalism': ['journalist', 'reporter', 'writer', 'author'],
        'military': ['veteran', 'military', 'soldier', 'navy', 'army'],
        'creative': ['artist', 'musician', 'designer', 'photographer'],
        'service': ['barista', 'waiter', 'retail', 'customer service'],
        'retired': ['retired', 'retiree']
    }

    for profession, keywords in profession_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return profession

    return 'unknown'


def extract_all_persona_attributes(persona_text: str) -> Dict[str, Any]:
    """Extract all persona attributes from description."""

    political_info = extract_political_leaning(persona_text)

    return {
        'gender': extract_gender(persona_text),
        'age_group': extract_age_group(persona_text),
        'race_ethnicity': extract_race_ethnicity(persona_text),
        'political_leaning': political_info['political_leaning'],
        'is_political': political_info['is_political'],
        'political_confidence': political_info['confidence'],
        'education_level': extract_education_level(persona_text),
        'profession': extract_profession(persona_text)
    }


def add_persona_attributes_to_dataframe(df: pd.DataFrame, persona_col: str = 'persona') -> pd.DataFrame:
    """Add persona attributes as columns to DataFrame."""

    print(f"Extracting persona attributes from {len(df)} rows...")

    # Extract all attributes
    persona_attrs = df[persona_col].apply(extract_all_persona_attributes)

    # Convert to DataFrame and merge
    attrs_df = pd.DataFrame(persona_attrs.tolist())
    result_df = pd.concat([df, attrs_df], axis=1)

    # Print summary
    print("\nExtracted attribute distributions:")
    for col in ['gender', 'age_group', 'race_ethnicity', 'political_leaning', 'education_level', 'profession']:
        if col in result_df.columns:
            dist = result_df[col].value_counts()
            print(f"\n{col}:")
            print(dist)

    return result_df


if __name__ == '__main__':
    # Test with sample persona
    sample_persona = """You are @example_user, a 35-year-old progressive woman working as a software engineer
    in Silicon Valley. You're passionate about social justice and frequently discuss issues related to
    technology, diversity, and women in tech."""

    attrs = extract_all_persona_attributes(sample_persona)
    print("Sample extraction:")
    for key, value in attrs.items():
        print(f"  {key}: {value}")
