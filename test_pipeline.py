"""
Test script to verify the pipeline works with all datasets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_dataset
from inference.metadata_inference import infer_tweet_metadata


def test_dataset_loading():
    """Test loading all three datasets."""
    print('='*80)
    print('TESTING DATASET LOADING')
    print('='*80)
    print()

    datasets = ['twitter', 'reddit', 'bluesky']

    for dataset_name in datasets:
        print(f'\nTesting {dataset_name}...')
        try:
            df = load_dataset(dataset_name, sample_size=100)
            print(f'  ✓ Loaded {len(df)} posts')
            print(f'  Columns: {list(df.columns)}')

            # Check text column
            text_col = 'message' if 'message' in df.columns else 'text'
            print(f'  Text column: {text_col}')
            print(f'  Sample text: {df[text_col].iloc[0][:100]}...')

        except Exception as e:
            print(f'  ✗ Error: {e}')

    print()


def test_metadata_inference():
    """Test metadata inference."""
    print('='*80)
    print('TESTING METADATA INFERENCE')
    print('='*80)
    print()

    # Load small sample
    df = load_dataset('twitter', sample_size=10)
    text_col = 'message' if 'message' in df.columns else 'text'

    print('Inferring metadata for 10 posts...')

    try:
        df_with_metadata = infer_tweet_metadata(
            df,
            text_column=text_col,
            sentiment_method='vader',
            topic_method='keyword',
            include_gender=False,
            include_political=False
        )

        print(f'  ✓ Metadata inference successful')
        print(f'  New columns: {[col for col in df_with_metadata.columns if col not in df.columns]}')

        # Show sample results
        print('\n  Sample metadata:')
        sample = df_with_metadata.iloc[0]
        print(f'    Sentiment: {sample["sentiment_label"]} (polarity: {sample["sentiment_polarity"]:.2f})')
        print(f'    Topic: {sample["primary_topic"]}')
        print(f'    Formality: {sample["formality_score"]:.2f}')
        print(f'    Polarization: {sample["polarization_score"]:.2f}')

    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()

    print()


def test_llm_client():
    """Test LLM client initialization."""
    print('='*80)
    print('TESTING LLM CLIENT')
    print('='*80)
    print()

    from utils.llm_client import get_llm_client
    import os

    # Test OpenAI (if key available)
    if os.environ.get('OPENAI_API_KEY'):
        print('Testing OpenAI client...')
        try:
            client = get_llm_client('openai', 'gpt-4o-mini')
            print(f'  ✓ OpenAI client initialized')
        except Exception as e:
            print(f'  ✗ Error: {e}')
    else:
        print('  ⊘ OPENAI_API_KEY not set, skipping OpenAI test')

    # Test Anthropic (if key available)
    if os.environ.get('ANTHROPIC_API_KEY'):
        print('\nTesting Anthropic client...')
        try:
            client = get_llm_client('anthropic', 'claude-3-5-sonnet-20241022')
            print(f'  ✓ Anthropic client initialized')
        except Exception as e:
            print(f'  ✗ Error: {e}')
    else:
        print('  ⊘ ANTHROPIC_API_KEY not set, skipping Anthropic test')

    # Test HuggingFace (will try to load model)
    print('\nTesting HuggingFace client...')
    print('  (Skipping actual model load to save time)')
    print('  To test: get_llm_client("huggingface", "meta-llama/Llama-3.1-8B-Instruct")')

    print()


def main():
    """Run all tests."""
    print()
    print('='*80)
    print('PIPELINE TEST SUITE')
    print('='*80)
    print()

    try:
        test_dataset_loading()
        test_metadata_inference()
        test_llm_client()

        print('='*80)
        print('ALL TESTS COMPLETED')
        print('='*80)
        print()

    except Exception as e:
        print(f'\nTest suite failed with error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
