#!/usr/bin/env python
"""Quick test script for Anthropic Claude API connection."""

from utils.llm_client import get_llm_client
import os

print('='*80)
print('TESTING ANTHROPIC CLAUDE API CONNECTION')
print('='*80)
print()

# Check API key
api_key_exists = "ANTHROPIC_API_KEY" in os.environ
print(f'API Key found in environment: {api_key_exists}')

if not api_key_exists:
    print()
    print('ERROR: ANTHROPIC_API_KEY not found!')
    print('Please run: export ANTHROPIC_API_KEY="your-api-key-here"')
    exit(1)

print()

# Test different models
models_to_test = [
    'claude-3-5-sonnet-20241022',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229'
]

for model_name in models_to_test:
    print(f'Testing {model_name}...')
    try:
        client = get_llm_client('anthropic', model_name)
        response = client.generate('Say "Hello from Claude!" in exactly 5 words.')
        print(f'  ✓ Response: {response}')
        stats = client.get_stats()
        print(f'  ✓ Tokens used: {stats["total_tokens"]}')
        print()
        break  # If one works, stop testing
    except Exception as e:
        print(f'  ✗ Error: {str(e)[:100]}')
        print()

print('='*80)
print('Claude is ready! You can now run experiments.')
print('='*80)
print()
print('Example commands:')
print()
print('# Fast and balanced (recommended):')
print('python run_experiment.py \\')
print('  --dataset reddit \\')
print('  --provider anthropic \\')
print('  --model claude-3-5-sonnet-20241022 \\')
print('  --dataset-size 5000 \\')
print('  --n-trials 100')
print()
print('# Fastest and cheapest:')
print('python run_experiment.py \\')
print('  --dataset twitter \\')
print('  --provider anthropic \\')
print('  --model claude-3-haiku-20240307 \\')
print('  --dataset-size 5000 \\')
print('  --n-trials 100')
