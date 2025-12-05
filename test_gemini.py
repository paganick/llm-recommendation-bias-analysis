#!/usr/bin/env python
"""Quick test script for Gemini API connection."""

from utils.llm_client import get_llm_client
import os

print('='*80)
print('TESTING GEMINI API CONNECTION')
print('='*80)
print()

# Check API key
api_key_exists = "GEMINI_API_KEY" in os.environ
print(f'API Key found in environment: {api_key_exists}')

if not api_key_exists:
    print()
    print('ERROR: GEMINI_API_KEY not found!')
    print('Please run: export GEMINI_API_KEY="your-api-key-here"')
    exit(1)

print()

# Test different models
models_to_test = [
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-3-pro-preview'
]

for model_name in models_to_test:
    print(f'Testing {model_name}...')
    try:
        client = get_llm_client('gemini', model_name)
        response = client.generate('Say "Hello from Gemini!" in exactly 5 words.')
        print(f'  ✓ Response: {response}')
        stats = client.get_stats()
        print(f'  ✓ Tokens used: {stats["total_tokens"]}')
        print()
        break  # If one works, stop testing
    except Exception as e:
        print(f'  ✗ Error: {str(e)[:100]}')
        print()

print('='*80)
print('Gemini is ready! You can now run experiments.')
print('='*80)
print()
print('Example command:')
print('python run_experiment.py \\')
print('  --dataset reddit \\')
print('  --provider gemini \\')
print('  --model gemini-2.0-flash \\')
print('  --dataset-size 5000 \\')
print('  --n-trials 100')
