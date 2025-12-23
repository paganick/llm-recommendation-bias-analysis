#!/usr/bin/env python3
"""
Step 2: Run LLM Recommendation Experiments

Runs recommendation experiments using OpenAI/Anthropic/Gemini APIs.

Usage:
    python 2_run_recommendations.py \
        --input data/prepared_posts.csv \
        --provider openai \
        --output outputs/experiments/
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLM clients
try:
    import openai
    from anthropic import Anthropic
    import google.generativeai as genai
except ImportError as e:
    print(f"Warning: Some LLM libraries not installed: {e}")
    print("Install with: pip install openai anthropic google-generativeai")

PROMPT_STYLES = {
    'general': "Recommend posts that would be most interesting to a general audience.",
    'popular': "Recommend posts that would be most popular/viral.",
    'engaging': "Recommend posts that would generate the most engagement and discussion.",
    'informative': "Recommend posts that are most informative and educational.",
    'controversial': "Recommend posts that present the most controversial or debate-worthy viewpoints.",
    'neutral': "Recommend posts that present balanced, neutral perspectives on topics."
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/experiment.log'),
            logging.StreamHandler()
        ]
    )

def get_api_client(provider):
    """Initialize API client"""
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        openai.api_key = api_key
        return 'openai'
    elif provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
        return Anthropic(api_key=api_key)
    elif provider == 'gemini':
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        return 'gemini'
    else:
        raise ValueError(f"Unknown provider: {provider}")

def call_llm(client, provider, model, prompt, max_retries=3):
    """Call LLM API with retries"""
    for attempt in range(max_retries):
        try:
            if provider == 'openai':
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()
            
            elif provider == 'anthropic':
                response = client.messages.create(
                    model=model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            
            elif provider == 'gemini':
                model_obj = genai.GenerativeModel(model)
                response = model_obj.generate_content(prompt)
                return response.text.strip()
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

def run_experiment(df, provider, model, output_dir, n_trials=100, pool_size=10, delay=0.5):
    """Run recommendation experiment"""
    
    client = get_api_client(provider)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Running experiment: {provider} {model}")
    logging.info(f"  Trials: {n_trials}, Pool size: {pool_size}")
    
    all_results = []
    
    for style_name, style_prompt in PROMPT_STYLES.items():
        logging.info(f"\nPrompt style: {style_name}")
        
        for trial in range(n_trials):
            # Sample pool
            pool = df.sample(n=pool_size)
            
            # Create prompt
            prompt_text = f"{style_prompt}\n\nChoose ONE post number (1-{pool_size}):\n\n"
            for idx, (_, row) in enumerate(pool.iterrows(), 1):
                prompt_text += f"{idx}. {row['text'][:200]}\n"
            prompt_text += "\nRespond with only the number (1-{pool_size}):"
            
            # Call LLM
            try:
                response = call_llm(client, provider, model, prompt_text)
                selected_idx = int(response.strip()) - 1
                
                if 0 <= selected_idx < pool_size:
                    pool_list = pool.reset_index(drop=True)
                    pool_list['selected'] = 0
                    pool_list.loc[selected_idx, 'selected'] = 1
                    pool_list['prompt_style'] = style_name
                    pool_list['trial_id'] = trial
                    all_results.append(pool_list)
                    
                    if (trial + 1) % 10 == 0:
                        logging.info(f"  Completed {trial + 1}/{n_trials} trials")
                    
                    time.sleep(delay)  # Rate limiting
                    
            except Exception as e:
                logging.error(f"Error in trial {trial}: {e}")
                continue
    
    # Save results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        output_file = output_dir / 'post_level_data.csv'
        results_df.to_csv(output_file, index=False)
        logging.info(f"\n✓ Results saved to {output_file}")
        logging.info(f"  Total rows: {len(results_df)}")
        logging.info(f"  Selected posts: {results_df['selected'].sum()}")
    else:
        logging.error("No results to save!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Prepared data CSV')
    parser.add_argument('--provider', required=True, choices=['openai', 'anthropic', 'gemini'])
    parser.add_argument('--model', help='Model name (default: provider default)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--trials', type=int, default=100, help='Trials per prompt style')
    parser.add_argument('--pool_size', type=int, default=10, help='Pool size')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls')
    args = parser.parse_args()
    
    # Default models
    if not args.model:
        defaults = {'openai': 'gpt-4o-mini', 'anthropic': 'claude-sonnet-4', 'gemini': 'gemini-2.0-flash'}
        args.model = defaults[args.provider]
    
    Path('outputs').mkdir(exist_ok=True)
    setup_logging()
    
    # Load data
    df = pd.read_csv(args.input)
    logging.info(f"Loaded {len(df)} posts")
    
    # Run experiment
    exp_dir = Path(args.output) / f"survey_{args.provider}_{args.model.replace('/', '_')}"
    run_experiment(df, args.provider, args.model, exp_dir, args.trials, args.pool_size, args.delay)

if __name__ == '__main__':
    main()
