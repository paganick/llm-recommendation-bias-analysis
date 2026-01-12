"""
Batch Experiment Runner

Orchestrates multiple bias testing experiments with different configurations:
- Multiple LLM models (GPT-4, Claude, etc.)
- Different prompt styles
- Multiple random seeds for statistical robustness
- Automated result aggregation and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import time

from data.loaders import load_dataset
from utils.llm_client import get_llm_client
from recommender.llm_recommender import OneShotRecommender
from inference.metadata_inference import infer_tweet_metadata
from analysis.bias_analysis import BiasAnalyzer
from analysis.visualization import create_bias_report


class ExperimentConfig:
    """Configuration for a single experiment."""

    def __init__(self,
                 name: str,
                 llm_provider: str,
                 llm_model: str,
                 prompt_style: str,
                 pool_size: int = 50,
                 k: int = 10,
                 random_seed: Optional[int] = None):
        """
        Initialize experiment configuration.

        Args:
            name: Experiment name
            llm_provider: 'anthropic' or 'openai'
            llm_model: Model name (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4')
            prompt_style: 'popular', 'engaging', 'informative', etc.
            pool_size: Number of tweets in pool
            k: Number of recommendations
            random_seed: Random seed for reproducibility
        """
        self.name = name
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.prompt_style = prompt_style
        self.pool_size = pool_size
        self.k = k
        self.random_seed = random_seed or int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'prompt_style': self.prompt_style,
            'pool_size': self.pool_size,
            'k': self.k,
            'random_seed': self.random_seed
        }


class ExperimentRunner:
    """
    Run batch experiments for bias analysis.

    Workflow:
    1. Load dataset
    2. Infer metadata (if not already done)
    3. For each experiment config:
        a. Sample tweet pool
        b. Get LLM recommendations
        c. Analyze bias
        d. Save results
    4. Aggregate results across experiments
    5. Generate visualizations
    """

    def __init__(self,
                 dataset_name: str = 'twitteraae',
                 dataset_version: str = 'all_aa',
                 dataset_sample_size: int = 10000,
                 output_dir: str = './experiments/results',
                 metadata_cache_path: Optional[str] = None):
        """
        Initialize experiment runner.

        Args:
            dataset_name: 'twitteraae' or 'dadit'
            dataset_version: Dataset version
            dataset_sample_size: Number of tweets to load
            output_dir: Directory to save results
            metadata_cache_path: Path to cached metadata (to avoid recomputation)
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.dataset_sample_size = dataset_sample_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_cache_path = metadata_cache_path

        self.tweets_df = None
        self.experiments_results = []

    def load_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load and prepare dataset.

        Returns:
            DataFrame with tweets and metadata
        """
        if verbose:
            print("="*60)
            print("LOADING DATASET")
            print("="*60)

        # Load dataset
        self.tweets_df = load_dataset(
            self.dataset_name,
            version=self.dataset_version,
            sample_size=self.dataset_sample_size
        )

        if verbose:
            print(f"Loaded {len(self.tweets_df)} tweets")

        # Check if metadata already exists or can be loaded from cache
        has_metadata = 'sentiment_polarity' in self.tweets_df.columns

        if not has_metadata:
            if self.metadata_cache_path and Path(self.metadata_cache_path).exists():
                if verbose:
                    print(f"\nLoading metadata from cache: {self.metadata_cache_path}")
                cached_df = pd.read_csv(self.metadata_cache_path)
                # Merge metadata
                self.tweets_df = pd.merge(
                    self.tweets_df,
                    cached_df,
                    on='tweet_id',
                    how='left',
                    suffixes=('', '_meta')
                )
            else:
                if verbose:
                    print("\nInferring metadata (this may take a while)...")
                self.tweets_df = infer_tweet_metadata(
                    self.tweets_df,
                    text_column='text',
                    sentiment_method='vader'
                )

                # Cache metadata if path provided
                if self.metadata_cache_path:
                    self.tweets_df.to_csv(self.metadata_cache_path, index=False)
                    if verbose:
                        print(f"Saved metadata cache to {self.metadata_cache_path}")

        if verbose:
            print(f"\nDataset ready with {len(self.tweets_df)} tweets and {len(self.tweets_df.columns)} columns")

        return self.tweets_df

    def run_experiment(self,
                      config: ExperimentConfig,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration
            verbose: Print progress

        Returns:
            Dict with experiment results
        """
        if verbose:
            print("\n" + "="*60)
            print(f"RUNNING EXPERIMENT: {config.name}")
            print("="*60)
            print(f"Model: {config.llm_provider}/{config.llm_model}")
            print(f"Prompt style: {config.prompt_style}")
            print(f"Pool size: {config.pool_size}, k: {config.k}")

        start_time = time.time()

        # Initialize LLM client
        try:
            llm = get_llm_client(
                provider=config.llm_provider,
                model=config.llm_model,
                temperature=0.7
            )
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            return {'error': str(e), 'config': config.to_dict()}

        # Sample tweet pool
        np.random.seed(config.random_seed)
        pool_df = self.tweets_df.sample(n=config.pool_size, random_state=config.random_seed)

        # Get recommendations
        recommender = OneShotRecommender(llm, k=config.k)

        try:
            recommended_df = recommender.recommend(
                pool_df,
                prompt_style=config.prompt_style,
                max_pool_size=config.pool_size
            )
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return {'error': str(e), 'config': config.to_dict()}

        # Analyze bias
        analyzer = BiasAnalyzer(alpha=0.05)
        bias_results = analyzer.comprehensive_bias_analysis(
            recommended_df,
            pool_df,
            demographic_cols=['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']
        )

        # Get LLM usage stats
        llm_stats = llm.get_stats()

        elapsed_time = time.time() - start_time

        # Compile results
        results = {
            'config': config.to_dict(),
            'bias_analysis': bias_results,
            'llm_stats': llm_stats,
            'elapsed_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }

        if verbose:
            print(f"\nCompleted in {elapsed_time:.1f}s")
            print(f"LLM calls: {llm_stats['call_count']}, Tokens: {llm_stats['total_tokens']}")

            summary = bias_results.get('summary', {})
            print(f"Significant biases detected: {summary.get('total_significant_biases', 0)}")

        return results

    def run_batch(self,
                 configs: List[ExperimentConfig],
                 save_individual: bool = True,
                 verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run batch of experiments.

        Args:
            configs: List of experiment configurations
            save_individual: Save individual results
            verbose: Print progress

        Returns:
            List of experiment results
        """
        if self.tweets_df is None:
            self.load_data(verbose=verbose)

        results = []

        for i, config in enumerate(configs, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"EXPERIMENT {i}/{len(configs)}")
                print(f"{'='*60}")

            result = self.run_experiment(config, verbose=verbose)
            results.append(result)

            # Save individual result
            if save_individual and 'error' not in result:
                self._save_experiment_result(result, config.name)

        self.experiments_results = results

        # Save batch summary
        self._save_batch_summary(results)

        if verbose:
            print("\n" + "="*60)
            print("BATCH COMPLETE")
            print("="*60)
            print(f"Total experiments: {len(results)}")
            print(f"Successful: {sum(1 for r in results if 'error' not in r)}")
            print(f"Failed: {sum(1 for r in results if 'error' in r)}")

        return results

    def generate_visualizations(self, verbose: bool = True) -> None:
        """Generate visualizations for all experiments."""
        if not self.experiments_results:
            print("No results to visualize")
            return

        if verbose:
            print("\n" + "="*60)
            print("GENERATING VISUALIZATIONS")
            print("="*60)

        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        for result in self.experiments_results:
            if 'error' in result:
                continue

            config = result['config']
            bias_analysis = result['bias_analysis']

            # Create bias report
            create_bias_report(
                bias_analysis,
                output_dir=viz_dir / config['name'],
                system_name=config['name']
            )

        if verbose:
            print(f"Visualizations saved to {viz_dir}")

    def _save_experiment_result(self, result: Dict[str, Any], name: str) -> None:
        """Save individual experiment result."""
        exp_dir = self.output_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save full results
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Save bias analysis separately
        if 'bias_analysis' in result:
            with open(exp_dir / 'bias_analysis.json', 'w') as f:
                json.dump(result['bias_analysis'], f, indent=2, default=str)

    def _save_batch_summary(self, results: List[Dict[str, Any]]) -> None:
        """Save summary of batch results."""
        summary = {
            'total_experiments': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'timestamp': datetime.now().isoformat(),
            'experiments': [
                {
                    'name': r['config']['name'],
                    'status': 'failed' if 'error' in r else 'success',
                    'error': r.get('error'),
                    'significant_biases': r.get('bias_analysis', {}).get('summary', {}).get('total_significant_biases', 0)
                }
                for r in results
            ]
        }

        with open(self.output_dir / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def create_default_configs() -> List[ExperimentConfig]:
    """
    Create default experiment configurations for testing multiple systems.

    Returns:
        List of experiment configs
    """
    configs = []

    # Claude configurations
    configs.append(ExperimentConfig(
        name='claude_sonnet_popular',
        llm_provider='anthropic',
        llm_model='claude-3-5-sonnet-20241022',
        prompt_style='popular',
        pool_size=50,
        k=10
    ))

    configs.append(ExperimentConfig(
        name='claude_sonnet_engaging',
        llm_provider='anthropic',
        llm_model='claude-3-5-sonnet-20241022',
        prompt_style='engaging',
        pool_size=50,
        k=10
    ))

    configs.append(ExperimentConfig(
        name='claude_sonnet_informative',
        llm_provider='anthropic',
        llm_model='claude-3-5-sonnet-20241022',
        prompt_style='informative',
        pool_size=50,
        k=10
    ))

    # OpenAI configurations (commented out - require OpenAI API key)
    # configs.append(ExperimentConfig(
    #     name='gpt4_popular',
    #     llm_provider='openai',
    #     llm_model='gpt-4',
    #     prompt_style='popular',
    #     pool_size=50,
    #     k=10
    # ))

    return configs


# Convenience function
def run_quick_experiment(llm_provider: str = 'anthropic',
                        llm_model: str = 'claude-3-5-sonnet-20241022',
                        prompt_style: str = 'popular',
                        pool_size: int = 50,
                        k: int = 10,
                        output_dir: str = './experiments/quick_test') -> Dict[str, Any]:
    """
    Quick experiment runner for testing.

    Args:
        llm_provider: 'anthropic' or 'openai'
        llm_model: Model name
        prompt_style: Recommendation prompt style
        pool_size: Tweet pool size
        k: Number of recommendations
        output_dir: Output directory

    Returns:
        Experiment results
    """
    config = ExperimentConfig(
        name='quick_test',
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_style=prompt_style,
        pool_size=pool_size,
        k=k
    )

    runner = ExperimentRunner(
        dataset_sample_size=5000,
        output_dir=output_dir
    )

    runner.load_data()
    result = runner.run_experiment(config)
    runner.generate_visualizations()

    return result
