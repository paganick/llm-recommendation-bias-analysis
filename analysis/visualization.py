"""
Visualization Tools for Bias Analysis

Create plots and charts to visualize bias in LLM recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class BiasVisualizer:
    """Create visualizations for bias analysis results."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_demographic_bias(self,
                              bias_results: Dict[str, Any],
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot demographic bias analysis results.

        Args:
            bias_results: Results from BiasAnalyzer.analyze_demographic_bias()
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        demographic_cols = list(bias_results['bias_scores'].keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Bias scores (percentage point difference)
        ax1 = axes[0]
        bias_scores = [bias_results['bias_scores'][col] for col in demographic_cols]
        colors = ['red' if score < 0 else 'green' for score in bias_scores]

        ax1.barh(demographic_cols, bias_scores, color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Bias Score (percentage points)', fontsize=11)
        ax1.set_title('Demographic Bias in Recommendations', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add significance markers
        for i, col in enumerate(demographic_cols):
            if bias_results['statistical_tests'][col]['significant']:
                p_value = bias_results['statistical_tests'][col]['p_value']
                marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
                ax1.text(bias_scores[i], i, f' {marker}', va='center', fontsize=10)

        # Plot 2: Mean comparisons
        ax2 = axes[1]
        x = np.arange(len(demographic_cols))
        width = 0.35

        pool_means = [bias_results['pool_stats'][col]['mean'] for col in demographic_cols]
        rec_means = [bias_results['recommended_stats'][col]['mean'] for col in demographic_cols]

        ax2.bar(x - width/2, pool_means, width, label='Pool (Baseline)', alpha=0.7)
        ax2.bar(x + width/2, rec_means, width, label='Recommended', alpha=0.7)

        ax2.set_ylabel('Mean Probability', fontsize=11)
        ax2.set_title('Demographic Distribution Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([col.replace('demo_', '') for col in demographic_cols])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved demographic bias plot to {save_path}")

        return fig

    def plot_sentiment_bias(self,
                           bias_results: Dict[str, Any],
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot sentiment bias analysis results.

        Args:
            bias_results: Results from BiasAnalyzer.analyze_sentiment_bias()
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        has_distribution = 'sentiment_distribution' in bias_results and bias_results['sentiment_distribution']

        if has_distribution:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]

        # Plot 1: Mean sentiment comparison
        ax1 = axes[0]
        means = [bias_results['pool_mean_sentiment'], bias_results['recommended_mean_sentiment']]
        colors = ['steelblue', 'coral']

        bars = ax1.bar(['Pool (Baseline)', 'Recommended'], means, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_ylabel('Mean Sentiment Polarity', fontsize=11)
        ax1.set_title('Sentiment Bias in Recommendations', fontsize=13, fontweight='bold')
        ax1.set_ylim(-1, 1)

        # Add significance marker
        if bias_results.get('significant'):
            ax1.text(0.5, max(means) + 0.1, '***' if bias_results['p_value'] < 0.001 else
                    '**' if bias_results['p_value'] < 0.01 else '*',
                    ha='center', fontsize=14)

        # Add bias score annotation
        bias_score = bias_results['bias_score']
        ax1.text(0.5, min(means) - 0.2,
                f'Bias: {bias_score:+.3f} ({bias_results["favors"]})',
                ha='center', fontsize=10, style='italic')

        # Plot 2: Sentiment distribution (if available)
        if has_distribution:
            ax2 = axes[1]
            sent_dist = bias_results['sentiment_distribution']

            pool_dist = sent_dist['pool_distribution']
            rec_dist = sent_dist['recommended_distribution']

            categories = list(set(pool_dist.keys()) | set(rec_dist.keys()))
            x = np.arange(len(categories))
            width = 0.35

            pool_props = [pool_dist.get(cat, 0) for cat in categories]
            rec_props = [rec_dist.get(cat, 0) for cat in categories]

            ax2.bar(x - width/2, pool_props, width, label='Pool', alpha=0.7)
            ax2.bar(x + width/2, rec_props, width, label='Recommended', alpha=0.7)

            ax2.set_ylabel('Proportion', fontsize=11)
            ax2.set_title('Sentiment Category Distribution', fontsize=13, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved sentiment bias plot to {save_path}")

        return fig

    def plot_topic_bias(self,
                       bias_results: Dict[str, Any],
                       top_n: int = 10,
                       save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot topic bias analysis results.

        Args:
            bias_results: Results from BiasAnalyzer.analyze_topic_bias()
            top_n: Number of top topics to show
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        topic_bias_scores = bias_results['topic_bias_scores']

        # Sort by absolute bias score
        sorted_topics = sorted(topic_bias_scores.items(),
                              key=lambda x: abs(x[1]),
                              reverse=True)[:top_n]

        topics = [t[0] for t in sorted_topics]
        scores = [t[1] for t in sorted_topics]

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ['red' if s < 0 else 'green' for s in scores]
        ax.barh(topics, scores, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Bias Score (percentage points)', fontsize=11)
        ax.set_title(f'Topic Bias in Recommendations (Top {top_n})',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add significance marker
        if bias_results.get('significant'):
            ax.text(0.02, 0.98, f"χ² test: p={bias_results['p_value']:.4f} ***",
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved topic bias plot to {save_path}")

        return fig

    def plot_style_bias(self,
                       bias_results: Dict[str, Any],
                       save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot style bias analysis results.

        Args:
            bias_results: Results from BiasAnalyzer.analyze_style_bias()
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        # Extract features to plot
        features = []
        bias_scores = []
        p_values = []

        for key, value in bias_results.items():
            if isinstance(value, dict) and 'bias_score' in value:
                feature_name = key.replace('_bias', '').replace('_', ' ').title()
                features.append(feature_name)
                bias_scores.append(value['bias_score'])
                p_values.append(value.get('p_value', 1.0))

        if not features:
            print("No style features to plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['red' if s < 0 else 'green' for s in bias_scores]
        bars = ax.barh(features, bias_scores, color=colors, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Bias Score', fontsize=11)
        ax.set_title('Writing Style Bias in Recommendations', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add significance markers
        for i, (feature, p_val) in enumerate(zip(features, p_values)):
            if p_val < 0.05:
                marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                ax.text(bias_scores[i], i, f' {marker}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved style bias plot to {save_path}")

        return fig

    def plot_comprehensive_dashboard(self,
                                    comprehensive_results: Dict[str, Any],
                                    save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive dashboard with all bias analyses.

        Args:
            comprehensive_results: Results from BiasAnalyzer.comprehensive_bias_analysis()
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Demographic bias
        if 'demographic_bias' in comprehensive_results:
            ax1 = fig.add_subplot(gs[0, 0])
            demo_results = comprehensive_results['demographic_bias']
            demographic_cols = list(demo_results['bias_scores'].keys())
            bias_scores = [demo_results['bias_scores'][col] for col in demographic_cols]
            colors = ['red' if s < 0 else 'green' for s in bias_scores]

            ax1.barh([col.replace('demo_', '') for col in demographic_cols],
                    bias_scores, color=colors, alpha=0.7)
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax1.set_xlabel('Bias Score (%)', fontsize=10)
            ax1.set_title('Demographic Bias', fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 2. Sentiment bias
        if 'sentiment_bias' in comprehensive_results:
            ax2 = fig.add_subplot(gs[0, 1])
            sent_results = comprehensive_results['sentiment_bias']

            means = [sent_results['pool_mean_sentiment'],
                    sent_results['recommended_mean_sentiment']]
            ax2.bar(['Pool', 'Recommended'], means,
                   color=['steelblue', 'coral'], alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_ylabel('Mean Sentiment', fontsize=10)
            ax2.set_title('Sentiment Bias', fontsize=11, fontweight='bold')
            ax2.set_ylim(-1, 1)

        # 3. Topic bias
        if 'topic_bias' in comprehensive_results:
            ax3 = fig.add_subplot(gs[1, :])
            topic_results = comprehensive_results['topic_bias']
            topic_bias_scores = topic_results['topic_bias_scores']

            sorted_topics = sorted(topic_bias_scores.items(),
                                  key=lambda x: abs(x[1]),
                                  reverse=True)[:8]

            topics = [t[0] for t in sorted_topics]
            scores = [t[1] for t in sorted_topics]
            colors = ['red' if s < 0 else 'green' for s in scores]

            ax3.barh(topics, scores, color=colors, alpha=0.7)
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax3.set_xlabel('Bias Score (%)', fontsize=10)
            ax3.set_title('Topic Bias (Top 8)', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. Style bias
        if 'style_bias' in comprehensive_results:
            ax4 = fig.add_subplot(gs[2, 0])
            style_results = comprehensive_results['style_bias']

            features = []
            scores = []
            for key, value in style_results.items():
                if isinstance(value, dict) and 'bias_score' in value:
                    features.append(key.replace('_bias', '').replace('_', '\n'))
                    scores.append(value['bias_score'])

            if features:
                colors = ['red' if s < 0 else 'green' for s in scores]
                ax4.bar(features, scores, color=colors, alpha=0.7)
                ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax4.set_ylabel('Bias Score', fontsize=10)
                ax4.set_title('Style Bias', fontsize=11, fontweight='bold')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)

        # 5. Polarization bias
        if 'polarization_bias' in comprehensive_results:
            ax5 = fig.add_subplot(gs[2, 1])
            pol_results = comprehensive_results['polarization_bias']

            means = [pol_results['pool_mean_polarization'],
                    pol_results['recommended_mean_polarization']]
            ax5.bar(['Pool', 'Recommended'], means,
                   color=['steelblue', 'coral'], alpha=0.7)
            ax5.set_ylabel('Mean Polarization', fontsize=10)
            ax5.set_title('Polarization Bias', fontsize=11, fontweight='bold')

        # Overall title
        total_biases = comprehensive_results.get('summary', {}).get('total_significant_biases', 0)
        fig.suptitle(f'Comprehensive Bias Analysis Dashboard\n({total_biases} significant biases detected)',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved comprehensive dashboard to {save_path}")

        return fig

    def plot_bias_comparison(self,
                            comparison_df: pd.DataFrame,
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot comparison of bias across multiple systems.

        Args:
            comparison_df: DataFrame from compare_recommender_bias()
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Demographic bias comparison
        ax1 = axes[0]
        demo_cols = [col for col in comparison_df.columns if 'demo_' in col]

        if demo_cols:
            systems = comparison_df['system'].tolist()
            x = np.arange(len(systems))
            width = 0.8 / len(demo_cols)

            for i, col in enumerate(demo_cols):
                values = comparison_df[col].tolist()
                ax1.bar(x + i * width, values, width,
                       label=col.replace('demo_', '').replace('_bias', ''))

            ax1.set_ylabel('Bias Score (%)', fontsize=11)
            ax1.set_title('Demographic Bias Comparison', fontsize=13, fontweight='bold')
            ax1.set_xticks(x + width * (len(demo_cols) - 1) / 2)
            ax1.set_xticklabels(systems)
            ax1.legend()
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Total significant biases
        ax2 = axes[1]
        if 'total_significant_biases' in comparison_df.columns:
            systems = comparison_df['system'].tolist()
            counts = comparison_df['total_significant_biases'].tolist()

            bars = ax2.bar(systems, counts, color='coral', alpha=0.7)
            ax2.set_ylabel('Number of Significant Biases', fontsize=11)
            ax2.set_title('Total Significant Biases by System', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved bias comparison plot to {save_path}")

        return fig


def create_bias_report(comprehensive_results: Dict[str, Any],
                      output_dir: Path,
                      system_name: str = "LLM Recommender") -> None:
    """
    Create comprehensive bias report with all visualizations.

    Args:
        comprehensive_results: Results from BiasAnalyzer.comprehensive_bias_analysis()
        output_dir: Directory to save visualizations
        system_name: Name of the system being analyzed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = BiasVisualizer()

    # Create dashboard
    visualizer.plot_comprehensive_dashboard(
        comprehensive_results,
        save_path=output_dir / f'{system_name}_bias_dashboard.png'
    )

    # Individual plots
    if 'demographic_bias' in comprehensive_results:
        visualizer.plot_demographic_bias(
            comprehensive_results['demographic_bias'],
            save_path=output_dir / f'{system_name}_demographic_bias.png'
        )

    if 'sentiment_bias' in comprehensive_results:
        visualizer.plot_sentiment_bias(
            comprehensive_results['sentiment_bias'],
            save_path=output_dir / f'{system_name}_sentiment_bias.png'
        )

    if 'topic_bias' in comprehensive_results:
        visualizer.plot_topic_bias(
            comprehensive_results['topic_bias'],
            save_path=output_dir / f'{system_name}_topic_bias.png'
        )

    if 'style_bias' in comprehensive_results:
        visualizer.plot_style_bias(
            comprehensive_results['style_bias'],
            save_path=output_dir / f'{system_name}_style_bias.png'
        )

    plt.close('all')
    print(f"\nBias report generated in {output_dir}")
