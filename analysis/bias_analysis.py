"""
Statistical Bias Analysis Framework

Analyzes systematic biases in LLM recommendations by comparing
recommended content distributions against baseline pools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from collections import defaultdict
import warnings


class BiasAnalyzer:
    """
    Analyze bias in recommendation results.

    Compares distributions of attributes (demographics, sentiment, topics, etc.)
    between recommended items and the baseline pool.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize bias analyzer.

        Args:
            alpha: Significance level for statistical tests (default 0.05)
        """
        self.alpha = alpha
        self.analysis_results = {}

    def analyze_demographic_bias(self,
                                  recommended_df: pd.DataFrame,
                                  pool_df: pd.DataFrame,
                                  demographic_cols: List[str]) -> Dict[str, Any]:
        """
        Analyze demographic bias in recommendations.

        Args:
            recommended_df: DataFrame of recommended items
            pool_df: DataFrame of baseline pool
            demographic_cols: List of demographic probability columns
                             (e.g., ['demo_aa', 'demo_white', 'demo_hispanic'])

        Returns:
            Dict with bias analysis results
        """
        results = {
            'pool_stats': {},
            'recommended_stats': {},
            'bias_scores': {},
            'statistical_tests': {}
        }

        for col in demographic_cols:
            if col not in pool_df.columns or col not in recommended_df.columns:
                warnings.warn(f"Column {col} not found in data")
                continue

            # Pool statistics
            pool_mean = pool_df[col].mean()
            pool_std = pool_df[col].std()

            # Recommended statistics
            rec_mean = recommended_df[col].mean()
            rec_std = recommended_df[col].std()

            # Bias score: difference in means (percentage points)
            bias_score = (rec_mean - pool_mean) * 100

            # Statistical test: two-sample t-test
            t_stat, p_value = stats.ttest_ind(
                recommended_df[col].dropna(),
                pool_df[col].dropna(),
                equal_var=False  # Welch's t-test
            )

            # Effect size: Cohen's d
            pooled_std = np.sqrt((pool_std**2 + rec_std**2) / 2)
            cohens_d = (rec_mean - pool_mean) / pooled_std if pooled_std > 0 else 0

            results['pool_stats'][col] = {
                'mean': pool_mean,
                'std': pool_std,
                'median': pool_df[col].median()
            }

            results['recommended_stats'][col] = {
                'mean': rec_mean,
                'std': rec_std,
                'median': recommended_df[col].median()
            }

            results['bias_scores'][col] = bias_score

            results['statistical_tests'][col] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d)
            }

        return results

    def analyze_sentiment_bias(self,
                               recommended_df: pd.DataFrame,
                               pool_df: pd.DataFrame,
                               sentiment_col: str = 'sentiment_polarity') -> Dict[str, Any]:
        """
        Analyze sentiment bias in recommendations.

        Args:
            recommended_df: DataFrame of recommended items
            pool_df: DataFrame of baseline pool
            sentiment_col: Column name for sentiment scores

        Returns:
            Dict with sentiment bias analysis
        """
        if sentiment_col not in pool_df.columns or sentiment_col not in recommended_df.columns:
            return {'error': f'Column {sentiment_col} not found'}

        pool_sentiment = pool_df[sentiment_col]
        rec_sentiment = recommended_df[sentiment_col]

        # Mean comparison
        pool_mean = pool_sentiment.mean()
        rec_mean = rec_sentiment.mean()
        bias_score = rec_mean - pool_mean

        # Statistical test
        t_stat, p_value = stats.ttest_ind(rec_sentiment.dropna(),
                                          pool_sentiment.dropna(),
                                          equal_var=False)

        # Distribution comparison (sentiment categories if available)
        sentiment_dist = {}
        if 'sentiment_label' in pool_df.columns:
            pool_dist = pool_df['sentiment_label'].value_counts(normalize=True)
            rec_dist = recommended_df['sentiment_label'].value_counts(normalize=True)

            # Chi-square test (using proportions scaled to same total)
            categories = list(set(pool_dist.index) | set(rec_dist.index))
            pool_counts = [pool_df[pool_df['sentiment_label'] == cat].shape[0] for cat in categories]
            rec_counts = [recommended_df[recommended_df['sentiment_label'] == cat].shape[0] for cat in categories]

            # Scale expected frequencies to match observed total
            rec_total = sum(rec_counts)
            pool_total = sum(pool_counts)
            pool_expected = [c * rec_total / pool_total if pool_total > 0 else 0 for c in pool_counts]

            chi2, chi_p = stats.chisquare(rec_counts, f_exp=pool_expected)

            sentiment_dist = {
                'pool_distribution': pool_dist.to_dict(),
                'recommended_distribution': rec_dist.to_dict(),
                'chi2_statistic': chi2,
                'chi2_p_value': chi_p,
                'chi2_significant': chi_p < self.alpha
            }

        return {
            'pool_mean_sentiment': pool_mean,
            'recommended_mean_sentiment': rec_mean,
            'bias_score': bias_score,
            'favors': 'positive' if bias_score > 0 else 'negative' if bias_score < 0 else 'neutral',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'sentiment_distribution': sentiment_dist
        }

    def analyze_topic_bias(self,
                          recommended_df: pd.DataFrame,
                          pool_df: pd.DataFrame,
                          topic_col: str = 'primary_topic') -> Dict[str, Any]:
        """
        Analyze topic bias in recommendations.

        Args:
            recommended_df: DataFrame of recommended items
            pool_df: DataFrame of baseline pool
            topic_col: Column name for topic labels

        Returns:
            Dict with topic bias analysis
        """
        if topic_col not in pool_df.columns or topic_col not in recommended_df.columns:
            return {'error': f'Column {topic_col} not found'}

        # Topic distributions
        pool_dist = pool_df[topic_col].value_counts(normalize=True)
        rec_dist = recommended_df[topic_col].value_counts(normalize=True)

        # Align distributions
        all_topics = list(set(pool_dist.index) | set(rec_dist.index))
        pool_props = {topic: pool_dist.get(topic, 0) for topic in all_topics}
        rec_props = {topic: rec_dist.get(topic, 0) for topic in all_topics}

        # Bias scores per topic (percentage point difference)
        topic_bias_scores = {
            topic: (rec_props[topic] - pool_props[topic]) * 100
            for topic in all_topics
        }

        # Chi-square test (scale expected frequencies to match observed total)
        pool_counts = pool_df[topic_col].value_counts()
        rec_counts = recommended_df[topic_col].value_counts()

        # Align counts
        pool_counts_aligned = [pool_counts.get(t, 0) for t in all_topics]
        rec_counts_aligned = [rec_counts.get(t, 0) for t in all_topics]

        # Scale expected frequencies
        rec_total = sum(rec_counts_aligned)
        pool_total = sum(pool_counts_aligned)
        pool_expected = [c * rec_total / pool_total if pool_total > 0 else 0 for c in pool_counts_aligned]

        chi2, p_value = stats.chisquare(rec_counts_aligned, f_exp=pool_expected)

        # Identify most biased topics
        sorted_topics = sorted(topic_bias_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'pool_distribution': pool_props,
            'recommended_distribution': rec_props,
            'topic_bias_scores': topic_bias_scores,
            'most_favored_topics': [t for t, s in sorted_topics if s > 2][:5],
            'most_disfavored_topics': [t for t, s in sorted_topics if s < -2][:5],
            'chi2_statistic': chi2,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

    def analyze_style_bias(self,
                          recommended_df: pd.DataFrame,
                          pool_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze writing style bias in recommendations.

        Tests for bias towards:
        - Formal vs. casual language
        - Presence of emojis, hashtags, URLs
        - Word count and complexity

        Args:
            recommended_df: DataFrame of recommended items
            pool_df: DataFrame of baseline pool

        Returns:
            Dict with style bias analysis
        """
        results = {}

        # Formality bias
        if 'formality_score' in pool_df.columns:
            pool_formality = pool_df['formality_score'].mean()
            rec_formality = recommended_df['formality_score'].mean()
            t_stat, p_value = stats.ttest_ind(
                recommended_df['formality_score'].dropna(),
                pool_df['formality_score'].dropna()
            )

            results['formality_bias'] = {
                'pool_mean': pool_formality,
                'recommended_mean': rec_formality,
                'bias_score': rec_formality - pool_formality,
                'favors': 'formal' if rec_formality > pool_formality else 'casual',
                'p_value': p_value,
                'significant': p_value < self.alpha
            }

        # Emoji, hashtag, URL bias
        for feature in ['has_emoji', 'has_hashtag', 'has_url']:
            if feature in pool_df.columns:
                pool_prop = pool_df[feature].mean()
                rec_prop = recommended_df[feature].mean()

                # Proportion test
                pool_count = pool_df[feature].sum()
                rec_count = recommended_df[feature].sum()
                pool_n = len(pool_df)
                rec_n = len(recommended_df)

                # Z-test for proportions
                z_stat, p_value = self._proportion_test(
                    rec_count, rec_n, pool_count, pool_n
                )

                results[f'{feature}_bias'] = {
                    'pool_proportion': pool_prop,
                    'recommended_proportion': rec_prop,
                    'bias_score': (rec_prop - pool_prop) * 100,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }

        # Word count bias
        if 'word_count' in pool_df.columns:
            pool_wc = pool_df['word_count'].mean()
            rec_wc = recommended_df['word_count'].mean()
            t_stat, p_value = stats.ttest_ind(
                recommended_df['word_count'].dropna(),
                pool_df['word_count'].dropna()
            )

            results['word_count_bias'] = {
                'pool_mean': pool_wc,
                'recommended_mean': rec_wc,
                'bias_score': rec_wc - pool_wc,
                'favors': 'longer' if rec_wc > pool_wc else 'shorter',
                'p_value': p_value,
                'significant': p_value < self.alpha
            }

        return results

    def analyze_polarization_bias(self,
                                  recommended_df: pd.DataFrame,
                                  pool_df: pd.DataFrame,
                                  polarization_col: str = 'polarization_score') -> Dict[str, Any]:
        """
        Analyze polarization/controversy bias in recommendations.

        Args:
            recommended_df: DataFrame of recommended items
            pool_df: DataFrame of baseline pool
            polarization_col: Column name for polarization scores

        Returns:
            Dict with polarization bias analysis
        """
        if polarization_col not in pool_df.columns or polarization_col not in recommended_df.columns:
            return {'error': f'Column {polarization_col} not found'}

        pool_pol = pool_df[polarization_col].mean()
        rec_pol = recommended_df[polarization_col].mean()

        t_stat, p_value = stats.ttest_ind(
            recommended_df[polarization_col].dropna(),
            pool_df[polarization_col].dropna()
        )

        return {
            'pool_mean_polarization': pool_pol,
            'recommended_mean_polarization': rec_pol,
            'bias_score': rec_pol - pool_pol,
            'favors': 'polarizing' if rec_pol > pool_pol else 'neutral',
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

    def comprehensive_bias_analysis(self,
                                   recommended_df: pd.DataFrame,
                                   pool_df: pd.DataFrame,
                                   demographic_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive bias analysis across all dimensions.

        Args:
            recommended_df: DataFrame of recommended items
            pool_df: DataFrame of baseline pool
            demographic_cols: List of demographic columns to analyze
                             (default: ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other'])

        Returns:
            Dict with all bias analyses
        """
        if demographic_cols is None:
            demographic_cols = ['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']

        results = {
            'metadata': {
                'pool_size': len(pool_df),
                'recommended_size': len(recommended_df),
                'recommendation_rate': len(recommended_df) / len(pool_df) if len(pool_df) > 0 else 0
            }
        }

        # Demographic bias
        if any(col in pool_df.columns for col in demographic_cols):
            available_demo_cols = [col for col in demographic_cols if col in pool_df.columns]
            results['demographic_bias'] = self.analyze_demographic_bias(
                recommended_df, pool_df, available_demo_cols
            )

        # Sentiment bias
        if 'sentiment_polarity' in pool_df.columns:
            results['sentiment_bias'] = self.analyze_sentiment_bias(
                recommended_df, pool_df
            )

        # Topic bias
        if 'primary_topic' in pool_df.columns:
            results['topic_bias'] = self.analyze_topic_bias(
                recommended_df, pool_df
            )

        # Style bias
        if 'formality_score' in pool_df.columns:
            results['style_bias'] = self.analyze_style_bias(
                recommended_df, pool_df
            )

        # Polarization bias
        if 'polarization_score' in pool_df.columns:
            results['polarization_bias'] = self.analyze_polarization_bias(
                recommended_df, pool_df
            )

        # Summary
        results['summary'] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of bias findings."""
        significant_biases = []

        # Check each bias type
        for bias_type in ['demographic_bias', 'sentiment_bias', 'topic_bias',
                         'style_bias', 'polarization_bias']:
            if bias_type in results:
                if bias_type == 'demographic_bias':
                    for demo, test in results[bias_type].get('statistical_tests', {}).items():
                        if test.get('significant'):
                            significant_biases.append({
                                'type': 'demographic',
                                'attribute': demo,
                                'bias_score': results[bias_type]['bias_scores'][demo],
                                'p_value': test['p_value']
                            })
                else:
                    if results[bias_type].get('significant'):
                        significant_biases.append({
                            'type': bias_type.replace('_bias', ''),
                            'p_value': results[bias_type].get('p_value')
                        })

        return {
            'total_significant_biases': len(significant_biases),
            'has_significant_bias': len(significant_biases) > 0,
            'significant_biases': significant_biases
        }

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def _proportion_test(self, x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float]:
        """Two-proportion z-test."""
        p1 = x1 / n1 if n1 > 0 else 0
        p2 = x2 / n2 if n2 > 0 else 0
        p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if (n1 > 0 and n2 > 0) else 1

        z_stat = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return z_stat, p_value


def compare_recommender_bias(results_list: List[Dict[str, Any]],
                             labels: List[str]) -> pd.DataFrame:
    """
    Compare bias across multiple recommender systems or configurations.

    Args:
        results_list: List of comprehensive_bias_analysis results
        labels: Labels for each result (e.g., ['GPT-4', 'Claude', 'Llama'])

    Returns:
        DataFrame comparing bias scores across systems
    """
    comparison_data = []

    for label, results in zip(labels, results_list):
        row = {'system': label}

        # Demographic bias scores
        if 'demographic_bias' in results:
            for demo, score in results['demographic_bias'].get('bias_scores', {}).items():
                row[f'{demo}_bias'] = score

        # Other bias scores
        for bias_type in ['sentiment_bias', 'polarization_bias']:
            if bias_type in results:
                row[f'{bias_type}'] = results[bias_type].get('bias_score', 0)

        # Significant bias count
        row['total_significant_biases'] = results.get('summary', {}).get('total_significant_biases', 0)

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)
