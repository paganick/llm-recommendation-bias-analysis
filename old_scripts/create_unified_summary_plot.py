"""
Create unified summary figure showing key findings from prompt style comparison
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

def load_and_process_results():
    """Load and process results for plotting."""

    with open('./outputs/prompt_style_comparison/prompt_style_results.pkl', 'rb') as f:
        results = pickle.load(f)

    styles = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

    data = {
        'style': [],
        'sentiment_polarity': [],
        'sentiment_polarity_se': [],
        'polarization_score': [],
        'polarization_score_se': [],
        'emoji_usage': [],
        'emoji_usage_se': [],
        'male_bias': [],
        'male_bias_se': [],
        'african_american_bias': [],
        'african_american_bias_se': [],
        'formality_score': [],
        'formality_score_se': []
    }

    for style in styles:
        style_results = [r for r in results if r['prompt_style'] == style]
        n = len(style_results)

        # Sentiment polarity
        sentiment_diffs = [r['diff_sentiment_polarity'] for r in style_results]
        data['sentiment_polarity'].append(np.mean(sentiment_diffs))
        data['sentiment_polarity_se'].append(np.std(sentiment_diffs, ddof=1) / np.sqrt(n))

        # Polarization
        polar_diffs = [r['diff_polarization_score'] for r in style_results]
        data['polarization_score'].append(np.mean(polar_diffs))
        data['polarization_score_se'].append(np.std(polar_diffs, ddof=1) / np.sqrt(n))

        # Emoji usage
        emoji_diffs = [r['diff_has_emoji_pct'] for r in style_results]
        data['emoji_usage'].append(np.mean(emoji_diffs))
        data['emoji_usage_se'].append(np.std(emoji_diffs, ddof=1) / np.sqrt(n))

        # Formality
        formality_diffs = [r['diff_formality_score'] for r in style_results]
        data['formality_score'].append(np.mean(formality_diffs))
        data['formality_score_se'].append(np.std(formality_diffs, ddof=1) / np.sqrt(n))

        # Gender bias (male)
        male_diffs = []
        for r in style_results:
            pool = r['pool_gender_value']
            rec = r['recommended_gender_value']
            diff = (rec.get('male', 0) - pool.get('male', 0)) * 100
            male_diffs.append(diff)
        data['male_bias'].append(np.mean(male_diffs))
        data['male_bias_se'].append(np.std(male_diffs, ddof=1) / np.sqrt(n))

        # Race bias (African American)
        aa_diffs = []
        for r in style_results:
            pool = r['pool_race_ethnicity_value']
            rec = r['recommended_race_ethnicity_value']
            diff = (rec.get('african_american', 0) - pool.get('african_american', 0)) * 100
            aa_diffs.append(diff)
        data['african_american_bias'].append(np.mean(aa_diffs))
        data['african_american_bias_se'].append(np.std(aa_diffs, ddof=1) / np.sqrt(n))

        data['style'].append(style)

    return pd.DataFrame(data)


def create_unified_plot(df):
    """Create unified summary plot."""

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Prompt Style Comparison: Key Biases in LLM Recommendations',
                 fontsize=18, fontweight='bold', y=0.995)

    x = np.arange(len(df))
    width = 0.6

    # Color scheme: highlight positive and negative
    def get_colors(values):
        colors = []
        for v in values:
            if v > 0:
                colors.append('#e74c3c' if v > 0.05 else '#f39c12')  # Red/orange for positive
            else:
                colors.append('#3498db' if v < -0.05 else '#95a5a6')  # Blue/gray for negative
        return colors

    # 1. Sentiment Polarity (TOP LEFT - MOST IMPORTANT)
    ax = axes[0, 0]
    colors = get_colors(df['sentiment_polarity'])
    bars = ax.bar(x, df['sentiment_polarity'], width, yerr=df['sentiment_polarity_se'],
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Sentiment Polarity\nDifference', fontsize=11, fontweight='bold')
    ax.set_title('Sentiment: Positive vs. Negative Content\n(Positive = More positive sentiment)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['sentiment_polarity'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # 2. Gender Bias - Male (TOP MIDDLE)
    ax = axes[0, 1]
    colors = get_colors(df['male_bias'])
    bars = ax.bar(x, df['male_bias'], width, yerr=df['male_bias_se'],
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Male Author Bias\n(percentage points)', fontsize=11, fontweight='bold')
    ax.set_title('Gender Bias: Male Over-representation\n(Positive = More male authors)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, df['male_bias'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}pp',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # 3. Polarization Score (TOP RIGHT)
    ax = axes[0, 2]
    colors = get_colors(df['polarization_score'])
    bars = ax.bar(x, df['polarization_score'], width, yerr=df['polarization_score_se'],
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Polarization Score\nDifference', fontsize=11, fontweight='bold')
    ax.set_title('Polarization: Controversial Content\n(Positive = More polarizing)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, df['polarization_score'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # 4. Emoji Usage (BOTTOM LEFT)
    ax = axes[1, 0]
    colors = get_colors(df['emoji_usage'])
    bars = ax.bar(x, df['emoji_usage'], width, yerr=df['emoji_usage_se'],
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Emoji Usage\n(percentage points)', fontsize=11, fontweight='bold')
    ax.set_title('Stylistic Bias: Emoji Usage\n(Positive = More emojis)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, df['emoji_usage'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}pp',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # 5. Formality (BOTTOM MIDDLE)
    ax = axes[1, 1]
    colors = get_colors(df['formality_score'])
    bars = ax.bar(x, df['formality_score'], width, yerr=df['formality_score_se'],
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Formality Score\nDifference', fontsize=11, fontweight='bold')
    ax.set_title('Stylistic Bias: Formality\n(Positive = More formal)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, df['formality_score'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # 6. Race Bias - African American (BOTTOM RIGHT)
    ax = axes[1, 2]
    colors = get_colors(df['african_american_bias'])
    bars = ax.bar(x, df['african_american_bias'], width, yerr=df['african_american_bias_se'],
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('African American Bias\n(percentage points)', fontsize=11, fontweight='bold')
    ax.set_title('Race Bias: African American Representation\n(Positive = Over-represented)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['style'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, df['african_american_bias'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}pp',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

    # Add overall summary text box
    summary_text = (
        "Key Findings:\n"
        "• 'Popular/viral' prompts select MORE POSITIVE content (not negative as hypothesized)\n"
        "• 'Popular' shows strongest male bias (+4.7pp) and emoji usage (+8.9pp)\n"
        "• 'Controversial' and 'neutral' select most negative content (-0.17, -0.15)\n"
        "• African American authors consistently under-represented across most styles\n"
        "• 'Informative' shows most formal content, least emoji usage"
    )

    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')

    plt.tight_layout(rect=[0, 0.06, 1, 0.99])

    return fig


def main():
    print('Creating unified summary plot...')

    # Load and process data
    df = load_and_process_results()

    # Create plot
    fig = create_unified_plot(df)

    # Save
    output_dir = Path('./outputs/prompt_style_comparison')
    output_path = output_dir / 'unified_summary.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved unified summary to {output_path}')

    plt.close()


if __name__ == '__main__':
    main()
