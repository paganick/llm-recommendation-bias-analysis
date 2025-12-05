"""
Create visualizations for persona metadata
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
sns.set_palette("husl")

# Load metadata
df = pd.read_pickle('./outputs/persona_analysis/user_metadata.pkl')

print(f"Loaded metadata for {len(df)} users")

# Create output directory
output_dir = Path('./outputs/persona_analysis/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 12))

# 1. Political Leaning
ax1 = plt.subplot(2, 4, 1)
political_counts = df['political_leaning'].value_counts()
colors = {'conservative': '#d62728', 'liberal': '#1f77b4', 'center': '#9467bd', 'unknown': '#7f7f7f'}
color_list = [colors.get(x, '#7f7f7f') for x in political_counts.index]
political_counts.plot(kind='bar', ax=ax1, color=color_list)
ax1.set_title('Political Leaning Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Political Leaning')
ax1.set_ylabel('Number of Users')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(political_counts):
    ax1.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Primary Topics
ax2 = plt.subplot(2, 4, 2)
topic_counts = df['primary_topic'].value_counts().head(8)
topic_counts.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_title('Primary Topics', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Users')
ax2.set_ylabel('Topic')
for i, v in enumerate(topic_counts):
    ax2.text(v + 1, i, str(v), ha='left', va='center', fontweight='bold')

# 3. Gender Distribution
ax3 = plt.subplot(2, 4, 3)
gender_counts = df['gender'].value_counts()
colors_gender = {'male': '#2ca02c', 'female': '#ff7f0e', 'unknown': '#7f7f7f'}
color_list_gender = [colors_gender.get(x, '#7f7f7f') for x in gender_counts.index]
wedges, texts, autotexts = ax3.pie(gender_counts, labels=gender_counts.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_gender)
ax3.set_title('Gender Distribution', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 4. Primary Tone
ax4 = plt.subplot(2, 4, 4)
tone_counts = df['primary_tone'].value_counts().head(8)
tone_counts.plot(kind='barh', ax=ax4, color='coral')
ax4.set_title('Primary Tone', fontsize=14, fontweight='bold')
ax4.set_xlabel('Number of Users')
ax4.set_ylabel('Tone')
for i, v in enumerate(tone_counts):
    ax4.text(v + 1, i, str(v), ha='left', va='center', fontweight='bold')

# 5. Tweets per User Distribution
ax5 = plt.subplot(2, 4, 5)
ax5.hist(df['tweet_count'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
ax5.axvline(df['tweet_count'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["tweet_count"].mean():.1f}')
ax5.axvline(df['tweet_count'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["tweet_count"].median():.1f}')
ax5.set_title('Tweets per User Distribution', fontsize=14, fontweight='bold')
ax5.set_xlabel('Number of Tweets')
ax5.set_ylabel('Number of Users')
ax5.legend()

# 6. Writing Style Features
ax6 = plt.subplot(2, 4, 6)
style_features = {
    'Uses Emoji': df['uses_emoji'].sum(),
    'Uses Slang': df['uses_slang'].sum(),
    'Uses Hashtags': df['uses_hashtags'].sum()
}
bars = ax6.bar(style_features.keys(), style_features.values(), color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
ax6.set_title('Writing Style Features', fontsize=14, fontweight='bold')
ax6.set_ylabel('Number of Users')
ax6.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 7. Profession (Top 10 excluding unknown)
ax7 = plt.subplot(2, 4, 7)
prof_counts = df[df['profession'] != 'unknown']['profession'].value_counts().head(10)
if len(prof_counts) > 0:
    prof_counts.plot(kind='barh', ax=ax7, color='teal')
    ax7.set_title('Top Professions (excl. unknown)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Number of Users')
    ax7.set_ylabel('Profession')
    for i, v in enumerate(prof_counts):
        ax7.text(v + 0.1, i, str(v), ha='left', va='center', fontweight='bold')
else:
    ax7.text(0.5, 0.5, 'No profession data', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Top Professions', fontsize=14, fontweight='bold')

# 8. Sentiment Polarity
ax8 = plt.subplot(2, 4, 8)
sentiment_counts = df['sentiment_polarity'].value_counts()
colors_sentiment = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
color_list_sentiment = [colors_sentiment.get(x, '#7f7f7f') for x in sentiment_counts.index]
sentiment_counts.plot(kind='bar', ax=ax8, color=color_list_sentiment)
ax8.set_title('Overall Sentiment Polarity', fontsize=14, fontweight='bold')
ax8.set_xlabel('Sentiment')
ax8.set_ylabel('Number of Users')
ax8.tick_params(axis='x', rotation=0)
for i, v in enumerate(sentiment_counts):
    ax8.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.suptitle('Persona Metadata Overview - 279 Twitter Users',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'metadata_overview.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'metadata_overview.png'}")

# Create political leaning breakdown
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Political users only (excluding unknown)
ax1 = axes[0]
political_only = df[df['political_leaning'] != 'unknown']['political_leaning'].value_counts()
colors_pol = {'conservative': '#d62728', 'liberal': '#1f77b4', 'center': '#9467bd'}
color_list_pol = [colors_pol.get(x, '#7f7f7f') for x in political_only.index]
wedges, texts, autotexts = ax1.pie(political_only, labels=political_only.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_pol)
ax1.set_title('Political Leaning\n(Political Users Only, n=169)',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

# All users including unknown
ax2 = axes[1]
all_political = df['political_leaning'].value_counts()
colors_all = {'conservative': '#d62728', 'liberal': '#1f77b4', 'center': '#9467bd', 'unknown': '#7f7f7f'}
color_list_all = [colors_all.get(x, '#7f7f7f') for x in all_political.index]
wedges, texts, autotexts = ax2.pie(all_political, labels=all_political.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_all)
ax2.set_title('Political Leaning\n(All Users, n=279)',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

plt.suptitle('Political Leaning Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'political_breakdown.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'political_breakdown.png'}")

# Create gender breakdown
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gender identifiable only
ax1 = axes[0]
gender_only = df[df['gender'] != 'unknown']['gender'].value_counts()
colors_gen = {'male': '#2ca02c', 'female': '#ff7f0e'}
color_list_gen = [colors_gen.get(x, '#7f7f7f') for x in gender_only.index]
wedges, texts, autotexts = ax1.pie(gender_only, labels=gender_only.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_gen)
ax1.set_title('Gender Distribution\n(Identifiable Only, n=149)',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

# All users including unknown
ax2 = axes[1]
all_gender = df['gender'].value_counts()
colors_all_gen = {'male': '#2ca02c', 'female': '#ff7f0e', 'unknown': '#7f7f7f'}
color_list_all_gen = [colors_all_gen.get(x, '#7f7f7f') for x in all_gender.index]
wedges, texts, autotexts = ax2.pie(all_gender, labels=all_gender.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_all_gen)
ax2.set_title('Gender Distribution\n(All Users, n=279)',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

plt.suptitle('Gender Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'gender_breakdown.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'gender_breakdown.png'}")

print("\n✓ All visualizations created successfully!")
