"""
Create visualizations for LLM-inferred user metadata
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
sns.set_palette("husl")

# Load LLM inference results
df = pd.read_csv('./outputs/llm_inference/user_inference_gpt_4o_mini.csv')

print(f"Loaded LLM inference results for {len(df)} users")

# Create output directory
output_dir = Path('./outputs/llm_inference/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Create comprehensive dashboard
fig = plt.figure(figsize=(22, 14))

# 1. Gender Distribution
ax1 = plt.subplot(3, 4, 1)
gender_counts = df['gender_value'].value_counts()
colors_gender = {'male': '#2ca02c', 'female': '#ff7f0e', 'unknown': '#7f7f7f'}
color_list = [colors_gender.get(x, '#7f7f7f') for x in gender_counts.index]
bars = ax1.bar(gender_counts.index, gender_counts.values, color=color_list, edgecolor='black', linewidth=1.5)
ax1.set_title('Gender Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Number of Users')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 2. Political Leaning
ax2 = plt.subplot(3, 4, 2)
pol_counts = df['political_leaning_value'].value_counts()
colors_pol = {'left': '#1f77b4', 'center-left': '#aec7e8', 'center': '#9467bd',
              'center-right': '#ffbb78', 'right': '#d62728', 'unknown': '#7f7f7f'}
color_list_pol = [colors_pol.get(x, '#7f7f7f') for x in pol_counts.index]
bars = ax2.barh(range(len(pol_counts)), pol_counts.values, color=color_list_pol, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(pol_counts)))
ax2.set_yticklabels(pol_counts.index)
ax2.set_title('Political Leaning Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Users')
for i, (bar, val) in enumerate(zip(bars, pol_counts.values)):
    ax2.text(val + 1, i, f'{int(val)} ({val/len(df)*100:.1f}%)',
             ha='left', va='center', fontweight='bold')

# 3. Race/Ethnicity
ax3 = plt.subplot(3, 4, 3)
race_counts = df['race_ethnicity_value'].value_counts()
race_counts.plot(kind='barh', ax=ax3, color='coral', edgecolor='black', linewidth=1.5)
ax3.set_title('Race/Ethnicity Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Users')
ax3.set_ylabel('Race/Ethnicity')
for i, v in enumerate(race_counts):
    ax3.text(v + 2, i, f'{v} ({v/len(df)*100:.1f}%)', ha='left', va='center', fontweight='bold')

# 4. Age/Generation
ax4 = plt.subplot(3, 4, 4)
age_counts = df['age_generation_value'].value_counts()
colors_age = {'gen_z': '#ff6b6b', 'millennial': '#4ecdc4', 'gen_x': '#45b7d1',
              'boomer': '#96ceb4', 'unknown': '#7f7f7f'}
color_list_age = [colors_age.get(x, '#7f7f7f') for x in age_counts.index]
wedges, texts, autotexts = ax4.pie(age_counts, labels=age_counts.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_age)
ax4.set_title('Age/Generation Distribution', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# 5. Gender Confidence (for known genders)
ax5 = plt.subplot(3, 4, 5)
gender_known = df[df['gender_value'] != 'unknown']
conf_counts = gender_known['gender_confidence'].value_counts()
colors_conf = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
color_list_conf = [colors_conf.get(x, '#7f7f7f') for x in conf_counts.index]
bars = ax5.bar(conf_counts.index, conf_counts.values, color=color_list_conf, edgecolor='black', linewidth=1.5)
ax5.set_title('Gender Inference Confidence\n(Known Genders Only)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Confidence Level')
ax5.set_ylabel('Number of Users')
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(gender_known)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 6. Political Confidence (for known political leanings)
ax6 = plt.subplot(3, 4, 6)
pol_known = df[df['political_leaning_value'] != 'unknown']
pol_conf_counts = pol_known['political_leaning_confidence'].value_counts()
color_list_pol_conf = [colors_conf.get(x, '#7f7f7f') for x in pol_conf_counts.index]
bars = ax6.bar(pol_conf_counts.index, pol_conf_counts.values, color=color_list_pol_conf, edgecolor='black', linewidth=1.5)
ax6.set_title('Political Inference Confidence\n(Known Politics Only)', fontsize=14, fontweight='bold')
ax6.set_xlabel('Confidence Level')
ax6.set_ylabel('Number of Users')
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(pol_known)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# 7. Education Level
ax7 = plt.subplot(3, 4, 7)
edu_counts = df['education_level_value'].value_counts()
edu_order = ['high_school', 'some_college', 'college', 'graduate', 'unknown']
edu_counts = edu_counts.reindex([x for x in edu_order if x in edu_counts.index])
bars = ax7.barh(range(len(edu_counts)), edu_counts.values, color='mediumpurple', edgecolor='black', linewidth=1.5)
ax7.set_yticks(range(len(edu_counts)))
ax7.set_yticklabels(edu_counts.index)
ax7.set_title('Education Level Distribution', fontsize=14, fontweight='bold')
ax7.set_xlabel('Number of Users')
for i, (bar, val) in enumerate(zip(bars, edu_counts.values)):
    ax7.text(val + 1, i, f'{int(val)} ({val/len(df)*100:.1f}%)',
             ha='left', va='center', fontweight='bold')

# 8. Geographic Origin (Top 10)
ax8 = plt.subplot(3, 4, 8)
geo_counts = df['geographic_origin_value'].value_counts().head(10)
geo_counts.plot(kind='barh', ax=ax8, color='steelblue', edgecolor='black', linewidth=1.5)
ax8.set_title('Geographic Origin (Top 10)', fontsize=14, fontweight='bold')
ax8.set_xlabel('Number of Users')
ax8.set_ylabel('Location')
for i, v in enumerate(geo_counts):
    ax8.text(v + 0.5, i, f'{v}', ha='left', va='center', fontweight='bold', fontsize=9)

# 9. Profession (Top 10, excluding unknown)
ax9 = plt.subplot(3, 4, 9)
prof_counts = df[df['profession_value'] != 'unknown']['profession_value'].value_counts().head(10)
if len(prof_counts) > 0:
    prof_counts.plot(kind='barh', ax=ax9, color='teal', edgecolor='black', linewidth=1.5)
    ax9.set_title('Top 10 Professions (excl. unknown)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Number of Users')
    for i, v in enumerate(prof_counts):
        ax9.text(v + 0.1, i, f'{v}', ha='left', va='center', fontweight='bold', fontsize=9)
else:
    ax9.text(0.5, 0.5, 'No profession data', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Top Professions', fontsize=14, fontweight='bold')

# 10. Classification Success Rates
ax10 = plt.subplot(3, 4, 10)
success_rates = {
    'Gender': (len(df) - (df['gender_value'] == 'unknown').sum()) / len(df) * 100,
    'Political': (len(df) - (df['political_leaning_value'] == 'unknown').sum()) / len(df) * 100,
    'Age': (len(df) - (df['age_generation_value'] == 'unknown').sum()) / len(df) * 100,
    'Race': (len(df) - (df['race_ethnicity_value'] == 'unknown').sum()) / len(df) * 100,
    'Education': (len(df) - (df['education_level_value'] == 'unknown').sum()) / len(df) * 100
}
bars = ax10.bar(success_rates.keys(), success_rates.values(),
                color=['#2ca02c', '#d62728', '#4ecdc4', '#ff7f0e', '#9467bd'],
                edgecolor='black', linewidth=1.5)
ax10.set_title('Classification Success Rates', fontsize=14, fontweight='bold')
ax10.set_ylabel('Success Rate (%)')
ax10.set_ylim(0, 100)
ax10.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
for bar in bars:
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold')

# 11. Gender by Political Leaning (known values only)
ax11 = plt.subplot(3, 4, 11)
gender_pol = df[(df['gender_value'] != 'unknown') & (df['political_leaning_value'].isin(['left', 'right']))]
if len(gender_pol) > 0:
    cross_tab = pd.crosstab(gender_pol['political_leaning_value'], gender_pol['gender_value'])
    cross_tab.plot(kind='bar', ax=ax11, color=['#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=1.5)
    ax11.set_title('Gender by Political Leaning\n(Left vs Right)', fontsize=14, fontweight='bold')
    ax11.set_xlabel('Political Leaning')
    ax11.set_ylabel('Count')
    ax11.legend(title='Gender')
    ax11.tick_params(axis='x', rotation=0)
else:
    ax11.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax11.transAxes)

# 12. Token Usage per User
ax12 = plt.subplot(3, 4, 12)
if 'tokens_used' in df.columns:
    ax12.hist(df['tokens_used'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax12.axvline(df['tokens_used'].mean(), color='red', linestyle='--', linewidth=2,
                 label=f'Mean: {df["tokens_used"].mean():.0f}')
    ax12.axvline(df['tokens_used'].median(), color='green', linestyle='--', linewidth=2,
                 label=f'Median: {df["tokens_used"].median():.0f}')
    ax12.set_title('Token Usage Distribution', fontsize=14, fontweight='bold')
    ax12.set_xlabel('Tokens per User')
    ax12.set_ylabel('Frequency')
    ax12.legend()
else:
    ax12.text(0.5, 0.5, 'No token data', ha='center', va='center', transform=ax12.transAxes)

plt.suptitle('LLM-Inferred User Metadata Overview - 279 Twitter Users (GPT-4o-mini)',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'llm_inference_overview.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'llm_inference_overview.png'}")

# Create detailed political breakdown
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Political users only (excluding unknown)
ax1 = axes[0]
political_only = df[df['political_leaning_value'] != 'unknown']['political_leaning_value'].value_counts()
colors_pol_pie = {'left': '#1f77b4', 'center-left': '#aec7e8', 'center': '#9467bd',
                  'center-right': '#ffbb78', 'right': '#d62728'}
color_list_pol_pie = [colors_pol_pie.get(x, '#7f7f7f') for x in political_only.index]
wedges, texts, autotexts = ax1.pie(political_only, labels=political_only.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_pol_pie)
ax1.set_title(f'Political Leaning\n(Political Users Only, n={len(political_only)})',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

# All users including unknown
ax2 = axes[1]
all_political = df['political_leaning_value'].value_counts()
colors_all_pol = {**colors_pol_pie, 'unknown': '#7f7f7f'}
color_list_all_pol = [colors_all_pol.get(x, '#7f7f7f') for x in all_political.index]
wedges, texts, autotexts = ax2.pie(all_political, labels=all_political.index,
                                     autopct='%1.1f%%', startangle=90,
                                     colors=color_list_all_pol)
ax2.set_title(f'Political Leaning\n(All Users, n={len(df)})',
              fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

plt.suptitle('Political Leaning Analysis (LLM Inferred)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'political_breakdown_llm.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'political_breakdown_llm.png'}")

# Create confidence analysis
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gender confidence breakdown
ax1 = axes[0, 0]
for gender in ['male', 'female']:
    gender_data = df[df['gender_value'] == gender]
    if len(gender_data) > 0:
        conf_dist = gender_data['gender_confidence'].value_counts()
        ax1.bar([f'{gender}_{c}' for c in conf_dist.index], conf_dist.values,
                label=gender.capitalize(), alpha=0.7, edgecolor='black')
ax1.set_title('Gender Inference Confidence by Gender', fontsize=14, fontweight='bold')
ax1.set_xlabel('Gender_Confidence')
ax1.set_ylabel('Count')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Political confidence breakdown
ax2 = axes[0, 1]
for pol in ['left', 'right']:
    pol_data = df[df['political_leaning_value'] == pol]
    if len(pol_data) > 0:
        conf_dist = pol_data['political_leaning_confidence'].value_counts()
        ax2.bar([f'{pol}_{c}' for c in conf_dist.index], conf_dist.values,
                label=pol.capitalize(), alpha=0.7, edgecolor='black')
ax2.set_title('Political Inference Confidence by Leaning', fontsize=14, fontweight='bold')
ax2.set_xlabel('Political_Confidence')
ax2.set_ylabel('Count')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# Race confidence (for known races)
ax3 = axes[1, 0]
race_known = df[df['race_ethnicity_value'] != 'unknown']
if len(race_known) > 0:
    race_conf = race_known['race_ethnicity_confidence'].value_counts()
    bars = ax3.bar(race_conf.index, race_conf.values, color='coral', edgecolor='black', linewidth=1.5)
    ax3.set_title('Race/Ethnicity Confidence\n(Known Only)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Confidence Level')
    ax3.set_ylabel('Count')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Overall confidence summary
ax4 = axes[1, 1]
confidence_summary = {
    'Gender\n(Known)': (df[df['gender_value'] != 'unknown']['gender_confidence'] == 'high').sum() /
                       (df['gender_value'] != 'unknown').sum() * 100,
    'Political\n(Known)': (df[df['political_leaning_value'] != 'unknown']['political_leaning_confidence'] == 'high').sum() /
                          (df['political_leaning_value'] != 'unknown').sum() * 100,
    'Race\n(Known)': (df[df['race_ethnicity_value'] != 'unknown']['race_ethnicity_confidence'] == 'high').sum() /
                     (df['race_ethnicity_value'] != 'unknown').sum() * 100 if (df['race_ethnicity_value'] != 'unknown').sum() > 0 else 0
}
bars = ax4.bar(confidence_summary.keys(), confidence_summary.values(),
               color=['#2ca02c', '#d62728', '#ff7f0e'], edgecolor='black', linewidth=1.5)
ax4.set_title('High Confidence Rate\n(% of Known Values)', fontsize=14, fontweight='bold')
ax4.set_ylabel('High Confidence (%)')
ax4.set_ylim(0, 100)
ax4.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
ax4.legend()

plt.suptitle('Inference Confidence Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'confidence_analysis.png'}")

print("\n✓ All visualizations created successfully!")
print(f"Output directory: {output_dir}")
