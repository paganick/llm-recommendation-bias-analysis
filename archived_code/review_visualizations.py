"""
Quick script to preview some key visualizations
"""
from pathlib import Path
import subprocess

viz_dir = Path('analysis_outputs/visualizations/static')

print("=" * 80)
print("VISUALIZATION REVIEW")
print("=" * 80)
print()

# Organize by category
categories = {
    'bias_heatmaps': list((viz_dir / 'bias_heatmaps').glob('*.png')),
    'cross_cutting': list((viz_dir / 'cross_cutting').glob('*.png'))
}

for cat, files in categories.items():
    print(f"\n{cat.upper()}: {len(files)} files")
    print("-" * 80)
    
    # Show sample filenames
    for f in sorted(files)[:5]:
        print(f"  - {f.name}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

print()
print("=" * 80)
print("KEY VISUALIZATIONS TO EXAMINE")
print("=" * 80)
print()

key_viz = [
    ("Text Length Bias (Strongest Effect)", 
     "bias_heatmaps/text_length_bias_fully_disaggregated.png"),
    ("Text Length by Model", 
     "bias_heatmaps/text_length_model_comparison.png"),
    ("Text Length by Prompt Style", 
     "bias_heatmaps/text_length_prompt_style_effect.png"),
    ("Model Agreement Analysis", 
     "cross_cutting/model_agreement_scatter.png"),
    ("Bias Magnitude vs Significance", 
     "cross_cutting/bias_magnitude_vs_significance.png"),
    ("Polarizing Content Bias", 
     "bias_heatmaps/has_polarizing_content_bias_fully_disaggregated.png"),
    ("Political Leaning Bias", 
     "bias_heatmaps/author_political_leaning_bias_fully_disaggregated.png"),
    ("Toxicity Bias", 
     "bias_heatmaps/toxicity_bias_fully_disaggregated.png"),
]

for i, (desc, path) in enumerate(key_viz, 1):
    full_path = viz_dir / path
    if full_path.exists():
        print(f"{i}. {desc}")
        print(f"   Location: {full_path}")
    else:
        print(f"{i}. {desc} - NOT FOUND")

print()
print("=" * 80)
print("To view these visualizations:")
print("  1. Download them from the cluster")
print("  2. Or use: display <filename>  (if X11 forwarding enabled)")
print("  3. Or use: open <filename>  (on macOS)")
print("=" * 80)

