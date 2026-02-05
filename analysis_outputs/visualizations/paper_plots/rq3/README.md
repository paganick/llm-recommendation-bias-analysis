# RQ3: Model-Specific Content and Safety Biases

This directory contains all figures and data for RQ3, which examines how different LLM providers (Anthropic Claude Sonnet 4.5, OpenAI GPT-4o-mini, Google Gemini 2.0 Flash) differ in their handling of content polarization, sentiment polarity, and toxicity.

## Generated Files

### Content Polarization

**Heatmap Version:**
- `polarization_score_heatmap.png` - Single heatmap showing directional bias across all models and prompts
- `polarization_score_heatmap_data.csv` - Aggregated data (averaged across datasets)

**Bar Plot Version:**
- `polarization_score_by_model.png` - Six subplots (one per prompt) comparing models across datasets
- `polarization_score_by_model_data.csv` - Detailed data by model, dataset, and prompt

### Sentiment Polarity

**Heatmap Version:**
- `sentiment_polarity_heatmap.png` - Single heatmap showing directional bias across all models and prompts
- `sentiment_polarity_heatmap_data.csv` - Aggregated data (averaged across datasets)

**Bar Plot Version:**
- `sentiment_polarity_by_model.png` - Six subplots (one per prompt) comparing models across datasets
- `sentiment_polarity_by_model_data.csv` - Detailed data by model, dataset, and prompt

### Toxicity

**Heatmap Version:**
- `toxicity_heatmap.png` - Single heatmap showing directional bias across all models and prompts
- `toxicity_heatmap_data.csv` - Aggregated data (averaged across datasets)

**Bar Plot Version:**
- `toxicity_by_model.png` - Six subplots (one per prompt) comparing models across datasets
- `toxicity_by_model_data.csv` - Detailed data by model, dataset, and prompt

### LaTeX Documentation

- `RQ3_section.tex` - Complete LaTeX section with figure descriptions, interpretations, and implications

## Key Findings Summary

### Content Polarization
- **All models show polarization preference** under controversial (mean = +0.092) and neutral (mean = +0.083) prompts
- **Gemini shows most adaptive behavior**: Strong preference under controversial (+0.091) but aversion under informative (−0.024)
- **GPT-4o-mini is most stable**: Consistent moderate polarization preference across all prompts (std = 0.027)
- **Claude balances adaptation and consistency**: Responds to prompts but maintains coherent patterns

### Sentiment Polarity
- **Informative/general prompts favor positive sentiment** (mean = +0.028, +0.022)
- **Engaging/neutral prompts favor negative sentiment** (mean = −0.105, −0.074)
- **Gemini shows strongest negative preference** overall (mean = −0.056, std = 0.099)
- **GPT-4o-mini is most stable** (std = 0.062)
- **Sentiment patterns highly consistent across datasets** (unlike polarization)

### Toxicity (Most Striking Patterns)
- **Universal toxicity aversion under informative prompts**: All models show mean = −0.087
- **Universal toxicity tolerance under engaging prompts**: All models show mean = +0.095
- **Concerning pattern**: Models associate toxicity with engagement across all providers
- **Claude shows highest variability** (std = 0.091), most context-dependent handling
- **GPT-4o-mini most balanced** (mean = +0.002, std = 0.037)
- **Gemini shows highest overall tolerance** (mean = +0.017)

## Plot Specifications

### Heatmap Plots
- **Rows**: Models in fixed order (Anthropic, OpenAI, Gemini)
- **Columns**: Prompts (neutral, general, popular, engaging, informative, controversial)
- **Color scale**: Diverging (blue-white-red), centered at zero, symmetric limits
- **Annotations**: Show exact values (3 decimal places)
- **Size**: 10" × 4" (optimized for paper width)

### Bar Plot Plots
- **Layout**: 2 rows × 3 columns = 6 subplots (one per prompt)
- **X-axis**: Models (Anthropic, OpenAI, Gemini) - abbreviated names
- **Y-axis**: Directional bias - **SHARED SCALE across all subplots**
- **Bars**: Grouped by dataset (Bluesky = blue, Reddit = red, Twitter/X = gray)
- **Reference line**: Horizontal line at y=0
- **Size**: 15" × 8"

## Interpretation Guide

### Directional Bias Interpretation
- **Positive values**: LLM recommends content with HIGHER values than the pool
  - Polarization: Prefers more polarized content
  - Sentiment: Prefers more positive sentiment
  - Toxicity: Tolerates toxic content (concerning)

- **Negative values**: LLM recommends content with LOWER values than the pool
  - Polarization: Avoids polarized content
  - Sentiment: Prefers more negative sentiment
  - Toxicity: Avoids toxic content (desirable)

### Color Coding
- **Red (positive)**: Over-representation or preference
- **Blue (negative)**: Under-representation or aversion
- **White**: No bias (recommended ≈ pool)

## Statistical Summary

### Polarization
- Overall range: [−0.024, +0.131]
- Most variable by prompt: Controversial (mean = +0.092) vs. Popular (mean = +0.020)
- Most consistent model: GPT-4o-mini (std = 0.027)

### Sentiment Polarity
- Overall range: [−0.258, +0.102]
- Most variable by prompt: Engaging (mean = −0.105) vs. Informative (mean = +0.028)
- Most consistent model: GPT-4o-mini (std = 0.062)

### Toxicity
- Overall range: [−0.149, +0.188]
- Most variable by prompt: Informative (mean = −0.087) vs. Engaging (mean = +0.095)
- Most consistent model: GPT-4o-mini (std = 0.037)

## Cross-Model Insights

1. **Consistency indicates industry-wide patterns**: Similar biases across Anthropic, OpenAI, and Google suggest shared training data characteristics and architectural properties

2. **Model-specific philosophies emerge**:
   - **Anthropic Claude**: Context-dependent, high variability, sophisticated prompt interpretation
   - **OpenAI GPT-4o-mini**: Balanced, stable, moderate biases across dimensions
   - **Google Gemini**: Adaptive polarization handling but highest toxicity tolerance

3. **Prompt engineering has limits**: While effective for polarization and sentiment, toxicity-engagement association persists across all prompts and models

4. **Safety-engagement tradeoff is fundamental**: The systematic inversion between informative (safe) and engaging (tolerant) prompts reveals a core challenge in LLM-based recommendation systems

## Usage in Paper

The LaTeX file (`RQ3_section.tex`) provides:
- Complete section text with subsections for each metric
- Figure captions with detailed descriptions
- Interpretation of patterns and model differences
- Discussion of safety implications
- Cross-model comparison analysis
- Data availability statement

Simply include this section in your main paper LaTeX file with:
```latex
\input{analysis_outputs/visualizations/paper_plots/rq3/RQ3_section.tex}
```

## Reproduction

To regenerate all plots:
```bash
python generate_rq3_plots.py
```

This will:
1. Load directional bias data from `analysis_outputs/directional_bias_data.csv`
2. Generate all 6 plot files (3 heatmaps + 3 bar plots)
3. Export all 6 CSV data files
4. Print summary statistics

## Data Format

### Heatmap CSV Format
```
provider,neutral,general,popular,engaging,informative,controversial,provider_display
anthropic,0.087,0.029,0.033,0.066,0.016,0.095,Anthropic Claude Sonnet 4.5
openai,0.089,0.029,0.033,0.063,0.032,0.093,OpenAI GPT-4o-mini
gemini,0.073,0.029,0.006,0.064,-0.024,0.091,Google Gemini 2.0 Flash
```

### Detailed CSV Format
```
provider,provider_display,dataset,dataset_display,prompt_style,prompt_display,directional_bias
anthropic,Anthropic Claude Sonnet 4.5,bluesky,Bluesky,neutral,Neutral,0.087
...
```

All values represent directional bias: recommended_value - pool_value
