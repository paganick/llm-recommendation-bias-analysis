# LLM Recommendation Bias Analysis

A comprehensive framework for analyzing systematic biases in LLM-based recommendation systems across multiple social media platforms, models, and prompt styles.

## Overview

This project evaluates whether large language model (LLM) recommender systems exhibit systematic biases when suggesting content. The framework provides end-to-end automation from running experiments to generating interactive dashboards with rigorous statistical analysis.

**Current Status**: Complete analysis of 9 experiments (3 datasets × 3 models × 6 prompt styles = 540,000 recommendations analyzed)

## Key Findings

- **Models are consistent**: Only 3 significant differences (2.9%) between OpenAI, Anthropic, and Google models
- **Datasets matter**: 30 significant differences (29.4%) across Twitter, Reddit, and Bluesky platforms
- **Universal predictors**: `formality_score`, `polarization_score`, and structural features (`has_url`, `has_emoji`, `has_hashtag`) consistently predict LLM recommendations
- **Prompt styles matter**: 40+ significant differences per experiment across 6 prompt styles

## Key Features

### Experiment Framework
- **Multiple Datasets**: Twitter, Reddit, Bluesky
- **Multiple Models**: OpenAI GPT-4o-mini, Anthropic Claude Sonnet 4.5, Google Gemini 2.0 Flash
- **6 Prompt Styles**: general, popular, engaging, informative, controversial, neutral
- **Large-scale Testing**: 100 trials × 100 posts per style = 60,000 recommendations per experiment

### Analysis Pipeline
- **Stratified Analysis**: Separate logistic regression per prompt style
- **Statistical Comparison**: Wald tests for coefficient differences across styles
- **Meta-Analysis**: Pooled effect sizes with heterogeneity testing (I²)
- **Interactive Dashboards**: Plotly HTML visualizations with drill-down capabilities

## Project Structure

```
llm-recommendation-bias-analysis/
├── data/                              # Data loading modules
├── inference/                         # Metadata inference (preprocessing)
├── utils/                             # LLM client wrappers
├── analysis/                          # Bias analysis modules
├── datasets/                          # Local dataset copies (not committed)
├── outputs/                           # All experiment outputs
│   ├── experiments/                   # Individual experiment results
│   │   └── {dataset}_{provider}_{model}/
│   │       ├── post_level_data.csv    # 60,000 recommendations
│   │       └── stratified_analysis/   # Per-style analysis
│   ├── meta_analysis/                 # Cross-experiment aggregation
│   └── dashboards/                    # Interactive HTML visualizations
├── run_experiment.py                  # Run single experiment
├── run_all_experiments.py             # Run multiple experiments in parallel
├── run_experiment_with_tracking.py    # Run with progress tracking
├── stratified_analysis.py             # Stratified regression analysis
├── meta_analysis.py                   # Cross-experiment meta-analysis
├── create_dashboards.py               # Generate interactive dashboards
├── run_full_analysis_pipeline.py      # Master orchestration script
└── PROGRESS.md                        # Detailed progress tracker
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd llm-recommendation-bias-analysis

# Install dependencies
pip install -r requirements.txt

# For HuggingFace models (optional)
pip install transformers torch accelerate
```

### 2. Setup Datasets

The framework expects datasets in `./datasets/{platform}/personas.pkl` format.

Copy your datasets:
```bash
mkdir -p datasets/twitter datasets/reddit datasets/bluesky
# Copy your persona datasets to these directories
```

Dataset format (pickle files with pandas DataFrame):
- Required columns: `message`, `username` (or `user_id`), `persona`, `training`
- `message`: Post text
- `persona`: User demographic/persona information
- `training`: Train/test split indicator (0 or 1)

### 3. Test Pipeline

```bash
python test_pipeline.py
```

This verifies:
- Dataset loading works
- Metadata inference works
- LLM clients initialize correctly

### 4. Run Complete Analysis Pipeline

The easiest way to get started is to use the master pipeline that handles everything:

```bash
# Run stratified analysis on all experiments + meta-analysis + dashboards
python run_full_analysis_pipeline.py --analyze-all

# Or analyze only experiments that haven't been analyzed yet
python run_full_analysis_pipeline.py --analyze-all --skip-existing
```

This will:
1. Run stratified regression analysis on each experiment
2. Perform cross-experiment meta-analysis
3. Generate interactive HTML dashboards
4. Create a comprehensive summary report

**Outputs**:
- `outputs/experiments/*/stratified_analysis/` - Per-experiment analysis
- `outputs/meta_analysis/` - Cross-experiment results
- `outputs/dashboards/` - Interactive HTML dashboards
- `outputs/analysis_summary_report.txt` - Summary of all experiments

## Analysis Workflow

### Step 1: Run Experiments

#### Single Experiment with Progress Tracking

```bash
python run_experiment_with_tracking.py \
  --dataset twitter \
  --provider openai \
  --model gpt-4o-mini \
  --dataset-size 5000 \
  --n-trials 100
```

This generates 60,000 recommendations (6 prompt styles × 100 trials × 100 posts)

#### Multiple Experiments in Parallel

```bash
python run_all_experiments.py
```

Runs all configured experiments in parallel (edit the script to configure which experiments to run)

### Step 2: Stratified Analysis (Per-Experiment)

```bash
python stratified_analysis.py \
  --experiment-dir outputs/experiments/twitter_openai_gpt-4o-mini
```

**Outputs**:
- `by_style/*.csv` - Logistic regression results per prompt style
- `comparison/coefficient_comparison.csv` - Statistical tests across styles
- `comparison/bias_by_style.csv` - Pool vs recommended bias analysis
- `comparison/feature_importance.csv` - Feature rankings
- `tables/regression_table_publication.csv` - Publication-ready tables

### Step 3: Meta-Analysis (Cross-Experiment)

```bash
python meta_analysis.py \
  --experiments-dir outputs/experiments \
  --output-dir outputs/meta_analysis
```

**Outputs**:
- `by_dataset/dataset_comparison.csv` - ANOVA comparing Twitter/Reddit/Bluesky
- `by_model/model_comparison.csv` - ANOVA comparing OpenAI/Anthropic/Gemini
- `meta_analysis/meta_effect_sizes.csv` - Pooled effects with heterogeneity (I²)
- `meta_analysis_summary.txt` - Human-readable summary

### Step 4: Interactive Dashboards

```bash
# Create all dashboards
python create_dashboards.py --all \
  --experiments-dir outputs/experiments \
  --meta-dir outputs/meta_analysis \
  --output-dir outputs/dashboards

# Or create dashboard for a single experiment
python create_dashboards.py \
  --experiment-dir outputs/experiments/twitter_openai_gpt-4o-mini \
  --output outputs/dashboards/twitter_dashboard.html
```

**Dashboard Features**:
- Interactive filtering and drill-down
- Hover tooltips with full statistics
- Three tabs: Bias by Style, Feature Importance, Coefficient Comparisons
- Cross-experiment dashboard with meta-analysis forest plots

## Advanced Usage

### Running Individual Experiments

#### Using OpenAI:
```bash
export OPENAI_API_KEY='your-key-here'

python run_experiment.py \
  --dataset twitter \
  --provider openai \
  --model gpt-4o-mini \
  --dataset-size 5000 \
  --pool-size 100 \
  --k 10 \
  --n-trials 20
```

#### Using Anthropic Claude:
```bash
export ANTHROPIC_API_KEY='your-key-here'

python run_experiment.py \
  --dataset reddit \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --dataset-size 5000 \
  --pool-size 100 \
  --k 10 \
  --n-trials 20
```

#### Using Google Gemini:
```bash
export GEMINI_API_KEY='your-key-here'

python run_experiment.py \
  --dataset twitter \
  --provider gemini \
  --model gemini-2.0-flash \
  --dataset-size 5000 \
  --pool-size 100 \
  --k 10 \
  --n-trials 20
```

#### Using HuggingFace (Local Model):
```bash
python run_experiment.py \
  --dataset bluesky \
  --provider huggingface \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-size 5000 \
  --pool-size 100 \
  --k 10 \
  --n-trials 20
```

Other HuggingFace models you can use:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-3.2-3B-Instruct`
- Any HuggingFace model with instruct/chat capability

### 5. Analyze Results

```bash
python analyze_experiment.py \
  --results-dir outputs/experiments/twitter_openai_gpt-4o-mini
```

This generates:
- `summary_statistics.csv`: Aggregated bias metrics
- `plots/`: Visualizations comparing prompt styles
  - `sentiment_comparison.png`: Sentiment bias
  - `sentiment-polarity-comparison.png`: Continuous sentiment scores
  - `polarization-score-comparison.png`: Controversy bias
  - `formality-score-comparison.png`: Formality bias
  - `topic_comparison.png`: Topic preferences

## Experiment Configuration

### Command-Line Arguments

```bash
python run_experiment.py --help
```

Key parameters:
- `--dataset`: Dataset to use (`twitter`, `reddit`, `bluesky`)
- `--provider`: LLM provider (`openai`, `anthropic`, `gemini`, `huggingface`)
- `--model`: Model name
- `--dataset-size`: Total posts to load
- `--pool-size`: Posts per recommendation pool
- `--k`: Number of recommendations per trial
- `--n-trials`: Trials per prompt style
- `--styles`: Prompt styles to test (default: all)

### Prompt Styles

The framework tests 6 different prompt styles:
1. **general**: Neutral, interesting to general audience
2. **popular**: Most popular/viral content
3. **engaging**: Maximum engagement (likes, shares)
4. **informative**: Educational/informative content
5. **controversial**: Thought-provoking/debate-generating
6. **neutral**: Simple ranking (baseline)

## Metadata Inference (Preprocessing)

The framework automatically infers metadata from post text:

- **Sentiment**: Polarity, subjectivity, label (positive/negative/neutral)
- **Topics**: Politics, sports, entertainment, technology, health, etc.
- **Style**: Formality, emoji usage, hashtags, mentions
- **Polarization**: Controversial/polarizing content detection

Method: VADER sentiment analysis + keyword-based topic classification

Cached automatically to `outputs/metadata_cache/` for faster re-runs.

## Bias Analysis (Postprocessing)

For each prompt style, the framework compares:
- **Pool distribution**: Content available to LLM
- **Recommended distribution**: Content selected by LLM
- **Differences**: Over/under-representation (percentage points)

Analyzed dimensions:
- Sentiment (positive vs. negative)
- Topics (politics, sports, entertainment, etc.)
- Formality (formal vs. casual)
- Polarization (controversial vs. neutral)
- Emoji usage

## Project Workflow

```
1. Load Dataset
   ↓
2. Infer Metadata (cached)
   ↓
3. For each prompt style:
   - Sample random pool of posts
   - Get LLM recommendations
   - Compare pool vs. recommendations
   ↓
4. Aggregate results across trials
   ↓
5. Generate visualizations
```

## Supported Datasets

### Format

All datasets use persona-based format with:
- `message`: Post text
- `username` / `user_id`: User identifier
- `persona`: User demographic/persona info
- `training`: Train/test split

### Loading Custom Dataset

```python
from data.loaders import PersonaDatasetLoader

loader = PersonaDatasetLoader('./path/to/your/personas.pkl')
df = loader.load(sample_size=5000, training_only=True)
```

## LLM Providers

### OpenAI

```python
from utils.llm_client import get_llm_client

client = get_llm_client('openai', 'gpt-4o-mini')
response = client.generate('Your prompt here')
```

Supported models: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`

### Anthropic

```python
client = get_llm_client('anthropic', 'claude-3-5-sonnet-20241022')
response = client.generate('Your prompt here')
```

Supported models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`

### Google Gemini

```python
client = get_llm_client('gemini', 'gemini-2.0-flash')
response = client.generate('Your prompt here')
```

Supported models:
- **Gemini 3.x**: `gemini-3-pro-preview` (latest, most powerful)
- **Gemini 2.5**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite`
- **Gemini 2.0**: `gemini-2.0-flash` (default), `gemini-2.0-flash-lite`
- **Gemini 1.5**: `gemini-1.5-pro`, `gemini-1.5-flash` (older)

### HuggingFace (Local)

```python
client = get_llm_client(
    'huggingface',
    'meta-llama/Llama-3.1-8B-Instruct',
    device='auto'  # Uses GPU if available
)
response = client.generate('Your prompt here')
```

Supported: Any HuggingFace causal LM with instruct/chat capability

## Statistical Methodology

### Stratified Analysis (Per-Experiment)

For each experiment, we run separate logistic regression models per prompt style:

**Model**: `logit(P(selected=1)) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ`

**Features** (17 total):
- **Sentiment**: polarity, subjectivity, positive, negative
- **Formality**: formality_score
- **Polarization**: polarization_score, controversy_level
- **Structure**: text_length, word_count, avg_word_length, has_url, has_mention, has_hashtag, has_emoji
- **Topic**: has_political_content, has_polarizing_content, misinformation_keywords

**Statistical Tests**:
1. **Wald Test** for coefficient differences between styles: `Z = (β₁ - β₂) / SE(β₁ - β₂)`
2. **T-test** for pool vs recommended bias per feature per style
3. **Multiple testing correction**: Bonferroni (α = 0.05/n_comparisons)

### Meta-Analysis (Cross-Experiment)

**Pooled Effect Sizes** using inverse-variance weighting:

```
θ_pooled = Σ(wᵢ × θᵢ) / Σwᵢ
where wᵢ = 1/SE(θᵢ)²
```

**Heterogeneity Testing**:
- **Cochran's Q statistic**: Tests if effects differ across experiments
- **I² statistic**: Quantifies proportion of variation due to heterogeneity
  - I² < 25%: Low heterogeneity
  - I² 25-75%: Moderate heterogeneity
  - I² > 75%: High heterogeneity

**Dataset/Model Comparisons**: One-way ANOVA on coefficients
- F-statistic tests if means differ across groups
- Post-hoc Tukey HSD for pairwise comparisons

### Power Analysis

With 60,000 observations per experiment:
- **Power** > 0.99 to detect small effects (OR ≥ 1.2) at α = 0.05
- **Precision**: SE(β) ≈ 0.01-0.05 depending on feature prevalence

## Results Structure

```
outputs/experiments/{dataset}_{provider}_{model}/
├── prompt_style_results.pkl   # Raw trial results
├── config.pkl                 # Experiment configuration
├── plots/                     # Visualizations
│   ├── sentiment_comparison.png
│   ├── sentiment-polarity-comparison.png
│   ├── polarization-score-comparison.png
│   ├── formality-score-comparison.png
│   └── topic_comparison.png
└── summary_statistics.csv     # Aggregated metrics
```

## Examples

### Compare Multiple Models

```bash
# Run with GPT-4o-mini
python run_experiment.py --dataset twitter --provider openai --model gpt-4o-mini

# Run with Claude
python run_experiment.py --dataset twitter --provider anthropic --model claude-3-5-sonnet-20241022

# Run with Llama
python run_experiment.py --dataset twitter --provider huggingface --model meta-llama/Llama-3.1-8B-Instruct

# Analyze each
python analyze_experiment.py --results-dir outputs/experiments/twitter_openai_gpt-4o-mini
python analyze_experiment.py --results-dir outputs/experiments/twitter_anthropic_claude-3-5-sonnet-20241022
python analyze_experiment.py --results-dir outputs/experiments/twitter_huggingface_meta-llama_Llama-3.1-8B-Instruct
```

### Test Specific Prompt Styles

```bash
python run_experiment.py \
  --dataset reddit \
  --provider openai \
  --model gpt-4o-mini \
  --styles popular controversial neutral \
  --n-trials 30
```

### Quick Test Run

```bash
python run_experiment.py \
  --dataset twitter \
  --provider openai \
  --model gpt-4o-mini \
  --dataset-size 1000 \
  --pool-size 50 \
  --k 5 \
  --n-trials 5
```

## Troubleshooting

### Dataset not found
Ensure datasets are in `./datasets/{platform}/personas.pkl` format

### API key errors
Set environment variables:
```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
```

### HuggingFace model loading slow
First load downloads model (can take 5-10 minutes for large models). Subsequent runs use cached model.

### Out of memory with HuggingFace
Use smaller models or reduce batch size:
```python
client = get_llm_client('huggingface', 'meta-llama/Llama-3.2-3B-Instruct')
```

## Contributing

When adding new features:
1. Follow existing code structure
2. Add tests to `test_pipeline.py`
3. Update README

## License

See LICENSE file for details.
