# LLM Recommendation Bias Analysis

Framework for analyzing bias in LLM-based recommendation systems across multiple social media platforms and models.

## Overview

This project evaluates whether LLM recommender systems exhibit systematic biases when suggesting content. It supports multiple datasets (Twitter, Reddit, Bluesky) and multiple LLM providers (OpenAI, Anthropic, HuggingFace local models).

## Key Features

- **Multiple Datasets**: Twitter, Reddit, Bluesky
- **Multiple Models**: OpenAI GPT, Anthropic Claude, HuggingFace models (Llama, Mistral, etc.)
- **Prompt Style Comparison**: Test how different prompt framings affect recommendations
- **Comprehensive Bias Analysis**: Sentiment, topics, style, polarization
- **Automated Metadata Inference**: Extract metadata from post text

## Project Structure

```
llm-recommendation-bias-analysis/
├── data/                  # Data loading
├── inference/             # Metadata inference (preprocessing)
├── utils/                 # LLM clients
├── analysis/              # Bias analysis (postprocessing)
├── datasets/              # Local dataset copies (not committed)
├── outputs/               # Experiment results
├── run_experiment.py      # Main experiment runner
├── analyze_experiment.py  # Results analysis
└── test_pipeline.py       # Pipeline tests
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

### 4. Run Experiment

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
- `--provider`: LLM provider (`openai`, `anthropic`, `huggingface`)
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
