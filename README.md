# LLM Recommendation Bias Analysis

Analysis framework for evaluating bias in LLM-based recommendation systems using real-world social media data.

## Overview

This project tests whether LLM recommender systems exhibit systematic biases when suggesting content. Unlike simulation-based approaches, this framework uses real-world tweets to perform one-shot recommendation analysis.

## Research Questions

Does the LLM recommender systematically favor:
- Content from specific demographic groups (gender, race)?
- Positive vs. negative sentiment?
- Polished vs. casual writing styles?
- Content with emojis?
- Authors with higher follower counts?
- Specific topics or political positions?

## Project Structure

```
llm-recommendation-bias-analysis/
├── data/                  # Data loading and preprocessing
├── inference/             # Metadata inference (sentiment, topics, demographics, etc.)
├── recommender/           # LLM-based recommendation system
├── analysis/              # Bias analysis and visualization
├── utils/                 # Shared utilities (LLM clients, config)
├── experiments/           # Experimental scenarios and results
└── outputs/              # Generated outputs and reports
```

## Datasets

### TwitterAAE (African American English Twitter Corpus)
- 59M+ tweets with demographic inferences
- Includes user IDs, timestamps, locations, full text
- Demographic probabilities: AA, Hispanic, Other, White
- Use case: Testing demographic bias in recommendations

### DADIT
- Mental health and user classification dataset
- Parquet format with multiple train/test splits
- Use case: TBD based on schema exploration

## Setup

```bash
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Edit config.yaml with your LLM API keys
```

## Usage

### Quick Start

1. **Set up your API key:**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
# or for OpenAI:
# export OPENAI_API_KEY='your-api-key-here'
```

2. **Run the complete analysis:**
```bash
python run_complete_analysis.py --provider anthropic --model claude-3-5-sonnet-20241022
```

This will:
- Load the TwitterAAE dataset
- Infer metadata (sentiment, topics, gender, political leaning, etc.)
- Get LLM recommendations
- Analyze bias in recommendations
- Generate visualizations

### Running a Simple Example

```python
from data.loaders import load_dataset
from utils.llm_client import get_llm_client
from inference.metadata_inference import infer_tweet_metadata
from recommender.llm_recommender import OneShotRecommender
from analysis.bias_analysis import BiasAnalyzer

# 1. Load data
tweets_df = load_dataset('twitteraae', version='all_aa', sample_size=1000)

# 2. Infer metadata
tweets_df = infer_tweet_metadata(
    tweets_df,
    sentiment_method='vader',
    include_gender=True,
    include_political=True
)

# 3. Initialize LLM
llm = get_llm_client(provider='anthropic', model='claude-3-5-sonnet-20241022')

# 4. Get recommendations
recommender = OneShotRecommender(llm, k=10)
pool = tweets_df.sample(n=50)
recommended = recommender.recommend(pool, prompt_style='popular')

# 5. Analyze bias
analyzer = BiasAnalyzer()
results = analyzer.comprehensive_bias_analysis(
    recommended, pool,
    demographic_cols=['demo_aa', 'demo_white', 'demo_hispanic', 'demo_other']
)

print(results['summary'])
```

### Metadata Inference

The framework can infer the following metadata from tweet text:

#### Available Analyzers

1. **Sentiment Analysis** (`SentimentAnalyzer`)
   - Methods: `textblob`, `vader`, `llm`
   - Output: polarity, subjectivity, sentiment label

2. **Topic Classification** (`TopicClassifier`)
   - Methods: `keyword`, `lda`, `llm`
   - Topics: politics, sports, entertainment, technology, health, etc.

3. **Gender Inference** (`GenderAnalyzer`) ⭐ NEW
   - Methods: `keyword`, `llm`
   - Output: male/female/unknown with confidence scores

4. **Political Leaning** (`PoliticalLeaningAnalyzer`) ⭐ NEW
   - Methods: `keyword`, `llm`
   - Output: left/right/center/unknown with confidence scores

5. **Style Analysis** (`StyleAnalyzer`)
   - Detects: emojis, hashtags, mentions, URLs, formality

6. **Polarization Analysis** (`PolarizationAnalyzer`)
   - Measures controversial/polarizing content

#### Using LLM-based Inference

For more accurate inference, use LLM-based methods:

```python
from utils.llm_client import get_llm_client
from inference.metadata_inference import MetadataInferenceEngine

# Initialize LLM client
llm = get_llm_client(provider='anthropic', model='claude-3-haiku-20240307')

# Create inference engine with LLM methods
engine = MetadataInferenceEngine(
    sentiment_method='llm',
    gender_method='llm',
    political_method='llm',
    llm_client=llm
)

# Infer metadata
result = engine.infer("Your tweet text here")
```

### Running Experiments

#### Single Experiment

```bash
python experiments/experiment_runner.py \
    --dataset twitteraae \
    --llm-provider anthropic \
    --llm-model claude-3-5-sonnet-20241022 \
    --prompt-style popular \
    --pool-size 50 \
    --k 10
```

#### Batch Experiments

```python
from experiments.experiment_runner import ExperimentRunner, ExperimentConfig

# Define experiment configurations
configs = [
    ExperimentConfig(
        name='claude_popular',
        llm_provider='anthropic',
        llm_model='claude-3-5-sonnet-20241022',
        prompt_style='popular',
        pool_size=50,
        k=10
    ),
    ExperimentConfig(
        name='claude_controversial',
        llm_provider='anthropic',
        llm_model='claude-3-5-sonnet-20241022',
        prompt_style='controversial',
        pool_size=50,
        k=10
    ),
]

# Run experiments
runner = ExperimentRunner(
    dataset_name='twitteraae',
    dataset_sample_size=5000,
    output_dir='./experiments/results'
)

runner.load_data()
results = runner.run_batch(configs)
runner.generate_visualizations()
```

### Bias Analysis

The framework analyzes bias across multiple dimensions:

- **Demographic Bias**: Over/under-representation of demographic groups
- **Sentiment Bias**: Preference for positive/negative content
- **Topic Bias**: Preference for specific topics
- **Style Bias**: Preference for formal vs. casual language, emojis, etc.
- **Polarization Bias**: Preference for controversial content
- **Gender Bias**: Preference for content from specific genders ⭐ NEW
- **Political Bias**: Preference for specific political leanings ⭐ NEW

#### Bias Metrics

- Statistical tests (t-tests, chi-square)
- Effect sizes (Cohen's d)
- Significance levels (p-values)
- Percentage point differences

### Visualization

Generate comprehensive bias reports:

```python
from analysis.visualization import create_bias_report

create_bias_report(
    bias_results,
    output_dir='./outputs/visualizations',
    system_name='Claude-Popular'
)
```

This generates:
- Comprehensive dashboard with all bias metrics
- Individual plots for each bias dimension
- Demographic bias comparisons
- Sentiment distribution plots
- Topic bias rankings
- Style and polarization visualizations

### Command-Line Options

The `run_complete_analysis.py` script supports many options:

```bash
python run_complete_analysis.py --help

Options:
  --provider {anthropic,openai}
                        LLM provider (default: anthropic)
  --model MODEL         LLM model name (default: claude-3-5-sonnet-20241022)
  --dataset {twitteraae,dadit}
                        Dataset to use (default: twitteraae)
  --dataset-version VERSION
                        Dataset version (default: all_aa)
  --dataset-size SIZE   Number of tweets to load (default: 5000)
  --pool-size SIZE      Number of tweets in recommendation pool (default: 50)
  --k K                 Number of recommendations (default: 10)
  --prompt-style {general,popular,engaging,informative,controversial}
                        Recommendation prompt style (default: popular)
  --include-gender      Include gender inference (default: True)
  --include-political   Include political leaning inference (default: True)
  --output-dir DIR      Output directory (default: ./outputs/complete_analysis)
  --metadata-cache PATH Path to metadata cache file
```

### Advanced Usage

#### Custom Recommender Prompts

```python
from recommender.llm_recommender import OneShotRecommender

recommender = OneShotRecommender(llm, k=10)

# Use different prompt styles
recommended = recommender.recommend(pool, prompt_style='controversial')
```

Available prompt styles:
- `general`: Interesting tweets for general audience
- `popular`: Tweets likely to be popular/viral
- `engaging`: Tweets that generate engagement
- `informative`: Educational/informative tweets
- `controversial`: Thought-provoking/debate-generating tweets

#### Comparing Multiple Systems

```python
from analysis.bias_analysis import compare_recommender_bias

# Run experiments with different models
results_claude = run_experiment(llm='claude-3-5-sonnet-20241022')
results_gpt = run_experiment(llm='gpt-4')

# Compare bias
comparison = compare_recommender_bias(
    [results_claude, results_gpt],
    labels=['Claude', 'GPT-4']
)

print(comparison)
```

## Citation

If you use the TwitterAAE dataset, please cite:
```
S. Blodgett, L. Green, and B. O'Connor. Demographic Dialectal Variation in Social Media:
A Case Study of African-American English. Proceedings of EMNLP. Austin. 2016.
```
