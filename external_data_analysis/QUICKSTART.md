# Quick Start Guide - Twitter Survey Data Analysis

## Step-by-Step Instructions

### 1. Installation

```bash
cd external_data_analysis
pip install -r requirements.txt
```

### 2. Prepare Your API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Prepare Your Data

You need TWO CSV files:
- `tweet_data.csv` - Tweet content and metadata
- `user_survey_data.csv` - User demographics and survey responses

See `docs/data_format_examples.md` for required format.

### 4. Run Data Preparation

```bash
python scripts/1_prepare_survey_data.py \
    --tweets /path/to/tweet_data.csv \
    --users /path/to/user_survey_data.csv \
    --output data/prepared_posts.csv
```

This creates `data/prepared_posts.csv` with all features extracted.

### 5. Run LLM Experiments

For each provider you want to test:

```bash
# OpenAI
python scripts/2_run_recommendations.py \
    --input data/prepared_posts.csv \
    --provider openai \
    --output outputs/experiments/

# Anthropic  
python scripts/2_run_recommendations.py \
    --input data/prepared_posts.csv \
    --provider anthropic \
    --output outputs/experiments/

# Gemini
python scripts/2_run_recommendations.py \
    --input data/prepared_posts.csv \
    --provider gemini \
    --output outputs/experiments/
```

**Note**: Each provider takes ~30-60 minutes to run 600 trials (100 per prompt style × 6 styles).

### 6. Analyze Results

```bash
cd ..  # Go back to main project directory
python run_comprehensive_analysis.py
```

This will analyze all experiments in `outputs/experiments/` and generate visualizations.

## Output Structure

After running, you'll have:

```
outputs/
├── experiments/
│   ├── survey_openai_gpt-4o-mini/
│   │   └── post_level_data.csv
│   ├── survey_anthropic_claude-sonnet-4/
│   │   └── post_level_data.csv
│   └── survey_gemini_gemini-2.0-flash/
│       └── post_level_data.csv
└── analysis/
    └── visualizations/
        ├── 1_distributions/
        ├── 2_bias_heatmaps/
        ├── 3_directional_bias/
        └── 4_feature_importance/
```

## Customization

### Adjust Feature Types

If your data uses different formats (e.g., ideology as 1-7 scale instead of categories), edit `scripts/1_prepare_survey_data.py`:

```python
FEATURE_TYPES = {
    'author_ideology': 'numerical',  # Changed from 'categorical'
    ...
}
```

### Change Column Names

If your columns have different names, edit the `COLUMN_MAPPING` dictionary:

```python
COLUMN_MAPPING = {
    'party': 'author_partisanship',  # Your column → standard name
    ...
}
```

## Troubleshooting

See README.md for detailed troubleshooting guide.

## Questions?

Check:
1. Main README.md - Full documentation
2. docs/data_format_examples.md - Data format requirements  
3. outputs/preprocessing.log - Data preparation logs
4. outputs/experiment.log - Experiment logs
