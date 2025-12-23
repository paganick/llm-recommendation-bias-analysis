# Twitter Survey Data - LLM Recommendation Bias Analysis

Complete pipeline for running LLM recommendation experiments and bias analysis on Twitter survey data.

## Overview

This module allows you to:
1. **Prepare** your Twitter survey data for LLM recommendation experiments
2. **Run** LLM recommendation experiments (OpenAI/Anthropic/Gemini)
3. **Analyze** bias across demographic and behavioral features

## What You Need

### 1. Your Data Files

**Required files** (CSV format):

`user_survey_data.csv` - One row per user with columns:
- `user_id` - Twitter user ID (must match Twitter data)
- Demographic variables (see Feature Assumptions section)

`tweet_data.csv` - One row per tweet with columns:
- `tweet_id` - Unique tweet ID
- `user_id` - Author's user ID
- `text` or `full_text` - Tweet content
- `created_at` - Timestamp
- (Optional) `favorite_count`, `retweet_count`, reply/retweet fields

### 2. API Keys

You'll need API keys for LLM providers you want to test:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google Gemini**: https://aistudio.google.com/app/apikey

## Quick Start

```bash
# 1. Set up environment
cd external_data_analysis
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# 3. Prepare data
python scripts/1_prepare_survey_data.py \
  --tweets ../path/to/tweet_data.csv \
  --users ../path/to/user_survey_data.csv \
  --output data/prepared_posts.csv

# 4. Run LLM recommendation experiments
python scripts/2_run_recommendations.py \
  --input data/prepared_posts.csv \
  --provider openai \
  --output outputs/experiments/

# 5. Analyze bias
python scripts/3_analyze_bias.py \
  --input outputs/experiments/survey_openai_*/post_level_data.csv \
  --output outputs/analysis/
```

## Feature Assumptions

The pipeline will extract these features from your data:

### Demographic Features (from survey)

**Assumed CATEGORICAL:**
- `gender` → `author_gender` (e.g., "male", "female", "other")
- `partisanship` → `author_partisanship` (e.g., "democrat", "republican", "independent")
- `race` → `author_race` (e.g., "white", "black", "hispanic", "asian", "other")
- `education` → `author_education` (e.g., "high_school", "college", "graduate")
- `marital_status` → `author_marital_status`
- `religiosity` → `author_religiosity` (e.g., "very_religious", "somewhat", "not_religious")

**Assumed ORDINAL (treated as categorical):**
- `ideology` → `author_ideology` (e.g., 1-7 scale: "very_liberal" to "very_conservative")
- `income` → `author_income` (e.g., "low", "medium", "high" or income brackets)

**Assumed NUMERICAL:**
- `age` → `author_age` (years)

### Tweet Features (computed)

**Text metrics (numerical):**
- `text_length` - Character count
- `avg_word_length` - Average word length
- `word_count` - Number of words

**Style (binary):**
- `has_url` - Contains URL
- `has_hashtag` - Contains #hashtag
- `has_mention` - Contains @mention
- `has_emoji` - Contains emoji

**Sentiment (numerical):**
- `sentiment_polarity` - [-1, 1] negative to positive
- `sentiment_subjectivity` - [0, 1] objective to subjective

**Tweet type (binary):**
- `is_reply` - Is a reply to another tweet
- `is_retweet` - Is a retweet
- `is_quote` - Is a quote tweet

### User Metadata (if available)

**Numerical:**
- `user_followers_count` - Number of followers
- `user_friends_count` - Number following
- `user_statuses_count` - Total tweets posted
- `user_account_age_days` - Days since account creation

**Binary:**
- `user_verified` - Verified account status

### Engagement (numerical):
- `engagement_score` - log(favorite_count + retweet_count + 1)

## ⚠️ IMPORTANT: Adjust Feature Types

**If your data uses different formats**, edit `scripts/1_prepare_survey_data.py`:

```python
# At the top of the file, find FEATURE_TYPES dictionary:

FEATURE_TYPES = {
    'author_gender': 'categorical',
    'author_ideology': 'categorical',  # ← Change to 'numerical' if using scale
    'author_income': 'categorical',    # ← Change to 'numerical' if using dollar amounts
    # ... etc
}
```

**Column name mapping**: If your columns have different names, edit the `COLUMN_MAPPING` dictionary:

```python
COLUMN_MAPPING = {
    'gender': 'author_gender',
    'party': 'author_partisanship',  # ← If your column is "party" not "partisanship"
    # ... etc
}
```

## Step-by-Step Instructions

### Step 1: Data Preparation

The preparation script will:
1. Merge tweet and user data
2. Extract all features listed above
3. Create persona descriptions for each user
4. Save prepared data in experiment format

```bash
python scripts/1_prepare_survey_data.py \
  --tweets /path/to/tweet_data.csv \
  --users /path/to/user_survey_data.csv \
  --output data/prepared_posts.csv \
  --sample_size 10000  # Optional: sample N tweets per experiment
```

**Output**: `data/prepared_posts.csv` with all features extracted

### Step 2: API Configuration

Create `.env` file with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=...
```

### Step 3: Run Recommendation Experiments

Run for each provider you want to test:

```bash
# OpenAI
python scripts/2_run_recommendations.py \
  --input data/prepared_posts.csv \
  --provider openai \
  --model gpt-4o-mini \
  --output outputs/experiments/

# Anthropic
python scripts/2_run_recommendations.py \
  --input data/prepared_posts.csv \
  --provider anthropic \
  --model claude-sonnet-4 \
  --output outputs/experiments/

# Gemini
python scripts/2_run_recommendations.py \
  --input data/prepared_posts.csv \
  --provider gemini \
  --model gemini-2.0-flash \
  --output outputs/experiments/
```

**What it does**:
- For each of 6 prompt styles (general, popular, engaging, informative, controversial, neutral)
- Run 100 trials, each selecting 1 post from 10 random posts
- Save results with all features to `outputs/experiments/survey_<provider>_<model>/`

**Runtime**: ~30-60 minutes per provider (depending on API rate limits)

### Step 4: Analyze Bias

```bash
python scripts/3_analyze_bias.py \
  --experiments outputs/experiments/*/post_level_data.csv \
  --output outputs/analysis/
```

**Outputs**:
- `outputs/analysis/visualizations/1_distributions/` - Feature distributions
- `outputs/analysis/visualizations/2_bias_heatmaps/` - Bias magnitude (Cramér's V, Cohen's d)
- `outputs/analysis/visualizations/3_directional_bias/` - Which categories favored
- `outputs/analysis/visualizations/4_feature_importance/` - Random Forest + SHAP analysis

## Understanding Your Results

### Bias Heatmaps
- **High values** (red): Feature strongly associated with recommendations
- **Low values** (white): Feature not associated with recommendations
- **Significance markers**: `*` p<0.05, `**` p<0.01, `***` p<0.001

### Directional Bias
- **Positive bars**: Category over-represented in recommendations
- **Negative bars**: Category under-represented
- Example: `author_gender=female: +0.05` means females are 5% more likely to be recommended

### Feature Importance
- **SHAP values**: How much each feature helps predict recommendations
- **High AUROC** (>0.8): Model can strongly predict what gets recommended based on features
- **Top features**: Most important predictors of recommendation

## Data Format Examples

See `docs/data_format_examples.md` for detailed examples of required data formats.

## Troubleshooting

### API Errors

**"Invalid API key"**: Check your `.env` file has correct keys

**"Rate limit exceeded"**: Add delays between requests:
```bash
python scripts/2_run_recommendations.py --delay 1.0  # 1 second between requests
```

### Data Preparation Errors

**"Column 'user_id' not found"**: Ensure both CSVs have `user_id` column

**"Merge resulted in 0 rows"**: Check `user_id` values match between files

**"No features extracted"**: Check column names match assumptions (see Feature Assumptions section)

### Analysis Errors

**"Not enough variance"**: You need both selected (1) and non-selected (0) tweets

**"Feature type mismatch"**: Adjust `FEATURE_TYPES` in `scripts/1_prepare_survey_data.py`

## Advanced Options

### Custom Prompt Styles

Edit `scripts/2_run_recommendations.py` to add custom prompts:

```python
CUSTOM_PROMPTS = {
    'my_style': 'Select posts that are [your criteria]'
}
```

### Additional Features

To add custom features, edit `scripts/1_prepare_survey_data.py`:

```python
def extract_custom_features(df):
    # Add your feature extraction code
    df['my_feature'] = ...
    return df
```

## Questions?

1. Check `docs/` for detailed documentation
2. Review example data in `docs/data_format_examples.md`
3. Check logs in `outputs/preprocessing.log` and `outputs/experiment.log`
