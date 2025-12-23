# Example Data Files

These are small fake datasets for testing the pipeline.

## Files

- `example_tweet_data.csv` - 20 fake tweets from 10 users
- `example_user_survey_data.csv` - 10 fake users with demographics
- `prepared_example.csv` - Output from preprocessing script
- `feature_metadata.csv` - List of all extracted features

## Test the Pipeline

```bash
cd external_data_analysis

# Run preprocessing
python scripts/1_prepare_survey_data.py \
  --tweets data/examples/example_tweet_data.csv \
  --users data/examples/example_user_survey_data.csv \
  --output data/test_output.csv

# Check output
head data/test_output.csv
wc -l data/test_output.csv  # Should be 21 (header + 20 rows)
```

## Features Extracted

The preprocessing extracted 24 features from the example data:

**Demographics (9):**
- author_gender, author_age, author_ideology, author_partisanship
- author_race, author_income, author_education, author_marital_status, author_religiosity

**Tweet Metrics (13):**
- text_length, word_count, avg_word_length
- has_url, has_hashtag, has_mention, has_emoji
- sentiment_polarity, sentiment_subjectivity (set to 0 if TextBlob not installed)
- is_reply, is_retweet, is_quote
- engagement_score

**User Metadata (4):**
- user_followers_count, user_friends_count, user_statuses_count, user_verified
- user_account_age_days

## Example Output Row

See `prepared_example.csv` for the full preprocessed dataset ready for LLM experiments.

