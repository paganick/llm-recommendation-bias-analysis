# Data Format Examples

## 1. tweet_level_data.csv

Minimum required format:

```csv
tweet_id,user_id,text,created_at
1001,501,"Just finished reading an amazing book about climate change!",2024-01-15 10:30:00
1002,501,"Coffee tastes better in the morning ☕",2024-01-15 11:45:00
1003,502,"New study shows renewable energy adoption increasing",2024-01-15 12:00:00
```

With optional fields:

```csv
tweet_id,user_id,text,created_at,favorite_count,retweet_count,in_reply_to_status_id_str
1001,501,"Just finished reading...",2024-01-15 10:30:00,42,5,
1002,501,"Coffee tastes better...",2024-01-15 11:45:00,103,12,
1003,502,"New study shows...",2024-01-15 12:00:00,215,48,1001
```

## 2. user_survey_data.csv

Example with demographic variables:

```csv
user_id,gender,age,education,partisanship,ideology,race,income,followers_count,verified
501,female,34,college,democrat,liberal,white,medium,1250,0
502,male,45,graduate,republican,conservative,white,high,3420,1
503,female,28,high_school,independent,moderate,hispanic,low,580,0
```

Minimum required (at least one demographic feature):

```csv
user_id,gender,age
501,female,34
502,male,45
503,female,28
```

## Column Requirements

### tweet_level_data.csv
- **Required**: `tweet_id`, `user_id`, `text` (or `full_text`)
- **Recommended**: `created_at`, `favorite_count`, `retweet_count`
- **Optional**: Reply/retweet/quote fields for tweet type classification

### user_survey_data.csv
- **Required**: `user_id` + at least ONE of:
  - `gender`, `age`, `education`, `partisanship`, `ideology`, `race`, `income`, `religiosity`, `marital_status`
- **Optional**: Twitter metadata (`followers_count`, `friends_count`, `verified`, etc.)

## Data Types

- **user_id**: String or integer (must match across both files)
- **tweet_id**: String or integer (unique)
- **text**: String (tweet content)
- **Demographic variables**: String (categorical) or numeric
- **Counts**: Integer
- **Dates**: ISO format recommended (YYYY-MM-DD HH:MM:SS)

## Notes

1. **Encoding**: Use UTF-8
2. **Missing values**: Use empty string or `NA`
3. **Headers**: Must match column names exactly
4. **File size**: No hard limit, but sampling recommended for >100K tweets
