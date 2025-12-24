# Test/Fake Data - DO NOT USE FOR RESEARCH

⚠️ **WARNING**: All data in this directory is FAKE/TEST DATA generated for demonstration purposes only.

## Contents

### `experiments/survey_gemini_gemini-2.0-flash/`
- **Status**: FAKE DATA
- **Purpose**: Demonstrates the expected output format from running recommendations
- **How it was created**: Generated using fake survey data for testing the pipeline
- **DO NOT USE**: This data should not be used for any research, analysis, or publication

### `analysis_full/`
- **Status**: FAKE ANALYSIS RESULTS
- **Purpose**: Shows the complete set of visualizations (87 total) that the analysis pipeline generates
- **Based on**: The fake experiment data above
- **DO NOT USE**: These are example outputs only

## Using With Real Data

To analyze your actual survey data:

1. **Prepare your real survey data**:
   ```bash
   python scripts/1_prepare_survey_data.py \
     --tweets /path/to/real/tweet_data.csv \
     --users /path/to/real/user_survey_data.csv \
     --output data/prepared_posts.csv
   ```

2. **Run recommendations** (this will replace the fake data):
   ```bash
   python scripts/2_run_recommendations.py \
     --input data/prepared_posts.csv \
     --provider gemini \
     --output outputs/experiments \
     --trials 100
   ```

3. **Generate analysis** (this will replace the fake visualizations):
   ```bash
   bash scripts/run_full_survey_analysis.sh
   ```

The fake data will be overwritten with real results.

## What to Expect

When you run the analysis on real data, you'll get:
- **87 visualizations** showing bias patterns in your survey data
- **4 directories**: 1_distributions, 2_bias_heatmaps, 3_directional_bias, 4_feature_importance
- **Auto-detected features**: All features from your survey data will be analyzed

See `../USAGE.md` for detailed instructions.
