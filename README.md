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

TBD - Under active development

## Citation

If you use the TwitterAAE dataset, please cite:
```
S. Blodgett, L. Green, and B. O'Connor. Demographic Dialectal Variation in Social Media:
A Case Study of African-American English. Proceedings of EMNLP. Austin. 2016.
```
