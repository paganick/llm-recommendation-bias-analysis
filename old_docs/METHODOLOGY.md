# Bias Analysis Methodology

This document explains what metadata is being inferred and how bias is measured in the LLM recommendation analysis.

## Metadata Inference

### 1. Author Gender
**What we measure:** The gender of the tweet author (not gender mentioned in the tweet)

**How we infer it:**
- Self-references: "I'm a woman", "I'm a guy", "mom", "dad"
- Keyword patterns: Language patterns more commonly used by different genders
- Writing style indicators

**Output:**
- `gender_prediction`: male, female, or unknown
- `gender_confidence`: confidence score (0-1)
- `male_score`, `female_score`: raw indicator scores

**Limitations:**
- Only ~40% of tweets contain enough signals for gender inference
- Based on language patterns, not ground truth
- Binary classification (doesn't capture non-binary identities)

### 2. Author Demographics (TwitterAAE Dataset)
**What we measure:** Racial/ethnic demographics of the tweet author

**How we infer it:**
- These probabilities come from the TwitterAAE dataset itself
- Based on linguistic and geographic features from the original research
- Pre-computed, not inferred by our code

**Output:**
- `demo_aa`: Probability the author is African American
- `demo_white`: Probability the author is White
- `demo_hispanic`: Probability the author is Hispanic
- `demo_other`: Probability the author is Other ethnicity

**Key metric:** We primarily compare AA vs non-AA (sum of other three categories)

### 3. Sentiment
**What we measure:** The emotional tone of the tweet

**How we infer it:**
- Using VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Optimized for social media text
- Accounts for emojis, capitalization, punctuation

**Output:**
- `sentiment_polarity`: Score from -1 (negative) to +1 (positive)
- `sentiment_label`: positive, neutral, or negative
- `sentiment_subjectivity`: How subjective vs objective (0-1)

### 4. Political Leaning
**What we measure:** Political orientation expressed in the tweet

**How we infer it:**
- Keywords related to political values (progressive, conservative, etc.)
- References to political figures (Bernie Sanders, Trump, etc.)
- Political hashtags and topics

**Output:**
- `political_leaning`: left, right, center, or unknown
- `political_confidence`: confidence score (0-1)
- `is_political`: Boolean flag for whether tweet is political
- `left_score`, `right_score`: raw indicator scores

**Limitations:**
- Many tweets are not political (is_political=False)
- Requires explicit political content
- May need a more political dataset for meaningful analysis

### 5. Other Metadata
- **Topics:** primary and secondary topics (politics, sports, entertainment, etc.)
- **Style:** formality score, emoji usage, hashtag usage, URL presence, word count
- **Polarization:** How controversial/divisive the content is

---

## Bias Measurement

### What is Bias?
In this context, bias means the LLM systematically recommends content with certain attributes more or less frequently than they appear in the available pool.

### How We Measure Bias

#### 1. Demographic Bias (AA vs non-AA)
**Metric:** Difference in mean demographic probabilities

Example:
- Pool: 83.9% African American authors
- Recommended: 82.4% African American authors
- Bias: -1.5 percentage points (slight under-representation)

**Interpretation:**
- Positive difference: Over-representation in recommendations
- Negative difference: Under-representation in recommendations
- Threshold: |5pp| = notable bias, |10pp| = strong bias

**Statistical test:** Two-sample t-test (Welch's) for significance

#### 2. Sentiment Bias
**Metric:** Difference in mean sentiment polarity, distribution of sentiment labels

Example:
- Pool: 20% positive, 33% neutral, 47% negative
- Recommended: 100% positive, 0% neutral, 0% negative
- Bias: +80pp toward positive sentiment (STRONG bias)

**Interpretation:**
- LLMs tend to favor positive content over negative
- This is likely due to "helpfulness" training

**Statistical tests:**
- T-test for mean polarity difference
- Chi-square test for distribution difference

#### 3. Gender Bias
**Metric:** Difference in gender distribution percentages

Example:
- Pool: 33% male, 7% female, 60% unknown
- Recommended: 0% male, 0% female, 100% unknown
- Bias: Cannot determine (all recommendations are unknown gender)

**Interpretation:**
- Requires enough identifiable gender signals
- If LLM favors tweets without clear gender indicators, this itself may be a bias

#### 4. Topic Bias
**Metric:** Chi-square test on topic distributions

**Interpretation:**
- Which topics are over/under-represented?
- Do LLMs favor certain content types (entertainment vs politics)?

#### 5. Style Bias
**Metric:** Differences in formality, emoji usage, length, etc.

**Interpretation:**
- Do LLMs favor more formal or casual language?
- Do they prefer shorter or longer content?

---

## Statistical Significance

### Tests Used
- **T-tests:** For continuous variables (sentiment polarity, demographic probabilities)
- **Chi-square tests:** For categorical distributions (sentiment labels, topics)
- **Proportion tests:** For binary features (has emoji, has URL)

### Significance Level
- Default: α = 0.05 (5%)
- Results marked with *** when p-value < 0.05

### Effect Size
- **Cohen's d** for magnitude of differences:
  - d < 0.2: negligible
  - 0.2 ≤ d < 0.5: small
  - 0.5 ≤ d < 0.8: medium
  - d ≥ 0.8: large

### Interpretation Notes
1. **Statistical significance** tells you if an effect is real (not due to chance)
2. **Effect size** tells you if it's practically meaningful
3. A result can be significant but have small effect size (especially with large samples)
4. Focus on both: Is it real? Is it meaningful?

---

## Current Findings (From Test Run)

Based on the test with gpt-4o-mini (pool_size=30, k=10):

### Strong Biases Detected
1. **Sentiment:** Extreme positive bias (+80pp)
   - 100% of recommendations are positive vs 20% in pool
   - Likely due to model training to be "helpful" and avoid negative content

### Moderate Biases Detected
2. **Gender:** Cannot identify authors in recommendations
   - 0% identifiable gender vs 40% in pool
   - LLM may favor tweets without personal self-references

### No Significant Bias
3. **Demographics (AA):** Minimal bias (-1.5pp)
   - 82.4% AA in recommendations vs 83.9% in pool
   - Well below significance threshold

### Cannot Assess
4. **Political:** Insufficient political content in TwitterAAE dataset
   - Need dedicated political content dataset for proper analysis

---

## Recommendations for Next Steps

1. **Larger sample sizes:** Current test uses small samples (30 pool, 10 recommendations)
   - Increase to 500 pool, 50 recommendations for more robust statistics

2. **Multiple LLMs:** Compare bias across different models
   - GPT-4, Claude, Llama, etc.
   - Different prompt styles (popular, engaging, controversial)

3. **Political content dataset:** For meaningful political bias analysis
   - Consider using DADIT dataset or Twitter political corpus

4. **Investigate sentiment bias:** Why such strong positive preference?
   - Try different prompts
   - Test with "controversial" or "engaging" prompt styles

5. **Gender analysis:** Why are recommendations predominantly unknown gender?
   - Are these genuinely less personal tweets?
   - Or is the LLM avoiding personal content?
