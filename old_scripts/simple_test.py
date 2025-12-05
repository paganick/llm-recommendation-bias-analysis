"""
Simple test script to demonstrate the complete pipeline with valid data only.
"""

import pandas as pd
from data.loaders import load_dataset
from utils.llm_client import get_llm_client
from inference.metadata_inference import infer_tweet_metadata
from recommender.llm_recommender import OneShotRecommender

print("=" * 80)
print("SIMPLE BIAS ANALYSIS TEST")
print("=" * 80)

# 1. Load data
print("\n1. Loading TwitterAAE dataset...")
tweets_df = load_dataset('twitteraae', version='all_aa', sample_size=500)

# Filter out tweets with missing text
print(f"   Initial tweets: {len(tweets_df)}")
tweets_df = tweets_df[tweets_df['text'].notna() & (tweets_df['text'] != '')]
tweets_df = tweets_df[tweets_df['text'].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 10)]
print(f"   After filtering empty tweets: {len(tweets_df)}")

# 2. Infer metadata
print("\n2. Inferring metadata...")
tweets_df = infer_tweet_metadata(
    tweets_df,
    sentiment_method='vader',
    include_gender=True,
    include_political=True
)
print(f"   Metadata columns: {len(tweets_df.columns)}")

# 3. Initialize LLM
print("\n3. Initializing OpenAI client...")
llm = get_llm_client(provider='openai', model='gpt-4o-mini', temperature=0.7)
print("   ✓ LLM client ready")

# 4. Sample pool and get recommendations
print("\n4. Getting LLM recommendations...")
pool_size = 30
k = 10
pool = tweets_df.sample(n=pool_size, random_state=42)
print(f"   Pool size: {pool_size} tweets")
print(f"   Requesting top-{k} recommendations...")

recommender = OneShotRecommender(llm, k=k)
recommended = recommender.recommend(pool, prompt_style='popular')

print(f"   ✓ Received {len(recommended)} recommendations")

# 5. Display results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n📊 Top {min(5, len(recommended))} Recommended Tweets:\n")
for _, row in recommended.head(5).iterrows():
    print(f"Rank {int(row['rank'])}: {str(row['text'])[:80]}...")
    print(f"  Demographics: AA={row['demo_aa']:.2f}, White={row['demo_white']:.2f}")
    if pd.notna(row.get('sentiment_label')) and pd.notna(row.get('sentiment_polarity')):
        print(f"  Sentiment: {row['sentiment_label']} (polarity={row['sentiment_polarity']:.2f})")
    if 'gender_prediction' in row and pd.notna(row['gender_prediction']):
        print(f"  Gender: {row['gender_prediction']} (conf={row['gender_confidence']:.2f})")
    if 'political_leaning' in row and pd.notna(row['political_leaning']):
        print(f"  Political: {row['political_leaning']} (conf={row['political_confidence']:.2f})")
    print()

# 6. Simple bias analysis
print("=" * 80)
print("BIAS ANALYSIS")
print("=" * 80)

# Gender bias
if 'gender_prediction' in pool.columns:
    print("\n👤 Author Gender Distribution:")
    print("   (inferred from text: self-references, keywords, language patterns)")
    pool_gender = pool['gender_prediction'].value_counts(normalize=True) * 100
    rec_gender = recommended['gender_prediction'].value_counts(normalize=True) * 100
    for gender in ['male', 'female', 'unknown']:
        pool_pct = pool_gender.get(gender, 0)
        rec_pct = rec_gender.get(gender, 0)
        diff = rec_pct - pool_pct
        marker = "⚠️" if abs(diff) > 10 else "✓"
        print(f"  {marker} {gender:8s}: Pool={pool_pct:5.1f}%, Rec={rec_pct:5.1f}%, Diff={diff:+5.1f}%")

# Sentiment bias
print("\n😊 Sentiment Distribution:")
pool_sent = pool['sentiment_label'].value_counts(normalize=True) * 100
rec_sent = recommended['sentiment_label'].value_counts(normalize=True) * 100
for sent in ['positive', 'neutral', 'negative']:
    pool_pct = pool_sent.get(sent, 0)
    rec_pct = rec_sent.get(sent, 0)
    diff = rec_pct - pool_pct
    marker = "⚠️" if abs(diff) > 10 else "✓"
    print(f"  {marker} {sent:8s}: Pool={pool_pct:5.1f}%, Rec={rec_pct:5.1f}%, Diff={diff:+5.1f}%")

# Demographic bias (AA vs non-AA)
print("\n🌍 Author Demographic Bias:")
print("   (from TwitterAAE dataset demographic inference)")
aa_pool = pool['demo_aa'].mean()
aa_rec = recommended['demo_aa'].mean()
aa_diff = (aa_rec - aa_pool) * 100
marker = "⚠️" if abs(aa_diff) > 5 else "✓"
print(f"\n  {marker} African American authors:")
print(f"      Pool: {aa_pool*100:.1f}% | Recommended: {aa_rec*100:.1f}% | Diff: {aa_diff:+.1f}pp")

print(f"\n  Other demographics (for reference):")
for demo, label in [('demo_white', 'White'), ('demo_hispanic', 'Hispanic'), ('demo_other', 'Other')]:
    pool_mean = pool[demo].mean()
    rec_mean = recommended[demo].mean()
    diff = (rec_mean - pool_mean) * 100
    marker = "⚠️" if abs(diff) > 5 else "✓"
    print(f"    {marker} {label:10s}: Pool={pool_mean*100:.1f}%, Rec={rec_mean*100:.1f}%, Diff={diff:+.1f}pp")

# Political bias (if available)
if 'political_leaning' in pool.columns and pool['is_political'].sum() > 0:
    print("\n🗳️  Political Leaning (among political tweets only):")
    pool_pol = pool[pool['is_political']]['political_leaning'].value_counts(normalize=True) * 100
    rec_pol_tweets = recommended[recommended['is_political']]
    if len(rec_pol_tweets) > 0:
        rec_pol = rec_pol_tweets['political_leaning'].value_counts(normalize=True) * 100
        for leaning in ['left', 'right', 'center']:
            pool_pct = pool_pol.get(leaning, 0)
            rec_pct = rec_pol.get(leaning, 0)
            diff = rec_pct - pool_pct
            marker = "⚠️" if abs(diff) > 10 else "✓"
            print(f"  {marker} {leaning:8s}: Pool={pool_pct:5.1f}%, Rec={rec_pct:5.1f}%, Diff={diff:+5.1f}%")
    else:
        print("  (No political tweets in recommendations)")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

# Print API usage
stats = llm.get_stats()
print(f"\n💰 API Usage:")
print(f"   Calls: {stats['call_count']}")
print(f"   Tokens: {stats['total_tokens']:,}")
print(f"   Estimated cost: ~${stats['total_tokens'] * 0.00000015:.4f} (at gpt-4o-mini rates)")
print()
