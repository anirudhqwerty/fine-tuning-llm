import json
import pandas as pd
from collections import Counter
import re

# Load your dataset
input_file = "dataset/better_dataset.jsonl"

print("🔍 DATASET INSPECTION")
print("="*60)

# Read dataset
df = pd.read_json(input_file, lines=True)
print(f"\n✅ Loaded {len(df):,} examples\n")

# Sample 10 random examples
print("📋 Random Sample of Examples:")
print("-"*60)
for i, row in df.sample(10).iterrows():
    print(f"\n[Example {i}]")
    print(f"Prompt: {row['prompt'][:100]}...")
    print(f"Response: {row['response'][:150]}...")
    print("-"*60)

# Check for contamination patterns
print("\n🚨 CONTAMINATION CHECK")
print("="*60)

contamination_patterns = {
    'Twitter handles (@)': r'@\w+',
    'Hashtags (#)': r'#\w+',
    'URLs': r'https?://\S+',
    'Emojis': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
    'Email addresses': r'\S+@\S+\.\S+',
    'Phone numbers': r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
    'Excessive caps': r'\b[A-Z]{5,}\b',
}

contaminated_examples = []

for pattern_name, pattern in contamination_patterns.items():
    count = 0
    examples = []
    
    for idx, row in df.iterrows():
        text = row['prompt'] + " " + row['response']
        matches = re.findall(pattern, text)
        if matches:
            count += 1
            if len(examples) < 3:  # Store first 3 examples
                examples.append((idx, matches[:5]))  # First 5 matches
    
    if count > 0:
        contaminated_examples.append(pattern_name)
        print(f"\n⚠️  {pattern_name}: Found in {count:,} examples ({count/len(df)*100:.1f}%)")
        if examples:
            print(f"   Examples:")
            for idx, matches in examples:
                print(f"     • Row {idx}: {matches}")

# Check response length distribution
print("\n\n📊 RESPONSE LENGTH ANALYSIS")
print("="*60)
df['response_length'] = df['response'].str.len()
print(f"Average: {df['response_length'].mean():.0f} chars")
print(f"Median: {df['response_length'].median():.0f} chars")
print(f"Min: {df['response_length'].min()} chars")
print(f"Max: {df['response_length'].max()} chars")

# Check for very long responses (likely spam)
long_responses = df[df['response_length'] > 500]
if len(long_responses) > 0:
    print(f"\n⚠️  {len(long_responses):,} responses are suspiciously long (>500 chars)")
    print("\nExample of long response:")
    print(long_responses.iloc[0]['response'][:300] + "...")

# Check for repeated names
print("\n\n🔍 REPEATED NAME PATTERNS")
print("="*60)
all_text = " ".join(df['response'].tolist())
# Find capitalized words that might be names
potential_names = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', all_text)
name_counts = Counter(potential_names)

# Show most common capitalized words
print("\nMost common capitalized words (potential names/spam):")
for name, count in name_counts.most_common(20):
    if count > 10:  # Only show if appears more than 10 times
        print(f"  • {name}: {count:,} times")

# Check for empty or very short responses
print("\n\n⚠️  QUALITY ISSUES")
print("="*60)
empty_responses = df[df['response'].str.strip() == '']
very_short = df[df['response_length'] < 10]
print(f"Empty responses: {len(empty_responses)}")
print(f"Very short responses (<10 chars): {len(very_short)}")

# Save contaminated examples for review
if contaminated_examples:
    print("\n\n💾 Saving contaminated examples for review...")
    contaminated_df = df[df.apply(
        lambda row: any(re.search(pattern, row['prompt'] + " " + row['response']) 
                       for pattern in contamination_patterns.values()),
        axis=1
    )]
    
    contaminated_df.head(100).to_json(
        "dataset/contaminated_examples.jsonl", 
        orient="records", 
        lines=True
    )
    print(f"✅ Saved {min(100, len(contaminated_df))} examples to dataset/contaminated_examples.jsonl")

print("\n\n" + "="*60)
print("🎯 RECOMMENDATION")
print("="*60)

if contaminated_examples:
    print("\n⚠️  Your dataset is CONTAMINATED!")
    print(f"   Issues found: {', '.join(contaminated_examples)}")
    print("\n   You need to CLEAN your data before training:")
    print("   1. Remove social media artifacts (@mentions, hashtags)")
    print("   2. Remove URLs and emojis")
    print("   3. Filter out spam responses with random names")
    print("   4. Keep only legitimate customer support examples")
else:
    print("\n✅ Dataset looks clean!")
    print("   The problem might be in the training process.")