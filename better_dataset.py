import pandas as pd
import random

# input and output paths
input_file = "dataset/final_dataset.jsonl"
output_file = "dataset/better_dataset.jsonl"

# number of samples
sample_size = 30000

# read the full dataset
print("📂 Loading dataset...")
df = pd.read_json(input_file, lines=True)

print(f"✅ Loaded {len(df):,} rows")

# if dataset is smaller than sample size, adjust automatically
sample_size = min(sample_size, len(df))

# random sample of 30k
print(f"🎲 Sampling {sample_size:,} examples...")
df_sampled = df.sample(n=sample_size, random_state=42)

# save sampled dataset
df_sampled.to_json(output_file, orient="records", lines=True, force_ascii=False)

print(f"💾 Saved sampled dataset to {output_file}")
print(f"✅ Final size: {len(df_sampled):,} examples")
