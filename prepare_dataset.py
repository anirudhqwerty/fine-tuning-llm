from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Paths
dataset_file = "dataset/better_dataset.jsonl"
output_dir = "dataset/tokenized_customer_support"
tokenizer_dir = "dataset/tokenizer"

print("📂 Loading dataset...")
dataset = load_dataset("json", data_files=dataset_file, split="train")
print(f"✅ Loaded {len(dataset):,} examples")

# Load and configure tokenizer
print("\n🔧 Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# CRITICAL FIX: Use eos_token as pad_token (standard for GPT-2)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   • Set pad_token = eos_token ('{tokenizer.eos_token}')")

print(f"   • Vocab size: {len(tokenizer):,}")
print(f"   • EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"   • PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# Save tokenizer for consistency
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save_pretrained(tokenizer_dir)
print(f"   • Saved tokenizer to {tokenizer_dir}")

# CRITICAL FIX: Proper tokenization with EOS token
def tokenize_function(example):
    """
    Tokenize prompt + response with proper EOS token.
    Format: <prompt> <response> <EOS>
    """
    # Combine prompt and response with EOS at the end
    texts = [
        p + " " + r + tokenizer.eos_token 
        for p, r in zip(example["prompt"], example["response"])
    ]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors=None,  # Return lists, not tensors
    )
    
    # CRITICAL: Set labels = input_ids for causal LM
    # This is what the model will learn to predict
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("\n⚙️  Tokenizing dataset...")
print("   • Max length: 256 tokens")
print("   • Padding: max_length")
print("   • Format: <prompt> <response> <EOS>")

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["prompt", "response"],
    desc="Tokenizing"
)

# Validate tokenization
print("\n🔍 Validation:")
sample = tokenized_dataset[0]
print(f"   • Keys: {list(sample.keys())}")
print(f"   • input_ids length: {len(sample['input_ids'])}")
print(f"   • attention_mask length: {len(sample['attention_mask'])}")
print(f"   • labels length: {len(sample['labels'])}")

# Decode first example to verify
decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
print(f"\n   • First example preview:")
print(f"     {decoded[:200]}...")

# Save tokenized dataset
tokenized_dataset.save_to_disk(output_dir)
print(f"\n💾 Tokenization complete!")
print(f"✅ Saved to {output_dir}")
print(f"   • Total examples: {len(tokenized_dataset):,}")
print(f"   • Features: {list(tokenized_dataset.features.keys())}")