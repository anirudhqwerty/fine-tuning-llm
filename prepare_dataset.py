import os
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

def prepare_dataset():
    """Tokenize dataset with proper error handling and validation."""
    
    input_file = "dataset/final_dataset.jsonl"
    output_dir = "dataset/tokenized_customer_support"
    model_name = "gpt2"
    max_length = 512  # Increased for better context
    
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        # Load dataset
        print(f"📂 Loading dataset from {input_file}...")
        dataset = load_dataset("json", data_files=input_file, split="train")
        print(f"✅ Loaded {len(dataset)} examples")
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Load tokenizer
        print(f"\n🔧 Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   • Set pad_token = eos_token")
        
        # Print sample before tokenization
        print(f"\n📝 Sample example:")
        print(f"   Prompt: {dataset[0]['prompt'][:100]}...")
        print(f"   Response: {dataset[0]['response'][:100]}...")
        
        # Tokenization function
        def tokenize_function(examples):
            """Combine prompt and response, then tokenize."""
            texts = [
                f"{prompt} {response}" 
                for prompt, response in zip(examples["prompt"], examples["response"])
            ]
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None,
            )
            
            return tokenized
        
        # Tokenize dataset
        print(f"\n🔄 Tokenizing dataset (max_length={max_length})...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=["prompt", "response"],
            desc="Tokenizing",
        )
        
        # Validate tokenization
        print(f"\n📊 Tokenization statistics:")
        sample_length = len(tokenized_dataset[0]["input_ids"])
        print(f"   • Token length: {sample_length}")
        print(f"   • Vocab size: {len(tokenizer)}")
        print(f"   • Total examples: {len(tokenized_dataset)}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save tokenized dataset
        print(f"\n💾 Saving tokenized dataset to {output_dir}...")
        tokenized_dataset.save_to_disk(output_dir)
        
        print(f"\n✅ Tokenization complete!")
        print(f"   • Output: {output_dir}")
        
    except Exception as e:
        raise RuntimeError(f"Dataset preparation failed: {str(e)}")

if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        exit(1)