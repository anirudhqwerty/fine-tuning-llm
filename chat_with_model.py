import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model_path = "outputs/final_customer_support_model"

print("🔧 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

print("✅ Model loaded!\n")

# Test prompts
test_cases = [
    "Customer: i did not recieve my order\nAgent:",
    "Customer: how do i track my package\nAgent:",
    "Customer: i want to return this item\nAgent:",
    "Customer: when will my order arrive\nAgent:",
    "Customer: the product is damaged\nAgent:",
]

print("="*60)
print("TESTING MODEL OUTPUTS")
print("="*60)

for prompt in test_cases:
    print(f"\n📝 Prompt: {prompt.split('Agent:')[0].strip()}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with PROPER settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,           # Limit response length
            min_new_tokens=10,            # Ensure some output
            do_sample=True,               # Use sampling
            temperature=0.7,              # Control randomness
            top_p=0.9,                    # Nucleus sampling
            top_k=50,                     # Top-k sampling
            repetition_penalty=1.2,       # Prevent repetition
            no_repeat_ngram_size=3,       # Avoid repeating phrases
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("Agent:")[-1].strip()
    
    print(f"🤖 Agent: {response}")
    print("-"*60)