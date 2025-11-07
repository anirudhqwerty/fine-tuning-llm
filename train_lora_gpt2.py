import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"💡 Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable: {100 * trainable_params / all_param:.2f}%"
    )

def train_model():
    """Train GPT-2 with LoRA - OPTIMIZED & FIXED VERSION."""
    
    # Configuration
    dataset_path = "dataset/tokenized_customer_support"
    tokenizer_path = "dataset/tokenizer"  # FIXED: Use saved tokenizer
    output_dir = "outputs/customer_support_lora"
    final_model_dir = "outputs/final_customer_support_model"
    model_name = "gpt2"
    
    # IMPROVED TRAINING SETTINGS (balanced speed & quality)
    batch_size = 8  # Reasonable batch size
    gradient_accumulation_steps = 8  # Effective batch size = 64
    max_steps = 3000  # More training steps for better results
    learning_rate = 2e-4  # Lower LR for stable fine-tuning
    warmup_steps = 100  # Gradual warmup
    
    # Validate paths
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}. Run prepare_dataset.py first!")
    
    try:
        # Load tokenized dataset
        logger.info(f"📂 Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        logger.info(f"✅ Loaded {len(dataset):,} examples")
        logger.info(f"   • Features: {list(dataset.features.keys())}")
        
        # Split dataset (95% train, 5% eval for better evaluation)
        logger.info("\n📊 Splitting dataset...")
        split = dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        logger.info(f"   • Training: {len(train_ds):,} examples")
        logger.info(f"   • Evaluation: {len(eval_ds):,} examples")
        
        # FIXED: Load the SAME tokenizer used during tokenization
        logger.info(f"\n🔧 Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"   • Vocab size: {len(tokenizer):,}")
        logger.info(f"   • PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        logger.info(f"   • EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        
        # Load base model
        logger.info(f"\n🤖 Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Resize embeddings to match tokenizer (in case of added tokens)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"   • Resized embeddings to {len(tokenizer):,}")
        
        # IMPROVED LoRA configuration (better coverage)
        logger.info("\n⚙️  Configuring LoRA...")
        peft_config = LoraConfig(
            r=16,  # Higher rank for better quality
            lora_alpha=32,  # Standard alpha = 2*r
            target_modules=["c_attn", "c_proj"],  # Both attention and projection
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)
        
        # FIXED: Data collator with proper label handling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(final_model_dir).mkdir(parents=True, exist_ok=True)
        
        # IMPROVED Training arguments
        logger.info("\n🏋️  Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Batch settings
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Training duration
            max_steps=max_steps,
            
            # Optimization (IMPROVED)
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,  # Gradient clipping
            
            # Mixed precision
            fp16=torch.cuda.is_available(),
            fp16_full_eval=torch.cuda.is_available(),
            
            # Evaluation & saving
            eval_strategy="steps",
            eval_steps=250,  # Evaluate more frequently
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,  # Keep more checkpoints
            load_best_model_at_end=True,  # Load best model
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_strategy="steps",
            logging_steps=50,
            logging_first_step=True,
            report_to=["none"],
            
            # Performance
            dataloader_num_workers=4 if os.cpu_count() > 4 else 2,
            dataloader_pin_memory=True,
            gradient_checkpointing=False,
            optim="adamw_torch" if not torch.cuda.is_available() else "adamw_torch_fused",
            
            # Reproducibility
            seed=42,
        )
        
        # Initialize trainer
        logger.info("\n🚀 Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("\n" + "="*60)
        logger.info("🔥 Starting training...")
        logger.info(f"   • Steps: {max_steps:,}")
        logger.info(f"   • Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"   • Learning rate: {learning_rate}")
        logger.info("="*60 + "\n")
        
        train_result = trainer.train()
        
        # Print training summary
        logger.info("\n" + "="*60)
        logger.info("✅ Training completed!")
        logger.info("="*60)
        logger.info(f"📈 Training Loss: {train_result.training_loss:.4f}")
        logger.info(f"⏱️  Training Time: {train_result.metrics['train_runtime']:.2f}s ({train_result.metrics['train_runtime']/60:.1f} min)")
        logger.info(f"🚄 Samples/Second: {train_result.metrics['train_samples_per_second']:.2f}")
        
        # Final evaluation
        logger.info("\n📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"📉 Eval Loss: {eval_results['eval_loss']:.4f}")
        perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
        logger.info(f"📖 Perplexity: {perplexity:.2f}")
        
        # Save final model
        logger.info(f"\n💾 Saving final model to {final_model_dir}...")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Save training info
        info_file = os.path.join(final_model_dir, "training_info.txt")
        with open(info_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"LoRA Config:\n")
            f.write(f"  • Rank (r): {peft_config.r}\n")
            f.write(f"  • Alpha: {peft_config.lora_alpha}\n")
            f.write(f"  • Target modules: {peft_config.target_modules}\n\n")
            f.write(f"Training:\n")
            f.write(f"  • Steps: {max_steps}\n")
            f.write(f"  • Learning rate: {learning_rate}\n")
            f.write(f"  • Batch size: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}\n")
            f.write(f"  • Training samples: {len(train_ds):,}\n")
            f.write(f"  • Eval samples: {len(eval_ds):,}\n\n")
            f.write(f"Results:\n")
            f.write(f"  • Training Loss: {train_result.training_loss:.4f}\n")
            f.write(f"  • Eval Loss: {eval_results['eval_loss']:.4f}\n")
            f.write(f"  • Perplexity: {perplexity:.2f}\n")
            f.write(f"  • Training Time: {train_result.metrics['train_runtime']:.2f}s ({train_result.metrics['train_runtime']/60:.1f} min)\n")
            f.write(f"  • Samples/Second: {train_result.metrics['train_samples_per_second']:.2f}\n")
        
        logger.info(f"✅ Training info saved to {info_file}")
        logger.info(f"\n🎉 All done! Model saved to: {final_model_dir}")
        logger.info(f"\n💡 To use this model:")
        logger.info(f"   from peft import AutoPeftModelForCausalLM")
        logger.info(f"   model = AutoPeftModelForCausalLM.from_pretrained('{final_model_dir}')")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Check for GPU
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("⚠️  No GPU detected! Training will be slow on CPU.")
        
        train_model()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)