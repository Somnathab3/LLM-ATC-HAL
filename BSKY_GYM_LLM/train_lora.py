#!/usr/bin/env python3
"""
Enhanced PEFT-based LoRA fine-tuning for Llama 3.1 8B on BlueSky Gym data.
Includes validation split, early stopping, hyperparameter tuning, and best practices.
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.trainer_callback import TrainerCallback

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('BSKY_GYM_LLM/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LossMonitorCallback(TrainerCallback):
    """Custom callback to monitor training and validation loss."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.best_eval_loss = float('inf')
        self.best_step = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log training metrics."""
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                # Track best validation loss
                if logs['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = logs['eval_loss']
                    self.best_step = state.global_step
                    logger.info(f"🎯 New best validation loss: {logs['eval_loss']:.4f} at step {state.global_step}")
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])
    
    def save_plots(self, output_dir: str):
        """Save training curves."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.train_losses and self.eval_losses:
            axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue', alpha=0.7)
            axes[0, 0].plot(self.eval_losses, label='Validation Loss', color='red', alpha=0.7)
            axes[0, 0].set_title('Training vs Validation Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if self.learning_rates:
            axes[0, 1].plot(self.learning_rates, label='Learning Rate', color='green')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        if self.train_losses and self.eval_losses:
            min_len = min(len(self.train_losses), len(self.eval_losses))
            train_subset = self.train_losses[:min_len]
            eval_subset = self.eval_losses[:min_len]
            loss_diff = [eval - train for eval, train in zip(eval_subset, train_subset)]
            
            axes[1, 0].plot(loss_diff, label='Validation - Training Loss', color='purple')
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Overfitting Indicator')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Model performance summary
        axes[1, 1].text(0.1, 0.8, f'Best Validation Loss: {self.best_eval_loss:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.7, f'Best Step: {self.best_step}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Final Training Loss: {self.train_losses[-1] if self.train_losses else "N/A"}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Final Validation Loss: {self.eval_losses[-1] if self.eval_losses else "N/A"}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'best_eval_loss': self.best_eval_loss,
            'best_step': self.best_step
        }
        with open(f"{output_dir}/training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

def get_hyperparameter_config() -> Dict[str, Any]:
    """Get optimized hyperparameters for LoRA fine-tuning."""
    return {
        "model": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "max_sequence_length": 2048,
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16"
            }
        },
        "lora": {
            "rank": 16,  # Increased from 8 for better capacity
            "alpha": 32,  # 2x rank is common practice
            "dropout": 0.1,  # Increased dropout for regularization
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"       # MLP
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "data": {
            "train_split": 0.85,      # 85% for training
            "validation_split": 0.10,  # 10% for validation
            "test_split": 0.05,        # 5% for testing (held out)
            "max_length": 2048,
            "padding": "max_length",
            "truncation": True
        },
        "training": {
            "num_epochs": 5,  # Increased epochs with early stopping
            "batch_size": 1,
            "gradient_accumulation_steps": 8,  # Reduced for memory efficiency
            "learning_rate": 2e-4,  # Standard LoRA learning rate
            "weight_decay": 0.01,   # L2 regularization
            "warmup_ratio": 0.1,    # 10% warmup
            "lr_scheduler_type": "cosine",  # Cosine annealing
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "logging_steps": 50,
            "eval_steps": 2000,     # Less frequent evaluation for large dataset
            "save_steps": 2000,     # Must be multiple of eval_steps for load_best_model_at_end
            "gradient_checkpointing": False,  # Disabled to avoid conflicts with quantized models
            "dataloader_num_workers": 4,     # Parallel data loading
            "fp16": True,
            "max_grad_norm": 1.0    # Gradient clipping
        },
        "early_stopping": {
            "patience": 2,           # More aggressive - stop after 2 evaluations without improvement
            "threshold": 0.0005,     # Lower threshold for more sensitive stopping
            "restore_best_weights": True
        },
        "optimization": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0
        }
    }

def verify_environment():
    """Verify CUDA and PyTorch setup with enhanced checks."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Memory information
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    else:
        logger.warning("CUDA not available - training will be very slow on CPU")

def load_and_prepare_data(config: Dict[str, Any]):
    """Load and split the training data with validation set."""
    logger.info("Loading dataset from JSONL...")
    
    # Load JSONL manually to handle data type inconsistencies
    data = []
    with open("data/gym_distill_intelligent.jsonl", 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # Extract only the fields we need for training
                clean_item = {
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": item["output"]
                }
                data.append(clean_item)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Enhanced prompt formatting with better structure
    def format_example(example):
        formatted_text = f"""<|start_header_id|>system<|end_header_id|>

You are an expert air traffic controller with extensive experience in conflict resolution and aircraft separation. You have been trained on thousands of scenarios from BlueSky air traffic simulation environments. Always provide clear, precise, and safe instructions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        return {"text": formatted_text}
    
    # Apply formatting
    formatted_data = [format_example(item) for item in data]
    
    # Split data into train/validation/test
    train_split = config["data"]["train_split"]
    val_split = config["data"]["validation_split"]
    test_split = config["data"]["test_split"]
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        formatted_data, 
        test_size=test_split, 
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation
    val_size = val_split / (train_split + val_split)  # Adjust for remaining data
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Dataset split:")
    logger.info(f"  Training: {len(train_data):,} examples ({len(train_data)/len(formatted_data)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_data):,} examples ({len(val_data)/len(formatted_data)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data):,} examples ({len(test_data)/len(formatted_data)*100:.1f}%)")
    
    # Create datasets
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Save test set for later evaluation
    os.makedirs("BSKY_GYM_LLM/data", exist_ok=True)
    with open("BSKY_GYM_LLM/data/test_set.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return train_dataset, val_dataset, test_dataset

def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model with 4-bit quantization and tokenizer."""
    model_name = config["model"]["name"]
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Add padding token with proper configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure proper padding side for causal LM
    tokenizer.padding_side = "right"
    
    logger.info("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["load_in_4bit"],
        bnb_4bit_use_double_quant=config["model"]["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"])
    )
    
    logger.info("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"]),
        attn_implementation=None  # Disable Flash Attention for compatibility
    )
    
    # Enable gradient checkpointing for memory efficiency (if not conflicting)
    if config["training"]["gradient_checkpointing"] and not model.config.quantization_config:
        model.gradient_checkpointing_enable()
    
    return model, tokenizer

def setup_lora(model, config: Dict[str, Any]):
    """Configure and attach LoRA adapters with enhanced settings."""
    logger.info("Configuring LoRA...")
    
    # Prepare base model for PEFT
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type=getattr(TaskType, config["lora"]["task_type"])
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable training mode and ensure gradients are enabled
    model.train()
    
    # More aggressive gradient enabling
    for name, param in model.named_parameters():
        if any(trainable_part in name for trainable_part in ['lora_', 'modules_to_save']):
            param.requires_grad = True
            logger.debug(f"Enabled gradients for: {name}")
    
    # Verify gradients are properly set
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    if not trainable_params:
        raise RuntimeError("No trainable parameters found! LoRA setup failed.")
    
    logger.info(f"Trainable parameter names: {trainable_params[:5]}...")  # Show first 5
    
    # Print detailed parameter info
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Trainable parameters: {trainable_count:,}")
    logger.info(f"Total parameters: {total_count:,}")
    logger.info(f"Trainable percentage: {100 * trainable_count / total_count:.4f}%")
    
    model.print_trainable_parameters()
    
    return model

def tokenize_dataset(dataset, tokenizer, config: Dict[str, Any]):
    """Tokenize the dataset with enhanced settings."""
    logger.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Tokenize with proper settings for causal LM
        result = tokenizer(
            examples["text"],
            padding=False,  # Dynamic padding in data collator
            truncation=config["data"]["truncation"],
            max_length=config["data"]["max_length"],
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=4  # Parallel processing
    )
    
    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, val_dataset, config: Dict[str, Any]):
    """Train the model with enhanced LoRA configuration."""
    logger.info("Setting up enhanced training...")
    
    # Enhanced data collator with padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Optimize for tensor cores
        return_tensors="pt"
    )
    
    # Output directory
    output_dir = "BSKY_GYM_LLM/models/llama3.1-bsky-lora"
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training schedule
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        
        # Optimization
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        adam_beta1=config["optimization"]["adam_beta1"],
        adam_beta2=config["optimization"]["adam_beta2"],
        adam_epsilon=config["optimization"]["adam_epsilon"],
        max_grad_norm=config["optimization"]["max_grad_norm"],
        
        # Learning rate schedule
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        
        # Evaluation and saving
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,  # Keep only 3 best checkpoints
        
        # Hardware optimization
        fp16=config["training"]["fp16"],
        gradient_checkpointing=False,  # Disabled for quantized models
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        
        # Logging and reporting
        report_to=[],  # Disable external reporting
        logging_dir=f"{output_dir}/logs",
        run_name="llama3.1-bsky",
        
        # Memory optimization
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        
        # Reproducibility
        seed=42,
        data_seed=42
    )
    
    # Initialize callbacks
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping"]["patience"],
        early_stopping_threshold=config["early_stopping"]["threshold"]
    )
    callbacks.append(early_stopping)
    
    # Loss monitoring callback
    loss_monitor = LossMonitorCallback()
    callbacks.append(loss_monitor)
    
    # Initialize trainer with enhanced configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # Train with error handling
    logger.info("Starting enhanced training...")
    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")
    logger.info(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    logger.info(f"Total training steps: {trainer.state.max_steps}")
    
    try:
        # Start training
        train_result = trainer.train()
        
        # Save final model and tokenizer
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training results
        with open(f"{output_dir}/train_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Generate and save plots
        loss_monitor.save_plots(output_dir)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        with open(f"{output_dir}/eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"Final validation loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        logger.info(f"Best validation loss: {loss_monitor.best_eval_loss:.4f} at step {loss_monitor.best_step}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Training curves saved to: {output_dir}/training_curves.png")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    return output_dir, loss_monitor

def main():
    """Enhanced main training function."""
    try:
        # Load configuration
        config = get_hyperparameter_config()
        
        # Create directories
        os.makedirs("BSKY_GYM_LLM/models", exist_ok=True)
        os.makedirs("BSKY_GYM_LLM/logs", exist_ok=True)
        os.makedirs("BSKY_GYM_LLM/data", exist_ok=True)
        os.makedirs("BSKY_GYM_LLM/config", exist_ok=True)
        
        # Save configuration
        with open("BSKY_GYM_LLM/config/training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("🚀 Starting Enhanced LoRA Fine-tuning Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Verify environment
        logger.info("Step 1: Verifying environment...")
        verify_environment()
        
        # Step 2: Load and prepare data with splits
        logger.info("Step 2: Loading and splitting data...")
        train_dataset, val_dataset, test_dataset = load_and_prepare_data(config)
        
        # Step 3: Setup model and tokenizer
        logger.info("Step 3: Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Step 4: Setup LoRA
        logger.info("Step 4: Configuring LoRA...")
        model = setup_lora(model, config)
        
        # Step 5: Tokenize datasets
        logger.info("Step 5: Tokenizing datasets...")
        train_tokenized = tokenize_dataset(train_dataset, tokenizer, config)
        val_tokenized = tokenize_dataset(val_dataset, tokenizer, config)
        
        # Step 6: Train model with validation and early stopping
        logger.info("Step 6: Training model with enhanced settings...")
        output_dir, loss_monitor = train_model(model, tokenizer, train_tokenized, val_tokenized, config)
        
        logger.info("🎉 Enhanced training pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("📋 Next steps:")
        logger.info("1. Review training curves: BSKY_GYM_LLM/models/llama3.1-bsky-lora/training_curves.png")
        logger.info("2. Test the model: python BSKY_GYM_LLM/test_model.py")
        logger.info("3. Update Modelfile to reference the enhanced adapter")
        logger.info("4. Run inference: ollama run llama3.1-bsky:latest")
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save error to log file
        with open("BSKY_GYM_LLM/logs/error.log", "a") as f:
            f.write(f"\n{traceback.format_exc()}\n")
        raise

if __name__ == "__main__":
    main()
