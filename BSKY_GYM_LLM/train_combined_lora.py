#!/usr/bin/env python3
"""
Updated LoRA Training Script for Combined ATC Dataset

This script loads the combined training dataset and performs LoRA fine-tuning
with proper handling of the new JSONL format.
"""

import os
import json
import logging
import torch
import numpy as np
import argparse
import hashlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.trainer_callback import TrainerCallback
from tqdm import tqdm

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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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


def validate_data_item(item: Dict[str, Any], line_number: int) -> Optional[Dict[str, Any]]:
    """Validate a single data item and return cleaned version or None if invalid."""
    try:
        # Check required fields exist
        required_fields = ["instruction", "input", "output", "metadata"]
        for field in required_fields:
            if field not in item:
                logger.warning(f"Line {line_number}: Missing required field '{field}'")
                return None
        
        # Check metadata has environment
        if "environment" not in item["metadata"]:
            logger.warning(f"Line {line_number}: Missing 'environment' in metadata")
            return None
        
        # Extract and validate content
        instruction = str(item["instruction"]).strip()
        input_text = str(item["input"]).strip()
        output_text = str(item["output"]).strip()
        environment = str(item["metadata"]["environment"]).strip()
        
        # Check for empty or very short content
        if not instruction or len(instruction) < 10:
            logger.warning(f"Line {line_number}: Instruction too short or empty")
            return None
        
        if not output_text or len(output_text) < 5:
            logger.warning(f"Line {line_number}: Output too short or empty")
            return None
        
        # Check for suspicious patterns
        if any(word in instruction.lower() for word in ["test", "debug", "placeholder", "todo"]):
            logger.warning(f"Line {line_number}: Suspicious content in instruction")
            return None
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "environment": environment
        }
    
    except Exception as e:
        logger.warning(f"Line {line_number}: Error validating item: {e}")
        return None


def create_content_hash(item: Dict[str, Any]) -> str:
    """Create a hash of the content for duplicate detection."""
    content = f"{item['instruction']}|||{item['input']}|||{item['output']}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def deduplicate_data(data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """Remove duplicate entries based on content hash and return deduplicated data and count."""
    logger.info("Checking for duplicates...")
    
    seen_hashes: Set[str] = set()
    deduplicated_data = []
    duplicate_count = 0
    
    for item in tqdm(data, desc="Deduplicating"):
        content_hash = create_content_hash(item)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduplicated_data.append(item)
        else:
            duplicate_count += 1
    
    logger.info(f"Removed {duplicate_count} duplicate entries")
    logger.info(f"Unique entries: {len(deduplicated_data)}")
    
    return deduplicated_data, duplicate_count


def load_combined_dataset(data_file: str, config: Dict[str, Any]):
    """Load and prepare the combined ATC dataset with validation and deduplication."""
    logger.info(f"Loading combined dataset from {data_file}...")
    
    # First pass: count total lines for progress bar
    total_lines = 0
    with open(data_file, 'r') as f:
        for _ in f:
            total_lines += 1
    
    logger.info(f"Processing {total_lines:,} lines...")
    
    # Load and validate JSONL data
    data = []
    invalid_count = 0
    
    with open(data_file, 'r') as f:
        for line_number, line in enumerate(tqdm(f, desc="Loading data", total=total_lines), 1):
            try:
                item = json.loads(line.strip())
                validated_item = validate_data_item(item, line_number)
                
                if validated_item:
                    data.append(validated_item)
                else:
                    invalid_count += 1
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_number}: JSON decode error: {e}")
                invalid_count += 1
                continue
    
    logger.info(f"Loaded {len(data):,} valid examples")
    logger.info(f"Skipped {invalid_count:,} invalid entries ({invalid_count/total_lines*100:.1f}%)")
    
    # Deduplicate data
    data, duplicate_count = deduplicate_data(data)
    
    # Log environment distribution
    env_counts = {}
    for item in data:
        env = item["environment"]
        env_counts[env] = env_counts.get(env, 0) + 1
    
    logger.info("Environment distribution:")
    for env, count in env_counts.items():
        logger.info(f"  {env}: {count} samples ({count/len(data)*100:.1f}%)")
    
    # Enhanced prompt formatting for Llama 3.1
    def format_example(example):
        formatted_text = f"""<|start_header_id|>system<|end_header_id|>

You are an expert air traffic controller with extensive experience in conflict resolution and aircraft separation. You have been trained on thousands of scenarios from BlueSky air traffic simulation environments including horizontal conflict resolution, vertical conflict resolution, sector management, and merge operations. Always provide clear, precise, and safe instructions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        return {"text": formatted_text}
    
    # Apply formatting with progress bar
    logger.info("Formatting examples for training...")
    formatted_data = []
    for item in tqdm(data, desc="Formatting examples"):
        formatted_data.append(format_example(item))
    
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
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Save test set for later evaluation with validation stats
    os.makedirs("BSKY_GYM_LLM/data", exist_ok=True)
    with open("BSKY_GYM_LLM/data/test_set.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Save data quality report
    quality_report = {
        "total_lines_processed": total_lines,
        "valid_examples": len(data),
        "invalid_examples": invalid_count,
        "duplicate_examples": duplicate_count,
        "final_unique_examples": len(data),
        "environment_distribution": env_counts,
        "data_splits": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data)
        }
    }
    
    with open("BSKY_GYM_LLM/data/data_quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    logger.info("Data quality report saved to BSKY_GYM_LLM/data/data_quality_report.json")
    
    return train_dataset, val_dataset, test_dataset


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model with 4-bit quantization and tokenizer."""
    model_name = config["model"]["name"]
    
    logger.info(f"Loading model: {model_name}")
    
    # Setup quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["load_in_4bit"],
        bnb_4bit_use_double_quant=config["model"]["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"])
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # Use eager attention as fallback
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Setup LoRA configuration."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none"
    )
    
    logger.info(f"LoRA Config:")
    logger.info(f"  Rank: {lora_config.r}")
    logger.info(f"  Alpha: {lora_config.lora_alpha}")
    logger.info(f"  Dropout: {lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    
    return lora_config


class ProgressTrainer(Trainer):
    """Enhanced Trainer with better progress tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = None
    
    def _get_train_sampler(self, train_dataset=None):
        """Override to get sampler info for progress tracking."""
        sampler = super()._get_train_sampler(train_dataset)
        return sampler
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to update progress bar."""
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        # Update progress bar if available
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            self.progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.lr_scheduler.get_last_lr()[0]:.2e}' if hasattr(self, 'lr_scheduler') else 'N/A'
            })
            self.progress_bar.update(1)
        
        return loss
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Override train method to add progress tracking."""
        # Calculate total steps
        total_steps = len(self.get_train_dataloader()) * self.args.num_train_epochs
        
        # Create progress bar
        self.progress_bar = tqdm(
            total=total_steps,
            desc="Training Progress",
            unit="step",
            leave=True
        )
        
        try:
            result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        finally:
            if self.progress_bar:
                self.progress_bar.close()
        
        return result


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train LoRA model on combined ATC dataset")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--data", required=True, help="Path to combined training data JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory for trained model")
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting LoRA training with combined ATC dataset...")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_combined_dataset(args.data, config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup LoRA
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Enable gradient computation for the model
    model.train()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Tokenize datasets with progress tracking
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Changed to max_length padding
            max_length=config["data"]["max_sequence_length"],
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    logger.info("Tokenizing training dataset...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing train"
    )
    
    logger.info("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing validation"
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Setup callbacks
    loss_monitor = LossMonitorCallback()
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        eval_strategy="steps",  # Fixed: evaluation_strategy -> eval_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # Disable wandb
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=42,
        data_seed=42
    )
    
    # Create enhanced trainer with progress tracking
    trainer = ProgressTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[loss_monitor, early_stopping]
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training plots and metrics
    loss_monitor.save_plots(str(output_dir))
    
    # Save config
    with open(output_dir / "training_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Training completed! Model saved to {output_dir}")
    logger.info(f"Best validation loss: {loss_monitor.best_eval_loss:.4f} at step {loss_monitor.best_step}")


if __name__ == "__main__":
    main()
