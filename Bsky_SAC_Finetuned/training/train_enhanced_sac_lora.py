#!/usr/bin/env python3
"""
Enhanced SAC LoRA Training Script for Combined ATC Dataset

This script loads the combined training dataset and performs LoRA fine-tuning
with enhanced progress tracking, proper deduplication before tensor conversion,
and comprehensive monitoring of the training process.
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
from typing import Dict, Any, Optional, List, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.trainer_callback import TrainerCallback
from tqdm import tqdm
import time
from collections import defaultdict
import gc

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
def setup_logging(log_dir: str) -> logging.Logger:
    """Setup comprehensive logging configuration with Windows encoding fix."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'sac_training.log')
    
    # Create custom formatter to avoid emoji issues on Windows
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("[START] Starting SAC LoRA Training Script")
    return logger


class EnhancedLossMonitorCallback(TrainerCallback):
    """Enhanced callback to monitor training with detailed progress tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.best_eval_loss = float('inf')
        self.best_step = 0
        self.config = config
        self.epoch_start_time = None
        self.step_times = []
        
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Track epoch start time."""
        self.epoch_start_time = time.time()
        logger.info(f"[EPOCH] Starting Epoch {state.epoch + 1}/{args.num_train_epochs}")
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Track epoch completion."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            logger.info(f"[SUCCESS] Completed Epoch {state.epoch + 1} in {epoch_time:.2f}s")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Track step timing with optimized memory management."""
        self.step_times.append(time.time())
        
        # Optimized GPU memory cleanup - only every 500 steps for better performance
        if state.global_step % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Enhanced logging with detailed metrics."""
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                # Track best validation loss
                if logs['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = logs['eval_loss']
                    self.best_step = state.global_step
                    logger.info(f"[BEST] New best validation loss: {logs['eval_loss']:.6f} at step {state.global_step}")
                    
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])
                
            if 'grad_norm' in logs:
                self.gradient_norms.append(logs['grad_norm'])
            
            # Log comprehensive training statistics
            if state.global_step % (args.logging_steps * 5) == 0:
                self._log_training_stats(state, logs)
    
    def _log_training_stats(self, state, logs):
        """Log detailed training statistics."""
        stats = []
        if self.train_losses:
            stats.append(f"Train Loss: {self.train_losses[-1]:.6f}")
        if self.eval_losses:
            stats.append(f"Val Loss: {self.eval_losses[-1]:.6f}")
        if self.learning_rates:
            stats.append(f"LR: {self.learning_rates[-1]:.2e}")
        if self.gradient_norms:
            stats.append(f"Grad Norm: {self.gradient_norms[-1]:.4f}")
        
        logger.info(f"[PROGRESS] Step {state.global_step}: {' | '.join(stats)}")
    
    def save_enhanced_plots(self, output_dir: str):
        """Save comprehensive training visualizations."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Loss curves
        if self.train_losses and self.eval_losses:
            axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue', alpha=0.8, linewidth=2)
            axes[0, 0].plot(self.eval_losses, label='Validation Loss', color='red', alpha=0.8, linewidth=2)
            axes[0, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if self.learning_rates:
            axes[0, 1].plot(self.learning_rates, label='Learning Rate', color='green', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        
        # Gradient norms
        if self.gradient_norms:
            axes[0, 2].plot(self.gradient_norms, label='Gradient Norm', color='orange', linewidth=2)
            axes[0, 2].set_title('Gradient Norms', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Steps')
            axes[0, 2].set_ylabel('Gradient Norm')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        if self.train_losses and self.eval_losses:
            min_len = min(len(self.train_losses), len(self.eval_losses))
            train_subset = self.train_losses[:min_len]
            eval_subset = self.eval_losses[:min_len]
            loss_diff = [eval - train for eval, train in zip(eval_subset, train_subset)]
            
            axes[1, 0].plot(loss_diff, label='Validation - Training Loss', color='purple', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss smoothed (moving average)
        if len(self.train_losses) > 10:
            window = min(50, len(self.train_losses) // 10)
            train_smooth = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            eval_smooth = np.convolve(self.eval_losses[:len(train_smooth)], np.ones(window)/window, mode='valid')
            
            axes[1, 1].plot(train_smooth, label='Training Loss (Smoothed)', color='lightblue', linewidth=2)
            axes[1, 1].plot(eval_smooth, label='Validation Loss (Smoothed)', color='lightcoral', linewidth=2)
            axes[1, 1].set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Model performance summary
        summary_text = [
            f'Best Validation Loss: {self.best_eval_loss:.6f}',
            f'Best Step: {self.best_step}',
            f'Final Training Loss: {self.train_losses[-1]:.6f}' if self.train_losses else 'N/A',
            f'Final Validation Loss: {self.eval_losses[-1]:.6f}' if self.eval_losses else 'N/A',
            f'Total Training Steps: {len(self.train_losses)}',
            f'Improvement: {((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100):.1f}%' if len(self.train_losses) > 1 else 'N/A'
        ]
        
        for i, text in enumerate(summary_text):
            axes[1, 2].text(0.1, 0.9 - i*0.12, text, transform=axes[1, 2].transAxes, 
                           fontsize=12, fontweight='bold' if i == 0 else 'normal')
        axes[1, 2].set_title('Training Summary', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/enhanced_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed metrics to JSON
        metrics = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms,
            'best_eval_loss': float(self.best_eval_loss),
            'best_step': int(self.best_step),
            'total_steps': len(self.train_losses),
            'improvement_percentage': float((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100) if len(self.train_losses) > 1 else 0.0
        }
        
        with open(f"{output_dir}/enhanced_training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)


def validate_data_item(item: Dict[str, Any], line_number: int) -> Optional[Dict[str, Any]]:
    """Enhanced validation with more comprehensive checks."""
    try:
        # Check required fields exist
        required_fields = ["instruction", "input", "output", "metadata"]
        for field in required_fields:
            if field not in item:
                return None
        
        # Check metadata has environment
        if "environment" not in item["metadata"]:
            return None
        
        # Extract and validate content
        instruction = str(item["instruction"]).strip()
        input_text = str(item["input"]).strip()
        output_text = str(item["output"]).strip()
        environment = str(item["metadata"]["environment"]).strip()
        
        # Enhanced content validation
        if not instruction or len(instruction) < 15:  # Increased minimum length
            return None
        
        if not output_text or len(output_text) < 10:  # Increased minimum length
            return None
        
        # Check for suspicious patterns
        suspicious_words = ["test", "debug", "placeholder", "todo", "fixme", "xxx", "lorem ipsum"]
        if any(word in instruction.lower() for word in suspicious_words):
            return None
        
        # Check for reasonable content length (not too long)
        if len(instruction) > 2000 or len(output_text) > 2000:
            return None
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "environment": environment
        }
    
    except Exception as e:
        return None


def create_content_hash(item: Dict[str, Any]) -> str:
    """Create a more robust hash for duplicate detection."""
    # Normalize text for better duplicate detection
    instruction = ' '.join(item['instruction'].lower().split())
    input_text = ' '.join(item['input'].lower().split())
    output_text = ' '.join(item['output'].lower().split())
    
    content = f"{instruction}|||{input_text}|||{output_text}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def enhanced_deduplicate_data(data: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Enhanced deduplication with detailed statistics and progress tracking."""
    logger.info("[DEDUP] Starting enhanced deduplication process...")
    
    dedup_config = config.get("data", {}).get("deduplication", {})
    enabled = dedup_config.get("enabled", True)
    
    if not enabled:
        logger.info("Deduplication disabled in config")
        return data, {"total": len(data), "duplicates": 0, "unique": len(data)}
    
    # Track deduplication statistics
    stats = {
        "total": len(data),
        "duplicates": 0,
        "unique": 0,
        "by_environment": defaultdict(int),
        "duplicate_pairs": []
    }
    
    seen_hashes: Set[str] = set()
    deduplicated_data = []
    
    # Create progress bar for deduplication
    with tqdm(total=len(data), desc="DEDUP - Deduplicating data", unit="samples") as pbar:
        for i, item in enumerate(data):
            content_hash = create_content_hash(item)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated_data.append(item)
                stats["by_environment"][item["environment"]] += 1
            else:
                stats["duplicates"] += 1
                # Track first few duplicate pairs for analysis
                if len(stats["duplicate_pairs"]) < 10:
                    stats["duplicate_pairs"].append({
                        "index": i,
                        "instruction": item["instruction"][:100] + "...",
                        "environment": item["environment"]
                    })
            
            pbar.update(1)
            pbar.set_postfix({
                "unique": len(deduplicated_data),
                "duplicates": stats["duplicates"]
            })
    
    stats["unique"] = len(deduplicated_data)
    
    # Log detailed deduplication results
    logger.info(f"[DATA] Deduplication Results:")
    logger.info(f"   Total samples: {stats['total']:,}")
    logger.info(f"   Unique samples: {stats['unique']:,}")
    logger.info(f"   Duplicates removed: {stats['duplicates']:,} ({stats['duplicates']/stats['total']*100:.1f}%)")
    
    logger.info("DEDUP - Unique samples by environment:")
    for env, count in stats["by_environment"].items():
        logger.info(f"   {env}: {count:,}")
    
    return deduplicated_data, stats


def load_combined_dataset(data_file: str, config: Dict[str, Any]):
    """Enhanced dataset loading with comprehensive progress tracking."""
    logger.info(f"DATA - Loading combined dataset from {data_file}...")
    
    # First pass: count total lines for accurate progress tracking
    logger.info("[DATA] Analyzing dataset size...")
    total_lines = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc="Counting lines", unit="lines"):
            total_lines += 1
    
    logger.info(f"[LIST] Processing {total_lines:,} lines...")
    
    # Load and validate JSONL data with progress tracking
    data = []
    invalid_count = 0
    parse_errors = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        with tqdm(total=total_lines, desc="LOAD - Loading & validating data", unit="lines") as pbar:
            for line_number, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    validated_item = validate_data_item(item, line_number)
                    
                    if validated_item:
                        data.append(validated_item)
                    else:
                        invalid_count += 1
                        
                except json.JSONDecodeError:
                    parse_errors += 1
                    invalid_count += 1
                
                # Update progress every 1000 lines
                if line_number % 1000 == 0:
                    pbar.set_postfix({
                        "valid": len(data),
                        "invalid": invalid_count,
                        "rate": f"{len(data)/line_number*100:.1f}%"
                    })
                
                pbar.update(1)
    
    logger.info(f"[SUCCESS] Loaded {len(data):,} valid examples")
    logger.info(f"[ERROR] Skipped {invalid_count:,} invalid entries ({invalid_count/total_lines*100:.1f}%)")
    logger.info(f"[TOOL] Parse errors: {parse_errors:,}")
    
    # Enhanced deduplication BEFORE tensor conversion
    data, dedup_stats = enhanced_deduplicate_data(data, config)
    
    # Log environment distribution
    env_counts = defaultdict(int)
    for item in data:
        env_counts[item["environment"]] += 1
    
    logger.info("ENV - Final environment distribution:")
    for env, count in sorted(env_counts.items()):
        logger.info(f"   {env}: {count:,} samples ({count/len(data)*100:.1f}%)")
    
    # Enhanced prompt formatting for Llama 3.1
    def format_example(example):
        formatted_text = f"""<|start_header_id|>system<|end_header_id|>

You are an expert air traffic controller with extensive experience in conflict resolution and aircraft separation. You have been trained on thousands of scenarios from BlueSky air traffic simulation environments including horizontal conflict resolution, vertical conflict resolution, sector management, and merge operations. Always provide clear, precise, and safe instructions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        return {"text": formatted_text}
    
    # Apply formatting with progress tracking
    logger.info("FORMAT - Formatting examples for training...")
    formatted_data = []
    with tqdm(data, desc="FORMAT - Formatting examples", unit="samples") as pbar:
        for item in pbar:
            formatted_data.append(format_example(item))
    
    # Split data into train/validation/test
    train_split = config["data"]["train_split"]
    val_split = config["data"]["validation_split"] 
    test_split = config["data"]["test_split"]
    
    logger.info("[DATA] Splitting dataset...")
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        formatted_data,
        test_size=test_split,
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation
    val_size = val_split / (train_split + val_split)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"[DATA] Dataset split complete:")
    logger.info(f"   Training: {len(train_data):,} examples ({len(train_data)/len(formatted_data)*100:.1f}%)")
    logger.info(f"   Validation: {len(val_data):,} examples ({len(val_data)/len(formatted_data)*100:.1f}%)")
    logger.info(f"   Test: {len(test_data):,} examples ({len(test_data)/len(formatted_data)*100:.1f}%)")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Save comprehensive data report
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "test_set.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Enhanced data quality report
    quality_report = {
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": data_file,
        "total_lines_processed": total_lines,
        "valid_examples": len(data),
        "invalid_examples": invalid_count,
        "parse_errors": parse_errors,
        "deduplication_stats": dedup_stats,
        "final_unique_examples": len(data),
        "environment_distribution": dict(env_counts),
        "data_splits": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data)
        },
        "quality_metrics": {
            "validation_rate": len(data) / total_lines,
            "duplication_rate": dedup_stats["duplicates"] / dedup_stats["total"] if dedup_stats["total"] > 0 else 0
        }
    }
    
    with open(output_dir / "enhanced_data_quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    logger.info(f"[LIST] Enhanced data quality report saved to {output_dir}/enhanced_data_quality_report.json")
    
    return train_dataset, val_dataset, test_dataset


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model with RTX 5070 Ti optimizations and fallback options."""
    model_name = config["model"]["name"]
    
    logger.info(f"[MODEL] Loading model: {model_name}")
    
    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"GPU - Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Optimized quantization config for RTX 5070 Ti
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["load_in_4bit"],
        bnb_4bit_use_double_quant=config["model"]["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["model"]["quantization"]["bnb_4bit_compute_dtype"])
    )
    
    # Try Flash Attention first, fallback to eager if not available
    attn_implementation = "eager"  # Default fallback
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        logger.info("[SPEED] Using Flash Attention 2 for optimal performance")
    except ImportError:
        logger.info("[FALLBACK] Flash Attention not available, using eager attention")
    
    # Load model with RTX 5070 Ti optimizations
    logger.info("[LOADING] Loading model (optimized for RTX 5070 Ti)...")
    with tqdm(desc="Loading model", unit="step") as pbar:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # Better for RTX 5070 Ti
                attn_implementation=attn_implementation,
                use_cache=False  # Disable during training for memory efficiency
            )
            pbar.update(1)
            
            # Prepare model for k-bit training
            pbar.set_description("Preparing for k-bit training")
            model = prepare_model_for_kbit_training(model)
            pbar.update(1)
            
        except Exception as e:
            # Fallback to basic settings if advanced features fail
            logger.warning(f"[FALLBACK] Advanced model loading failed: {e}")
            logger.info("[FALLBACK] Loading with basic settings...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                use_cache=False
            )
            model = prepare_model_for_kbit_training(model)
            pbar.update(2)
    
    # Load tokenizer with optimizations
    logger.info("[TOKENIZER] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=True  # Use fast tokenizer for speed
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("[SUCCESS] Model and tokenizer loaded successfully")
    
    return model, tokenizer


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Setup LoRA configuration with detailed logging."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none"
    )
    
    logger.info(f"[TOOL] LoRA Configuration:")
    logger.info(f"   Rank: {lora_config.r}")
    logger.info(f"   Alpha: {lora_config.lora_alpha}")
    logger.info(f"   Dropout: {lora_config.lora_dropout}")
    logger.info(f"   Target modules: {lora_config.target_modules}")
    
    return lora_config


class EnhancedProgressTrainer(Trainer):
    """Enhanced Trainer with comprehensive progress tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_progress_bar = None
        self.epoch_progress_bar = None
        self.global_step_start_time = time.time()
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Optimized training step with minimal overhead."""
        # Perform training step with minimal tracking for speed
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        # Minimal progress updates to reduce overhead
        if hasattr(self, 'training_progress_bar') and self.training_progress_bar is not None:
            if self.state.global_step % 50 == 0:  # Update every 50 steps
                self.training_progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'step': self.state.global_step
                })
                self.training_progress_bar.update(50)  # Update by 50 to catch up
        
        return loss
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Enhanced train method with comprehensive progress tracking."""
        # Calculate total steps
        total_steps = len(self.get_train_dataloader()) * self.args.num_train_epochs
        
        # Create main training progress bar
        self.training_progress_bar = tqdm(
            total=total_steps,
            desc="TRAIN - Training Progress",
            unit="step",
            leave=True,
            dynamic_ncols=True
        )
        
        logger.info(f"TRAIN - Starting training: {total_steps:,} total steps over {self.args.num_train_epochs} epochs")
        
        try:
            result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        finally:
            if self.training_progress_bar:
                self.training_progress_bar.close()
        
        logger.info("[SUCCESS] Training completed successfully!")
        return result


def main():
    """Enhanced main training function with comprehensive progress tracking."""
    parser = argparse.ArgumentParser(description="Enhanced SAC LoRA Training on Combined ATC Dataset")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--data", required=True, help="Path to combined training data JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory for trained model")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup logging first
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(str(output_dir / "logs"))
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("[START] Starting Enhanced SAC LoRA Training...")
    logger.info(f"DATA - Configuration loaded from: {args.config}")
    logger.info(f"DATA - Data file: {args.data}")
    logger.info(f"SAVE - Output directory: {args.output_dir}")
    
    # Load and prepare data (with deduplication BEFORE tensor conversion)
    train_dataset, val_dataset, test_dataset = load_combined_dataset(args.data, config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup LoRA
    lora_config = setup_lora_config(config)
    logger.info("[TOOL] Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    # Enable gradient computation for the model
    model.train()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[DATA] Model Parameters:")
    logger.info(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"   Total: {total_params:,}")
    
    # Tokenize datasets with enhanced progress tracking
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config["data"]["max_sequence_length"],
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    logger.info("[TOKENIZE] Tokenizing datasets...")
    
    # Tokenize with single-process for small datasets (faster for <50k samples)
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="TOKENIZE - Training data"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="TOKENIZE - Validation data"
    )
    
    logger.info("[SUCCESS] Tokenization complete")
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Setup enhanced callbacks
    loss_monitor = EnhancedLossMonitorCallback(config)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Setup optimized training arguments for RTX 5070 Ti (speed optimized)
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
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # Disable wandb
        dataloader_pin_memory=config.get("hardware", {}).get("pin_memory", True),
        gradient_checkpointing=False,  # Disable for speed - we have enough VRAM
        bf16=True,  # Use bf16 for RTX 5070 Ti
        fp16=False,  # Disable fp16 when using bf16
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        prediction_loss_only=True,  # Faster evaluation
        ddp_find_unused_parameters=False,  # Faster training
        dataloader_num_workers=0,  # Single process for small datasets
        optim="adamw_torch",  # Use PyTorch AdamW (stable and fast)
        lr_scheduler_type="linear",  # Simple linear scheduler
        save_safetensors=True,  # Modern format
        disable_tqdm=False,  # Keep progress bars
        log_level="info",
        # Additional speed optimizations
        skip_memory_metrics=True,  # Skip memory tracking for speed
        include_num_input_tokens_seen=False,  # Skip token counting
        group_by_length=True,  # Group similar length sequences for efficiency
        length_column_name="length",
        auto_find_batch_size=False,  # Disable auto batch size detection
    )
    
    # Create enhanced trainer
    trainer = EnhancedProgressTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[loss_monitor, early_stopping]
    )
    
    # Start training
    logger.info("[TARGET] Starting training process...")
    if args.resume:
        logger.info(f"[FOLDER] Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save final model
    logger.info("[SAVE] Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save enhanced training visualizations and metrics
    logger.info("[DATA] Generating training visualizations...")
    loss_monitor.save_enhanced_plots(str(output_dir))
    
    # Save config
    with open(output_dir / "training_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Final summary
    logger.info("[COMPLETE] Training completed successfully!")
    logger.info(f"[DATA] Final Results:")
    logger.info(f"   Best validation loss: {loss_monitor.best_eval_loss:.6f}")
    logger.info(f"   Best step: {loss_monitor.best_step}")
    logger.info(f"   Model saved to: {output_dir}")
    logger.info(f"   Training curves saved to: {output_dir}/enhanced_training_curves.png")


if __name__ == "__main__":
    main()
