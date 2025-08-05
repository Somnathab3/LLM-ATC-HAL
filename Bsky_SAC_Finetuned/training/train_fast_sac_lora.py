#!/usr/bin/env python3
"""
Fast SAC LoRA Training Script - Optimized for Speed

This is a simplified, speed-optimized version of the training script
focusing on maximum training speed with minimal overhead.
"""

import os
import json
import logging
import torch
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import time

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
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.model_selection import train_test_split

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup simple logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'fast_training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_and_process_data(data_file: str, config: Dict[str, Any], logger):
    """Load and process data with minimal overhead."""
    logger.info(f"Loading data from {data_file}")
    
    # Load JSONL data
    data = []
    total_lines = 0
    skipped = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            total_lines += 1
            try:
                item = json.loads(line.strip())
                if all(field in item for field in ["instruction", "input", "output", "metadata"]):
                    if "environment" in item["metadata"]:
                        data.append({
                            "instruction": str(item["instruction"]).strip(),
                            "input": str(item["input"]).strip(),
                            "output": str(item["output"]).strip(),
                            "environment": str(item["metadata"]["environment"]).strip()
                        })
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue
    
    logger.info(f"Processed {total_lines} total lines")
    logger.info(f"Loaded {len(data)} valid examples")
    logger.info(f"Skipped {skipped} invalid examples")
    
    # Improved deduplication that considers input variations
    logger.info("Deduplicating data...")
    seen = set()
    deduplicated = []
    for item in data:
        # Create a key that includes more of the input context to preserve variations
        # Extract step, drift, waypoint info from input to differentiate similar scenarios
        input_text = item['input']
        key = f"{item['environment']}|||{input_text}|||{item['output'][:50]}"
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    
    logger.info(f"After deduplication: {len(deduplicated)} unique examples")
    
    # If too much data was removed, disable deduplication 
    if len(deduplicated) < len(data) * 0.3:  # If less than 30% remains
        logger.warning(f"Deduplication removed too much data ({len(data)} -> {len(deduplicated)}), using less aggressive deduplication...")
        
        # Try less aggressive deduplication - only remove exact duplicates
        seen = set()
        deduplicated = []
        for item in data:
            key = f"{item['instruction']}|||{item['input']}|||{item['output']}"
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        
        logger.info(f"After less aggressive deduplication: {len(deduplicated)} unique examples")
    
    # Format for training
    def format_example(example):
        return {
            "text": f"""<|start_header_id|>system<|end_header_id|>

You are an expert air traffic controller with extensive experience in conflict resolution and aircraft separation.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        }
    
    formatted_data = [format_example(item) for item in deduplicated]
    
    # Split data
    train_data, temp_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def setup_model_and_tokenizer(config: Dict[str, Any], logger):
    """Setup model with speed optimizations."""
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model
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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fast SAC LoRA Training")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--data", required=True, help="Path to training data JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir / "logs"))
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting fast SAC LoRA training...")
    
    # Load data
    train_dataset, val_dataset = load_and_process_data(args.data, config, logger)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config, logger)
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=list(config["lora"]["target_modules"]),
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.train()
    
    # Tokenize datasets
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
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Training arguments - optimized for speed
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        save_safetensors=True,
        prediction_loss_only=True,
        group_by_length=True,
        report_to=[],
        disable_tqdm=False,
        skip_memory_metrics=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
