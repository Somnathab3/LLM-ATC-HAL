#!/usr/bin/env python3
"""
LLM Fine-tuning Script for ATC Decision-Making
==============================================

This script fine-tunes Llama models using LoRA for environment-specific 
ATC decision-making based on expert SAC policy demonstrations.

Features:
- LoRA-based fine-tuning for parameter efficiency
- Environment-specific model training
- Comprehensive evaluation and validation
- Integration with HuggingFace Transformers and PEFT
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning process"""
    environment_name: str
    base_model_name: str
    training_data_path: str
    output_dir: str
    lora_config: Dict[str, Any]
    training_args: Dict[str, Any]
    prompts: Dict[str, str]


class ATCDatasetProcessor:
    """Processes ATC training data for LLM fine-tuning"""
    
    def __init__(self, config: FineTuningConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = 2048
        
    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from JSON file"""
        with open(self.config.training_data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training samples from {self.config.training_data_path}")
        return data
    
    def format_training_sample(self, sample: Dict[str, Any]) -> str:
        """Format a training sample into prompt-response format"""
        system_prompt = self.config.prompts["system_prompt"]
        
        # Extract sample information
        scenario_desc = sample.get("scenario_description", "")
        obs_summary = sample.get("observation_summary", "")
        expert_action = sample.get("expert_action", "")
        reasoning = sample.get("reasoning", "")
        
        # Create user prompt with observation details
        user_prompt = f"""Scenario: {scenario_desc}

Current Situation: {obs_summary}

Based on this air traffic control situation, what action should be taken?

Consider safety requirements, efficiency, and standard ATC procedures in your response."""
        
        # Create assistant response with action and reasoning
        assistant_response = f"""{expert_action}

Reasoning: {reasoning}"""
        
        # Format as conversation
        conversation = f"""<|system|>
{system_prompt}

<|user|>
{user_prompt}

<|assistant|>
{assistant_response}<|end|>"""
        
        return conversation
    
    def create_dataset(self) -> Dataset:
        """Create HuggingFace dataset from training samples"""
        training_data = self.load_training_data()
        
        # Format samples
        formatted_samples = []
        for sample in training_data:
            formatted_text = self.format_training_sample(sample)
            formatted_samples.append({"text": formatted_text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_samples)
        
        # Tokenize dataset
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )
            
            # Set labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info(f"Created tokenized dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.9) -> tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets"""
        split_dataset = dataset.train_test_split(test_size=1-train_ratio, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
        return train_dataset, eval_dataset


class ATCModelTrainer:
    """Handles LLM fine-tuning for ATC decision-making"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()
        self.dataset_processor = ATCDatasetProcessor(config, self.tokenizer)
        
    def _setup_tokenizer(self):
        """Setup tokenizer with special tokens"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True
        )
        
        # Add special tokens if not present
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|end|>",
            "bos_token": "<|begin|>",
        }
        
        added_tokens = []
        for token_type, token in special_tokens.items():
            if getattr(tokenizer, token_type) is None:
                added_tokens.append(token)
                setattr(tokenizer, token_type, token)
        
        if added_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
            logger.info(f"Added special tokens: {added_tokens}")
        
        return tokenizer
    
    def _setup_model(self):
        """Setup base model with LoRA configuration"""
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Resize embeddings if we added tokens
        if len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized embeddings to {len(self.tokenizer)}")
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **self.config.lora_config
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def prepare_training_data(self) -> tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        dataset = self.dataset_processor.create_dataset()
        train_dataset, eval_dataset = self.dataset_processor.split_dataset(dataset)
        return train_dataset, eval_dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset) -> Trainer:
        """Setup HuggingFace Trainer"""
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            **self.config.training_args,
            report_to=None,  # Disable wandb/tensorboard for now
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self) -> None:
        """Execute the training process"""
        logger.info(f"Starting fine-tuning for {self.config.environment_name}")
        
        # Prepare data
        train_dataset, eval_dataset = self.prepare_training_data()
        
        # Setup trainer
        trainer = self.setup_trainer(train_dataset, eval_dataset)
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        self._save_training_metrics(trainer)
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
    
    def _save_training_metrics(self, trainer: Trainer) -> None:
        """Save training metrics and configuration"""
        metrics = {
            "training_config": {
                "environment": self.config.environment_name,
                "base_model": self.config.base_model_name,
                "lora_config": self.config.lora_config,
                "training_args": self.config.training_args,
            },
            "final_metrics": trainer.state.log_history[-1] if trainer.state.log_history else {},
        }
        
        metrics_file = Path(self.config.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_file}")


class ConfigurationManager:
    """Manages configuration loading and validation"""
    
    @staticmethod
    def load_config(config_path: str) -> FineTuningConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract configuration sections
        env_config = config_dict["environment"]
        llm_config = config_dict["llm_training"]
        prompts = config_dict["prompts"]
        
        # Determine training data path
        env_name = env_config["name"]
        data_file_map = {
            "HorizontalCREnv-v0": "horizontal_cr_samples.json",
            "VerticalCREnv-v0": "vertical_cr_samples.json",
            "SectorCREnv-v0": "sector_cr_samples.json",
            "MergeEnv-v0": "merge_samples.json"
        }
        
        base_path = Path(config_path).parent.parent
        training_data_path = base_path / "training_data" / data_file_map[env_name]
        
        return FineTuningConfig(
            environment_name=env_name,
            base_model_name=llm_config["base_model"],
            training_data_path=str(training_data_path),
            output_dir=llm_config["training_args"]["output_dir"],
            lora_config=llm_config["lora_config"],
            training_args=llm_config["training_args"],
            prompts=prompts
        )
    
    @staticmethod
    def validate_config(config: FineTuningConfig) -> bool:
        """Validate configuration"""
        # Check if training data exists
        if not Path(config.training_data_path).exists():
            logger.error(f"Training data not found: {config.training_data_path}")
            return False
        
        # Check if output directory can be created
        output_dir = Path(config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create output directory {output_dir}: {e}")
            return False
        
        # Validate LoRA configuration
        required_lora_keys = ["r", "lora_alpha", "target_modules", "lora_dropout"]
        for key in required_lora_keys:
            if key not in config.lora_config:
                logger.error(f"Missing LoRA configuration key: {key}")
                return False
        
        return True


def fine_tune_single_environment(config_path: str) -> bool:
    """Fine-tune model for a single environment"""
    try:
        # Load and validate configuration
        config = ConfigurationManager.load_config(config_path)
        if not ConfigurationManager.validate_config(config):
            return False
        
        # Create trainer and train model
        trainer = ATCModelTrainer(config)
        trainer.train()
        
        return True
        
    except Exception as e:
        logger.error(f"Fine-tuning failed for {config_path}: {e}")
        return False


def main():
    """Main function to fine-tune models for all environments"""
    base_path = Path(__file__).parent.parent
    
    config_files = [
        "horizontal_config.yaml",
        "vertical_config.yaml", 
        "sector_config.yaml",
        "merge_config.yaml"
    ]
    
    results = {}
    
    for config_file in config_files:
        config_path = base_path / "configs" / config_file
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            results[config_file] = False
            continue
        
        logger.info(f"Starting fine-tuning for {config_file}")
        success = fine_tune_single_environment(str(config_path))
        results[config_file] = success
        
        if success:
            logger.info(f"Successfully completed fine-tuning for {config_file}")
        else:
            logger.error(f"Fine-tuning failed for {config_file}")
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    
    logger.info(f"Fine-tuning completed: {successful}/{total} environments successful")
    
    # Save results summary
    results_file = base_path / "models" / "training_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "successful": successful,
                "total": total,
                "success_rate": successful / total if total > 0 else 0
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results summary saved to {results_file}")


if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available. Training will be slow on CPU.")
    
    main()
