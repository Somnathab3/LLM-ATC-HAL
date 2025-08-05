#!/usr/bin/env python3
"""
Combine All Training Datasets and Train Ollama Model

This script combines all environment-specific training datasets into a unified format
and trains an Ollama model using LoRA fine-tuning for air traffic control tasks.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ollama_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatasetCombiner:
    """Combines multiple environment training datasets into unified format."""
    
    def __init__(self, training_data_dir: str, output_dir: str):
        self.training_data_dir = Path(training_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define environment dataset files
        self.dataset_files = {
            'horizontal': 'horizontal_cr_samples.json',
            'vertical': 'vertical_cr_samples.json',
            'sector': 'sector_cr_samples.json',
            'merge': 'merge_samples.json'
        }
        
    def load_dataset(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load a single dataset file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {filepath.name}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return []
    
    def convert_to_jsonl_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert training sample to JSONL format for Ollama training."""
        instruction = "As an air traffic controller, analyze the current situation and provide the appropriate action with explanation."
        
        # Extract environment info and scenario
        environment_name = sample.get('environment', 'Unknown')
        scenario = sample.get('scenario_description', '')
        
        # Format input with environment context and scenario
        input_text = f"Environment: {environment_name}\n\n{scenario}"
        
        # Extract expert action and reasoning
        expert_action = sample.get('expert_action', '')
        reasoning = sample.get('reasoning', '')
        
        # Format output with action and explanation
        output_text = f"Action: {expert_action}\n\nExplanation: {reasoning}"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "environment": environment_name,
                "source_file": sample.get('source_file', ''),
                "generated_timestamp": datetime.now().isoformat()
            }
        }
    
    def combine_datasets(self) -> str:
        """Combine all datasets into unified JSONL format."""
        combined_data: List[Dict[str, Any]] = []
        
        # Load and process each environment dataset
        for filename in self.dataset_files.values():
            filepath = self.training_data_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Dataset file not found: {filepath}")
                continue
                
            dataset = self.load_dataset(filepath)
            
            # Convert each sample to JSONL format
            for sample in dataset:
                sample['source_file'] = filename
                jsonl_sample = self.convert_to_jsonl_format(sample)
                combined_data.append(jsonl_sample)
        
        # Save combined dataset as JSONL
        output_file = self.output_dir / "combined_atc_training.jsonl"
        
        with open(output_file, 'w') as f:
            for sample in combined_data:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Combined {len(combined_data)} samples into {output_file}")
        
        # Generate statistics
        self._generate_statistics(combined_data, output_file.parent / "combination_stats.json")
        
        return str(output_file)
    
    def _generate_statistics(self, data: List[Dict[str, Any]], output_file: Path):
        """Generate statistics about the combined dataset."""
        stats: Dict[str, Any] = {
            "total_samples": len(data),
            "environments": {},
            "generation_timestamp": datetime.now().isoformat()
        }
        
        # Count samples per environment
        for sample in data:
            env = sample["metadata"]["environment"]
            stats["environments"][env] = stats["environments"].get(env, 0) + 1
        
        # Save statistics
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {output_file}")
        
        # Log summary
        logger.info("Dataset Combination Summary:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        for env, count in stats["environments"].items():
            logger.info(f"  {env}: {count} samples")


class OllamaTrainer:
    """Handles Ollama model training with LoRA fine-tuning."""
    
    def __init__(self, config_file: str, data_file: str):
        self.config_file = Path(config_file)
        self.data_file = Path(data_file)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded training config from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def prepare_training_environment(self):
        """Prepare the training environment and dependencies."""
        logger.info("Preparing training environment...")
        
        # Check if required directories exist
        model_dir = Path("models/llama3.1-bsky-lora")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Verify data file exists
        if not self.data_file.exists():
            raise FileNotFoundError(f"Training data file not found: {self.data_file}")
        
        logger.info("Training environment prepared successfully")
    
    def start_training(self):
        """Start the LoRA fine-tuning process."""
        logger.info("Starting LoRA fine-tuning for Ollama model...")
        
        try:
            # Prepare environment
            self.prepare_training_environment()
            
            # Run training script with combined dataset support
            training_script = Path("BSKY_GYM_LLM/train_combined_lora.py")
            
            if not training_script.exists():
                raise FileNotFoundError(f"Training script not found: {training_script}")
            
            # Construct training command
            cmd = [
                sys.executable, 
                str(training_script),
                "--config", str(self.config_file),
                "--data", str(self.data_file),
                "--output-dir", "models/llama3.1-bsky-lora"
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute training
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Training completed successfully!")
                logger.info(f"Training output:\n{result.stdout}")
            else:
                logger.error(f"Training failed with return code {result.returncode}")
                logger.error(f"Error output:\n{result.stderr}")
                raise RuntimeError("Training process failed")
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def convert_to_ollama(self):
        """Convert the trained LoRA model to Ollama format."""
        logger.info("Converting trained model to Ollama format...")
        
        try:
            # Run conversion script
            conversion_script = Path("BSKY_GYM_LLM/convert_to_ollama.py")
            
            if not conversion_script.exists():
                raise FileNotFoundError(f"Conversion script not found: {conversion_script}")
            
            cmd: List[str] = [
                sys.executable,
                str(conversion_script),
                "--model-dir", "models/llama3.1-bsky-lora",
                "--output-name", self.config["model"]["output_name"]
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Model conversion completed successfully!")
                logger.info(f"Conversion output:\n{result.stdout}")
            else:
                logger.error(f"Conversion failed with return code {result.returncode}")
                logger.error(f"Error output:\n{result.stderr}")
                raise RuntimeError("Conversion process failed")
                
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise


def main():
    """Main execution function."""
    logger.info("Starting Ollama model training pipeline...")
    
    try:
        # Configuration
        training_data_dir = "Bsky_SAC_Finetuned/training_data"
        output_dir = "BSKY_GYM_LLM/data"
        config_file = "BSKY_GYM_LLM/config/training_config.yaml"
        
        # Step 1: Combine datasets
        logger.info("Step 1: Combining training datasets...")
        combiner = DatasetCombiner(training_data_dir, output_dir)
        combined_data_file = combiner.combine_datasets()
        
        # Step 2: Train model
        logger.info("Step 2: Training Ollama model...")
        trainer = OllamaTrainer(config_file, combined_data_file)
        trainer.start_training()
        
        # Step 3: Convert to Ollama format
        logger.info("Step 3: Converting to Ollama format...")
        trainer.convert_to_ollama()
        
        logger.info("Ollama model training pipeline completed successfully!")
        logger.info("Your trained model is now available for use with Ollama.")
        
        # Provide usage instructions
        model_name = trainer.config["model"]["output_name"]
        logger.info(f"\nTo use your trained model:")
        logger.info(f"  ollama run {model_name}")
        logger.info(f"\nExample prompt:")
        logger.info(f'  "As an air traffic controller, what action should I take for two aircraft approaching each other?"')
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
