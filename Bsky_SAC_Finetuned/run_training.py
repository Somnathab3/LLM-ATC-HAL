#!/usr/bin/env python3
"""
SAC LoRA Training Runner Script

This script provides an easy way to run the enhanced SAC LoRA training
with sensible defaults and progress tracking.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Main runner function."""
    parser = argparse.ArgumentParser(description="Run Enhanced SAC LoRA Training")
    parser.add_argument("--config", 
                       default="config/training_config.yaml",
                       help="Path to training config YAML (default: config/training_config.yaml)")
    parser.add_argument("--data", 
                       default="data/combined_atc_training.jsonl",
                       help="Path to training data JSONL (default: data/combined_atc_training.jsonl)")
    parser.add_argument("--output-dir", 
                       default="models/llama3.1-bsky-sac-lora",
                       help="Output directory for trained model (default: models/llama3.1-bsky-sac-lora)")
    parser.add_argument("--resume", 
                       help="Path to checkpoint to resume from")
    parser.add_argument("--install-deps", 
                       action="store_true",
                       help="Install dependencies before training")
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Convert relative paths to absolute paths based on script location
    config_path = script_dir / args.config
    data_path = script_dir / args.data
    output_dir = script_dir / args.output_dir
    
    print("[START] Enhanced SAC LoRA Training Runner")
    print("=" * 50)
    print(f"[FOLDER] Script directory: {script_dir}")
    print(f"[CONFIG]  Config file: {config_path}")
    print(f"[DATA] Data file: {data_path}")
    print(f"[SAVE] Output directory: {output_dir}")
    
    # Check if files exist
    if not config_path.exists():
        print(f"[ERROR] Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if not data_path.exists():
        print(f"[ERROR] Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        print("\n📦 Installing dependencies...")
        requirements_file = script_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        else:
            print("[WARNING]  Warning: requirements.txt not found, skipping dependency installation")
    
    # Prepare command
    training_script = script_dir / "training" / "train_enhanced_sac_lora.py"
    
    if not training_script.exists():
        print(f"[ERROR] Error: Training script not found: {training_script}")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        str(training_script),
        "--config", str(config_path),
        "--data", str(data_path),
        "--output-dir", str(output_dir)
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    print(f"\n[TARGET] Starting training...")
    print(f"[COMMAND] Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("\n[COMPLETE] Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[STOP]  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
