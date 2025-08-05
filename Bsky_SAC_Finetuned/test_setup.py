#!/usr/bin/env python3
"""
Test Script for Enhanced SAC LoRA Training Setup

This script verifies that all dependencies and files are properly set up
for the enhanced training pipeline.
"""

import sys
import os
from pathlib import Path
import importlib

def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists and report status."""
    if file_path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False

def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError:
        print(f"❌ {module_name} (NOT AVAILABLE)")
        return False

def main():
    """Main test function."""
    print("🧪 Enhanced SAC LoRA Training Setup Test")
    print("=" * 50)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Check essential files
    print("\n📁 Checking Essential Files:")
    files_ok = True
    
    essential_files = [
        (script_dir / "config" / "training_config.yaml", "Training config"),
        (script_dir / "training" / "train_enhanced_sac_lora.py", "Enhanced training script"),
        (script_dir / "run_training.py", "Runner script"),
        (script_dir / "requirements.txt", "Requirements file"),
        (script_dir / "data" / "combined_atc_training.jsonl", "Training data")
    ]
    
    for file_path, description in essential_files:
        if not check_file_exists(file_path, description):
            files_ok = False
    
    # Check Python dependencies
    print("\n📦 Checking Python Dependencies:")
    deps_ok = True
    
    essential_deps = [
        "torch",
        "transformers", 
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "yaml"
    ]
    
    for dep in essential_deps:
        if not check_import(dep):
            deps_ok = False
    
    # Check data file size
    print("\n📊 Checking Data File:")
    data_file = script_dir / "data" / "combined_atc_training.jsonl"
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"✅ Data file size: {size_mb:.1f} MB")
        
        # Quick line count
        try:
            with open(data_file, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"✅ Data file contains: {line_count:,} lines")
        except Exception as e:
            print(f"⚠️  Could not count lines: {e}")
    
    # Check GPU availability
    print("\n🎮 Checking GPU Availability:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            print(f"✅ GPU Available: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            print(f"✅ GPU Count: {gpu_count}")
        else:
            print("⚠️  No GPU available - training will use CPU (very slow)")
    except ImportError:
        print("❌ PyTorch not available - cannot check GPU")
    
    # Final status
    print("\n🎯 Setup Status:")
    if files_ok and deps_ok:
        print("✅ Setup appears to be complete!")
        print("\n🚀 You can now run training with:")
        print("   python run_training.py")
        print("\n📚 For more options, see:")
        print("   python run_training.py --help")
        return 0
    else:
        print("❌ Setup is incomplete!")
        if not files_ok:
            print("   - Some essential files are missing")
        if not deps_ok:
            print("   - Some Python dependencies are missing")
            print("   - Try: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
