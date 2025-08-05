#!/usr/bin/env python3
"""
Setup Script for BlueSky-Gym ATC Fine-tuning System
==================================================

This script helps set up the environment, install dependencies,
and prepare the system for ATC LLM fine-tuning.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], description: str, check: bool = True) -> bool:
    """Run a command and handle errors gracefully"""
    print(f"\n📋 {description}")
    print(f"🔄 Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            return True
        else:
            print(f"❌ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_cuda_availability():
    """Check CUDA availability for GPU training"""
    print("\n🔍 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available with {gpu_count} GPU(s): {gpu_name}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU training (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - CUDA check will be performed after installation")
        return False


def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "models",
        "training_data", 
        "data_generation/sac_models",
        "data_generation/expert_demos",
        "logs/training",
        "logs/evaluation",
        "results/plots",
        "results/reports"
    ]
    
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    # Install core dependencies
    success = run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        "Installing core dependencies"
    )
    
    if not success:
        print("❌ Failed to install dependencies")
        return False
    
    # Try to install optional BlueSky-Gym if available
    print("\n📦 Attempting to install BlueSky-Gym (optional)...")
    blusky_success = run_command(
        [sys.executable, "-m", "pip", "install", "blusky-gym"],
        "Installing BlueSky-Gym",
        check=False
    )
    
    if not blusky_success:
        print("⚠️  BlueSky-Gym not available - will use mock environments")
    
    return True


def verify_installation():
    """Verify that key packages are installed"""
    print("\n🔍 Verifying installation...")
    
    key_packages = [
        "torch",
        "transformers", 
        "peft",
        "stable_baselines3",
        "gymnasium",
        "numpy",
        "pandas",
        "yaml"
    ]
    
    failed_packages = []
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to import: {', '.join(failed_packages)}")
        return False
    
    print("\n✅ All key packages verified!")
    return True


def download_base_models():
    """Offer to download base models"""
    print("\n🤖 Base Model Setup")
    print("The system uses Llama-2 models as base models for fine-tuning.")
    print("These models require HuggingFace authentication and agreement to license terms.")
    
    response = input("\nWould you like to test model loading? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🔄 Testing model access...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Try to load a small test model first
            model_name = "microsoft/DialoGPT-small"  # Small test model
            print(f"🔄 Loading test model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✅ Tokenizer loaded successfully")
            
            # Don't actually load the full model to save time/memory
            print("✅ Model loading test completed")
            
        except Exception as e:
            print(f"❌ Model loading test failed: {e}")
            print("💡 You may need to:")
            print("   1. Install git-lfs: https://git-lfs.github.io/")
            print("   2. Login to HuggingFace: huggingface-cli login")
            print("   3. Accept model license terms on HuggingFace")
            return False
    
    return True


def create_sample_config():
    """Create a sample configuration file"""
    print("\n⚙️  Creating sample configuration...")
    
    sample_config = """# Sample configuration for testing
environment:
  name: "HorizontalCREnv-v0"
  max_episode_length: 200
  observation_space_size: 8
  action_space_size: 1

sac_model:
  path: "data_generation/sac_models/horizontal_cr_model.zip"
  device: "auto"

data_generation:
  num_episodes: 10  # Small number for testing
  num_samples_per_episode: 5
  reasoning_templates:
    - "conflict_avoidance"
    - "separation_maintenance"

llm_training:
  base_model: "microsoft/DialoGPT-small"  # Small model for testing
  max_seq_length: 512
  batch_size: 2
  learning_rate: 0.0002
  num_epochs: 1
  lora_config:
    r: 8
    lora_alpha: 16
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.1

prompts:
  system_prompt: |
    You are an expert air traffic controller responsible for ensuring
    safe and efficient aircraft operations. Analyze each situation
    carefully and provide clear, actionable guidance.
"""
    
    config_path = Path(__file__).parent / "configs" / "test_config.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(sample_config)
    
    print(f"✅ Sample config created: {config_path}")
    return True


def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    test_script = """
import sys
sys.path.insert(0, '.')

# Test imports
try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    print("✅ Core imports successful")
    
    # Test basic functionality
    data = np.random.randn(10, 5)
    print(f"✅ NumPy working: {data.shape}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} devices")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    print("✅ Basic functionality test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
"""
    
    test_file = Path(__file__).parent / "test_setup.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    success = run_command(
        [sys.executable, str(test_file)],
        "Running basic functionality test"
    )
    
    # Clean up test file
    test_file.unlink()
    
    return success


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Obtain SAC models for your environments:")
    print("   - Place .zip model files in data_generation/sac_models/")
    print("   - Or use the system in mock mode for testing")
    
    print("\n2. Generate training data:")
    print("   cd scripts")
    print("   python generate_training_data.py --environment HorizontalCREnv-v0 --num_samples 100")
    
    print("\n3. Fine-tune a model:")
    print("   python finetune_llama.py --config ../configs/test_config.yaml")
    
    print("\n4. Evaluate the model:")
    print("   python test_models.py")
    
    print("\n💡 Tips:")
    print("- Start with small datasets and models for testing")
    print("- Check logs/ directory for detailed training logs")
    print("- Use --help flag with scripts for more options")
    print("- Review README.md for comprehensive documentation")
    
    print("\n🔗 Useful Commands:")
    print("- Check CUDA: python -c 'import torch; print(torch.cuda.is_available())'")
    print("- HuggingFace login: huggingface-cli login")
    print("- List environments: python -c 'import gymnasium; print(gymnasium.envs.registry.all())'")


def main():
    """Main setup function"""
    print("🚀 BlueSky-Gym ATC Fine-tuning System Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directory_structure():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Verify installation
    if not verify_installation():
        return False
    
    # Check CUDA
    check_cuda_availability()
    
    # Test model access
    if not download_base_models():
        print("⚠️  Model access test failed - you may need additional setup")
    
    # Create sample config
    if not create_sample_config():
        return False
    
    # Run basic test
    if not run_basic_test():
        return False
    
    # Print next steps
    print_next_steps()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)
