#!/usr/bin/env python3
"""
Quick Performance Test Runner
Tests different configurations to find optimal settings for RTX 5070 Ti
"""

import subprocess
import sys
import time
import yaml
from pathlib import Path

def test_configuration(config_name, config_updates):
    """Test a specific configuration and measure performance."""
    print(f"\n{'='*50}")
    print(f"Testing: {config_name}")
    print(f"{'='*50}")
    
    # Load base config
    config_file = Path("config/training_config.yaml")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply updates
    for key_path, value in config_updates.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    
    # Save test config
    test_config_file = Path(f"config/test_{config_name.lower().replace(' ', '_')}.yaml")
    with open(test_config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training for a few steps
    cmd = [
        sys.executable,
        "training/train_enhanced_sac_lora.py",
        "--config", str(test_config_file),
        "--data", "data/combined_atc_training.jsonl",
        "--output-dir", f"models/test_{config_name.lower().replace(' ', '_')}"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # Run for just 10 training steps to measure speed
        result = subprocess.run(cmd, timeout=300, capture_output=True, text=True)  # 5 minute timeout
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ {config_name}: Completed in {duration:.1f}s")
            
            # Extract some metrics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Check last 20 lines
                if "step" in line.lower() and ("loss" in line.lower() or "time" in line.lower()):
                    print(f"   {line.strip()}")
                    
        else:
            print(f"❌ {config_name}: Failed")
            print(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {config_name}: Timeout (too slow)")
    except Exception as e:
        print(f"❌ {config_name}: Exception - {e}")

def main():
    """Run performance tests with different configurations."""
    print("RTX 5070 Ti Performance Testing")
    print("Testing different configurations to find optimal settings...")
    
    # Test configurations
    test_configs = [
        ("Conservative", {
            "training.batch_size": 4,
            "training.gradient_accumulation_steps": 2,
            "data.max_sequence_length": 1024,
            "lora.rank": 16,
            "hardware.dataloader_num_workers": 2
        }),
        
        ("Optimized", {
            "training.batch_size": 8,
            "training.gradient_accumulation_steps": 1,
            "data.max_sequence_length": 1024,
            "lora.rank": 32,
            "hardware.dataloader_num_workers": 4
        }),
        
        ("Aggressive", {
            "training.batch_size": 12,
            "training.gradient_accumulation_steps": 1,
            "data.max_sequence_length": 1024,
            "lora.rank": 32,
            "hardware.dataloader_num_workers": 6
        }),
        
        ("Short Sequence", {
            "training.batch_size": 16,
            "training.gradient_accumulation_steps": 1,
            "data.max_sequence_length": 512,
            "lora.rank": 32,
            "hardware.dataloader_num_workers": 4
        })
    ]
    
    results = []
    
    for config_name, config_updates in test_configs:
        try:
            test_configuration(config_name, config_updates)
            results.append((config_name, "Completed"))
        except Exception as e:
            results.append((config_name, f"Failed: {e}"))
    
    # Summary
    print(f"\n{'='*50}")
    print("PERFORMANCE TEST SUMMARY")
    print(f"{'='*50}")
    
    for config_name, result in results:
        print(f"{config_name:15}: {result}")
    
    print(f"\nRecommendations:")
    print("1. Use the fastest successful configuration")
    print("2. Monitor GPU memory usage during training")
    print("3. Adjust batch size based on available VRAM")
    print("4. Consider reducing sequence length if memory limited")

if __name__ == "__main__":
    main()
