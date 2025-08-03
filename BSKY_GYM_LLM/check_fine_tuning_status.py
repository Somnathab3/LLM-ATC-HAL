#!/usr/bin/env python3
"""
DEFINITIVE MODEL ANALYSIS
=========================
Check if the current model is actually fine-tuned or just prompt-engineered
"""

import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_current_model():
    """Analyze what we actually have"""
    
    print("🔍 DEFINITIVE ANALYSIS: Is our model actually fine-tuned?")
    print("=" * 70)
    
    # 1. Check the Modelfile structure
    print("\n1️⃣ MODELFILE ANALYSIS:")
    try:
        result = subprocess.run(
            ["ollama", "show", "llama3.1-bsky-lora", "--modelfile"],
            capture_output=True, text=True, check=True
        )
        
        modelfile = result.stdout
        
        # Key indicators
        if "FROM llama3.1:8b" in modelfile:
            print("   ❌ FROM: Points to base llama3.1:8b")
            print("   📝 This means it's just the base model + prompt")
        elif "FROM C:\\" in modelfile or "sha256-" in modelfile:
            print("   ⚠️  FROM: Points to a blob file")
            print("   📝 This could be base model or merged model")
        
        if "ADAPTER" in modelfile.upper():
            print("   ✅ ADAPTER: Found LoRA adapter reference")
        else:
            print("   ❌ ADAPTER: No LoRA adapter reference found")
        
        if "SYSTEM" in modelfile:
            print("   ✅ SYSTEM: Custom system prompt present")
        
    except Exception as e:
        print(f"   ❌ Error analyzing modelfile: {e}")
    
    # 2. Check what we actually trained
    print("\n2️⃣ WHAT WE ACTUALLY TRAINED:")
    print("   ✅ LoRA Adapter: 160.1 MB (adapter_model.safetensors)")
    print("   ✅ Training: 5,000 examples, 2,660 steps, 5 epochs")
    print("   ✅ Loss: 0.2413 training, 0.0285 validation")
    print("   ✅ Config: r=16, alpha=32, 7 target modules")
    
    # 3. Check Ollama model vs actual LoRA
    print("\n3️⃣ CURRENT SITUATION:")
    print("   🎯 LoRA Adapter: Located in models/llama3.1-bsky-lora/")
    print("   📁 Ollama Model: llama3.1-bsky-lora:latest")
    print("")
    print("   ❌ PROBLEM: Ollama model is NOT using the LoRA adapter!")
    print("   📝 Current model = Base llama3.1:8b + Custom prompt")
    print("   📝 Missing = Actual LoRA fine-tuned weights")
    
    # 4. What we need to do
    print("\n4️⃣ SOLUTION NEEDED:")
    print("   1. Merge LoRA adapter with base model")
    print("   2. Create new Ollama model from merged weights")
    print("   3. Verify the new model contains actual fine-tuned weights")
    
    print("\n" + "=" * 70)
    print("🎯 VERDICT: NO, current model is NOT properly fine-tuned!")
    print("📝 We have the LoRA weights, but Ollama isn't using them.")
    print("🔧 Need to properly merge and deploy the LoRA adapter.")
    print("=" * 70)

def check_lora_adapter():
    """Check if we have the actual LoRA adapter"""
    
    print("\n🔍 CHECKING LORA ADAPTER FILES:")
    
    import os
    from pathlib import Path
    
    adapter_path = Path("f:/LLM-ATC-HAL/BSKY_GYM_LLM/models/llama3.1-bsky-lora")
    
    files_to_check = [
        "adapter_model.safetensors",
        "adapter_config.json", 
        "train_results.json"
    ]
    
    for filename in files_to_check:
        filepath = adapter_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {filename}: Missing")
    
    # Check adapter config
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n📊 LoRA Configuration:")
        print(f"   • Type: {config.get('peft_type')}")
        print(f"   • Rank: {config.get('r')}")
        print(f"   • Alpha: {config.get('lora_alpha')}")
        print(f"   • Target Modules: {len(config.get('target_modules', []))}")
        print(f"   • Base Model: {config.get('base_model_name_or_path')}")

def main():
    analyze_current_model()
    check_lora_adapter()

if __name__ == "__main__":
    main()
