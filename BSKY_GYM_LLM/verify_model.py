#!/usr/bin/env python3
"""Simple verification that the model was saved correctly"""

import os
import json

def verify_model():
    print("🔍 Verifying Enhanced LoRA Model...")
    print("=" * 50)
    
    model_path = "models/llama3.1-bsky-lora"
    
    # Check essential files
    essential_files = [
        "adapter_model.safetensors",
        "adapter_config.json", 
        "tokenizer.json",
        "train_results.json",
        "eval_results.json"
    ]
    
    print("📁 Checking essential files:")
    all_exist = True
    for file in essential_files:
        full_path = os.path.join(model_path, file)
        exists = os.path.exists(full_path)
        size = os.path.getsize(full_path) if exists else 0
        print(f"  {'✅' if exists else '❌'} {file}: {size:,} bytes")
        if not exists:
            all_exist = False
    
    # Check training results
    if os.path.exists(os.path.join(model_path, "train_results.json")):
        with open(os.path.join(model_path, "train_results.json"), 'r') as f:
            train_results = json.load(f)
        print(f"\n📊 Training Results:")
        print(f"  Final Training Loss: {train_results.get('train_loss', 'N/A'):.4f}")
        print(f"  Training Runtime: {train_results.get('train_runtime', 'N/A'):.1f} seconds")
        print(f"  Epochs Completed: {train_results.get('epoch', 'N/A')}")
    
    if os.path.exists(os.path.join(model_path, "eval_results.json")):
        with open(os.path.join(model_path, "eval_results.json"), 'r') as f:
            eval_results = json.load(f)
        print(f"  Final Validation Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    
    # Check adapter size
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        adapter_size = os.path.getsize(adapter_path)
        print(f"\n🧠 Adapter Model:")
        print(f"  Size: {adapter_size:,} bytes ({adapter_size/1024/1024:.1f} MB)")
        print(f"  Expected: ~160-170 MB for LoRA adapter")
    
    # Check checkpoints
    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
    print(f"\n💾 Checkpoints Found: {len(checkpoints)}")
    for checkpoint in sorted(checkpoints):
        print(f"  📁 {checkpoint}")
    
    print("\n" + "=" * 50)
    if all_exist and adapter_size > 100_000_000:  # At least 100MB
        print("🎉 MODEL VERIFICATION SUCCESSFUL!")
        print("✅ All essential files present")
        print("✅ Adapter model properly saved")
        print("✅ Training completed successfully")
        print("\n🚀 Your fine-tuned model is ready for use!")
    else:
        print("⚠️ MODEL VERIFICATION ISSUES DETECTED")
        print("Some files may be missing or incomplete")
    
    print("=" * 50)
    return all_exist

if __name__ == "__main__":
    verify_model()
