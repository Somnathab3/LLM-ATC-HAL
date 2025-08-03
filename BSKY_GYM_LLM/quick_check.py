#!/usr/bin/env python3
"""
Simple Model Fine-tuning Verification
====================================
Quick check to see if models are actually fine-tuned.
"""

import subprocess
from pathlib import Path

def check_lora_weights():
    """Check if we have real LoRA weights"""
    print("🔍 CHECKING LORA ADAPTER WEIGHTS")
    print("=" * 50)
    
    adapter_path = Path("models/llama3.1-bsky-lora/adapter_model.safetensors")
    
    if adapter_path.exists():
        size_mb = adapter_path.stat().st_size / (1024 * 1024)
        print(f"📁 LoRA adapter file: {size_mb:.1f} MB")
        
        if size_mb > 100:
            print("✅ REAL FINE-TUNED WEIGHTS FOUND!")
            print("   (Large adapter file indicates actual training)")
            return True
        else:
            print("❌ File too small - likely not trained")
            return False
    else:
        print("❌ No LoRA adapter found")
        return False

def check_ollama_modelfile():
    """Check Ollama modelfile structure"""
    print("\n🔍 CHECKING OLLAMA MODEL")
    print("=" * 50)
    
    try:
        # Use PowerShell to avoid encoding issues
        result = subprocess.run(
            ["powershell", "-Command", "ollama show llama3.1-bsky-lora --modelfile | Select-String 'FROM'"],
            capture_output=True, text=True, check=True
        )
        
        from_line = result.stdout.strip()
        print(f"📋 Ollama FROM line: {from_line}")
        
        if "llama3.1:8b" in from_line:
            print("❌ USES BASE MODEL - NOT FINE-TUNED!")
            print("   (Just references base model with custom prompt)")
            return False
        elif "blobs/sha256" in from_line:
            print("❌ USES BLOB FILE - LIKELY NOT FINE-TUNED!")
            print("   (Probably just a copy of base model)")
            return False
        else:
            print("✅ USES CUSTOM MODEL FILE")
            return True
            
    except Exception as e:
        print(f"❌ Could not check Ollama model: {e}")
        return False

def simple_response_test():
    """Simple test of model responses"""
    print("\n🔍 TESTING MODEL RESPONSES")
    print("=" * 50)
    
    try:
        import ollama
        client = ollama.Client()
        
        # Test ATC-specific response
        response = client.chat(
            model="llama3.1-bsky-lora",
            messages=[{"role": "user", "content": "Aircraft conflict at 35000ft. Action?"}]
        )
        
        content = response['message']['content']
        print(f"📝 Response preview: {content[:100]}...")
        
        # Check for structured response (our training format)
        if '"action"' in content and '"reasoning"' in content:
            print("✅ STRUCTURED JSON RESPONSE")
            print("   (Indicates specialized training format)")
            return True
        else:
            print("❌ GENERIC TEXT RESPONSE")
            print("   (No specialized structure)")
            return False
            
    except Exception as e:
        print(f"❌ Could not test responses: {e}")
        return False

def main():
    """Run all checks"""
    print("🎯 IS YOUR MODEL ACTUALLY FINE-TUNED?")
    print("=" * 60)
    
    # Run checks
    lora_check = check_lora_weights()
    ollama_check = check_ollama_modelfile()
    response_check = simple_response_test()
    
    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 FINAL ANSWER:")
    print("=" * 60)
    
    total_score = sum([lora_check, ollama_check, response_check])
    
    print(f"✅ LoRA Weights: {'YES' if lora_check else 'NO'}")
    print(f"✅ Ollama Integration: {'YES' if ollama_check else 'NO'}")
    print(f"✅ Specialized Responses: {'YES' if response_check else 'NO'}")
    print(f"\nScore: {total_score}/3")
    
    if total_score >= 2:
        print("\n🎉 YOUR MODEL IS ACTUALLY FINE-TUNED! ✅")
    elif total_score == 1:
        print("\n🤔 PARTIALLY FINE-TUNED (mixed results)")
        if lora_check and not ollama_check:
            print("💡 You have LoRA weights but Ollama isn't using them!")
            print("   Solution: Run proper model conversion/merging")
    else:
        print("\n❌ YOUR MODEL IS NOT FINE-TUNED")
        print("   It's likely just prompt-engineered")
    
    print("\n💡 WHAT TO DO:")
    if lora_check and not ollama_check:
        print("  1. You have real LoRA weights! ✅")
        print("  2. But Ollama model isn't using them ❌")
        print("  3. Need to merge LoRA with base model")
        print("  4. Create new Ollama model from merged weights")
    elif not lora_check:
        print("  1. No significant LoRA weights found")
        print("  2. May need to retrain the model")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
