#!/usr/bin/env python3
"""
How to Verify if a Model is Actually Fine-tuned
==============================================
Comprehensive guide to check if models contain actual fine-tuned weights
vs just prompt engineering.
"""

import json
import subprocess
import ollama
from pathlib import Path
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_lora_adapter_exists():
    """Check if we have actual LoRA adapter files"""
    
    print("🔍 METHOD 1: CHECK LORA ADAPTER FILES")
    print("=" * 60)
    
    adapter_path = Path("models/llama3.1-bsky-lora")
    
    # Key files that indicate real fine-tuning
    key_files = {
        "adapter_model.safetensors": "LoRA weights (MUST be >100MB for real training)",
        "adapter_config.json": "LoRA configuration",
        "training_args.bin": "Training arguments",
        "train_results.json": "Training loss curves"
    }
    
    print("📁 LoRA Adapter Files Check:")
    has_real_weights = False
    
    for filename, description in key_files.items():
        filepath = adapter_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:<25} {size_mb:>8.1f} MB - {description}")
            if filename == "adapter_model.safetensors" and size_mb > 100:
                has_real_weights = True
        else:
            print(f"  ❌ {filename:<25} {'Missing':<8} - {description}")
    
    if has_real_weights:
        print("  🎯 VERDICT: ✅ REAL LORA WEIGHTS FOUND (>100MB)")
    else:
        print("  🎯 VERDICT: ❌ NO SIGNIFICANT WEIGHTS FOUND")
    
    return has_real_weights

def check_ollama_model_structure():
    """Check if Ollama model uses actual weights or just prompts"""
    
    print("\n🔍 METHOD 2: CHECK OLLAMA MODEL STRUCTURE")
    print("=" * 60)
    
    model_name = "llama3.1-bsky-lora"
    
    try:
        # Get modelfile
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True, text=True, check=True
        )
        
        modelfile = result.stdout
        
        print("📋 Ollama Modelfile Analysis:")
        
        # Check what the FROM line points to
        from_lines = [line for line in modelfile.split('\n') if line.startswith('FROM ')]
        if from_lines:
            from_line = from_lines[0]
            print(f"  • FROM: {from_line}")
            
            if "llama3.1:8b" in from_line:
                print("  🎯 VERDICT: ❌ USES BASE MODEL (no fine-tuning)")
                return False
            elif "blobs/sha256" in from_line:
                print("  🎯 VERDICT: ❌ USES BLOB FILE (likely just base model copy)")
                return False
            elif "models/" in from_line:
                print("  🎯 VERDICT: ✅ MIGHT USE CUSTOM WEIGHTS")
                return True
        else:
            print("  ❌ No FROM line found")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to get modelfile: {e}")
        return False

def check_ollama_api_details():
    """Check Ollama API for model details"""
    
    print("\n🔍 METHOD 3: CHECK OLLAMA API DETAILS")
    print("=" * 60)
    
    try:
        # Get detailed model info via API
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            
            # Find our model
            for model in models_data.get('models', []):
                if 'llama3.1-bsky-lora' in model.get('name', ''):
                    print("📊 Model Details from API:")
                    print(f"  • Name: {model.get('name')}")
                    print(f"  • Size: {model.get('size', 0) / (1024**3):.1f} GB")
                    print(f"  • Modified: {model.get('modified_at')}")
                    print(f"  • Digest: {model.get('digest', 'Unknown')[:20]}...")
                    
                    # Check if size is similar to base model
                    size_gb = model.get('size', 0) / (1024**3)
                    if 4.5 <= size_gb <= 5.1:  # Base llama3.1:8b is ~4.9GB
                        print("  🎯 VERDICT: ❌ SIZE MATCHES BASE MODEL (likely not fine-tuned)")
                        return False
                    elif size_gb > 5.5:
                        print("  🎯 VERDICT: ✅ LARGER THAN BASE (might contain fine-tuned weights)")
                        return True
                    else:
                        print("  🎯 VERDICT: ❓ UNCLEAR FROM SIZE ALONE")
                        return None
    
    except Exception as e:
        print(f"  ❌ API check failed: {e}")
        return None

def test_model_responses():
    """Test if model shows signs of actual fine-tuning in responses"""
    
    print("\n🔍 METHOD 4: BEHAVIORAL TESTING")
    print("=" * 60)
    
    try:
        client = ollama.Client()
        
        # Test with very specific ATC scenario
        test_prompts = [
            {
                "prompt": "Aircraft at 35000ft, heading 090, speed 450kt. Conflict in 2 minutes. Action?",
                "expected_indicators": ["turn", "heading", "altitude", "descend", "climb"]
            },
            {
                "prompt": "What is the minimum separation for aircraft?",
                "expected_indicators": ["5 nautical miles", "1000 feet", "separation"]
            }
        ]
        
        print("🧪 Testing Model Responses:")
        
        fine_tuned_score = 0
        total_tests = len(test_prompts)
        
        for i, test in enumerate(test_prompts, 1):
            print(f"\n  Test {i}: {test['prompt'][:50]}...")
            
            response = client.chat(
                model="llama3.1-bsky-lora",
                messages=[{"role": "user", "content": test['prompt']}]
            )
            
            content = response['message']['content'].lower()
            
            # Check for expected indicators
            found_indicators = [ind for ind in test['expected_indicators'] if ind.lower() in content]
            score = len(found_indicators) / len(test['expected_indicators'])
            
            print(f"    Found: {found_indicators}")
            print(f"    Score: {score:.2f}")
            
            if score > 0.5:
                fine_tuned_score += 1
            
            # Check for structured JSON (our training format)
            if '"action"' in content and '"reasoning"' in content:
                print(f"    ✅ Structured response (fine-tuning indicator)")
                fine_tuned_score += 0.5
            else:
                print(f"    ❌ Unstructured response")
        
        final_score = fine_tuned_score / total_tests
        print(f"\n  🎯 BEHAVIORAL SCORE: {final_score:.2f}")
        
        if final_score > 0.7:
            print(f"  🎯 VERDICT: ✅ SHOWS SIGNS OF FINE-TUNING")
            return True
        elif final_score > 0.4:
            print(f"  🎯 VERDICT: ❓ SOME SPECIALIZATION (unclear)")
            return None
        else:
            print(f"  🎯 VERDICT: ❌ GENERIC RESPONSES (no fine-tuning)")
            return False
    
    except Exception as e:
        print(f"  ❌ Behavioral test failed: {e}")
        return None

def compare_with_base_model():
    """Compare responses with base model to see differences"""
    
    print("\n🔍 METHOD 5: COMPARISON WITH BASE MODEL")
    print("=" * 60)
    
    try:
        client = ollama.Client()
        
        test_prompt = "Aircraft conflict: 35000ft, opposite headings. Immediate action?"
        
        # Test our model
        print("🔬 Testing our model:")
        our_response = client.chat(
            model="llama3.1-bsky-lora",
            messages=[{"role": "user", "content": test_prompt}]
        )
        our_content = our_response['message']['content']
        print(f"  Response length: {len(our_content)} chars")
        print(f"  Preview: {our_content[:100]}...")
        
        # Test base model
        print("\n🔬 Testing base model:")
        base_response = client.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": test_prompt}]
        )
        base_content = base_response['message']['content']
        print(f"  Response length: {len(base_content)} chars")
        print(f"  Preview: {base_content[:100]}...")
        
        # Compare responses
        similarity = len(set(our_content.lower().split()) & set(base_content.lower().split()))
        total_words = len(set(our_content.lower().split()) | set(base_content.lower().split()))
        similarity_ratio = similarity / total_words if total_words > 0 else 0
        
        print(f"\n  📊 Response Similarity: {similarity_ratio:.2f}")
        
        if similarity_ratio < 0.3:
            print(f"  🎯 VERDICT: ✅ VERY DIFFERENT (likely fine-tuned)")
            return True
        elif similarity_ratio < 0.6:
            print(f"  🎯 VERDICT: ❓ SOMEWHAT DIFFERENT")
            return None
        else:
            print(f"  🎯 VERDICT: ❌ TOO SIMILAR (not fine-tuned)")
            return False
    
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        return None

def final_verdict():
    """Run all checks and provide final verdict"""
    
    print("\n" + "=" * 70)
    print("🎯 FINAL VERDICT: IS THE MODEL ACTUALLY FINE-TUNED?")
    print("=" * 70)
    
    # Run all checks
    checks = [
        ("LoRA Adapter Files", check_lora_adapter_exists()),
        ("Ollama Model Structure", check_ollama_model_structure()),
        ("API Details", check_ollama_api_details()),
        ("Behavioral Testing", test_model_responses()),
        ("Base Model Comparison", compare_with_base_model())
    ]
    
    print("\n📋 SUMMARY OF CHECKS:")
    positive_checks = 0
    total_checks = 0
    
    for check_name, result in checks:
        if result is True:
            print(f"  ✅ {check_name}: FINE-TUNED")
            positive_checks += 1
        elif result is False:
            print(f"  ❌ {check_name}: NOT FINE-TUNED")
        else:
            print(f"  ❓ {check_name}: UNCLEAR")
        
        if result is not None:
            total_checks += 1
    
    confidence = positive_checks / total_checks if total_checks > 0 else 0
    
    print(f"\n🎯 CONFIDENCE SCORE: {confidence:.2f} ({positive_checks}/{total_checks})")
    
    if confidence >= 0.7:
        print("🎉 FINAL VERDICT: ✅ MODEL IS ACTUALLY FINE-TUNED!")
        print("   The model contains real fine-tuned weights.")
    elif confidence >= 0.4:
        print("🤔 FINAL VERDICT: ❓ PARTIALLY FINE-TUNED")
        print("   Some evidence of fine-tuning, but mixed results.")
    else:
        print("❌ FINAL VERDICT: ❌ MODEL IS NOT FINE-TUNED!")
        print("   The model is likely just prompt-engineered.")
    
    print("\n💡 RECOMMENDATIONS:")
    if confidence < 0.7:
        print("  1. The current Ollama model may not use actual LoRA weights")
        print("  2. We need to properly merge LoRA adapter with base model")
        print("  3. Create new Ollama model from merged weights")
        print("  4. Use: python convert_to_ollama.py")

def main():
    """Run comprehensive fine-tuning verification"""
    
    print("🔍 COMPREHENSIVE FINE-TUNING VERIFICATION")
    print("=" * 70)
    print("This will check if your model is actually fine-tuned")
    print("or just using prompt engineering.")
    print("=" * 70)
    
    final_verdict()

if __name__ == "__main__":
    main()
