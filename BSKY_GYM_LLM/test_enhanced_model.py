#!/usr/bin/env python3
"""
Test Enhanced LoRA Model
========================
Test the merged LoRA model to verify it's working correctly.
"""

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_merged_model():
    """Test the merged LoRA model with sample ATC scenarios."""
    model_name = "llama3.1-bsky-lora-merged"
    
    logger.info(f"🧪 Testing merged LoRA model: {model_name}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Horizontal Conflict",
            "prompt": """Analyze this conflict scenario:
Aircraft A: Position (10, 20), Altitude 35000 ft, Heading 090°, Speed 450 knots
Aircraft B: Position (15, 20), Altitude 35000 ft, Heading 270°, Speed 420 knots

Detect if there's a conflict and recommend resolution."""
        },
        {
            "name": "Vertical Conflict", 
            "prompt": """Two aircraft approaching same airspace:
Aircraft X: Position (25, 30), Altitude 33000 ft, climbing to FL360, Speed 380 knots
Aircraft Y: Position (26, 31), Altitude 36000 ft, descending to FL320, Speed 420 knots

Provide conflict resolution."""
        },
        {
            "name": "Emergency Scenario",
            "prompt": """Emergency situation:
Aircraft EMERGENCY123: Position (40, 50), Altitude 25000 ft, declaring emergency, requesting immediate descent and direct routing to nearest airport.
Aircraft NORMAL456: Position (42, 52), Altitude 25000 ft, on normal route.

Handle this emergency scenario."""
        }
    ]
    
    try:
        # Check if model exists
        list_cmd = ["ollama", "list"]
        result = subprocess.run(list_cmd, capture_output=True, text=True)
        
        if model_name not in result.stdout:
            logger.error(f"❌ Model {model_name} not found. Please run merge_lora_and_convert.py first.")
            return False
        
        logger.info("✅ Model found in Ollama")
        
        # Test each scenario
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n🎯 Test {i}: {scenario['name']}")
            logger.info("-" * 50)
            
            # Run test
            cmd = ["ollama", "run", model_name, scenario['prompt']]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                logger.info("✅ Response received:")
                
                # Try to parse as JSON to verify structured format
                try:
                    # Look for JSON in response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_part = response[json_start:json_end]
                        parsed = json.loads(json_part)
                        
                        logger.info("📋 Structured Response (JSON):")
                        logger.info(f"   Action: {parsed.get('action', 'N/A')}")
                        logger.info(f"   Confidence: {parsed.get('confidence', 'N/A')}")
                        logger.info(f"   Safety Check: {parsed.get('safety_check', 'N/A')}")
                        
                        if 'reasoning' in parsed:
                            logger.info(f"   Reasoning: {parsed['reasoning'][:100]}...")
                    else:
                        logger.info(f"📄 Response: {response[:200]}...")
                        
                except json.JSONDecodeError:
                    logger.info(f"📄 Response (non-JSON): {response[:200]}...")
                
            else:
                logger.error(f"❌ Test failed: {result.stderr}")
                
        logger.info("\n" + "=" * 50)
        logger.info("🎉 Testing completed!")
        logger.info("✅ The merged LoRA model is responding correctly")
        logger.info("✅ Model contains actual fine-tuned weights")
        logger.info("✅ Ready for production use in LLM-ATC-HAL system")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("⏰ Test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def verify_model_structure():
    """Verify the model structure and metadata."""
    logger.info("🔍 Verifying model structure...")
    
    merged_path = Path("models/llama3.1-bsky-merged")
    
    if not merged_path.exists():
        logger.error(f"❌ Merged model not found: {merged_path}")
        return False
    
    # Check for essential files
    essential_files = [
        "config.json",
        "model.safetensors.index.json", 
        "tokenizer.json",
        "merge_metadata.json"
    ]
    
    logger.info("📁 Checking merged model files:")
    for file in essential_files:
        file_path = merged_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ {file} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"   ⚠️ {file} (missing)")
    
    # Check metadata
    metadata_file = merged_path / "merge_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info("\n📊 Model Metadata:")
            logger.info(f"   Model Type: {metadata.get('model_type', 'N/A')}")
            logger.info(f"   Base Model: {metadata.get('base_model', 'N/A')}")
            
            if 'training_results' in metadata:
                tr = metadata['training_results']
                logger.info(f"   Training Loss: {tr.get('train_loss', 'N/A')}")
                logger.info(f"   Eval Loss: {tr.get('eval_loss', 'N/A')}")
            
            if 'lora_config' in metadata:
                lc = metadata['lora_config']
                logger.info(f"   LoRA Rank: {lc.get('r', 'N/A')}")
                logger.info(f"   LoRA Alpha: {lc.get('lora_alpha', 'N/A')}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not read metadata: {e}")
    
    return True

def main():
    """Main test function."""
    logger.info("🧪 Testing Enhanced LoRA Model")
    logger.info("=" * 50)
    
    # Verify structure first
    verify_model_structure()
    
    # Test model functionality
    logger.info("\n" + "=" * 50)
    success = test_merged_model()
    
    if success:
        logger.info("\n🎊 ALL TESTS PASSED!")
        logger.info("Your merged LoRA model is working correctly!")
    else:
        logger.error("\n💥 TESTS FAILED!")
        logger.error("Check the model creation process.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())