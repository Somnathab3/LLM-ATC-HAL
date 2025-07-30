#!/usr/bin/env python3
"""
Test script to verify Ollama integration with LLM Prompt Engine
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools.llm_prompt_engine import LLMPromptEngine, ConflictPromptData
from llm_interface.llm_client import LLMClient


def test_basic_ollama_connection():
    """Test basic Ollama connection"""
    print("🔗 Testing basic Ollama connection...")
    
    try:
        client = LLMClient(model='llama3.1:8b')
        response = client.ask("Say 'Hello from Ollama!' and nothing else.")
        print(f"✅ Ollama connection successful!")
        print(f"📝 Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False


def test_prompt_engine_basic():
    """Test basic prompt engine functionality"""
    print("\n🧠 Testing LLM Prompt Engine basic functionality...")
    
    try:
        engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
        
        # Create sample conflict data
        sample_conflict = {
            'aircraft_1_id': 'AAL123',
            'aircraft_2_id': 'UAL456',
            'time_to_conflict': 90.0,
            'closest_approach_distance': 3.2,
            'conflict_type': 'convergent',
            'urgency_level': 'high',
            'aircraft_1': {
                'lat': 52.3676,
                'lon': 4.9041,
                'alt': 35000,
                'hdg': 90,
                'spd': 450,
                'type': 'B738'
            },
            'aircraft_2': {
                'lat': 52.3700,
                'lon': 4.9100,
                'alt': 35000,
                'hdg': 270,
                'spd': 460,
                'type': 'A320'
            },
            'environmental_conditions': {
                'wind_direction_deg': 270,
                'wind_speed_kts': 15,
                'visibility_km': '10+ km',
                'conditions': 'Clear'
            }
        }
        
        # Test prompt formatting
        prompt = engine.format_conflict_prompt(sample_conflict)
        print(f"✅ Prompt formatting successful! (length: {len(prompt)} chars)")
        
        return True
    except Exception as e:
        print(f"❌ Prompt engine test failed: {e}")
        return False


def test_conflict_resolution():
    """Test full conflict resolution pipeline"""
    print("\n🛡️ Testing conflict resolution pipeline...")
    
    try:
        engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
        
        # Sample conflict scenario
        sample_conflict = {
            'aircraft_1_id': 'DAL789',
            'aircraft_2_id': 'SWA321',
            'time_to_conflict': 120.0,
            'closest_approach_distance': 2.8,
            'conflict_type': 'convergent',
            'urgency_level': 'medium',
            'aircraft_1': {
                'lat': 40.7128,
                'lon': -74.0060,
                'alt': 36000,
                'hdg': 180,
                'spd': 480,
                'type': 'B777'
            },
            'aircraft_2': {
                'lat': 40.7200,
                'lon': -74.0100,
                'alt': 36000,
                'hdg': 360,
                'spd': 470,
                'type': 'B737'
            },
            'environmental_conditions': {
                'wind_direction_deg': 270,
                'wind_speed_kts': 20,
                'visibility_km': '10+ km',
                'conditions': 'Clear'
            }
        }
        
        # Get conflict resolution
        print("🤔 Asking Ollama for conflict resolution...")
        resolution_command = engine.get_conflict_resolution(sample_conflict)
        
        if resolution_command:
            print(f"✅ Resolution generated: {resolution_command}")
            
            # Test response parsing
            sample_response = f"""
Command: {resolution_command}
Aircraft: {sample_conflict['aircraft_1_id']}
Maneuver: heading
Rationale: Turn aircraft to avoid conflict and maintain separation
Confidence: 0.85
"""
            parsed = engine.parse_resolution_response(sample_response)
            if parsed:
                print(f"✅ Response parsing successful!")
                print(f"   📋 Command: {parsed.command}")
                print(f"   ✈️ Aircraft: {parsed.aircraft_id}")
                print(f"   🔄 Maneuver: {parsed.maneuver_type}")
                print(f"   💭 Rationale: {parsed.rationale}")
                print(f"   🎯 Confidence: {parsed.confidence}")
            else:
                print("❌ Response parsing failed")
        else:
            print("❌ No resolution generated")
            
        return resolution_command is not None
        
    except Exception as e:
        print(f"❌ Conflict resolution test failed: {e}")
        return False


def test_detection_capability():
    """Test conflict detection capability"""
    print("\n🔍 Testing conflict detection...")
    
    try:
        engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
        
        # Sample aircraft states
        aircraft_states = [
            {
                'id': 'AAL101',
                'lat': 52.3676,
                'lon': 4.9041,
                'alt': 35000,
                'hdg': 90,
                'spd': 450,
                'vs': 0
            },
            {
                'id': 'UAL202',
                'lat': 52.3700,
                'lon': 4.9100,
                'alt': 35000,
                'hdg': 270,
                'spd': 460,
                'vs': 0
            },
            {
                'id': 'DAL303',
                'lat': 53.0000,
                'lon': 5.0000,
                'alt': 37000,
                'hdg': 180,
                'spd': 420,
                'vs': -500
            }
        ]
        
        # Test detection
        print("🤔 Asking Ollama for conflict detection...")
        detection_result = engine.detect_conflict_via_llm(aircraft_states, time_horizon=5.0)
        
        print(f"✅ Detection completed!")
        print(f"   🚨 Conflict detected: {detection_result.get('conflict_detected', False)}")
        print(f"   ✈️ Aircraft pairs: {detection_result.get('aircraft_pairs', [])}")
        print(f"   ⏰ Time to conflict: {detection_result.get('time_to_conflict', [])}")
        print(f"   🎯 Confidence: {detection_result.get('confidence', 0.0)}")
        print(f"   📊 Priority: {detection_result.get('priority', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 Starting Ollama Integration Tests for LLM Prompt Engine")
    print("=" * 60)
    
    tests = [
        ("Basic Ollama Connection", test_basic_ollama_connection),
        ("Prompt Engine Basic", test_prompt_engine_basic),
        ("Conflict Resolution", test_conflict_resolution),
        ("Conflict Detection", test_detection_capability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 60)
    print(f"🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ollama integration is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
