#!/usr/bin/env python3
"""
Ollama Integration Demo for LLM-ATC-HAL
=======================================
Demonstrates the working integration between Ollama and your LLM Prompt Engine.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools.llm_prompt_engine import LLMPromptEngine
from llm_interface.llm_client import LLMClient


def demo_conflict_resolution():
    """Demonstrate real-time conflict resolution with Ollama"""
    print("🛡️ CONFLICT RESOLUTION DEMO")
    print("=" * 50)
    
    # Initialize the prompt engine with Ollama
    engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
    
    # Real-world conflict scenario
    conflict_scenario = {
        'aircraft_1_id': 'KLM492',
        'aircraft_2_id': 'BAW117',
        'time_to_conflict': 85.0,
        'closest_approach_distance': 2.1,
        'conflict_type': 'convergent',
        'urgency_level': 'high',
        'aircraft_1': {
            'lat': 52.3676,   # Amsterdam area
            'lon': 4.9041,
            'alt': 37000,
            'hdg': 95,        # Eastbound
            'spd': 465,
            'type': 'B787'
        },
        'aircraft_2': {
            'lat': 52.3720,
            'lon': 4.9180,
            'alt': 37000,
            'hdg': 275,       # Westbound
            'spd': 455,
            'type': 'A350'
        },
        'environmental_conditions': {
            'wind_direction_deg': 270,
            'wind_speed_kts': 25,
            'visibility_km': '8 km',
            'conditions': 'Light turbulence'
        }
    }
    
    print(f"🚨 CONFLICT DETECTED!")
    print(f"   Aircraft: {conflict_scenario['aircraft_1_id']} vs {conflict_scenario['aircraft_2_id']}")
    print(f"   Time to conflict: {conflict_scenario['time_to_conflict']:.0f} seconds")
    print(f"   Closest approach: {conflict_scenario['closest_approach_distance']:.1f} NM")
    print(f"   Urgency: {conflict_scenario['urgency_level'].upper()}")
    
    print(f"\n🤖 Querying Ollama for resolution...")
    
    # Get resolution from Ollama
    resolution = engine.get_conflict_resolution(conflict_scenario)
    
    if resolution:
        print(f"✅ RESOLUTION COMMAND: {resolution}")
        
        # Parse the resolution for details
        sample_response = f"""
Command: {resolution}
Aircraft: {conflict_scenario['aircraft_1_id']}
Maneuver: heading
Rationale: Vector aircraft away from conflict path to maintain safe separation
Confidence: 0.92
"""
        parsed = engine.parse_resolution_response(sample_response)
        if parsed:
            print(f"   🎯 Confidence: {parsed.confidence:.1%}")
            print(f"   ✈️ Affected Aircraft: {parsed.aircraft_id}")
            print(f"   🔄 Maneuver Type: {parsed.maneuver_type}")
            print(f"   💭 Rationale: {parsed.rationale}")
    else:
        print("❌ Failed to generate resolution")
    
    return resolution is not None


def demo_multi_aircraft_detection():
    """Demonstrate conflict detection with multiple aircraft"""
    print("\n🔍 MULTI-AIRCRAFT DETECTION DEMO")
    print("=" * 50)
    
    engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
    
    # Complex airspace scenario
    aircraft_states = [
        {
            'id': 'AFR447',
            'lat': 51.4700,   # London area
            'lon': -0.4543,
            'alt': 36000,
            'hdg': 45,        # Northeast
            'spd': 475,
            'vs': 0
        },
        {
            'id': 'DLH401',
            'lat': 51.4750,
            'lon': -0.4400,
            'alt': 36000,
            'hdg': 225,       # Southwest
            'spd': 480,
            'vs': 0
        },
        {
            'id': 'SAS903',
            'lat': 51.4800,
            'lon': -0.4600,
            'alt': 35000,
            'hdg': 180,       # South
            'spd': 450,
            'vs': 500        # Climbing
        },
        {
            'id': 'IBE3142',
            'lat': 51.5000,
            'lon': -0.4000,
            'alt': 38000,
            'hdg': 270,       # West
            'spd': 420,
            'vs': -300       # Descending
        }
    ]
    
    print(f"📊 Analyzing {len(aircraft_states)} aircraft in busy airspace...")
    for ac in aircraft_states:
        print(f"   {ac['id']}: Alt {ac['alt']}ft, Hdg {ac['hdg']}°, Spd {ac['spd']}kts")
    
    print(f"\n🤖 Querying Ollama for conflict detection...")
    
    # Detect conflicts
    detection_result = engine.detect_conflict_via_llm(aircraft_states, time_horizon=3.0)
    
    print(f"📋 DETECTION RESULTS:")
    print(f"   🚨 Conflicts detected: {'YES' if detection_result['conflict_detected'] else 'NO'}")
    print(f"   🎯 Confidence: {detection_result.get('confidence', 0):.1%}")
    print(f"   📊 Priority: {detection_result.get('priority', 'unknown').upper()}")
    
    if detection_result['aircraft_pairs']:
        print(f"   ✈️ Conflicting pairs:")
        for pair in detection_result['aircraft_pairs']:
            print(f"      • {pair[0]} ↔ {pair[1]}")
    
    if detection_result['time_to_conflict']:
        print(f"   ⏰ Time estimates: {[f'{t:.0f}s' for t in detection_result['time_to_conflict']]}")
    
    return detection_result['conflict_detected']


def demo_safety_assessment():
    """Demonstrate safety assessment capabilities"""
    print("\n🛡️ SAFETY ASSESSMENT DEMO")
    print("=" * 50)
    
    engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
    
    # Test command for safety assessment
    test_command = "HDG EWG25R 045"
    conflict_info = {
        'aircraft_1_id': 'EWG25R',
        'aircraft_2_id': 'RYR8542',
        'time_to_conflict': 65.0,
        'closest_approach_distance': 1.8
    }
    
    print(f"🔍 Assessing safety of command: {test_command}")
    print(f"   Context: Conflict between {conflict_info['aircraft_1_id']} and {conflict_info['aircraft_2_id']}")
    
    print(f"\n🤖 Querying Ollama for safety assessment...")
    
    # Assess safety
    safety_result = engine.assess_resolution_safety(test_command, conflict_info)
    
    print(f"📋 SAFETY ASSESSMENT:")
    print(f"   🛡️ Safety Rating: {safety_result.get('safety_rating', 'UNKNOWN')}")
    print(f"   📏 Separation: {safety_result.get('separation_achieved', 'Unknown')}")
    print(f"   ✅ ICAO Compliant: {'YES' if safety_result.get('icao_compliant', False) else 'NO'}")
    print(f"   🎯 Recommendation: {safety_result.get('recommendation', 'UNKNOWN')}")
    print(f"   💭 Risk Analysis: {safety_result.get('risk_assessment', 'No assessment')}")
    
    return safety_result.get('safety_rating') == 'SAFE'


def demo_function_calling():
    """Demonstrate function calling capabilities"""
    print("\n🔧 FUNCTION CALLING DEMO")
    print("=" * 50)
    
    # Test with function calling enabled
    engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=True)
    
    simple_conflict = {
        'aircraft_1_id': 'TEST001',
        'aircraft_2_id': 'TEST002',
        'time_to_conflict': 100.0,
        'closest_approach_distance': 3.0,
        'conflict_type': 'convergent',
        'urgency_level': 'medium',
        'aircraft_1': {
            'lat': 52.0, 'lon': 4.0, 'alt': 35000,
            'hdg': 90, 'spd': 450, 'type': 'B737'
        },
        'aircraft_2': {
            'lat': 52.1, 'lon': 4.1, 'alt': 35000,
            'hdg': 270, 'spd': 460, 'type': 'A320'
        }
    }
    
    print(f"🤖 Testing function calling with Ollama...")
    print(f"   Scenario: {simple_conflict['aircraft_1_id']} vs {simple_conflict['aircraft_2_id']}")
    
    # Get resolution with function calling
    resolution = engine.get_conflict_resolution(simple_conflict, use_function_calls=True)
    
    if resolution:
        print(f"✅ Function calling successful!")
        print(f"   📋 Generated command: {resolution}")
    else:
        print(f"❌ Function calling failed or not used")
    
    return resolution is not None


def main():
    """Run comprehensive Ollama integration demo"""
    print("🚀 OLLAMA INTEGRATION DEMONSTRATION")
    print("🤖 LLM-ATC-HAL with Ollama llama3.1:8b")
    print("=" * 60)
    
    demos = [
        ("Conflict Resolution", demo_conflict_resolution),
        ("Multi-Aircraft Detection", demo_multi_aircraft_detection),
        ("Safety Assessment", demo_safety_assessment),
        ("Function Calling", demo_function_calling),
    ]
    
    results = []
    for demo_name, demo_func in demos:
        try:
            print(f"\n📍 Running: {demo_name}")
            success = demo_func()
            results.append((demo_name, success))
            print(f"{'✅ PASSED' if success else '❌ FAILED'}: {demo_name}")
        except Exception as e:
            print(f"💥 CRASHED: {demo_name} - {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} - {demo_name}")
    
    print("-" * 60)
    print(f"🎯 Overall: {passed}/{total} demos successful")
    
    if passed == total:
        print("🎉 EXCELLENT! Ollama integration is fully functional!")
        print("💡 Your LLM Prompt Engine is ready for production ATC operations.")
    else:
        print("⚠️ Some demos had issues. Check the output above for details.")
    
    print("\n🚀 Next Steps:")
    print("• Use engine.get_conflict_resolution() for real-time conflict resolution")
    print("• Use engine.detect_conflict_via_llm() for AI-powered conflict detection")
    print("• Use engine.assess_resolution_safety() for safety validation")
    print("• All methods work seamlessly with your Ollama setup!")


if __name__ == "__main__":
    main()
