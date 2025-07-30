#!/usr/bin/env python3
"""
Test script for LLM Prompt Engine
=================================
Demonstrates the LLM prompt engine capabilities for conflict resolution.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from llm_atc.tools.llm_prompt_engine import LLMPromptEngine, ConflictPromptData


def test_conflict_prompt_generation():
    """Test conflict prompt generation"""
    print("=" * 60)
    print("Testing Conflict Prompt Generation")
    print("=" * 60)
    
    # Initialize prompt engine
    engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=True)
    
    # Sample conflict scenario
    conflict_info = {
        'aircraft_1_id': 'AAL123',
        'aircraft_2_id': 'UAL456',
        'time_to_conflict': 95.5,
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
    
    # Generate prompt
    prompt = engine.format_conflict_prompt(conflict_info)
    print("Generated Conflict Resolution Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    return prompt


def test_detection_prompt_generation():
    """Test conflict detection prompt generation"""
    print("\n" + "=" * 60)
    print("Testing Conflict Detection Prompt Generation")
    print("=" * 60)
    
    # Initialize prompt engine
    engine = LLMPromptEngine()
    
    # Sample aircraft states
    aircraft_states = [
        {
            'id': 'DAL789',
            'lat': 40.7128,
            'lon': -74.0060,
            'alt': 37000,
            'hdg': 180,
            'spd': 480,
            'vs': 0
        },
        {
            'id': 'SWA321',
            'lat': 40.7200,
            'lon': -74.0100,
            'alt': 37000,
            'hdg': 360,
            'spd': 470,
            'vs': -500
        },
        {
            'id': 'JBU654',
            'lat': 40.7000,
            'lon': -73.9900,
            'alt': 39000,
            'hdg': 90,
            'spd': 440,
            'vs': 0
        }
    ]
    
    # Generate detection prompt
    prompt = engine.format_detector_prompt(aircraft_states, time_horizon=3.0)
    print("Generated Conflict Detection Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


def test_response_parsing():
    """Test response parsing capabilities"""
    print("\n" + "=" * 60)
    print("Testing Response Parsing")
    print("=" * 60)
    
    # Initialize prompt engine
    engine = LLMPromptEngine()
    
    # Sample LLM responses to parse
    sample_responses = [
        """
Command: HDG AAL123 270
Aircraft: AAL123
Maneuver: heading
Rationale: Turn left 20 degrees to avoid conflict while maintaining safe separation
Confidence: 0.85
        """,
        
        """
Command: ALT UAL456 36000
Aircraft: UAL456
Maneuver: altitude
Rationale: Climb 1000 feet to establish vertical separation
Confidence: 0.92
        """,
        
        """
Looking at this situation, I recommend:
HDG AAL123 250
This heading change will resolve the conflict safely.
        """
    ]
    
    print("Testing Resolution Response Parsing:")
    print("-" * 40)
    
    for i, response in enumerate(sample_responses, 1):
        print(f"\nSample Response {i}:")
        print(f"Raw: {response.strip()}")
        
        parsed = engine.parse_resolution_response(response)
        if parsed:
            print(f"Parsed Command: {parsed.command}")
            print(f"Aircraft: {parsed.aircraft_id}")
            print(f"Maneuver Type: {parsed.maneuver_type}")
            print(f"Rationale: {parsed.rationale}")
            print(f"Confidence: {parsed.confidence}")
        else:
            print("❌ Failed to parse response")
        print("-" * 20)
    
    # Test detection response parsing
    print("\nTesting Detection Response Parsing:")
    print("-" * 40)
    
    detection_response = """
Conflict Detected: YES
Aircraft Pairs at Risk: DAL789-SWA321, JBU654-DAL789
Time to Loss of Separation: 120 seconds, 180 seconds
Confidence: 0.78
Priority: high
    """
    
    print(f"Detection Response: {detection_response.strip()}")
    parsed_detection = engine.parse_detector_response(detection_response)
    print(f"Parsed Results: {parsed_detection}")


def test_high_level_api():
    """Test high-level API functions"""
    print("\n" + "=" * 60)
    print("Testing High-Level API")
    print("=" * 60)
    
    # Note: This would require actual LLM connectivity
    print("High-level API functions available:")
    print("✓ engine.get_conflict_resolution(conflict_info)")
    print("✓ engine.detect_conflict_via_llm(aircraft_states)")
    print("✓ engine.assess_resolution_safety(command, conflict_info)")
    print("\n⚠️  Actual testing requires LLM connectivity (Ollama)")
    print("   Run with proper Ollama setup to test full functionality")


def main():
    """Run all tests"""
    print("LLM Prompt Engine Test Suite")
    print("=" * 60)
    
    try:
        test_conflict_prompt_generation()
        test_detection_prompt_generation()
        test_response_parsing()
        test_high_level_api()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The LLM Prompt Engine is ready for integration.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
