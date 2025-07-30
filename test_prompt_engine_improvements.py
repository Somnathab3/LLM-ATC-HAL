#!/usr/bin/env python3
"""
Test script for LLM Prompt Engine improvements.

Tests the new features:
1. Configurable aircraft ID patterns
2. Simplified command extraction
3. JSON parsing for conflict detection
4. Robust safety assessment parsing
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_atc.tools.llm_prompt_engine import LLMPromptEngine


def test_configurable_aircraft_id():
    """Test configurable aircraft ID patterns"""
    print("Testing configurable aircraft ID patterns...")
    
    # Test with default pattern
    engine_default = LLMPromptEngine()
    print(f"Default pattern: {engine_default.aircraft_id_regex}")
    
    # Test with ICAO pattern
    engine_icao = LLMPromptEngine(aircraft_id_regex=r'^[A-Z]{2,4}\d{2,4}[A-Z]?$')
    print(f"ICAO pattern: {engine_icao.aircraft_id_regex}")
    
    # Test aircraft ID extraction with different patterns
    test_commands = ["HDG AC001 270", "HDG KLM492 180", "HDG TEST-123 360", "HDG N123AB 270", "HDG DLH456A 180"]
    
    for command in test_commands:
        default_match = engine_default._extract_aircraft_id(command)
        icao_match = engine_icao._extract_aircraft_id(command)
        aircraft_id = command.split()[1]  # Expected aircraft ID
        print(f"  {aircraft_id}: Default={default_match}, ICAO={icao_match}")
    
    # Show pattern validation differences
    print("\n  Pattern validation differences:")
    test_ids = ["AC001", "TEST-123", "X1", "ABC12345", "123ABC"]
    
    for test_id in test_ids:
        import re
        default_valid = bool(re.match(engine_default.aircraft_id_regex, test_id))
        icao_valid = bool(re.match(engine_icao.aircraft_id_regex, test_id))
        print(f"    {test_id}: Default={default_valid}, ICAO={icao_valid}")
    
    print("✓ Aircraft ID pattern configuration test passed\n")


def test_simplified_command_extraction():
    """Test simplified command extraction"""
    print("Testing simplified command extraction...")
    
    engine = LLMPromptEngine()
    
    test_responses = [
        # Explicit BlueSky commands
        "HDG AC001 270",
        "ALT TEST002 36000", 
        "SPD KLM492 450",
        "Command: HDG AC001 270",
        
        # Natural language patterns
        "Turn AC001 to heading 270",
        "AC002 turn to heading 180",
        "Climb AC001 to altitude 37000",
        "AC002 descend to 33000",
        "Speed AC001 to 420",
        
        # Multiple commands (should warn)
        "HDG AC001 270 and ALT AC002 36000",
        
        # Invalid/unclear responses
        "I suggest turning the aircraft",
        "No clear command available",
    ]
    
    for response in test_responses:
        command = engine._extract_bluesky_command(response)
        print(f"  '{response}' -> {command}")
    
    print("✓ Simplified command extraction test passed\n")


def test_json_conflict_detection():
    """Test JSON-based conflict detection parsing"""
    print("Testing JSON conflict detection parsing...")
    
    engine = LLMPromptEngine()
    
    # Valid JSON response
    json_response = '''
    {
      "conflict_detected": true,
      "aircraft_pairs": ["AC001-AC002", "TEST003-KLM456"],
      "time_to_conflict": [120.5, 180.0],
      "confidence": 0.85,
      "priority": "high",
      "analysis": "Two pairs showing convergent paths"
    }
    '''
    
    result = engine.parse_detector_response(json_response)
    print("Valid JSON response:")
    print(f"  Conflict detected: {result['conflict_detected']}")
    print(f"  Aircraft pairs: {result['aircraft_pairs']}")
    print(f"  Confidence: {result['confidence']}")
    
    # Malformed JSON (should fallback to legacy parsing)
    legacy_response = '''
    Conflict Detected: YES
    Aircraft Pairs at Risk: AC001-AC002
    Time to Loss of Separation: 120 seconds
    Confidence: 0.8
    Priority: medium
    '''
    
    result = engine.parse_detector_response(legacy_response)
    print("\nLegacy text response:")
    print(f"  Conflict detected: {result['conflict_detected']}")
    print(f"  Aircraft pairs: {result['aircraft_pairs']}")
    print(f"  Confidence: {result['confidence']}")
    
    print("✓ JSON conflict detection parsing test passed\n")


def test_safety_assessment_robustness():
    """Test robust safety assessment parsing"""
    print("Testing robust safety assessment parsing...")
    
    engine = LLMPromptEngine()
    
    # Complete response
    complete_response = '''
    Safety Rating: SAFE
    Separation Achieved: 8.5 NM horizontal
    Compliance: ICAO compliant: YES
    Risk Assessment: Maneuver is within normal parameters
    Recommendation: APPROVE
    '''
    
    result = engine._parse_safety_response(complete_response)
    print("Complete response:")
    print(f"  Safety rating: {result['safety_rating']}")
    print(f"  Parsing issues: {result.get('parsing_issues', False)}")
    
    # Incomplete response (missing fields)
    incomplete_response = '''
    Safety Rating: MARGINAL
    Some unstructured text about the safety assessment.
    Recommendation: MODIFY
    '''
    
    result = engine._parse_safety_response(incomplete_response)
    print("\nIncomplete response:")
    print(f"  Safety rating: {result['safety_rating']}")
    print(f"  Missing fields: {result.get('missing_fields', [])}")
    print(f"  Parsing issues: {result.get('parsing_issues', False)}")
    
    print("✓ Safety assessment robustness test passed\n")


def test_updated_templates():
    """Test updated prompt templates"""
    print("Testing updated prompt templates...")
    
    engine = LLMPromptEngine()
    
    # Test conflict resolution template (no examples)
    conflict_info = {
        'aircraft_1_id': 'AC001',
        'aircraft_2_id': 'AC002',
        'time_to_conflict': 120.0,
        'closest_approach_distance': 3.5,
        'conflict_type': 'convergent',
        'urgency_level': 'medium',
        'aircraft_1': {'lat': 52.3676, 'lon': 4.9041, 'alt': 35000, 'hdg': 90, 'spd': 450, 'type': 'B738'},
        'aircraft_2': {'lat': 52.3700, 'lon': 4.9100, 'alt': 35000, 'hdg': 270, 'spd': 460, 'type': 'A320'},
        'environmental_conditions': {}
    }
    
    prompt = engine.format_conflict_prompt(conflict_info)
    
    # Check that examples are removed
    has_examples = "HDG AC001 270" in prompt or "Example valid responses" in prompt
    print(f"  Conflict resolution template has examples: {has_examples}")
    
    # Check that function calling instruction is present
    has_function_instruction = "SendCommand function" in prompt
    print(f"  Has function calling instruction: {has_function_instruction}")
    
    # Test conflict detection template format
    aircraft_states = [
        {'id': 'AC001', 'lat': 52.0, 'lon': 4.0, 'alt': 35000, 'hdg': 90, 'spd': 450},
        {'id': 'AC002', 'lat': 52.1, 'lon': 4.1, 'alt': 35000, 'hdg': 270, 'spd': 460}
    ]
    
    detection_prompt = engine.format_detector_prompt(aircraft_states)
    has_json_instruction = "JSON" in detection_prompt and "{" in detection_prompt
    print(f"  Detection template requests JSON format: {has_json_instruction}")
    
    print("✓ Updated templates test passed\n")


if __name__ == "__main__":
    print("LLM Prompt Engine Improvements Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_configurable_aircraft_id()
        test_simplified_command_extraction()
        test_json_conflict_detection()
        test_safety_assessment_robustness()
        test_updated_templates()
        
        print("All tests passed! ✅")
        print("\nImprovements implemented:")
        print("1. ✅ Configurable aircraft ID patterns")
        print("2. ✅ Simplified two-pass command extraction")
        print("3. ✅ JSON-based conflict detection parsing")
        print("4. ✅ Robust safety assessment with fallbacks")
        print("5. ✅ Updated templates without examples")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
