#!/usr/bin/env python3
"""
Enhanced training data analysis script to identify duplication issues and conflict patterns
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any

def analyze_data(data_file, max_samples=None):
    print(f"Analyzing {data_file}...")
    
    # Load data
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                data.append(item)
                if max_samples and line_num >= max_samples - 1:
                    break
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                continue
    
    print(f"Loaded {len(data)} samples for analysis")
    
from typing import Dict, List, Any, Optional

def analyze_data(data_file: str, max_samples: Optional[int] = None) -> None:
    print(f"Analyzing {data_file}...")
    
    # Load data
    data: List[Dict[str, Any]] = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                data.append(item)
                if max_samples and line_num >= max_samples - 1:
                    break
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                continue
    
    print(f"Loaded {len(data)} samples for analysis")
    
    # Basic instruction analysis
    instructions = [item.get('instruction', '') for item in data]
    unique_instructions = set(instructions)
    print(f"\nInstructions:")
    print(f"  Total: {len(instructions)}")
    print(f"  Unique: {len(unique_instructions)}")
    print(f"  Most common:")
    for instr, count in Counter(instructions).most_common(5):
        print(f"    {count:3d}: {instr[:80]}...")
    
    # Enhanced output analysis with action categorization
    outputs = [item.get('output', '') for item in data]
    unique_outputs = set(outputs)
    print(f"\nOutputs:")
    print(f"  Total: {len(outputs)}")
    print(f"  Unique: {len(unique_outputs)}")
    print(f"  Most common:")
    for output, count in Counter(outputs).most_common(10):
        print(f"    {count:3d}: {output[:80]}...")
    
    # Analyze action types
    action_types = analyze_action_types(outputs)
    print(f"\nAction Type Distribution:")
    for action_type, count in action_types.most_common():
        percentage = (count / len(outputs)) * 100
        print(f"  {action_type}: {count} ({percentage:.1f}%)")
    
    # Analyze conflict indicators
    conflict_analysis = analyze_conflicts(data)
    print(f"\nConflict Analysis:")
    print(f"  Samples with explicit conflicts: {conflict_analysis['explicit_conflicts']}")
    print(f"  Samples with multiple intruders: {conflict_analysis['multiple_intruders']}")
    print(f"  Samples with close proximity: {conflict_analysis['close_proximity']}")
    print(f"  Samples requiring evasive action: {conflict_analysis['evasive_actions']}")
    
    # Environment distribution
    environments = [item.get('metadata', {}).get('environment', '') for item in data]
    print(f"\nEnvironments:")
    for env, count in Counter(environments).most_common():
        print(f"  {env}: {count}")
    
    # Enhanced duplication analysis
    analyze_duplication_patterns(data)
    
    # Scenario complexity analysis
    analyze_scenario_complexity(data)


def analyze_action_types(outputs: List[str]) -> Counter[str]:
    """Categorize actions into types based on output content"""
    action_counter: Counter[str] = Counter()
    
    for output in outputs:
        if "maintain current heading" in output.lower():
            action_counter["Maintain Heading"] += 1
        elif "turn left" in output.lower() or "heading left" in output.lower():
            action_counter["Turn Left"] += 1
        elif "turn right" in output.lower() or "heading right" in output.lower():
            action_counter["Turn Right"] += 1
        elif "climb" in output.lower() or "altitude" in output.lower() and "increase" in output.lower():
            action_counter["Climb"] += 1
        elif "descend" in output.lower() or "altitude" in output.lower() and "decrease" in output.lower():
            action_counter["Descend"] += 1
        elif "speed" in output.lower() and "increase" in output.lower():
            action_counter["Speed Up"] += 1
        elif "speed" in output.lower() and ("decrease" in output.lower() or "reduce" in output.lower()):
            action_counter["Slow Down"] += 1
        elif "vector" in output.lower():
            action_counter["Vector"] += 1
        else:
            action_counter["Other/Maintain"] += 1
    
    return action_counter


def analyze_conflicts(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze conflict indicators in the training data"""
    analysis = {
        'explicit_conflicts': 0,
        'multiple_intruders': 0,
        'close_proximity': 0,
        'evasive_actions': 0
    }
    
    for item in data:
        input_text = item.get('input', '').lower()
        output_text = item.get('output', '').lower()
        
        # Check for explicit conflict mentions
        if any(word in input_text for word in ['conflict', 'collision', 'separation', 'breach']):
            analysis['explicit_conflicts'] += 1
        
        # Check for multiple intruders (potential conflicts)
        intruder_match = re.search(r'(\d+)\s+intruders?', input_text)
        if intruder_match and int(intruder_match.group(1)) > 1:
            analysis['multiple_intruders'] += 1
        
        # Check for close proximity indicators
        distance_patterns = [r'(\d+\.?\d*)\s*nm', r'waypoint\s+(\d+\.?\d*)', r'distance\s+(\d+\.?\d*)']
        for pattern in distance_patterns:
            distance_match = re.search(pattern, input_text)
            if distance_match and float(distance_match.group(1)) < 2.0:  # Less than 2 NM
                analysis['close_proximity'] += 1
                break
        
        # Check for evasive actions in output
        evasive_words = ['avoid', 'evade', 'turn', 'climb', 'descend', 'vector', 'immediate']
        if any(word in output_text for word in evasive_words) and 'maintain' not in output_text:
            analysis['evasive_actions'] += 1
    
    return analysis


def analyze_duplication_patterns(data: List[Dict[str, Any]]) -> None:
    """Analyze specific duplication patterns in the data"""
    print(f"\nDuplication Pattern Analysis:")
    
    # Analyze input variations
    inputs = [item.get('input', '') for item in data]
    unique_inputs = set(inputs)
    input_duplication = (1 - len(unique_inputs) / len(inputs)) * 100
    print(f"  Input duplication rate: {input_duplication:.1f}%")
    
    # Analyze parameter variations in inputs
    step_numbers: List[int] = []
    drift_angles: List[float] = []
    waypoint_distances: List[float] = []
    
    for input_text in inputs:
        # Extract step numbers
        step_match = re.search(r'step\s+(\d+)', input_text.lower())
        if step_match:
            step_numbers.append(int(step_match.group(1)))
        
        # Extract drift angles
        drift_match = re.search(r'(\d+\.?\d*)\s*°?\s*drift', input_text.lower())
        if drift_match:
            drift_angles.append(float(drift_match.group(1)))
        
        # Extract waypoint distances
        waypoint_match = re.search(r'waypoint\s+(\d+\.?\d*)\s*nm', input_text.lower())
        if waypoint_match:
            waypoint_distances.append(float(waypoint_match.group(1)))
    
    print(f"  Parameter variations:")
    if step_numbers:
        print(f"    Step numbers: {len(set(step_numbers))} unique values (range: {min(step_numbers)}-{max(step_numbers)})")
    if drift_angles:
        print(f"    Drift angles: {len(set(drift_angles))} unique values (range: {min(drift_angles):.1f}°-{max(drift_angles):.1f}°)")
    if waypoint_distances:
        print(f"    Waypoint distances: {len(set(waypoint_distances))} unique values (range: {min(waypoint_distances):.1f}-{max(waypoint_distances):.1f} NM)")
    
    # Analyze template-based generation
    combined = [f"{item.get('instruction', '')}|||{item.get('output', '')}" for item in data]
    unique_combined = set(combined)
    overall_duplication = (1 - len(unique_combined) / len(combined)) * 100
    print(f"  Overall duplication rate: {overall_duplication:.1f}%")


def analyze_scenario_complexity(data: List[Dict[str, Any]]) -> None:
    """Analyze the complexity and variety of scenarios"""
    print(f"\nScenario Complexity Analysis:")
    
    complexity_scores: List[int] = []
    for item in data:
        input_text = item.get('input', '').lower()
        output_text = item.get('output', '').lower()
        
        score = 0
        
        # Add complexity for number of intruders
        intruder_match = re.search(r'(\d+)\s+intruders?', input_text)
        if intruder_match:
            score += int(intruder_match.group(1))
        
        # Add complexity for close proximity
        if 'immediate attention' in input_text:
            score += 2
        
        # Add complexity for active maneuvering
        if any(action in output_text for action in ['turn', 'climb', 'descend', 'vector']) and 'maintain' not in output_text:
            score += 3
        
        # Add complexity for multiple action types
        action_count = sum(1 for action in ['turn', 'climb', 'descend', 'speed', 'vector'] if action in output_text)
        score += action_count
        
        complexity_scores.append(score)
    
    if complexity_scores:
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        print(f"  Average complexity score: {avg_complexity:.1f}")
        print(f"  Complexity distribution:")
        complexity_counter = Counter(complexity_scores)
        for score in sorted(complexity_counter.keys()):
            count = complexity_counter[score]
            percentage = (count / len(complexity_scores)) * 100
            print(f"    Score {score}: {count} samples ({percentage:.1f}%)")


def generate_improvement_recommendations() -> None:
    """Generate specific recommendations for improving training data diversity"""
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR IMPROVED TRAINING DATA")
    print(f"{'='*60}")
    
    recommendations = [
        "1. INCREASE CONFLICT DENSITY:",
        "   - Generate scenarios with guaranteed conflicts (separation < 5 NM)",
        "   - Force intruder proximity within critical distance zones",
        "   - Create time-pressure scenarios requiring immediate action",
        "",
        "2. DIVERSIFY ACTION TEMPLATES:",
        "   - Add heading change commands (turn left/right X degrees)",
        "   - Include altitude change instructions (climb/descend to FL X)",
        "   - Add speed adjustment commands (reduce/increase speed to X knots)",
        "   - Include vectoring instructions (vector heading X for separation)",
        "",
        "3. VARY SCENARIO COMPLEXITY:",
        "   - Multi-aircraft conflict scenarios (3+ aircraft)",
        "   - Sequential conflict chains (resolve A, then handle B)",
        "   - Weather-related diversions",
        "   - Emergency priority handling",
        "",
        "4. ENHANCE CONTEXTUAL DIVERSITY:",
        "   - Different airspace types (terminal, enroute, approach)",
        "   - Various aircraft types (heavy, light, military)",
        "   - Different weather conditions",
        "   - Time-of-day variations (rush hour, night ops)",
        "",
        "5. IMPROVE REASONING QUALITY:",
        "   - Include separation distance calculations",
        "   - Explain conflict geometry (head-on, overtaking, crossing)",
        "   - Mention relevant ATC procedures and regulations",
        "   - Include risk assessment rationale"
    ]
    
    for recommendation in recommendations:
        print(recommendation)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ATC training data for duplication and diversity issues")
    parser.add_argument("--data-file", default="f:\\LLM-ATC-HAL\\Bsky_SAC_Finetuned\\data\\combined_atc_training.jsonl",
                       help="Path to training data file")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to analyze (default: all)")
    parser.add_argument("--generate-recommendations", action="store_true",
                       help="Generate improvement recommendations")
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_data(args.data_file, args.max_samples)
    
    if args.generate_recommendations:
        generate_improvement_recommendations()
