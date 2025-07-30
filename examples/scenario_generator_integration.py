# examples/scenario_generator_integration.py
"""
Example integration of the Scenario Generator Module with the 
LLM-ATC-HAL framework, particularly the distribution shift experiment runner.

This shows how to use the environment-specific scenario generation 
within the existing experiment infrastructure.
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.scenario_generator import (
    ScenarioGenerator,
    HorizontalCREnv,
    VerticalCREnv,
    SectorCREnv,
    ComplexityTier
)


def example_horizontal_testing():
    """Example: Testing horizontal conflict detection with controlled scenarios"""
    print("🧪 Example: Horizontal Conflict Detection Testing")
    print("="*60)
    
    h_env = HorizontalCREnv()
    
    # Generate test suite for horizontal conflicts
    test_cases = []
    
    # True positive cases (should detect conflicts)
    for i in range(5):
        scenario = h_env.generate_scenario(n_aircraft=2, conflict=True)
        test_cases.append({
            'scenario': scenario,
            'expected_result': 'conflict',
            'ground_truth_count': len(scenario.ground_truth_conflicts)
        })
    
    # True negative cases (should not detect conflicts)
    for i in range(5):
        scenario = h_env.generate_scenario(n_aircraft=2, conflict=False)
        test_cases.append({
            'scenario': scenario,
            'expected_result': 'safe',
            'ground_truth_count': len(scenario.ground_truth_conflicts)
        })
    
    print(f"Generated {len(test_cases)} horizontal test scenarios")
    
    # Simulate testing with an LLM conflict detector
    fp_count = 0  # False positives
    fn_count = 0  # False negatives
    tp_count = 0  # True positives  
    tn_count = 0  # True negatives
    
    for i, test_case in enumerate(test_cases):
        scenario = test_case['scenario']
        expected = test_case['expected_result']
        ground_truth = test_case['ground_truth_count']
        
        # Simulate LLM detection (mock)
        # In real usage, you would call your LLM system here
        detected_conflicts = mock_llm_conflict_detection(scenario)
        
        # Evaluate results
        if expected == 'conflict' and detected_conflicts > 0:
            tp_count += 1
            result = "TP"
        elif expected == 'safe' and detected_conflicts == 0:
            tn_count += 1
            result = "TN"
        elif expected == 'conflict' and detected_conflicts == 0:
            fn_count += 1
            result = "FN"
        else:
            fp_count += 1
            result = "FP"
        
        print(f"Test {i+1:2d}: {result} - Expected: {expected:8s}, "
              f"Ground truth: {ground_truth}, Detected: {detected_conflicts}")
    
    # Calculate metrics
    total = len(test_cases)
    accuracy = (tp_count + tn_count) / total
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  TP: {tp_count}, TN: {tn_count}, FP: {fp_count}, FN: {fn_count}")


def example_distribution_shift_integration():
    """Example: Integration with distribution shift experiments"""
    print("\n🌪️  Example: Distribution Shift Integration")
    print("="*60)
    
    generator = ScenarioGenerator()
    shift_levels = ["in_distribution", "moderate_shift", "extreme_shift"]
    complexities = [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX]
    
    results = {}
    
    for shift_level in shift_levels:
        for complexity in complexities:
            scenarios = []
            total_conflicts = 0
            
            # Generate 5 scenarios for each combination
            for i in range(5):
                try:
                    scenario = generator.generate_sector_scenario(
                        complexity=complexity,
                        shift_level=shift_level,
                        force_conflicts=True
                    )
                    scenarios.append(scenario)
                    total_conflicts += len(scenario.ground_truth_conflicts)
                except Exception as e:
                    print(f"  ⚠️  Failed to generate {shift_level}/{complexity.value}: {e}")
                    continue
            
            if scenarios:
                avg_aircraft = sum(s.aircraft_count for s in scenarios) / len(scenarios)
                avg_conflicts = total_conflicts / len(scenarios)
                
                key = f"{shift_level}_{complexity.value}"
                results[key] = {
                    'scenarios': len(scenarios),
                    'avg_aircraft': avg_aircraft,
                    'avg_conflicts': avg_conflicts,
                    'shift_level': shift_level,
                    'complexity': complexity.value
                }
                
                print(f"{shift_level:15s} {complexity.value:8s}: "
                      f"{avg_aircraft:5.1f} aircraft, {avg_conflicts:4.1f} conflicts")
    
    # Analyze distribution shift effects
    print(f"\nDistribution Shift Analysis:")
    for complexity in complexities:
        baseline_key = f"in_distribution_{complexity.value}"
        moderate_key = f"moderate_shift_{complexity.value}"
        extreme_key = f"extreme_shift_{complexity.value}"
        
        if all(key in results for key in [baseline_key, moderate_key, extreme_key]):
            baseline = results[baseline_key]
            moderate = results[moderate_key]
            extreme = results[extreme_key]
            
            print(f"  {complexity.value.upper()}:")
            print(f"    Aircraft count increase: "
                  f"Baseline→Moderate: +{moderate['avg_aircraft'] - baseline['avg_aircraft']:.1f}, "
                  f"Baseline→Extreme: +{extreme['avg_aircraft'] - baseline['avg_aircraft']:.1f}")
            print(f"    Conflict count increase: "
                  f"Baseline→Moderate: +{moderate['avg_conflicts'] - baseline['avg_conflicts']:.1f}, "
                  f"Baseline→Extreme: +{extreme['avg_conflicts'] - baseline['avg_conflicts']:.1f}")


def example_vertical_conflict_analysis():
    """Example: Detailed vertical conflict analysis"""
    print("\n📈 Example: Vertical Conflict Analysis")
    print("="*60)
    
    v_env = VerticalCREnv()
    
    # Generate vertical scenarios with different aircraft counts
    for n_aircraft in [2, 3]:
        print(f"\n{n_aircraft}-Aircraft Vertical Scenarios:")
        
        conflicts_detected = 0
        total_scenarios = 10
        
        for i in range(total_scenarios):
            scenario = v_env.generate_scenario(n_aircraft=n_aircraft, conflict=True)
            
            # Analyze altitude differences
            altitudes = [state['altitude'] for state in scenario.initial_states]
            altitude_diffs = []
            for j in range(len(altitudes)):
                for k in range(j + 1, len(altitudes)):
                    altitude_diffs.append(abs(altitudes[j] - altitudes[k]))
            
            min_alt_diff = min(altitude_diffs) if altitude_diffs else 0
            
            # Check for vertical maneuver commands
            has_vertical_commands = any('ALT' in cmd or 'VS' in cmd for cmd in scenario.commands)
            
            # Count ground truth conflicts
            ground_truth_conflicts = len(scenario.ground_truth_conflicts)
            conflicts_detected += ground_truth_conflicts
            
            print(f"  Scenario {i+1:2d}: {ground_truth_conflicts} conflicts, "
                  f"min alt diff: {min_alt_diff:4.0f} ft, "
                  f"vertical cmds: {'Yes' if has_vertical_commands else 'No'}")
        
        conflict_rate = conflicts_detected / total_scenarios
        print(f"  Average conflicts per scenario: {conflict_rate:.1f}")


def example_batch_scenario_generation():
    """Example: Batch generation for comprehensive testing"""
    print("\n📦 Example: Batch Scenario Generation")
    print("="*60)
    
    generator = ScenarioGenerator()
    
    # Define test matrix
    test_matrix = {
        'horizontal': {
            'aircraft_counts': [2, 3, 4],
            'conflict_settings': [True, False]
        },
        'vertical': {
            'aircraft_counts': [2, 3],
            'conflict_settings': [True, False]
        },
        'sector': {
            'complexities': [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX],
            'shift_levels': ['in_distribution', 'moderate_shift']
        }
    }
    
    all_scenarios = []
    
    # Generate horizontal scenarios
    print("Generating horizontal scenarios...")
    for n_aircraft in test_matrix['horizontal']['aircraft_counts']:
        for conflict in test_matrix['horizontal']['conflict_settings']:
            scenario = generator.generate_horizontal_scenario(
                n_aircraft=n_aircraft,
                conflict=conflict
            )
            all_scenarios.append({
                'type': 'horizontal',
                'scenario': scenario,
                'params': {'n_aircraft': n_aircraft, 'conflict': conflict}
            })
    
    # Generate vertical scenarios
    print("Generating vertical scenarios...")
    for n_aircraft in test_matrix['vertical']['aircraft_counts']:
        for conflict in test_matrix['vertical']['conflict_settings']:
            scenario = generator.generate_vertical_scenario(
                n_aircraft=n_aircraft,
                conflict=conflict
            )
            all_scenarios.append({
                'type': 'vertical',
                'scenario': scenario,
                'params': {'n_aircraft': n_aircraft, 'conflict': conflict}
            })
    
    # Generate sector scenarios
    print("Generating sector scenarios...")
    for complexity in test_matrix['sector']['complexities']:
        for shift_level in test_matrix['sector']['shift_levels']:
            try:
                scenario = generator.generate_sector_scenario(
                    complexity=complexity,
                    shift_level=shift_level,
                    force_conflicts=True
                )
                all_scenarios.append({
                    'type': 'sector',
                    'scenario': scenario,
                    'params': {'complexity': complexity.value, 'shift_level': shift_level}
                })
            except Exception as e:
                print(f"  Failed to generate sector {complexity.value}/{shift_level}: {e}")
    
    print(f"\nGenerated {len(all_scenarios)} total scenarios:")
    
    # Summarize by type
    type_counts = {}
    conflict_counts = {}
    
    for item in all_scenarios:
        scenario_type = item['type']
        scenario = item['scenario']
        
        type_counts[scenario_type] = type_counts.get(scenario_type, 0) + 1
        conflict_counts[scenario_type] = conflict_counts.get(scenario_type, 0) + len(scenario.ground_truth_conflicts)
    
    for scenario_type in ['horizontal', 'vertical', 'sector']:
        if scenario_type in type_counts:
            count = type_counts[scenario_type]
            conflicts = conflict_counts[scenario_type]
            avg_conflicts = conflicts / count if count > 0 else 0
            print(f"  {scenario_type:10s}: {count:2d} scenarios, {conflicts:3d} total conflicts ({avg_conflicts:.1f} avg)")


def mock_llm_conflict_detection(scenario):
    """Mock LLM conflict detection for demonstration purposes"""
    # Simulate LLM analysis with some randomness and bias
    import random
    
    # Base detection on ground truth with some noise
    ground_truth_count = len(scenario.ground_truth_conflicts)
    
    # Simulate 85% accuracy with some false positives/negatives
    if ground_truth_count > 0:
        # True positive case - detect most conflicts
        if random.random() < 0.85:
            return max(1, ground_truth_count + random.randint(-1, 1))
        else:
            return 0  # False negative
    else:
        # True negative case - most should detect no conflicts
        if random.random() < 0.90:
            return 0
        else:
            return random.randint(1, 2)  # False positive


def main():
    """Run all integration examples"""
    print("🔗 SCENARIO GENERATOR INTEGRATION EXAMPLES")
    print("="*70)
    print("Demonstrating integration with LLM-ATC-HAL framework")
    print("="*70)
    
    # Setup logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    try:
        example_horizontal_testing()
        example_distribution_shift_integration()
        example_vertical_conflict_analysis()
        example_batch_scenario_generation()
        
        print("\n" + "="*70)
        print("✅ All integration examples completed successfully!")
        print("\n💡 Key Integration Points:")
        print("• Use environment-specific generators for targeted testing")
        print("• Leverage ground truth data for accurate FP/FN analysis")
        print("• Integrate with distribution shift experiments")
        print("• Generate comprehensive test suites efficiently")
        print("• Analyze conflict patterns across scenario types")
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
