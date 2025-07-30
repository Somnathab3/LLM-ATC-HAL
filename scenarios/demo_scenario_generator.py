# scenarios/demo_scenario_generator.py
"""
Demonstration of the Scenario Generator Module
==============================================
This script demonstrates the three environment classes and their capabilities:
- HorizontalCREnv: Same-altitude conflict scenarios
- VerticalCREnv: Altitude-based conflict scenarios
- SectorCREnv: Full-sector realistic scenarios

Shows how to generate scenarios with precise ground truth for testing
false positive/negative rates in conflict detection systems.
"""

import time
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.scenario_generator import (
    ScenarioGenerator,
    ScenarioType,
    HorizontalCREnv,
    VerticalCREnv,  
    SectorCREnv,
    ComplexityTier,
    generate_horizontal_scenario,
    generate_vertical_scenario,
    generate_sector_scenario
)


def demo_horizontal_scenarios():
    """Demonstrate horizontal conflict scenario generation"""
    print("🛩️  HORIZONTAL CONFLICT SCENARIOS")
    print("="*50)
    
    env = HorizontalCREnv()
    
    # Generate conflict scenario
    print("1. Generating conflict scenario...")
    conflict_scenario = env.generate_scenario(n_aircraft=3, conflict=True)
    
    print(f"   Scenario ID: {conflict_scenario.scenario_id}")
    print(f"   Aircraft count: {conflict_scenario.aircraft_count}")
    
    # Check altitudes are same
    altitudes = [state['altitude'] for state in conflict_scenario.initial_states]
    print(f"   Altitudes: {altitudes} ft (all same = {len(set(altitudes)) == 1})")
    
    # Show conflicts
    print(f"   Ground truth conflicts: {len(conflict_scenario.ground_truth_conflicts)}")
    for i, conflict in enumerate(conflict_scenario.ground_truth_conflicts):
        print(f"      Conflict {i+1}: {conflict.aircraft_pair} - {conflict.conflict_type} - {conflict.severity}")
    
    # Generate safe scenario
    print("\n2. Generating safe scenario...")
    safe_scenario = env.generate_scenario(n_aircraft=3, conflict=False)
    print(f"   Safe scenario conflicts: {len(safe_scenario.ground_truth_conflicts)}")
    
    # Show some commands
    print(f"\n3. Sample BlueSky commands:")
    for cmd in conflict_scenario.commands[:5]:
        print(f"      {cmd}")
    
    print()


def demo_vertical_scenarios():
    """Demonstrate vertical conflict scenario generation"""
    print("🛩️  VERTICAL CONFLICT SCENARIOS")
    print("="*50)
    
    env = VerticalCREnv()
    
    # Generate vertical conflict scenario
    print("1. Generating vertical conflict scenario...")
    scenario = env.generate_scenario(n_aircraft=2, conflict=True)
    
    print(f"   Scenario ID: {scenario.scenario_id}")
    print(f"   Aircraft count: {scenario.aircraft_count}")
    
    # Check altitudes are different
    altitudes = [state['altitude'] for state in scenario.initial_states]
    print(f"   Altitudes: {altitudes} ft (different = {len(set(altitudes)) > 1})")
    
    # Check for vertical rates
    vertical_rates = [state.get('vertical_rate', 0) for state in scenario.initial_states]
    print(f"   Vertical rates: {vertical_rates} fpm")
    
    # Show conflicts
    print(f"   Ground truth conflicts: {len(scenario.ground_truth_conflicts)}")
    for i, conflict in enumerate(scenario.ground_truth_conflicts):
        print(f"      Conflict {i+1}: {conflict.aircraft_pair} - {conflict.conflict_type}")
        sep = conflict.min_separation
        print(f"         Min separation: {sep['horizontal_nm']:.1f} nm horizontal, {sep['vertical_ft']:.0f} ft vertical")
    
    # Look for altitude/vertical speed commands
    alt_vs_commands = [cmd for cmd in scenario.commands if 'ALT' in cmd or 'VS' in cmd]
    if alt_vs_commands:
        print(f"\n2. Vertical maneuver commands:")
        for cmd in alt_vs_commands:
            print(f"      {cmd}")
    
    print()


def demo_sector_scenarios():
    """Demonstrate sector scenario generation"""
    print("🛩️  SECTOR SCENARIOS")
    print("="*50)
    
    env = SectorCREnv()
    
    # Generate scenarios of different complexity
    complexities = [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX]
    
    for complexity in complexities:
        print(f"{complexity.value.upper()} Complexity:")
        scenario = env.generate_scenario(complexity=complexity, force_conflicts=False)
        
        print(f"   Aircraft count: {scenario.aircraft_count}")
        print(f"   Airspace region: {scenario.airspace_region}")
        print(f"   Environmental conditions:")
        env_cond = scenario.environmental_conditions
        print(f"      Wind: {env_cond.get('wind_speed_kts', 0)} kts @ {env_cond.get('wind_direction_deg', 0)}°")
        print(f"      Visibility: {env_cond.get('visibility_nm', 10)} nm")
        print(f"      Turbulence: {env_cond.get('turbulence_intensity', 0):.2f}")
        
        print(f"   Ground truth conflicts: {len(scenario.ground_truth_conflicts)}")
        print(f"   BlueSky commands: {len(scenario.commands)}")
        print()


def demo_distribution_shift():
    """Demonstrate distribution shift scenarios"""
    print("🛩️  DISTRIBUTION SHIFT SCENARIOS")
    print("="*50)
    
    generator = ScenarioGenerator()
    shift_levels = ["in_distribution", "moderate_shift", "extreme_shift"]
    
    for shift_level in shift_levels:
        print(f"{shift_level.upper()}:")
        try:
            scenario = generator.generate_sector_scenario(
                complexity=ComplexityTier.MODERATE,
                shift_level=shift_level,
                force_conflicts=True
            )
            
            print(f"   Aircraft count: {scenario.aircraft_count}")
            print(f"   Distribution shift: {scenario.distribution_shift_tier}")
            
            env_cond = scenario.environmental_conditions
            print(f"   Wind conditions: {env_cond.get('wind_speed_kts', 0)} kts")
            print(f"   Conflicts detected: {len(scenario.ground_truth_conflicts)}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to generate {shift_level}: {e}")
        
        print()


def demo_ground_truth_analysis():
    """Demonstrate ground truth conflict analysis"""
    print("🛩️  GROUND TRUTH CONFLICT ANALYSIS")
    print("="*50)
    
    # Generate a complex scenario
    generator = ScenarioGenerator()
    scenario = generator.generate_sector_scenario(
        complexity=ComplexityTier.COMPLEX,
        force_conflicts=True
    )
    
    print(f"Analyzing scenario: {scenario.scenario_id}")
    print(f"Aircraft count: {scenario.aircraft_count}")
    print(f"Total conflicts: {len(scenario.ground_truth_conflicts)}")
    print()
    
    # Analyze conflicts by type
    conflict_types = {}
    severity_counts = {}
    
    for conflict in scenario.ground_truth_conflicts:
        # Count by type
        conflict_types[conflict.conflict_type] = conflict_types.get(conflict.conflict_type, 0) + 1
        
        # Count by severity
        severity_counts[conflict.severity] = severity_counts.get(conflict.severity, 0) + 1
        
        # Show detailed info for first few conflicts
        if len([c for c in scenario.ground_truth_conflicts if c == conflict]) <= 3:
            print(f"Conflict: {conflict.aircraft_pair}")
            print(f"   Type: {conflict.conflict_type}")
            print(f"   Severity: {conflict.severity}")
            print(f"   Time to conflict: {conflict.time_to_conflict:.0f} seconds")
            print(f"   Is actual violation: {conflict.is_actual_conflict}")
            sep = conflict.min_separation
            print(f"   Min separation: {sep['horizontal_nm']:.1f} nm, {sep['vertical_ft']:.0f} ft")
            print()
    
    print("Conflict Summary:")
    print(f"   By type: {conflict_types}")
    print(f"   By severity: {severity_counts}")
    
    # Calculate false positive/negative baseline
    actual_conflicts = sum(1 for c in scenario.ground_truth_conflicts if c.is_actual_conflict)
    potential_conflicts = len(scenario.ground_truth_conflicts)
    
    print(f"   Actual separation violations: {actual_conflicts}")
    print(f"   Potential conflicts: {potential_conflicts}")
    print(f"   Expected detection rate: {actual_conflicts / max(potential_conflicts, 1):.2%}")


def demo_scenario_integration():
    """Demonstrate integration with existing code"""
    print("🛩️  INTEGRATION WITH EXISTING CODE")
    print("="*50)
    
    # Show how scenarios convert to existing format
    scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)
    
    # Convert to dict for compatibility
    scenario_dict = scenario.to_dict()
    
    print("Scenario structure for integration:")
    print(f"   scenario_id: {scenario_dict['scenario_id']}")
    print(f"   aircraft_count: {scenario_dict['aircraft_count']}")
    print(f"   commands: {len(scenario_dict['commands'])} BlueSky commands")
    print(f"   initial_states: {len(scenario_dict['initial_states'])} aircraft")
    print(f"   ground_truth_conflicts: {len(scenario_dict['ground_truth_conflicts'])} conflicts")
    
    # Show aircraft state format
    print("\nSample aircraft state:")
    if scenario.initial_states:
        state = scenario.initial_states[0]
        for key, value in state.items():
            print(f"   {key}: {value}")
    
    # Show commands format
    print(f"\nSample BlueSky commands:")
    for cmd in scenario.commands[:3]:
        print(f"   {cmd}")


def main():
    """Main demonstration"""
    print("🎯 SCENARIO GENERATOR MODULE DEMONSTRATION")
    print("="*60)
    print("Demonstrating environment-specific scenario generation")
    print("for Horizontal, Vertical, and Sector conflict scenarios.")
    print("="*60)
    print()
    
    try:
        # Run all demonstrations
        demo_horizontal_scenarios()
        demo_vertical_scenarios()
        demo_sector_scenarios()
        demo_distribution_shift()
        demo_ground_truth_analysis()
        demo_scenario_integration()
        
        print("✅ All demonstrations completed successfully!")
        print("\n🎉 The scenario generator module is ready for use!")
        print("\nKey capabilities demonstrated:")
        print("• Horizontal conflicts with same altitude")
        print("• Vertical conflicts with climb/descent")
        print("• Sector scenarios with varying complexity")
        print("• Distribution shift testing")
        print("• Ground truth conflict labeling")
        print("• Integration with existing codebase")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
