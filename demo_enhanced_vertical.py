#!/usr/bin/env python3
"""
Demo script for enhanced vertical scenario generation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.scenario_generator import ScenarioGenerator

def demo_enhanced_vertical_generation():
    """Demonstrate enhanced vertical scenario generation"""
    generator = ScenarioGenerator()
    
    print('=== Enhanced Vertical Scenario Generation Demo ===')
    print()
    
    # Test with custom climb rates and crossing altitudes
    custom_climb_rates = [-2000, 1500, -1000]
    custom_crossing_altitudes = [32000, 36000, 34000]
    
    scenario = generator.generate_vertical_scenario(
        n_aircraft=3,
        conflict=True,
        climb_rates=custom_climb_rates,
        crossing_altitudes=custom_crossing_altitudes
    )
    
    print(f'Scenario ID: {scenario.scenario_id}')
    print(f'Aircraft Count: {scenario.aircraft_count}')
    print(f'Scenario Type: {scenario.scenario_type.value}')
    print(f'Has Conflicts: {scenario.has_conflicts}')
    print()
    
    print('Initial Aircraft States:')
    for i, state in enumerate(scenario.initial_states):
        callsign = state['callsign']
        altitude = state['altitude']
        target_altitude = state['target_altitude']
        climb_rate = state['assigned_climb_rate']
        print(f'  {callsign}: Alt={altitude}ft, Target={target_altitude}ft, Rate={climb_rate}fpm')
    
    print()
    print('Extended Fields (for benchmark runner):')
    print(f'  Predicted Conflicts: {len(scenario.predicted_conflicts)} (empty - to be filled by LLM)')
    print(f'  Resolution Commands: {len(scenario.resolution_commands)} (empty - to be filled by LLM)')
    print(f'  Trajectories: {len(scenario.trajectories)} (empty - to be filled by simulation)')
    print(f'  Success: {scenario.success} (None - to be determined by benchmark)')
    
    print()
    print('BlueSky Commands Generated:')
    for i, cmd in enumerate(scenario.commands[:10]):  # Show first 10 commands
        print(f'  {i+1:2d}: {cmd}')
    if len(scenario.commands) > 10:
        print(f'  ... and {len(scenario.commands)-10} more commands')

if __name__ == '__main__':
    demo_enhanced_vertical_generation()
