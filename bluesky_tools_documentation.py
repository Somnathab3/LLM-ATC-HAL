#!/usr/bin/env python3
"""
Extended BlueSky Tools Documentation and Examples
================================================
Demonstrates the new capabilities added to BlueSky tools for Monte Carlo testing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools import (
    get_distance,
    step_simulation,
    reset_simulation,
    get_minimum_separation,
    check_separation_violation,
    TOOL_REGISTRY
)


def demonstrate_distance_calculation():
    """Demonstrate distance calculation capabilities"""
    print("📏 DISTANCE CALCULATION")
    print("=" * 50)
    print("""
The get_distance() function computes separation between aircraft using:
• Haversine formula for horizontal distance (great circle distance)
• Absolute altitude difference for vertical separation
• 3D Euclidean distance combining both components

Example usage:
""")
    
    # Example calculation
    result = get_distance("AAL123", "DLH456")
    
    print(f"""
distance_result = get_distance("AAL123", "DLH456")

Result:
{result}

Key metrics:
• horizontal_nm: Great circle distance in nautical miles
• vertical_ft: Absolute altitude difference in feet
• total_3d_nm: Combined 3D separation distance
""")


def demonstrate_simulation_control():
    """Demonstrate simulation time control"""
    print("\n⏰ SIMULATION TIME CONTROL")
    print("=" * 50)
    print("""
The step_simulation() function advances BlueSky simulation time:
• Sends DT (Delta Time) commands to BlueSky
• Accounts for simulation speed multiplier (dtmult)
• Returns comprehensive status information

Example usage:
""")
    
    # Example simulation step
    step_result = step_simulation(minutes=1.5, dtmult=2.0)
    
    print(f"""
step_result = step_simulation(minutes=1.5, dtmult=2.0)

Result:
{{
    "minutes_advanced": {step_result['minutes_advanced']},
    "seconds_advanced": {step_result['seconds_advanced']},
    "dtmult": {step_result['dtmult']},
    "command_sent": "{step_result['command_sent']}",
    "status": "{step_result['status']}"
}}

This advances simulation by 90 seconds at 2x speed.
""")


def demonstrate_simulation_reset():
    """Demonstrate simulation reset capabilities"""
    print("\n🔄 SIMULATION RESET")
    print("=" * 50)
    print("""
The reset_simulation() function completely resets BlueSky:
• Sends RESET command to clear all aircraft
• Applies standard initialization commands
• Sets up conflict detection parameters

Example usage:
""")
    
    # Example reset
    reset_result = reset_simulation()
    
    print(f"""
reset_result = reset_simulation()

Automatic setup commands executed:
• RESET - Clear all aircraft and state
• DTMULT 1 - Set normal time speed
• CDMETHOD SWARM - Enable swarm-based conflict detection
• CDSEP 5.0 1000 - Set separation minimums (5nm horizontal, 1000ft vertical)

Result status: {reset_result['status']}
""")


def demonstrate_separation_monitoring():
    """Demonstrate separation monitoring capabilities"""
    print("\n📐 SEPARATION MONITORING")
    print("=" * 50)
    print("""
Separation monitoring includes two key functions:

1. get_minimum_separation() - Returns current separation standards
2. check_separation_violation() - Checks if aircraft violate separation

Standards include different requirements for:
• Standard en-route operations (5nm/1000ft)
• Approach phases (3nm/500ft)
• Terminal areas (3nm/1000ft)
• Oceanic operations (10nm/1000ft)
• RVSM airspace (1000ft vertical)
""")
    
    # Get separation standards
    min_sep = get_minimum_separation()
    print(f"\nCurrent separation standards:")
    for key, value in min_sep.items():
        unit = "nm" if "horizontal" in key or "oceanic" in key else "ft"
        print(f"• {key}: {value} {unit}")
    
    # Check violation
    violation = check_separation_violation("AAL123", "DLH456")
    violations = violation.get('violations', {})
    
    print(f"""
Separation violation check example:
• Aircraft pair: {violation['aircraft_pair']}
• Horizontal violation: {'YES' if violations.get('horizontal') else 'NO'}
• Vertical violation: {'YES' if violations.get('vertical') else 'NO'}
• Separation loss: {'YES' if violations.get('separation_loss') else 'NO'}

Note: Separation loss occurs only when BOTH horizontal AND vertical 
minimums are violated simultaneously.
""")


def demonstrate_expanded_commands():
    """Demonstrate expanded command support"""
    print("\n🎮 EXPANDED COMMAND SUPPORT")
    print("=" * 50)
    print("""
Extended BlueSky command validation now supports:

Time Control:
• DT <seconds> - Advance simulation time
• DTMULT <factor> - Set time acceleration multiplier
• PAUSE - Pause simulation
• UNPAUSE - Resume simulation  
• FF <factor> - Fast-forward multiplier

Aircraft Control:
• VS <aircraft> <rate> - Set vertical speed (feet/min)
• CRE <aircraft> <type> <lat> <lon> <hdg> <alt> <spd> - Create aircraft
• DEL <aircraft> - Delete aircraft

Simulation Setup:
• RESET - Reset simulation to initial state
• AREA <lat>,<lon> - Set simulation area center
• CDMETHOD <method> - Set conflict detection method
• CDSEP <h_sep> <v_sep> - Set separation minimums

Environment:
• WIND <lat>,<lon>,<layer>,<dir>,<spd> - Set wind conditions
• TURB <intensity> - Set turbulence level
• IC - Initial conditions command
• GO - Start/continue simulation

All commands are validated and logged for debugging purposes.
""")


def demonstrate_tool_registry():
    """Demonstrate function calling registry"""
    print("\n🔧 FUNCTION CALLING REGISTRY")
    print("=" * 50)
    print("""
The TOOL_REGISTRY has been expanded to include all new functions,
enabling LLM function calling capabilities:
""")
    
    print("Available tools for LLM function calling:")
    for i, tool_name in enumerate(sorted(TOOL_REGISTRY.keys()), 1):
        print(f"{i:2d}. {tool_name}")
    
    print(f"""
Total: {len(TOOL_REGISTRY)} tools available

New tools added for Monte Carlo testing:
• GetDistance - Calculate aircraft separation
• StepSimulation - Advance simulation time
• ResetSimulation - Reset to initial state
• GetMinimumSeparation - Get separation standards
• CheckSeparationViolation - Monitor separation compliance

These tools enable the Monte Carlo runner to:
1. Reset simulations between scenarios
2. Load scenario commands step by step
3. Monitor aircraft separation continuously
4. Detect and verify conflict resolutions
5. Calculate path length deviations
""")


def demonstrate_monte_carlo_integration():
    """Demonstrate integration with Monte Carlo testing"""
    print("\n🎯 MONTE CARLO INTEGRATION")
    print("=" * 50)
    print("""
The extended BlueSky tools enable the Monte Carlo runner to:

1. SCENARIO SETUP:
   • reset_simulation() - Clean slate for each test
   • send_command() - Load scenario aircraft and commands
   • step_simulation() - Advance to conflict detection point

2. CONFLICT DETECTION:
   • get_all_aircraft_info() - Get current aircraft states
   • get_distance() - Calculate separations
   • check_separation_violation() - Identify conflicts

3. RESOLUTION TESTING:
   • send_command() - Execute LLM-generated resolutions
   • step_simulation() - Advance time to verify effectiveness
   • get_distance() - Monitor separation continuously

4. VERIFICATION:
   • check_separation_violation() - Confirm conflict resolution
   • Calculate minimum achieved separation
   • Measure extra path length from original trajectory

Example Monte Carlo pipeline:
""")
    
    print("""
# 1. Setup
reset_simulation()
for cmd in scenario.commands:
    send_command(cmd)

# 2. Detection
aircraft_info = get_all_aircraft_info()
conflicts = detect_conflicts(aircraft_info)

# 3. Resolution
if conflicts:
    resolution_cmd = llm_prompt_engine.get_conflict_resolution(conflict_info)
    send_command(resolution_cmd)

# 4. Verification
step_simulation(minutes=5.0)
final_separation = get_distance(aircraft1, aircraft2)
violation_status = check_separation_violation(aircraft1, aircraft2)

# 5. Metrics
min_separation = track_minimum_separation_over_time()
extra_path_length = calculate_trajectory_deviation()
""")


def main():
    """Run all demonstrations"""
    print("📚 EXTENDED BLUESKY TOOLS DOCUMENTATION")
    print("=" * 60)
    print("Comprehensive guide to new Monte Carlo testing capabilities")
    
    demonstrations = [
        demonstrate_distance_calculation,
        demonstrate_simulation_control,
        demonstrate_simulation_reset,
        demonstrate_separation_monitoring,
        demonstrate_expanded_commands,
        demonstrate_tool_registry,
        demonstrate_monte_carlo_integration,
    ]
    
    for demo in demonstrations:
        demo()
    
    print("\n" + "=" * 60)
    print("✅ DOCUMENTATION COMPLETE")
    print("=" * 60)
    print("""
The extended BlueSky tools provide comprehensive capabilities for:
• Aircraft separation monitoring and calculation
• Simulation time control and stepping
• Command validation and execution
• Function calling integration with LLMs
• Monte Carlo testing pipeline support

These tools enable sophisticated testing of LLM-based conflict
resolution in realistic air traffic control scenarios.

For implementation examples, see:
• test_extended_bluesky_tools.py - Comprehensive test suite
• scenarios/monte_carlo_runner.py - Integration example
• llm_atc/tools/bluesky_tools.py - Source implementation
""")


if __name__ == "__main__":
    main()
