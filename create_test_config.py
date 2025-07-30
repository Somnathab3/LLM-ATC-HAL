#!/usr/bin/env python3
"""
Quick test configuration for comprehensive_hallucination_tester_v2.py
This runs a minimal OFAT sweep to test the real LLM integration.
"""

import yaml

# Create a minimal test configuration
minimal_config = {
    'altitude': {
        'vertical_rate_fpm': [500, 1500]  # Only one parameter for quick test
    },
    'speed': {
        'cruise_speed_variation': [0.85, 1.15]
    },
    'traffic': {
        'density_multiplier': [0.5, 1.5]
    }
}

# Save the minimal configuration
with open('test_scenario_ranges.yaml', 'w') as f:
    yaml.dump(minimal_config, f, default_flow_style=False)

print("Created minimal test configuration in test_scenario_ranges.yaml")
print("You can now run:")
print("python comprehensive_hallucination_tester_v2.py")
print("(after modifying it to use test_scenario_ranges.yaml)")
