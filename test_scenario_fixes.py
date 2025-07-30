#!/usr/bin/env python3
"""
Test script to verify the fixed distribution shift functionality.
This script tests the scenarios that were previously failing with 
"empty range in randrange" errors.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from scenarios.scenario_generator import ScenarioGenerator, ComplexityTier

def test_distribution_shifts():
    """Test all distribution shift combinations that were previously failing."""
    
    print("=== Testing Fixed Distribution Shift Functionality ===\n")
    
    generator = ScenarioGenerator()
    
    # Test cases that were previously failing
    test_cases = [
        ('in_distribution', ComplexityTier.SIMPLE),
        ('in_distribution', ComplexityTier.MODERATE), 
        ('in_distribution', ComplexityTier.COMPLEX),
        ('moderate_shift', ComplexityTier.SIMPLE),
        ('moderate_shift', ComplexityTier.MODERATE),
        ('moderate_shift', ComplexityTier.COMPLEX),
        ('extreme_shift', ComplexityTier.SIMPLE),
        ('extreme_shift', ComplexityTier.MODERATE),
        ('extreme_shift', ComplexityTier.COMPLEX)
    ]
    
    print("Testing all distribution shift and complexity combinations:")
    print("=" * 70)
    success_count = 0
    
    for shift_level, complexity in test_cases:
        try:
            scenario = generator.generate_sector_scenario(
                complexity=complexity,
                shift_level=shift_level,
                force_conflicts=True
            )
            
            status = "✓"
            result = f"{scenario.aircraft_count:2d} aircraft, {len(scenario.ground_truth_conflicts)} conflicts"
            success_count += 1
            
        except Exception as e:
            status = "✗"
            result = f"FAILED - {str(e)[:50]}"
        
        print(f"{status} {shift_level:15} {complexity.value:8}: {result}")
    
    print("=" * 70)
    print(f"Success rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    
    if success_count == len(test_cases):
        print("🎉 All tests PASSED! Distribution shift issues have been fixed.")
    else:
        print(f"⚠️  {len(test_cases) - success_count} tests still failing.")
    
    return success_count == len(test_cases)


def test_scenario_types():
    """Test all three scenario environment types."""
    
    print("\n=== Testing Scenario Environment Types ===\n")
    
    generator = ScenarioGenerator()
    
    test_cases = [
        ("Horizontal", lambda: generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)),
        ("Vertical", lambda: generator.generate_vertical_scenario(n_aircraft=2, conflict=True)),
        ("Sector", lambda: generator.generate_sector_scenario(complexity=ComplexityTier.MODERATE))
    ]
    
    success_count = 0
    for name, test_func in test_cases:
        try:
            scenario = test_func()
            print(f"✓ {name:10}: {scenario.aircraft_count} aircraft, {len(scenario.ground_truth_conflicts)} conflicts")
            success_count += 1
        except Exception as e:
            print(f"✗ {name:10}: FAILED - {e}")
    
    print(f"\nEnvironment types: {success_count}/{len(test_cases)} working")
    return success_count == len(test_cases)


if __name__ == "__main__":
    print("Testing Scenario Generator Fixes")
    print("=" * 50)
    
    # Test distribution shifts
    shift_success = test_distribution_shifts()
    
    # Test scenario types
    env_success = test_scenario_types()
    
    print("\n" + "=" * 50)
    if shift_success and env_success:
        print("🎉 ALL TESTS PASSED! Scenario generator is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)
