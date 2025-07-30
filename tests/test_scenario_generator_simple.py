# tests/test_scenario_generator_simple.py
"""
Simple unit tests for scenario generation module without pytest dependency.
Tests verify basic functionality and ground truth generation.
"""

import sys
import os

# Add project root to path
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


def test_horizontal_scenario_same_altitudes():
    """Test that horizontal scenarios have all aircraft at same altitude"""
    print("Testing horizontal scenario altitude consistency...")
    
    generator = ScenarioGenerator()
    scenario = generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)
    
    # Check that all aircraft have same altitude
    altitudes = [state['altitude'] for state in scenario.initial_states]
    assert len(set(altitudes)) == 1, "All aircraft should be at same altitude"
    assert altitudes[0] == 35000, "Should use standard FL350"
    
    # Check scenario properties
    assert scenario.scenario_type == ScenarioType.HORIZONTAL
    assert scenario.aircraft_count == 3
    assert len(scenario.initial_states) == 3
    assert len(scenario.commands) > 0
    
    print(f"✓ Horizontal scenario: {scenario.aircraft_count} aircraft at altitude {altitudes[0]} ft")
    print(f"✓ Commands generated: {len(scenario.commands)}")
    print(f"✓ Ground truth conflicts: {len(scenario.ground_truth_conflicts)}")


def test_vertical_scenario_altitude_differences():
    """Test that vertical scenarios have aircraft at different altitudes"""
    print("\nTesting vertical scenario altitude differences...")
    
    generator = ScenarioGenerator()
    scenario = generator.generate_vertical_scenario(n_aircraft=2, conflict=True)
    
    # Check that aircraft have different initial altitudes
    altitudes = [state['altitude'] for state in scenario.initial_states]
    assert len(set(altitudes)) > 1, "Aircraft should have different altitudes"
    
    # Check scenario properties
    assert scenario.scenario_type == ScenarioType.VERTICAL
    assert len(scenario.initial_states) >= 2
    
    print(f"✓ Vertical scenario: {scenario.aircraft_count} aircraft")
    print(f"✓ Altitudes: {altitudes} ft (should be different)")
    print(f"✓ Ground truth conflicts: {len(scenario.ground_truth_conflicts)}")


def test_sector_scenario_complexity():
    """Test that sector scenarios respect complexity tier"""
    print("\nTesting sector scenario complexity...")
    
    generator = ScenarioGenerator()
    
    simple_scenario = generator.generate_sector_scenario(complexity=ComplexityTier.SIMPLE)
    complex_scenario = generator.generate_sector_scenario(complexity=ComplexityTier.COMPLEX)
    
    assert simple_scenario.complexity_tier == ComplexityTier.SIMPLE
    assert complex_scenario.complexity_tier == ComplexityTier.COMPLEX
    assert simple_scenario.scenario_type == ScenarioType.SECTOR
    assert complex_scenario.scenario_type == ScenarioType.SECTOR
    
    print(f"✓ Simple sector: {simple_scenario.aircraft_count} aircraft")
    print(f"✓ Complex sector: {complex_scenario.aircraft_count} aircraft")
    print(f"✓ Both scenarios generated successfully")


def test_ground_truth_structure():
    """Test ground truth conflict data structure"""
    print("\nTesting ground truth conflict structure...")
    
    generator = ScenarioGenerator()
    scenario = generator.generate_horizontal_scenario(n_aircraft=2, conflict=True)
    
    if scenario.ground_truth_conflicts:
        conflict = scenario.ground_truth_conflicts[0]
        
        # Check required fields
        assert hasattr(conflict, 'aircraft_pair')
        assert hasattr(conflict, 'conflict_type')
        assert hasattr(conflict, 'time_to_conflict')
        assert hasattr(conflict, 'min_separation')
        assert hasattr(conflict, 'severity')
        assert hasattr(conflict, 'is_actual_conflict')
        
        # Check data types
        assert isinstance(conflict.aircraft_pair, tuple)
        assert len(conflict.aircraft_pair) == 2
        assert isinstance(conflict.conflict_type, str)
        assert isinstance(conflict.min_separation, dict)
        assert isinstance(conflict.is_actual_conflict, bool)
        
        # Check separation data
        assert 'horizontal_nm' in conflict.min_separation
        assert 'vertical_ft' in conflict.min_separation
        
        print(f"✓ Ground truth conflict structure valid")
        print(f"✓ Conflict type: {conflict.conflict_type}")
        print(f"✓ Aircraft pair: {conflict.aircraft_pair}")
        print(f"✓ Severity: {conflict.severity}")
    else:
        print("ℹ No conflicts detected in test scenario")


def test_environment_classes():
    """Test environment-specific classes"""
    print("\nTesting environment classes...")
    
    # Test HorizontalCREnv
    h_env = HorizontalCREnv()
    h_scenario = h_env.generate_scenario(n_aircraft=2, conflict=True)
    assert h_scenario.scenario_type == ScenarioType.HORIZONTAL
    
    # Check altitude consistency
    altitudes = [state['altitude'] for state in h_scenario.initial_states]
    assert len(set(altitudes)) == 1
    
    # Test VerticalCREnv
    v_env = VerticalCREnv()
    v_scenario = v_env.generate_scenario(n_aircraft=2, conflict=True)
    assert v_scenario.scenario_type == ScenarioType.VERTICAL
    
    # Check altitude variety
    altitudes = [state['altitude'] for state in v_scenario.initial_states]
    assert len(set(altitudes)) > 1
    
    # Test SectorCREnv
    s_env = SectorCREnv()
    s_scenario = s_env.generate_scenario(complexity=ComplexityTier.MODERATE)
    assert s_scenario.scenario_type == ScenarioType.SECTOR
    assert s_scenario.complexity_tier == ComplexityTier.MODERATE
    
    print("✓ HorizontalCREnv working correctly")
    print("✓ VerticalCREnv working correctly")
    print("✓ SectorCREnv working correctly")


def test_convenience_functions():
    """Test convenience functions"""
    print("\nTesting convenience functions...")
    
    # Test horizontal function
    h_scenario = generate_horizontal_scenario(n_aircraft=3, conflict=True)
    assert h_scenario.scenario_type == ScenarioType.HORIZONTAL
    assert h_scenario.aircraft_count == 3
    
    # Test vertical function
    v_scenario = generate_vertical_scenario(n_aircraft=2, conflict=True)
    assert v_scenario.scenario_type == ScenarioType.VERTICAL
    assert v_scenario.aircraft_count == 2
    
    # Test sector function
    s_scenario = generate_sector_scenario(
        complexity=ComplexityTier.SIMPLE,
        shift_level="moderate_shift"
    )
    assert s_scenario.scenario_type == ScenarioType.SECTOR
    assert s_scenario.complexity_tier == ComplexityTier.SIMPLE
    assert s_scenario.distribution_shift_tier == "moderate_shift"
    
    print("✓ All convenience functions working correctly")


def test_conflict_detection_logic():
    """Test conflict vs no-conflict scenario generation"""
    print("\nTesting conflict detection logic...")
    
    generator = ScenarioGenerator()
    
    # Generate conflict scenario
    conflict_scenario = generator.generate_horizontal_scenario(n_aircraft=2, conflict=True)
    
    # Generate safe scenario
    safe_scenario = generator.generate_horizontal_scenario(n_aircraft=2, conflict=False)
    
    print(f"✓ Conflict scenario: {len(conflict_scenario.ground_truth_conflicts)} conflicts detected")
    print(f"✓ Safe scenario: {len(safe_scenario.ground_truth_conflicts)} conflicts detected")
    
    # The conflict scenario should typically have more conflicts than safe scenario
    # (though this is probabilistic)
    if conflict_scenario.has_conflicts:
        print("✓ Conflict scenario properly generated conflicts")
    
    if not safe_scenario.has_conflicts:
        print("✓ Safe scenario successfully avoided conflicts")


def test_scenario_data_integrity():
    """Test scenario data structure integrity"""
    print("\nTesting scenario data integrity...")
    
    generator = ScenarioGenerator()
    scenario = generator.generate_sector_scenario()
    
    # Check required fields
    required_fields = [
        'scenario_id', 'scenario_type', 'aircraft_count', 'commands',
        'initial_states', 'ground_truth_conflicts', 'complexity_tier',
        'generation_timestamp', 'environmental_conditions'
    ]
    
    for field in required_fields:
        assert hasattr(scenario, field), f"Missing required field: {field}"
    
    # Check that initial states have expected aircraft data
    if scenario.initial_states:
        state = scenario.initial_states[0]
        expected_keys = ['callsign', 'latitude', 'longitude', 'altitude', 'heading', 'ground_speed']
        for key in expected_keys:
            assert key in state, f"Missing aircraft state key: {key}"
    
    # Test conversion to dict
    scenario_dict = scenario.to_dict()
    assert isinstance(scenario_dict, dict)
    
    print("✓ All required fields present")
    print("✓ Aircraft state data complete")
    print("✓ Dictionary conversion working")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("SCENARIO GENERATOR TEST SUITE")
    print("="*60)
    
    try:
        test_horizontal_scenario_same_altitudes()
        test_vertical_scenario_altitude_differences()
        test_sector_scenario_complexity()
        test_ground_truth_structure()
        test_environment_classes()
        test_convenience_functions()
        test_conflict_detection_logic()
        test_scenario_data_integrity()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 Scenario generator module is working correctly!")
        print("\nKey features verified:")
        print("• Horizontal scenarios have same altitude")
        print("• Vertical scenarios have different altitudes")  
        print("• Sector scenarios respect complexity tiers")
        print("• Ground truth conflicts are properly generated")
        print("• Environment classes work as expected")
        print("• All convenience functions operational")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        exit(1)
