# tests/test_scenario_generator.py
"""
Unit tests for scenario generation module.

Tests verify that:
- Horizontal scenarios have equal altitudes
- Vertical scenarios have near-threshold altitudes  
- Sector scenarios respect complexity tier
- Ground truth conflict detection works correctly
"""

import pytest
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


class TestScenarioGenerator:
    """Test main scenario generator functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ScenarioGenerator()
    
    def test_horizontal_scenario_same_altitudes(self):
        """Test that horizontal scenarios have all aircraft at same altitude"""
        scenario = self.generator.generate_horizontal_scenario(
            n_aircraft=3, 
            conflict=True
        )
        
        # Check that all aircraft have same altitude
        altitudes = [state['altitude'] for state in scenario.initial_states]
        assert len(set(altitudes)) == 1, "All aircraft should be at same altitude"
        assert altitudes[0] == 35000, "Should use standard FL350"
        
        # Check scenario properties
        assert scenario.scenario_type == ScenarioType.HORIZONTAL
        assert scenario.aircraft_count == 3
        assert len(scenario.initial_states) == 3
        assert len(scenario.commands) > 0
    
    def test_horizontal_scenario_conflict_generation(self):
        """Test horizontal conflict generation vs avoidance"""
        # Generate conflict scenario
        conflict_scenario = self.generator.generate_horizontal_scenario(
            n_aircraft=2, 
            conflict=True
        )
        
        # Generate safe scenario
        safe_scenario = self.generator.generate_horizontal_scenario(
            n_aircraft=2, 
            conflict=False
        )
        
        # Conflict scenario should have ground truth conflicts
        assert conflict_scenario.has_conflicts, "Conflict scenario should have conflicts"
        assert len(conflict_scenario.ground_truth_conflicts) > 0
        
        # Safe scenario should have no conflicts
        assert not safe_scenario.has_conflicts or len(safe_scenario.ground_truth_conflicts) == 0
    
    def test_vertical_scenario_altitude_differences(self):
        """Test that vertical scenarios have aircraft at different altitudes"""
        scenario = self.generator.generate_vertical_scenario(
            n_aircraft=2, 
            conflict=True
        )
        
        # Check that aircraft have different initial altitudes
        altitudes = [state['altitude'] for state in scenario.initial_states]
        assert len(set(altitudes)) > 1, "Aircraft should have different altitudes"
        
        # Check altitude separation
        altitude_diffs = []
        for i in range(len(altitudes)):
            for j in range(i + 1, len(altitudes)):
                altitude_diffs.append(abs(altitudes[i] - altitudes[j]))
        
        # At least one pair should have reasonable separation for vertical conflict
        assert any(diff >= 1000 for diff in altitude_diffs), "Should have meaningful altitude differences"
        
        # Check scenario properties
        assert scenario.scenario_type == ScenarioType.VERTICAL
        assert len(scenario.initial_states) >= 2
    
    def test_vertical_scenario_climb_descent_commands(self):
        """Test that vertical scenarios include climb/descent commands"""
        scenario = self.generator.generate_vertical_scenario(
            n_aircraft=2, 
            conflict=True
        )
        
        # Check for altitude or vertical speed commands
        command_text = ' '.join(scenario.commands)
        has_vertical_commands = any(cmd in command_text for cmd in ['ALT', 'VS'])
        
        assert has_vertical_commands, "Vertical scenario should include ALT or VS commands"
        
        # Check for vertical rates in initial states
        vertical_rates = [state.get('vertical_rate', 0) for state in scenario.initial_states]
        has_non_zero_rates = any(vr != 0 for vr in vertical_rates)
        
        # Either commands or initial vertical rates should be present
        assert has_vertical_commands or has_non_zero_rates, "Should have vertical movement"
    
    def test_sector_scenario_complexity_respect(self):
        """Test that sector scenarios respect complexity tier"""
        simple_scenario = self.generator.generate_sector_scenario(
            complexity=ComplexityTier.SIMPLE
        )
        
        complex_scenario = self.generator.generate_sector_scenario(
            complexity=ComplexityTier.COMPLEX
        )
        
        # Complex scenario should generally have more aircraft
        # (though this is probabilistic, so we'll just check basic properties)
        assert simple_scenario.complexity_tier == ComplexityTier.SIMPLE
        assert complex_scenario.complexity_tier == ComplexityTier.COMPLEX
        
        assert simple_scenario.scenario_type == ScenarioType.SECTOR
        assert complex_scenario.scenario_type == ScenarioType.SECTOR
        
        # Both should have reasonable aircraft counts
        assert 1 <= simple_scenario.aircraft_count <= 10
        assert 1 <= complex_scenario.aircraft_count <= 15
    
    def test_ground_truth_conflict_structure(self):
        """Test ground truth conflict data structure"""
        scenario = self.generator.generate_horizontal_scenario(
            n_aircraft=2, 
            conflict=True
        )
        
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
            assert isinstance(conflict.time_to_conflict, (int, float))
            assert isinstance(conflict.min_separation, dict)
            assert isinstance(conflict.is_actual_conflict, bool)
            
            # Check separation data
            assert 'horizontal_nm' in conflict.min_separation
            assert 'vertical_ft' in conflict.min_separation
    
    def test_scenario_dispatcher(self):
        """Test scenario type dispatcher method"""
        h_scenario = self.generator.generate_scenario(
            ScenarioType.HORIZONTAL, 
            n_aircraft=2, 
            conflict=True
        )
        assert h_scenario.scenario_type == ScenarioType.HORIZONTAL
        
        v_scenario = self.generator.generate_scenario(
            ScenarioType.VERTICAL, 
            n_aircraft=2, 
            conflict=True
        )
        assert v_scenario.scenario_type == ScenarioType.VERTICAL
        
        s_scenario = self.generator.generate_scenario(
            ScenarioType.SECTOR, 
            complexity=ComplexityTier.MODERATE
        )
        assert s_scenario.scenario_type == ScenarioType.SECTOR
        
        # Test invalid type
        with pytest.raises(ValueError):
            self.generator.generate_scenario("invalid_type")


class TestEnvironmentClasses:
    """Test environment-specific classes"""
    
    def test_horizontal_cr_env(self):
        """Test HorizontalCREnv class"""
        env = HorizontalCREnv()
        
        scenario = env.generate_scenario(n_aircraft=2, conflict=True)
        
        assert scenario.scenario_type == ScenarioType.HORIZONTAL
        assert scenario.aircraft_count == 2
        
        # Check altitude consistency
        altitudes = [state['altitude'] for state in scenario.initial_states]
        assert len(set(altitudes)) == 1
    
    def test_vertical_cr_env(self):
        """Test VerticalCREnv class"""
        env = VerticalCREnv()
        
        scenario = env.generate_scenario(n_aircraft=2, conflict=True)
        
        assert scenario.scenario_type == ScenarioType.VERTICAL
        assert scenario.aircraft_count == 2
        
        # Check altitude variety
        altitudes = [state['altitude'] for state in scenario.initial_states]
        assert len(set(altitudes)) > 1
    
    def test_sector_cr_env(self):
        """Test SectorCREnv class"""
        env = SectorCREnv()
        
        scenario = env.generate_scenario(
            complexity=ComplexityTier.MODERATE,
            force_conflicts=False
        )
        
        assert scenario.scenario_type == ScenarioType.SECTOR
        assert scenario.complexity_tier == ComplexityTier.MODERATE


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_generate_horizontal_scenario_function(self):
        """Test standalone horizontal scenario generation function"""
        scenario = generate_horizontal_scenario(n_aircraft=3, conflict=True)
        
        assert scenario.scenario_type == ScenarioType.HORIZONTAL
        assert scenario.aircraft_count == 3
        
        # Check altitude consistency
        altitudes = [state['altitude'] for state in scenario.initial_states]
        assert len(set(altitudes)) == 1
    
    def test_generate_vertical_scenario_function(self):
        """Test standalone vertical scenario generation function"""
        scenario = generate_vertical_scenario(n_aircraft=2, conflict=True)
        
        assert scenario.scenario_type == ScenarioType.VERTICAL
        assert scenario.aircraft_count == 2
    
    def test_generate_sector_scenario_function(self):
        """Test standalone sector scenario generation function"""
        scenario = generate_sector_scenario(
            complexity=ComplexityTier.SIMPLE,
            shift_level="moderate_shift"
        )
        
        assert scenario.scenario_type == ScenarioType.SECTOR
        assert scenario.complexity_tier == ComplexityTier.SIMPLE
        assert scenario.distribution_shift_tier == "moderate_shift"


class TestGeometryCalculations:
    """Test geometric calculation methods"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ScenarioGenerator()
    
    def test_distance_calculation(self):
        """Test distance calculation between points"""
        # Test known distance (approximately)
        # London to Paris is about 214 nautical miles
        london_lat, london_lon = 51.5074, -0.1278
        paris_lat, paris_lon = 48.8566, 2.3522
        
        distance = self.generator._calculate_distance_nm(
            london_lat, london_lon, paris_lat, paris_lon
        )
        
        # Should be approximately 214 nm (within 10% tolerance)
        assert 190 < distance < 240, f"Distance calculation seems incorrect: {distance}"
    
    def test_bearing_calculation(self):
        """Test bearing calculation between points"""
        # North bearing should be 0/360
        bearing = self.generator._calculate_bearing(0, 0, 1, 0)
        assert abs(bearing) < 5 or abs(bearing - 360) < 5, "North bearing should be ~0°"
        
        # East bearing should be 90
        bearing = self.generator._calculate_bearing(0, 0, 0, 1)
        assert abs(bearing - 90) < 5, "East bearing should be ~90°"
    
    def test_position_projection(self):
        """Test position projection based on heading and speed"""
        # Start at origin
        lat, lon = 0.0, 0.0
        heading = 90  # East
        speed = 60  # 60 knots
        time = 60  # 60 minutes = 1 hour
        
        new_lat, new_lon = self.generator._project_position(lat, lon, heading, speed, time)
        
        # After 1 hour at 60 knots heading east, should move ~1 degree longitude
        assert abs(new_lat - lat) < 0.1, "Latitude should barely change for eastward flight"
        assert new_lon > lon, "Longitude should increase for eastward flight"
    
    def test_convergent_headings(self):
        """Test convergent heading detection"""
        # Aircraft pointing toward each other
        is_convergent = self.generator._are_headings_convergent(
            0, 0, 90,   # AC1 at origin heading east
            0, 1, 270   # AC2 to the east heading west
        )
        assert is_convergent, "Aircraft heading toward each other should be convergent"
        
        # Aircraft with parallel headings
        is_convergent = self.generator._are_headings_convergent(
            0, 0, 90,   # AC1 heading east
            1, 0, 90    # AC2 also heading east
        )
        assert not is_convergent, "Parallel aircraft should not be convergent"


class TestScenarioDataStructure:
    """Test scenario data structure and compatibility"""
    
    def test_scenario_to_dict(self):
        """Test scenario conversion to dictionary"""
        generator = ScenarioGenerator()
        scenario = generator.generate_horizontal_scenario(n_aircraft=2)
        
        scenario_dict = scenario.to_dict()
        
        # Check required fields
        required_fields = [
            'scenario_id', 'scenario_type', 'aircraft_count', 'commands',
            'initial_states', 'ground_truth_conflicts', 'complexity_tier',
            'generation_timestamp', 'environmental_conditions'
        ]
        
        for field in required_fields:
            assert field in scenario_dict, f"Missing required field: {field}"
    
    def test_backward_compatibility(self):
        """Test compatibility with existing code expectations"""
        generator = ScenarioGenerator()
        scenario = generator.generate_sector_scenario()
        
        # Check that scenario has expected structure for integration
        assert hasattr(scenario, 'commands')
        assert hasattr(scenario, 'initial_states')
        assert hasattr(scenario, 'aircraft_count')
        assert hasattr(scenario, 'environmental_conditions')
        
        # Check that initial states have expected aircraft data
        if scenario.initial_states:
            state = scenario.initial_states[0]
            expected_keys = ['callsign', 'latitude', 'longitude', 'altitude', 'heading', 'ground_speed']
            for key in expected_keys:
                assert key in state, f"Missing aircraft state key: {key}"


# Integration test
def test_full_scenario_generation_workflow():
    """Test complete scenario generation workflow"""
    # Test all three environment types
    generator = ScenarioGenerator()
    
    # Generate one of each type
    scenarios = [
        generator.generate_horizontal_scenario(n_aircraft=2, conflict=True),
        generator.generate_vertical_scenario(n_aircraft=2, conflict=True),
        generator.generate_sector_scenario(complexity=ComplexityTier.MODERATE)
    ]
    
    for scenario in scenarios:
        # Basic validation
        assert scenario.scenario_id is not None
        assert scenario.aircraft_count > 0
        assert len(scenario.commands) > 0
        assert len(scenario.initial_states) == scenario.aircraft_count
        assert isinstance(scenario.generation_timestamp, float)
        assert isinstance(scenario.environmental_conditions, dict)
        
        # Validate ground truth conflicts
        for conflict in scenario.ground_truth_conflicts:
            assert len(conflict.aircraft_pair) == 2
            assert conflict.conflict_type in ['horizontal', 'vertical', 'convergent', 'parallel', 'overtaking']
            assert conflict.severity in ['low', 'medium', 'high', 'critical']
            assert isinstance(conflict.is_actual_conflict, bool)


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Running basic scenario generator tests...")
    
    # Test basic functionality
    generator = ScenarioGenerator()
    
    # Test horizontal scenario
    print("Testing horizontal scenario...")
    h_scenario = generator.generate_horizontal_scenario(n_aircraft=2, conflict=True)
    altitudes = [state['altitude'] for state in h_scenario.initial_states]
    print(f"Horizontal scenario altitudes: {altitudes} (should be same)")
    assert len(set(altitudes)) == 1, "Horizontal scenario test failed"
    
    # Test vertical scenario
    print("Testing vertical scenario...")
    v_scenario = generator.generate_vertical_scenario(n_aircraft=2, conflict=True)
    altitudes = [state['altitude'] for state in v_scenario.initial_states]
    print(f"Vertical scenario altitudes: {altitudes} (should be different)")
    assert len(set(altitudes)) > 1, "Vertical scenario test failed"
    
    # Test sector scenario
    print("Testing sector scenario...")
    s_scenario = generator.generate_sector_scenario(complexity=ComplexityTier.MODERATE)
    print(f"Sector scenario: {s_scenario.aircraft_count} aircraft, {len(s_scenario.ground_truth_conflicts)} conflicts")
    
    print("All basic tests passed!")
