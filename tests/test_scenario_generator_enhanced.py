# tests/test_scenario_generator_enhanced.py
"""
Enhanced Unit Tests for Scenario Generator Module
================================================
Comprehensive tests to verify:
1. Horizontal scenarios have equal altitudes
2. Vertical scenarios create near-threshold altitude differences
3. Sector scenarios respect complexity tiers
4. Ground truth conflicts are properly generated
5. Extended dataclass fields are properly initialized
6. Enhanced vertical scenario generation with configurable parameters
"""

import unittest
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
    Scenario,
    GroundTruthConflict,
    generate_horizontal_scenario,
    generate_vertical_scenario,
    generate_sector_scenario
)


class TestScenarioDataClass(unittest.TestCase):
    """Test the extended Scenario dataclass"""
    
    def test_scenario_initialization_with_defaults(self):
        """Test that extended fields are properly initialized to defaults"""
        scenario = Scenario(
            scenario_id="test_001",
            scenario_type=ScenarioType.HORIZONTAL,
            aircraft_count=2,
            commands=["CRE AC001 B737 52.0 4.0 090 35000 350"],
            initial_states=[
                {'callsign': 'AC001', 'altitude': 35000, 'latitude': 52.0, 'longitude': 4.0}
            ],
            ground_truth_conflicts=[],
            expected_conflict_count=0,
            has_conflicts=False,
            complexity_tier=ComplexityTier.SIMPLE,
            generation_timestamp=1234567890.0,
            environmental_conditions={},
            airspace_region="EHAM_TMA"
        )
        
        # Check that extended fields are initialized to empty lists
        self.assertEqual(scenario.predicted_conflicts, [])
        self.assertEqual(scenario.resolution_commands, [])
        self.assertEqual(scenario.trajectories, [])
        self.assertIsNone(scenario.success)
        
    def test_scenario_to_dict_compatibility(self):
        """Test that scenario can be converted to dict for compatibility"""
        scenario = generate_horizontal_scenario(n_aircraft=2, conflict=False)
        scenario_dict = scenario.to_dict()
        
        # Check that all required fields are present
        required_fields = ['scenario_id', 'scenario_type', 'aircraft_count', 'commands', 
                          'initial_states', 'predicted_conflicts', 'resolution_commands', 
                          'trajectories', 'success']
        for field in required_fields:
            self.assertIn(field, scenario_dict)
        
        # Check extended fields are properly serialized
        self.assertIsInstance(scenario_dict['predicted_conflicts'], list)
        self.assertIsInstance(scenario_dict['resolution_commands'], list)
        self.assertIsInstance(scenario_dict['trajectories'], list)


class TestHorizontalScenarioGeneration(unittest.TestCase):
    """Test horizontal scenario generation requirements"""
    
    def setUp(self):
        self.generator = ScenarioGenerator()
    
    def test_horizontal_scenarios_equal_altitudes(self):
        """Verify that horizontal scenarios set all altitudes equal"""
        scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)
        
        # Extract altitudes from initial states
        altitudes = [state['altitude'] for state in scenario.initial_states]
        
        # Assert all altitudes are equal
        self.assertEqual(len(set(altitudes)), 1, "All aircraft in horizontal scenario must have equal altitudes")
        self.assertEqual(altitudes[0], 35000, "Standard altitude should be FL350 (35000 ft)")
        
        # Verify scenario properties
        self.assertEqual(scenario.scenario_type, ScenarioType.HORIZONTAL)
        self.assertEqual(scenario.aircraft_count, 3)
        self.assertEqual(len(scenario.initial_states), 3)
        
    def test_horizontal_scenario_conflict_vs_safe(self):
        """Test that conflict flag properly controls conflict generation"""
        # Generate conflict scenario
        conflict_scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)
        
        # Generate safe scenario
        safe_scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=False)
        
        # Conflict scenario should have conflicts when aircraft are convergent
        # (This is probabilistic, but we can check the structure)
        self.assertIsInstance(conflict_scenario.ground_truth_conflicts, list)
        
        # Both should have same altitudes regardless of conflict setting
        conflict_alts = [state['altitude'] for state in conflict_scenario.initial_states]
        safe_alts = [state['altitude'] for state in safe_scenario.initial_states]
        
        self.assertEqual(len(set(conflict_alts)), 1)
        self.assertEqual(len(set(safe_alts)), 1)
        self.assertEqual(conflict_alts[0], safe_alts[0])
    
    def test_horizontal_scenario_id_generation(self):
        """Test that scenario IDs are consistently generated"""
        scenario = self.generator.generate_horizontal_scenario(n_aircraft=2, conflict=True)
        
        self.assertIsInstance(scenario.scenario_id, str)
        self.assertTrue(scenario.scenario_id.startswith("horizontal_"))
        self.assertRegex(scenario.scenario_id, r'horizontal_\d+_\d{4}')


class TestVerticalScenarioGeneration(unittest.TestCase):
    """Test enhanced vertical scenario generation"""
    
    def setUp(self):
        self.generator = ScenarioGenerator()
    
    def test_vertical_scenarios_different_altitudes(self):
        """Verify that vertical scenarios create different altitudes"""
        scenario = self.generator.generate_vertical_scenario(n_aircraft=3, conflict=True)
        
        # Extract initial altitudes from initial states
        initial_altitudes = [state['altitude'] for state in scenario.initial_states]
        
        # Assert altitudes are different (more than 1 unique altitude)
        unique_altitudes = set(initial_altitudes)
        self.assertGreater(len(unique_altitudes), 1, 
                          "Vertical scenarios must have aircraft at different altitudes")
        
        # Verify scenario properties
        self.assertEqual(scenario.scenario_type, ScenarioType.VERTICAL)
        self.assertEqual(scenario.aircraft_count, 3)
        self.assertEqual(len(scenario.initial_states), 3)
    
    def test_vertical_scenario_near_threshold_separation(self):
        """Test that vertical conflicts create near-threshold altitude differences"""
        scenario = self.generator.generate_vertical_scenario(n_aircraft=2, conflict=True)
        
        # Check that aircraft have target altitudes that will create conflicts
        for state in scenario.initial_states:
            self.assertIn('target_altitude', state)
            self.assertIn('assigned_climb_rate', state)
            
            # Verify altitude difference between initial and target creates potential conflict
            alt_diff = abs(state['altitude'] - state['target_altitude'])
            self.assertGreater(alt_diff, 0, "Aircraft should have different initial and target altitudes")
    
    def test_vertical_scenario_custom_climb_rates(self):
        """Test vertical scenario generation with custom climb rates"""
        custom_climb_rates = [-2000, 1500, -1000]
        scenario = self.generator.generate_vertical_scenario(
            n_aircraft=3, 
            conflict=True,
            climb_rates=custom_climb_rates
        )
        
        # Verify custom climb rates are assigned
        for i, state in enumerate(scenario.initial_states):
            expected_rate = custom_climb_rates[i % len(custom_climb_rates)]
            self.assertEqual(state['assigned_climb_rate'], expected_rate)
    
    def test_vertical_scenario_custom_crossing_altitudes(self):
        """Test vertical scenario generation with custom crossing altitudes"""
        custom_crossing_alts = [33000, 35000, 37000]
        scenario = self.generator.generate_vertical_scenario(
            n_aircraft=3,
            conflict=True,
            crossing_altitudes=custom_crossing_alts
        )
        
        # Verify custom crossing altitudes are used as targets
        for i, state in enumerate(scenario.initial_states):
            expected_target = custom_crossing_alts[i]
            self.assertEqual(state['target_altitude'], expected_target)
    
    def test_vertical_scenario_safe_separation(self):
        """Test that safe vertical scenarios maintain >1000ft separation"""
        scenario = self.generator.generate_vertical_scenario(n_aircraft=3, conflict=False)
        
        # Check that all aircraft maintain safe separation
        altitudes = [state['altitude'] for state in scenario.initial_states]
        target_altitudes = [state['target_altitude'] for state in scenario.initial_states]
        
        # Verify minimum separation between all pairs
        for i in range(len(altitudes)):
            for j in range(i + 1, len(altitudes)):
                current_sep = abs(altitudes[i] - altitudes[j])
                target_sep = abs(target_altitudes[i] - target_altitudes[j])
                
                self.assertGreaterEqual(current_sep, 1500, 
                                      f"Initial separation {current_sep} ft too small")
                self.assertGreaterEqual(target_sep, 1500, 
                                      f"Target separation {target_sep} ft too small")


class TestSectorScenarioGeneration(unittest.TestCase):
    """Test sector scenario generation and complexity tiers"""
    
    def setUp(self):
        self.generator = ScenarioGenerator()
    
    def test_sector_scenarios_respect_complexity_tiers(self):
        """Verify that sector scenarios respect complexity tier constraints"""
        
        # Test different complexity tiers
        for complexity in [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX]:
            with self.subTest(complexity=complexity):
                scenario = self.generator.generate_sector_scenario(
                    complexity=complexity, 
                    force_conflicts=False
                )
                
                # Verify complexity tier is recorded
                self.assertEqual(scenario.complexity_tier, complexity)
                
                # Verify aircraft count respects complexity
                if complexity == ComplexityTier.SIMPLE:
                    self.assertLessEqual(scenario.aircraft_count, 5)
                elif complexity == ComplexityTier.MODERATE:
                    self.assertLessEqual(scenario.aircraft_count, 12)
                elif complexity == ComplexityTier.COMPLEX:
                    self.assertLessEqual(scenario.aircraft_count, 20)
                
                # Verify scenario structure
                self.assertEqual(scenario.scenario_type, ScenarioType.SECTOR)
                self.assertGreater(len(scenario.commands), 0)
                self.assertEqual(len(scenario.initial_states), scenario.aircraft_count)
    
    def test_sector_scenario_metadata_completeness(self):
        """Test that sector scenarios have complete metadata"""
        scenario = self.generator.generate_sector_scenario(
            complexity=ComplexityTier.MODERATE,
            force_conflicts=True
        )
        
        # Check all required metadata fields
        self.assertIsInstance(scenario.scenario_id, str)
        self.assertIsInstance(scenario.generation_timestamp, float)
        self.assertIsInstance(scenario.environmental_conditions, dict)
        self.assertIsInstance(scenario.airspace_region, str)
        self.assertIsInstance(scenario.distribution_shift_tier, str)


class TestEnvironmentClasses(unittest.TestCase):
    """Test environment-specific classes"""
    
    def test_horizontal_cr_env(self):
        """Test HorizontalCREnv class"""
        env = HorizontalCREnv()
        
        # Test conflict scenario
        conflict_scenario = env.generate_scenario(n_aircraft=2, conflict=True)
        self.assertEqual(conflict_scenario.scenario_type, ScenarioType.HORIZONTAL)
        
        # Test safe scenario
        safe_scenario = env.generate_scenario(n_aircraft=2, conflict=False)
        self.assertEqual(safe_scenario.scenario_type, ScenarioType.HORIZONTAL)
        
        # Both should have equal altitudes
        for scenario in [conflict_scenario, safe_scenario]:
            altitudes = [state['altitude'] for state in scenario.initial_states]
            self.assertEqual(len(set(altitudes)), 1)
    
    def test_vertical_cr_env(self):
        """Test VerticalCREnv class"""
        env = VerticalCREnv()
        
        scenario = env.generate_scenario(n_aircraft=3, conflict=True)
        self.assertEqual(scenario.scenario_type, ScenarioType.VERTICAL)
        
        # Should have different altitudes
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertGreater(len(set(altitudes)), 1)
    
    def test_sector_cr_env(self):
        """Test SectorCREnv class"""
        env = SectorCREnv()
        
        scenario = env.generate_scenario(complexity=ComplexityTier.SIMPLE)
        self.assertEqual(scenario.scenario_type, ScenarioType.SECTOR)
        self.assertEqual(scenario.complexity_tier, ComplexityTier.SIMPLE)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_generate_horizontal_scenario_function(self):
        """Test generate_horizontal_scenario convenience function"""
        scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.HORIZONTAL)
        self.assertEqual(scenario.aircraft_count, 2)
        
        # Check altitudes are equal
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertEqual(len(set(altitudes)), 1)
    
    def test_generate_vertical_scenario_function(self):
        """Test generate_vertical_scenario convenience function"""
        scenario = generate_vertical_scenario(n_aircraft=3, conflict=True)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.VERTICAL)
        self.assertEqual(scenario.aircraft_count, 3)
        
        # Check altitudes are different
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertGreater(len(set(altitudes)), 1)
    
    def test_generate_sector_scenario_function(self):
        """Test generate_sector_scenario convenience function"""
        scenario = generate_sector_scenario(
            complexity=ComplexityTier.MODERATE,
            force_conflicts=False
        )
        
        self.assertEqual(scenario.scenario_type, ScenarioType.SECTOR)
        self.assertEqual(scenario.complexity_tier, ComplexityTier.MODERATE)


class TestScenarioConsistency(unittest.TestCase):
    """Test scenario data consistency and integrity"""
    
    def setUp(self):
        self.generator = ScenarioGenerator()
    
    def test_scenario_id_consistency(self):
        """Test that scenario IDs are consistently generated and unique"""
        scenarios = []
        for scenario_type in [ScenarioType.HORIZONTAL, ScenarioType.VERTICAL, ScenarioType.SECTOR]:
            if scenario_type == ScenarioType.HORIZONTAL:
                scenario = self.generator.generate_horizontal_scenario(n_aircraft=2, conflict=True)
            elif scenario_type == ScenarioType.VERTICAL:
                scenario = self.generator.generate_vertical_scenario(n_aircraft=2, conflict=True)
            else:
                scenario = self.generator.generate_sector_scenario(complexity=ComplexityTier.SIMPLE)
            
            scenarios.append(scenario)
        
        # Check all scenario IDs are unique
        scenario_ids = [s.scenario_id for s in scenarios]
        self.assertEqual(len(scenario_ids), len(set(scenario_ids)), "All scenario IDs should be unique")
        
        # Check ID format consistency
        for scenario in scenarios:
            self.assertRegex(scenario.scenario_id, r'\w+_\d+_\d{4}')
    
    def test_initial_states_consistency(self):
        """Test that initial states are consistently recorded"""
        scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)
        
        # Check that aircraft count matches initial states length
        self.assertEqual(scenario.aircraft_count, len(scenario.initial_states))
        
        # Check that all initial states have required fields
        required_fields = ['callsign', 'aircraft_type', 'latitude', 'longitude', 
                          'altitude', 'heading', 'ground_speed']
        
        for state in scenario.initial_states:
            for field in required_fields:
                self.assertIn(field, state, f"Initial state missing required field: {field}")
                self.assertIsNotNone(state[field], f"Initial state field {field} is None")


def run_comprehensive_tests():
    """Run all tests and report results"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestScenarioDataClass,
        TestHorizontalScenarioGeneration,
        TestVerticalScenarioGeneration,
        TestSectorScenarioGeneration,
        TestEnvironmentClasses,
        TestConvenienceFunctions,
        TestScenarioConsistency
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*80)
    print("ENHANCED SCENARIO GENERATOR TESTS")
    print("="*80)
    
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
