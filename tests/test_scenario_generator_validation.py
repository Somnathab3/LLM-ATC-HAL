# tests/test_scenario_generator_validation.py
"""
Validation tests for the Scenario Generator Module
==================================================
Tests to verify that:
1. Horizontal scenarios have equal altitudes
2. Vertical scenarios have different altitudes with vertical maneuvers
3. Sector scenarios respect complexity tiers
4. Ground truth conflicts are properly generated
5. Environment classes work correctly
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
    generate_horizontal_scenario,
    generate_vertical_scenario,
    generate_sector_scenario
)


class TestScenarioGenerator(unittest.TestCase):
    """Test the main ScenarioGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ScenarioGenerator()
    
    def test_horizontal_scenario_same_altitudes(self):
        """Test that horizontal scenarios have all aircraft at same altitude"""
        scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)
        
        # Check that all aircraft have same altitude
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertEqual(len(set(altitudes)), 1, "All aircraft should be at same altitude")
        self.assertEqual(altitudes[0], 35000, "Standard altitude should be FL350")
        
        # Check scenario type
        self.assertEqual(scenario.scenario_type, ScenarioType.HORIZONTAL)
        
        # Check aircraft count
        self.assertEqual(scenario.aircraft_count, 3)
        
        # Should have conflicts for conflict=True
        self.assertTrue(scenario.has_conflicts)
        self.assertGreater(len(scenario.ground_truth_conflicts), 0)
    
    def test_horizontal_scenario_no_conflicts(self):
        """Test that horizontal safe scenarios avoid conflicts"""
        scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=False)
        
        # Check that all aircraft have same altitude
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertEqual(len(set(altitudes)), 1, "All aircraft should be at same altitude")
        
        # Should have no conflicts for conflict=False
        self.assertFalse(scenario.has_conflicts)
        self.assertEqual(len(scenario.ground_truth_conflicts), 0)
    
    def test_vertical_scenario_different_altitudes(self):
        """Test that vertical scenarios have aircraft at different altitudes"""
        scenario = self.generator.generate_vertical_scenario(n_aircraft=2, conflict=True)
        
        # Check that aircraft have different altitudes
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertGreater(len(set(altitudes)), 1, "Aircraft should be at different altitudes")
        
        # Check scenario type
        self.assertEqual(scenario.scenario_type, ScenarioType.VERTICAL)
        
        # Check aircraft count
        self.assertEqual(scenario.aircraft_count, 2)
        
        # Should have conflicts for conflict=True
        self.assertTrue(scenario.has_conflicts)
        self.assertGreater(len(scenario.ground_truth_conflicts), 0)
        
        # Check for vertical maneuver commands
        alt_vs_commands = [cmd for cmd in scenario.commands if 'ALT' in cmd or 'VS' in cmd]
        self.assertGreater(len(alt_vs_commands), 0, "Should have altitude or vertical speed commands")
    
    def test_vertical_scenario_safe(self):
        """Test that vertical safe scenarios maintain safe separation"""
        scenario = self.generator.generate_vertical_scenario(n_aircraft=2, conflict=False)
        
        # Check that aircraft have different altitudes with safe separation
        altitudes = [state['altitude'] for state in scenario.initial_states]
        if len(altitudes) >= 2:
            alt_diff = abs(altitudes[0] - altitudes[1])
            self.assertGreaterEqual(alt_diff, 1000, "Safe vertical scenarios should have >1000ft separation")
        
        # Should have no conflicts for conflict=False
        self.assertFalse(scenario.has_conflicts)
        self.assertEqual(len(scenario.ground_truth_conflicts), 0)
    
    def test_sector_scenario_complexity_tiers(self):
        """Test that sector scenarios respect complexity tiers"""
        # Test different complexity tiers
        complexities = [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX]
        expected_ranges = {
            ComplexityTier.SIMPLE: (2, 3),
            ComplexityTier.MODERATE: (4, 6), 
            ComplexityTier.COMPLEX: (8, 15)  # Allow some flexibility
        }
        
        for complexity in complexities:
            with self.subTest(complexity=complexity):
                scenario = self.generator.generate_sector_scenario(
                    complexity=complexity, 
                    force_conflicts=False
                )
                
                # Check scenario type
                self.assertEqual(scenario.scenario_type, ScenarioType.SECTOR)
                self.assertEqual(scenario.complexity_tier, complexity)
                
                # Check aircraft count is in expected range
                min_count, max_count = expected_ranges[complexity]
                self.assertGreaterEqual(scenario.aircraft_count, min_count)
                self.assertLessEqual(scenario.aircraft_count, max_count)
    
    def test_ground_truth_conflicts_structure(self):
        """Test that ground truth conflicts have proper structure"""
        scenario = self.generator.generate_horizontal_scenario(n_aircraft=3, conflict=True)
        
        for conflict in scenario.ground_truth_conflicts:
            # Check required fields
            self.assertIsInstance(conflict.aircraft_pair, tuple)
            self.assertEqual(len(conflict.aircraft_pair), 2)
            self.assertIsInstance(conflict.conflict_type, str)
            self.assertIsInstance(conflict.time_to_conflict, (int, float))
            self.assertIsInstance(conflict.min_separation, dict)
            self.assertIsInstance(conflict.severity, str)
            self.assertIsInstance(conflict.is_actual_conflict, bool)
            
            # Check min_separation structure
            self.assertIn('horizontal_nm', conflict.min_separation)
            self.assertIn('vertical_ft', conflict.min_separation)
            
            # Check severity values
            self.assertIn(conflict.severity, ['low', 'medium', 'high', 'critical'])
    
    def test_scenario_dispatcher(self):
        """Test the scenario type dispatcher"""
        # Test horizontal dispatch
        h_scenario = self.generator.generate_scenario(ScenarioType.HORIZONTAL, n_aircraft=2, conflict=True)
        self.assertEqual(h_scenario.scenario_type, ScenarioType.HORIZONTAL)
        
        # Test vertical dispatch
        v_scenario = self.generator.generate_scenario(ScenarioType.VERTICAL, n_aircraft=2, conflict=True)
        self.assertEqual(v_scenario.scenario_type, ScenarioType.VERTICAL)
        
        # Test sector dispatch
        s_scenario = self.generator.generate_scenario(ScenarioType.SECTOR, complexity=ComplexityTier.SIMPLE)
        self.assertEqual(s_scenario.scenario_type, ScenarioType.SECTOR)


class TestEnvironmentClasses(unittest.TestCase):
    """Test the environment-specific classes"""
    
    def test_horizontal_cr_env(self):
        """Test HorizontalCREnv class"""
        env = HorizontalCREnv()
        scenario = env.generate_scenario(n_aircraft=2, conflict=True)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.HORIZONTAL)
        self.assertEqual(scenario.aircraft_count, 2)
        
        # Check same altitudes
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertEqual(len(set(altitudes)), 1)
    
    def test_vertical_cr_env(self):
        """Test VerticalCREnv class"""
        env = VerticalCREnv()
        scenario = env.generate_scenario(n_aircraft=2, conflict=True)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.VERTICAL)
        self.assertEqual(scenario.aircraft_count, 2)
        
        # Check different altitudes
        altitudes = [state['altitude'] for state in scenario.initial_states]
        self.assertGreater(len(set(altitudes)), 1)
    
    def test_sector_cr_env(self):
        """Test SectorCREnv class"""
        env = SectorCREnv()
        scenario = env.generate_scenario(complexity=ComplexityTier.MODERATE)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.SECTOR)
        self.assertEqual(scenario.complexity_tier, ComplexityTier.MODERATE)
        self.assertGreaterEqual(scenario.aircraft_count, 4)
        self.assertLessEqual(scenario.aircraft_count, 6)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_generate_horizontal_scenario(self):
        """Test generate_horizontal_scenario convenience function"""
        scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.HORIZONTAL)
        self.assertEqual(scenario.aircraft_count, 2)
        self.assertTrue(scenario.has_conflicts)
    
    def test_generate_vertical_scenario(self):
        """Test generate_vertical_scenario convenience function"""
        scenario = generate_vertical_scenario(n_aircraft=2, conflict=True)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.VERTICAL)
        self.assertEqual(scenario.aircraft_count, 2)
        self.assertTrue(scenario.has_conflicts)
    
    def test_generate_sector_scenario(self):
        """Test generate_sector_scenario convenience function"""
        scenario = generate_sector_scenario(complexity=ComplexityTier.SIMPLE, force_conflicts=False)
        
        self.assertEqual(scenario.scenario_type, ScenarioType.SECTOR)
        self.assertEqual(scenario.complexity_tier, ComplexityTier.SIMPLE)
        self.assertGreaterEqual(scenario.aircraft_count, 2)
        self.assertLessEqual(scenario.aircraft_count, 3)


class TestScenarioDataStructure(unittest.TestCase):
    """Test scenario data structure compatibility"""
    
    def test_scenario_to_dict(self):
        """Test scenario conversion to dictionary"""
        scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)
        scenario_dict = scenario.to_dict()
        
        # Check all required fields present
        required_fields = [
            'scenario_id', 'scenario_type', 'aircraft_count', 'commands',
            'initial_states', 'ground_truth_conflicts', 'expected_conflict_count',
            'has_conflicts', 'complexity_tier', 'generation_timestamp',
            'environmental_conditions', 'airspace_region'
        ]
        
        for field in required_fields:
            self.assertIn(field, scenario_dict)
    
    def test_aircraft_state_format(self):
        """Test aircraft state format matches expected structure"""
        scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)
        
        required_state_fields = [
            'callsign', 'aircraft_type', 'latitude', 'longitude',
            'altitude', 'heading', 'ground_speed', 'vertical_rate'
        ]
        
        for state in scenario.initial_states:
            for field in required_state_fields:
                self.assertIn(field, state)
    
    def test_bluesky_commands_format(self):
        """Test BlueSky commands are properly formatted"""
        scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)
        
        # Should have CRE commands for each aircraft
        cre_commands = [cmd for cmd in scenario.commands if cmd.startswith('CRE')]
        self.assertGreaterEqual(len(cre_commands), scenario.aircraft_count)
        
        # Check CRE command format
        for cmd in cre_commands:
            parts = cmd.split(',')
            self.assertGreaterEqual(len(parts), 7)  # CRE callsign,type,lat,lon,hdg,alt,speed


def run_validation_tests():
    """Run all validation tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestScenarioGenerator,
        TestEnvironmentClasses,
        TestConvenienceFunctions,
        TestScenarioDataStructure
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 SCENARIO GENERATOR VALIDATION TESTS")
    print("="*50)
    
    success = run_validation_tests()
    
    if success:
        print("\n✅ All validation tests passed!")
        print("The scenario generator module is working correctly.")
    else:
        print("\n❌ Some validation tests failed!")
        print("Check the output above for details.")
    
    exit(0 if success else 1)
