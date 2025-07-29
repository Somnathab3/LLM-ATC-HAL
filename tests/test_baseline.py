# tests/test_baseline.py
"""
Test suite for baseline models
"""

import sys
import os
import unittest
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_atc.baseline_models.conflict_detector import BaselineConflictDetector, ConflictPrediction
from llm_atc.baseline_models.conflict_resolver import BaselineConflictResolver, ManeuverType, ResolutionManeuver
from llm_atc.baseline_models.evaluate import BaselineEvaluator


class TestBaselineConflictDetector(unittest.TestCase):
    """Test cases for BaselineConflictDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = BaselineConflictDetector("random_forest")
        self.sample_scenario = {
            'aircraft': [
                {
                    'id': 'AC001',
                    'lat': 52.0,
                    'lon': 4.0,
                    'alt': 35000,
                    'speed': 250,
                    'heading': 90,
                    'vertical_speed': 0,
                    'type': 'commercial',
                    'flight_phase': 'cruise'
                },
                {
                    'id': 'AC002',
                    'lat': 52.01,
                    'lon': 4.01,
                    'alt': 35000,
                    'speed': 260,
                    'heading': 270,
                    'vertical_speed': 0,
                    'type': 'commercial',
                    'flight_phase': 'cruise'
                }
            ],
            'traffic_density': 0.5,
            'weather_severity': 0.2,
            'time_horizon': 600
        }
    
    def test_feature_extraction(self):
        """Test feature extraction from scenario"""
        features = self.detector.extract_features(self.sample_scenario)
        
        # Check feature vector properties
        self.assertEqual(len(features), 20)  # Expected feature count
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        
        # Check specific feature values
        self.assertEqual(features[0], 2)  # Number of aircraft
        self.assertEqual(features[1], 600)  # Time horizon
        self.assertEqual(features[2], 0.5)  # Traffic density
        self.assertEqual(features[3], 0.2)  # Weather severity
    
    def test_feature_extraction_insufficient_aircraft(self):
        """Test feature extraction with insufficient aircraft"""
        scenario = {
            'aircraft': [
                {
                    'id': 'AC001',
                    'lat': 52.0,
                    'lon': 4.0,
                    'alt': 35000,
                    'speed': 250,
                    'heading': 90
                }
            ]
        }
        
        features = self.detector.extract_features(scenario)
        
        # Should return zero vector for insufficient aircraft
        self.assertEqual(len(features), 20)
        self.assertTrue(np.all(features == 0))
    
    def test_aircraft_type_encoding(self):
        """Test aircraft type encoding"""
        self.assertEqual(self.detector._encode_aircraft_type('commercial'), 1.0)
        self.assertEqual(self.detector._encode_aircraft_type('cargo'), 2.0)
        self.assertEqual(self.detector._encode_aircraft_type('private'), 3.0)
        self.assertEqual(self.detector._encode_aircraft_type('military'), 4.0)
        self.assertEqual(self.detector._encode_aircraft_type('unknown'), 0.0)
        self.assertEqual(self.detector._encode_aircraft_type('invalid'), 0.0)
    
    def test_flight_phase_encoding(self):
        """Test flight phase encoding"""
        self.assertEqual(self.detector._encode_flight_phase('takeoff'), 1.0)
        self.assertEqual(self.detector._encode_flight_phase('climb'), 2.0)
        self.assertEqual(self.detector._encode_flight_phase('cruise'), 3.0)
        self.assertEqual(self.detector._encode_flight_phase('descent'), 4.0)
        self.assertEqual(self.detector._encode_flight_phase('approach'), 5.0)
        self.assertEqual(self.detector._encode_flight_phase('landing'), 6.0)
        self.assertEqual(self.detector._encode_flight_phase('unknown'), 3.0)  # Default to cruise
    
    def test_training_and_prediction(self):
        """Test model training and prediction"""
        # Generate synthetic training data
        training_scenarios = []
        labels = []
        
        for i in range(50):
            scenario = {
                'aircraft': [
                    {
                        'id': f'AC{i*2}',
                        'lat': 52.0 + np.random.normal(0, 0.1),
                        'lon': 4.0 + np.random.normal(0, 0.1),
                        'alt': 35000 + np.random.normal(0, 2000),
                        'speed': 250 + np.random.normal(0, 50),
                        'heading': np.random.uniform(0, 360),
                        'vertical_speed': np.random.normal(0, 500),
                        'type': 'commercial',
                        'flight_phase': 'cruise'
                    },
                    {
                        'id': f'AC{i*2+1}',
                        'lat': 52.0 + np.random.normal(0, 0.1),
                        'lon': 4.0 + np.random.normal(0, 0.1),
                        'alt': 35000 + np.random.normal(0, 2000),
                        'speed': 250 + np.random.normal(0, 50),
                        'heading': np.random.uniform(0, 360),
                        'vertical_speed': np.random.normal(0, 500),
                        'type': 'commercial',
                        'flight_phase': 'cruise'
                    }
                ],
                'traffic_density': np.random.uniform(0, 1),
                'weather_severity': np.random.uniform(0, 0.5)
            }
            training_scenarios.append(scenario)
            labels.append(np.random.random() > 0.6)  # Random labels
        
        # Train model
        try:
            metrics = self.detector.train(training_scenarios, labels)
            
            # Check training metrics
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1_score', metrics)
            self.assertIn('roc_auc', metrics)
            
            # Test prediction
            prediction = self.detector.predict(self.sample_scenario)
            
            # Check prediction structure
            self.assertIsInstance(prediction, ConflictPrediction)
            self.assertIsInstance(prediction.has_conflict, bool)
            self.assertIsInstance(prediction.confidence, float)
            self.assertIsInstance(prediction.time_to_conflict, float)
            self.assertIsInstance(prediction.conflict_pairs, list)
            self.assertIsInstance(prediction.risk_factors, dict)
            
            # Check confidence range
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)
            
        except ImportError:
            # Skip test if scikit-learn is not available
            self.skipTest("scikit-learn not available")
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pkl")
                
                # Train a simple model
                training_data = [self.sample_scenario] * 10
                labels = [True] * 5 + [False] * 5
                
                self.detector.train(training_data, labels)
                
                # Save model
                self.detector.save_model(model_path)
                self.assertTrue(os.path.exists(model_path))
                
                # Create new detector and load model
                new_detector = BaselineConflictDetector("random_forest")
                new_detector.load_model(model_path)
                
                # Test that loaded model works
                prediction = new_detector.predict(self.sample_scenario)
                self.assertIsInstance(prediction, ConflictPrediction)
                
        except ImportError:
            self.skipTest("scikit-learn not available")


class TestBaselineConflictResolver(unittest.TestCase):
    """Test cases for BaselineConflictResolver"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resolver = BaselineConflictResolver()
        self.sample_conflict_scenario = {
            'aircraft': [
                {
                    'id': 'AC001',
                    'lat': 52.0,
                    'lon': 4.0,
                    'alt': 35000,
                    'speed': 250,
                    'heading': 90,
                    'type': 'commercial',
                    'flight_phase': 'cruise'
                },
                {
                    'id': 'AC002',
                    'lat': 52.05,
                    'lon': 4.05,
                    'alt': 35000,
                    'speed': 260,
                    'heading': 270,
                    'type': 'commercial',
                    'flight_phase': 'cruise'
                }
            ],
            'conflicts': [
                {
                    'aircraft1_id': 'AC001',
                    'aircraft2_id': 'AC002',
                    'time_to_conflict': 300,
                    'closest_approach_distance': 3.5
                }
            ],
            'environmental_conditions': {
                'weather_severity': 0.2,
                'traffic_density': 0.4
            }
        }
    
    def test_conflict_resolution(self):
        """Test conflict resolution generation"""
        maneuvers = self.resolver.resolve_conflicts(self.sample_conflict_scenario)
        
        # Should generate at least one maneuver
        self.assertGreater(len(maneuvers), 0)
        
        # Check maneuver structure
        for maneuver in maneuvers:
            self.assertIsInstance(maneuver, ResolutionManeuver)
            self.assertIn(maneuver.aircraft_id, ['AC001', 'AC002'])
            self.assertIsInstance(maneuver.maneuver_type, ManeuverType)
            self.assertIsInstance(maneuver.parameters, dict)
            self.assertIsInstance(maneuver.safety_score, float)
            self.assertIsInstance(maneuver.priority, int)
            
            # Check safety score range
            self.assertGreaterEqual(maneuver.safety_score, 0.0)
            self.assertLessEqual(maneuver.safety_score, 1.0)
            
            # Check priority range
            self.assertGreaterEqual(maneuver.priority, 1)
            self.assertLessEqual(maneuver.priority, 5)
    
    def test_no_conflicts(self):
        """Test behavior with no conflicts"""
        no_conflict_scenario = {
            'aircraft': self.sample_conflict_scenario['aircraft'],
            'conflicts': [],
            'environmental_conditions': {}
        }
        
        maneuvers = self.resolver.resolve_conflicts(no_conflict_scenario)
        self.assertEqual(len(maneuvers), 0)
    
    def test_conflict_geometry_analysis(self):
        """Test conflict geometry analysis"""
        ac1 = self.sample_conflict_scenario['aircraft'][0]
        ac2 = self.sample_conflict_scenario['aircraft'][1]
        conflict = self.sample_conflict_scenario['conflicts'][0]
        
        geometry = self.resolver._analyze_conflict_geometry(ac1, ac2, conflict)
        
        # Check geometry structure
        self.assertIn('bearing', geometry)
        self.assertIn('distance', geometry)
        self.assertIn('altitude_difference', geometry)
        self.assertIn('relative_speed', geometry)
        self.assertIn('time_to_closest_approach', geometry)
        
        # Check value ranges
        self.assertIsInstance(geometry['bearing'], float)
        self.assertGreaterEqual(geometry['distance'], 0.0)
        self.assertGreaterEqual(geometry['altitude_difference'], 0.0)
        self.assertGreaterEqual(geometry['relative_speed'], 0.0)
    
    def test_aircraft_priority(self):
        """Test aircraft priority calculation"""
        # Test emergency aircraft
        emergency_ac = {'type': 'emergency', 'flight_phase': 'cruise'}
        priority = self.resolver._get_aircraft_priority(emergency_ac)
        self.assertEqual(priority, 1)
        
        # Test commercial aircraft
        commercial_ac = {'type': 'commercial', 'flight_phase': 'cruise'}
        priority = self.resolver._get_aircraft_priority(commercial_ac)
        self.assertEqual(priority, 2)
        
        # Test with critical flight phase
        approach_ac = {'type': 'commercial', 'flight_phase': 'approach'}
        priority = self.resolver._get_aircraft_priority(approach_ac)
        self.assertLessEqual(priority, 2)  # Should have higher priority
    
    def test_speed_limits(self):
        """Test speed limit calculation"""
        # Test cruise phase
        cruise_ac = {'type': 'commercial', 'flight_phase': 'cruise'}
        min_speed, max_speed = self.resolver._get_speed_limits(cruise_ac)
        self.assertGreaterEqual(min_speed, 0)
        self.assertGreater(max_speed, min_speed)
        
        # Test approach phase
        approach_ac = {'type': 'commercial', 'flight_phase': 'approach'}
        min_speed, max_speed = self.resolver._get_speed_limits(approach_ac)
        self.assertGreaterEqual(min_speed, 0)
        self.assertGreater(max_speed, min_speed)
    
    def test_maneuver_deduplication(self):
        """Test maneuver deduplication"""
        # Create duplicate maneuvers for same aircraft
        maneuvers = [
            ResolutionManeuver(
                aircraft_id='AC001',
                maneuver_type=ManeuverType.ALTITUDE_CHANGE,
                parameters={'altitude_change': 1000},
                priority=1,
                safety_score=0.8,
                estimated_delay=60,
                fuel_penalty=10
            ),
            ResolutionManeuver(
                aircraft_id='AC001',
                maneuver_type=ManeuverType.HEADING_CHANGE,
                parameters={'heading_change': 20},
                priority=2,
                safety_score=0.7,
                estimated_delay=30,
                fuel_penalty=5
            ),
            ResolutionManeuver(
                aircraft_id='AC002',
                maneuver_type=ManeuverType.SPEED_CHANGE,
                parameters={'speed_change': -20},
                priority=3,
                safety_score=0.6,
                estimated_delay=15,
                fuel_penalty=3
            )
        ]
        
        deduplicated = self.resolver._deduplicate_maneuvers(maneuvers)
        
        # Should keep only one maneuver per aircraft
        aircraft_ids = [m.aircraft_id for m in deduplicated]
        self.assertEqual(len(set(aircraft_ids)), len(deduplicated))
        self.assertIn('AC001', aircraft_ids)
        self.assertIn('AC002', aircraft_ids)


class TestBaselineEvaluator(unittest.TestCase):
    """Test cases for BaselineEvaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = BaselineEvaluator(model_dir=self.temp_dir)
        
        self.sample_scenario = {
            'id': 'TEST_001',
            'aircraft': [
                {
                    'id': 'AC001',
                    'lat': 52.0,
                    'lon': 4.0,
                    'alt': 35000,
                    'speed': 250,
                    'heading': 90,
                    'type': 'commercial',
                    'flight_phase': 'cruise'
                },
                {
                    'id': 'AC002',
                    'lat': 52.05,
                    'lon': 4.05,
                    'alt': 35000,
                    'speed': 260,
                    'heading': 270,
                    'type': 'commercial',
                    'flight_phase': 'cruise'
                }
            ],
            'environmental_conditions': {
                'weather_severity': 0.2,
                'traffic_density': 0.4
            }
        }
        
        self.sample_ground_truth = {
            'conflicts': [
                {
                    'id1': 'AC001',
                    'id2': 'AC002',
                    'time': 300,
                    'severity': 'medium'
                }
            ]
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scenario_evaluation(self):
        """Test single scenario evaluation"""
        try:
            result = self.evaluator.evaluate_scenario(self.sample_scenario, self.sample_ground_truth)
            
            # Check result structure
            self.assertIn('scenario_id', result)
            self.assertIn('detection_result', result)
            self.assertIn('resolution_maneuvers', result)
            self.assertIn('metrics', result)
            self.assertIn('evaluation_time', result)
            self.assertIn('model_type', result)
            
            # Check detection result structure
            detection = result['detection_result']
            self.assertIn('has_conflict', detection)
            self.assertIn('confidence', detection)
            self.assertIn('time_to_conflict', detection)
            self.assertIn('conflict_pairs', detection)
            self.assertIn('risk_factors', detection)
            
            # Check metrics structure
            metrics = result['metrics']
            self.assertIn('detection_confidence', metrics)
            self.assertIn('intervention_count', metrics)
            
            # Check data types
            self.assertEqual(result['model_type'], 'baseline')
            self.assertIsInstance(result['evaluation_time'], float)
            
        except ImportError:
            self.skipTest("Required dependencies not available")
    
    def test_batch_evaluation(self):
        """Test batch scenario evaluation"""
        try:
            scenarios = [self.sample_scenario] * 3
            ground_truths = [self.sample_ground_truth] * 3
            
            results = self.evaluator.evaluate_batch(scenarios, ground_truths)
            
            # Check results
            self.assertEqual(len(results), 3)
            
            for result in results:
                self.assertIn('scenario_id', result)
                self.assertIn('model_type', result)
                self.assertEqual(result['model_type'], 'baseline')
                
        except ImportError:
            self.skipTest("Required dependencies not available")
    
    def test_training_data_generation(self):
        """Test training data generation"""
        scenarios = [self.sample_scenario] * 5
        ground_truths = [self.sample_ground_truth] * 5
        
        training_data, labels = self.evaluator.generate_training_data(scenarios, ground_truths)
        
        # Check output
        self.assertEqual(len(training_data), 5)
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(isinstance(label, bool) for label in labels))
    
    def test_error_handling(self):
        """Test error handling in evaluation"""
        # Create invalid scenario
        invalid_scenario = {'invalid': 'data'}
        
        result = self.evaluator.evaluate_scenario(invalid_scenario)
        
        # Should return error result
        self.assertIn('error', result)
        self.assertEqual(result['model_type'], 'baseline')


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestBaselineConflictDetector))
    suite.addTest(unittest.makeSuite(TestBaselineConflictResolver))
    suite.addTest(unittest.makeSuite(TestBaselineEvaluator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
