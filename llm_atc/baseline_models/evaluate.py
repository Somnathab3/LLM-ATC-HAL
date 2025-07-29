# baseline_models/evaluate.py
"""
Evaluation Pipeline for Baseline Models
Integrates baseline models into the same metrics pipeline as LLM models
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baseline_models.conflict_detector import BaselineConflictDetector
from baseline_models.conflict_resolver import BaselineConflictResolver
from analysis.metrics import calc_fp_fn, aggregate_thesis_metrics
from llm_atc.metrics.safety_margin_quantifier import calc_separation_margin, calc_efficiency_penalty, count_interventions
from analysis.shift_quantifier import compute_shift_score


class BaselineEvaluator:
    """
    Evaluates baseline models using the same metrics as LLM models.
    Enables direct comparison between traditional and LLM-based approaches.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize baseline evaluator.
        
        Args:
            model_dir: Directory containing trained baseline models
        """
        self.model_dir = Path(model_dir) if model_dir else Path("baseline_models/trained")
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector = BaselineConflictDetector("random_forest")
        self.resolver = BaselineConflictResolver()
        
        # Load pre-trained models if available
        self._load_trained_models()
    
    def evaluate_scenario(self, scenario: Dict[str, Any], 
                         ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate baseline models on a single scenario.
        
        Args:
            scenario: Scenario dictionary with aircraft states
            ground_truth: Optional ground truth for validation
            
        Returns:
            Evaluation results dictionary
        """
        start_time = time.time()
        
        try:
            # 1. Conflict Detection
            detection_result = self.detector.predict(scenario)
            
            # 2. Conflict Resolution (if conflicts detected)
            resolution_maneuvers = []
            if detection_result.has_conflict:
                # Prepare conflict scenario for resolver
                conflict_scenario = self._prepare_conflict_scenario(scenario, detection_result)
                resolution_maneuvers = self.resolver.resolve_conflicts(conflict_scenario)
            
            # 3. Calculate metrics
            metrics = self._calculate_metrics(
                scenario, detection_result, resolution_maneuvers, ground_truth
            )
            
            # 4. Compile results
            evaluation_time = time.time() - start_time
            
            results = {
                'scenario_id': scenario.get('id', 'unknown'),
                'detection_result': {
                    'has_conflict': detection_result.has_conflict,
                    'confidence': detection_result.confidence,
                    'time_to_conflict': detection_result.time_to_conflict,
                    'conflict_pairs': detection_result.conflict_pairs,
                    'risk_factors': detection_result.risk_factors
                },
                'resolution_maneuvers': [
                    {
                        'aircraft_id': m.aircraft_id,
                        'maneuver_type': m.maneuver_type.value,
                        'parameters': m.parameters,
                        'safety_score': m.safety_score,
                        'priority': m.priority,
                        'estimated_delay': m.estimated_delay,
                        'fuel_penalty': m.fuel_penalty
                    }
                    for m in resolution_maneuvers
                ],
                'metrics': metrics,
                'evaluation_time': evaluation_time,
                'model_type': 'baseline'
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating scenario: {e}")
            return self._create_error_result(scenario, str(e))
    
    def evaluate_batch(self, scenarios: List[Dict[str, Any]], 
                      ground_truths: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate baseline models on a batch of scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            ground_truths: Optional list of ground truth dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, scenario in enumerate(scenarios):
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            result = self.evaluate_scenario(scenario, ground_truth)
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                self.logger.info(f"Evaluated {i + 1}/{len(scenarios)} scenarios")
        
        return results
    
    def train_detector(self, training_scenarios: List[Dict[str, Any]], 
                      labels: List[bool]) -> Dict[str, float]:
        """
        Train the baseline conflict detector.
        
        Args:
            training_scenarios: List of training scenarios
            labels: List of conflict labels (True/False)
            
        Returns:
            Training metrics
        """
        try:
            metrics = self.detector.train(training_scenarios, labels)
            
            # Save trained model
            self.model_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.model_dir / "conflict_detector.pkl"
            self.detector.save_model(str(model_path))
            
            self.logger.info(f"Detector trained and saved to {model_path}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training detector: {e}")
            return {}
    
    def generate_training_data(self, scenarios: List[Dict[str, Any]], 
                             ground_truths: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[bool]]:
        """
        Generate training data from scenarios and ground truth.
        
        Args:
            scenarios: List of scenarios
            ground_truths: List of ground truth conflict data
            
        Returns:
            Tuple of (training_scenarios, labels)
        """
        training_data = []
        labels = []
        
        for scenario, gt in zip(scenarios, ground_truths):
            training_data.append(scenario)
            # Extract conflict label from ground truth
            has_conflict = len(gt.get('conflicts', [])) > 0
            labels.append(has_conflict)
        
        return training_data, labels
    
    def _prepare_conflict_scenario(self, scenario: Dict[str, Any], 
                                 detection_result) -> Dict[str, Any]:
        """Prepare conflict scenario for resolver"""
        conflicts = []
        
        # Convert detection result to conflict format
        for ac1_id, ac2_id in detection_result.conflict_pairs:
            conflicts.append({
                'aircraft1_id': ac1_id,
                'aircraft2_id': ac2_id,
                'time_to_conflict': detection_result.time_to_conflict,
                'confidence': detection_result.confidence
            })
        
        return {
            'aircraft': scenario.get('aircraft', []),
            'conflicts': conflicts,
            'environmental_conditions': scenario.get('environmental_conditions', {})
        }
    
    def _calculate_metrics(self, scenario: Dict[str, Any], 
                          detection_result, resolution_maneuvers: List,
                          ground_truth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Detection metrics
        if ground_truth:
            # Extract predicted and ground truth conflicts
            pred_conflicts = [
                {'id1': pair[0], 'id2': pair[1], 'time': detection_result.time_to_conflict}
                for pair in detection_result.conflict_pairs
            ]
            gt_conflicts = ground_truth.get('conflicts', [])
            
            # Calculate FP/FN rates
            fp_rate, fn_rate = calc_fp_fn(pred_conflicts, gt_conflicts)
            metrics['fp_rate'] = fp_rate
            metrics['fn_rate'] = fn_rate
        
        # Safety metrics
        if resolution_maneuvers:
            # Calculate safety scores
            safety_scores = [m.safety_score for m in resolution_maneuvers]
            metrics['avg_safety_score'] = sum(safety_scores) / len(safety_scores)
            metrics['min_safety_score'] = min(safety_scores)
            
            # Calculate intervention count
            commands = [
                {
                    'type': m.maneuver_type.value,
                    'aircraft_id': m.aircraft_id,
                    **m.parameters
                }
                for m in resolution_maneuvers
            ]
            metrics['intervention_count'] = count_interventions(commands)
            
            # Calculate efficiency penalties
            total_delay = sum(m.estimated_delay for m in resolution_maneuvers)
            total_fuel_penalty = sum(m.fuel_penalty for m in resolution_maneuvers)
            metrics['total_delay'] = total_delay
            metrics['total_fuel_penalty'] = total_fuel_penalty
        else:
            metrics['avg_safety_score'] = 1.0  # No conflicts = perfect safety
            metrics['min_safety_score'] = 1.0
            metrics['intervention_count'] = 0
            metrics['total_delay'] = 0
            metrics['total_fuel_penalty'] = 0
        
        # Detection confidence
        metrics['detection_confidence'] = detection_result.confidence
        
        # Risk assessment
        if detection_result.risk_factors:
            metrics['max_risk_factor'] = max(detection_result.risk_factors.values())
            metrics['avg_risk_factor'] = sum(detection_result.risk_factors.values()) / len(detection_result.risk_factors)
        else:
            metrics['max_risk_factor'] = 0.0
            metrics['avg_risk_factor'] = 0.0
        
        return metrics
    
    def _load_trained_models(self):
        """Load pre-trained models if available"""
        detector_path = self.model_dir / "conflict_detector.pkl"
        
        if detector_path.exists():
            try:
                self.detector.load_model(str(detector_path))
                self.logger.info(f"Loaded trained detector from {detector_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load detector: {e}")
    
    def _create_error_result(self, scenario: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create error result for failed evaluations"""
        return {
            'scenario_id': scenario.get('id', 'unknown'),
            'detection_result': {
                'has_conflict': False,
                'confidence': 0.0,
                'time_to_conflict': 0.0,
                'conflict_pairs': [],
                'risk_factors': {}
            },
            'resolution_maneuvers': [],
            'metrics': {
                'fp_rate': 0.0,
                'fn_rate': 0.0,
                'avg_safety_score': 0.0,
                'min_safety_score': 0.0,
                'intervention_count': 0,
                'total_delay': 0,
                'total_fuel_penalty': 0,
                'detection_confidence': 0.0,
                'max_risk_factor': 0.0,
                'avg_risk_factor': 0.0
            },
            'evaluation_time': 0.0,
            'model_type': 'baseline',
            'error': error_msg
        }


# Convenience functions for integration with experiment framework
def evaluate_baseline_on_scenarios(scenarios: List[Dict[str, Any]], 
                                  ground_truths: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to evaluate baseline models on scenarios.
    
    Args:
        scenarios: List of scenarios
        ground_truths: Optional ground truths
        
    Returns:
        List of evaluation results
    """
    evaluator = BaselineEvaluator()
    return evaluator.evaluate_batch(scenarios, ground_truths)


def train_baseline_detector(training_scenarios: List[Dict[str, Any]], 
                           labels: List[bool]) -> Dict[str, float]:
    """
    Convenience function to train baseline detector.
    
    Args:
        training_scenarios: Training scenarios
        labels: Conflict labels
        
    Returns:
        Training metrics
    """
    evaluator = BaselineEvaluator()
    return evaluator.train_detector(training_scenarios, labels)


# Example usage and testing
if __name__ == "__main__":
    # Create sample scenarios for testing
    test_scenarios = [
        {
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
    ]
    
    # Test ground truth
    ground_truths = [
        {
            'conflicts': [
                {
                    'id1': 'AC001',
                    'id2': 'AC002',
                    'time': 300,
                    'severity': 'medium'
                }
            ]
        }
    ]
    
    # Test evaluation
    try:
        evaluator = BaselineEvaluator()
        results = evaluator.evaluate_batch(test_scenarios, ground_truths)
        
        print("Baseline Evaluation Results:")
        for result in results:
            print(f"Scenario: {result['scenario_id']}")
            print(f"Conflicts detected: {result['detection_result']['has_conflict']}")
            print(f"Confidence: {result['detection_result']['confidence']:.3f}")
            print(f"Resolution maneuvers: {len(result['resolution_maneuvers'])}")
            print(f"Metrics: {result['metrics']}")
            print("---")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
