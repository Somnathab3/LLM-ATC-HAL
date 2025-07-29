# testing/test_executor.py
"""
Test Execution Module for LLM-ATC-HAL Framework
Handles individual test execution and result processing
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from validation.input_validator import validator, ValidationResult


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    scenario_type: str
    complexity_level: str
    model_used: str
    
    # Timing metrics
    response_time: float
    processing_time: float
    
    # Detection metrics
    hallucination_detected: bool
    hallucination_types: List[str]
    confidence_score: float
    
    # Safety metrics
    safety_margin: float
    icao_compliant: bool
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    
    # Raw data
    scenario_data: Dict[str, Any]
    llm_response: Dict[str, Any]
    ground_truth: Dict[str, Any]
    errors: List[str]


class TestExecutor:
    """Handles individual test execution with security validation and error handling"""
    
    def __init__(self, ensemble_client, hallucination_detector, safety_quantifier, timeout: float = 30.0):
        self.ensemble_client = ensemble_client
        self.hallucination_detector = hallucination_detector
        self.safety_quantifier = safety_quantifier
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    async def execute_test(self, scenario: Dict[str, Any], models_to_test: List[str]) -> List[TestResult]:
        """Execute test for a scenario across multiple models"""
        start_time = time.time()
        # Try to get scenario_id first, then test_id, then generate one
        test_id = scenario.get('scenario_id') or scenario.get('test_id') or f"test_{int(time.time())}"
        
        # Input validation for security hardening
        validation_result = validator.validate_scenario_data(scenario)
        if not validation_result.is_valid:
            self.logger.error(f"Scenario validation failed for {test_id}: {validation_result.errors}")
            return [self._create_error_result(test_id, scenario, "none", validation_result.errors, start_time)]
        
        # Use sanitized data if available
        if validation_result.sanitized_data:
            scenario = validation_result.sanitized_data
            if validation_result.warnings:
                self.logger.warning(f"Security warnings for {test_id}: {'; '.join(validation_result.warnings)}")
        
        results = []
        
        # Test each model in the configuration
        for model in models_to_test:
            model_start_time = time.time()
            
            try:
                # Query LLM for conflict resolution with timeout
                llm_response = await asyncio.wait_for(
                    self._query_model_safely(model, scenario),
                    timeout=self.timeout
                )
                
                model_end_time = time.time()
                response_time = model_end_time - model_start_time
                
                # Detect hallucinations
                conflict_context = self._create_conflict_context(scenario)
                
                # Prepare response data for hallucination detection
                prepared_response = {
                    'decision_text': llm_response.get('decision_text', ''),
                    'reasoning': llm_response.get('reasoning', ''),
                    'confidence': llm_response.get('confidence', 0.0),
                    'response_time': response_time
                }
                
                hallucination_result = self.hallucination_detector.detect_hallucinations(
                    prepared_response,
                    {'response': 'baseline_response'},  # Simple baseline
                    conflict_context
                )
                
                # Calculate safety margins
                try:
                    safety_result = self._calculate_safety_margins_safely(scenario, llm_response)
                except Exception as e:
                    self.logger.warning(f"Safety margin calculation failed: {str(e)}")
                    safety_result = {'effective_margin': 0.0, 'icao_compliant': False}
                
                # Generate ground truth (baseline algorithm)
                baseline_result = self._generate_baseline_solution(scenario)
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    llm_response, baseline_result, hallucination_result
                )
                
                model_response_time = time.time() - model_start_time
                
                # Create test result
                test_result = TestResult(
                    test_id=f"{test_id}_{model}",
                    scenario_type=scenario.get('scenario_type', 'standard'),
                    complexity_level=scenario.get('complexity_level', 'moderate'),
                    model_used=model,
                    response_time=model_response_time,
                    processing_time=time.time() - start_time,
                    hallucination_detected=hallucination_result.detected if hallucination_result else False,
                    hallucination_types=[t.value for t in hallucination_result.types] if hallucination_result and hallucination_result.types else [],
                    confidence_score=hallucination_result.confidence if hallucination_result else 0.0,
                    safety_margin=safety_result.get('effective_margin', 0.0) if isinstance(safety_result, dict) else 0.0,
                    icao_compliant=safety_result.get('icao_compliant', False) if isinstance(safety_result, dict) else False,
                    accuracy=performance_metrics.get('accuracy', 0.0),
                    precision=performance_metrics.get('precision', 0.0),
                    recall=performance_metrics.get('recall', 0.0),
                    scenario_data=scenario,
                    llm_response=llm_response,
                    ground_truth=baseline_result,
                    errors=[]
                )
                
                results.append(test_result)
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Test timeout for model {model} on scenario {test_id}")
                results.append(self._create_error_result(
                    test_id, scenario, model, [f"Test timeout after {self.timeout}s"], model_start_time
                ))
            except (ConnectionError, OSError) as e:
                self.logger.error(f"Connection error for model {model} on scenario {test_id}: {e}")
                results.append(self._create_error_result(
                    test_id, scenario, model, [f"Connection error: {str(e)}"], model_start_time
                ))
            except ValueError as e:
                self.logger.error(f"Invalid data for model {model} on scenario {test_id}: {e}")
                results.append(self._create_error_result(
                    test_id, scenario, model, [f"Invalid data: {str(e)}"], model_start_time
                ))
            except Exception as e:
                self.logger.exception(f"Unexpected error for model {model} on scenario {test_id}: {e}")
                results.append(self._create_error_result(
                    test_id, scenario, model, [f"Unexpected error: {str(e)}"], model_start_time
                ))
        
        return results
    
    def _create_error_result(self, test_id: str, scenario: Dict[str, Any], model: str, 
                           errors: List[str], start_time: float) -> TestResult:
        """Create a TestResult for failed tests"""
        return TestResult(
            test_id=f"{test_id}_{model}",
            scenario_type=scenario.get('scenario_type', 'unknown'),
            complexity_level=scenario.get('complexity_level', 'unknown'),
            model_used=model,
            response_time=0.0,
            processing_time=time.time() - start_time,
            hallucination_detected=False,
            hallucination_types=[],
            confidence_score=0.0,
            safety_margin=0.0,
            icao_compliant=False,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            scenario_data=scenario,
            llm_response={},
            ground_truth={},
            errors=errors
        )
    
    async def _query_model_safely(self, model: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Query model with input validation"""
        # Create prompt from scenario
        prompt_data = {
            "prompt": self._create_prompt_from_scenario(scenario),
            "context": scenario,
            "model": model
        }
        
        # Validate prompt
        validation_result = validator.validate_llm_prompt(prompt_data)
        if not validation_result.is_valid:
            raise ValueError(f"LLM prompt validation failed: {'; '.join(validation_result.errors)}")
        
        # Use sanitized prompt
        if validation_result.sanitized_data:
            prompt_data = validation_result.sanitized_data
        
        # Query ensemble (this would be implemented based on your ensemble client interface)
        response = await self._query_ensemble(prompt_data)
        return response
    
    def _create_prompt_from_scenario(self, scenario: Dict[str, Any]) -> str:
        """Create LLM prompt from scenario data"""
        aircraft_list = scenario.get('aircraft_list', [])
        env_conditions = scenario.get('environmental_conditions', {})
        
        prompt = f"Air Traffic Control Conflict Resolution:\n\n"
        prompt += f"Aircraft involved: {len(aircraft_list)} aircraft\n"
        prompt += f"Weather conditions: {env_conditions.get('weather', 'CLEAR')}\n"
        prompt += f"Visibility: {env_conditions.get('visibility', 10)} nm\n\n"
        prompt += "Please provide conflict resolution instructions following ICAO standards."
        
        return prompt
    
    async def _query_ensemble(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query the ensemble client (mock implementation)"""
        # This would integrate with your actual ensemble client
        return {
            "decision_text": "Maintain current heading and altitude",
            "confidence": 0.85,
            "safety_score": 0.9
        }
    
    def _create_conflict_context(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create conflict context from scenario"""
        return {
            'scenario_id': scenario.get('test_id', 'unknown'),
            'complexity': scenario.get('complexity_level', 'unknown'),
            'aircraft_count': len(scenario.get('aircraft_list', [])),
            'environmental_conditions': scenario.get('environmental_conditions', {}),
            'timestamp': scenario.get('generation_timestamp', time.time())
        }
    
    def _calculate_safety_margins_safely(self, scenario: Dict[str, Any], llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate safety margins with proper error handling"""
        try:
            # Extract aircraft positions from scenario
            aircraft_list = scenario.get('aircraft_list', [])
            if len(aircraft_list) < 2:
                return {'effective_margin': 5.0, 'icao_compliant': True}  # Safe default for single aircraft
            
            # Create mock ConflictGeometry for now (this would be properly implemented with real conflict detection)
            from metrics.safety_margin_quantifier import ConflictGeometry
            
            aircraft1 = aircraft_list[0]
            aircraft2 = aircraft_list[1]
            
            conflict_geometry = ConflictGeometry(
                aircraft1_pos=(aircraft1.get('latitude', 0), aircraft1.get('longitude', 0), aircraft1.get('altitude', 0)),
                aircraft2_pos=(aircraft2.get('latitude', 0), aircraft2.get('longitude', 0), aircraft2.get('altitude', 0)),
                aircraft1_velocity=(aircraft1.get('ground_speed', 0), aircraft1.get('vertical_rate', 0), aircraft1.get('heading', 0)),
                aircraft2_velocity=(aircraft2.get('ground_speed', 0), aircraft2.get('vertical_rate', 0), aircraft2.get('heading', 0)),
                time_to_closest_approach=60.0,  # Mock value
                closest_approach_distance=5.0,  # Mock value - 5 nautical miles
                closest_approach_altitude_diff=1000.0  # Mock value - 1000 feet
            )
            
            # Mock resolution maneuver
            resolution_maneuver = {
                'type': 'heading_change',
                'aircraft_id': aircraft1.get('id', 'unknown'),
                'heading_change': 10,  # degrees
                'altitude_change': 0   # feet
            }
            
            # Calculate safety margins
            safety_margin = self.safety_quantifier.calculate_safety_margins(
                conflict_geometry, 
                resolution_maneuver,
                scenario.get('environmental_conditions', {})
            )
            
            # Convert SafetyMargin object to dictionary
            return {
                'effective_margin': safety_margin.effective_margin,
                'icao_compliant': safety_margin.safety_level in ['adequate', 'excellent'],
                'safety_level': safety_margin.safety_level,
                'horizontal_margin': safety_margin.horizontal_margin,
                'vertical_margin': safety_margin.vertical_margin,
                'temporal_margin': safety_margin.temporal_margin
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate safety margins: {str(e)}")
            return {'effective_margin': 0.0, 'icao_compliant': False}
    
    def _generate_baseline_solution(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Generate baseline solution (mock implementation)"""
        return {
            "decision": "standard_separation",
            "confidence": 0.75,
            "safety_score": 0.8
        }
    
    def _calculate_performance_metrics(self, llm_response: Dict[str, Any], 
                                     baseline_result: Dict[str, Any], 
                                     hallucination_result: Any) -> Dict[str, float]:
        """Calculate performance metrics"""
        # Mock implementation - would calculate actual metrics
        # hallucination_result can be either HallucinationResult object or dict
        return {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90
        }
