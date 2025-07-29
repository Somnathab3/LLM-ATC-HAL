# testing/scenario_manager.py
"""
Scenario Management Module for LLM-ATC-HAL Framework
Handles scenario generation and edge case management
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import asdict

from scenarios.monte_carlo_framework import MonteCarloScenarioGenerator, ComplexityTier


class ScenarioManager:
    """Manages scenario generation and edge case creation"""
    
    def __init__(self):
        self.monte_carlo_generator = MonteCarloScenarioGenerator()
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_scenarios(self, num_scenarios: int, 
                                       complexity_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate comprehensive test scenarios across all complexity levels"""
        self.logger.info(f"Generating {num_scenarios} test scenarios...")
        
        scenarios = []
        
        # Calculate scenario distribution
        for complexity_level, proportion in complexity_distribution.items():
            num_scenarios_for_level = int(num_scenarios * proportion)
            
            # Convert string to ComplexityTier enum
            try:
                complexity_tier = getattr(ComplexityTier, complexity_level.upper())
            except AttributeError:
                self.logger.warning(f"Unknown complexity level: {complexity_level}, skipping")
                continue
            
            for i in range(num_scenarios_for_level):
                try:
                    # Generate scenario using Monte Carlo framework
                    scenario = self.monte_carlo_generator.generate_scenario(complexity_tier)
                    
                    # Convert scenario object to dict for compatibility
                    # Use the environmental property which provides compatibility
                    env_conditions = scenario.environmental
                    
                    # Convert aircraft list and ensure numpy types are converted to Python native types
                    aircraft_list = []
                    for aircraft in scenario.aircraft_list:
                        # Handle both dataclass and dict aircraft data
                        if hasattr(aircraft, '__dict__'):
                            aircraft_dict = asdict(aircraft)
                        else:
                            aircraft_dict = dict(aircraft)
                            
                        # Convert numpy types to native Python types
                        for key, value in aircraft_dict.items():
                            if hasattr(value, 'item'):  # numpy scalar
                                aircraft_dict[key] = value.item()
                            elif hasattr(value, 'tolist'):  # numpy array
                                aircraft_dict[key] = value.tolist()
                        aircraft_list.append(aircraft_dict)
                    
                    # Convert environmental conditions numpy types
                    for key, value in env_conditions.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            env_conditions[key] = value.item()
                        elif hasattr(value, 'tolist'):  # numpy array
                            env_conditions[key] = value.tolist()
                    
                    scenario_dict = {
                        'scenario_id': f"{complexity_level}_{i:04d}",
                        'test_id': f"{complexity_level}_{i:04d}",  # Keep for backward compatibility
                        'complexity_level': complexity_level,
                        'generation_timestamp': time.time(),
                        'aircraft_list': aircraft_list,
                        'environmental': env_conditions,
                        'environmental_conditions': env_conditions,  # Keep both for compatibility
                        'airspace_region': scenario.airspace_region,
                        'scenario_type': 'monte_carlo_generated'
                    }
                    
                    scenarios.append(scenario_dict)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate scenario {complexity_level}_{i}: {str(e)}")
        
        # Add edge case scenarios
        edge_cases = self.generate_edge_case_scenarios()
        scenarios.extend(edge_cases)
        
        self.logger.info(f"Generated {len(scenarios)} total scenarios")
        return scenarios
    
    def generate_edge_case_scenarios(self) -> List[Dict[str, Any]]:
        """Generate specific edge case scenarios for comprehensive testing"""
        edge_cases = []
        
        # Extreme weather scenarios
        extreme_weather = {
            'scenario_id': 'edge_extreme_weather',
            'test_id': 'edge_extreme_weather',  # Keep for backward compatibility
            'complexity_level': 'extreme',
            'scenario_type': 'edge_case',
            'generation_timestamp': time.time(),
            'aircraft_list': [
                {
                    'id': 'TEST001',
                    'aircraft_type': 'A380',
                    'latitude': 40.0,
                    'longitude': -74.0,
                    'altitude': 35000,
                    'heading': 90,
                    'ground_speed': 520,
                    'vertical_rate': 0
                },
                {
                    'id': 'TEST002',
                    'aircraft_type': 'B737',
                    'latitude': 40.0,
                    'longitude': -73.8,
                    'altitude': 35000,
                    'heading': 270,
                    'ground_speed': 480,
                    'vertical_rate': 0
                }
            ],
            'environmental_conditions': {
                'weather': 'STORM',
                'wind_speed': 80,
                'visibility': 0.5,
                'turbulence_intensity': 0.9
            },
            'airspace_region': 'KJFK_TMA'
        }
        edge_cases.append(extreme_weather)
        
        # Multiple aircraft convergence
        multi_aircraft = {
            'scenario_id': 'edge_multi_convergence',
            'test_id': 'edge_multi_convergence',  # Keep for backward compatibility
            'complexity_level': 'extreme',
            'scenario_type': 'edge_case',
            'generation_timestamp': time.time(),
            'aircraft_list': [],
            'environmental_conditions': {
                'weather': 'CLEAR',
                'wind_speed': 15,
                'visibility': 10,
                'turbulence_intensity': 0.1
            },
            'airspace_region': 'KJFK_TMA'
        }
        
        # Generate 5 converging aircraft
        aircraft_types = ['A320', 'B737', 'A380', 'B777', 'CRJ900']
        for i in range(5):
            aircraft = {
                'id': f'CONV{i+1:02d}',
                'aircraft_type': aircraft_types[i],
                'latitude': 40.0 + 0.1 * np.cos(2 * np.pi * i / 5),
                'longitude': -74.0 + 0.1 * np.sin(2 * np.pi * i / 5),
                'altitude': 35000,
                'heading': (180 + 72 * i) % 360,  # Converging headings, ensure 0-359
                'ground_speed': 450 + i * 20,
                'vertical_rate': 0
            }
            multi_aircraft['aircraft_list'].append(aircraft)
        
        edge_cases.append(multi_aircraft)
        
        # System failure scenarios
        system_failure = {
            'scenario_id': 'edge_system_failure',
            'test_id': 'edge_system_failure',  # Keep for backward compatibility
            'complexity_level': 'extreme',
            'scenario_type': 'edge_case',
            'generation_timestamp': time.time(),
            'aircraft_list': [
                {
                    'id': 'FAIL001',
                    'aircraft_type': 'B777',
                    'latitude': 41.0,
                    'longitude': -74.0,
                    'altitude': 37000,
                    'heading': 180,
                    'ground_speed': 500,
                    'vertical_rate': 0
                }
            ],
            'environmental_conditions': {
                'weather': 'CLEAR',
                'wind_speed': 10,
                'visibility': 8,
                'turbulence_intensity': 0.2
            },
            'airspace_region': 'KJFK_TMA',
            'system_failures': {
                'transponder_failure': True,
                'communication_degraded': True,
                'navigation_accuracy_reduced': 0.3
            }
        }
        edge_cases.append(system_failure)
        
        return edge_cases
    
    def validate_scenario_integrity(self, scenario: Dict[str, Any]) -> bool:
        """Validate scenario integrity and logical consistency"""
        try:
            # Check required fields  
            required_fields = ['scenario_id', 'aircraft_list', 'environmental_conditions']
            for field in required_fields:
                if field not in scenario:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Check aircraft list
            aircraft_list = scenario['aircraft_list']
            if not isinstance(aircraft_list, list) or len(aircraft_list) == 0:
                self.logger.error("Invalid aircraft list")
                return False
            
            # Check each aircraft
            for aircraft in aircraft_list:
                if not self._validate_aircraft(aircraft):
                    return False
            
            # Check environmental conditions
            env_conditions = scenario['environmental_conditions']
            if not self._validate_environmental_conditions(env_conditions):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scenario validation error: {e}")
            return False
    
    def _validate_aircraft(self, aircraft: Dict[str, Any]) -> bool:
        """Validate individual aircraft data"""
        required_fields = ['id', 'aircraft_type', 'latitude', 'longitude', 'altitude', 'heading', 'ground_speed']
        
        for field in required_fields:
            if field not in aircraft:
                self.logger.error(f"Missing aircraft field: {field}")
                return False
        
        # Check ranges
        if not (-90 <= aircraft['latitude'] <= 90):
            self.logger.error(f"Invalid latitude: {aircraft['latitude']}")
            return False
        
        if not (-180 <= aircraft['longitude'] <= 180):
            self.logger.error(f"Invalid longitude: {aircraft['longitude']}")
            return False
        
        if not (0 <= aircraft['altitude'] <= 60000):
            self.logger.error(f"Invalid altitude: {aircraft['altitude']}")
            return False
        
        if not (0 <= aircraft['heading'] < 360):
            self.logger.error(f"Invalid heading: {aircraft['heading']}")
            return False
        
        if not (0 <= aircraft['ground_speed'] <= 1000):
            self.logger.error(f"Invalid ground speed: {aircraft['ground_speed']}")
            return False
        
        return True
    
    def _validate_environmental_conditions(self, env_conditions: Dict[str, Any]) -> bool:
        """Validate environmental conditions"""
        required_fields = ['weather', 'wind_speed', 'visibility']
        
        for field in required_fields:
            if field not in env_conditions:
                self.logger.error(f"Missing environmental field: {field}")
                return False
        
        # Check ranges
        if not (0 <= env_conditions['wind_speed'] <= 200):
            self.logger.error(f"Invalid wind speed: {env_conditions['wind_speed']}")
            return False
        
        if not (0 <= env_conditions['visibility'] <= 20):
            self.logger.error(f"Invalid visibility: {env_conditions['visibility']}")
            return False
        
        return True
