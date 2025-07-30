#!/usr/bin/env python3
"""
Monte Carlo Benchmark Runner for LLM-ATC-HAL
============================================
Orchestrates the three-stage pipeline: conflict detection, resolution, and verification
across batches of scenarios to produce comprehensive metrics and logs.

Pipeline Stages:
1. Conflict Detection - Use ground truth and/or LLM detection
2. Conflict Resolution - Generate LLM commands for detected conflicts  
3. Conflict Verification - Execute resolutions and measure outcomes

Key Features:
- Configurable scenario generation across complexity tiers
- Distribution shift testing capability
- Comprehensive logging and metrics collection
- False positive/negative analysis 
- Safety margin quantification
- Visualization of results
- Robust error handling and recovery
"""

import json
import csv
import os
import logging
import time
import uuid
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# LLM-ATC-HAL imports
from scenarios.scenario_generator import (
    ScenarioGenerator, ScenarioType, HorizontalCREnv, 
    VerticalCREnv, SectorCREnv, ComplexityTier,
    generate_horizontal_scenario, generate_vertical_scenario, 
    generate_sector_scenario
)
from scenarios.monte_carlo_framework import (
    BlueSkyScenarioGenerator, ScenarioConfiguration
)
from llm_atc.tools.llm_prompt_engine import LLMPromptEngine, ConflictPromptData, ResolutionResponse
from llm_atc.tools import bluesky_tools
from llm_atc.tools.bluesky_tools import AircraftInfo, ConflictInfo, BlueSkyToolsError


@dataclass 
class BenchmarkConfiguration:
    """Configuration for Monte Carlo benchmark runs"""
    # Scenario parameters
    num_scenarios_per_type: int = 50
    scenario_types: List[ScenarioType] = None
    complexity_tiers: List[ComplexityTier] = None
    distribution_shift_levels: List[str] = None
    
    # Simulation parameters
    time_horizon_minutes: float = 10.0
    max_interventions_per_scenario: int = 5
    step_size_seconds: float = 10.0
    
    # Output configuration
    output_directory: str = "output/monte_carlo_benchmark"
    generate_visualizations: bool = True
    detailed_logging: bool = True
    
    # LLM configuration
    llm_model: str = "llama3.1:8b"
    enable_llm_detection: bool = True
    enable_function_calls: bool = False
    
    # Performance thresholds
    min_separation_nm: float = 5.0
    min_separation_ft: float = 1000.0
    
    def __post_init__(self):
        """Set defaults for mutable fields"""
        if self.scenario_types is None:
            self.scenario_types = [ScenarioType.HORIZONTAL, ScenarioType.VERTICAL, ScenarioType.SECTOR]
        if self.complexity_tiers is None:
            self.complexity_tiers = [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX]
        if self.distribution_shift_levels is None:
            self.distribution_shift_levels = ["in_distribution", "moderate_shift", "extreme_shift"]


@dataclass
class ScenarioResult:
    """Results from executing a single scenario"""
    # Scenario metadata
    scenario_id: str
    scenario_type: str
    complexity_tier: str
    distribution_shift_tier: str
    aircraft_count: int
    duration_minutes: float
    
    # Ground truth
    true_conflicts: List[Dict[str, Any]]
    num_true_conflicts: int
    
    # Detection results
    predicted_conflicts: List[Dict[str, Any]]
    num_predicted_conflicts: int
    detection_method: str  # 'ground_truth', 'llm', 'hybrid'
    
    # Resolution results
    llm_commands: List[str]
    resolution_success: bool
    num_interventions: int
    
    # Safety metrics
    min_separation_nm: float
    min_separation_ft: float
    separation_violations: int
    safety_margin_hz: float
    safety_margin_vt: float
    
    # Efficiency metrics
    extra_distance_nm: float
    total_delay_seconds: float
    fuel_penalty_percent: float
    
    # Performance metrics
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    detection_accuracy: float
    precision: float
    recall: float
    
    # Execution metadata
    execution_time_seconds: float
    errors: List[str]
    warnings: List[str]
    timestamp: str
    
    # Environmental factors
    wind_speed_kts: float
    visibility_nm: float
    turbulence_level: float


class MonteCarloBenchmark:
    """
    Monte Carlo benchmark runner for LLM-ATC-HAL performance evaluation.
    
    Orchestrates comprehensive testing across scenario types, complexity levels,
    and distribution shifts to generate robust performance metrics.
    """
    
    def __init__(self, config: Optional[BenchmarkConfiguration] = None):
        """
        Initialize the Monte Carlo benchmark runner.
        
        Args:
            config: Benchmark configuration. If None, uses defaults.
        """
        self.config = config or BenchmarkConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator()
        self.llm_engine = LLMPromptEngine(
            model=self.config.llm_model,
            enable_function_calls=self.config.enable_function_calls
        )
        
        # Results storage
        self.results: List[ScenarioResult] = []
        self.benchmark_start_time: Optional[datetime] = None
        self.benchmark_id = str(uuid.uuid4())[:8]
        
        # Setup output directory
        self._setup_output_directory()
        
        # Initialize logging
        self._setup_logging()
        
        self.logger.info(f"Initialized Monte Carlo benchmark {self.benchmark_id}")
        self.logger.info(f"Configuration: {self.config}")
    
    def _setup_output_directory(self):
        """Create output directory structure"""
        base_dir = Path(self.config.output_directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.output_dir = base_dir / f"benchmark_{self.benchmark_id}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup detailed logging for benchmark execution"""
        if not self.config.detailed_logging:
            return
            
        log_file = self.output_dir / "logs" / "benchmark.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete Monte Carlo benchmark.
        
        Returns:
            Summary statistics and results overview
        """
        self.benchmark_start_time = datetime.now()
        self.logger.info("Starting Monte Carlo benchmark execution")
        
        try:
            # Generate and execute scenarios
            total_scenarios = self._calculate_total_scenarios()
            self.logger.info(f"Planning to execute {total_scenarios} scenarios")
            
            scenario_count = 0
            for scenario_type in self.config.scenario_types:
                for complexity_tier in self.config.complexity_tiers:
                    for shift_level in self.config.distribution_shift_levels:
                        scenario_count += self._run_scenario_batch(
                            scenario_type, complexity_tier, shift_level
                        )
            
            # Generate summary and visualizations
            summary = self._generate_summary()
            
            if self.config.generate_visualizations:
                self._generate_visualizations()
            
            # Save results
            self._save_results()
            
            self.logger.info(f"Benchmark completed successfully: {scenario_count} scenarios executed")
            return summary
            
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _calculate_total_scenarios(self) -> int:
        """Calculate total number of scenarios to be executed"""
        return (
            len(self.config.scenario_types) * 
            len(self.config.complexity_tiers) * 
            len(self.config.distribution_shift_levels) * 
            self.config.num_scenarios_per_type
        )
    
    def _run_scenario_batch(self, scenario_type: ScenarioType, 
                           complexity_tier: ComplexityTier, 
                           shift_level: str) -> int:
        """
        Execute a batch of scenarios for given parameters.
        
        Args:
            scenario_type: Type of scenario to generate
            complexity_tier: Complexity level
            shift_level: Distribution shift level
            
        Returns:
            Number of scenarios successfully executed
        """
        batch_id = f"{scenario_type.value}_{complexity_tier.value}_{shift_level}"
        self.logger.info(f"Starting batch: {batch_id}")
        
        successful_scenarios = 0
        
        for i in range(self.config.num_scenarios_per_type):
            scenario_id = f"{batch_id}_{i+1:03d}"
            
            try:
                # Generate scenario
                scenario = self._generate_scenario(
                    scenario_type, complexity_tier, shift_level, scenario_id
                )
                
                # Execute three-stage pipeline
                result = self._execute_scenario_pipeline(scenario, scenario_id)
                
                # Store result
                self.results.append(result)
                successful_scenarios += 1
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Batch {batch_id}: completed {i+1}/{self.config.num_scenarios_per_type}")
                
            except Exception as e:
                self.logger.error(f"Failed to execute scenario {scenario_id}: {e}")
                
                # Create error result
                error_result = self._create_error_result(scenario_id, scenario_type, 
                                                       complexity_tier, shift_level, str(e))
                self.results.append(error_result)
        
        self.logger.info(f"Batch {batch_id} completed: {successful_scenarios}/{self.config.num_scenarios_per_type} successful")
        return successful_scenarios
    
    def _generate_scenario(self, scenario_type: ScenarioType, 
                          complexity_tier: ComplexityTier,
                          shift_level: str, scenario_id: str) -> Any:
        """Generate a scenario based on type and parameters"""
        
        if scenario_type == ScenarioType.HORIZONTAL:
            return generate_horizontal_scenario(
                n_aircraft=self._get_aircraft_count_for_complexity(complexity_tier),
                conflict=True,  # Force conflicts for testing
                complexity_tier=complexity_tier,
                distribution_shift_tier=shift_level
            )
        elif scenario_type == ScenarioType.VERTICAL:
            return generate_vertical_scenario(
                n_aircraft=self._get_aircraft_count_for_complexity(complexity_tier),
                conflict=True,
                complexity_tier=complexity_tier,
                distribution_shift_tier=shift_level
            )
        elif scenario_type == ScenarioType.SECTOR:
            return generate_sector_scenario(
                complexity=complexity_tier,
                shift_level=shift_level,
                force_conflicts=True
            )
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def _get_aircraft_count_for_complexity(self, complexity_tier: ComplexityTier) -> int:
        """Get appropriate aircraft count for complexity level"""
        counts = {
            ComplexityTier.SIMPLE: 2,
            ComplexityTier.MODERATE: 4,
            ComplexityTier.COMPLEX: 6,
            ComplexityTier.EXTREME: 8
        }
        return counts.get(complexity_tier, 3)
    
    def _execute_scenario_pipeline(self, scenario: Any, scenario_id: str) -> ScenarioResult:
        """
        Execute the three-stage pipeline for a single scenario.
        
        Args:
            scenario: Generated scenario object
            scenario_id: Unique scenario identifier
            
        Returns:
            ScenarioResult with complete execution data
        """
        pipeline_start = time.time()
        errors = []
        warnings = []
        
        try:
            # Stage 1: Setup and Reset BlueSky
            self._reset_bluesky_simulation()
            self._load_scenario_commands(scenario)
            
            # Stage 2: Conflict Detection
            ground_truth_conflicts = self._extract_ground_truth_conflicts(scenario)
            detected_conflicts = self._detect_conflicts(scenario)
            
            # Stage 3: Conflict Resolution
            resolutions = []
            if detected_conflicts:
                resolutions = self._resolve_conflicts(detected_conflicts, scenario)
            
            # Stage 4: Verification and Monitoring
            verification_results = self._verify_resolutions(scenario, resolutions)
            
            # Stage 5: Calculate Metrics
            metrics = self._calculate_scenario_metrics(
                ground_truth_conflicts, detected_conflicts, 
                resolutions, verification_results
            )
            
            # Create result object
            result = ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=scenario.scenario_type.value if hasattr(scenario, 'scenario_type') else 'unknown',
                complexity_tier=getattr(scenario, 'complexity_tier', ComplexityTier.MODERATE).value,
                distribution_shift_tier=getattr(scenario, 'distribution_shift_tier', 'in_distribution'),
                aircraft_count=getattr(scenario, 'aircraft_count', len(scenario.initial_states)),
                duration_minutes=getattr(scenario, 'duration_minutes', self.config.time_horizon_minutes),
                
                # Ground truth
                true_conflicts=ground_truth_conflicts,
                num_true_conflicts=len(ground_truth_conflicts),
                
                # Detection
                predicted_conflicts=detected_conflicts,
                num_predicted_conflicts=len(detected_conflicts),
                detection_method='hybrid' if self.config.enable_llm_detection else 'ground_truth',
                
                # Resolution
                llm_commands=[r.get('command', '') for r in resolutions],
                resolution_success=verification_results.get('resolution_success', False),
                num_interventions=len(resolutions),
                
                # Safety metrics
                min_separation_nm=verification_results.get('min_separation_nm', 999.0),
                min_separation_ft=verification_results.get('min_separation_ft', 999999.0),
                separation_violations=verification_results.get('violations', 0),
                safety_margin_hz=max(0, verification_results.get('min_separation_nm', 0) - self.config.min_separation_nm),
                safety_margin_vt=max(0, verification_results.get('min_separation_ft', 0) - self.config.min_separation_ft),
                
                # Efficiency metrics
                extra_distance_nm=verification_results.get('extra_distance_nm', 0.0),
                total_delay_seconds=verification_results.get('total_delay', 0.0),
                fuel_penalty_percent=verification_results.get('fuel_penalty', 0.0),
                
                # Performance metrics
                **metrics,
                
                # Execution metadata
                execution_time_seconds=time.time() - pipeline_start,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now().isoformat(),
                
                # Environmental factors
                wind_speed_kts=getattr(scenario, 'environmental_conditions', {}).get('wind_speed_kts', 0),
                visibility_nm=getattr(scenario, 'environmental_conditions', {}).get('visibility_nm', 10),
                turbulence_level=getattr(scenario, 'environmental_conditions', {}).get('turbulence_intensity', 0),
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed for {scenario_id}: {e}")
            errors.append(str(e))
            
            return self._create_error_result(
                scenario_id, 
                getattr(scenario, 'scenario_type', ScenarioType.SECTOR),
                getattr(scenario, 'complexity_tier', ComplexityTier.MODERATE),
                getattr(scenario, 'distribution_shift_tier', 'in_distribution'),
                str(e)
            )
    
    def _reset_bluesky_simulation(self):
        """Reset BlueSky simulation to clean state"""
        try:
            result = bluesky_tools.send_command('RESET')
            self.logger.debug(f"BlueSky reset: {result}")
        except Exception as e:
            self.logger.warning(f"BlueSky reset failed: {e}")
    
    def _load_scenario_commands(self, scenario: Any):
        """Load scenario commands into BlueSky"""
        try:
            commands = getattr(scenario, 'commands', [])
            if not commands and hasattr(scenario, 'bluesky_commands'):
                commands = scenario.bluesky_commands
            
            for command in commands:
                result = bluesky_tools.send_command(command)
                self.logger.debug(f"Executed command '{command}': {result}")
                
        except Exception as e:
            self.logger.error(f"Failed to load scenario commands: {e}")
            raise
    
    def _extract_ground_truth_conflicts(self, scenario: Any) -> List[Dict[str, Any]]:
        """Extract ground truth conflicts from scenario"""
        try:
            if hasattr(scenario, 'ground_truth_conflicts'):
                return [asdict(conflict) for conflict in scenario.ground_truth_conflicts]
            else:
                # Create mock ground truth for testing
                return [{
                    'aircraft_pair': ('AC001', 'AC002'),
                    'conflict_type': 'horizontal',
                    'time_to_conflict': 120.0,
                    'min_separation': {'horizontal_nm': 3.5, 'vertical_ft': 0},
                    'severity': 'medium',
                    'is_actual_conflict': True
                }]
        except Exception as e:
            self.logger.warning(f"Failed to extract ground truth: {e}")
            return []
    
    def _detect_conflicts(self, scenario: Any) -> List[Dict[str, Any]]:
        """Perform conflict detection using available methods"""
        detected_conflicts = []
        
        try:
            # Method 1: BlueSky built-in conflict detection
            bluesky_conflicts = bluesky_tools.get_conflict_info()
            for conflict in bluesky_conflicts.get('conflicts', []):
                detected_conflicts.append({
                    'source': 'bluesky',
                    'aircraft_1': conflict['aircraft_1'],
                    'aircraft_2': conflict['aircraft_2'],
                    'horizontal_separation': conflict['horizontal_separation'],
                    'vertical_separation': conflict['vertical_separation'],
                    'time_to_cpa': conflict['time_to_cpa'],
                    'severity': conflict['severity']
                })
            
            # Method 2: LLM-based detection (if enabled)
            if self.config.enable_llm_detection:
                aircraft_states = self._get_aircraft_states_for_llm()
                llm_detection = self.llm_engine.detect_conflict_via_llm(
                    aircraft_states, self.config.time_horizon_minutes
                )
                
                if llm_detection.get('conflict_detected', False):
                    for pair in llm_detection.get('aircraft_pairs', []):
                        detected_conflicts.append({
                            'source': 'llm',
                            'aircraft_1': pair[0],
                            'aircraft_2': pair[1],
                            'confidence': llm_detection.get('confidence', 0.5),
                            'priority': llm_detection.get('priority', 'unknown')
                        })
            
        except Exception as e:
            self.logger.error(f"Conflict detection failed: {e}")
        
        return detected_conflicts
    
    def _get_aircraft_states_for_llm(self) -> List[Dict[str, Any]]:
        """Get current aircraft states formatted for LLM"""
        try:
            aircraft_info = bluesky_tools.get_all_aircraft_info()
            states = []
            
            for aircraft_id, info in aircraft_info.get('aircraft', {}).items():
                states.append({
                    'id': aircraft_id,
                    'lat': info['lat'],
                    'lon': info['lon'],
                    'alt': info['alt'],
                    'hdg': info['hdg'],
                    'spd': info['spd'],
                    'vs': info['vs']
                })
            
            return states
            
        except Exception as e:
            self.logger.error(f"Failed to get aircraft states: {e}")
            return []
    
    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]], scenario: Any) -> List[Dict[str, Any]]:
        """Generate LLM-based conflict resolutions"""
        resolutions = []
        
        for conflict in conflicts:
            try:
                # Create conflict info for LLM
                conflict_info = self._format_conflict_for_llm(conflict, scenario)
                
                # Get LLM resolution
                resolution_command = self.llm_engine.get_conflict_resolution(conflict_info)
                
                if resolution_command:
                    # Execute the command
                    execution_result = bluesky_tools.send_command(resolution_command)
                    
                    resolutions.append({
                        'conflict': conflict,
                        'command': resolution_command,
                        'execution_result': execution_result,
                        'timestamp': time.time()
                    })
                    
                    self.logger.info(f"Executed resolution: {resolution_command}")
                
            except Exception as e:
                self.logger.error(f"Failed to resolve conflict {conflict}: {e}")
        
        return resolutions
    
    def _format_conflict_for_llm(self, conflict: Dict[str, Any], scenario: Any) -> Dict[str, Any]:
        """Format conflict data for LLM prompt engine"""
        try:
            # Get aircraft information
            aircraft_info = bluesky_tools.get_all_aircraft_info()
            
            ac1_id = conflict.get('aircraft_1', 'AC001')
            ac2_id = conflict.get('aircraft_2', 'AC002')
            
            ac1_info = aircraft_info.get('aircraft', {}).get(ac1_id, {})
            ac2_info = aircraft_info.get('aircraft', {}).get(ac2_id, {})
            
            return {
                'aircraft_1_id': ac1_id,
                'aircraft_2_id': ac2_id,
                'time_to_conflict': conflict.get('time_to_cpa', 120.0),
                'closest_approach_distance': conflict.get('horizontal_separation', 3.5),
                'conflict_type': 'convergent',
                'urgency_level': conflict.get('severity', 'medium'),
                'aircraft_1': ac1_info,
                'aircraft_2': ac2_info,
                'environmental_conditions': getattr(scenario, 'environmental_conditions', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to format conflict for LLM: {e}")
            return {}
    
    def _verify_resolutions(self, scenario: Any, resolutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify resolution effectiveness by stepping simulation"""
        verification_results = {
            'resolution_success': False,
            'min_separation_nm': 999.0,
            'min_separation_ft': 999999.0,
            'violations': 0,
            'extra_distance_nm': 0.0,
            'total_delay': 0.0,
            'fuel_penalty': 0.0
        }
        
        try:
            # Step simulation forward
            total_steps = int((self.config.time_horizon_minutes * 60) / self.config.step_size_seconds)
            
            min_separation_recorded = []
            
            for step in range(total_steps):
                # Step simulation
                bluesky_tools.step_simulation(self.config.step_size_seconds)
                
                # Check separations
                aircraft_info = bluesky_tools.get_all_aircraft_info()
                separations = self._calculate_all_separations(aircraft_info)
                
                if separations:
                    min_hz = min(sep['horizontal_nm'] for sep in separations)
                    min_vt = min(sep['vertical_ft'] for sep in separations)
                    
                    min_separation_recorded.append({
                        'time': step * self.config.step_size_seconds,
                        'horizontal_nm': min_hz,
                        'vertical_ft': min_vt
                    })
                    
                    # Check for violations
                    if (min_hz < self.config.min_separation_nm and 
                        min_vt < self.config.min_separation_ft):
                        verification_results['violations'] += 1
            
            # Calculate final metrics
            if min_separation_recorded:
                verification_results['min_separation_nm'] = min(
                    s['horizontal_nm'] for s in min_separation_recorded
                )
                verification_results['min_separation_ft'] = min(
                    s['vertical_ft'] for s in min_separation_recorded
                )
                
                # Resolution is successful if no violations occurred
                verification_results['resolution_success'] = verification_results['violations'] == 0
                
                # Calculate efficiency metrics (simplified)
                verification_results['extra_distance_nm'] = len(resolutions) * 5.0  # Estimate
                verification_results['total_delay'] = len(resolutions) * 30.0  # Estimate in seconds
                verification_results['fuel_penalty'] = len(resolutions) * 2.0  # Estimate percentage
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
        
        return verification_results
    
    def _calculate_all_separations(self, aircraft_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate separations between all aircraft pairs"""
        separations = []
        
        try:
            aircraft_list = list(aircraft_info.get('aircraft', {}).values())
            
            for i in range(len(aircraft_list)):
                for j in range(i + 1, len(aircraft_list)):
                    ac1 = aircraft_list[i]
                    ac2 = aircraft_list[j]
                    
                    # Calculate horizontal separation (simplified)
                    lat_diff = ac1['lat'] - ac2['lat']
                    lon_diff = ac1['lon'] - ac2['lon']
                    horizontal_nm = ((lat_diff**2 + lon_diff**2)**0.5) * 60  # Rough conversion
                    
                    # Calculate vertical separation
                    vertical_ft = abs(ac1['alt'] - ac2['alt'])
                    
                    separations.append({
                        'aircraft_1': ac1['id'],
                        'aircraft_2': ac2['id'],
                        'horizontal_nm': horizontal_nm,
                        'vertical_ft': vertical_ft
                    })
        
        except Exception as e:
            self.logger.error(f"Failed to calculate separations: {e}")
        
        return separations
    
    def _calculate_scenario_metrics(self, ground_truth: List[Dict[str, Any]], 
                                  detected: List[Dict[str, Any]], 
                                  resolutions: List[Dict[str, Any]],
                                  verification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for scenario"""
        
        # Convert to sets for easier comparison
        true_conflicts = set()
        for gt in ground_truth:
            if gt.get('is_actual_conflict', True):
                pair = gt.get('aircraft_pair', ('AC001', 'AC002'))
                true_conflicts.add(tuple(sorted(pair)))
        
        detected_conflicts = set()
        for det in detected:
            ac1 = det.get('aircraft_1', 'AC001')
            ac2 = det.get('aircraft_2', 'AC002')
            detected_conflicts.add(tuple(sorted([ac1, ac2])))
        
        # Calculate confusion matrix
        tp = len(true_conflicts.intersection(detected_conflicts))
        fp = len(detected_conflicts - true_conflicts)
        fn = len(true_conflicts - detected_conflicts)
        tn = max(0, 10 - tp - fp - fn)  # Estimate based on potential pairs
        
        # Calculate metrics
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        
        return {
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn,
            'detection_accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def _create_error_result(self, scenario_id: str, scenario_type: ScenarioType,
                           complexity_tier: ComplexityTier, shift_level: str, 
                           error: str) -> ScenarioResult:
        """Create error result for failed scenarios"""
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=scenario_type.value,
            complexity_tier=complexity_tier.value,
            distribution_shift_tier=shift_level,
            aircraft_count=0,
            duration_minutes=0.0,
            true_conflicts=[], num_true_conflicts=0,
            predicted_conflicts=[], num_predicted_conflicts=0,
            detection_method='error',
            llm_commands=[], resolution_success=False, num_interventions=0,
            min_separation_nm=0.0, min_separation_ft=0.0, separation_violations=999,
            safety_margin_hz=0.0, safety_margin_vt=0.0,
            extra_distance_nm=0.0, total_delay_seconds=0.0, fuel_penalty_percent=0.0,
            false_positives=0, false_negatives=0, true_positives=0, true_negatives=0,
            detection_accuracy=0.0, precision=0.0, recall=0.0,
            execution_time_seconds=0.0,
            errors=[error], warnings=[],
            timestamp=datetime.now().isoformat(),
            wind_speed_kts=0.0, visibility_nm=0.0, turbulence_level=0.0
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from all results"""
        if not self.results:
            return {'error': 'No results to summarize'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Overall statistics
        total_scenarios = len(df)
        successful_scenarios = len(df[df['errors'].apply(len) == 0])
        
        # Detection performance
        avg_accuracy = df['detection_accuracy'].mean()
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        
        # Safety metrics
        avg_min_separation = df['min_separation_nm'].mean()
        total_violations = df['separation_violations'].sum()
        
        # Efficiency metrics
        avg_extra_distance = df['extra_distance_nm'].mean()
        avg_delay = df['total_delay_seconds'].mean()
        
        # By scenario type
        type_stats = df.groupby('scenario_type').agg({
            'detection_accuracy': 'mean',
            'resolution_success': lambda x: x.sum() / len(x),
            'min_separation_nm': 'mean'
        }).to_dict()
        
        # By complexity
        complexity_stats = df.groupby('complexity_tier').agg({
            'detection_accuracy': 'mean',
            'execution_time_seconds': 'mean',
            'num_interventions': 'mean'
        }).to_dict()
        
        # By distribution shift
        shift_stats = df.groupby('distribution_shift_tier').agg({
            'detection_accuracy': 'mean',
            'false_positives': 'sum',
            'false_negatives': 'sum'
        }).to_dict()
        
        summary = {
            'benchmark_id': self.benchmark_id,
            'execution_time': str(datetime.now() - self.benchmark_start_time),
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': successful_scenarios / total_scenarios,
            
            'overall_performance': {
                'detection_accuracy': avg_accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'avg_min_separation_nm': avg_min_separation,
                'total_violations': int(total_violations),
                'avg_extra_distance_nm': avg_extra_distance,
                'avg_delay_seconds': avg_delay
            },
            
            'by_scenario_type': type_stats,
            'by_complexity': complexity_stats,
            'by_distribution_shift': shift_stats,
            
            'configuration': asdict(self.config)
        }
        
        return summary
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations of results"""
        if not self.results:
            self.logger.warning("No results available for visualization")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        # Set up matplotlib
        plt.style.use('default')
        fig_size = (12, 8)
        
        # 1. Detection Performance Overview
        self._plot_detection_performance(df, fig_size)
        
        # 2. Safety Margins Distribution
        self._plot_safety_margins(df, fig_size)
        
        # 3. Efficiency Metrics
        self._plot_efficiency_metrics(df, fig_size)
        
        # 4. Performance by Scenario Type
        self._plot_performance_by_type(df, fig_size)
        
        # 5. Distribution Shift Impact
        self._plot_distribution_shift_impact(df, fig_size)
        
        self.logger.info("Visualizations generated successfully")
    
    def _plot_detection_performance(self, df: pd.DataFrame, fig_size: Tuple[int, int]):
        """Plot detection performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Conflict Detection Performance', fontsize=16, fontweight='bold')
        
        # Accuracy histogram
        axes[0,0].hist(df['detection_accuracy'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_xlabel('Detection Accuracy')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Detection Accuracy Distribution')
        axes[0,0].axvline(df['detection_accuracy'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["detection_accuracy"].mean():.3f}')
        axes[0,0].legend()
        
        # Precision vs Recall scatter
        axes[0,1].scatter(df['recall'], df['precision'], alpha=0.6, color='green')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision vs Recall')
        axes[0,1].grid(True, alpha=0.3)
        
        # False Positives vs False Negatives
        fp_fn_data = df.groupby(['false_positives', 'false_negatives']).size().reset_index(name='count')
        scatter = axes[1,0].scatter(fp_fn_data['false_positives'], fp_fn_data['false_negatives'], 
                                   s=fp_fn_data['count']*20, alpha=0.6, color='orange')
        axes[1,0].set_xlabel('False Positives')
        axes[1,0].set_ylabel('False Negatives')
        axes[1,0].set_title('False Positives vs False Negatives')
        axes[1,0].grid(True, alpha=0.3)
        
        # Success rate by complexity
        success_by_complexity = df.groupby('complexity_tier')['resolution_success'].mean()
        axes[1,1].bar(success_by_complexity.index, success_by_complexity.values, 
                     color='purple', alpha=0.7)
        axes[1,1].set_xlabel('Complexity Tier')
        axes[1,1].set_ylabel('Resolution Success Rate')
        axes[1,1].set_title('Success Rate by Complexity')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "detection_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_safety_margins(self, df: pd.DataFrame, fig_size: Tuple[int, int]):
        """Plot safety margin distributions"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Safety Margin Analysis', fontsize=16, fontweight='bold')
        
        # Horizontal separation distribution
        axes[0,0].hist(df['min_separation_nm'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(self.config.min_separation_nm, color='red', linestyle='--', 
                         label=f'Min Required: {self.config.min_separation_nm} NM')
        axes[0,0].set_xlabel('Minimum Separation (NM)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Horizontal Separation Distribution')
        axes[0,0].legend()
        
        # Vertical separation distribution
        axes[0,1].hist(df['min_separation_ft'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].axvline(self.config.min_separation_ft, color='red', linestyle='--', 
                         label=f'Min Required: {self.config.min_separation_ft} ft')
        axes[0,1].set_xlabel('Minimum Separation (ft)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Vertical Separation Distribution')
        axes[0,1].legend()
        
        # Violations by scenario type
        violations_by_type = df.groupby('scenario_type')['separation_violations'].sum()
        axes[1,0].bar(violations_by_type.index, violations_by_type.values, 
                     color='red', alpha=0.7)
        axes[1,0].set_xlabel('Scenario Type')
        axes[1,0].set_ylabel('Total Violations')
        axes[1,0].set_title('Separation Violations by Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Safety margin correlation
        axes[1,1].scatter(df['safety_margin_hz'], df['safety_margin_vt'], alpha=0.6, color='purple')
        axes[1,1].set_xlabel('Horizontal Safety Margin (NM)')
        axes[1,1].set_ylabel('Vertical Safety Margin (ft)')
        axes[1,1].set_title('Safety Margin Correlation')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "safety_margins.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_metrics(self, df: pd.DataFrame, fig_size: Tuple[int, int]):
        """Plot efficiency and cost metrics"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Efficiency Metrics', fontsize=16, fontweight='bold')
        
        # Extra distance distribution
        axes[0,0].hist(df['extra_distance_nm'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0,0].set_xlabel('Extra Distance (NM)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Extra Path Distance Distribution')
        
        # Delay distribution
        axes[0,1].hist(df['total_delay_seconds'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0,1].set_xlabel('Total Delay (seconds)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Delay Distribution')
        
        # Interventions vs Efficiency
        axes[1,0].scatter(df['num_interventions'], df['extra_distance_nm'], alpha=0.6, color='blue')
        axes[1,0].set_xlabel('Number of Interventions')
        axes[1,0].set_ylabel('Extra Distance (NM)')
        axes[1,0].set_title('Interventions vs Extra Distance')
        axes[1,0].grid(True, alpha=0.3)
        
        # Fuel penalty by complexity
        fuel_by_complexity = df.groupby('complexity_tier')['fuel_penalty_percent'].mean()
        axes[1,1].bar(fuel_by_complexity.index, fuel_by_complexity.values, 
                     color='green', alpha=0.7)
        axes[1,1].set_xlabel('Complexity Tier')
        axes[1,1].set_ylabel('Fuel Penalty (%)')
        axes[1,1].set_title('Fuel Penalty by Complexity')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "efficiency_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_by_type(self, df: pd.DataFrame, fig_size: Tuple[int, int]):
        """Plot performance metrics by scenario type"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Performance by Scenario Type', fontsize=16, fontweight='bold')
        
        # Accuracy by type
        acc_by_type = df.groupby('scenario_type')['detection_accuracy'].mean()
        axes[0,0].bar(acc_by_type.index, acc_by_type.values, color='blue', alpha=0.7)
        axes[0,0].set_xlabel('Scenario Type')
        axes[0,0].set_ylabel('Detection Accuracy')
        axes[0,0].set_title('Detection Accuracy by Type')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Execution time by type
        time_by_type = df.groupby('scenario_type')['execution_time_seconds'].mean()
        axes[0,1].bar(time_by_type.index, time_by_type.values, color='green', alpha=0.7)
        axes[0,1].set_xlabel('Scenario Type')
        axes[0,1].set_ylabel('Execution Time (s)')
        axes[0,1].set_title('Execution Time by Type')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Box plot of separations by type
        type_sep_data = [df[df['scenario_type'] == t]['min_separation_nm'].values 
                        for t in df['scenario_type'].unique()]
        axes[1,0].boxplot(type_sep_data, labels=df['scenario_type'].unique())
        axes[1,0].set_xlabel('Scenario Type')
        axes[1,0].set_ylabel('Min Separation (NM)')
        axes[1,0].set_title('Separation Distribution by Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        success_by_type = df.groupby('scenario_type')['resolution_success'].mean()
        axes[1,1].bar(success_by_type.index, success_by_type.values, color='purple', alpha=0.7)
        axes[1,1].set_xlabel('Scenario Type')
        axes[1,1].set_ylabel('Resolution Success Rate')
        axes[1,1].set_title('Resolution Success by Type')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "performance_by_type.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_shift_impact(self, df: pd.DataFrame, fig_size: Tuple[int, int]):
        """Plot impact of distribution shift on performance"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Distribution Shift Impact', fontsize=16, fontweight='bold')
        
        # Accuracy across shifts
        acc_by_shift = df.groupby('distribution_shift_tier')['detection_accuracy'].mean()
        colors = ['green', 'orange', 'red']
        axes[0,0].bar(acc_by_shift.index, acc_by_shift.values, 
                     color=colors[:len(acc_by_shift)], alpha=0.7)
        axes[0,0].set_xlabel('Distribution Shift Level')
        axes[0,0].set_ylabel('Detection Accuracy')
        axes[0,0].set_title('Accuracy vs Distribution Shift')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # False positives/negatives by shift
        fp_by_shift = df.groupby('distribution_shift_tier')['false_positives'].mean()
        fn_by_shift = df.groupby('distribution_shift_tier')['false_negatives'].mean()
        x = range(len(fp_by_shift))
        width = 0.35
        axes[0,1].bar([i - width/2 for i in x], fp_by_shift.values, width, 
                     label='False Positives', color='red', alpha=0.7)
        axes[0,1].bar([i + width/2 for i in x], fn_by_shift.values, width, 
                     label='False Negatives', color='blue', alpha=0.7)
        axes[0,1].set_xlabel('Distribution Shift Level')
        axes[0,1].set_ylabel('Average Count')
        axes[0,1].set_title('FP/FN vs Distribution Shift')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(fp_by_shift.index, rotation=45)
        axes[0,1].legend()
        
        # Safety margin degradation
        safety_by_shift = df.groupby('distribution_shift_tier')['safety_margin_hz'].mean()
        axes[1,0].bar(safety_by_shift.index, safety_by_shift.values, 
                     color='purple', alpha=0.7)
        axes[1,0].set_xlabel('Distribution Shift Level')
        axes[1,0].set_ylabel('Average Safety Margin (NM)')
        axes[1,0].set_title('Safety Margin vs Distribution Shift')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Execution time impact
        time_by_shift = df.groupby('distribution_shift_tier')['execution_time_seconds'].mean()
        axes[1,1].bar(time_by_shift.index, time_by_shift.values, 
                     color='orange', alpha=0.7)
        axes[1,1].set_xlabel('Distribution Shift Level')
        axes[1,1].set_ylabel('Execution Time (s)')
        axes[1,1].set_title('Execution Time vs Distribution Shift')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "distribution_shift_impact.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save results in multiple formats"""
        
        # 1. Save detailed results as JSON
        results_data = [asdict(result) for result in self.results]
        with open(self.output_dir / "raw_data" / "detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # 2. Save summary CSV
        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "summaries" / "results_summary.csv", index=False)
        
        # 3. Save benchmark summary
        summary = self._generate_summary()
        with open(self.output_dir / "summaries" / "benchmark_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 4. Save configuration
        with open(self.output_dir / "summaries" / "configuration.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("MONTE CARLO BENCHMARK SUMMARY")
        print("="*60)
        print(f"Benchmark ID: {self.benchmark_id}")
        print(f"Total scenarios: {len(self.results)}")
        print(f"Successful scenarios: {summary['successful_scenarios']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"\nOverall Performance:")
        print(f"  Detection accuracy: {summary['overall_performance']['detection_accuracy']:.3f}")
        print(f"  Average separation: {summary['overall_performance']['avg_min_separation_nm']:.1f} NM")
        print(f"  Total violations: {summary['overall_performance']['total_violations']}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*60)


def run_benchmark_with_config(config_path: str = None) -> Dict[str, Any]:
    """
    Run Monte Carlo benchmark with configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Benchmark summary results
    """
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string enums back to objects
        if 'scenario_types' in config_dict:
            config_dict['scenario_types'] = [ScenarioType(t) for t in config_dict['scenario_types']]
        if 'complexity_tiers' in config_dict:
            config_dict['complexity_tiers'] = [ComplexityTier(t) for t in config_dict['complexity_tiers']]
        
        config = BenchmarkConfiguration(**config_dict)
    else:
        config = BenchmarkConfiguration()
    
    # Initialize and run benchmark
    benchmark = MonteCarloBenchmark(config)
    return benchmark.run()


def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monte Carlo Benchmark Runner for LLM-ATC-HAL')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--scenarios', '-n', type=int, default=10, 
                       help='Number of scenarios per type (default: 10)')
    parser.add_argument('--output', '-o', default='output/monte_carlo_benchmark',
                       help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    if args.config:
        summary = run_benchmark_with_config(args.config)
    else:
        config = BenchmarkConfiguration(
            num_scenarios_per_type=args.scenarios,
            output_directory=args.output
        )
        benchmark = MonteCarloBenchmark(config)
        summary = benchmark.run()
    
    print("\nBenchmark completed successfully!")
    return summary


if __name__ == "__main__":
    main()
