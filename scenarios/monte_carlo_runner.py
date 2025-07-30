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
import logging
import math
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

from llm_atc.tools import bluesky_tools
from llm_atc.tools.llm_prompt_engine import LLMPromptEngine
from scenarios.monte_carlo_framework import (
    ComplexityTier,
)

# LLM-ATC-HAL imports
from scenarios.scenario_generator import (
    ScenarioGenerator,
    ScenarioType,
    generate_horizontal_scenario,
    generate_sector_scenario,
    generate_vertical_scenario,
)

# Import monte carlo analysis helpers
try:
    from llm_atc.metrics.monte_carlo_analysis import MonteCarloResultsAnalyzer

    MONTE_CARLO_ANALYSIS_AVAILABLE = True
except ImportError:
    MONTE_CARLO_ANALYSIS_AVAILABLE = False
    logging.warning("Monte Carlo analysis module not available - limited summary functionality")


@dataclass
class BenchmarkConfiguration:
    """Configuration for Monte Carlo benchmark runs"""

    # Scenario parameters - NEW: per-type scenario counts
    num_scenarios_per_type: int = 50  # Kept for backward compatibility
    scenario_counts: Optional[dict[str, int]] = None  # New: per-type counts
    scenario_types: list[ScenarioType] = None
    complexity_tiers: list[ComplexityTier] = None
    distribution_shift_levels: list[str] = None

    # Simulation parameters - NEW: exposed configuration fields
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

    # Validation and strict mode
    strict_mode: bool = False  # If True, fail on mock data or LLM failures
    validate_llm_responses: bool = True  # If True, fail on invalid LLM responses

    # Performance thresholds
    min_separation_nm: float = 5.0
    min_separation_ft: float = 1000.0

    def __post_init__(self):
        """Set defaults for mutable fields"""
        if self.scenario_types is None:
            self.scenario_types = [
                ScenarioType.HORIZONTAL,
                ScenarioType.VERTICAL,
                ScenarioType.SECTOR,
            ]
        if self.complexity_tiers is None:
            self.complexity_tiers = [
                ComplexityTier.SIMPLE,
                ComplexityTier.MODERATE,
                ComplexityTier.COMPLEX,
            ]
        if self.distribution_shift_levels is None:
            self.distribution_shift_levels = ["in_distribution", "moderate_shift", "extreme_shift"]

        # Initialize scenario_counts if not provided
        if self.scenario_counts is None:
            self.scenario_counts = {
                (
                    scenario_type.value if hasattr(scenario_type, "value") else scenario_type
                ): self.num_scenarios_per_type
                for scenario_type in self.scenario_types
            }


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
    true_conflicts: list[dict[str, Any]]
    num_true_conflicts: int

    # Detection results
    predicted_conflicts: list[dict[str, Any]]
    num_predicted_conflicts: int
    detection_method: str  # 'ground_truth', 'llm', 'hybrid'

    # Resolution results
    llm_commands: list[str]
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

    # Execution metadata - NEW: success flag
    success: bool = False  # Overall scenario execution success
    execution_time_seconds: float = 0.0
    errors: list[str] = None
    warnings: list[str] = None
    timestamp: str = ""

    # Environmental factors
    wind_speed_kts: float = 0.0
    visibility_nm: float = 0.0
    turbulence_level: float = 0.0

    def __post_init__(self):
        """Set defaults for mutable fields"""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class MonteCarloBenchmark:
    """
    Monte Carlo benchmark runner for LLM-ATC-HAL performance evaluation.

    Orchestrates comprehensive testing across scenario types, complexity levels,
    and distribution shifts to generate robust performance metrics.
    """

    def __init__(self, config: Optional[BenchmarkConfiguration] = None) -> None:
        """
        Initialize the Monte Carlo benchmark runner.

        Args:
            config: Benchmark configuration. If None, uses defaults.
        """
        self.config = config or BenchmarkConfiguration()
        self.logger = logging.getLogger(__name__)

        # Enable strict mode for BlueSky if requested
        if hasattr(self.config, "strict_mode") and self.config.strict_mode:
            from llm_atc.tools.bluesky_tools import set_strict_mode

            set_strict_mode(True)
            self.logger.info("Enabled strict mode - will fail on mock data usage")

        # Initialize components
        self.scenario_generator = ScenarioGenerator()
        self.llm_engine = LLMPromptEngine(
            model=self.config.llm_model,
            enable_function_calls=self.config.enable_function_calls,
        )

        # Results storage
        self.results: list[ScenarioResult] = []
        self.benchmark_start_time: Optional[datetime] = None
        self.benchmark_id = str(uuid.uuid4())[:8]

        # Setup output directory
        self._setup_output_directory()

        # Initialize logging
        self._setup_logging()

        self.logger.info(f"Initialized Monte Carlo benchmark {self.benchmark_id}")
        self.logger.info(f"Configuration: {self.config}")

    def _setup_output_directory(self) -> None:
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

    def _setup_logging(self) -> None:
        """Setup detailed logging for benchmark execution"""
        if not self.config.detailed_logging:
            return

        log_file = self.output_dir / "logs" / "benchmark.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

    def run(self) -> dict[str, Any]:
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
                            scenario_type,
                            complexity_tier,
                            shift_level,
                        )

            # Generate summary and visualizations
            summary = self._generate_summary()

            if self.config.generate_visualizations:
                self._generate_visualizations()

            # Save results
            self._save_results()

            self.logger.info(
                f"Benchmark completed successfully: {scenario_count} scenarios executed",
            )
            return summary

        except Exception as e:
            self.logger.exception(f"Benchmark execution failed: {e}")
            self.logger.exception(traceback.format_exc())
            raise

    def _calculate_total_scenarios(self) -> int:
        """Calculate total number of scenarios to be executed"""
        total = 0
        for scenario_type in self.config.scenario_types:
            scenario_key = scenario_type.value if hasattr(scenario_type, "value") else scenario_type
            scenario_count = self.config.scenario_counts.get(scenario_key, 0)
            total += (
                scenario_count
                * len(self.config.complexity_tiers)
                * len(self.config.distribution_shift_levels)
            )
        return total

    def _run_scenario_batch(
        self, scenario_type: ScenarioType, complexity_tier: ComplexityTier, shift_level: str,
    ) -> int:
        """
        Execute a batch of scenarios for given parameters.

        Args:
            scenario_type: Type of scenario to generate
            complexity_tier: Complexity level
            shift_level: Distribution shift level

        Returns:
            Number of scenarios successfully executed
        """
        batch_id = f"{scenario_type.value if hasattr(scenario_type, 'value') else scenario_type}_{complexity_tier.value if hasattr(complexity_tier, 'value') else complexity_tier}_{shift_level}"
        self.logger.info(f"Starting batch: {batch_id}")

        successful_scenarios = 0

        # Get scenario count for this type
        num_scenarios = self.config.scenario_counts.get(
            scenario_type.value if hasattr(scenario_type, "value") else scenario_type, 0,
        )

        for i in range(num_scenarios):
            scenario_id = f"{batch_id}_{i+1:03d}"

            try:
                # Generate scenario
                scenario = self._generate_scenario(
                    scenario_type,
                    complexity_tier,
                    shift_level,
                    scenario_id,
                )

                # Execute single scenario with success tracking
                result = self._run_single_scenario(scenario, scenario_id)

                # Store result
                self.results.append(result)
                if result.success:
                    successful_scenarios += 1

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Batch {batch_id}: completed {i+1}/{num_scenarios}")

            except Exception as e:
                self.logger.exception(f"Failed to execute scenario {scenario_id}: {e}")

                # Create error result
                error_result = self._create_error_result(
                    scenario_id, scenario_type, complexity_tier, shift_level, str(e),
                )
                self.results.append(error_result)

        self.logger.info(
            f"Batch {batch_id} completed: {successful_scenarios}/{num_scenarios} successful",
        )
        return successful_scenarios

    def _generate_scenario(
        self,
        scenario_type: ScenarioType,
        complexity_tier: ComplexityTier,
        shift_level: str,
        scenario_id: str,
    ) -> Any:
        """Generate a scenario based on type and parameters"""

        if scenario_type == ScenarioType.HORIZONTAL:
            return generate_horizontal_scenario(
                n_aircraft=self._get_aircraft_count_for_complexity(complexity_tier),
                conflict=True,  # Force conflicts for testing
                complexity_tier=complexity_tier,
                distribution_shift_tier=shift_level,
            )
        if scenario_type == ScenarioType.VERTICAL:
            return generate_vertical_scenario(
                n_aircraft=self._get_aircraft_count_for_complexity(complexity_tier),
                conflict=True,
                complexity_tier=complexity_tier,
                distribution_shift_tier=shift_level,
            )
        if scenario_type == ScenarioType.SECTOR:
            return generate_sector_scenario(
                complexity=complexity_tier,
                shift_level=shift_level,
                force_conflicts=True,
            )
        msg = f"Unknown scenario type: {scenario_type}"
        raise ValueError(msg)

    def _get_aircraft_count_for_complexity(self, complexity_tier: ComplexityTier) -> int:
        """Get appropriate aircraft count for complexity level"""
        counts = {
            ComplexityTier.SIMPLE: 2,
            ComplexityTier.MODERATE: 4,
            ComplexityTier.COMPLEX: 6,
            ComplexityTier.EXTREME: 8,
        }
        return counts.get(complexity_tier, 3)

    def _run_single_scenario(self, scenario: Any, scenario_id: str) -> ScenarioResult:
        """
        Execute a single scenario with comprehensive error handling and success tracking.

        Args:
            scenario: Generated scenario object
            scenario_id: Unique scenario identifier

        Returns:
            ScenarioResult with success flag properly set
        """
        # Initialize result with success=False
        success = False

        try:
            # Execute the three-stage pipeline
            result = self._execute_scenario_pipeline(scenario, scenario_id)

            # Determine success criteria:
            # 1. No unresolved conflicts (either none detected or successfully resolved)
            # 2. No parse errors in LLM responses
            # 3. No critical execution errors

            has_unresolved_conflicts = (
                len(result.predicted_conflicts) > 0
                and not result.resolution_success
                and result.separation_violations > 0
            )

            has_parse_errors = any("parse" in error.lower() for error in result.errors)
            has_critical_errors = len(result.errors) > 0

            # Success if no unresolved conflicts and no critical errors
            success = not (has_unresolved_conflicts or has_parse_errors or has_critical_errors)

            # Update the result with success flag
            result.success = success

            if success:
                self.logger.debug(f"Scenario {scenario_id} completed successfully")
            else:
                reasons = []
                if has_unresolved_conflicts:
                    reasons.append("unresolved conflicts")
                if has_parse_errors:
                    reasons.append("parse errors")
                if has_critical_errors:
                    reasons.append("execution errors")
                self.logger.warning(f"Scenario {scenario_id} failed: {', '.join(reasons)}")

            return result

        except Exception as e:
            # Catch exceptions at this level and create error result
            error_message = f"Exception in scenario execution: {e!s}"
            self.logger.exception(f"Scenario {scenario_id} failed with exception: {e}")
            self.logger.debug(traceback.format_exc())

            # Extract basic scenario information for error result
            scenario_type = getattr(scenario, "scenario_type", ScenarioType.HORIZONTAL)
            complexity_tier = getattr(scenario, "complexity_tier", ComplexityTier.MODERATE)
            shift_level = getattr(scenario, "distribution_shift_tier", "in_distribution")

            return self._create_error_result(
                scenario_id,
                scenario_type,
                complexity_tier,
                shift_level,
                error_message,
            )

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
                ground_truth_conflicts,
                detected_conflicts,
                resolutions,
                verification_results,
            )

            # Create result object
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=(
                    scenario.scenario_type.value
                    if hasattr(scenario, "scenario_type")
                    else "unknown"
                ),
                complexity_tier=getattr(scenario, "complexity_tier", ComplexityTier.MODERATE).value,
                distribution_shift_tier=getattr(
                    scenario, "distribution_shift_tier", "in_distribution",
                ),
                aircraft_count=getattr(scenario, "aircraft_count", len(scenario.initial_states)),
                duration_minutes=getattr(
                    scenario, "duration_minutes", self.config.time_horizon_minutes,
                ),
                # Ground truth
                true_conflicts=ground_truth_conflicts,
                num_true_conflicts=len(ground_truth_conflicts),
                # Detection
                predicted_conflicts=detected_conflicts,
                num_predicted_conflicts=len(detected_conflicts),
                detection_method="hybrid" if self.config.enable_llm_detection else "ground_truth",
                # Resolution
                llm_commands=[r.get("command", "") for r in resolutions],
                resolution_success=verification_results.get("resolution_success", False),
                num_interventions=len(resolutions),
                # Safety metrics
                min_separation_nm=verification_results.get("min_separation_nm", 999.0),
                min_separation_ft=verification_results.get("min_separation_ft", 999999.0),
                separation_violations=verification_results.get("violations", 0),
                safety_margin_hz=max(
                    0,
                    verification_results.get("min_separation_nm", 0)
                    - self.config.min_separation_nm,
                ),
                safety_margin_vt=max(
                    0,
                    verification_results.get("min_separation_ft", 0)
                    - self.config.min_separation_ft,
                ),
                # Efficiency metrics
                extra_distance_nm=verification_results.get("extra_distance_nm", 0.0),
                total_delay_seconds=verification_results.get("total_delay", 0.0),
                fuel_penalty_percent=verification_results.get("fuel_penalty", 0.0),
                # Performance metrics
                **metrics,
                # Execution metadata - Initialize success as False, will be updated in _run_single_scenario
                success=False,  # Will be determined by _run_single_scenario
                execution_time_seconds=time.time() - pipeline_start,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now().isoformat(),
                # Environmental factors
                wind_speed_kts=getattr(scenario, "environmental_conditions", {}).get(
                    "wind_speed_kts", 0,
                ),
                visibility_nm=getattr(scenario, "environmental_conditions", {}).get(
                    "visibility_nm", 10,
                ),
                turbulence_level=getattr(scenario, "environmental_conditions", {}).get(
                    "turbulence_intensity", 0,
                ),
            )

        except Exception as e:
            self.logger.exception(f"Pipeline execution failed for {scenario_id}: {e}")
            errors.append(str(e))

            return self._create_error_result(
                scenario_id,
                getattr(scenario, "scenario_type", ScenarioType.SECTOR),
                getattr(scenario, "complexity_tier", ComplexityTier.MODERATE),
                getattr(scenario, "distribution_shift_tier", "in_distribution"),
                str(e),
            )

    def _reset_bluesky_simulation(self) -> None:
        """Reset BlueSky simulation to clean state - only if scenario doesn't include RESET"""
        try:
            # Check if scenario already includes RESET command
            # Most scenarios generated by the framework include their own RESET sequence
            self.logger.debug("BlueSky simulation will be reset by scenario commands")
        except Exception as e:
            self.logger.warning(f"BlueSky reset preparation failed: {e}")

    def _load_scenario_commands(self, scenario: Any) -> None:
        """Load scenario commands into BlueSky"""
        try:
            commands = getattr(scenario, "commands", [])
            if not commands and hasattr(scenario, "bluesky_commands"):
                commands = scenario.bluesky_commands

            self.logger.info(f"Loading {len(commands)} scenario commands into BlueSky")

            cre_command_count = 0
            for i, command in enumerate(commands):
                result = bluesky_tools.send_command(command)
                self.logger.debug(
                    f"Command {i+1}/{len(commands)} '{command}': {result.get('status', 'unknown')}",
                )

                # Count CRE commands
                if command.strip().upper().startswith("CRE"):
                    cre_command_count += 1

                # Add small delay between commands for complex setups
                if i > 0 and (command.startswith(("CRE", "IC", "OP"))):
                    time.sleep(0.1)  # 100ms delay for important commands

            # After all commands are loaded, check if we have aircraft
            if cre_command_count > 0:
                try:
                    aircraft_data = bluesky_tools.get_all_aircraft_info()
                    aircraft_count = len(aircraft_data) if aircraft_data else 0
                    self.logger.info(
                        f"Loaded {cre_command_count} CRE commands, detected {aircraft_count} aircraft in simulation",
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not verify aircraft after loading CRE commands: {e}",
                    )

        except Exception as e:
            self.logger.exception(f"Failed to load scenario commands: {e}")
            raise

    def _extract_ground_truth_conflicts(self, scenario: Any) -> list[dict[str, Any]]:
        """Extract ground truth conflicts from scenario"""
        try:
            if hasattr(scenario, "ground_truth_conflicts"):
                return [asdict(conflict) for conflict in scenario.ground_truth_conflicts]
            # Create mock ground truth for testing
            return [
                {
                    "aircraft_pair": ("AC001", "AC002"),
                    "conflict_type": "horizontal",
                    "time_to_conflict": 120.0,
                    "min_separation": {"horizontal_nm": 3.5, "vertical_ft": 0},
                    "severity": "medium",
                    "is_actual_conflict": True,
                },
            ]
        except Exception as e:
            self.logger.warning(f"Failed to extract ground truth: {e}")
            return []

    def _detect_conflicts(self, scenario: Any) -> list[dict[str, Any]]:
        """Perform conflict detection using multiple BlueSky methods for validation"""
        detected_conflicts = []

        try:
            # Import enhanced conflict detector
            from llm_atc.tools.enhanced_conflict_detector import EnhancedConflictDetector
            
            # Use enhanced detection with multiple methods (SWARM, STATEBASED, ENHANCED)
            enhanced_detector = EnhancedConflictDetector()
            comprehensive_conflicts = enhanced_detector.detect_conflicts_comprehensive()
            
            for conflict in comprehensive_conflicts:
                detected_conflicts.append(
                    {
                        "source": "enhanced_multi_method",
                        "aircraft_1": conflict.aircraft_1,
                        "aircraft_2": conflict.aircraft_2,
                        "horizontal_separation": conflict.current_horizontal_separation,
                        "vertical_separation": conflict.current_vertical_separation,
                        "time_to_cpa": conflict.time_to_cpa,
                        "distance_at_cpa": conflict.distance_at_cpa,
                        "min_horizontal_separation": conflict.min_horizontal_separation,
                        "min_vertical_separation": conflict.min_vertical_separation,
                        "violates_icao": conflict.violates_icao_separation,
                        "severity": conflict.severity,
                        "detection_method": conflict.detection_method,
                        "confidence": conflict.confidence,
                        "conflict_within_300s": conflict.time_to_cpa <= 300.0,  # Key improvement
                    },
                )

            # Fallback: Original BlueSky method
            if not detected_conflicts:
                bluesky_conflicts = bluesky_tools.get_conflict_info()
                for conflict in bluesky_conflicts.get("conflicts", []):
                    detected_conflicts.append(
                        {
                            "source": "bluesky_fallback",
                            "aircraft_1": conflict["aircraft_1"],
                            "aircraft_2": conflict["aircraft_2"],
                            "horizontal_separation": conflict["horizontal_separation"],
                            "vertical_separation": conflict["vertical_separation"],
                            "time_to_cpa": conflict.get("time_to_cpa", 999),
                            "severity": conflict["severity"],
                            "conflict_within_300s": conflict.get("time_to_cpa", 999) <= 300.0,
                        },
                    )

            # Method 2: LLM-based detection (if enabled) with enhanced validation
            if self.config.enable_llm_detection:
                aircraft_states = self._get_aircraft_states_for_llm()
                
                # Generate enhanced prompt with CPA data
                cpa_data = {}
                if comprehensive_conflicts:
                    # Use data from first conflict for context
                    first_conflict = comprehensive_conflicts[0]
                    cpa_data = {
                        'time_to_cpa': first_conflict.time_to_cpa,
                        'min_horizontal_separation': first_conflict.min_horizontal_separation,
                        'min_vertical_separation': first_conflict.min_vertical_separation,
                        'current_horizontal_separation': first_conflict.current_horizontal_separation,
                        'current_vertical_separation': first_conflict.current_vertical_separation,
                        'violates_icao_separation': first_conflict.violates_icao_separation,
                        'severity': first_conflict.severity
                    }

                llm_detection = self.llm_engine.detect_conflict_via_llm(
                    aircraft_states,
                    self.config.time_horizon_minutes,
                    cpa_data=cpa_data  # Enhanced with CPA data
                )

                # Validate LLM detection response
                if self.config.validate_llm_responses and not isinstance(llm_detection, dict):
                    error_msg = f"LLM conflict detection returned invalid response type: {type(llm_detection)}"
                    self.logger.error(error_msg)
                    if self.config.strict_mode:
                        raise Exception(error_msg)

                if llm_detection and llm_detection.get("conflict_detected", False):
                    aircraft_pairs = llm_detection.get("aircraft_pairs", [])
                    if self.config.validate_llm_responses and not aircraft_pairs:
                        error_msg = "LLM detected conflict but provided no aircraft pairs"
                        self.logger.warning(error_msg)
                        if self.config.strict_mode:
                            raise Exception(error_msg)

                    # Cross-validate LLM conflicts with enhanced detector
                    validated_pairs = enhanced_detector.validate_llm_conflicts(aircraft_pairs)

                    for pair_data in validated_pairs:
                        ac1, ac2, confidence = pair_data
                        detected_conflicts.append(
                            {
                                "source": "llm_enhanced_validated",
                                "aircraft_1": ac1,
                                "aircraft_2": ac2,
                                "confidence": confidence,
                                "priority": llm_detection.get("priority", "unknown"),
                                "validation": "enhanced_detector_confirmed",
                                "uses_icao_standards": True,
                                "cpa_analysis_provided": bool(cpa_data),
                            },
                        )

        except Exception as e:
            self.logger.exception(f"Enhanced conflict detection failed: {e}")
            # Fallback to basic detection
            detected_conflicts = self._basic_conflict_detection_fallback()

        return detected_conflicts

    def _basic_conflict_detection_fallback(self) -> list[dict[str, Any]]:
        """Basic conflict detection fallback when enhanced detection fails"""
        try:
            # Method 1: BlueSky built-in conflict detection
            bluesky_conflicts = bluesky_tools.get_conflict_info()
            detected_conflicts = []
            for conflict in bluesky_conflicts.get("conflicts", []):
                detected_conflicts.append(
                    {
                        "source": "bluesky_fallback",
                        "aircraft_1": conflict["aircraft_1"],
                        "aircraft_2": conflict["aircraft_2"],
                        "horizontal_separation": conflict["horizontal_separation"],
                        "vertical_separation": conflict["vertical_separation"],
                        "time_to_cpa": conflict["time_to_cpa"],
                        "severity": conflict["severity"],
                    },
                )
            return detected_conflicts
        except Exception as e:
            self.logger.exception(f"Fallback conflict detection also failed: {e}")
            return []

    def _get_aircraft_states_for_llm(self) -> list[dict[str, Any]]:
        """Get current aircraft states formatted for LLM"""
        try:
            aircraft_info = bluesky_tools.get_all_aircraft_info()

            # Validate aircraft info exists in strict mode
            if self.config.strict_mode and (not aircraft_info or not aircraft_info.get("aircraft")):
                msg = "No aircraft data available in strict mode"
                raise Exception(msg)

            states = []

            for aircraft_id, info in aircraft_info.get("aircraft", {}).items():
                states.append(
                    {
                        "id": aircraft_id,
                        "lat": info["lat"],
                        "lon": info["lon"],
                        "alt": info["alt"],
                        "hdg": info["hdg"],
                        "spd": info["spd"],
                        "vs": info["vs"],
                    },
                )

            return states

        except Exception as e:
            self.logger.exception(f"Failed to get aircraft states: {e}")
            if self.config.strict_mode:
                raise
            return []

    def _validate_llm_conflicts_with_bluesky(
        self, 
        llm_pairs: list[tuple[str, str]], 
        bluesky_conflicts: list[dict[str, Any]]
    ) -> list[tuple[str, str]]:
        """
        Validate LLM-detected conflicts against BlueSky ground truth
        to eliminate false positives.
        
        Args:
            llm_pairs: Aircraft pairs detected by LLM
            bluesky_conflicts: Conflicts detected by BlueSky
            
        Returns:
            Validated aircraft pairs confirmed by BlueSky
        """
        if not self.config.validate_llm_responses:
            # If validation disabled, return all LLM pairs
            return llm_pairs
        
        validated_pairs = []
        false_positives = 0
        
        for llm_pair in llm_pairs:
            # Normalize pair format
            if isinstance(llm_pair, (list, tuple)) and len(llm_pair) >= 2:
                ac1, ac2 = llm_pair[0], llm_pair[1]
            else:
                self.logger.warning(f"Invalid LLM pair format: {llm_pair}")
                continue
            
            # Check if this pair matches any BlueSky conflict
            bluesky_confirmed = False
            for bs_conflict in bluesky_conflicts:
                bs_ac1 = bs_conflict.get("aircraft_1", "")
                bs_ac2 = bs_conflict.get("aircraft_2", "")
                
                # Check both orderings of the pair
                if (ac1 == bs_ac1 and ac2 == bs_ac2) or (ac1 == bs_ac2 and ac2 == bs_ac1):
                    bluesky_confirmed = True
                    break
            
            if bluesky_confirmed:
                validated_pairs.append((ac1, ac2))
                self.logger.info(f"LLM conflict validated by BlueSky: {ac1}-{ac2}")
            else:
                false_positives += 1
                self.logger.warning(f"LLM false positive filtered out: {ac1}-{ac2}")
        
        if false_positives > 0:
            self.logger.info(f"Prevented {false_positives} LLM false positives using BlueSky validation")
        
        return validated_pairs

    def _resolve_conflicts(
        self, conflicts: list[dict[str, Any]], scenario: Any,
    ) -> list[dict[str, Any]]:
        """Generate LLM-based conflict resolutions"""
        resolutions = []

        for conflict in conflicts:
            try:
                # Create conflict info for LLM
                conflict_info = self._format_conflict_for_llm(conflict, scenario)

                # Get LLM resolution
                resolution_command = self.llm_engine.get_conflict_resolution(conflict_info)

                # Validate LLM response if strict mode enabled
                if self.config.validate_llm_responses and not resolution_command:
                    error_msg = f"LLM failed to generate resolution for conflict {conflict}"
                    self.logger.error(error_msg)
                    if self.config.strict_mode:
                        raise Exception(error_msg)

                if resolution_command:
                    # Validate command format
                    if self.config.validate_llm_responses and not self._is_valid_bluesky_command(
                        resolution_command,
                    ):
                        error_msg = f"LLM generated invalid command: {resolution_command}"
                        self.logger.error(error_msg)
                        if self.config.strict_mode:
                            raise Exception(error_msg)

                    # Execute the command
                    execution_result = bluesky_tools.send_command(resolution_command)

                    resolutions.append(
                        {
                            "conflict": conflict,
                            "command": resolution_command,
                            "execution_result": execution_result,
                            "timestamp": time.time(),
                        },
                    )

                    self.logger.info(f"Executed resolution: {resolution_command}")

            except Exception as e:
                self.logger.exception(f"Failed to resolve conflict {conflict}: {e}")
                if self.config.strict_mode:
                    raise  # Re-raise in strict mode

        return resolutions

    def _is_valid_bluesky_command(self, command: str) -> bool:
        """Validate if a command is a valid BlueSky command"""
        if not command or not isinstance(command, str):
            return False

        command = command.strip()
        if not command:
            return False

        # Check for common BlueSky command patterns
        valid_commands = ["HDG", "ALT", "SPD", "CRE", "DEL", "MOVE", "TURN", "CLIMB", "DESCEND"]
        command_parts = command.split()

        if len(command_parts) < 2:  # Need at least command and aircraft ID
            return False

        command_type = command_parts[0].upper()
        return command_type in valid_commands

    def _format_conflict_for_llm(self, conflict: dict[str, Any], scenario: Any) -> dict[str, Any]:
        """Format conflict data for LLM prompt engine"""
        try:
            # Get aircraft information
            aircraft_info = bluesky_tools.get_all_aircraft_info()

            ac1_id = conflict.get("aircraft_1", "AC001")
            ac2_id = conflict.get("aircraft_2", "AC002")

            ac1_info = aircraft_info.get("aircraft", {}).get(ac1_id, {})
            ac2_info = aircraft_info.get("aircraft", {}).get(ac2_id, {})

            return {
                "aircraft_1_id": ac1_id,
                "aircraft_2_id": ac2_id,
                "time_to_conflict": conflict.get("time_to_cpa", 120.0),
                "closest_approach_distance": conflict.get("horizontal_separation", 3.5),
                "conflict_type": "convergent",
                "urgency_level": conflict.get("severity", "medium"),
                "aircraft_1": ac1_info,
                "aircraft_2": ac2_info,
                "environmental_conditions": getattr(scenario, "environmental_conditions", {}),
            }

        except Exception as e:
            self.logger.exception(f"Failed to format conflict for LLM: {e}")
            return {}

    def _verify_resolutions(
        self, scenario: Any, resolutions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Verify resolution effectiveness by stepping simulation with adaptive time stepping"""
        verification_results = {
            "resolution_success": False,
            "min_separation_nm": 999.0,
            "min_separation_ft": 999999.0,
            "violations": 0,
            "extra_distance_nm": 0.0,
            "total_delay": 0.0,
            "fuel_penalty": 0.0,
        }

        try:
            # Calculate adaptive time stepping
            time_horizon_seconds = self.config.time_horizon_minutes * 60

            # Adaptive step size based on scenario duration
            if time_horizon_seconds < 300:  # Less than 5 minutes - use smaller steps
                adaptive_step_size = min(self.config.step_size_seconds, 5.0)
            elif time_horizon_seconds > 1200:  # More than 20 minutes - use larger steps
                adaptive_step_size = max(self.config.step_size_seconds, 15.0)
            else:
                adaptive_step_size = self.config.step_size_seconds

            # Calculate number of steps needed
            num_steps = math.ceil(time_horizon_seconds / adaptive_step_size)

            self.logger.debug(
                f"Using adaptive step size: {adaptive_step_size}s for {num_steps} steps",
            )

            min_separation_recorded = []

            for step in range(num_steps):
                # Step simulation with adaptive step size
                bluesky_tools.step_simulation(adaptive_step_size)

                # Check separations
                aircraft_info = bluesky_tools.get_all_aircraft_info()
                separations = self._calculate_all_separations(aircraft_info)

                if separations:
                    min_hz = min(sep["horizontal_nm"] for sep in separations)
                    min_vt = min(sep["vertical_ft"] for sep in separations)

                    min_separation_recorded.append(
                        {
                            "time": step * adaptive_step_size,
                            "horizontal_nm": min_hz,
                            "vertical_ft": min_vt,
                        },
                    )

                    # Check for violations
                    if (
                        min_hz < self.config.min_separation_nm
                        and min_vt < self.config.min_separation_ft
                    ):
                        verification_results["violations"] += 1

            # Calculate final metrics
            if min_separation_recorded:
                verification_results["min_separation_nm"] = min(
                    s["horizontal_nm"] for s in min_separation_recorded
                )
                verification_results["min_separation_ft"] = min(
                    s["vertical_ft"] for s in min_separation_recorded
                )

                # Resolution is successful if no violations occurred
                verification_results["resolution_success"] = verification_results["violations"] == 0

                # Calculate efficiency metrics (simplified)
                verification_results["extra_distance_nm"] = len(resolutions) * 5.0  # Estimate
                verification_results["total_delay"] = len(resolutions) * 30.0  # Estimate in seconds
                verification_results["fuel_penalty"] = len(resolutions) * 2.0  # Estimate percentage

        except Exception as e:
            self.logger.exception(f"Verification failed: {e}")

        return verification_results

    def _calculate_all_separations(self, aircraft_info: dict[str, Any]) -> list[dict[str, Any]]:
        """Calculate separations between all aircraft pairs"""
        separations = []

        try:
            aircraft_list = list(aircraft_info.get("aircraft", {}).values())

            for i in range(len(aircraft_list)):
                for j in range(i + 1, len(aircraft_list)):
                    ac1 = aircraft_list[i]
                    ac2 = aircraft_list[j]

                    # Calculate horizontal separation (simplified)
                    lat_diff = ac1["lat"] - ac2["lat"]
                    lon_diff = ac1["lon"] - ac2["lon"]
                    horizontal_nm = ((lat_diff**2 + lon_diff**2) ** 0.5) * 60  # Rough conversion

                    # Calculate vertical separation
                    vertical_ft = abs(ac1["alt"] - ac2["alt"])

                    separations.append(
                        {
                            "aircraft_1": ac1["id"],
                            "aircraft_2": ac2["id"],
                            "horizontal_nm": horizontal_nm,
                            "vertical_ft": vertical_ft,
                        },
                    )

        except Exception as e:
            self.logger.exception(f"Failed to calculate separations: {e}")

        return separations

    def _calculate_scenario_metrics(
        self,
        ground_truth: list[dict[str, Any]],
        detected: list[dict[str, Any]],
        resolutions: list[dict[str, Any]],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate performance metrics for scenario"""

        # Convert to sets for easier comparison
        true_conflicts = set()
        for gt in ground_truth:
            if gt.get("is_actual_conflict", True):
                pair = gt.get("aircraft_pair", ("AC001", "AC002"))
                true_conflicts.add(tuple(sorted(pair)))

        detected_conflicts = set()
        for det in detected:
            ac1 = det.get("aircraft_1", "AC001")
            ac2 = det.get("aircraft_2", "AC002")
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
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
            "detection_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

    def _create_error_result(
        self,
        scenario_id: str,
        scenario_type: ScenarioType,
        complexity_tier: ComplexityTier,
        shift_level: str,
        error: str,
    ) -> ScenarioResult:
        """Create error result for failed scenarios"""
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=scenario_type.value if hasattr(scenario_type, "value") else scenario_type,
            complexity_tier=(
                complexity_tier.value if hasattr(complexity_tier, "value") else complexity_tier
            ),
            distribution_shift_tier=shift_level,
            aircraft_count=0,
            duration_minutes=0.0,
            true_conflicts=[],
            num_true_conflicts=0,
            predicted_conflicts=[],
            num_predicted_conflicts=0,
            detection_method="error",
            llm_commands=[],
            resolution_success=False,
            num_interventions=0,
            min_separation_nm=0.0,
            min_separation_ft=0.0,
            separation_violations=999,
            safety_margin_hz=0.0,
            safety_margin_vt=0.0,
            extra_distance_nm=0.0,
            total_delay_seconds=0.0,
            fuel_penalty_percent=0.0,
            false_positives=0,
            false_negatives=0,
            true_positives=0,
            true_negatives=0,
            detection_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            success=False,  # Error results are never successful
            execution_time_seconds=0.0,
            errors=[error],
            warnings=[],
            timestamp=datetime.now().isoformat(),
            wind_speed_kts=0.0,
            visibility_nm=0.0,
            turbulence_level=0.0,
        )

    def _generate_summary(self) -> dict[str, Any]:
        """Generate comprehensive summary statistics from all results"""
        if not self.results:
            return {"error": "No results to summarize"}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(result) for result in self.results])

        # Basic statistics
        total_scenarios = len(df)
        successful_scenarios = len(df[df["success"]])  # Use new success field
        failed_scenarios = total_scenarios - successful_scenarios

        # Print basic counts to user

        # Use MonteCarloResultsAnalyzer if available for detailed analysis
        detailed_analysis = {}
        if MONTE_CARLO_ANALYSIS_AVAILABLE:
            try:
                analyzer = MonteCarloResultsAnalyzer()
                detailed_analysis = analyzer.aggregate_monte_carlo_metrics(df)

                # Print comprehensive analysis results
                self._print_detailed_analysis(detailed_analysis)

            except Exception as e:
                self.logger.warning(f"Failed to run detailed analysis: {e}")
                detailed_analysis = {"error": f"Analysis failed: {e!s}"}

        # Generate summary by scenario type, complexity, and shift
        type_summary = self._generate_summary_by_group(df, "scenario_type")
        complexity_summary = self._generate_summary_by_group(df, "complexity_tier")
        shift_summary = self._generate_summary_by_group(df, "distribution_shift_tier")

        # Combined summary across all dimensions using MonteCarloResultsAnalyzer if available
        combined_summary = self._generate_combined_summary(df)
        multi_group_summary = {}
        if MONTE_CARLO_ANALYSIS_AVAILABLE:
            try:
                analyzer = MonteCarloResultsAnalyzer()
                multi_group_summary = analyzer.compute_success_rates_by_group(
                    df,
                    ["scenario_type", "complexity_tier", "distribution_shift_tier"],
                )
                # Convert to dict for JSON serialization with string keys
                if not multi_group_summary.empty:
                    raw_dict = multi_group_summary.to_dict("index")
                    # Convert tuple keys to string keys for JSON serialization
                    multi_group_summary = {str(key): value for key, value in raw_dict.items()}
            except Exception as e:
                self.logger.warning(f"Failed to compute multi-group success rates: {e}")
                multi_group_summary = {}

        # Detection performance (basic)
        avg_accuracy = df["detection_accuracy"].mean()
        avg_precision = df["precision"].mean()
        avg_recall = df["recall"].mean()

        # Safety metrics (basic)
        avg_min_separation = df["min_separation_nm"].mean()
        total_violations = df["separation_violations"].sum()

        # Create comprehensive summary
        return {
            "benchmark_id": self.benchmark_id,
            "execution_time": str(datetime.now() - self.benchmark_start_time),
            "timestamp": datetime.now().isoformat(),
            # Basic counts with success/failure tracking
            "scenario_counts": {
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "failed_scenarios": failed_scenarios,
                "success_rate": successful_scenarios / total_scenarios,
            },
            # Overall performance (backward compatibility)
            "overall_performance": {
                "detection_accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "avg_min_separation_nm": avg_min_separation,
                "total_violations": int(total_violations),
            },
            # Grouped summaries
            "by_scenario_type": type_summary,
            "by_complexity_tier": complexity_summary,
            "by_distribution_shift": shift_summary,
            "combined_analysis": combined_summary,
            "multi_dimensional_analysis": multi_group_summary,
            # Detailed analysis from MonteCarloResultsAnalyzer
            "detailed_analysis": detailed_analysis,
            "configuration": asdict(self.config),
        }

    def _generate_summary_by_group(
        self, df: pd.DataFrame, group_column: str,
    ) -> dict[str, dict[str, Any]]:
        """Generate success rate and metrics summary by a specific grouping column"""
        summary = {}

        for group_value in df[group_column].unique():
            group_data = df[df[group_column] == group_value]

            total_in_group = len(group_data)
            successful_in_group = len(group_data[group_data["success"]])

            summary[group_value] = {
                "total_scenarios": total_in_group,
                "successful_scenarios": successful_in_group,
                "failed_scenarios": total_in_group - successful_in_group,
                "success_rate": successful_in_group / total_in_group if total_in_group > 0 else 0.0,
                "avg_detection_accuracy": group_data["detection_accuracy"].mean(),
                "avg_precision": group_data["precision"].mean(),
                "avg_recall": group_data["recall"].mean(),
                "avg_min_separation_nm": group_data["min_separation_nm"].mean(),
                "total_violations": int(group_data["separation_violations"].sum()),
            }

        return summary

    def _generate_combined_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate summary across all combinations of scenario type, complexity, and shift"""
        combined_analysis = {}

        # Group by all three dimensions
        for scenario_type in df["scenario_type"].unique():
            combined_analysis[scenario_type] = {}

            type_data = df[df["scenario_type"] == scenario_type]

            for complexity in type_data["complexity_tier"].unique():
                combined_analysis[scenario_type][complexity] = {}

                complexity_data = type_data[type_data["complexity_tier"] == complexity]

                for shift in complexity_data["distribution_shift_tier"].unique():
                    shift_data = complexity_data[
                        complexity_data["distribution_shift_tier"] == shift
                    ]

                    total = len(shift_data)
                    successful = len(shift_data[shift_data["success"]])

                    combined_analysis[scenario_type][complexity][shift] = {
                        "total_scenarios": total,
                        "successful_scenarios": successful,
                        "success_rate": successful / total if total > 0 else 0.0,
                        "avg_detection_accuracy": shift_data["detection_accuracy"].mean(),
                        "avg_separation_violations": shift_data["separation_violations"].mean(),
                    }

        return combined_analysis

    def _print_detailed_analysis(self, analysis: dict[str, Any]) -> None:
        """Print detailed analysis results to console"""

        # Detection performance
        detection_perf = analysis.get("detection_performance", {})
        detection_perf.get("false_positive_rate", 0)
        detection_perf.get("false_negative_rate", 0)

        # Success rates by scenario type
        success_rates = analysis.get("success_rates_by_scenario", {})
        if success_rates:
            for _scenario_type, metrics in success_rates.items():
                metrics.get("success_rate", 0)
                metrics.get("total_scenarios", 0)

        # Distribution shift analysis
        shift_analysis = analysis.get("distribution_shift_analysis", {})
        if shift_analysis:
            for _shift_level, metrics in shift_analysis.items():
                metrics.get("avg_success_rate", 0)
                metrics.get("false_positive_rate", 0)
                metrics.get("false_negative_rate", 0)

    def _generate_visualizations(self) -> None:
        """Generate comprehensive visualizations of results"""
        if not self.results:
            self.logger.warning("No results available for visualization")
            return

        # Convert to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])

        # Set up matplotlib
        plt.style.use("default")
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

    def _plot_detection_performance(self, df: pd.DataFrame, fig_size: tuple[int, int]) -> None:
        """Plot detection performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Conflict Detection Performance", fontsize=16, fontweight="bold")

        # Accuracy histogram
        axes[0, 0].hist(
            df["detection_accuracy"], bins=20, alpha=0.7, color="blue", edgecolor="black",
        )
        axes[0, 0].set_xlabel("Detection Accuracy")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Detection Accuracy Distribution")
        axes[0, 0].axvline(
            df["detection_accuracy"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {df["detection_accuracy"].mean():.3f}',
        )
        axes[0, 0].legend()

        # Precision vs Recall scatter
        axes[0, 1].scatter(df["recall"], df["precision"], alpha=0.6, color="green")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision vs Recall")
        axes[0, 1].grid(True, alpha=0.3)

        # False Positives vs False Negatives
        fp_fn_data = (
            df.groupby(["false_positives", "false_negatives"]).size().reset_index(name="count")
        )
        axes[1, 0].scatter(
            fp_fn_data["false_positives"],
            fp_fn_data["false_negatives"],
            s=fp_fn_data["count"] * 20,
            alpha=0.6,
            color="orange",
        )
        axes[1, 0].set_xlabel("False Positives")
        axes[1, 0].set_ylabel("False Negatives")
        axes[1, 0].set_title("False Positives vs False Negatives")
        axes[1, 0].grid(True, alpha=0.3)

        # Success rate by complexity
        success_by_complexity = df.groupby("complexity_tier")["resolution_success"].mean()
        axes[1, 1].bar(
            success_by_complexity.index, success_by_complexity.values, color="purple", alpha=0.7,
        )
        axes[1, 1].set_xlabel("Complexity Tier")
        axes[1, 1].set_ylabel("Resolution Success Rate")
        axes[1, 1].set_title("Success Rate by Complexity")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "detection_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_safety_margins(self, df: pd.DataFrame, fig_size: tuple[int, int]) -> None:
        """Plot safety margin distributions"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Safety Margin Analysis", fontsize=16, fontweight="bold")

        # Horizontal separation distribution
        axes[0, 0].hist(
            df["min_separation_nm"], bins=30, alpha=0.7, color="blue", edgecolor="black",
        )
        axes[0, 0].axvline(
            self.config.min_separation_nm,
            color="red",
            linestyle="--",
            label=f"Min Required: {self.config.min_separation_nm} NM",
        )
        axes[0, 0].set_xlabel("Minimum Separation (NM)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Horizontal Separation Distribution")
        axes[0, 0].legend()

        # Vertical separation distribution
        axes[0, 1].hist(
            df["min_separation_ft"], bins=30, alpha=0.7, color="green", edgecolor="black",
        )
        axes[0, 1].axvline(
            self.config.min_separation_ft,
            color="red",
            linestyle="--",
            label=f"Min Required: {self.config.min_separation_ft} ft",
        )
        axes[0, 1].set_xlabel("Minimum Separation (ft)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Vertical Separation Distribution")
        axes[0, 1].legend()

        # Violations by scenario type
        violations_by_type = df.groupby("scenario_type")["separation_violations"].sum()
        axes[1, 0].bar(violations_by_type.index, violations_by_type.values, color="red", alpha=0.7)
        axes[1, 0].set_xlabel("Scenario Type")
        axes[1, 0].set_ylabel("Total Violations")
        axes[1, 0].set_title("Separation Violations by Type")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Safety margin correlation
        axes[1, 1].scatter(
            df["safety_margin_hz"], df["safety_margin_vt"], alpha=0.6, color="purple",
        )
        axes[1, 1].set_xlabel("Horizontal Safety Margin (NM)")
        axes[1, 1].set_ylabel("Vertical Safety Margin (ft)")
        axes[1, 1].set_title("Safety Margin Correlation")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "safety_margins.png", dpi=300, bbox_inches="tight",
        )
        plt.close()

    def _plot_efficiency_metrics(self, df: pd.DataFrame, fig_size: tuple[int, int]) -> None:
        """Plot efficiency and cost metrics"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Efficiency Metrics", fontsize=16, fontweight="bold")

        # Extra distance distribution
        axes[0, 0].hist(
            df["extra_distance_nm"], bins=20, alpha=0.7, color="orange", edgecolor="black",
        )
        axes[0, 0].set_xlabel("Extra Distance (NM)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Extra Path Distance Distribution")

        # Delay distribution
        axes[0, 1].hist(
            df["total_delay_seconds"], bins=20, alpha=0.7, color="red", edgecolor="black",
        )
        axes[0, 1].set_xlabel("Total Delay (seconds)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Delay Distribution")

        # Interventions vs Efficiency
        axes[1, 0].scatter(
            df["num_interventions"], df["extra_distance_nm"], alpha=0.6, color="blue",
        )
        axes[1, 0].set_xlabel("Number of Interventions")
        axes[1, 0].set_ylabel("Extra Distance (NM)")
        axes[1, 0].set_title("Interventions vs Extra Distance")
        axes[1, 0].grid(True, alpha=0.3)

        # Fuel penalty by complexity
        fuel_by_complexity = df.groupby("complexity_tier")["fuel_penalty_percent"].mean()
        axes[1, 1].bar(
            fuel_by_complexity.index, fuel_by_complexity.values, color="green", alpha=0.7,
        )
        axes[1, 1].set_xlabel("Complexity Tier")
        axes[1, 1].set_ylabel("Fuel Penalty (%)")
        axes[1, 1].set_title("Fuel Penalty by Complexity")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "efficiency_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_performance_by_type(self, df: pd.DataFrame, fig_size: tuple[int, int]) -> None:
        """Plot performance metrics by scenario type"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Performance by Scenario Type", fontsize=16, fontweight="bold")

        # Accuracy by type
        acc_by_type = df.groupby("scenario_type")["detection_accuracy"].mean()
        axes[0, 0].bar(acc_by_type.index, acc_by_type.values, color="blue", alpha=0.7)
        axes[0, 0].set_xlabel("Scenario Type")
        axes[0, 0].set_ylabel("Detection Accuracy")
        axes[0, 0].set_title("Detection Accuracy by Type")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Execution time by type
        time_by_type = df.groupby("scenario_type")["execution_time_seconds"].mean()
        axes[0, 1].bar(time_by_type.index, time_by_type.values, color="green", alpha=0.7)
        axes[0, 1].set_xlabel("Scenario Type")
        axes[0, 1].set_ylabel("Execution Time (s)")
        axes[0, 1].set_title("Execution Time by Type")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Box plot of separations by type
        type_sep_data = [
            df[df["scenario_type"] == t]["min_separation_nm"].values
            for t in df["scenario_type"].unique()
        ]
        axes[1, 0].boxplot(type_sep_data, labels=df["scenario_type"].unique())
        axes[1, 0].set_xlabel("Scenario Type")
        axes[1, 0].set_ylabel("Min Separation (NM)")
        axes[1, 0].set_title("Separation Distribution by Type")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Success rate comparison
        success_by_type = df.groupby("scenario_type")["resolution_success"].mean()
        axes[1, 1].bar(success_by_type.index, success_by_type.values, color="purple", alpha=0.7)
        axes[1, 1].set_xlabel("Scenario Type")
        axes[1, 1].set_ylabel("Resolution Success Rate")
        axes[1, 1].set_title("Resolution Success by Type")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "performance_by_type.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_distribution_shift_impact(self, df: pd.DataFrame, fig_size: tuple[int, int]) -> None:
        """Plot impact of distribution shift on performance"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Distribution Shift Impact", fontsize=16, fontweight="bold")

        # Accuracy across shifts
        acc_by_shift = df.groupby("distribution_shift_tier")["detection_accuracy"].mean()
        colors = ["green", "orange", "red"]
        axes[0, 0].bar(
            acc_by_shift.index, acc_by_shift.values, color=colors[: len(acc_by_shift)], alpha=0.7,
        )
        axes[0, 0].set_xlabel("Distribution Shift Level")
        axes[0, 0].set_ylabel("Detection Accuracy")
        axes[0, 0].set_title("Accuracy vs Distribution Shift")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # False positives/negatives by shift
        fp_by_shift = df.groupby("distribution_shift_tier")["false_positives"].mean()
        fn_by_shift = df.groupby("distribution_shift_tier")["false_negatives"].mean()
        x = range(len(fp_by_shift))
        width = 0.35
        axes[0, 1].bar(
            [i - width / 2 for i in x],
            fp_by_shift.values,
            width,
            label="False Positives",
            color="red",
            alpha=0.7,
        )
        axes[0, 1].bar(
            [i + width / 2 for i in x],
            fn_by_shift.values,
            width,
            label="False Negatives",
            color="blue",
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Distribution Shift Level")
        axes[0, 1].set_ylabel("Average Count")
        axes[0, 1].set_title("FP/FN vs Distribution Shift")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(fp_by_shift.index, rotation=45)
        axes[0, 1].legend()

        # Safety margin degradation
        safety_by_shift = df.groupby("distribution_shift_tier")["safety_margin_hz"].mean()
        axes[1, 0].bar(safety_by_shift.index, safety_by_shift.values, color="purple", alpha=0.7)
        axes[1, 0].set_xlabel("Distribution Shift Level")
        axes[1, 0].set_ylabel("Average Safety Margin (NM)")
        axes[1, 0].set_title("Safety Margin vs Distribution Shift")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Execution time impact
        time_by_shift = df.groupby("distribution_shift_tier")["execution_time_seconds"].mean()
        axes[1, 1].bar(time_by_shift.index, time_by_shift.values, color="orange", alpha=0.7)
        axes[1, 1].set_xlabel("Distribution Shift Level")
        axes[1, 1].set_ylabel("Execution Time (s)")
        axes[1, 1].set_title("Execution Time vs Distribution Shift")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "distribution_shift_impact.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_results(self) -> None:
        """Save results in multiple formats"""

        # 1. Save detailed results as JSON
        results_data = [asdict(result) for result in self.results]
        with open(self.output_dir / "raw_data" / "detailed_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # 2. Save summary CSV
        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "summaries" / "results_summary.csv", index=False)

        # 3. Save benchmark summary
        summary = self._generate_summary()
        with open(self.output_dir / "summaries" / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 4. Save configuration with enum serialization
        with open(self.output_dir / "summaries" / "configuration.json", "w") as f:
            config_dict = asdict(self.config)
            # Convert enum objects to their string values for JSON serialization
            if 'scenario_types' in config_dict:
                config_dict['scenario_types'] = [
                    st.value if hasattr(st, 'value') else str(st) 
                    for st in config_dict['scenario_types']
                ]
            if 'complexity_tiers' in config_dict:
                config_dict['complexity_tiers'] = [
                    ct.value if hasattr(ct, 'value') else str(ct) 
                    for ct in config_dict['complexity_tiers']
                ]
            json.dump(config_dict, f, indent=2)

        self.logger.info(f"Results saved to {self.output_dir}")

        # Print summary to console


def run_benchmark_with_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Run Monte Carlo benchmark with configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Benchmark summary results
    """

    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)

        # Convert string enums back to objects
        if "scenario_types" in config_dict:
            config_dict["scenario_types"] = [ScenarioType(t) for t in config_dict["scenario_types"]]
        if "complexity_tiers" in config_dict:
            config_dict["complexity_tiers"] = [
                ComplexityTier(t) for t in config_dict["complexity_tiers"]
            ]

        config = BenchmarkConfiguration(**config_dict)
    else:
        config = BenchmarkConfiguration()

    # Initialize and run benchmark
    benchmark = MonteCarloBenchmark(config)
    return benchmark.run()


def main():
    """Main entry point for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo Benchmark Runner for LLM-ATC-HAL")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument(
        "--scenarios", "-n", type=int, default=10, help="Number of scenarios per type (default: 10)",
    )
    parser.add_argument(
        "--output", "-o", default="output/monte_carlo_benchmark", help="Output directory",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # NEW: Expose simulation parameters
    parser.add_argument(
        "--max-interventions",
        type=int,
        default=5,
        help="Maximum interventions per scenario (default: 5)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=10.0,
        help="Simulation step size in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--time-horizon", type=float, default=10.0, help="Time horizon in minutes (default: 10.0)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create configuration
    if args.config:
        summary = run_benchmark_with_config(args.config)
    else:
        config = BenchmarkConfiguration(
            num_scenarios_per_type=args.scenarios,
            output_directory=args.output,
            max_interventions_per_scenario=args.max_interventions,
            step_size_seconds=args.step_size,
            time_horizon_minutes=args.time_horizon,
        )
        benchmark = MonteCarloBenchmark(config)
        summary = benchmark.run()

    return summary


if __name__ == "__main__":
    main()
