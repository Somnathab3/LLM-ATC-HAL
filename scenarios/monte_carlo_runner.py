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
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict

import matplotlib.pyplot as plt
import pandas as pd

from llm_atc.tools import bluesky_tools
from llm_atc.tools.llm_prompt_engine import LLMPromptEngine
from llm_atc.tools.bluesky_command_validator import get_validator, auto_correct_command
from llm_atc.tools.baseline_resolution_strategy import (
    get_baseline_strategy, 
    ConflictGeometry as BaselineConflictGeometry,
    ResolutionCommand
)
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
    logging.warning(
        "Monte Carlo analysis module not available - limited summary functionality"
    )


@dataclass
class DetectionComparison:
    """Enhanced comparison record for detection analysis"""

    scenario_id: str
    scenario_type: str
    complexity_tier: str
    shift_level: str

    # Ground truth data
    ground_truth_conflicts: int
    ground_truth_pairs: str  # JSON string of pairs

    # BlueSky detection results
    bluesky_conflicts: int
    bluesky_pairs: str  # JSON string of pairs
    bluesky_confidence: float

    # LLM detection results
    llm_prompt: str
    llm_response: str
    llm_conflicts: int
    llm_pairs: str  # JSON string of pairs
    llm_confidence: float

    # Resolution data
    resolution_prompt: str
    resolution_response: str
    resolution_commands: str  # JSON string of commands
    bluesky_commands_executed: str  # JSON string of executed commands

    # Final results
    final_separation_status: str
    detection_accuracy: str  # TP, FP, TN, FN
    resolution_success: bool
    execution_time_ms: float


@dataclass
class BenchmarkConfiguration:
    """Configuration for Monte Carlo benchmark runs"""

    # Scenario parameters - NEW: per-type scenario counts
    num_scenarios_per_type: int = 50  # Kept for backward compatibility
    scenario_counts: Optional[dict[str, int]] = None  # New: per-type counts
    scenario_types: Optional[List[ScenarioType]] = None
    complexity_tiers: Optional[List[ComplexityTier]] = None
    distribution_shift_levels: Optional[List[str]] = None

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
    enable_planner_crosscheck: bool = True  # If True, use Planner.assess_conflict for cross-validation
    
    # Baseline resolution configuration
    baseline_resolution_mode: bool = False  # If True, use baseline strategy instead of LLM
    baseline_preferred_method: Optional[str] = None  # "horizontal", "vertical", "speed", or None for automatic
    baseline_asas_mode: bool = False  # If True, use ASAS-like automated resolution logic
    enable_baseline_comparison: bool = True  # If True, generate baseline resolutions for comparison

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
            self.distribution_shift_levels = [
                "in_distribution",
                "moderate_shift",
                "extreme_shift",
            ]

        # Initialize scenario_counts if not provided
        if self.scenario_counts is None:
            self.scenario_counts = {
                (
                    scenario_type.value
                    if hasattr(scenario_type, "value")
                    else scenario_type
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

    # Performance metrics - Enhanced FP/FN standardization
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    false_positive_rate: float  # Standardized FP rate from calc_fp_fn
    false_negative_rate: float  # Standardized FN rate from calc_fp_fn  
    detection_accuracy: float
    precision: float
    recall: float

    # Environmental factors
    wind_speed_kts: float
    visibility_nm: float
    turbulence_level: float

    # LLM interaction logs for detailed analysis
    llm_prompt: str
    llm_response: str
    resolution_prompt: str
    resolution_response: str

    # Fields with defaults must come after fields without defaults
    # CPA-based Resolution Metrics
    cpa_post_resolution: list[dict[str, Any]] = field(default_factory=list)  # CPA data after resolutions
    cpa_pre_resolution: list[dict[str, Any]] = field(default_factory=list)   # CPA data before resolutions
    resolution_effectiveness: dict[str, Any] = field(default_factory=dict)   # Per-conflict resolution analysis
    insufficient_cpa_resolutions: int = 0             # Count of resolutions with CPA < threshold
    average_cpa_improvement_nm: float = 0.0           # Average improvement in CPA distance
    successful_cpa_resolutions: int = 0               # Count of resolutions with safe CPA

    # Enhanced Safety Margin Quality Assessment
    safety_margin_quality: str = "unknown"  # critical, marginal, adequate, excellent
    separation_breach_count: int = 0  # Number of separation violations
    worst_case_separation_nm: float = 999.0  # Minimum separation achieved
    worst_case_separation_ft: float = 999999.0  # Minimum vertical separation achieved
    resolution_quality_score: float = 0.0  # Quality score for LLM resolutions (0-1)
    low_quality_resolutions: int = 0  # Count of resolutions with <0.5 NM margin

    # Execution metadata
    success: bool = False  # Overall scenario execution success
    execution_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: str = ""

    # Detection tracking for enhanced analysis
    detected_conflicts: list[dict[str, Any]] = field(default_factory=list)  # All detected conflicts with metadata
    bluesky_conflicts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Set defaults for mutable fields"""
        # No longer needed since we use field(default_factory=list) 
        # All mutable fields are properly initialized
        pass


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

        # Enhanced logging storage
        self.detection_comparisons: List[DetectionComparison] = []

        # Current scenario LLM interaction data
        self.current_llm_prompt: str = ""
        self.current_llm_response: str = ""
        self.current_resolution_prompt: str = ""
        self.current_resolution_response: str = ""

        # Setup output directory
        self._setup_output_directory()

        # Initialize logging
        self._setup_logging()

        # Setup enhanced logging components
        self._setup_enhanced_logging()

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

    def _setup_enhanced_logging(self) -> None:
        """Setup enhanced logging with separate loggers for different components"""
        if not self.config.detailed_logging:
            return

        # Create specialized loggers
        self.llm_logger = logging.getLogger(f"{__name__}.llm")
        self.debug_logger = logging.getLogger(f"{__name__}.debug")

        # LLM interactions log
        llm_log_file = self.output_dir / "logs" / "llm_interactions.log"
        llm_handler = logging.FileHandler(llm_log_file, encoding="utf-8")
        llm_handler.setLevel(logging.INFO)
        llm_formatter = logging.Formatter(
            "%(asctime)s - SCENARIO:%(scenario_id)s - %(levelname)s - %(message)s"
        )
        llm_handler.setFormatter(llm_formatter)
        self.llm_logger.addHandler(llm_handler)
        self.llm_logger.setLevel(logging.INFO)

        # Debug log
        debug_log_file = self.output_dir / "logs" / "debug.log"
        debug_handler = logging.FileHandler(debug_log_file, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            "%(asctime)s - SCENARIO:%(scenario_id)s - %(levelname)s - %(message)s"
        )
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)
        self.debug_logger.setLevel(logging.DEBUG)

        # CSV output for detection comparison
        self.csv_path = self.output_dir / "detection_comparison.csv"
        self._init_csv_file()

    def _init_csv_file(self) -> None:
        """Initialize CSV file with headers"""
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "scenario_id",
                        "scenario_type",
                        "complexity_tier",
                        "shift_level",
                        "ground_truth_conflicts",
                        "ground_truth_pairs",
                        "bluesky_conflicts",
                        "bluesky_pairs",
                        "bluesky_confidence",
                        "llm_prompt",
                        "llm_response",
                        "llm_conflicts",
                        "llm_pairs",
                        "llm_confidence",
                        "resolution_prompt",
                        "resolution_response",
                        "resolution_commands",
                        "bluesky_commands_executed",
                        "final_separation_status",
                        "detection_accuracy",
                        "resolution_success",
                        "execution_time_ms",
                        # CPA-based metrics
                        "cpa_pre_resolution_count",
                        "cpa_post_resolution_count", 
                        "successful_cpa_resolutions",
                        "insufficient_cpa_resolutions",
                        "average_cpa_improvement_nm",
                        "cpa_success_rate_percent",
                        "min_cpa_distance_post_resolution",
                        "cpa_analysis_details",
                    ]
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize CSV file: {e}")

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
            # Try to get scenario count using different key formats
            scenario_count = 0

            # Try enum object as key first
            if scenario_type in self.config.scenario_counts:
                scenario_count = self.config.scenario_counts[scenario_type]
            else:
                # Try string value as key
                scenario_key = (
                    scenario_type.value
                    if hasattr(scenario_type, "value")
                    else str(scenario_type)
                )
                scenario_count = self.config.scenario_counts.get(scenario_key, 0)

            total += (
                scenario_count
                * len(self.config.complexity_tiers)
                * len(self.config.distribution_shift_levels)
            )
        return total

    def _run_scenario_batch(
        self,
        scenario_type: ScenarioType,
        complexity_tier: ComplexityTier,
        shift_level: str,
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
        num_scenarios = 0
        if scenario_type in self.config.scenario_counts:
            num_scenarios = self.config.scenario_counts[scenario_type]
        else:
            scenario_key = (
                scenario_type.value
                if hasattr(scenario_type, "value")
                else str(scenario_type)
            )
            num_scenarios = self.config.scenario_counts.get(scenario_key, 0)

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

                # Execute enhanced scenario with CSV logging
                result = self._run_enhanced_scenario(scenario, scenario_id)

                # Store result
                self.results.append(result)
                if result.success:
                    successful_scenarios += 1

                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Batch {batch_id}: completed {i+1}/{num_scenarios}"
                    )

            except Exception as e:
                self.logger.exception(f"Failed to execute scenario {scenario_id}: {e}")

                # Create error result
                error_result = self._create_error_result(
                    scenario_id,
                    scenario_type,
                    complexity_tier,
                    shift_level,
                    str(e),
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

    def _get_aircraft_count_for_complexity(
        self, complexity_tier: ComplexityTier
    ) -> int:
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
            success = not (
                has_unresolved_conflicts or has_parse_errors or has_critical_errors
            )

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
                self.logger.warning(
                    f"Scenario {scenario_id} failed: {', '.join(reasons)}"
                )

            return result

        except Exception as e:
            # Catch exceptions at this level and create error result
            error_message = f"Exception in scenario execution: {e!s}"
            self.logger.exception(f"Scenario {scenario_id} failed with exception: {e}")
            self.logger.debug(traceback.format_exc())

            # Extract basic scenario information for error result
            scenario_type = getattr(scenario, "scenario_type", ScenarioType.HORIZONTAL)
            complexity_tier = getattr(
                scenario, "complexity_tier", ComplexityTier.MODERATE
            )
            shift_level = getattr(
                scenario, "distribution_shift_tier", "in_distribution"
            )

            return self._create_error_result(
                scenario_id,
                scenario_type,
                complexity_tier,
                shift_level,
                error_message,
            )

    def _execute_scenario_pipeline(
        self, scenario: Any, scenario_id: str
    ) -> ScenarioResult:
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

        # Reset LLM interaction data for this scenario
        self.current_llm_prompt = ""
        self.current_llm_response = ""
        self.current_resolution_prompt = ""
        self.current_resolution_response = ""

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

            # Extract BlueSky-specific conflicts from detected conflicts
            bluesky_conflicts = [
                conflict
                for conflict in detected_conflicts
                if conflict.get("source", "").startswith(("enhanced", "bluesky"))
            ]

            # Create result object
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=(
                    scenario.scenario_type.value
                    if hasattr(scenario, "scenario_type")
                    else "unknown"
                ),
                complexity_tier=getattr(
                    scenario, "complexity_tier", ComplexityTier.MODERATE
                ).value,
                distribution_shift_tier=getattr(
                    scenario,
                    "distribution_shift_tier",
                    "in_distribution",
                ),
                aircraft_count=getattr(
                    scenario, "aircraft_count", len(scenario.initial_states)
                ),
                duration_minutes=getattr(
                    scenario,
                    "duration_minutes",
                    self.config.time_horizon_minutes,
                ),
                # Ground truth
                true_conflicts=ground_truth_conflicts,
                num_true_conflicts=len(ground_truth_conflicts),
                # Detection
                predicted_conflicts=detected_conflicts,
                num_predicted_conflicts=len(detected_conflicts),
                detection_method=(
                    "hybrid" if self.config.enable_llm_detection else "ground_truth"
                ),
                # BlueSky specific conflicts
                bluesky_conflicts=bluesky_conflicts,
                # Resolution
                llm_commands=[r.get("command", "") for r in resolutions],
                resolution_success=verification_results.get(
                    "resolution_success", False
                ),
                num_interventions=len(resolutions),
                # Safety metrics
                min_separation_nm=verification_results.get("min_separation_nm", 999.0),
                min_separation_ft=verification_results.get(
                    "min_separation_ft", 999999.0
                ),
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
                # CPA-based Resolution Metrics
                cpa_post_resolution=verification_results.get("cpa_post_resolution", []),
                cpa_pre_resolution=verification_results.get("cpa_pre_resolution", []),
                resolution_effectiveness=verification_results.get("resolution_effectiveness", {}),
                insufficient_cpa_resolutions=verification_results.get("insufficient_cpa_resolutions", 0),
                average_cpa_improvement_nm=verification_results.get("average_cpa_improvement_nm", 0.0),
                successful_cpa_resolutions=verification_results.get("successful_cpa_resolutions", 0),
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
                    "wind_speed_kts",
                    0,
                ),
                visibility_nm=getattr(scenario, "environmental_conditions", {}).get(
                    "visibility_nm",
                    10,
                ),
                turbulence_level=getattr(scenario, "environmental_conditions", {}).get(
                    "turbulence_intensity",
                    0,
                ),
                # LLM interaction logs for detailed analysis
                llm_prompt=self.current_llm_prompt,
                llm_response=self.current_llm_response,
                resolution_prompt=self.current_resolution_prompt,
                resolution_response=self.current_resolution_response,
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
        """Extract ground truth conflicts from scenario or generate them from simulation"""
        try:
            # First, try to get ground truth from the scenario object
            if hasattr(scenario, "ground_truth_conflicts") and scenario.ground_truth_conflicts:
                return [
                    asdict(conflict) for conflict in scenario.ground_truth_conflicts
                ]
            
            # If no ground truth available, generate it using EnhancedConflictDetector
            # This ensures we always have accurate ground truth rather than using mock data
            self.logger.warning("Scenario missing ground truth conflicts - generating from simulation state")
            
            try:
                from llm_atc.tools.enhanced_conflict_detector import EnhancedConflictDetector
                
                # Run the simulation briefly to establish aircraft states
                # Most scenario files should include RESET and aircraft creation commands
                if hasattr(scenario, 'commands') or hasattr(scenario, 'bluesky_commands'):
                    # Let the scenario commands execute first (this happens in _load_scenario_commands)
                    # Then extract ground truth from the resulting simulation state
                    
                    enhanced_detector = EnhancedConflictDetector()
                    
                    # Generate comprehensive ground truth using all detection methods
                    comprehensive_conflicts = enhanced_detector.detect_conflicts_comprehensive()
                    
                    ground_truth_conflicts = []
                    for conflict in comprehensive_conflicts:
                        # Convert to standard ground truth format
                        ground_truth_conflicts.append({
                            "aircraft_pair": [conflict.aircraft_1, conflict.aircraft_2],
                            "horizontal_separation": conflict.current_horizontal_separation,
                            "vertical_separation": conflict.current_vertical_separation,
                            "time_to_cpa": conflict.time_to_cpa,
                            "min_horizontal_separation": conflict.min_horizontal_separation,
                            "min_vertical_separation": conflict.min_vertical_separation,
                            "violates_icao": conflict.violates_icao_separation,
                            "severity": conflict.severity,
                            "confidence": conflict.confidence,
                            "detection_method": "generated_from_simulation",
                            "source": "enhanced_detector_ground_truth"
                        })
                    
                    if ground_truth_conflicts:
                        self.logger.info(f"Generated {len(ground_truth_conflicts)} ground truth conflicts from simulation")
                        return ground_truth_conflicts
                    else:
                        self.logger.info("No conflicts detected in simulation - scenario appears conflict-free")
                        return []
                        
            except Exception as e:
                self.logger.warning(f"Failed to generate ground truth from simulation: {e}")
                # Fall back to empty list rather than mock data
                return []
            
            # Final fallback - empty list (no mock data)
            self.logger.info("No conflicts detected - returning empty ground truth")
            return []
            
        except Exception as e:
            self.logger.warning(f"Failed to extract ground truth: {e}")
            return []

    def _detect_conflicts(self, scenario: Any) -> list[dict[str, Any]]:
        """Perform conflict detection using multiple BlueSky methods for validation"""
        detected_conflicts = []

        try:
            # Import enhanced conflict detector
            from llm_atc.tools.enhanced_conflict_detector import (
                EnhancedConflictDetector,
            )

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
                        "conflict_within_300s": conflict.time_to_cpa
                        <= 300.0,  # Key improvement
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
                            "conflict_within_300s": conflict.get("time_to_cpa", 999)
                            <= 300.0,
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
                        "time_to_cpa": first_conflict.time_to_cpa,
                        "min_horizontal_separation": first_conflict.min_horizontal_separation,
                        "min_vertical_separation": first_conflict.min_vertical_separation,
                        "current_horizontal_separation": first_conflict.current_horizontal_separation,
                        "current_vertical_separation": first_conflict.current_vertical_separation,
                        "violates_icao_separation": first_conflict.violates_icao_separation,
                        "severity": first_conflict.severity,
                    }

                llm_detection = self.llm_engine.detect_conflict_via_llm_with_prompts(
                    aircraft_states,
                    self.config.time_horizon_minutes,
                    cpa_data=cpa_data,  # Enhanced with CPA data
                )

                # Store LLM prompt and response for CSV output
                self.current_llm_prompt = llm_detection.get("llm_prompt", "")
                self.current_llm_response = llm_detection.get("llm_response", "")

                # Validate LLM detection response
                if self.config.validate_llm_responses and not isinstance(
                    llm_detection, dict
                ):
                    error_msg = f"LLM conflict detection returned invalid response type: {type(llm_detection)}"
                    self.logger.error(error_msg)
                    if self.config.strict_mode:
                        raise Exception(error_msg)

                if llm_detection and llm_detection.get("conflict_detected", False):
                    aircraft_pairs = llm_detection.get("aircraft_pairs", [])
                    if self.config.validate_llm_responses and not aircraft_pairs:
                        error_msg = (
                            "LLM detected conflict but provided no aircraft pairs"
                        )
                        self.logger.warning(error_msg)
                        if self.config.strict_mode:
                            raise Exception(error_msg)

                    # Enhanced LLM vs BlueSky conflict comparison for false positive prevention
                    detected_pairs = [(pair[0], pair[1]) for pair in aircraft_pairs if len(pair) >= 2]
                    
                    # Get BlueSky conflicts for comparison
                    bs_conflicts = bluesky_tools.get_conflict_info().get("conflicts", [])
                    bs_pairs = [(c['aircraft_1'], c['aircraft_2']) for c in bs_conflicts]
                    
                    # Cross-validate LLM predictions with BlueSky ground truth
                    validated_pairs = self._validate_llm_conflicts_with_bluesky(detected_pairs, bs_pairs)
                    
                    # Calculate and log false positive statistics
                    original_count = len(detected_pairs)
                    validated_count = len(validated_pairs)
                    false_positives = original_count - validated_count
                    
                    if false_positives > 0:
                        false_positive_rate = (false_positives / original_count) * 100 if original_count > 0 else 0
                        self.logger.warning(
                            f"LLM conflict detection false positives: {false_positives}/{original_count} "
                            f"({false_positive_rate:.1f}%) conflicts dropped as hallucinations"
                        )
                        
                        # Log detailed false positive analysis
                        false_positive_pairs = [pair for pair in detected_pairs if pair not in validated_pairs]
                        for fp_pair in false_positive_pairs:
                            self.logger.debug(f"False positive conflict dropped: {fp_pair[0]} <-> {fp_pair[1]}")
                    else:
                        self.logger.info(f"LLM conflict detection: All {validated_count} conflicts validated by BlueSky")

                    # Use unified enhanced detector validation to reduce hallucinations
                    # This replaces the old _validate_llm_conflicts_with_bluesky method for confidence scoring
                    enhanced_validated_pairs = enhanced_detector.validate_llm_conflicts(
                        aircraft_pairs
                    )

                    for pair_data in enhanced_validated_pairs:
                        ac1, ac2, confidence = pair_data
                        # Only add conflicts that passed both BlueSky and enhanced validation
                        if (ac1, ac2) in validated_pairs or (ac2, ac1) in validated_pairs:
                            detected_conflicts.append(
                                {
                                    "source": "llm_enhanced_validated",
                                    "aircraft_1": ac1,
                                    "aircraft_2": ac2,
                                    "confidence": confidence,
                                    "priority": llm_detection.get("priority", "unknown"),
                                    "validation": "bluesky_and_enhanced_confirmed",
                                    "uses_icao_standards": True,
                                    "cpa_analysis_provided": bool(cpa_data),
                                    "false_positive_filtered": True,
                                },
                            )

            # Optional: Cross-verify all detected conflicts with Planner assessment
            if self.config.enable_planner_crosscheck and detected_conflicts:
                planner_verification = self._verify_conflicts_with_planner(detected_conflicts)
                self.logger.info(f"Planner cross-verification completed: {len(planner_verification)} conflicts verified")
                
                # Add planner verification metadata to existing conflicts
                for conflict in detected_conflicts:
                    conflict["planner_verified"] = any(
                        pv["aircraft_1"] == conflict["aircraft_1"] and pv["aircraft_2"] == conflict["aircraft_2"]
                        for pv in planner_verification
                    )

        except Exception as e:
            self.logger.exception(f"Enhanced conflict detection failed: {e}")
            # Fallback to basic detection
            detected_conflicts = self._basic_conflict_detection_fallback()

        return detected_conflicts

    def _basic_conflict_detection_fallback(self) -> list[dict[str, Any]]:
        """Basic conflict detection fallback when enhanced detection fails"""
        detected_conflicts = []
        
        try:
            # Method 1: BlueSky built-in conflict detection
            bluesky_conflicts = bluesky_tools.get_conflict_info()
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
                
            # Method 2: Planner.assess_conflict as backup/cross-check
            if not detected_conflicts or self.config.enable_planner_crosscheck:
                planner_conflicts = self._detect_conflicts_with_planner()
                detected_conflicts.extend(planner_conflicts)
                
        except Exception as e:
            self.logger.exception(f"Fallback conflict detection also failed: {e}")
            
        return detected_conflicts

    def _validate_llm_conflicts_with_bluesky(
        self, 
        llm_pairs: list[tuple[str, str]], 
        bs_pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """
        Validate LLM-detected conflicts against BlueSky ground truth.
        
        Args:
            llm_pairs: List of aircraft pairs detected by LLM
            bs_pairs: List of aircraft pairs detected by BlueSky ASAS
            
        Returns:
            List of validated aircraft pairs that are confirmed by BlueSky
        """
        validated_pairs = []
        
        # Normalize pairs to ensure consistent comparison (sorted order)
        normalized_bs_pairs = set()
        for pair in bs_pairs:
            normalized_bs_pairs.add(tuple(sorted([pair[0], pair[1]])))
        
        # Check each LLM pair against BlueSky ground truth
        for llm_pair in llm_pairs:
            normalized_llm_pair = tuple(sorted([llm_pair[0], llm_pair[1]]))
            
            if normalized_llm_pair in normalized_bs_pairs:
                # True positive - LLM correctly detected conflict confirmed by BlueSky
                validated_pairs.append(llm_pair)  # Keep original order
                self.logger.debug(f"LLM conflict validated by BlueSky: {llm_pair[0]} <-> {llm_pair[1]}")
            else:
                # False positive - LLM detected conflict not confirmed by BlueSky ASAS
                self.logger.debug(f"LLM false positive filtered out: {llm_pair[0]} <-> {llm_pair[1]}")
        
        return validated_pairs

    def _detect_conflicts_with_planner(self) -> list[dict[str, Any]]:
        """Use Planner.assess_conflict for independent conflict detection"""
        try:
            from llm_atc.agents.planner import Planner
            
            planner = Planner()
            aircraft_info = bluesky_tools.get_all_aircraft_info()
            
            if not aircraft_info or not aircraft_info.get("aircraft"):
                self.logger.warning("No aircraft data available for planner assessment")
                return []
            
            # Use planner's conflict assessment
            assessment = planner.assess_conflict(aircraft_info)
            
            if not assessment:
                self.logger.info("Planner detected no conflicts")
                return []
            
            # Convert planner assessment to standard format
            planner_conflicts = []
            if len(assessment.aircraft_involved) >= 2:
                aircraft_1 = assessment.aircraft_involved[0]
                aircraft_2 = assessment.aircraft_involved[1]
                
                # Get separation data from planner's internal calculation
                ac1_data = aircraft_info["aircraft"].get(aircraft_1, {})
                ac2_data = aircraft_info["aircraft"].get(aircraft_2, {})
                
                # Calculate separation using planner's method for consistency
                separation = self._calculate_planner_separation(ac1_data, ac2_data)
                
                conflict_data = {
                    "source": "planner_assessment",
                    "aircraft_1": aircraft_1,
                    "aircraft_2": aircraft_2,
                    "horizontal_separation": separation.get("horizontal", 999),
                    "vertical_separation": separation.get("vertical", 999),
                    "time_to_cpa": assessment.time_to_conflict,
                    "severity": assessment.severity,
                    "confidence": assessment.confidence,
                    "recommended_action": assessment.recommended_action.value,
                    "reasoning": assessment.reasoning,
                    "uses_icao_standards": True,  # Planner uses same ICAO standards
                    "conflict_id": assessment.conflict_id,
                }
                planner_conflicts.append(conflict_data)
                
                self.logger.info(
                    f"Planner detected conflict: {aircraft_1}-{aircraft_2}, "
                    f"severity={assessment.severity}, confidence={assessment.confidence}"
                )
            
            return planner_conflicts
            
        except Exception as e:
            self.logger.exception(f"Planner conflict detection failed: {e}")
            return []

    def _calculate_planner_separation(self, ac1_data: dict, ac2_data: dict) -> dict[str, float]:
        """Calculate separation using same method as Planner for consistency"""
        try:
            # Simplified calculation matching planner.py implementation
            lat1, lon1, alt1 = (
                ac1_data.get("lat", 0),
                ac1_data.get("lon", 0),
                ac1_data.get("alt", 0),
            )
            lat2, lon2, alt2 = (
                ac2_data.get("lat", 0),
                ac2_data.get("lon", 0),
                ac2_data.get("alt", 0),
            )

            # Horizontal distance in nautical miles (simplified, matches planner.py)
            DEGREES_TO_NM_FACTOR = 60.0
            horizontal_nm = (
                (lat2 - lat1) ** 2 + (lon2 - lon1) ** 2
            ) ** 0.5 * DEGREES_TO_NM_FACTOR

            # Vertical separation in feet
            vertical_ft = abs(alt2 - alt1)

            return {
                "horizontal": horizontal_nm,
                "vertical": vertical_ft,
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate planner separation: {e}")
            return {"horizontal": 999, "vertical": 999}

    def _verify_conflicts_with_planner(self, detected_conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Use Planner.assess_conflict to cross-verify detected conflicts"""
        try:
            from llm_atc.agents.planner import Planner
            
            planner = Planner()
            aircraft_info = bluesky_tools.get_all_aircraft_info()
            
            if not aircraft_info or not aircraft_info.get("aircraft"):
                self.logger.warning("No aircraft data available for planner verification")
                return []
            
            # Get planner's independent assessment
            assessment = planner.assess_conflict(aircraft_info)
            
            if not assessment:
                self.logger.info("Planner verification: no conflicts detected")
                return []
            
            # Cross-check planner's conflicts against our detected conflicts
            verified_conflicts = []
            
            if len(assessment.aircraft_involved) >= 2:
                planner_ac1 = assessment.aircraft_involved[0]
                planner_ac2 = assessment.aircraft_involved[1]
                
                # Check if planner's conflict matches any of our detected conflicts
                for conflict in detected_conflicts:
                    conflict_ac1 = conflict.get("aircraft_1", "")
                    conflict_ac2 = conflict.get("aircraft_2", "")
                    
                    # Check both orderings
                    if ((planner_ac1 == conflict_ac1 and planner_ac2 == conflict_ac2) or
                        (planner_ac1 == conflict_ac2 and planner_ac2 == conflict_ac1)):
                        
                        verified_conflicts.append({
                            "aircraft_1": planner_ac1,
                            "aircraft_2": planner_ac2,
                            "severity": assessment.severity,
                            "confidence": assessment.confidence,
                            "planner_conflict_id": assessment.conflict_id,
                            "time_to_conflict": assessment.time_to_conflict,
                            "recommended_action": assessment.recommended_action.value,
                            "reasoning": assessment.reasoning,
                            "source": "planner_verification",
                            "original_source": conflict.get("source", "unknown"),
                        })
                        
                        self.logger.info(
                            f"Planner verified conflict: {planner_ac1}-{planner_ac2}, "
                            f"severity={assessment.severity} (matches {conflict.get('source', 'unknown')})"
                        )
                        break
            
            return verified_conflicts
            
        except Exception as e:
            self.logger.exception(f"Planner verification failed: {e}")
            return []

    def _get_aircraft_states_for_llm(self) -> list[dict[str, Any]]:
        """Get current aircraft states formatted for LLM"""
        try:
            aircraft_info = bluesky_tools.get_all_aircraft_info()

            # Validate aircraft info exists in strict mode
            if self.config.strict_mode and (
                not aircraft_info or not aircraft_info.get("aircraft")
            ):
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

    def _resolve_conflicts(
        self,
        conflicts: list[dict[str, Any]],
        scenario: Any,
    ) -> list[dict[str, Any]]:
        """Generate conflict resolutions using LLM or baseline strategy"""
        resolutions = []

        for conflict in conflicts:
            try:
                resolution_command = None
                baseline_command = None
                
                # Generate baseline resolution if enabled
                if self.config.enable_baseline_comparison or self.config.baseline_resolution_mode:
                    baseline_command = self._generate_baseline_resolution(conflict)
                
                # Generate LLM resolution unless in baseline-only mode
                if not self.config.baseline_resolution_mode:
                    # Create conflict info for LLM
                    conflict_info = self._format_conflict_for_llm(conflict, scenario)

                    # Get LLM resolution with prompts
                    resolution_data = self.llm_engine.get_conflict_resolution_with_prompts(
                        conflict_info
                    )
                    resolution_command = resolution_data.get("command")

                    # Store resolution prompt and response for CSV output (only for first conflict)
                    if (
                        not self.current_resolution_prompt
                    ):  # Only store first resolution's data
                        self.current_resolution_prompt = resolution_data.get(
                            "resolution_prompt", ""
                        )
                        self.current_resolution_response = resolution_data.get(
                        "resolution_response", ""
                    )

                # Validate LLM response if strict mode enabled
                if self.config.validate_llm_responses and not resolution_command:
                    error_msg = (
                        f"LLM failed to generate resolution for conflict {conflict}"
                    )
                    self.logger.error(error_msg)
                    if self.config.strict_mode:
                        raise Exception(error_msg)

                if resolution_command:
                    # Enhanced command validation with auto-correction
                    validator = get_validator()
                    is_valid, error_msg, suggestion = validator.validate_command(resolution_command)
                    
                    if not is_valid:
                        self.logger.warning(f"Invalid command: {resolution_command}. Error: {error_msg}")
                        
                        # Attempt auto-correction
                        corrected_command, warnings = auto_correct_command(
                            resolution_command, 
                            strict_mode=self.config.strict_mode
                        )
                        
                        if corrected_command:
                            self.logger.info(f"Auto-corrected command: {resolution_command} -> {corrected_command}")
                            for warning in warnings:
                                self.logger.warning(warning)
                            resolution_command = corrected_command
                        else:
                            error_msg = f"LLM generated invalid command that could not be corrected: {resolution_command}"
                            self.logger.error(error_msg)
                            if self.config.strict_mode:
                                raise Exception(error_msg)
                            continue  # Skip this command

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

    def _generate_baseline_resolution(self, conflict: dict[str, Any]) -> Optional[str]:
        """Generate baseline resolution command using conventional ATC strategies"""
        try:
            # Convert conflict data to baseline format
            baseline_conflict = self._convert_conflict_to_baseline_format(conflict)
            
            # Get baseline strategy
            baseline_strategy = get_baseline_strategy()
            
            # Generate baseline resolution commands
            baseline_commands = baseline_strategy.generate_baseline_resolution(
                baseline_conflict,
                preferred_method=self.config.baseline_preferred_method,
                asas_mode=self.config.baseline_asas_mode
            )
            
            if baseline_commands:
                # Return the first (highest priority) command
                primary_command = baseline_commands[0]
                self.logger.info(
                    f"Generated baseline resolution: {primary_command.command} "
                    f"(rationale: {primary_command.rationale})"
                )
                return primary_command.command
            else:
                self.logger.warning("No baseline resolution generated")
                return None
                
        except Exception as e:
            self.logger.exception(f"Failed to generate baseline resolution: {e}")
            return None
    
    def _convert_conflict_to_baseline_format(self, conflict: dict[str, Any]) -> BaselineConflictGeometry:
        """Convert internal conflict format to baseline strategy format"""
        # Extract aircraft information - try different key formats
        aircraft_ids = conflict.get("aircraft_ids", [])
        if len(aircraft_ids) < 2:
            # Try alternative format
            ac1_id = conflict.get("aircraft_1")
            ac2_id = conflict.get("aircraft_2")
            if not ac1_id or not ac2_id:
                raise ValueError("Conflict must involve at least 2 aircraft")
            aircraft_ids = [ac1_id, ac2_id]
        
        # Get aircraft data from current simulation state
        aircraft_data = bluesky_tools.get_all_aircraft_info()
        
        # Find the two aircraft involved in the conflict
        ac1_id = aircraft_ids[0]
        ac2_id = aircraft_ids[1]
        
        # Aircraft data is a dict where keys are aircraft IDs and values are aircraft info
        aircraft_dict = aircraft_data.get("aircraft", {})
        
        ac1_data = aircraft_dict.get(ac1_id)
        ac2_data = aircraft_dict.get(ac2_id)
        
        if not ac1_data or not ac2_data:
            raise ValueError(f"Could not find aircraft data for {ac1_id} or {ac2_id}")
        
        # Calculate conflict geometry
        horizontal_distance = conflict.get("horizontal_distance_nm", 0.0)
        vertical_separation = abs(ac1_data.get("alt", 0) - ac2_data.get("alt", 0))
        time_to_cpa = conflict.get("time_to_closest_approach_min", 5.0)
        
        # Calculate relative bearing
        ac1_lat = ac1_data.get("lat", 0.0)
        ac1_lon = ac1_data.get("lon", 0.0)
        ac2_lat = ac2_data.get("lat", 0.0)
        ac2_lon = ac2_data.get("lon", 0.0)
        
        relative_bearing = self._calculate_bearing(ac1_lat, ac1_lon, ac2_lat, ac2_lon)
        
        # Calculate closing speed
        ac1_speed = ac1_data.get("spd", 250.0)
        ac2_speed = ac2_data.get("spd", 250.0)
        closing_speed = abs(ac1_speed - ac2_speed)  # Simplified calculation
        
        return BaselineConflictGeometry(
            aircraft1_id=ac1_id,
            aircraft2_id=ac2_id,
            horizontal_distance_nm=horizontal_distance,
            vertical_separation_ft=vertical_separation,
            time_to_closest_approach_min=time_to_cpa,
            relative_bearing_deg=relative_bearing,
            aircraft1_heading=ac1_data.get("hdg", 0.0),
            aircraft2_heading=ac2_data.get("hdg", 0.0),
            aircraft1_altitude=ac1_data.get("alt", 35000.0),
            aircraft2_altitude=ac2_data.get("alt", 35000.0),
            aircraft1_speed=ac1_speed,
            aircraft2_speed=ac2_speed,
            closing_speed_kts=closing_speed
        )
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2"""
        import math
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        return (bearing_deg + 360) % 360

    def _is_valid_bluesky_command(self, command: str) -> bool:
        """Validate if a command is a valid BlueSky command"""
        if not command or not isinstance(command, str):
            return False

        command = command.strip()
        if not command:
            return False

        # Check for common BlueSky command patterns
        valid_commands = [
            "HDG",
            "ALT",
            "SPD",
            "CRE",
            "DEL",
            "MOVE",
            "TURN",
            "CLIMB",
            "DESCEND",
        ]
        command_parts = command.split()

        if len(command_parts) < 2:  # Need at least command and aircraft ID
            return False

        command_type = command_parts[0].upper()
        return command_type in valid_commands

    def _format_conflict_for_llm(
        self, conflict: dict[str, Any], scenario: Any
    ) -> dict[str, Any]:
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
                "environmental_conditions": getattr(
                    scenario, "environmental_conditions", {}
                ),
            }

        except Exception as e:
            self.logger.exception(f"Failed to format conflict for LLM: {e}")
            return {}

    def _verify_resolutions(
        self,
        scenario: Any,
        resolutions: list[dict[str, Any]],
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
            # NEW: CPA-based metrics
            "cpa_metrics": {},
            "cpa_post_resolution": [],
            "cpa_pre_resolution": [],
            "resolution_effectiveness": {},
            "insufficient_cpa_resolutions": 0,
            "successful_cpa_resolutions": 0,
            "average_cpa_improvement_nm": 0.0,
        }

        try:
            # Capture pre-resolution CPA data for comparison
            if resolutions:
                pre_resolution_cpa = self._capture_cpa_data("pre_resolution")
                verification_results["cpa_pre_resolution"] = pre_resolution_cpa
                self.logger.debug(f"Captured pre-resolution CPA data: {len(pre_resolution_cpa)} conflicts")

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

            # Capture post-resolution CPA data
            if resolutions:
                post_resolution_cpa = self._capture_cpa_data("post_resolution")
                verification_results["cpa_post_resolution"] = post_resolution_cpa
                self.logger.debug(f"Captured post-resolution CPA data: {len(post_resolution_cpa)} conflicts")

                # Calculate CPA-based resolution effectiveness
                cpa_effectiveness = self._analyze_cpa_resolution_effectiveness(
                    verification_results["cpa_pre_resolution"],
                    post_resolution_cpa,
                    resolutions
                )
                verification_results.update(cpa_effectiveness)

            # Calculate final metrics
            if min_separation_recorded:
                verification_results["min_separation_nm"] = min(
                    s["horizontal_nm"] for s in min_separation_recorded
                )
                verification_results["min_separation_ft"] = min(
                    s["vertical_ft"] for s in min_separation_recorded
                )

                # Resolution is successful if no violations occurred
                verification_results["resolution_success"] = (
                    verification_results["violations"] == 0
                )

                # Calculate efficiency metrics (simplified)
                verification_results["extra_distance_nm"] = (
                    len(resolutions) * 5.0
                )  # Estimate
                verification_results["total_delay"] = (
                    len(resolutions) * 30.0
                )  # Estimate in seconds
                verification_results["fuel_penalty"] = (
                    len(resolutions) * 2.0
                )  # Estimate percentage

        except Exception as e:
            self.logger.exception(f"Verification failed: {e}")

        return verification_results

    def _capture_cpa_data(self, stage: str) -> list[dict[str, Any]]:
        """
        Capture CPA (Closest Point of Approach) data from BlueSky conflict detection.
        
        Args:
            stage: "pre_resolution" or "post_resolution" for logging purposes
            
        Returns:
            List of CPA data for each detected conflict
        """
        cpa_data = []
        
        try:
            # Get BlueSky conflict information with CPA data
            bs_conflicts = bluesky_tools.get_conflict_info()
            
            for conflict in bs_conflicts.get("conflicts", []):
                cpa_info = {
                    "aircraft_1": conflict.get("aircraft_1", ""),
                    "aircraft_2": conflict.get("aircraft_2", ""),
                    "time_to_cpa": conflict.get("time_to_cpa", 999.0),
                    "distance_at_cpa": conflict.get("distance_at_cpa", 999.0),
                    "horizontal_separation": conflict.get("horizontal_separation", 999.0),
                    "vertical_separation": conflict.get("vertical_separation", 999999.0),
                    "stage": stage,
                    "timestamp": time.time(),
                    "violates_icao": (
                        conflict.get("distance_at_cpa", 999.0) < self.config.min_separation_nm
                        and conflict.get("vertical_separation", 999999.0) < self.config.min_separation_ft
                    ),
                }
                cpa_data.append(cpa_info)
                
            self.logger.debug(f"Captured {len(cpa_data)} CPA conflicts at {stage}")
            
        except Exception as e:
            self.logger.warning(f"Failed to capture CPA data at {stage}: {e}")
            
        return cpa_data

    def _analyze_cpa_resolution_effectiveness(
        self,
        pre_resolution_cpa: list[dict[str, Any]],
        post_resolution_cpa: list[dict[str, Any]], 
        resolutions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze resolution effectiveness using CPA metrics.
        
        Args:
            pre_resolution_cpa: CPA data before resolutions
            post_resolution_cpa: CPA data after resolutions
            resolutions: List of resolution commands applied
            
        Returns:
            Dictionary with CPA-based effectiveness metrics
        """
        analysis = {
            "resolution_effectiveness": {},
            "insufficient_cpa_resolutions": 0,
            "successful_cpa_resolutions": 0,
            "average_cpa_improvement_nm": 0.0,
            "cpa_metrics": {},
        }
        
        try:
            # Create lookup maps for aircraft pairs
            pre_cpa_map = {}
            for cpa in pre_resolution_cpa:
                pair_key = tuple(sorted([cpa["aircraft_1"], cpa["aircraft_2"]]))
                pre_cpa_map[pair_key] = cpa
                
            post_cpa_map = {}
            for cpa in post_resolution_cpa:
                pair_key = tuple(sorted([cpa["aircraft_1"], cpa["aircraft_2"]]))
                post_cpa_map[pair_key] = cpa
            
            improvements = []
            per_conflict_analysis = {}
            
            # Analyze each conflict that had a resolution
            for pair_key in pre_cpa_map.keys():
                pre_cpa = pre_cpa_map[pair_key]
                post_cpa = post_cpa_map.get(pair_key)
                
                if post_cpa:
                    # Calculate CPA improvement
                    pre_distance = pre_cpa["distance_at_cpa"]
                    post_distance = post_cpa["distance_at_cpa"]
                    improvement = post_distance - pre_distance
                    
                    improvements.append(improvement)
                    
                    # Determine if resolution was sufficient
                    is_safe_cpa = (
                        post_distance >= self.config.min_separation_nm
                        and post_cpa["vertical_separation"] >= self.config.min_separation_ft
                    )
                    
                    if is_safe_cpa:
                        analysis["successful_cpa_resolutions"] += 1
                    else:
                        analysis["insufficient_cpa_resolutions"] += 1
                        self.logger.warning(
                            f"Insufficient CPA resolution for {pair_key}: "
                            f"CPA distance {post_distance:.2f} NM < {self.config.min_separation_nm} NM"
                        )
                    
                    # Store per-conflict analysis
                    conflict_id = f"{pair_key[0]}-{pair_key[1]}"
                    per_conflict_analysis[conflict_id] = {
                        "pre_cpa_distance": pre_distance,
                        "post_cpa_distance": post_distance,
                        "improvement_nm": improvement,
                        "pre_time_to_cpa": pre_cpa["time_to_cpa"],
                        "post_time_to_cpa": post_cpa["time_to_cpa"],
                        "is_safe_resolution": is_safe_cpa,
                        "violates_icao_pre": pre_cpa["violates_icao"],
                        "violates_icao_post": post_cpa["violates_icao"],
                    }
                    
                    self.logger.debug(
                        f"CPA analysis for {conflict_id}: "
                        f"pre={pre_distance:.2f}nm, post={post_distance:.2f}nm, "
                        f"improvement={improvement:.2f}nm, safe={is_safe_cpa}"
                    )
                else:
                    # Conflict resolved completely (no longer detected)
                    analysis["successful_cpa_resolutions"] += 1
                    conflict_id = f"{pair_key[0]}-{pair_key[1]}"
                    per_conflict_analysis[conflict_id] = {
                        "pre_cpa_distance": pre_cpa["distance_at_cpa"],
                        "post_cpa_distance": 999.0,  # No conflict detected
                        "improvement_nm": 999.0 - pre_cpa["distance_at_cpa"],
                        "pre_time_to_cpa": pre_cpa["time_to_cpa"],
                        "post_time_to_cpa": 999.0,
                        "is_safe_resolution": True,
                        "violates_icao_pre": pre_cpa["violates_icao"],
                        "violates_icao_post": False,
                        "conflict_resolved_completely": True,
                    }
                    
                    self.logger.info(f"Conflict {conflict_id} resolved completely - no longer detected")
            
            # Calculate average improvement
            if improvements:
                analysis["average_cpa_improvement_nm"] = sum(improvements) / len(improvements)
            
            analysis["resolution_effectiveness"] = per_conflict_analysis
            
            # Overall CPA metrics summary
            total_conflicts = len(pre_cpa_map)
            analysis["cpa_metrics"] = {
                "total_conflicts_with_cpa_data": total_conflicts,
                "successful_cpa_resolutions": analysis["successful_cpa_resolutions"],
                "insufficient_cpa_resolutions": analysis["insufficient_cpa_resolutions"],
                "cpa_success_rate": (
                    analysis["successful_cpa_resolutions"] / total_conflicts * 100 
                    if total_conflicts > 0 else 0.0
                ),
                "average_improvement_nm": analysis["average_cpa_improvement_nm"],
                "resolutions_applied": len(resolutions),
            }
            
            self.logger.info(
                f"CPA Resolution Analysis: {analysis['successful_cpa_resolutions']}/{total_conflicts} "
                f"successful ({analysis['cpa_metrics']['cpa_success_rate']:.1f}%), "
                f"avg improvement: {analysis['average_cpa_improvement_nm']:.2f} NM"
            )
            
        except Exception as e:
            self.logger.exception(f"Failed to analyze CPA resolution effectiveness: {e}")
            
        return analysis

    def _calculate_all_separations(
        self, aircraft_info: dict[str, Any]
    ) -> list[dict[str, Any]]:
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
                    horizontal_nm = (
                        (lat_diff**2 + lon_diff**2) ** 0.5
                    ) * 60  # Rough conversion

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

        # Calculate confusion matrix using both methods for comparison
        tp = len(true_conflicts.intersection(detected_conflicts))
        fp = len(detected_conflicts - true_conflicts)
        fn = len(true_conflicts - detected_conflicts)
        tn = max(0, 10 - tp - fp - fn)  # Estimate based on potential pairs

        # Also use standardized metrics.calc_fp_fn for consistency
        try:
            from llm_atc.metrics import calc_fp_fn
            
            # Convert conflicts to standard format for calc_fp_fn
            pred_conflicts_list = [
                {"aircraft_1": pair[0], "aircraft_2": pair[1]} 
                for pair in detected_conflicts
            ]
            gt_conflicts_list = [
                {"aircraft_1": pair[0], "aircraft_2": pair[1]} 
                for pair in true_conflicts
            ]
            
            fp_rate, fn_rate = calc_fp_fn(pred_conflicts_list, gt_conflicts_list)
            
            self.logger.debug(
                f"Metrics comparison - Local FP/FN: {fp}/{fn}, "
                f"Standardized FP/FN rates: {fp_rate:.3f}/{fn_rate:.3f}"
            )
            
        except ImportError:
            fp_rate, fn_rate = fp / max(1, tp + fp), fn / max(1, tp + fn)
            self.logger.warning("Using fallback FP/FN calculation")

        # Calculate metrics
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        return {
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "true_positives": tp,
            "true_negatives": tn,
            "detection_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            # Enhanced safety margin quality assessment
            "safety_margin_quality": self._assess_safety_margin_quality(verification),
            "separation_breach_count": verification.get("violations", 0),
            "worst_case_separation_nm": verification.get("min_separation_nm", 999.0),
            "worst_case_separation_ft": verification.get("min_separation_ft", 999999.0),
            "resolution_quality_score": self._calculate_resolution_quality_score(verification),
            "low_quality_resolutions": self._count_low_quality_resolutions(verification),
        }

    def _assess_safety_margin_quality(self, verification: dict[str, Any]) -> str:
        """Assess the quality of safety margins based on ICAO standards."""
        min_sep_nm = verification.get("min_separation_nm", 999.0)
        min_sep_ft = verification.get("min_separation_ft", 999999.0)
        violations = verification.get("violations", 0)
        
        # If there are violations, it's critical
        if violations > 0:
            return "critical"
        
        # Assess based on minimum separations achieved
        # ICAO standard: 5 NM horizontal, 1000 ft vertical
        horizontal_margin = min_sep_nm - 5.0
        vertical_margin = min_sep_ft - 1000.0
        
        if horizontal_margin < 0 or vertical_margin < 0:
            return "critical"
        elif horizontal_margin < 1.0 or vertical_margin < 500.0:  # <20% buffer
            return "marginal"
        elif horizontal_margin < 2.5 or vertical_margin < 1000.0:  # <50% buffer  
            return "adequate"
        else:
            return "excellent"
    
    def _calculate_resolution_quality_score(self, verification: dict[str, Any]) -> float:
        """Calculate a quality score for LLM resolutions (0-1 scale)."""
        min_sep_nm = verification.get("min_separation_nm", 999.0)
        min_sep_ft = verification.get("min_separation_ft", 999999.0)
        violations = verification.get("violations", 0)
        resolution_success = verification.get("resolution_success", False)
        
        # Base score from resolution success
        base_score = 0.8 if resolution_success else 0.2
        
        # Penalty for violations
        violation_penalty = min(violations * 0.3, 0.6)
        
        # Bonus for good safety margins
        horizontal_bonus = min((min_sep_nm - 5.0) / 5.0 * 0.1, 0.1)  # Up to 0.1 bonus
        vertical_bonus = min((min_sep_ft - 1000.0) / 1000.0 * 0.1, 0.1)  # Up to 0.1 bonus
        
        score = base_score - violation_penalty + horizontal_bonus + vertical_bonus
        return max(0.0, min(1.0, score))
    
    def _count_low_quality_resolutions(self, verification: dict[str, Any]) -> int:
        """Count resolutions that left less than 0.5 NM safety margin."""
        min_sep_nm = verification.get("min_separation_nm", 999.0)
        
        # Count as low quality if horizontal separation < 5.5 NM (0.5 NM buffer)
        if min_sep_nm < 5.5:
            return 1
        return 0

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
            scenario_type=(
                scenario_type.value
                if hasattr(scenario_type, "value")
                else scenario_type
            ),
            complexity_tier=(
                complexity_tier.value
                if hasattr(complexity_tier, "value")
                else complexity_tier
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
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            true_positives=0,
            true_negatives=0,
            detection_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            safety_margin_quality="critical",
            separation_breach_count=999,
            worst_case_separation_nm=0.0,
            worst_case_separation_ft=0.0,
            resolution_quality_score=0.0,
            low_quality_resolutions=999,
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
                    multi_group_summary = {
                        str(key): value for key, value in raw_dict.items()
                    }
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
            "configuration": self._get_serializable_config(),
        }

    def _get_serializable_config(self) -> dict[str, Any]:
        """Get configuration as JSON-serializable dictionary with enum conversion"""
        config_dict = asdict(self.config)

        # Convert enum objects to their string values for JSON serialization
        if "scenario_types" in config_dict:
            config_dict["scenario_types"] = [
                st.value if hasattr(st, "value") else str(st)
                for st in config_dict["scenario_types"]
            ]
        if "complexity_tiers" in config_dict:
            config_dict["complexity_tiers"] = [
                ct.value if hasattr(ct, "value") else str(ct)
                for ct in config_dict["complexity_tiers"]
            ]
        # Convert scenario_counts dictionary keys from enum to string
        if "scenario_counts" in config_dict and config_dict["scenario_counts"]:
            new_scenario_counts = {}
            for key, value in config_dict["scenario_counts"].items():
                if hasattr(key, "value"):
                    new_scenario_counts[key.value] = value
                else:
                    new_scenario_counts[str(key)] = value
            config_dict["scenario_counts"] = new_scenario_counts

        return config_dict

    def _generate_summary_by_group(
        self,
        df: pd.DataFrame,
        group_column: str,
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
                "success_rate": (
                    successful_in_group / total_in_group if total_in_group > 0 else 0.0
                ),
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
                        "avg_detection_accuracy": shift_data[
                            "detection_accuracy"
                        ].mean(),
                        "avg_separation_violations": shift_data[
                            "separation_violations"
                        ].mean(),
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

    def _plot_detection_performance(
        self, df: pd.DataFrame, fig_size: tuple[int, int]
    ) -> None:
        """Plot detection performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Conflict Detection Performance", fontsize=16, fontweight="bold")

        # Accuracy histogram
        axes[0, 0].hist(
            df["detection_accuracy"],
            bins=20,
            alpha=0.7,
            color="blue",
            edgecolor="black",
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
            df.groupby(["false_positives", "false_negatives"])
            .size()
            .reset_index(name="count")
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
        success_by_complexity = df.groupby("complexity_tier")[
            "resolution_success"
        ].mean()
        axes[1, 1].bar(
            success_by_complexity.index,
            success_by_complexity.values,
            color="purple",
            alpha=0.7,
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
            df["min_separation_nm"],
            bins=30,
            alpha=0.7,
            color="blue",
            edgecolor="black",
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
            df["min_separation_ft"],
            bins=30,
            alpha=0.7,
            color="green",
            edgecolor="black",
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
        axes[1, 0].bar(
            violations_by_type.index, violations_by_type.values, color="red", alpha=0.7
        )
        axes[1, 0].set_xlabel("Scenario Type")
        axes[1, 0].set_ylabel("Total Violations")
        axes[1, 0].set_title("Separation Violations by Type")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Safety margin correlation
        axes[1, 1].scatter(
            df["safety_margin_hz"],
            df["safety_margin_vt"],
            alpha=0.6,
            color="purple",
        )
        axes[1, 1].set_xlabel("Horizontal Safety Margin (NM)")
        axes[1, 1].set_ylabel("Vertical Safety Margin (ft)")
        axes[1, 1].set_title("Safety Margin Correlation")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "visualizations" / "safety_margins.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_efficiency_metrics(
        self, df: pd.DataFrame, fig_size: tuple[int, int]
    ) -> None:
        """Plot efficiency and cost metrics"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Efficiency Metrics", fontsize=16, fontweight="bold")

        # Extra distance distribution
        axes[0, 0].hist(
            df["extra_distance_nm"],
            bins=20,
            alpha=0.7,
            color="orange",
            edgecolor="black",
        )
        axes[0, 0].set_xlabel("Extra Distance (NM)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Extra Path Distance Distribution")

        # Delay distribution
        axes[0, 1].hist(
            df["total_delay_seconds"],
            bins=20,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        axes[0, 1].set_xlabel("Total Delay (seconds)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Delay Distribution")

        # Interventions vs Efficiency
        axes[1, 0].scatter(
            df["num_interventions"],
            df["extra_distance_nm"],
            alpha=0.6,
            color="blue",
        )
        axes[1, 0].set_xlabel("Number of Interventions")
        axes[1, 0].set_ylabel("Extra Distance (NM)")
        axes[1, 0].set_title("Interventions vs Extra Distance")
        axes[1, 0].grid(True, alpha=0.3)

        # Fuel penalty by complexity
        fuel_by_complexity = df.groupby("complexity_tier")[
            "fuel_penalty_percent"
        ].mean()
        axes[1, 1].bar(
            fuel_by_complexity.index,
            fuel_by_complexity.values,
            color="green",
            alpha=0.7,
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

    def _plot_performance_by_type(
        self, df: pd.DataFrame, fig_size: tuple[int, int]
    ) -> None:
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
        axes[0, 1].bar(
            time_by_type.index, time_by_type.values, color="green", alpha=0.7
        )
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
        axes[1, 1].bar(
            success_by_type.index, success_by_type.values, color="purple", alpha=0.7
        )
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

    def _plot_distribution_shift_impact(
        self, df: pd.DataFrame, fig_size: tuple[int, int]
    ) -> None:
        """Plot impact of distribution shift on performance"""
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle("Distribution Shift Impact", fontsize=16, fontweight="bold")

        # Accuracy across shifts
        acc_by_shift = df.groupby("distribution_shift_tier")[
            "detection_accuracy"
        ].mean()
        colors = ["green", "orange", "red"]
        axes[0, 0].bar(
            acc_by_shift.index,
            acc_by_shift.values,
            color=colors[: len(acc_by_shift)],
            alpha=0.7,
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
        safety_by_shift = df.groupby("distribution_shift_tier")[
            "safety_margin_hz"
        ].mean()
        axes[1, 0].bar(
            safety_by_shift.index, safety_by_shift.values, color="purple", alpha=0.7
        )
        axes[1, 0].set_xlabel("Distribution Shift Level")
        axes[1, 0].set_ylabel("Average Safety Margin (NM)")
        axes[1, 0].set_title("Safety Margin vs Distribution Shift")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Execution time impact
        time_by_shift = df.groupby("distribution_shift_tier")[
            "execution_time_seconds"
        ].mean()
        axes[1, 1].bar(
            time_by_shift.index, time_by_shift.values, color="orange", alpha=0.7
        )
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
            if "scenario_types" in config_dict:
                config_dict["scenario_types"] = [
                    st.value if hasattr(st, "value") else str(st)
                    for st in config_dict["scenario_types"]
                ]
            if "complexity_tiers" in config_dict:
                config_dict["complexity_tiers"] = [
                    ct.value if hasattr(ct, "value") else str(ct)
                    for ct in config_dict["complexity_tiers"]
                ]
            # Convert scenario_counts dictionary keys from enum to string
            if "scenario_counts" in config_dict and config_dict["scenario_counts"]:
                new_scenario_counts = {}
                for key, value in config_dict["scenario_counts"].items():
                    if hasattr(key, "value"):
                        new_scenario_counts[key.value] = value
                    else:
                        new_scenario_counts[str(key)] = value
                config_dict["scenario_counts"] = new_scenario_counts

            json.dump(config_dict, f, indent=2)

        self.logger.info(f"Results saved to {self.output_dir}")

        # Print summary to console

    # Enhanced methods for sophisticated LLM integration
    def _run_enhanced_scenario(self, scenario: Any, scenario_id: str) -> ScenarioResult:
        """Enhanced scenario execution with detailed logging"""
        start_time = time.time()

        try:
            self.debug_logger.info(
                "=== ENHANCED SCENARIO EXECUTION ===",
                extra={"scenario_id": scenario_id},
            )

            # Run base scenario execution
            result = self._run_single_scenario(scenario, scenario_id)

            # Create enhanced comparison record
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            comparison = self._create_detection_comparison(
                scenario, scenario_id, result, execution_time
            )

            # Store comparison
            self.detection_comparisons.append(comparison)
            self._write_csv_row(comparison)

            return result

        except Exception as e:
            self.debug_logger.exception(
                "Enhanced scenario execution failed", extra={"scenario_id": scenario_id}
            )
            raise

    def _create_detection_comparison(
        self,
        scenario: Any,
        scenario_id: str,
        result: ScenarioResult,
        execution_time: float,
    ) -> DetectionComparison:
        """Create detection comparison record"""

        # Extract components from scenario_id
        parts = scenario_id.split("_")
        scenario_type = parts[0] if len(parts) > 0 else "unknown"
        complexity_tier = parts[1] if len(parts) > 1 else "unknown"
        shift_level = parts[2] if len(parts) > 2 else "unknown"

        # Get metadata from result attributes
        metadata = {
            "llm_prompt": getattr(result, "llm_prompt", ""),
            "llm_response": getattr(result, "llm_response", ""),
            "resolution_prompt": getattr(result, "resolution_prompt", ""),
            "resolution_response": getattr(result, "resolution_response", ""),
            "commands_executed": getattr(result, "llm_commands", []),
            "bluesky_conflicts": getattr(result, "bluesky_conflicts", []),
        }

        # Determine detection accuracy
        gt_conflicts = getattr(result, "num_true_conflicts", 0)
        llm_conflicts = getattr(result, "num_predicted_conflicts", 0)

        if gt_conflicts > 0 and llm_conflicts > 0:
            accuracy = "TP"  # True Positive
        elif gt_conflicts > 0 and llm_conflicts == 0:
            accuracy = "FN"  # False Negative
        elif gt_conflicts == 0 and llm_conflicts > 0:
            accuracy = "FP"  # False Positive
        else:
            accuracy = "TN"  # True Negative

        return DetectionComparison(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            complexity_tier=complexity_tier,
            shift_level=shift_level,
            ground_truth_conflicts=gt_conflicts,
            ground_truth_pairs=json.dumps([]),  # TODO: Extract actual pairs
            bluesky_conflicts=(
                len(result.bluesky_conflicts) if result.bluesky_conflicts else 0
            ),
            bluesky_pairs=json.dumps([]),
            bluesky_confidence=0.9,  # Default confidence
            llm_prompt=metadata.get("llm_prompt", ""),
            llm_response=metadata.get("llm_response", ""),
            llm_conflicts=llm_conflicts,
            llm_pairs=json.dumps([]),
            llm_confidence=0.8,  # Default confidence
            resolution_prompt=metadata.get("resolution_prompt", ""),
            resolution_response=metadata.get("resolution_response", ""),
            resolution_commands=json.dumps(metadata.get("commands_executed", [])),
            bluesky_commands_executed=json.dumps(metadata.get("commands_executed", [])),
            final_separation_status="OK" if result.success else "CONFLICT",
            detection_accuracy=accuracy,
            resolution_success=result.success,
            execution_time_ms=execution_time,
        )

    def _write_csv_row(self, comparison: DetectionComparison) -> None:
        """Write detection comparison to CSV"""
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        comparison.scenario_id,
                        comparison.scenario_type,
                        comparison.complexity_tier,
                        comparison.shift_level,
                        comparison.ground_truth_conflicts,
                        comparison.ground_truth_pairs,
                        comparison.bluesky_conflicts,
                        comparison.bluesky_pairs,
                        comparison.bluesky_confidence,
                        comparison.llm_prompt,
                        comparison.llm_response,
                        comparison.llm_conflicts,
                        comparison.llm_pairs,
                        comparison.llm_confidence,
                        comparison.resolution_prompt,
                        comparison.resolution_response,
                        comparison.resolution_commands,
                        comparison.bluesky_commands_executed,
                        comparison.final_separation_status,
                        comparison.detection_accuracy,
                        comparison.resolution_success,
                        comparison.execution_time_ms,
                    ]
                )
        except Exception as e:
            self.logger.error(f"Failed to write CSV row: {e}")

    def _save_detection_analysis(self) -> None:
        """Save detection analysis summary"""
        try:
            if not self.detection_comparisons:
                return

            # Create DataFrame for analysis
            df = pd.DataFrame([asdict(comp) for comp in self.detection_comparisons])

            # Generate analysis summary
            analysis = {
                "total_scenarios": len(df),
                "detection_accuracy": {
                    "true_positives": len(df[df["detection_accuracy"] == "TP"]),
                    "false_positives": len(df[df["detection_accuracy"] == "FP"]),
                    "true_negatives": len(df[df["detection_accuracy"] == "TN"]),
                    "false_negatives": len(df[df["detection_accuracy"] == "FN"]),
                },
                "success_rates": {
                    "overall": df["resolution_success"].mean(),
                    "by_scenario_type": df.groupby("scenario_type")[
                        "resolution_success"
                    ]
                    .mean()
                    .to_dict(),
                    "by_complexity": df.groupby("complexity_tier")["resolution_success"]
                    .mean()
                    .to_dict(),
                },
                "execution_times": {
                    "mean": df["execution_time_ms"].mean(),
                    "median": df["execution_time_ms"].median(),
                    "std": df["execution_time_ms"].std(),
                },
            }

            # Save analysis
            analysis_path = self.output_dir / "detection_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)

            self.logger.info(f"Detection analysis saved to {analysis_path}")

        except Exception as e:
            self.logger.error(f"Failed to save detection analysis: {e}")


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
            config_dict["scenario_types"] = [
                ScenarioType(t) for t in config_dict["scenario_types"]
            ]
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

    parser = argparse.ArgumentParser(
        description="Monte Carlo Benchmark Runner for LLM-ATC-HAL"
    )
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument(
        "--scenarios",
        "-n",
        type=int,
        default=10,
        help="Number of scenarios per type (default: 10)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/monte_carlo_benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

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
        "--time-horizon",
        type=float,
        default=10.0,
        help="Time horizon in minutes (default: 10.0)",
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
