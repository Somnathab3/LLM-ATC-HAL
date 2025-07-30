# scenarios/scenario_generator.py
"""
Environment-Specific Scenario Generation Module
==============================================
This module encapsulates environment-specific scenario creation logic for
Horizontal, Vertical and Sector conflict scenarios, wrapping the existing
Monte Carlo generator.

Implements three environment classes as specified:
- HorizontalCREnv: Same-altitude conflict scenarios
- VerticalCREnv: Altitude-based conflict scenarios
- SectorCREnv: Full-sector realistic scenarios

Each environment provides targeted scenario generation with precise
ground truth conflict labeling for false positive/negative analysis.
"""

import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

# Import from existing Monte Carlo framework
from scenarios.monte_carlo_framework import (
    BlueSkyScenarioGenerator,
    ComplexityTier,
    ScenarioConfiguration,
)


class ScenarioType(Enum):
    """Environment-specific scenario types"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    SECTOR = "sector"


@dataclass
class GroundTruthConflict:
    """Ground truth conflict information for validation"""
    aircraft_pair: tuple[str, str]
    conflict_type: str  # 'horizontal', 'vertical', 'convergent', 'overtaking'
    time_to_conflict: float  # seconds
    min_separation: dict[str, float]  # {'horizontal_nm': X, 'vertical_ft': Y}
    severity: str  # 'low', 'medium', 'high', 'critical'
    is_actual_conflict: bool  # True if separation will be violated


@dataclass
class Scenario:
    """Enhanced scenario representation with ground truth"""
    # Core scenario data
    scenario_id: str
    scenario_type: ScenarioType
    aircraft_count: int
    commands: list[str]  # BlueSky commands
    initial_states: list[dict[str, Any]]  # Aircraft initial states

    # Ground truth information
    ground_truth_conflicts: list[GroundTruthConflict]
    expected_conflict_count: int
    has_conflicts: bool

    # Metadata
    complexity_tier: ComplexityTier
    generation_timestamp: float
    environmental_conditions: dict[str, Any]
    airspace_region: str

    # Additional scenario configuration
    duration_minutes: float = 10.0
    distribution_shift_tier: str = "in_distribution"

    # Extended fields for benchmark runner
    predicted_conflicts: list[GroundTruthConflict] = None  # LLM predicted conflicts
    resolution_commands: list[str] = None  # Resolution commands from LLM
    success: Optional[bool] = None  # Whether resolution was successful
    trajectories: list[dict[str, Any]] = None  # Aircraft trajectory recordings

    def __post_init__(self):
        """Initialize optional fields to sensible defaults"""
        if self.predicted_conflicts is None:
            self.predicted_conflicts = []
        if self.resolution_commands is None:
            self.resolution_commands = []
        if self.trajectories is None:
            self.trajectories = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with existing code"""
        return asdict(self)


class ScenarioGenerator:
    """
    Main scenario generator wrapping BlueSkyScenarioGenerator for
    environment-specific scenario creation.
    """

    def __init__(self, ranges_file: str = "scenario_ranges.yaml",
                 distribution_shift_file: str = "distribution_shift_levels.yaml") -> None:
        """
        Initialize scenario generator.

        Args:
            ranges_file: Path to scenario ranges YAML
            distribution_shift_file: Path to distribution shift config YAML
        """
        self.base_generator = BlueSkyScenarioGenerator(
            ranges_file=ranges_file,
            distribution_shift_file=distribution_shift_file,
        )
        self.logger = logging.getLogger(__name__)

        # Standard separation minimums (ICAO)
        self.MIN_HORIZONTAL_SEP_NM = 5.0  # nautical miles
        self.MIN_VERTICAL_SEP_FT = 1000.0  # feet
        self.CRITICAL_HORIZONTAL_SEP_NM = 3.0  # critical threshold
        self.CRITICAL_VERTICAL_SEP_FT = 500.0  # critical threshold

    def generate_scenario(self, scenario_type: ScenarioType, **kwargs) -> Scenario:
        """
        Dispatcher method to generate scenarios by type.

        Args:
            scenario_type: Type of scenario to generate
            **kwargs: Type-specific arguments

        Returns:
            Generated scenario with ground truth
        """
        if scenario_type == ScenarioType.HORIZONTAL:
            return self.generate_horizontal_scenario(**kwargs)
        if scenario_type == ScenarioType.VERTICAL:
            return self.generate_vertical_scenario(**kwargs)
        if scenario_type == ScenarioType.SECTOR:
            return self.generate_sector_scenario(**kwargs)
        msg = f"Unknown scenario type: {scenario_type}"
        raise ValueError(msg)

    def generate_horizontal_scenario(self,
                                   n_aircraft: int = 2,
                                   conflict: bool = True,
                                   complexity_tier: ComplexityTier = ComplexityTier.SIMPLE,
                                   distribution_shift_tier: str = "in_distribution") -> Scenario:
        """
        Generate horizontal conflict scenario.

        All aircraft at same flight level to eliminate vertical separation.
        Adjust headings to create/avoid horizontal conflicts.

        Args:
            n_aircraft: Number of aircraft (2-5)
            conflict: Whether to create conflicts (True) or safe scenarios (False)
            complexity_tier: Scenario complexity
            distribution_shift_tier: Distribution shift level

        Returns:
            Horizontal conflict scenario with ground truth
        """
        self.logger.info(f"Generating horizontal scenario: {n_aircraft} aircraft, conflict={conflict}")

        # Generate base scenario using Monte Carlo framework
        base_scenario = self.base_generator.generate_scenario(
            complexity_tier=complexity_tier,
            force_conflicts=conflict,
            distribution_shift_tier=distribution_shift_tier,
        )

        # Override aircraft count if specified
        if n_aircraft != base_scenario.aircraft_count:
            # Regenerate with specific aircraft count by modifying ranges
            custom_ranges = self.base_generator.ranges.copy()
            custom_ranges["aircraft"]["count"][complexity_tier.value] = [n_aircraft, n_aircraft]

            custom_generator = BlueSkyScenarioGenerator(ranges_dict=custom_ranges)
            base_scenario = custom_generator.generate_scenario(
                complexity_tier=complexity_tier,
                force_conflicts=conflict,
                distribution_shift_tier=distribution_shift_tier,
            )

        # Force all aircraft to same altitude (eliminate vertical separation)
        standard_altitude = 35000  # FL350
        modified_commands = []
        modified_states = []

        for i, aircraft in enumerate(base_scenario.aircraft_list):
            callsign = f"AC{i+1:03d}"

            # Create aircraft with standard altitude
            create_cmd = f"CRE {callsign},{aircraft['aircraft_type']},{aircraft['latitude']},{aircraft['longitude']},{aircraft['heading']},{standard_altitude},{aircraft['ground_speed']}"
            modified_commands.append(create_cmd)

            # Store initial state
            initial_state = {
                "callsign": callsign,
                "aircraft_type": aircraft["aircraft_type"],
                "latitude": aircraft["latitude"],
                "longitude": aircraft["longitude"],
                "altitude": standard_altitude,
                "heading": aircraft["heading"],
                "ground_speed": aircraft["ground_speed"],
                "vertical_rate": 0,
            }
            modified_states.append(initial_state)

        # Adjust headings for conflict creation/avoidance
        if conflict and n_aircraft >= 2:
            modified_commands.extend(self._create_horizontal_conflicts(modified_states))
        elif not conflict:
            modified_commands.extend(self._avoid_horizontal_conflicts(modified_states))

        # Add environmental commands
        modified_commands.extend(self._add_environmental_commands(base_scenario.environmental_conditions))

        # Calculate ground truth conflicts
        ground_truth_conflicts = self._calculate_horizontal_ground_truth(modified_states, conflict)

        # Create scenario
        scenario_id = f"horizontal_{int(time.time())}_{random.randint(1000, 9999)}"
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.HORIZONTAL,
            aircraft_count=len(modified_states),
            commands=modified_commands,
            initial_states=modified_states,
            ground_truth_conflicts=ground_truth_conflicts,
            expected_conflict_count=len(ground_truth_conflicts),
            has_conflicts=len(ground_truth_conflicts) > 0,
            complexity_tier=complexity_tier,
            generation_timestamp=time.time(),
            environmental_conditions=base_scenario.environmental_conditions,
            airspace_region=base_scenario.airspace_region,
            distribution_shift_tier=distribution_shift_tier,
        )

        self.logger.info(f"Generated horizontal scenario {scenario_id} with {len(ground_truth_conflicts)} conflicts")
        return scenario

    def generate_vertical_scenario(self,
                                 n_aircraft: int = 3,
                                 conflict: bool = True,
                                 climb_rates: Optional[list[int]] = None,
                                 crossing_altitudes: Optional[list[int]] = None,
                                 complexity_tier: ComplexityTier = ComplexityTier.SIMPLE,
                                 distribution_shift_tier: str = "in_distribution") -> Scenario:
        """
        Generate vertical conflict scenario.

        Aircraft at different altitudes with climb/descent creating vertical conflicts.

        Args:
            n_aircraft: Number of aircraft (2-5)
            conflict: Whether to create conflicts (True) or safe scenarios (False)
            climb_rates: List of climb/descent rates in fpm (default: [-1500, 0, 1500])
            crossing_altitudes: List of target altitudes for vertical maneuvers (default: auto-generated)
            complexity_tier: Scenario complexity
            distribution_shift_tier: Distribution shift level

        Returns:
            Vertical conflict scenario with ground truth
        """
        self.logger.info(f"Generating vertical scenario: {n_aircraft} aircraft, conflict={conflict}")

        # Set default climb rates if not provided
        if climb_rates is None:
            climb_rates = [-1500, 0, 1500, -1000, 1000]  # feet per minute

        # Generate base scenario
        base_scenario = self.base_generator.generate_scenario(
            complexity_tier=complexity_tier,
            force_conflicts=False,  # We'll create our own vertical conflicts
            distribution_shift_tier=distribution_shift_tier,
        )

        # Limit to reasonable number for vertical conflicts
        n_aircraft = min(n_aircraft, 5)

        modified_commands = []
        modified_states = []

        # Set up aircraft at different altitudes
        if crossing_altitudes is None:
            # Auto-generate crossing altitudes based on conflict requirement
            if conflict:
                # Create altitudes that will intersect when vertical maneuvers are applied
                crossing_altitudes = [33000, 35000, 37000, 34000, 36000][:n_aircraft]
            else:
                # Create well-separated altitudes for safe scenarios
                crossing_altitudes = [31000 + i * 3000 for i in range(n_aircraft)]

        # Ensure we have enough crossing altitudes
        while len(crossing_altitudes) < n_aircraft:
            crossing_altitudes.append(crossing_altitudes[-1] + 2000)

        for i in range(n_aircraft):
            aircraft = base_scenario.aircraft_list[i] if i < len(base_scenario.aircraft_list) else base_scenario.aircraft_list[0]
            callsign = f"AC{i+1:03d}"

            # Assign initial altitude (offset from crossing altitude)
            if conflict:
                # Start at different altitude from target to create crossing paths
                initial_altitude = crossing_altitudes[i] + (2000 if i % 2 == 0 else -2000)
            else:
                # Start at safe altitude with no crossing potential
                initial_altitude = crossing_altitudes[i]

            # Create aircraft
            create_cmd = f"CRE {callsign},{aircraft['aircraft_type']},{aircraft['latitude']},{aircraft['longitude']},{aircraft['heading']},{initial_altitude},{aircraft['ground_speed']}"
            modified_commands.append(create_cmd)

            # Store initial state
            initial_state = {
                "callsign": callsign,
                "aircraft_type": aircraft["aircraft_type"],
                "latitude": aircraft["latitude"],
                "longitude": aircraft["longitude"],
                "altitude": initial_altitude,
                "heading": aircraft["heading"],
                "ground_speed": aircraft["ground_speed"],
                "vertical_rate": 0,
                "target_altitude": crossing_altitudes[i],
                "assigned_climb_rate": climb_rates[i % len(climb_rates)],
            }
            modified_states.append(initial_state)

        # Add vertical maneuvers to create/avoid conflicts
        if conflict and n_aircraft >= 2:
            modified_commands.extend(self._create_vertical_conflicts_enhanced(modified_states, climb_rates))
        elif not conflict:
            modified_commands.extend(self._avoid_vertical_conflicts_enhanced(modified_states, climb_rates))

        # Add environmental commands
        modified_commands.extend(self._add_environmental_commands(base_scenario.environmental_conditions))

        # Calculate ground truth conflicts
        ground_truth_conflicts = self._calculate_vertical_ground_truth(modified_states, conflict)

        # Create scenario
        scenario_id = f"vertical_{int(time.time())}_{random.randint(1000, 9999)}"
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.VERTICAL,
            aircraft_count=len(modified_states),
            commands=modified_commands,
            initial_states=modified_states,
            ground_truth_conflicts=ground_truth_conflicts,
            expected_conflict_count=len(ground_truth_conflicts),
            has_conflicts=len(ground_truth_conflicts) > 0,
            complexity_tier=complexity_tier,
            generation_timestamp=time.time(),
            environmental_conditions=base_scenario.environmental_conditions,
            airspace_region=base_scenario.airspace_region,
            distribution_shift_tier=distribution_shift_tier,
        )

        self.logger.info(f"Generated vertical scenario {scenario_id} with {len(ground_truth_conflicts)} conflicts")
        return scenario

    def generate_sector_scenario(self,
                               complexity: ComplexityTier = ComplexityTier.MODERATE,
                               shift_level: str = "in_distribution",
                               force_conflicts: bool = False) -> Scenario:
        """
        Generate realistic sector scenario.

        Uses full Monte Carlo generation for organic sector scenarios.

        Args:
            complexity: Scenario complexity tier
            shift_level: Distribution shift level
            force_conflicts: Whether to force conflicts (False for realistic scenarios)

        Returns:
            Sector scenario with ground truth
        """
        self.logger.info(f"Generating sector scenario: {complexity.value}, shift={shift_level}, force_conflicts={force_conflicts}")

        # Generate base scenario using full Monte Carlo framework
        base_scenario = self.base_generator.generate_scenario(
            complexity_tier=complexity,
            force_conflicts=force_conflicts,
            distribution_shift_tier=shift_level,
        )

        # Convert to our scenario format
        initial_states = []
        for i, aircraft in enumerate(base_scenario.aircraft_list):
            callsign = f"AC{i+1:03d}"
            initial_state = {
                "callsign": callsign,
                "aircraft_type": aircraft["aircraft_type"],
                "latitude": aircraft["latitude"],
                "longitude": aircraft["longitude"],
                "altitude": aircraft["altitude"],
                "heading": aircraft["heading"],
                "ground_speed": aircraft["ground_speed"],
                "vertical_rate": aircraft.get("vertical_rate", 0),
            }
            initial_states.append(initial_state)

        # Calculate ground truth for sector scenario
        ground_truth_conflicts = self._calculate_sector_ground_truth(initial_states, base_scenario)

        # Create scenario
        scenario_id = f"sector_{int(time.time())}_{random.randint(1000, 9999)}"
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.SECTOR,
            aircraft_count=base_scenario.aircraft_count,
            commands=base_scenario.bluesky_commands,
            initial_states=initial_states,
            ground_truth_conflicts=ground_truth_conflicts,
            expected_conflict_count=len(ground_truth_conflicts),
            has_conflicts=len(ground_truth_conflicts) > 0,
            complexity_tier=complexity,
            generation_timestamp=time.time(),
            environmental_conditions=base_scenario.environmental_conditions,
            airspace_region=base_scenario.airspace_region,
            duration_minutes=base_scenario.duration_minutes,
            distribution_shift_tier=shift_level,
        )

        self.logger.info(f"Generated sector scenario {scenario_id} with {len(ground_truth_conflicts)} conflicts")
        return scenario

    def _create_horizontal_conflicts(self, aircraft_states: list[dict[str, Any]]) -> list[str]:
        """Create heading adjustments to generate horizontal conflicts"""
        commands = []

        if len(aircraft_states) >= 2:
            # Make first two aircraft converge
            ac1 = aircraft_states[0]
            ac2 = aircraft_states[1]

            # Calculate bearing between aircraft
            bearing = self._calculate_bearing(
                ac1["latitude"], ac1["longitude"],
                ac2["latitude"], ac2["longitude"],
            )

            # Set convergent headings
            hdg1 = int(bearing) % 360
            hdg2 = int(bearing + 180) % 360

            commands.append(f"HDG {ac1['callsign']} {hdg1}")
            commands.append(f"HDG {ac2['callsign']} {hdg2}")

            # Update states
            aircraft_states[0]["heading"] = hdg1
            aircraft_states[1]["heading"] = hdg2

        return commands

    def _avoid_horizontal_conflicts(self, aircraft_states: list[dict[str, Any]]) -> list[str]:
        """Create heading adjustments to avoid horizontal conflicts"""
        commands = []

        # Spread aircraft headings to avoid convergence
        base_heading = 0
        heading_increment = 360 // len(aircraft_states)

        for i, aircraft in enumerate(aircraft_states):
            safe_heading = (base_heading + i * heading_increment) % 360
            commands.append(f"HDG {aircraft['callsign']} {safe_heading}")
            aircraft["heading"] = safe_heading

        return commands

    def _create_vertical_conflicts(self, aircraft_states: list[dict[str, Any]]) -> list[str]:
        """Create altitude/climb commands to generate vertical conflicts"""
        commands = []

        if len(aircraft_states) >= 2:
            # First aircraft climbs, second descends to create vertical conflict
            ac1 = aircraft_states[0]
            ac2 = aircraft_states[1]

            # Aircraft 1 climbs from lower altitude
            target_alt1 = ac2["altitude"]  # Climb to AC2's altitude
            commands.append(f"ALT {ac1['callsign']} {target_alt1}")

            # Aircraft 2 descends slightly to create near-miss
            target_alt2 = ac1["altitude"]  # Descend to AC1's altitude
            commands.append(f"ALT {ac2['callsign']} {target_alt2}")

            # Add vertical rates for realism
            commands.append(f"VS {ac1['callsign']} 1500")  # 1500 fpm climb
            commands.append(f"VS {ac2['callsign']} -1500")  # 1500 fpm descent

            # Update states to reflect commands
            aircraft_states[0]["vertical_rate"] = 1500
            aircraft_states[1]["vertical_rate"] = -1500

        return commands

    def _avoid_vertical_conflicts(self, aircraft_states: list[dict[str, Any]]) -> list[str]:
        """Ensure safe vertical separation"""
        commands = []

        # Ensure all aircraft have safe vertical separation (>1000 ft)
        for i, aircraft in enumerate(aircraft_states):
            safe_altitude = 33000 + (i * 2000)  # 2000 ft separation
            if aircraft["altitude"] != safe_altitude:
                commands.append(f"ALT {aircraft['callsign']} {safe_altitude}")
                aircraft["altitude"] = safe_altitude

        return commands

    def _create_vertical_conflicts_enhanced(self, aircraft_states: list[dict[str, Any]],
                                          climb_rates: list[int]) -> list[str]:
        """Enhanced vertical conflict creation with configurable climb rates"""
        commands = []

        # Create crossing vertical paths that will result in conflicts
        for i in range(len(aircraft_states)):
            aircraft = aircraft_states[i]
            callsign = aircraft["callsign"]
            current_alt = aircraft["altitude"]
            target_alt = aircraft["target_altitude"]
            climb_rate = aircraft["assigned_climb_rate"]

            # Only issue altitude commands if different from current
            if target_alt != current_alt:
                commands.append(f"ALT {callsign} {target_alt}")

                # Set vertical speed if non-zero climb rate
                if climb_rate != 0:
                    commands.append(f"VS {callsign} {climb_rate}")
                    aircraft["vertical_rate"] = climb_rate

        # Calculate conflict timing to ensure near-threshold separation
        if len(aircraft_states) >= 2:
            self._optimize_conflict_timing(aircraft_states, commands)

        return commands

    def _avoid_vertical_conflicts_enhanced(self, aircraft_states: list[dict[str, Any]],
                                         climb_rates: list[int]) -> list[str]:
        """Enhanced vertical conflict avoidance ensuring >1000ft separation"""
        commands = []

        # Calculate safe altitudes ensuring no vertical conflicts
        safe_altitudes = []
        min_separation = 1500  # feet - buffer above ICAO minimum

        for i, aircraft in enumerate(aircraft_states):
            # Calculate safe altitude based on other aircraft paths
            safe_alt = 30000 + (i * min_separation)

            # Verify this altitude doesn't conflict with any other aircraft's path
            for j, other_aircraft in enumerate(aircraft_states):
                if i != j:
                    other_target = other_aircraft.get("target_altitude", other_aircraft["altitude"])
                    # Ensure separation at all times during climb/descent
                    while abs(safe_alt - other_target) < min_separation:
                        safe_alt += min_separation

            safe_altitudes.append(safe_alt)

            # Issue altitude command if needed
            if aircraft["altitude"] != safe_alt:
                commands.append(f"ALT {aircraft['callsign']} {safe_alt}")
                aircraft["altitude"] = safe_alt
                aircraft["target_altitude"] = safe_alt

                # Use conservative climb rate for safety
                safe_climb_rate = 500 if safe_alt > aircraft["altitude"] else -500
                commands.append(f"VS {aircraft['callsign']} {safe_climb_rate}")
                aircraft["vertical_rate"] = safe_climb_rate

        return commands

    def _optimize_conflict_timing(self, aircraft_states: list[dict[str, Any]],
                                commands: list[str]) -> None:
        """Optimize timing to create near-threshold vertical conflicts"""
        # Add timing commands to ensure conflicts occur within simulation window
        for i in range(len(aircraft_states) - 1):
            ac1 = aircraft_states[i]
            ac2 = aircraft_states[i + 1]

            # Calculate time when vertical paths will intersect
            alt_diff = abs(ac1["target_altitude"] - ac2["target_altitude"])
            rate_sum = abs(ac1.get("vertical_rate", 0)) + abs(ac2.get("vertical_rate", 0))

            if rate_sum > 0:
                conflict_time_min = (alt_diff / rate_sum) * 60  # Convert to minutes

                # If conflict occurs too late, add delayed commands
                if conflict_time_min > 3:  # Start maneuvers after 3 minutes
                    delay_sec = 180  # 3 minutes
                    # Add delayed commands (would need simulation timing logic)
                    commands.append(f"# Delayed maneuver at {delay_sec}s")

    def _add_environmental_commands(self, env_conditions: dict[str, Any]) -> list[str]:
        """Add environmental condition commands"""
        commands = []

        # Add wind if specified
        if "wind_speed_kts" in env_conditions and env_conditions["wind_speed_kts"] > 0:
            wind_dir = env_conditions.get("wind_direction_deg", 270)
            wind_speed = env_conditions["wind_speed_kts"]
            commands.append(f"WIND {wind_dir} {wind_speed}")

        # Add turbulence if specified
        if "turbulence_intensity" in env_conditions and env_conditions["turbulence_intensity"] > 0:
            turb_level = min(int(env_conditions["turbulence_intensity"] * 10), 9)
            commands.append(f"TURB {turb_level}")

        return commands

    def _calculate_horizontal_ground_truth(self, aircraft_states: list[dict[str, Any]],
                                         expect_conflicts: bool) -> list[GroundTruthConflict]:
        """Calculate ground truth conflicts for horizontal scenarios"""
        conflicts = []

        # For horizontal scenarios, check all pairs for convergent trajectories
        for i in range(len(aircraft_states)):
            for j in range(i + 1, len(aircraft_states)):
                ac1 = aircraft_states[i]
                ac2 = aircraft_states[j]

                # Calculate horizontal distance
                distance_nm = self._calculate_distance_nm(
                    ac1["latitude"], ac1["longitude"],
                    ac2["latitude"], ac2["longitude"],
                )

                # Check if headings are convergent
                is_convergent = self._are_headings_convergent(
                    ac1["latitude"], ac1["longitude"], ac1["heading"],
                    ac2["latitude"], ac2["longitude"], ac2["heading"],
                )

                if expect_conflicts and is_convergent:
                    # Estimate time to conflict (simplified)
                    avg_speed = (ac1["ground_speed"] + ac2["ground_speed"]) / 2
                    time_to_conflict = (distance_nm / avg_speed) * 3600  # Convert to seconds

                    conflict = GroundTruthConflict(
                        aircraft_pair=(ac1["callsign"], ac2["callsign"]),
                        conflict_type="horizontal",
                        time_to_conflict=time_to_conflict,
                        min_separation={
                            "horizontal_nm": min(distance_nm, self.CRITICAL_HORIZONTAL_SEP_NM),
                            "vertical_ft": 0,  # Same altitude
                        },
                        severity="high" if distance_nm < self.CRITICAL_HORIZONTAL_SEP_NM else "medium",
                        is_actual_conflict=distance_nm < self.MIN_HORIZONTAL_SEP_NM,
                    )
                    conflicts.append(conflict)

        return conflicts

    def _calculate_vertical_ground_truth(self, aircraft_states: list[dict[str, Any]],
                                       expect_conflicts: bool) -> list[GroundTruthConflict]:
        """Calculate ground truth conflicts for vertical scenarios"""
        conflicts = []

        for i in range(len(aircraft_states)):
            for j in range(i + 1, len(aircraft_states)):
                ac1 = aircraft_states[i]
                ac2 = aircraft_states[j]

                # Calculate vertical separation
                vertical_sep_ft = abs(ac1["altitude"] - ac2["altitude"])

                # Check if vertical rates will cause conflict
                vr1 = ac1.get("vertical_rate", 0)
                vr2 = ac2.get("vertical_rate", 0)

                if expect_conflicts and (vr1 != 0 or vr2 != 0):
                    # Estimate time when altitudes will be closest
                    if vr1 != vr2:  # Aircraft have different vertical rates
                        time_to_closest = vertical_sep_ft / abs(vr1 - vr2) * 60  # Convert to seconds

                        conflict = GroundTruthConflict(
                            aircraft_pair=(ac1["callsign"], ac2["callsign"]),
                            conflict_type="vertical",
                            time_to_conflict=time_to_closest,
                            min_separation={
                                "horizontal_nm": 0,  # Assuming same horizontal position
                                "vertical_ft": min(vertical_sep_ft, self.CRITICAL_VERTICAL_SEP_FT),
                            },
                            severity="critical" if vertical_sep_ft < self.CRITICAL_VERTICAL_SEP_FT else "high",
                            is_actual_conflict=vertical_sep_ft < self.MIN_VERTICAL_SEP_FT,
                        )
                        conflicts.append(conflict)

        return conflicts

    def _calculate_sector_ground_truth(self, aircraft_states: list[dict[str, Any]],
                                     base_scenario: ScenarioConfiguration) -> list[GroundTruthConflict]:
        """Calculate ground truth conflicts for sector scenarios using trajectory analysis"""
        conflicts = []

        # Analyze all aircraft pairs for potential conflicts
        for i in range(len(aircraft_states)):
            for j in range(i + 1, len(aircraft_states)):
                ac1 = aircraft_states[i]
                ac2 = aircraft_states[j]

                # Calculate current separation
                self._calculate_distance_nm(
                    ac1["latitude"], ac1["longitude"],
                    ac2["latitude"], ac2["longitude"],
                )
                abs(ac1["altitude"] - ac2["altitude"])

                # Check if trajectories will converge
                conflict_analysis = self._analyze_trajectory_conflict(ac1, ac2)

                if conflict_analysis["has_conflict"]:
                    severity = self._determine_conflict_severity(
                        conflict_analysis["min_horizontal_sep"],
                        conflict_analysis["min_vertical_sep"],
                    )

                    conflict = GroundTruthConflict(
                        aircraft_pair=(ac1["callsign"], ac2["callsign"]),
                        conflict_type=conflict_analysis["conflict_type"],
                        time_to_conflict=conflict_analysis["time_to_conflict"],
                        min_separation={
                            "horizontal_nm": conflict_analysis["min_horizontal_sep"],
                            "vertical_ft": conflict_analysis["min_vertical_sep"],
                        },
                        severity=severity,
                        is_actual_conflict=conflict_analysis["violates_separation"],
                    )
                    conflicts.append(conflict)

        return conflicts

    def _analyze_trajectory_conflict(self, ac1: dict[str, Any], ac2: dict[str, Any]) -> dict[str, Any]:
        """Analyze if two aircraft trajectories will conflict"""
        # Simplified trajectory analysis
        # In a full implementation, this would use proper 4D trajectory prediction

        # Current positions
        lat1, lon1, alt1 = ac1["latitude"], ac1["longitude"], ac1["altitude"]
        lat2, lon2, alt2 = ac2["latitude"], ac2["longitude"], ac2["altitude"]

        # Current separation
        horizontal_dist = self._calculate_distance_nm(lat1, lon1, lat2, lon2)
        vertical_sep = abs(alt1 - alt2)

        # Simple prediction: project 5 minutes ahead
        projection_time = 5.0  # minutes

        # Project positions based on headings and speeds
        new_lat1, new_lon1 = self._project_position(
            lat1, lon1, ac1["heading"], ac1["ground_speed"], projection_time,
        )
        new_lat2, new_lon2 = self._project_position(
            lat2, lon2, ac2["heading"], ac2["ground_speed"], projection_time,
        )

        # Project altitudes based on vertical rates
        vr1 = ac1.get("vertical_rate", 0)
        vr2 = ac2.get("vertical_rate", 0)
        new_alt1 = alt1 + (vr1 * projection_time)
        new_alt2 = alt2 + (vr2 * projection_time)

        # Calculate future separation
        future_horizontal_dist = self._calculate_distance_nm(new_lat1, new_lon1, new_lat2, new_lon2)
        future_vertical_sep = abs(new_alt1 - new_alt2)

        # Determine if this constitutes a conflict
        min_horizontal_sep = min(horizontal_dist, future_horizontal_dist)
        min_vertical_sep = min(vertical_sep, future_vertical_sep)

        has_conflict = (
            min_horizontal_sep < self.MIN_HORIZONTAL_SEP_NM * 1.5 and  # Within 1.5x minimum
            min_vertical_sep < self.MIN_VERTICAL_SEP_FT * 1.5
        )

        violates_separation = (
            min_horizontal_sep < self.MIN_HORIZONTAL_SEP_NM and
            min_vertical_sep < self.MIN_VERTICAL_SEP_FT
        )

        # Estimate time to closest approach
        if has_conflict:
            time_to_conflict = projection_time * 60 / 2  # Rough estimate in seconds
        else:
            time_to_conflict = float("inf")

        # Determine conflict type
        if vertical_sep < self.MIN_VERTICAL_SEP_FT and min_vertical_sep < vertical_sep:
            conflict_type = "vertical"
        elif self._are_headings_convergent(lat1, lon1, ac1["heading"], lat2, lon2, ac2["heading"]):
            conflict_type = "convergent"
        else:
            conflict_type = "parallel"

        return {
            "has_conflict": has_conflict,
            "violates_separation": violates_separation,
            "time_to_conflict": time_to_conflict,
            "min_horizontal_sep": min_horizontal_sep,
            "min_vertical_sep": min_vertical_sep,
            "conflict_type": conflict_type,
        }

    def _determine_conflict_severity(self, horizontal_sep: float, vertical_sep: float) -> str:
        """Determine conflict severity based on separation"""
        if (horizontal_sep < self.CRITICAL_HORIZONTAL_SEP_NM and
            vertical_sep < self.CRITICAL_VERTICAL_SEP_FT):
            return "critical"
        if (horizontal_sep < self.MIN_HORIZONTAL_SEP_NM and
              vertical_sep < self.MIN_VERTICAL_SEP_FT):
            return "high"
        if (horizontal_sep < self.MIN_HORIZONTAL_SEP_NM * 1.5 or
              vertical_sep < self.MIN_VERTICAL_SEP_FT * 1.5):
            return "medium"
        return "low"

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)

        return (bearing_deg + 360) % 360

    def _calculate_distance_nm(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        # Haversine formula
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)

        a = (math.sin(dlat_rad/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        # Earth radius in nautical miles
        R_nm = 3440.065
        return R_nm * c


    def _are_headings_convergent(self, lat1: float, lon1: float, hdg1: float,
                                lat2: float, lon2: float, hdg2: float) -> bool:
        """Check if two aircraft headings are convergent"""
        # Calculate bearing from AC1 to AC2
        bearing_1_to_2 = self._calculate_bearing(lat1, lon1, lat2, lon2)
        bearing_2_to_1 = (bearing_1_to_2 + 180) % 360

        # Check if headings are roughly pointing toward each other
        hdg1_diff = abs(hdg1 - bearing_1_to_2)
        hdg2_diff = abs(hdg2 - bearing_2_to_1)

        # Account for circular nature of headings
        hdg1_diff = min(hdg1_diff, 360 - hdg1_diff)
        hdg2_diff = min(hdg2_diff, 360 - hdg2_diff)

        # Consider convergent if both aircraft are heading roughly toward each other
        return hdg1_diff < 45 and hdg2_diff < 45

    def _project_position(self, lat: float, lon: float, heading: float,
                         speed_kts: float, time_min: float) -> tuple[float, float]:
        """Project aircraft position based on heading and speed"""
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        heading_rad = math.radians(heading)

        # Distance traveled in nautical miles
        distance_nm = speed_kts * (time_min / 60.0)

        # Earth radius in nautical miles
        R_nm = 3440.065
        angular_distance = distance_nm / R_nm

        # Calculate new position
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(heading_rad),
        )

        new_lon_rad = lon_rad + math.atan2(
            math.sin(heading_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad),
        )

        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


# Environment-specific classes as requested
class HorizontalCREnv:
    """Horizontal Conflict Resolution Environment"""

    def __init__(self, generator: Optional[ScenarioGenerator] = None) -> None:
        self.generator = generator or ScenarioGenerator()

    def generate_scenario(self, n_aircraft: int = 2, conflict: bool = True, **kwargs) -> Scenario:
        """Generate horizontal conflict scenario"""
        return self.generator.generate_horizontal_scenario(
            n_aircraft=n_aircraft,
            conflict=conflict,
            **kwargs,
        )


class VerticalCREnv:
    """Vertical Conflict Resolution Environment"""

    def __init__(self, generator: Optional[ScenarioGenerator] = None) -> None:
        self.generator = generator or ScenarioGenerator()

    def generate_scenario(self, n_aircraft: int = 2, conflict: bool = True, **kwargs) -> Scenario:
        """Generate vertical conflict scenario"""
        return self.generator.generate_vertical_scenario(
            n_aircraft=n_aircraft,
            conflict=conflict,
            **kwargs,
        )


class SectorCREnv:
    """Sector Conflict Resolution Environment"""

    def __init__(self, generator: Optional[ScenarioGenerator] = None) -> None:
        self.generator = generator or ScenarioGenerator()

    def generate_scenario(self, complexity: ComplexityTier = ComplexityTier.MODERATE,
                         shift_level: str = "in_distribution",
                         force_conflicts: bool = False, **kwargs) -> Scenario:
        """Generate sector scenario"""
        return self.generator.generate_sector_scenario(
            complexity=complexity,
            shift_level=shift_level,
            force_conflicts=force_conflicts,
            **kwargs,
        )


# Convenience functions for compatibility
def generate_horizontal_scenario(n_aircraft: int = 2, conflict: bool = True, **kwargs) -> Scenario:
    """Generate horizontal scenario - convenience function"""
    generator = ScenarioGenerator()
    return generator.generate_horizontal_scenario(n_aircraft, conflict, **kwargs)


def generate_vertical_scenario(n_aircraft: int = 2, conflict: bool = True, **kwargs) -> Scenario:
    """Generate vertical scenario - convenience function"""
    generator = ScenarioGenerator()
    return generator.generate_vertical_scenario(n_aircraft, conflict, **kwargs)


def generate_sector_scenario(complexity: ComplexityTier = ComplexityTier.MODERATE,
                            shift_level: str = "in_distribution",
                            force_conflicts: bool = False, **kwargs) -> Scenario:
    """Generate sector scenario - convenience function"""
    generator = ScenarioGenerator()
    return generator.generate_sector_scenario(complexity, shift_level, force_conflicts, **kwargs)


# Example usage
if __name__ == "__main__":
    # Demonstration code moved to tests/test_scenario_generator.py
    # Run comprehensive tests with:
    #   python tests/test_scenario_generator.py
    #   python tests/test_scenario_generator_enhanced.py

    pass
