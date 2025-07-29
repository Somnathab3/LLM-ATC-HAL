# metrics/safety_margin_quantifier.py
"""
Safety Margin Quantification for ATC Conflict Resolution
Based on ICAO Doc 9689 and real-time safety assessment principles
"""

import json
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SeparationStandard(Enum):
    """ICAO separation standards"""
    HORIZONTAL_MIN = 5.0  # nautical miles
    VERTICAL_MIN = 1000   # feet
    TIME_MIN = 60         # seconds

@dataclass
class SafetyMargin:
    """Safety margin calculation result"""
    horizontal_margin: float  # nautical miles
    vertical_margin: float    # feet
    temporal_margin: float    # seconds
    effective_margin: float   # combined normalized margin
    margin_to_uncertainty_ratio: float
    degradation_factor: float
    safety_level: str  # 'critical', 'marginal', 'adequate', 'excellent'

@dataclass
class ConflictGeometry:
    """3D conflict geometry representation"""
    aircraft1_pos: Tuple[float, float, float]  # lat, lon, alt
    aircraft2_pos: Tuple[float, float, float]
    aircraft1_velocity: Tuple[float, float, float]  # ground speed, vertical rate, heading
    aircraft2_velocity: Tuple[float, float, float]
    time_to_closest_approach: float
    closest_approach_distance: float
    closest_approach_altitude_diff: float

class SafetyMarginQuantifier:
    """
    Quantifies safety margins according to ICAO standards and research best practices
    """

    def __init__(self):
        self.separation_standards = {
            "horizontal": SeparationStandard.HORIZONTAL_MIN.value,
            "vertical": SeparationStandard.VERTICAL_MIN.value,
            "temporal": SeparationStandard.TIME_MIN.value,
        }

        # Uncertainty factors for different conditions
        self.uncertainty_factors = {
            "navigation_accuracy": 0.1,  # 0.1 NM typical GPS accuracy
            "pilot_response_time": 5.0,  # 5 seconds typical response
            "turbulence_factor": 0.05,   # 5% additional uncertainty
            "equipment_error": 0.02,      # 2% equipment uncertainty
        }

        # Safety level thresholds
        self.safety_thresholds = {
            "critical": 0.2,    # < 20% of minimum separation
            "marginal": 0.5,    # 20-50% of minimum separation
            "adequate": 1.0,    # 50-100% of minimum separation
            "excellent": 2.0,    # > 100% of minimum separation
        }

    def calculate_safety_margins(self,
                                conflict_geometry: ConflictGeometry,
                                resolution_maneuver: Dict,
                                environmental_conditions: Optional[Dict] = None) -> SafetyMargin:
        """
        Calculate comprehensive safety margins for a conflict resolution
        """
        try:
            # Apply resolution maneuver to predict future geometry
            future_geometry = self._apply_resolution_maneuver(conflict_geometry, resolution_maneuver)

            # Calculate individual margin components
            horizontal_margin = self._calculate_horizontal_margin(future_geometry)
            vertical_margin = self._calculate_vertical_margin(future_geometry)
            temporal_margin = self._calculate_temporal_margin(future_geometry)

            # Account for environmental conditions
            if environmental_conditions:
                horizontal_margin *= (1 - environmental_conditions.get("turbulence_factor", 0))
                vertical_margin *= (1 - environmental_conditions.get("wind_shear_factor", 0))
                temporal_margin *= (1 - environmental_conditions.get("response_delay_factor", 0))

            # Calculate effective margin (weighted combination)
            effective_margin = self._calculate_effective_margin(
                horizontal_margin, vertical_margin, temporal_margin,
            )

            # Calculate margin-to-uncertainty ratio
            total_uncertainty = self._calculate_total_uncertainty(environmental_conditions)
            margin_uncertainty_ratio = effective_margin / (total_uncertainty + 1e-6)

            # Calculate degradation factor compared to baseline
            baseline_margin = self._calculate_baseline_margin(conflict_geometry)
            degradation_factor = effective_margin / (baseline_margin + 1e-6)

            # Determine safety level
            safety_level = self._determine_safety_level(effective_margin)

            return SafetyMargin(
                horizontal_margin=float(horizontal_margin),
                vertical_margin=float(vertical_margin),
                temporal_margin=float(temporal_margin),
                effective_margin=float(effective_margin),
                margin_to_uncertainty_ratio=float(margin_uncertainty_ratio),
                degradation_factor=float(degradation_factor),
                safety_level=safety_level,
            )

        except Exception as e:
            logging.exception(f"Safety margin calculation failed: {e}")
            return self._create_default_safety_margin()

    def _apply_resolution_maneuver(self,
                                 geometry: ConflictGeometry,
                                 maneuver: Dict) -> ConflictGeometry:
        """Apply resolution maneuver and predict future conflict geometry"""
        try:
            # Extract current positions and velocities
            ac1_pos = list(geometry.aircraft1_pos)
            ac2_pos = list(geometry.aircraft2_pos)
            ac1_vel = list(geometry.aircraft1_velocity)
            ac2_vel = list(geometry.aircraft2_velocity)

            # Apply maneuver based on type
            maneuver_type = maneuver.get("type", "").lower()
            target_aircraft = maneuver.get("aircraft_id", geometry.aircraft1_pos)  # Default to AC1

            if maneuver_type == "heading":
                heading_change = maneuver.get("heading_change", 0)
                if target_aircraft == "AC001":  # Assuming AC001 is aircraft1
                    # Convert heading change to velocity components
                    current_heading = ac1_vel[2]  # Assuming velocity[2] is heading
                    new_heading = (current_heading + heading_change) % 360
                    ac1_vel[2] = new_heading

            elif maneuver_type == "altitude":
                altitude_change = maneuver.get("altitude_change", 0)
                if target_aircraft == "AC001":
                    ac1_pos[2] += altitude_change  # Apply altitude change
                    ac1_vel[1] = altitude_change / 60  # Assume 1 minute to reach new altitude

            elif maneuver_type == "speed":
                speed_change = maneuver.get("speed_change", 0)
                if target_aircraft == "AC001":
                    ac1_vel[0] += speed_change  # Modify ground speed

            # Predict future positions after maneuver execution
            execution_time = 30  # Assume 30 seconds to execute maneuver

            # Update positions based on modified velocities
            future_ac1_pos = self._predict_position(ac1_pos, ac1_vel, execution_time)
            future_ac2_pos = self._predict_position(ac2_pos, ac2_vel, execution_time)

            # Calculate new closest approach
            time_to_ca, ca_distance, ca_alt_diff = self._calculate_closest_approach(
                future_ac1_pos, future_ac2_pos, ac1_vel, ac2_vel,
            )

            return ConflictGeometry(
                aircraft1_pos=tuple(future_ac1_pos),
                aircraft2_pos=tuple(future_ac2_pos),
                aircraft1_velocity=tuple(ac1_vel),
                aircraft2_velocity=tuple(ac2_vel),
                time_to_closest_approach=time_to_ca,
                closest_approach_distance=ca_distance,
                closest_approach_altitude_diff=ca_alt_diff,
            )

        except Exception as e:
            logging.warning(f"Failed to apply maneuver: {e}")
            return geometry  # Return original geometry if calculation fails

    def _predict_position(self, position: List[float], velocity: List[float], time: float) -> List[float]:
        """Predict future position based on current velocity"""
        try:
            # Simple linear prediction
            # position[0] = latitude, position[1] = longitude, position[2] = altitude
            # velocity[0] = ground_speed (knots), velocity[1] = vertical_rate (ft/min), velocity[2] = heading

            ground_speed_ms = velocity[0] * 0.514444  # Convert knots to m/s
            heading_rad = math.radians(velocity[2])

            # Calculate distance traveled
            distance_m = ground_speed_ms * time

            # Convert to lat/lon changes (approximate)
            lat_change = (distance_m * math.cos(heading_rad)) / 111320  # meters to degrees lat
            lon_change = (distance_m * math.sin(heading_rad)) / (111320 * math.cos(math.radians(position[0])))

            # Altitude change
            alt_change = velocity[1] * (time / 60)  # ft/min to ft over time period

            return [
                position[0] + lat_change,
                position[1] + lon_change,
                position[2] + alt_change,
            ]

        except Exception as e:
            logging.warning(f"Position prediction failed: {e}")
            return position

    def _calculate_closest_approach(self,
                                  pos1: List[float], pos2: List[float],
                                  vel1: List[float], vel2: List[float]) -> Tuple[float, float, float]:
        """Calculate time and distance of closest approach"""
        try:
            # Relative position and velocity
            rel_pos = [pos2[i] - pos1[i] for i in range(3)]
            rel_vel = [vel2[i] - vel1[i] for i in range(3)]

            # Time to closest approach (dot product calculation)
            rel_pos_magnitude = math.sqrt(sum(x**2 for x in rel_pos[:2]))  # Horizontal only
            rel_vel_magnitude = math.sqrt(sum(x**2 for x in rel_vel[:2]))  # Horizontal only

            if rel_vel_magnitude < 1e-6:  # Essentially no relative motion
                return 999999, rel_pos_magnitude, abs(rel_pos[2])

            # Simplified calculation for time to closest approach
            time_to_ca = max(0, -sum(rel_pos[i] * rel_vel[i] for i in range(2)) /
                           sum(rel_vel[i]**2 for i in range(2)))

            # Distance at closest approach
            future_rel_pos = [rel_pos[i] + rel_vel[i] * time_to_ca for i in range(3)]
            ca_distance = math.sqrt(sum(x**2 for x in future_rel_pos[:2]))
            ca_alt_diff = abs(future_rel_pos[2])

            return time_to_ca, ca_distance, ca_alt_diff

        except Exception as e:
            logging.warning(f"Closest approach calculation failed: {e}")
            return 120, 5.0, 1000  # Default values

    def _calculate_horizontal_margin(self, geometry: ConflictGeometry) -> float:
        """Calculate horizontal separation margin"""
        min_horizontal = self.separation_standards["horizontal"]
        actual_separation = geometry.closest_approach_distance

        # Convert to nautical miles if needed (assuming input is already in NM)
        margin = actual_separation - min_horizontal
        return max(margin, 0)  # Negative margin indicates violation

    def _calculate_vertical_margin(self, geometry: ConflictGeometry) -> float:
        """Calculate vertical separation margin"""
        min_vertical = self.separation_standards["vertical"]
        actual_separation = geometry.closest_approach_altitude_diff

        margin = actual_separation - min_vertical
        return max(margin, 0)  # Negative margin indicates violation

    def _calculate_temporal_margin(self, geometry: ConflictGeometry) -> float:
        """Calculate temporal margin (time available for corrective action)"""
        min_time = self.separation_standards["temporal"]
        time_to_conflict = geometry.time_to_closest_approach

        margin = time_to_conflict - min_time
        return max(margin, 0)  # Negative margin indicates immediate action needed

    def _calculate_effective_margin(self, h_margin: float, v_margin: float, t_margin: float) -> float:
        """Calculate effective combined margin using weighted approach"""
        # Weights based on criticality (horizontal separation most critical)
        weights = {"horizontal": 0.5, "vertical": 0.3, "temporal": 0.2}

        # Normalize margins to standard units
        h_normalized = h_margin / self.separation_standards["horizontal"]
        v_normalized = v_margin / self.separation_standards["vertical"]
        t_normalized = t_margin / self.separation_standards["temporal"]

        effective = (weights["horizontal"] * h_normalized +
                    weights["vertical"] * v_normalized +
                    weights["temporal"] * t_normalized)

        return max(effective, 0)

    def _calculate_total_uncertainty(self, environmental_conditions: Optional[Dict]) -> float:
        """Calculate total uncertainty in the system"""
        base_uncertainty = sum(self.uncertainty_factors.values())

        if environmental_conditions:
            # Add environmental uncertainties
            weather_uncertainty = environmental_conditions.get("weather_uncertainty", 0)
            traffic_density_factor = environmental_conditions.get("traffic_density", 1.0)

            total_uncertainty = base_uncertainty * traffic_density_factor + weather_uncertainty
        else:
            total_uncertainty = base_uncertainty

        return total_uncertainty

    def _calculate_baseline_margin(self, geometry: ConflictGeometry) -> float:
        """Calculate baseline margin without any resolution maneuver"""
        # This would be the margin if no action were taken
        baseline_horizontal = max(0, geometry.closest_approach_distance - self.separation_standards["horizontal"])
        baseline_vertical = max(0, geometry.closest_approach_altitude_diff - self.separation_standards["vertical"])
        baseline_temporal = max(0, geometry.time_to_closest_approach - self.separation_standards["temporal"])

        return self._calculate_effective_margin(baseline_horizontal, baseline_vertical, baseline_temporal)

    def _determine_safety_level(self, effective_margin: float) -> str:
        """Determine safety level based on effective margin"""
        if effective_margin < self.safety_thresholds["critical"]:
            return "critical"
        if effective_margin < self.safety_thresholds["marginal"]:
            return "marginal"
        if effective_margin < self.safety_thresholds["adequate"]:
            return "adequate"
        return "excellent"

    def _create_default_safety_margin(self) -> SafetyMargin:
        """Create default safety margin for error cases"""
        return SafetyMargin(
            horizontal_margin=0.0,
            vertical_margin=0.0,
            temporal_margin=0.0,
            effective_margin=0.0,
            margin_to_uncertainty_ratio=0.0,
            degradation_factor=1.0,
            safety_level="critical",
        )

class SafetyMetricsAggregator:
    """Aggregates safety metrics across multiple scenarios and conflicts"""

    def __init__(self):
        self.metrics_history = []
        self.quantifier = SafetyMarginQuantifier()

    def add_conflict_resolution(self,
                               conflict_id: str,
                               geometry: ConflictGeometry,
                               llm_resolution: Dict,
                               baseline_resolution: Dict,
                               environmental_conditions: Optional[Dict] = None) -> Dict:
        """Add a conflict resolution case and compute comparative metrics"""

        # Calculate safety margins for both resolutions
        llm_margins = self.quantifier.calculate_safety_margins(
            geometry, llm_resolution, environmental_conditions,
        )

        baseline_margins = self.quantifier.calculate_safety_margins(
            geometry, baseline_resolution, environmental_conditions,
        )

        # Create comparison metrics
        comparison = {
            "conflict_id": conflict_id,
            "timestamp": time.time(),
            "llm_margins": llm_margins,
            "baseline_margins": baseline_margins,
            "margin_difference": llm_margins.effective_margin - baseline_margins.effective_margin,
            "safety_degradation": baseline_margins.effective_margin - llm_margins.effective_margin,
            "uncertainty_ratio_diff": (llm_margins.margin_to_uncertainty_ratio -
                                     baseline_margins.margin_to_uncertainty_ratio),
            "environmental_conditions": environmental_conditions or {},
        }

        self.metrics_history.append(comparison)
        return comparison

    def generate_safety_summary(self) -> Dict:
        """Generate comprehensive safety summary across all conflicts"""
        if not self.metrics_history:
            return {"error": "No metrics data available"}

        # Extract key metrics
        margin_differences = [m["margin_difference"] for m in self.metrics_history]
        safety_degradations = [m["safety_degradation"] for m in self.metrics_history]
        llm_safety_levels = [m["llm_margins"].safety_level for m in self.metrics_history]
        baseline_safety_levels = [m["baseline_margins"].safety_level for m in self.metrics_history]

        # Calculate statistics
        summary = {
            "total_conflicts": len(self.metrics_history),
            "average_margin_difference": float(np.mean(margin_differences)),
            "std_margin_difference": float(np.std(margin_differences)),
            "max_safety_degradation": float(max(safety_degradations)),
            "min_safety_degradation": float(min(safety_degradations)),
            "llm_critical_cases": sum(1 for level in llm_safety_levels if level == "critical"),
            "baseline_critical_cases": sum(1 for level in baseline_safety_levels if level == "critical"),
            "safety_improvement_cases": sum(1 for diff in margin_differences if diff > 0),
            "safety_degradation_cases": sum(1 for diff in margin_differences if diff < 0),
            "safety_level_distribution": {
                "llm": {level: llm_safety_levels.count(level) for level in ["critical", "marginal", "adequate", "excellent"]},
                "baseline": {level: baseline_safety_levels.count(level) for level in ["critical", "marginal", "adequate", "excellent"]},
            },
        }

        return summary

    def export_detailed_metrics(self, filepath: str):
        """Export detailed metrics to JSON file"""
        try:
            # Convert SafetyMargin objects to dictionaries for JSON serialization
            exportable_data = []
            for metric in self.metrics_history:
                exportable_metric = {
                    "conflict_id": metric["conflict_id"],
                    "timestamp": metric["timestamp"],
                    "llm_margins": {
                        "horizontal_margin": metric["llm_margins"].horizontal_margin,
                        "vertical_margin": metric["llm_margins"].vertical_margin,
                        "temporal_margin": metric["llm_margins"].temporal_margin,
                        "effective_margin": metric["llm_margins"].effective_margin,
                        "margin_to_uncertainty_ratio": metric["llm_margins"].margin_to_uncertainty_ratio,
                        "degradation_factor": metric["llm_margins"].degradation_factor,
                        "safety_level": metric["llm_margins"].safety_level,
                    },
                    "baseline_margins": {
                        "horizontal_margin": metric["baseline_margins"].horizontal_margin,
                        "vertical_margin": metric["baseline_margins"].vertical_margin,
                        "temporal_margin": metric["baseline_margins"].temporal_margin,
                        "effective_margin": metric["baseline_margins"].effective_margin,
                        "margin_to_uncertainty_ratio": metric["baseline_margins"].margin_to_uncertainty_ratio,
                        "degradation_factor": metric["baseline_margins"].degradation_factor,
                        "safety_level": metric["baseline_margins"].safety_level,
                    },
                    "margin_difference": metric["margin_difference"],
                    "safety_degradation": metric["safety_degradation"],
                    "uncertainty_ratio_diff": metric["uncertainty_ratio_diff"],
                    "environmental_conditions": metric["environmental_conditions"],
                }
                exportable_data.append(exportable_metric)

            with open(filepath, "w") as f:
                json.dump(exportable_data, f, indent=2)

            logging.info(f"Safety metrics exported to {filepath}")

        except Exception as e:
            logging.exception(f"Failed to export metrics: {e}")


def calc_separation_margin(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate horizontal and vertical separation margins from trajectories.
    
    Args:
        trajectories: List of aircraft trajectories with format:
                     [{'aircraft_id': str, 'path': [{'lat': float, 'lon': float, 
                       'alt': float, 'time': float}]}]
    
    Returns:
        Dict with 'hz' (horizontal) and 'vt' (vertical) margins in nm and ft
    """
    if len(trajectories) < 2:
        return {"hz": float("inf"), "vt": float("inf")}

    min_horizontal = float("inf")
    min_vertical = float("inf")

    # Compare all pairs of aircraft
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            traj1 = trajectories[i]["path"]
            traj2 = trajectories[j]["path"]

            # Find closest approach between these two aircraft
            for point1 in traj1:
                for point2 in traj2:
                    # Only compare points at similar times (±30 seconds)
                    if abs(point1["time"] - point2["time"]) <= 30:
                        # Calculate horizontal distance in nautical miles
                        lat1, lon1 = math.radians(point1["lat"]), math.radians(point1["lon"])
                        lat2, lon2 = math.radians(point2["lat"]), math.radians(point2["lon"])

                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = (math.sin(dlat/2)**2 +
                             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
                        c = 2 * math.asin(math.sqrt(a))
                        horizontal_dist = 3440.065 * c  # Convert to nautical miles

                        # Calculate vertical distance in feet
                        vertical_dist = abs(point1["alt"] - point2["alt"])

                        min_horizontal = min(min_horizontal, horizontal_dist)
                        min_vertical = min(min_vertical, vertical_dist)

    return {
        "hz": max(0, min_horizontal - SeparationStandard.HORIZONTAL_MIN.value),
        "vt": max(0, min_vertical - SeparationStandard.VERTICAL_MIN.value),
    }


def calc_efficiency_penalty(planned_path: List[Dict[str, Any]],
                           executed_path: List[Dict[str, Any]]) -> float:
    """
    Calculate efficiency penalty as extra distance traveled due to conflict resolution.
    
    Args:
        planned_path: Original planned trajectory points
                     [{'lat': float, 'lon': float, 'alt': float, 'time': float}]
        executed_path: Actual executed trajectory points (same format)
    
    Returns:
        Extra distance in nautical miles
    """
    def calculate_path_distance(path: List[Dict[str, Any]]) -> float:
        """Calculate total distance of a path in nautical miles"""
        total_distance = 0.0

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            lat1, lon1 = math.radians(p1["lat"]), math.radians(p1["lon"])
            lat2, lon2 = math.radians(p2["lat"]), math.radians(p2["lon"])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat/2)**2 +
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            c = 2 * math.asin(math.sqrt(a))
            distance = 3440.065 * c  # Convert to nautical miles
            total_distance += distance

        return total_distance

    if not planned_path or not executed_path:
        return 0.0

    planned_distance = calculate_path_distance(planned_path)
    executed_distance = calculate_path_distance(executed_path)

    return max(0.0, executed_distance - planned_distance)


def count_interventions(commands: List[Dict[str, Any]]) -> int:
    """
    Count the number of ATC interventions in a command sequence.
    
    Args:
        commands: List of ATC commands with format:
                 [{'type': str, 'aircraft_id': str, 'timestamp': float, ...}]
    
    Returns:
        Number of intervention commands
    """
    if not commands:
        return 0

    intervention_types = {
        "heading_change", "altitude_change", "speed_change",
        "vector", "climb", "descend", "turn", "direct",
        "hold", "expedite", "reduce_speed",
    }

    intervention_count = 0

    for command in commands:
        cmd_type = command.get("type", "").lower()

        # Count as intervention if it's a control command
        if any(interv_type in cmd_type for interv_type in intervention_types):
            intervention_count += 1

        # Also count based on specific fields
        if any(key in command for key in ["heading_change", "altitude_change", "speed_change"]):
            intervention_count += 1

    return intervention_count


# Example usage and testing
if __name__ == "__main__":
    import time

    # Create test scenario
    conflict_geometry = ConflictGeometry(
        aircraft1_pos=(52.3, 4.8, 35000),
        aircraft2_pos=(52.4, 4.6, 35000),
        aircraft1_velocity=(350, 0, 90),  # 350 knots, level, heading 90
        aircraft2_velocity=(350, 0, 270), # 350 knots, level, heading 270
        time_to_closest_approach=120,
        closest_approach_distance=3.0,
        closest_approach_altitude_diff=0,
    )

    llm_resolution = {
        "type": "heading",
        "aircraft_id": "AC001",
        "heading_change": 20,
        "safety_score": 0.8,
    }

    baseline_resolution = {
        "type": "altitude",
        "aircraft_id": "AC001",
        "altitude_change": 1000,
        "safety_score": 0.9,
    }

    # Test safety margin calculation
    quantifier = SafetyMarginQuantifier()
    margins = quantifier.calculate_safety_margins(conflict_geometry, llm_resolution)

    print("Safety Margin Analysis:")
    print(f"Horizontal Margin: {margins.horizontal_margin:.2f} NM")
    print(f"Vertical Margin: {margins.vertical_margin:.0f} ft")
    print(f"Temporal Margin: {margins.temporal_margin:.0f} seconds")
    print(f"Effective Margin: {margins.effective_margin:.3f}")
    print(f"Margin/Uncertainty Ratio: {margins.margin_to_uncertainty_ratio:.2f}")
    print(f"Safety Level: {margins.safety_level}")

    # Test aggregator
    aggregator = SafetyMetricsAggregator()
    comparison = aggregator.add_conflict_resolution(
        "TEST-001", conflict_geometry, llm_resolution, baseline_resolution,
    )

    summary = aggregator.generate_safety_summary()
    print("\nSafety Summary:")
    print(json.dumps(summary, indent=2))
