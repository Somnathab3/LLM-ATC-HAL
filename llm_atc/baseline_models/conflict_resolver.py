# baseline_models/conflict_resolver.py
"""
Baseline Conflict Resolver using Rule-based Vertical-then-Lateral Heuristic
Traditional ATC resolution strategy as baseline comparison
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ManeuverType(Enum):
    """Types of conflict resolution maneuvers"""

    ALTITUDE_CHANGE = "altitude_change"
    HEADING_CHANGE = "heading_change"
    SPEED_CHANGE = "speed_change"
    VECTOR = "vector"
    HOLD = "hold"
    NO_ACTION = "no_action"


@dataclass
class ResolutionManeuver:
    """Conflict resolution maneuver"""

    aircraft_id: str
    maneuver_type: ManeuverType
    parameters: dict[str, float]  # e.g., {'altitude_change': 1000, 'heading_change': 20}
    priority: int  # 1=high, 2=medium, 3=low
    safety_score: float  # 0.0-1.0
    estimated_delay: float  # seconds
    fuel_penalty: float  # additional fuel consumption


class BaselineConflictResolver:
    """
    Rule-based conflict resolver implementing vertical-then-lateral strategy.
    Follows traditional ATC procedures as baseline for comparison.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Standard separation requirements (ICAO)
        self.min_horizontal_separation = 5.0  # nautical miles
        self.min_vertical_separation = 1000  # feet

        # Standard maneuver parameters
        self.standard_altitude_change = 1000  # feet
        self.standard_heading_change = 20  # degrees
        self.standard_speed_change = 20  # knots

        # Resolution preferences (priority order)
        self.resolution_hierarchy = [
            ManeuverType.ALTITUDE_CHANGE,
            ManeuverType.HEADING_CHANGE,
            ManeuverType.SPEED_CHANGE,
            ManeuverType.VECTOR,
            ManeuverType.HOLD,
        ]

    def resolve_conflicts(self, conflict_scenario: dict[str, Any]) -> list[ResolutionManeuver]:
        """
        Generate conflict resolution maneuvers using rule-based approach.

        Args:
            conflict_scenario: Dictionary containing:
                - aircraft: List of aircraft with states
                - conflicts: List of detected conflicts
                - environmental_conditions: Weather, traffic density, etc.

        Returns:
            List of resolution maneuvers
        """
        aircraft_list = conflict_scenario.get("aircraft", [])
        conflicts = conflict_scenario.get("conflicts", [])
        environmental = conflict_scenario.get("environmental_conditions", {})

        if not conflicts:
            return []

        maneuvers = []

        # Process each conflict using vertical-then-lateral strategy
        for conflict in conflicts:
            conflict_maneuvers = self._resolve_single_conflict(
                conflict,
                aircraft_list,
                environmental,
            )
            maneuvers.extend(conflict_maneuvers)

        # Remove duplicate maneuvers for same aircraft
        maneuvers = self._deduplicate_maneuvers(maneuvers)

        # Sort by priority
        maneuvers.sort(key=lambda x: x.priority)

        return maneuvers

    def _resolve_single_conflict(
        self,
        conflict: dict[str, Any],
        aircraft_list: list[dict[str, Any]],
        environmental: dict[str, Any],
    ) -> list[ResolutionManeuver]:
        """Resolve a single conflict using rule-based strategy"""

        ac1_id = conflict.get("aircraft1_id", conflict.get("id1", ""))
        ac2_id = conflict.get("aircraft2_id", conflict.get("id2", ""))

        # Find aircraft data
        ac1_data = next((ac for ac in aircraft_list if ac.get("id") == ac1_id), None)
        ac2_data = next((ac for ac in aircraft_list if ac.get("id") == ac2_id), None)

        if not ac1_data or not ac2_data:
            self.logger.warning("Aircraft data not found for conflict %s-%s", ac1_id, ac2_id)
            return []

        # Analyze conflict geometry
        geometry = self._analyze_conflict_geometry(ac1_data, ac2_data, conflict)

        # Apply vertical-then-lateral strategy
        maneuvers = []

        # 1. Try vertical separation first (preferred)
        vertical_maneuver = self._try_vertical_resolution(
            ac1_data,
            ac2_data,
            geometry,
            environmental,
        )
        if vertical_maneuver:
            maneuvers.append(vertical_maneuver)
        else:
            # 2. Try lateral separation (heading change)
            lateral_maneuver = self._try_lateral_resolution(
                ac1_data,
                ac2_data,
                geometry,
                environmental,
            )
            if lateral_maneuver:
                maneuvers.append(lateral_maneuver)
            else:
                # 3. Try speed-based resolution
                speed_maneuver = self._try_speed_resolution(
                    ac1_data,
                    ac2_data,
                    geometry,
                    environmental,
                )
                if speed_maneuver:
                    maneuvers.append(speed_maneuver)
                else:
                    # 4. Last resort: vector or hold
                    fallback_maneuver = self._try_fallback_resolution(
                        ac1_data,
                        ac2_data,
                        geometry,
                        environmental,
                    )
                    if fallback_maneuver:
                        maneuvers.append(fallback_maneuver)

        return maneuvers

    def _try_vertical_resolution(
        self, ac1: dict, ac2: dict, geometry: dict, environmental: dict,
    ) -> Optional[ResolutionManeuver]:
        """Try to resolve conflict with altitude change"""

        # Determine which aircraft should change altitude
        ac1_alt = ac1.get("alt", 0)
        ac2_alt = ac2.get("alt", 0)

        # Priority rules:
        # 1. Aircraft at lower altitude climbs (if possible)
        # 2. Consider flight phases and aircraft capabilities
        # 3. Minimize disruption to traffic flow

        target_aircraft = ac1 if ac1_alt <= ac2_alt else ac2
        other_aircraft = ac2 if target_aircraft == ac1 else ac1

        # Check if altitude change is feasible
        current_alt = target_aircraft.get("alt", 0)
        flight_phase = target_aircraft.get("flight_phase", "cruise")

        # Determine altitude change direction and amount
        if flight_phase in ["climb", "cruise"] and current_alt < 40000:
            # Climb
            altitude_change = self.standard_altitude_change
        elif flight_phase in ["descent", "cruise"] and current_alt > 20000:
            # Descend
            altitude_change = -self.standard_altitude_change
        else:
            # No feasible altitude change
            return None

        # Calculate safety score
        safety_score = self._calculate_vertical_safety_score(
            target_aircraft,
            other_aircraft,
            altitude_change,
            environmental,
        )

        # Estimate delay and fuel penalty
        estimated_delay = self._estimate_altitude_delay(altitude_change)
        fuel_penalty = self._estimate_altitude_fuel_penalty(altitude_change)

        return ResolutionManeuver(
            aircraft_id=target_aircraft["id"],
            maneuver_type=ManeuverType.ALTITUDE_CHANGE,
            parameters={"altitude_change": altitude_change},
            priority=1,  # High priority for vertical separation
            safety_score=safety_score,
            estimated_delay=estimated_delay,
            fuel_penalty=fuel_penalty,
        )

    def _try_lateral_resolution(
        self, ac1: dict, ac2: dict, geometry: dict, environmental: dict,
    ) -> Optional[ResolutionManeuver]:
        """Try to resolve conflict with heading change"""

        # Determine which aircraft should change heading
        # Usually the aircraft with more flexibility or lower priority

        ac1_priority = self._get_aircraft_priority(ac1)
        ac2_priority = self._get_aircraft_priority(ac2)

        target_aircraft = ac1 if ac1_priority >= ac2_priority else ac2

        # Calculate optimal heading change
        current_heading = target_aircraft.get("heading", 0)
        conflict_bearing = geometry.get("bearing", 0)

        # Turn away from conflict (perpendicular avoidance)
        heading_change = self._calculate_optimal_heading_change(
            current_heading,
            conflict_bearing,
            geometry,
        )

        # Calculate safety score
        safety_score = self._calculate_lateral_safety_score(
            target_aircraft,
            heading_change,
            environmental,
        )

        # Estimate delay and fuel penalty
        estimated_delay = self._estimate_heading_delay(heading_change)
        fuel_penalty = self._estimate_heading_fuel_penalty(heading_change)

        return ResolutionManeuver(
            aircraft_id=target_aircraft["id"],
            maneuver_type=ManeuverType.HEADING_CHANGE,
            parameters={"heading_change": heading_change},
            priority=2,  # Medium priority for lateral separation
            safety_score=safety_score,
            estimated_delay=estimated_delay,
            fuel_penalty=fuel_penalty,
        )

    def _try_speed_resolution(
        self, ac1: dict, ac2: dict, geometry: dict, environmental: dict,
    ) -> Optional[ResolutionManeuver]:
        """Try to resolve conflict with speed change"""

        # Determine which aircraft should change speed
        ac1_speed = ac1.get("speed", 250)
        ac2_speed = ac2.get("speed", 250)

        # Usually slow down the faster aircraft or speed up the slower one
        if ac1_speed > ac2_speed:
            target_aircraft = ac1
            speed_change = -self.standard_speed_change
        else:
            target_aircraft = ac2
            speed_change = self.standard_speed_change

        # Check speed limits
        current_speed = target_aircraft.get("speed", 250)
        new_speed = current_speed + speed_change

        # Apply speed limits based on aircraft type and flight phase
        min_speed, max_speed = self._get_speed_limits(target_aircraft)

        if new_speed < min_speed or new_speed > max_speed:
            return None  # Speed change not feasible

        # Calculate safety score
        safety_score = self._calculate_speed_safety_score(
            target_aircraft,
            speed_change,
            environmental,
        )

        # Estimate delay and fuel penalty
        estimated_delay = self._estimate_speed_delay(speed_change)
        fuel_penalty = self._estimate_speed_fuel_penalty(speed_change)

        return ResolutionManeuver(
            aircraft_id=target_aircraft["id"],
            maneuver_type=ManeuverType.SPEED_CHANGE,
            parameters={"speed_change": speed_change},
            priority=3,  # Lower priority for speed changes
            safety_score=safety_score,
            estimated_delay=estimated_delay,
            fuel_penalty=fuel_penalty,
        )

    def _try_fallback_resolution(
        self, ac1: dict, ac2: dict, geometry: dict, environmental: dict,
    ) -> Optional[ResolutionManeuver]:
        """Try fallback resolution (vector or hold)"""

        # Choose aircraft with lower priority for fallback maneuver
        ac1_priority = self._get_aircraft_priority(ac1)
        ac2_priority = self._get_aircraft_priority(ac2)

        target_aircraft = ac1 if ac1_priority >= ac2_priority else ac2

        # Default to vector maneuver (heading + speed)
        heading_change = 30  # More aggressive turn
        speed_change = -10  # Slight speed reduction

        safety_score = 0.6  # Lower safety score for fallback
        estimated_delay = 180  # 3 minutes
        fuel_penalty = 15  # kg

        return ResolutionManeuver(
            aircraft_id=target_aircraft["id"],
            maneuver_type=ManeuverType.VECTOR,
            parameters={
                "heading_change": heading_change,
                "speed_change": speed_change,
            },
            priority=4,  # Low priority
            safety_score=safety_score,
            estimated_delay=estimated_delay,
            fuel_penalty=fuel_penalty,
        )

    def _analyze_conflict_geometry(self, ac1: dict, ac2: dict, conflict: dict) -> dict[str, float]:
        """Analyze geometric relationship between conflicting aircraft"""

        # Calculate bearing between aircraft
        lat1, lon1 = ac1.get("lat", 0), ac1.get("lon", 0)
        lat2, lon2 = ac2.get("lat", 0), ac2.get("lon", 0)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        bearing = math.atan2(dlon, dlat) * 180 / math.pi

        # Calculate distance
        distance = math.sqrt(dlat**2 + dlon**2) * 60  # Approximate nm

        # Calculate altitude difference
        alt_diff = abs(ac1.get("alt", 0) - ac2.get("alt", 0))

        # Calculate relative speed
        speed1 = ac1.get("speed", 250)
        speed2 = ac2.get("speed", 250)
        heading1 = ac1.get("heading", 0)
        heading2 = ac2.get("heading", 0)

        # Relative velocity components
        vx1 = speed1 * math.sin(math.radians(heading1))
        vy1 = speed1 * math.cos(math.radians(heading1))
        vx2 = speed2 * math.sin(math.radians(heading2))
        vy2 = speed2 * math.cos(math.radians(heading2))

        rel_speed = math.sqrt((vx1 - vx2) ** 2 + (vy1 - vy2) ** 2)

        return {
            "bearing": bearing,
            "distance": distance,
            "altitude_difference": alt_diff,
            "relative_speed": rel_speed,
            "time_to_closest_approach": distance / max(rel_speed, 1) * 3600,  # seconds
        }

    def _get_aircraft_priority(self, aircraft: dict) -> int:
        """Get aircraft priority (1=highest, 3=lowest)"""
        aircraft_type = aircraft.get("type", "commercial").lower()
        flight_phase = aircraft.get("flight_phase", "cruise").lower()

        # Priority based on type
        base_priority = 3  # Default priority
        if aircraft_type in {"emergency", "military"}:
            base_priority = 1
        elif aircraft_type == "commercial":
            base_priority = 2

        # Adjust for flight phase
        if flight_phase in ["takeoff", "landing", "approach"]:
            return max(1, base_priority - 1)  # Higher priority for critical phases

        return base_priority

    def _calculate_optimal_heading_change(
        self, current_heading: float, conflict_bearing: float, geometry: dict,
    ) -> float:
        """Calculate optimal heading change to avoid conflict"""

        # Turn perpendicular to conflict bearing
        relative_bearing = conflict_bearing - current_heading

        # Choose shorter turn direction
        if -180 <= relative_bearing <= 0:
            heading_change = -self.standard_heading_change
        else:
            heading_change = self.standard_heading_change

        return heading_change

    def _calculate_vertical_safety_score(
        self, target_ac: dict, other_ac: dict, altitude_change: float, env: dict,
    ) -> float:
        """Calculate safety score for altitude change"""
        base_score = 0.85

        # Reduce score for extreme altitude changes
        if abs(altitude_change) > 2000:
            base_score -= 0.1

        # Consider weather
        weather_severity = env.get("weather_severity", 0)
        base_score -= weather_severity * 0.2

        # Consider traffic density
        traffic_density = env.get("traffic_density", 0.5)
        base_score -= traffic_density * 0.1

        return max(0.5, base_score)

    def _calculate_lateral_safety_score(
        self, target_ac: dict, heading_change: float, env: dict,
    ) -> float:
        """Calculate safety score for heading change"""
        base_score = 0.75

        # Reduce score for large heading changes
        if abs(heading_change) > 30:
            base_score -= 0.15

        # Consider environmental factors
        weather_severity = env.get("weather_severity", 0)
        base_score -= weather_severity * 0.15

        return max(0.5, base_score)

    def _calculate_speed_safety_score(
        self, target_ac: dict, speed_change: float, env: dict,
    ) -> float:
        """Calculate safety score for speed change"""
        base_score = 0.70

        # Consider speed change magnitude
        if abs(speed_change) > 30:
            base_score -= 0.1

        return max(0.5, base_score)

    def _get_speed_limits(self, aircraft: dict) -> tuple[float, float]:
        """Get speed limits for aircraft"""
        aircraft_type = aircraft.get("type", "commercial").lower()
        flight_phase = aircraft.get("flight_phase", "cruise").lower()

        if flight_phase == "cruise":
            if aircraft_type == "commercial":
                return (220, 350)  # Typical cruise speeds
            return (200, 300)
        if flight_phase in ["approach", "landing"]:
            return (120, 250)
        return (150, 280)

    def _estimate_altitude_delay(self, altitude_change: float) -> float:
        """Estimate delay caused by altitude change"""
        # Assume 1000 fpm climb/descent rate
        return abs(altitude_change) / 1000 * 60  # seconds

    def _estimate_heading_delay(self, heading_change: float) -> float:
        """Estimate delay caused by heading change"""
        # Approximate delay based on heading change
        return abs(heading_change) * 3  # seconds per degree

    def _estimate_speed_delay(self, speed_change: float) -> float:
        """Estimate delay caused by speed change"""
        # Speed changes cause minimal direct delay
        return abs(speed_change) * 2  # seconds

    def _estimate_altitude_fuel_penalty(self, altitude_change: float) -> float:
        """Estimate fuel penalty for altitude change"""
        return abs(altitude_change) * 0.05  # kg per foot

    def _estimate_heading_fuel_penalty(self, heading_change: float) -> float:
        """Estimate fuel penalty for heading change"""
        return abs(heading_change) * 0.5  # kg per degree

    def _estimate_speed_fuel_penalty(self, speed_change: float) -> float:
        """Estimate fuel penalty for speed change"""
        return abs(speed_change) * 0.3  # kg per knot

    def _deduplicate_maneuvers(
        self, maneuvers: list[ResolutionManeuver],
    ) -> list[ResolutionManeuver]:
        """Remove duplicate maneuvers for same aircraft"""
        seen_aircraft = set()
        unique_maneuvers = []

        for maneuver in maneuvers:
            if maneuver.aircraft_id not in seen_aircraft:
                unique_maneuvers.append(maneuver)
                seen_aircraft.add(maneuver.aircraft_id)

        return unique_maneuvers


# Example usage and testing
if __name__ == "__main__":
    # Create sample conflict scenario
    conflict_scenario = {
        "aircraft": [
            {
                "id": "AC001",
                "lat": 52.0,
                "lon": 4.0,
                "alt": 35000,
                "speed": 250,
                "heading": 90,
                "type": "commercial",
                "flight_phase": "cruise",
            },
            {
                "id": "AC002",
                "lat": 52.05,
                "lon": 4.05,
                "alt": 35000,
                "speed": 260,
                "heading": 270,
                "type": "commercial",
                "flight_phase": "cruise",
            },
        ],
        "conflicts": [
            {
                "aircraft1_id": "AC001",
                "aircraft2_id": "AC002",
                "time_to_conflict": 300,
                "closest_approach_distance": 3.5,
            },
        ],
        "environmental_conditions": {
            "weather_severity": 0.2,
            "traffic_density": 0.4,
        },
    }

    # Test resolver
    resolver = BaselineConflictResolver()
    maneuvers = resolver.resolve_conflicts(conflict_scenario)

    for _maneuver in maneuvers:
        pass
