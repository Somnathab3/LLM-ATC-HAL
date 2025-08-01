# agents/planner.py
"""
Planner Agent - Assesses conflicts and generates action plans
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

# Constants for separation standards and thresholds
MIN_HORIZONTAL_SEPARATION_NM = 5.0
MIN_VERTICAL_SEPARATION_FT = 1000.0
CRITICAL_HORIZONTAL_SEPARATION_NM = 1.0
CRITICAL_VERTICAL_SEPARATION_FT = 200.0
HIGH_HORIZONTAL_SEPARATION_NM = 2.0
HIGH_VERTICAL_SEPARATION_FT = 500.0
MEDIUM_HORIZONTAL_SEPARATION_NM = 3.0
MEDIUM_VERTICAL_SEPARATION_FT = 800.0
DEGREES_TO_NM_FACTOR = 60.0
DEFAULT_TIME_TO_CONFLICT_SEC = 120.0
URGENT_TIME_THRESHOLD_SEC = 60.0
NORMAL_TIME_THRESHOLD_SEC = 180.0
DEFAULT_EXPECTED_SEPARATION_INCREASE_NM = 2.0
DEFAULT_RESOLUTION_TIME_SEC = 180.0
DEFAULT_SAFETY_MARGIN_IMPROVEMENT = 0.15
DEFAULT_CONFIDENCE = 0.85


class PlanType(Enum):
    MONITOR = "monitor"
    VECTOR_CHANGE = "vector_change"
    ALTITUDE_CHANGE = "altitude_change"
    SPEED_CHANGE = "speed_change"
    EMERGENCY_DESCENT = "emergency_descent"
    HOLD_PATTERN = "hold_pattern"


@dataclass
class ConflictAssessment:
    """Assessment of a conflict situation"""

    conflict_id: str
    aircraft_involved: list[str]
    severity: str  # low, medium, high, critical
    time_to_conflict: float
    recommended_action: PlanType
    confidence: float
    reasoning: str
    metadata: dict[str, Any]


@dataclass
class ActionPlan:
    """Generated action plan for conflict resolution"""

    plan_id: str
    conflict_id: str
    plan_type: PlanType
    target_aircraft: list[str]
    commands: list[str]
    priority: int
    expected_outcome: dict[str, Any]
    confidence: float
    reasoning: str
    created_at: float


class Planner:
    """
    Planner agent responsible for conflict assessment and action plan generation
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.assessment_history: list[ConflictAssessment] = []
        self.plan_history: list[ActionPlan] = []

    def assess_conflict(
        self, aircraft_info: dict[str, Any]
    ) -> Optional[ConflictAssessment]:
        """
        Assess current aircraft situation for potential conflicts

        Args:
            aircraft_info: Dictionary containing all aircraft information

        Returns:
            ConflictAssessment or None if no conflicts detected
        """
        try:
            self.logger.info("Starting conflict assessment")

            # Extract aircraft positions and trajectories
            aircraft_data = aircraft_info.get("aircraft", {})

            if not aircraft_data:
                self.logger.info("No aircraft data available")
                return None

            # Check for proximity conflicts
            conflicts = self._detect_proximity_conflicts(aircraft_data)

            if not conflicts:
                self.logger.info("No conflicts detected")
                return None

            # Assess the most critical conflict
            critical_conflict = self._prioritize_conflicts(conflicts)

            # Generate assessment
            assessment = self._generate_assessment(critical_conflict, aircraft_data)

            # Store in history
            self.assessment_history.append(assessment)

            self.logger.info(
                "Conflict assessment completed: %s", assessment.conflict_id
            )
            return assessment

        except Exception:
            self.logger.exception("Error in conflict assessment")
            return None

    def generate_action_plan(
        self, assessment: ConflictAssessment
    ) -> Optional[ActionPlan]:
        """
        Generate detailed action plan based on conflict assessment

        Args:
            assessment: ConflictAssessment from assess_conflict

        Returns:
            ActionPlan with specific commands and expected outcomes
        """
        try:
            self.logger.info(
                "Generating action plan for conflict %s", assessment.conflict_id
            )

            # Generate plan ID
            plan_id = f"plan_{int(time.time() * 1000)}"

            # Determine appropriate action type and commands
            commands = self._generate_commands(assessment)

            # Calculate expected outcome
            expected_outcome = self._calculate_expected_outcome(assessment, commands)

            # Create action plan
            plan = ActionPlan(
                plan_id=plan_id,
                conflict_id=assessment.conflict_id,
                plan_type=assessment.recommended_action,
                target_aircraft=assessment.aircraft_involved,
                commands=commands,
                priority=self._calculate_priority(assessment),
                expected_outcome=expected_outcome,
                confidence=assessment.confidence,
                reasoning=assessment.reasoning,
                created_at=time.time(),
            )

            # Store in history
            self.plan_history.append(plan)

            self.logger.info("Action plan generated: %s", plan.plan_id)
            return plan

        except Exception:
            self.logger.exception("Error generating action plan")
            return None

    def _detect_proximity_conflicts(
        self, aircraft_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Detect proximity-based conflicts between aircraft"""
        conflicts = []
        aircraft_list = list(aircraft_data.keys())

        # Check all pairs of aircraft
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                ac1_id = aircraft_list[i]
                ac2_id = aircraft_list[j]

                ac1_data = aircraft_data[ac1_id]
                ac2_data = aircraft_data[ac2_id]

                # Calculate separation
                separation = self._calculate_separation(ac1_data, ac2_data)

                # Check if below minimum separation
                if (
                    separation["horizontal"] < MIN_HORIZONTAL_SEPARATION_NM
                    or separation["vertical"] < MIN_VERTICAL_SEPARATION_FT
                ):
                    conflicts.append(
                        {
                            "aircraft": [ac1_id, ac2_id],
                            "separation": separation,
                            "severity": self._assess_severity(separation),
                            "time_to_conflict": self._estimate_time_to_conflict(
                                ac1_data, ac2_data
                            ),
                        },
                    )

        return conflicts

    def _calculate_separation(self, ac1_data: dict, ac2_data: dict) -> dict[str, float]:
        """Calculate horizontal and vertical separation between aircraft"""
        # Simplified calculation - in real implementation would use proper geodetic calculations
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

        # Horizontal distance in nautical miles (simplified)
        horizontal_nm = (
            (lat2 - lat1) ** 2 + (lon2 - lon1) ** 2
        ) ** 0.5 * DEGREES_TO_NM_FACTOR

        # Vertical separation in feet
        vertical_ft = abs(alt2 - alt1)

        return {
            "horizontal": horizontal_nm,
            "vertical": vertical_ft,
        }

    def _assess_severity(self, separation: dict[str, float]) -> str:
        """Assess conflict severity based on separation"""
        if (
            separation["horizontal"] < CRITICAL_HORIZONTAL_SEPARATION_NM
            or separation["vertical"] < CRITICAL_VERTICAL_SEPARATION_FT
        ):
            return "critical"
        if (
            separation["horizontal"] < HIGH_HORIZONTAL_SEPARATION_NM
            or separation["vertical"] < HIGH_VERTICAL_SEPARATION_FT
        ):
            return "high"
        if (
            separation["horizontal"] < MEDIUM_HORIZONTAL_SEPARATION_NM
            or separation["vertical"] < MEDIUM_VERTICAL_SEPARATION_FT
        ):
            return "medium"
        return "low"

    def _estimate_time_to_conflict(self, _ac1_data: dict, _ac2_data: dict) -> float:
        """Estimate time to conflict in seconds"""
        # Simplified calculation based on current trajectories
        # In real implementation would use proper trajectory prediction
        return DEFAULT_TIME_TO_CONFLICT_SEC  # Default 2 minutes

    def _prioritize_conflicts(self, conflicts: list[dict]) -> dict[str, Any]:
        """Select the most critical conflict to address first"""
        if not conflicts:
            return None

        # Sort by severity and time to conflict
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        return max(
            conflicts,
            key=lambda c: (
                severity_order.get(c["severity"], 0),
                -c["time_to_conflict"],  # Negative to prioritize shorter times
            ),
        )

    def _generate_assessment(
        self, conflict: dict, aircraft_data: dict
    ) -> ConflictAssessment:
        """Generate comprehensive conflict assessment"""
        conflict_id = f"conflict_{int(time.time() * 1000)}"

        # Determine recommended action based on severity and aircraft capabilities
        recommended_action = self._determine_recommended_action(conflict, aircraft_data)

        # Generate reasoning
        reasoning = self._generate_reasoning(conflict, recommended_action)

        return ConflictAssessment(
            conflict_id=conflict_id,
            aircraft_involved=conflict["aircraft"],
            severity=conflict["severity"],
            time_to_conflict=conflict["time_to_conflict"],
            recommended_action=recommended_action,
            confidence=0.85,  # Would be calculated from LLM uncertainty
            reasoning=reasoning,
            metadata={
                "separation": conflict["separation"],
                "detection_time": time.time(),
            },
        )

    def _determine_recommended_action(
        self, conflict: dict, _aircraft_data: dict
    ) -> PlanType:
        """Determine the most appropriate action type for conflict resolution"""
        severity = conflict["severity"]

        if severity == "critical":
            return PlanType.EMERGENCY_DESCENT
        if severity == "high":
            return PlanType.ALTITUDE_CHANGE
        if severity == "medium":
            return PlanType.VECTOR_CHANGE
        return PlanType.MONITOR

    def _generate_reasoning(self, conflict: dict, action: PlanType) -> str:
        """Generate human-readable reasoning for the recommended action"""
        aircraft = ", ".join(conflict["aircraft"])
        severity = conflict["severity"]
        time_to_conflict = conflict["time_to_conflict"]

        return (
            f"Detected {severity} conflict between {aircraft} with "
            f"{time_to_conflict:.0f}s to impact. "
            f"Recommended {action.value} to ensure safe separation."
        )

    def _generate_commands(self, assessment: ConflictAssessment) -> list[str]:
        """Generate specific BlueSky commands for conflict resolution"""
        commands = []
        aircraft = assessment.aircraft_involved

        if assessment.recommended_action == PlanType.ALTITUDE_CHANGE:
            # Generate altitude change command
            for ac_id in aircraft[:1]:  # Change altitude for first aircraft
                commands.append(f"ALT {ac_id} FL350")

        elif assessment.recommended_action == PlanType.VECTOR_CHANGE:
            # Generate heading change command
            for ac_id in aircraft[:1]:
                commands.append(f"HDG {ac_id} 090")

        elif assessment.recommended_action == PlanType.SPEED_CHANGE:
            # Generate speed change command
            for ac_id in aircraft[:1]:
                commands.append(f"SPD {ac_id} 250")

        elif assessment.recommended_action == PlanType.EMERGENCY_DESCENT:
            # Generate emergency descent
            for ac_id in aircraft:
                commands.append(f"ALT {ac_id} FL100")
                commands.append(f"SPD {ac_id} 350")

        else:  # MONITOR
            commands.append("CONTINUE_MONITORING")

        return commands

    def _calculate_expected_outcome(
        self,
        _assessment: ConflictAssessment,
        _commands: list[str],
    ) -> dict[str, Any]:
        """Calculate expected outcome of executing the commands"""
        return {
            "expected_separation_increase": DEFAULT_EXPECTED_SEPARATION_INCREASE_NM,  # nm
            "resolution_time": DEFAULT_RESOLUTION_TIME_SEC,  # seconds
            "safety_margin_improvement": DEFAULT_SAFETY_MARGIN_IMPROVEMENT,
            "icao_compliance": True,
        }

    def _calculate_priority(self, assessment: ConflictAssessment) -> int:
        """Calculate plan priority (1-10, higher is more urgent)"""
        severity_priority = {
            "critical": 10,
            "high": 7,
            "medium": 5,
            "low": 3,
        }

        base_priority = severity_priority.get(assessment.severity, 1)

        # Adjust based on time to conflict
        if assessment.time_to_conflict < URGENT_TIME_THRESHOLD_SEC:
            base_priority += 2
        elif assessment.time_to_conflict < NORMAL_TIME_THRESHOLD_SEC:
            base_priority += 1

        return min(base_priority, 10)

    def get_assessment_history(self) -> list[ConflictAssessment]:
        """Get history of conflict assessments"""
        return self.assessment_history.copy()

    def get_plan_history(self) -> list[ActionPlan]:
        """Get history of generated plans"""
        return self.plan_history.copy()
