# analysis/enhanced_hallucination_detection.py
"""
Enhanced Hallucination Detection for LLM-ATC-HAL Framework
Detects various types of hallucinations in ATC decision-making
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class HallucinationType(Enum):
    """Types of hallucinations that can occur in ATC decisions"""
    AIRCRAFT_EXISTENCE = "aircraft_existence"
    ALTITUDE_CONFUSION = "altitude_confusion"
    HEADING_CONFUSION = "heading_confusion"
    AIRSPACE_VIOLATION = "airspace_violation"
    PROTOCOL_VIOLATION = "protocol_violation"
    IMPOSSIBLE_MANEUVER = "impossible_maneuver"
    NONSENSICAL_RESPONSE = "nonsensical_response"

@dataclass
class HallucinationResult:
    """Result of hallucination detection"""
    detected: bool
    confidence: float
    types: list[HallucinationType]
    explanation: str
    severity: str  # 'low', 'medium', 'high', 'critical'

class EnhancedHallucinationDetector:
    """
    Enhanced hallucination detector using multiple detection strategies
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._init_detection_patterns()

    def _init_detection_patterns(self) -> None:
        """Initialize detection patterns for various hallucination types"""

        # Patterns for common ATC hallucinations
        self.aircraft_id_pattern = re.compile(r"[A-Z]{3}[0-9]{1,4}")
        self.altitude_pattern = re.compile(r"FL[0-9]{3}|[0-9]{3,5}\s*(?:ft|feet)")
        self.heading_pattern = re.compile(r"[0-9]{1,3}(?:\.[0-9])?°?")

        # Known valid aircraft types
        self.valid_aircraft_types = {
            "A320", "A330", "A340", "A350", "A380",
            "B737", "B747", "B757", "B767", "B777", "B787",
            "CRJ200", "CRJ700", "CRJ900", "DHC8", "E170", "E190",
        }

        # Valid altitude ranges
        self.min_altitude = 0
        self.max_altitude = 60000

        # Valid heading range
        self.min_heading = 0
        self.max_heading = 360

    def detect_hallucinations(self, llm_response: dict[str, Any],
                            baseline_response: dict[str, Any],
                            context: dict[str, Any]) -> HallucinationResult:
        """
        Detect hallucinations in LLM response using multiple strategies

        Args:
            llm_response: Response from LLM model
            baseline_response: Baseline/ground truth response
            context: Context information including scenario data

        Returns:
            HallucinationResult with detection details
        """

        detected_types = []
        explanations = []
        max_confidence = 0.0

        try:
            # Extract response text
            response_text = llm_response.get("decision_text", "")

            # 1. Aircraft existence check
            aircraft_hallucination = self._check_aircraft_existence(response_text, context)
            if aircraft_hallucination:
                detected_types.append(HallucinationType.AIRCRAFT_EXISTENCE)
                explanations.append("Referenced non-existent aircraft")
                max_confidence = max(max_confidence, 0.9)

            # 2. Altitude validity check
            altitude_hallucination = self._check_altitude_validity(response_text)
            if altitude_hallucination:
                detected_types.append(HallucinationType.ALTITUDE_CONFUSION)
                explanations.append("Invalid altitude values detected")
                max_confidence = max(max_confidence, 0.8)

            # 3. Heading validity check
            heading_hallucination = self._check_heading_validity(response_text)
            if heading_hallucination:
                detected_types.append(HallucinationType.HEADING_CONFUSION)
                explanations.append("Invalid heading values detected")
                max_confidence = max(max_confidence, 0.8)

            # 4. Protocol violation check
            protocol_hallucination = self._check_protocol_violations(response_text)
            if protocol_hallucination:
                detected_types.append(HallucinationType.PROTOCOL_VIOLATION)
                explanations.append("ATC protocol violations detected")
                max_confidence = max(max_confidence, 0.7)

            # 5. Impossible maneuver check
            maneuver_hallucination = self._check_impossible_maneuvers(response_text, context)
            if maneuver_hallucination:
                detected_types.append(HallucinationType.IMPOSSIBLE_MANEUVER)
                explanations.append("Physically impossible maneuvers suggested")
                max_confidence = max(max_confidence, 0.9)

            # 6. Nonsensical response check
            nonsensical_hallucination = self._check_nonsensical_response(response_text)
            if nonsensical_hallucination:
                detected_types.append(HallucinationType.NONSENSICAL_RESPONSE)
                explanations.append("Response contains nonsensical content")
                max_confidence = max(max_confidence, 0.95)

            # Determine overall detection and severity
            detected = len(detected_types) > 0
            severity = self._determine_severity(detected_types)

            return HallucinationResult(
                detected=detected,
                confidence=max_confidence,
                types=detected_types,
                explanation="; ".join(explanations) if explanations else "No hallucinations detected",
                severity=severity,
            )

        except Exception as e:
            self.logger.exception(f"Error in hallucination detection: {e}")
            return HallucinationResult(
                detected=False,
                confidence=0.0,
                types=[],
                explanation=f"Detection error: {e!s}",
                severity="low",
            )

    def _check_aircraft_existence(self, response_text: str, context: dict[str, Any]) -> bool:
        """Check if response references non-existent aircraft"""
        try:
            # Extract aircraft IDs from response
            mentioned_aircraft = set(self.aircraft_id_pattern.findall(response_text))

            # Get actual aircraft from context
            scenario_aircraft = set()
            aircraft_list = context.get("aircraft_list", [])
            for aircraft in aircraft_list:
                if isinstance(aircraft, dict):
                    aircraft_id = aircraft.get("id", "")
                    if aircraft_id:
                        scenario_aircraft.add(aircraft_id)

            # Check for non-existent aircraft
            non_existent = mentioned_aircraft - scenario_aircraft
            return len(non_existent) > 0

        except Exception as e:
            self.logger.warning(f"Aircraft existence check failed: {e}")
            return False

    def _check_altitude_validity(self, response_text: str) -> bool:
        """Check for invalid altitude values"""
        try:
            altitude_matches = self.altitude_pattern.findall(response_text)

            for match in altitude_matches:
                # Extract numeric value
                if match.startswith("FL"):
                    altitude = int(match[2:]) * 100  # Flight level to feet
                else:
                    altitude = int(re.findall(r"[0-9]+", match)[0])

                # Check if altitude is valid
                if altitude < self.min_altitude or altitude > self.max_altitude:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Altitude validity check failed: {e}")
            return False

    def _check_heading_validity(self, response_text: str) -> bool:
        """Check for invalid heading values"""
        try:
            heading_matches = self.heading_pattern.findall(response_text)

            for match in heading_matches:
                heading = float(match.replace("°", ""))

                # Check if heading is valid
                if heading < self.min_heading or heading >= self.max_heading:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Heading validity check failed: {e}")
            return False

    def _check_protocol_violations(self, response_text: str) -> bool:
        """Check for ATC protocol violations"""
        try:
            # Check for common protocol violations
            violations = [
                # Missing standard phraseology
                r"(?i)please|thank you|could you",  # Politeness not used in ATC
                # Incorrect format
                r"(?i)turn left to heading|turn right to heading",  # Should be "turn left heading"
                # Missing aircraft identification
                r"(?i)^(?!.*[A-Z]{3}[0-9]).*turn|climb|descend",  # Commands without aircraft ID
            ]

            for violation_pattern in violations:
                if re.search(violation_pattern, response_text):
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Protocol violation check failed: {e}")
            return False

    def _check_impossible_maneuvers(self, response_text: str, context: dict[str, Any]) -> bool:
        """Check for physically impossible maneuvers"""
        try:
            # Check for extreme climb/descent rates
            if re.search(r"climb.*[5-9][0-9]{3}|descend.*[5-9][0-9]{3}", response_text, re.IGNORECASE):
                return True  # >5000 fpm is extreme

            # Check for impossible speed changes
            if re.search(r"reduce speed.*[0-9]{1,2}|increase speed.*[5-9][0-9]{2}", response_text, re.IGNORECASE):
                return True  # Too slow or too fast

            # Check for contradictory instructions
            if re.search(r"climb.*descend|turn left.*turn right", response_text, re.IGNORECASE):
                return True  # Contradictory commands

            return False

        except Exception as e:
            self.logger.warning(f"Impossible maneuver check failed: {e}")
            return False

    def _check_nonsensical_response(self, response_text: str) -> bool:
        """Check for nonsensical or gibberish responses"""
        try:
            # Check for very short responses
            if len(response_text.strip()) < 10:
                return True

            # Check for repetitive patterns
            words = response_text.split()
            if len(set(words)) < len(words) * 0.5:  # More than 50% repeated words
                return True

            # Check for non-English gibberish
            return bool(re.search(r"[^a-zA-Z0-9\s\.,!?;:-]", response_text))

        except Exception as e:
            self.logger.warning(f"Nonsensical response check failed: {e}")
            return False

    def _determine_severity(self, detected_types: list[HallucinationType]) -> str:
        """Determine severity based on detected hallucination types"""
        if not detected_types:
            return "low"

        critical_types = {
            HallucinationType.AIRCRAFT_EXISTENCE,
            HallucinationType.IMPOSSIBLE_MANEUVER,
            HallucinationType.NONSENSICAL_RESPONSE,
        }

        high_types = {
            HallucinationType.AIRSPACE_VIOLATION,
            HallucinationType.PROTOCOL_VIOLATION,
        }

        if any(t in critical_types for t in detected_types):
            return "critical"
        if any(t in high_types for t in detected_types):
            return "high"
        if len(detected_types) > 2:
            return "medium"
        return "low"

def create_enhanced_detector() -> EnhancedHallucinationDetector:
    """Factory function to create enhanced hallucination detector"""
    return EnhancedHallucinationDetector()
