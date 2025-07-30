# analysis/hallucination_taxonomy.py
"""
Hallucination Taxonomy Module for LLM-ATC-HAL
Categorizes and analyzes different types of hallucinations in ATC context
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HallucinationType(Enum):
    """Types of hallucinations in ATC context"""
    FACTUAL_ERROR = "factual_error"              # Wrong aircraft positions, speeds, etc.
    TEMPORAL_ERROR = "temporal_error"            # Wrong timestamps, sequence errors
    PROCEDURAL_ERROR = "procedural_error"        # Incorrect ATC procedures
    SAFETY_VIOLATION = "safety_violation"        # Unsafe commands or recommendations
    PHANTOM_AIRCRAFT = "phantom_aircraft"        # Non-existent aircraft references
    CONFLICT_MISIDENTIFICATION = "conflict_misidentification"  # Wrong conflict detection
    SPATIAL_ERROR = "spatial_error"              # Wrong coordinates, distances
    COMMUNICATION_ERROR = "communication_error"   # Garbled or nonsensical commands


@dataclass
class HallucinationEvent:
    """Represents a detected hallucination event"""
    event_id: str
    hallucination_type: HallucinationType
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # Detection confidence 0-1
    timestamp: float
    context: dict[str, Any]
    evidence: list[str]
    suggested_correction: Optional[str] = None
    safety_impact: Optional[str] = None


class HallucinationTaxonomy:
    """Classifies and analyzes hallucination patterns in ATC context"""

    def __init__(self) -> None:
        self.taxonomy_rules = self._initialize_taxonomy_rules()
        self.severity_thresholds = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }

    def _initialize_taxonomy_rules(self) -> dict[str, dict]:
        """Initialize classification rules for different hallucination types"""
        return {
            "factual_error": {
                "keywords": ["altitude", "speed", "heading", "position", "callsign"],
                "patterns": ["incorrect numerical values", "impossible physics"],
                "safety_impact": "medium",
            },
            "temporal_error": {
                "keywords": ["time", "sequence", "before", "after", "when"],
                "patterns": ["timeline inconsistency", "causality violation"],
                "safety_impact": "medium",
            },
            "procedural_error": {
                "keywords": ["clearance", "instruction", "procedure", "protocol"],
                "patterns": ["non-standard phraseology", "invalid procedure"],
                "safety_impact": "high",
            },
            "safety_violation": {
                "keywords": ["separation", "conflict", "emergency", "unsafe"],
                "patterns": ["minimum separation violation", "dangerous command"],
                "safety_impact": "critical",
            },
            "phantom_aircraft": {
                "keywords": ["aircraft", "callsign", "flight"],
                "patterns": ["non-existent identifier", "phantom reference"],
                "safety_impact": "medium",
            },
            "conflict_misidentification": {
                "keywords": ["conflict", "separation", "collision", "traffic"],
                "patterns": ["false positive conflict", "missed conflict"],
                "safety_impact": "high",
            },
            "spatial_error": {
                "keywords": ["coordinate", "distance", "bearing", "location"],
                "patterns": ["impossible geometry", "coordinate errors"],
                "safety_impact": "medium",
            },
            "communication_error": {
                "keywords": ["roger", "wilco", "negative", "affirm"],
                "patterns": ["garbled transmission", "nonsensical response"],
                "safety_impact": "low",
            },
        }

    def classify_hallucination(self, text: str, context: dict[str, Any]) -> Optional[HallucinationEvent]:
        """Classify a potential hallucination event"""
        try:
            # Analyze text for hallucination indicators
            detected_types = []
            confidence_scores = []

            text_lower = text.lower()

            for hall_type, rules in self.taxonomy_rules.items():
                score = self._calculate_hallucination_score(text_lower, rules, context)
                if score > 0.3:  # Threshold for detection
                    detected_types.append((hall_type, score))
                    confidence_scores.append(score)

            if not detected_types:
                return None

            # Select the most likely hallucination type
            best_type, best_score = max(detected_types, key=lambda x: x[1])

            # Generate event
            return HallucinationEvent(
                event_id=f"hall_{int(time.time() * 1000)}",
                hallucination_type=HallucinationType(best_type),
                severity=self._determine_severity(best_score, best_type),
                confidence=best_score,
                timestamp=time.time(),
                context=context,
                evidence=self._extract_evidence(text, best_type),
                safety_impact=self.taxonomy_rules[best_type]["safety_impact"],
            )


        except Exception as e:
            logger.exception(f"Error classifying hallucination: {e!s}")
            return None

    def _calculate_hallucination_score(self, text: str, rules: dict, context: dict[str, Any]) -> float:
        """Calculate hallucination likelihood score for a specific type"""

        # Keyword matching
        keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in text)
        keyword_score = min(keyword_matches / len(rules["keywords"]), 1.0) * 0.4

        # Pattern matching (simplified)
        pattern_score = 0.0
        for pattern in rules["patterns"]:
            if any(word in text for word in pattern.split()):
                pattern_score += 0.2
        pattern_score = min(pattern_score, 0.4)

        # Context-based scoring
        context_score = self._evaluate_context_consistency(text, context) * 0.2

        return keyword_score + pattern_score + context_score

    def _evaluate_context_consistency(self, text: str, context: dict[str, Any]) -> float:
        """Evaluate consistency with known context"""
        try:
            # Check for consistency with aircraft data
            if "aircraft_data" in context:
                aircraft_ids = [ac.get("callsign", "") for ac in context["aircraft_data"]]
                # Check if text references non-existent aircraft
                for word in text.split():
                    if word.upper().startswith("AC") and word.upper() not in aircraft_ids:
                        return 0.8  # High inconsistency score

            # Add more context checks here
            return 0.1  # Default low inconsistency

        except Exception:
            return 0.0

    def _determine_severity(self, confidence: float, hallucination_type: str) -> str:
        """Determine severity based on confidence and type"""
        base_severity = self.taxonomy_rules[hallucination_type]["safety_impact"]

        # Adjust severity based on confidence
        if confidence > 0.8:
            if base_severity == "low":
                return "medium"
            if base_severity == "medium":
                return "high"
            if base_severity == "high":
                return "critical"

        return base_severity

    def _extract_evidence(self, text: str, hallucination_type: str) -> list[str]:
        """Extract evidence snippets that support the hallucination classification"""
        evidence = []
        keywords = self.taxonomy_rules[hallucination_type]["keywords"]

        sentences = text.split(".")
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                evidence.append(sentence.strip())

        return evidence[:3]  # Return top 3 evidence snippets

    def analyze_hallucination_patterns(self, events: list[HallucinationEvent]) -> dict[str, Any]:
        """Analyze patterns across multiple hallucination events"""
        if not events:
            return {"total_events": 0}

        analysis = {
            "total_events": len(events),
            "by_type": {},
            "by_severity": {},
            "average_confidence": sum(e.confidence for e in events) / len(events),
            "safety_critical_count": sum(1 for e in events if e.severity == "critical"),
            "temporal_pattern": self._analyze_temporal_patterns(events),
        }

        # Analyze by type
        for event in events:
            type_name = event.hallucination_type.value
            analysis["by_type"][type_name] = analysis["by_type"].get(type_name, 0) + 1

        # Analyze by severity
        for event in events:
            analysis["by_severity"][event.severity] = analysis["by_severity"].get(event.severity, 0) + 1

        return analysis

    def _analyze_temporal_patterns(self, events: list[HallucinationEvent]) -> dict[str, Any]:
        """Analyze temporal patterns in hallucination events"""
        if len(events) < 2:
            return {"insufficient_data": True}

        timestamps = [e.timestamp for e in events]
        timestamps.sort()

        # Calculate time gaps
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        return {
            "time_span": timestamps[-1] - timestamps[0],
            "average_gap": sum(gaps) / len(gaps) if gaps else 0,
            "clustering_detected": any(gap < 10 for gap in gaps),  # Events within 10 seconds
        }


# Global taxonomy instance
taxonomy = HallucinationTaxonomy()


def classify_hallucination(text: str, context: Optional[dict[str, Any]] = None) -> Optional[HallucinationEvent]:
    """Convenience function for hallucination classification"""
    if context is None:
        context = {}
    return taxonomy.classify_hallucination(text, context)


def analyze_hallucination_patterns(events: list[HallucinationEvent]) -> dict[str, Any]:
    """Convenience function for pattern analysis"""
    return taxonomy.analyze_hallucination_patterns(events)


# For compatibility with existing code
import time


def analyze_hallucinations_in_log(log_file: str) -> dict[str, Any]:
    """Analyze hallucinations in a log file (legacy compatibility function)"""
    try:
        with open(log_file, encoding="utf-8") as f:
            content = f.read()

        events = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if "hallucination" in line.lower() or "error" in line.lower():
                event = classify_hallucination(line, {"line_number": i, "source": "log_file"})
                if event:
                    events.append(event)

        return analyze_hallucination_patterns(events)

    except Exception as e:
        logger.exception(f"Error analyzing log file {log_file}: {e!s}")
        return {"error": str(e), "total_events": 0}


if __name__ == "__main__":
    # Test the taxonomy system
    test_cases = [
        "Aircraft AC001 is at altitude 35000 feet with speed 500 knots",
        "Phantom aircraft XYZ999 detected in sector 7",
        "Unsafe separation between AC001 and AC002 - immediate action required",
        "Turn left heading 270 degrees for traffic avoidance",
    ]


    for _i, test_text in enumerate(test_cases):
        event = classify_hallucination(test_text, {"aircraft_data": [{"callsign": "AC001"}, {"callsign": "AC002"}]})

        if event:
            pass
        else:
            pass
