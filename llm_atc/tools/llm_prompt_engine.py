#!/usr/bin/env python3
"""
LLM Prompt Engine for ATC Conflict Resolution
============================================
Centralized prompt generation and response parsing for conflict detection
and resolution using Large Language Models.

This module provides:
- Standardized prompt templates for conflict scenarios
- Response parsing for BlueSky commands
- High-level API for LLM-based conflict resolution
- Support for function calling and direct command generation
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from llm_interface.llm_client import LLMClient


@dataclass
class ConflictPromptData:
    """Data structure for conflict prompt generation"""
    aircraft_pair: tuple[str, str]
    aircraft_1_info: dict[str, Any]
    aircraft_2_info: dict[str, Any]
    time_to_conflict: float
    closest_approach_distance: float
    conflict_type: str  # 'horizontal', 'vertical', 'convergent'
    urgency_level: str  # 'low', 'medium', 'high', 'critical'
    environmental_conditions: dict[str, Any]


@dataclass
class ResolutionResponse:
    """Parsed resolution response from LLM"""
    command: str
    aircraft_id: str
    maneuver_type: str  # 'heading', 'altitude', 'speed'
    rationale: str
    confidence: float
    safety_assessment: str


class LLMPromptEngine:
    """
    Centralized LLM prompt generation and response parsing engine
    for ATC conflict resolution tasks.
    """

    def __init__(self, model: str = "llama3.1:8b", enable_function_calls: bool = True,
                 aircraft_id_regex: str = r"^[A-Z0-9-]+$") -> None:
        """
        Initialize the LLM prompt engine.

        Args:
            model: LLM model to use for queries
            enable_function_calls: Whether to enable function calling capabilities
            aircraft_id_regex: Regular expression pattern for validating aircraft callsigns.
                              Default pattern accepts alphanumeric characters and hyphens.
                              Examples: r'^[A-Z]{2,4}\\d{2,4}[A-Z]?$' for traditional ICAO format,
                                       r'^[A-Z0-9-]+$' for flexible alphanumeric with hyphens.
        """
        self.llm_client = LLMClient(model=model)
        self.enable_function_calls = enable_function_calls
        self.aircraft_id_regex = aircraft_id_regex
        self.logger = logging.getLogger(__name__)

        # Standard separation requirements
        self.min_horizontal_separation_nm = 5.0
        self.min_vertical_separation_ft = 1000.0

        # Prompt templates
        self._init_prompt_templates()

    def _init_prompt_templates(self) -> None:
        """Initialize standardized prompt templates"""

        self.conflict_resolution_template = """
You are an expert Air Traffic Controller responsible for maintaining aircraft separation.

CONFLICT SITUATION:
Aircraft: {aircraft_1_id} and {aircraft_2_id}
Time to Conflict: {time_to_conflict:.1f} seconds
Closest Approach: {closest_approach_distance:.1f} NM
Conflict Type: {conflict_type}
Urgency: {urgency_level}

AIRCRAFT DETAILS:
{aircraft_1_id}:
  Position: {ac1_lat:.4f}°N, {ac1_lon:.4f}°E
  Altitude: {ac1_alt:.0f} ft
  Heading: {ac1_hdg:.0f}°
  Speed: {ac1_spd:.0f} kts
  Type: {ac1_type}

{aircraft_2_id}:
  Position: {ac2_lat:.4f}°N, {ac2_lon:.4f}°E
  Altitude: {ac2_alt:.0f} ft
  Heading: {ac2_hdg:.0f}°
  Speed: {ac2_spd:.0f} kts
  Type: {ac2_type}

ENVIRONMENTAL CONDITIONS:
Wind: {wind_direction}° at {wind_speed} kts
Visibility: {visibility}
Weather: {weather_conditions}

REQUIREMENTS:
- Maintain minimum separation: 5 NM horizontal OR 1000 ft vertical
- Minimize disruption to flight paths
- Ensure ICAO compliance
- Prioritize safety over efficiency

INSTRUCTIONS:
You MUST respond with EXACTLY this format. Do not include any other text:

COMMAND: [HDG/ALT/SPD/VS] [AIRCRAFT_ID] [VALUE]
AIRCRAFT: [AIRCRAFT_ID]
MANEUVER: [heading_change/altitude_change/speed_change/vertical_speed_change]
RATIONALE: [Brief explanation]
CONFIDENCE: [0.0-1.0]

EXAMPLES:
COMMAND: HDG UAL890 045
AIRCRAFT: UAL890
MANEUVER: heading_change
RATIONALE: Turn right 25 degrees to avoid conflict
CONFIDENCE: 0.92

COMMAND: ALT AAL123 37000
AIRCRAFT: AAL123
MANEUVER: altitude_change
RATIONALE: Climb 2000 feet for vertical separation
CONFIDENCE: 0.88

Choose ONE aircraft and provide the resolution in EXACTLY the above format.
"""

        self.conflict_detection_template = """
You are an expert Air Traffic Controller analyzing aircraft positions for potential conflicts.

AIRCRAFT STATUS:
{aircraft_list}

ANALYSIS REQUIREMENTS:
- Check for pairs that may lose minimum separation (5 NM horizontal OR 1000 ft vertical)
- Consider current trajectories and speeds
- Look ahead {time_horizon} minutes
- Account for pilot response time (~15-30 seconds)

RESPONSE FORMAT (JSON):
{{
  "conflict_detected": true/false,
  "aircraft_pairs": ["AC001-AC002", "AC003-AC004"],
  "time_to_conflict": [120.5, 180.0],
  "confidence": 0.85,
  "priority": "high",
  "analysis": "Brief explanation of findings"
}}

Provide your analysis as valid JSON only.
"""

        self.safety_assessment_template = """
You are a safety expert evaluating an ATC resolution maneuver.

PROPOSED MANEUVER:
Command: {command}
Aircraft: {aircraft_id}
Situation: {conflict_description}

SAFETY CRITERIA:
1. Will this maintain required separation (5 NM / 1000 ft)?
2. Is the maneuver within aircraft performance limits?
3. Does it comply with ICAO standards?
4. Are there any secondary conflict risks?
5. Is pilot workload reasonable?

You MUST respond with EXACTLY this format:

SAFETY_RATING: [SAFE/MARGINAL/UNSAFE]
SEPARATION_ACHIEVED: [Distance in NM or ft with units]
ICAO_COMPLIANT: [YES/NO]
RISK_ASSESSMENT: [Brief risk analysis in one sentence]
RECOMMENDATION: [APPROVE/MODIFY/REJECT]

EXAMPLE:
SAFETY_RATING: SAFE
SEPARATION_ACHIEVED: 6.2 NM horizontal
ICAO_COMPLIANT: YES
RISK_ASSESSMENT: Maneuver provides adequate separation with minimal disruption
RECOMMENDATION: APPROVE
"""

    def format_conflict_prompt(self, conflict_info: dict[str, Any]) -> str:
        """
        Create a descriptive natural-language prompt for conflict resolution.

        Args:
            conflict_info: Dictionary containing conflict and aircraft information

        Returns:
            Formatted prompt string for LLM query
        """
        try:
            # Extract aircraft information
            ac1_id = conflict_info.get("aircraft_1_id", "AC001")
            ac2_id = conflict_info.get("aircraft_2_id", "AC002")

            ac1_info = conflict_info.get("aircraft_1", {})
            ac2_info = conflict_info.get("aircraft_2", {})

            # Environmental conditions with defaults
            env_conditions = conflict_info.get("environmental_conditions", {})

            return self.conflict_resolution_template.format(
                aircraft_1_id=ac1_id,
                aircraft_2_id=ac2_id,
                time_to_conflict=conflict_info.get("time_to_conflict", 120.0),
                closest_approach_distance=conflict_info.get("closest_approach_distance", 3.5),
                conflict_type=conflict_info.get("conflict_type", "convergent"),
                urgency_level=conflict_info.get("urgency_level", "medium"),

                # Aircraft 1 details
                ac1_lat=ac1_info.get("lat", 52.3676),
                ac1_lon=ac1_info.get("lon", 4.9041),
                ac1_alt=ac1_info.get("alt", 35000),
                ac1_hdg=ac1_info.get("hdg", 90),
                ac1_spd=ac1_info.get("spd", 450),
                ac1_type=ac1_info.get("type", "B738"),

                # Aircraft 2 details
                ac2_lat=ac2_info.get("lat", 52.3700),
                ac2_lon=ac2_info.get("lon", 4.9100),
                ac2_alt=ac2_info.get("alt", 35000),
                ac2_hdg=ac2_info.get("hdg", 270),
                ac2_spd=ac2_info.get("spd", 460),
                ac2_type=ac2_info.get("type", "A320"),

                # Environmental conditions
                wind_direction=env_conditions.get("wind_direction_deg", 270),
                wind_speed=env_conditions.get("wind_speed_kts", 15),
                visibility=env_conditions.get("visibility_km", "10+ km"),
                weather_conditions=env_conditions.get("conditions", "Clear"),
            )


        except Exception as e:
            self.logger.exception(f"Error formatting conflict prompt: {e}")
            return self._get_fallback_conflict_prompt(conflict_info)

    def format_detector_prompt(self, aircraft_states: list[dict[str, Any]],
                              time_horizon: float = 5.0) -> str:
        """
        Create a prompt for LLM-based conflict detection.

        Args:
            aircraft_states: List of aircraft state dictionaries
            time_horizon: Time horizon in minutes for conflict detection

        Returns:
            Formatted detection prompt string
        """
        # Format aircraft list
        aircraft_list = []
        for i, aircraft in enumerate(aircraft_states):
            aircraft_str = f"""
Aircraft {aircraft.get('id', f'AC{i+1:03d}')}:
  Position: {aircraft.get('lat', 0):.4f}°N, {aircraft.get('lon', 0):.4f}°E
  Altitude: {aircraft.get('alt', 0):.0f} ft
  Heading: {aircraft.get('hdg', 0):.0f}°
  Speed: {aircraft.get('spd', 0):.0f} kts
  Vertical Speed: {aircraft.get('vs', 0):.0f} fpm"""
            aircraft_list.append(aircraft_str)

        return self.conflict_detection_template.format(
            aircraft_list="\n".join(aircraft_list),
            time_horizon=time_horizon,
        )

    def parse_resolution_response(self, response_text: str) -> Optional[ResolutionResponse]:
        """
        Extract BlueSky command from LLM response.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed ResolutionResponse object or None if parsing fails
        """
        try:
            # Handle function call responses
            if isinstance(response_text, dict) and response_text.get("type") == "function_call":
                return self._parse_function_call_response(response_text)

            # Parse structured text response using the new format
            command_match = re.search(r"COMMAND:\s*([^\n]+)", response_text, re.IGNORECASE)
            re.search(r"AIRCRAFT:\s*([^\n]+)", response_text, re.IGNORECASE)
            re.search(r"MANEUVER:\s*([^\n]+)", response_text, re.IGNORECASE)
            rationale_match = re.search(r"RATIONALE:\s*([^\n]+)", response_text, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response_text, re.IGNORECASE)

            command = None
            if command_match:
                command = command_match.group(1).strip()
                # Validate and normalize the command
                normalized_command = self._normalize_bluesky_command(command)
                if normalized_command:
                    command = normalized_command
                else:
                    self.logger.warning(f"Could not normalize structured command: {command}")
                    return None
            else:
                # Fallback to legacy parsing if structured format not found
                self.logger.warning("Structured format not found, trying legacy parsing")
                command = self._extract_bluesky_command(response_text)

                if not command:
                    # Try even more flexible patterns for commands like "Turn TEST002 to heading 180"
                    flexible_patterns = [
                        r"turn\s+([A-Z]{2,4}\d{2,4}[A-Z]?)\s+to\s+heading\s+(\d+)",
                        r"([A-Z]{2,4}\d{2,4}[A-Z]?)\s+turn\s+(?:to\s+)?(?:heading\s+)?(\d+)",
                        r"heading\s+(\d+)\s+for\s+([A-Z]{2,4}\d{2,4}[A-Z]?)",
                        r"altitude\s+(\d+)\s+for\s+([A-Z]{2,4}\d{2,4}[A-Z]?)",
                    ]

                    for pattern in flexible_patterns:
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            if "heading" in pattern or "turn" in pattern:
                                if len(match.groups()) >= 2:
                                    if match.group(1).isdigit():
                                        command = f"HDG {match.group(2).upper()} {match.group(1)}"
                                    else:
                                        command = f"HDG {match.group(1).upper()} {match.group(2)}"
                                    break
                            elif "altitude" in pattern:
                                command = f"ALT {match.group(2).upper()} {match.group(1)}"
                                break

            if not command:
                self.logger.warning(f"Could not extract command from response: {response_text[:200]}...")
                return None

            # Extract aircraft ID from command
            aircraft_id = self._extract_aircraft_id(command)

            # Determine maneuver type
            maneuver_type = self._determine_maneuver_type(command)

            return ResolutionResponse(
                command=command,
                aircraft_id=aircraft_id,
                maneuver_type=maneuver_type,
                rationale=rationale_match.group(1).strip() if rationale_match else "No rationale provided",
                confidence=float(confidence_match.group(1)) if confidence_match else 0.5,
                safety_assessment="Pending verification",
            )

        except Exception as e:
            self.logger.exception(f"Error parsing resolution response: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return None

    def parse_detector_response(self, response_text: str) -> dict[str, Any]:
        """
        Interpret conflict detection response from LLM (JSON format).

        Args:
            response_text: Raw LLM response text (expected to be JSON)

        Returns:
            Dictionary with detection results
        """
        # Default response structure
        result = {
            "conflict_detected": False,
            "aircraft_pairs": [],
            "time_to_conflict": [],
            "confidence": 0.5,
            "priority": "low",
            "analysis": "No analysis provided",
        }

        try:
            # First, try to parse as JSON
            # Clean the response text to extract JSON
            json_text = response_text.strip()

            # Look for JSON block in case there's extra text
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)

            json_data = json.loads(json_text)

            # Extract fields with defaults
            result["conflict_detected"] = json_data.get("conflict_detected", False)
            result["aircraft_pairs"] = json_data.get("aircraft_pairs", [])
            result["time_to_conflict"] = json_data.get("time_to_conflict", [])
            result["confidence"] = float(json_data.get("confidence", 0.5))
            result["priority"] = json_data.get("priority", "low").lower()
            result["analysis"] = json_data.get("analysis", "No analysis provided")

            # Convert aircraft pairs to tuples if they're strings
            if result["aircraft_pairs"]:
                pairs = []
                for pair in result["aircraft_pairs"]:
                    if isinstance(pair, str) and "-" in pair:
                        pairs.append(tuple(pair.split("-", 1)))
                    elif isinstance(pair, list) and len(pair) >= 2:
                        pairs.append(tuple(pair[:2]))
                    else:
                        pairs.append(pair)
                result["aircraft_pairs"] = pairs

            return result

        except json.JSONDecodeError:
            # Fallback to legacy text parsing
            self.logger.warning("JSON parsing failed, falling back to text parsing")
            return self._parse_detector_response_legacy(response_text)

        except Exception as e:
            self.logger.exception(f"Error parsing detector response: {e}")
            result["error"] = str(e)
            return result

    def _parse_detector_response_legacy(self, response_text: str) -> dict[str, Any]:
        """Legacy text-based parsing for detector responses"""
        result = {
            "conflict_detected": False,
            "aircraft_pairs": [],
            "time_to_conflict": [],
            "confidence": 0.5,
            "priority": "low",
            "analysis": "Legacy parsing used",
        }

        try:
            # Parse conflict detection
            conflict_match = re.search(r"Conflict Detected:\s*(YES|NO)", response_text, re.IGNORECASE)
            if conflict_match:
                result["conflict_detected"] = conflict_match.group(1).upper() == "YES"

            # Parse aircraft pairs
            pairs_match = re.search(r"Aircraft Pairs at Risk:\s*([^\n]+)", response_text, re.IGNORECASE)
            if pairs_match:
                pairs_text = pairs_match.group(1).strip()
                if pairs_text.lower() not in ["none", "n/a", ""]:
                    result["aircraft_pairs"] = self._parse_aircraft_pairs(pairs_text)

            # Parse time to conflict
            time_match = re.search(r"Time to Loss of Separation:\s*([^\n]+)", response_text, re.IGNORECASE)
            if time_match:
                result["time_to_conflict"] = self._parse_time_values(time_match.group(1))

            # Parse confidence
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", response_text, re.IGNORECASE)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))

            # Parse priority
            priority_match = re.search(r"Priority:\s*(\w+)", response_text, re.IGNORECASE)
            if priority_match:
                result["priority"] = priority_match.group(1).lower()

            return result

        except Exception as e:
            self.logger.exception(f"Error in legacy detector response parsing: {e}")
            result["error"] = str(e)
            return result

    def get_conflict_resolution(self, conflict_info: dict[str, Any],
                               use_function_calls: Optional[bool] = None) -> Optional[str]:
        """
        High-level API for getting conflict resolution from LLM.

        Args:
            conflict_info: Conflict scenario information
            use_function_calls: Override function calling setting

        Returns:
            BlueSky command string or None if resolution fails
        """
        try:
            # Format the prompt
            prompt = self.format_conflict_prompt(conflict_info)

            # Determine function calling setting
            enable_calls = use_function_calls if use_function_calls is not None else self.enable_function_calls

            # Query the LLM
            response = self.llm_client.ask(prompt, enable_function_calls=enable_calls)

            # Parse the response
            resolution = self.parse_resolution_response(response)

            if resolution:
                self.logger.info(f"Generated resolution: {resolution.command} (confidence: {resolution.confidence:.2f})")
                return resolution.command
            self.logger.warning("Failed to parse resolution from LLM response")
            return None

        except Exception as e:
            self.logger.exception(f"Error getting conflict resolution: {e}")
            return None

    def detect_conflict_via_llm(self, aircraft_states: list[dict[str, Any]],
                               time_horizon: float = 5.0) -> dict[str, Any]:
        """
        High-level API for LLM-based conflict detection.

        Args:
            aircraft_states: List of aircraft state dictionaries
            time_horizon: Time horizon in minutes

        Returns:
            Dictionary with detection results
        """
        try:
            # Format the detection prompt
            prompt = self.format_detector_prompt(aircraft_states, time_horizon)

            # Query the LLM
            response = self.llm_client.ask(prompt, enable_function_calls=False)

            # Parse the response
            detection_result = self.parse_detector_response(response)

            self.logger.info(f"Conflict detection: {detection_result['conflict_detected']} "
                           f"(confidence: {detection_result['confidence']:.2f})")

            return detection_result

        except Exception as e:
            self.logger.exception(f"Error in LLM-based conflict detection: {e}")
            return {"conflict_detected": False, "error": str(e)}

    def assess_resolution_safety(self, command: str, conflict_info: dict[str, Any]) -> dict[str, Any]:
        """
        Use LLM to assess the safety of a proposed resolution.

        Args:
            command: Proposed BlueSky command
            conflict_info: Original conflict information

        Returns:
            Safety assessment dictionary
        """
        try:
            # Create conflict description
            conflict_desc = f"Conflict between {conflict_info.get('aircraft_1_id', 'AC1')} " \
                          f"and {conflict_info.get('aircraft_2_id', 'AC2')} with " \
                          f"{conflict_info.get('time_to_conflict', 120):.0f}s to impact"

            # Format safety assessment prompt
            prompt = self.safety_assessment_template.format(
                command=command,
                aircraft_id=self._extract_aircraft_id(command),
                conflict_description=conflict_desc,
            )

            # Query LLM for safety assessment
            response = self.llm_client.ask(prompt, enable_function_calls=False)

            # Parse safety response
            return self._parse_safety_response(response)

        except Exception as e:
            self.logger.exception(f"Error in safety assessment: {e}")
            return {"safety_rating": "UNKNOWN", "error": str(e)}

    # Helper methods

    def _get_fallback_conflict_prompt(self, conflict_info: dict[str, Any]) -> str:
        """Generate a simple fallback prompt when main formatting fails"""
        return f"""
Aircraft conflict detected between {conflict_info.get('aircraft_1_id', 'AC1')}
and {conflict_info.get('aircraft_2_id', 'AC2')}.

Please provide a single BlueSky command to resolve this conflict safely.
Maintain minimum separation of 5 NM horizontal or 1000 ft vertical.

Command:
"""

    def _parse_function_call_response(self, response_dict: dict[str, Any]) -> Optional[ResolutionResponse]:
        """Parse function call response into ResolutionResponse"""
        try:
            function_name = response_dict.get("function_name", "")
            result = response_dict.get("result", {})

            if function_name == "SendCommand" and result.get("success"):
                command = result.get("command", "")
                return ResolutionResponse(
                    command=command,
                    aircraft_id=self._extract_aircraft_id(command),
                    maneuver_type=self._determine_maneuver_type(command),
                    rationale="Generated via function call",
                    confidence=0.8,
                    safety_assessment="Function call successful",
                )
            return None
        except Exception:
            return None

    def _extract_bluesky_command(self, text: str) -> Optional[str]:
        """
        Extract BlueSky command using simplified two-pass approach.

        First pass: Look for explicit BlueSky commands (HDG/ALT/SPD/VS)
        Second pass: Look for natural language patterns
        Third pass: Check for function call format
        """
        if not text:
            return None

        # Clean text: remove degree symbols and other formatting
        cleaned_text = text.replace("°", "").replace("degrees", "").replace("deg", "")
        cleaned_text = re.sub(r"[^\w\s:]+", " ", cleaned_text)  # Remove special chars except colons

        # Check for SendCommand format: **SendCommand("CLB 1000ft", "UAL890")**
        sendcommand_match = re.search(r'SendCommand\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\)', text, re.IGNORECASE)
        if sendcommand_match:
            command_part = sendcommand_match.group(1).strip()
            aircraft_part = sendcommand_match.group(2).strip()

            # Parse command part like "CLB 1000ft" or "HDG 040"
            cmd_match = re.search(r"(CLB|HDG|ALT|SPD|VS)\s*(\d+)", command_part, re.IGNORECASE)
            if cmd_match:
                cmd_type = cmd_match.group(1).upper()
                if cmd_type == "CLB":
                    cmd_type = "ALT"  # Convert CLB to ALT
                value = cmd_match.group(2)

                if re.match(self.aircraft_id_regex, aircraft_part.upper()):
                    return f"{cmd_type} {aircraft_part.upper()} {value}"

        # First pass: Explicit BlueSky command patterns
        explicit_patterns = [
            r"\b(HDG|ALT|SPD|VS)\s+([A-Z0-9-]+)\s+(\d+)\b",  # HDG AC001 270
            r"\b([A-Z0-9-]+)\s+(HDG|ALT|SPD|VS)\s+(\d+)\b",   # AC001 HDG 270
            r"Command:\s*(HDG|ALT|SPD|VS)\s+([A-Z0-9-]+)\s+(\d+)",  # Command: HDG AC001 270
            r"(HDG|ALT|SPD|VS)\s+([A-Z0-9-]+)\s+(\d+)",  # More flexible version
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match and len(match.groups()) >= 3:
                # Validate aircraft ID with configured pattern
                aircraft_candidate = None
                cmd_candidate = None
                value_candidate = None

                if match.group(1).upper() in ["HDG", "ALT", "SPD", "VS"]:
                    cmd_candidate = match.group(1).upper()
                    aircraft_candidate = match.group(2).upper()
                    value_candidate = match.group(3)
                elif match.group(2).upper() in ["HDG", "ALT", "SPD", "VS"]:
                    aircraft_candidate = match.group(1).upper()
                    cmd_candidate = match.group(2).upper()
                    value_candidate = match.group(3)

                if (aircraft_candidate and cmd_candidate and value_candidate and
                    re.match(self.aircraft_id_regex, aircraft_candidate) and
                    value_candidate.isdigit()):
                    return f"{cmd_candidate} {aircraft_candidate} {value_candidate}"

        # Second pass: Natural language patterns (with more flexible matching)
        natural_patterns = [
            r"turn\s+(?:aircraft\s+)?([A-Z0-9-]+)\s+to\s+heading\s+(\d+)",  # Fixed: turn aircraft UAL890 to heading 040
            r"([A-Z0-9-]+)\s+turn\s+(?:to\s+)?(?:heading\s+)?(\d+)",
            r"([A-Z0-9-]+)\s+heading\s+(\d+)",  # New: AC001 heading 045
            r"heading\s+(\d+)\s+([A-Z0-9-]+)",  # New: heading 045 AC001
            r"([A-Z0-9-]+)\s+(\d+)",  # Very flexible: AC001 045
            r"climb\s+([A-Z0-9-]+)\s+to\s+(?:altitude\s+)?(\d+)",
            r"descend\s+([A-Z0-9-]+)\s+to\s+(?:altitude\s+)?(\d+)",
            r"([A-Z0-9-]+)\s+climb\s+to\s+(\d+)",
            r"([A-Z0-9-]+)\s+descend\s+to\s+(\d+)",
            r"speed\s+([A-Z0-9-]+)\s+to\s+(\d+)",
            r"([A-Z0-9-]+)\s+speed\s+(\d+)",
        ]

        for pattern in natural_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match and len(match.groups()) >= 2:
                aircraft = match.group(1).upper()
                value = match.group(2)

                # For the flexible pattern, we need to determine context
                if pattern == r"([A-Z0-9-]+)\s+(\d+)":
                    # Check if this appears to be a heading (0-359) vs altitude (>1000)
                    val_int = int(value)
                    if val_int <= 359 and "heading" in text.lower():
                        cmd_type = "HDG"
                    elif val_int > 1000:
                        cmd_type = "ALT"
                    else:
                        cmd_type = "HDG"  # Default to heading for ambiguous cases
                else:
                    # Determine command type based on pattern
                    pattern_lower = pattern.lower()
                    if "heading" in pattern_lower or "turn" in pattern_lower:
                        cmd_type = "HDG"
                    elif "climb" in pattern_lower or "descend" in pattern_lower or "altitude" in pattern_lower:
                        cmd_type = "ALT"
                    elif "speed" in pattern_lower:
                        cmd_type = "SPD"
                    else:
                        cmd_type = "HDG"  # Default

                if re.match(self.aircraft_id_regex, aircraft) and value.isdigit():
                    return f"{cmd_type} {aircraft} {value}"

        # Third pass: Check for multiple commands or extraneous text
        command_count = len(re.findall(r"\b(HDG|ALT|SPD|VS)\s+[A-Z0-9-]+\s+\d+", cleaned_text, re.IGNORECASE))
        if command_count > 1:
            self.logger.warning(f"Multiple commands detected in response: {text[:100]}...")
            # Return the first valid command found
            first_match = re.search(r"\b(HDG|ALT|SPD|VS)\s+([A-Z0-9-]+)\s+(\d+)", cleaned_text, re.IGNORECASE)
            if first_match:
                cmd, aircraft, value = first_match.groups()
                aircraft = aircraft.upper()
                if re.match(self.aircraft_id_regex, aircraft):
                    return f"{cmd.upper()} {aircraft} {value}"

        return None

    def _normalize_bluesky_command(self, command: str) -> Optional[str]:
        """Normalize and validate BlueSky command format using configurable aircraft ID pattern"""
        if not command:
            return None

        # Clean command: remove degree symbols and other formatting
        cleaned_command = command.replace("°", "").replace("degrees", "").replace("deg", "")
        cleaned_command = re.sub(r"[^\w\s-]", " ", cleaned_command)  # Remove special chars except hyphens

        # Remove extra whitespace and convert to uppercase
        cleaned_command = " ".join(cleaned_command.upper().split())

        # Validate basic command structure
        parts = cleaned_command.split()
        if len(parts) < 3:
            return None

        # Ensure proper command format: CMD AIRCRAFT VALUE or AIRCRAFT CMD VALUE
        valid_commands = ["HDG", "ALT", "SPD", "VS"]

        if parts[0] in valid_commands:
            # Format: HDG AC001 270
            cmd, aircraft, value = parts[0], parts[1], parts[2]
        elif len(parts) >= 3 and parts[1] in valid_commands:
            # Format: AC001 HDG 270
            aircraft, cmd, value = parts[0], parts[1], parts[2]
        else:
            # Try to find valid command anywhere in the parts
            for i, part in enumerate(parts):
                if part in valid_commands and i > 0 and i < len(parts) - 1:
                    aircraft, cmd, value = parts[i-1], part, parts[i+1]
                    break
            else:
                return None

        # Validate aircraft ID pattern using configurable regex
        if not re.match(self.aircraft_id_regex, aircraft):
            return None

        # Validate and clean value
        value = re.sub(r"[^\d]", "", value)  # Remove any non-digit characters
        if not value or not value.isdigit():
            return None

        return f"{cmd} {aircraft} {value}"

    def _extract_aircraft_id(self, command: str) -> str:
        """Extract aircraft ID from BlueSky command using configurable pattern"""
        if not command:
            return ""

        parts = command.split()
        command_keywords = {"HDG", "ALT", "SPD", "VS"}

        for part in parts:
            # Skip command keywords and use configurable aircraft ID pattern
            if part.upper() not in command_keywords and re.match(self.aircraft_id_regex, part):
                return part

        # Fallback: return second part if it exists (traditional CMD AIRCRAFT VALUE format)
        return parts[1] if len(parts) > 1 and parts[1].upper() not in command_keywords else ""

    def _determine_maneuver_type(self, command: str) -> str:
        """Determine maneuver type from BlueSky command"""
        if not command:
            return "unknown"

        command_upper = command.upper()
        if "HDG" in command_upper:
            return "heading"
        if "ALT" in command_upper:
            return "altitude"
        if "SPD" in command_upper:
            return "speed"
        if "VS" in command_upper:
            return "vertical_speed"

        return "unknown"

    def _parse_aircraft_pairs(self, pairs_text: str) -> list[tuple[str, str]]:
        """Parse aircraft pairs from text using configurable aircraft ID pattern"""
        pairs = []

        # Create dynamic patterns based on the configurable aircraft ID regex
        # Remove ^ and $ anchors to use in the middle of patterns
        id_pattern = self.aircraft_id_regex.replace("^", "").replace("$", "")

        # Look for patterns like "AC001-AC002" or "AC001 and AC002"
        pair_patterns = [
            rf"({id_pattern})-({id_pattern})",
            rf"({id_pattern})\s+and\s+({id_pattern})",
            rf"({id_pattern}),\s*({id_pattern})",
            rf"({id_pattern})\s+vs\s+({id_pattern})",
        ]

        for pattern in pair_patterns:
            matches = re.findall(pattern, pairs_text, re.IGNORECASE)
            for match in matches:
                # Validate both aircraft IDs match the full pattern
                ac1, ac2 = match[0].upper(), match[1].upper()
                if (re.match(self.aircraft_id_regex, ac1) and
                    re.match(self.aircraft_id_regex, ac2)):
                    pairs.append((ac1, ac2))

        return pairs

    def _parse_time_values(self, time_text: str) -> list[float]:
        """Parse time values from text"""
        times = []
        # Extract numeric values that could be times
        time_matches = re.findall(r"(\d+(?:\.\d+)?)", time_text)
        for match in time_matches:
            try:
                times.append(float(match))
            except ValueError:
                continue
        return times

    def _parse_safety_response(self, response_text: str) -> dict[str, Any]:
        """Parse safety assessment response with robust fallbacks for missing fields"""
        result = {
            "safety_rating": "UNKNOWN",
            "separation_achieved": "Unknown",
            "icao_compliant": False,
            "risk_assessment": "No assessment provided",
            "recommendation": "UNKNOWN",
        }

        missing_fields = []

        try:
            # Parse safety rating (new format with underscores)
            rating_match = re.search(r"SAFETY_RATING:\s*(SAFE|MARGINAL|UNSAFE)", response_text, re.IGNORECASE)
            if not rating_match:
                # Fallback to old format
                rating_match = re.search(r"Safety Rating:\s*(SAFE|MARGINAL|UNSAFE)", response_text, re.IGNORECASE)

            if rating_match:
                result["safety_rating"] = rating_match.group(1).upper()
            else:
                missing_fields.append("Safety Rating")

            # Parse separation (new format)
            sep_match = re.search(r"SEPARATION_ACHIEVED:\s*([^\n]+)", response_text, re.IGNORECASE)
            if not sep_match:
                # Fallback to old format
                sep_match = re.search(r"Separation Achieved:\s*([^\n]+)", response_text, re.IGNORECASE)

            if sep_match:
                result["separation_achieved"] = sep_match.group(1).strip()
            else:
                missing_fields.append("Separation Achieved")

            # Parse compliance (new format)
            compliance_match = re.search(r"ICAO_COMPLIANT:\s*(YES|NO)", response_text, re.IGNORECASE)
            if not compliance_match:
                # Fallback to old format
                compliance_match = re.search(r"ICAO compliant:\s*(YES|NO)", response_text, re.IGNORECASE)

            if compliance_match:
                result["icao_compliant"] = compliance_match.group(1).upper() == "YES"
            else:
                missing_fields.append("ICAO Compliance")

            # Parse risk assessment (new format)
            risk_match = re.search(r"RISK_ASSESSMENT:\s*([^\n]+)", response_text, re.IGNORECASE)
            if not risk_match:
                # Fallback to old format
                risk_match = re.search(r"Risk Assessment:\s*([^\n]+)", response_text, re.IGNORECASE)

            if risk_match:
                result["risk_assessment"] = risk_match.group(1).strip()
            else:
                missing_fields.append("Risk Assessment")

            # Parse recommendation (new format)
            rec_match = re.search(r"RECOMMENDATION:\s*(APPROVE|MODIFY|REJECT)", response_text, re.IGNORECASE)
            if not rec_match:
                # Fallback to old format
                rec_match = re.search(r"Recommendation:\s*(APPROVE|MODIFY|REJECT)", response_text, re.IGNORECASE)

            if rec_match:
                result["recommendation"] = rec_match.group(1).upper()
            else:
                missing_fields.append("Recommendation")

            # Log warnings for missing fields
            if missing_fields:
                self.logger.warning(f"Missing safety assessment fields: {', '.join(missing_fields)}")
                result["missing_fields"] = missing_fields
                result["parsing_issues"] = True
            else:
                result["parsing_issues"] = False

        except Exception as e:
            self.logger.exception(f"Error parsing safety response: {e}")
            result["error"] = str(e)
            result["parsing_issues"] = True

        return result
