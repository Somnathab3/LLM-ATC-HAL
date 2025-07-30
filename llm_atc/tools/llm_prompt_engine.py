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
from typing import Dict, List, Any, Optional, Union, Tuple

from llm_interface.llm_client import LLMClient
from llm_atc.tools.bluesky_tools import AircraftInfo, ConflictInfo


@dataclass
class ConflictPromptData:
    """Data structure for conflict prompt generation"""
    aircraft_pair: Tuple[str, str]
    aircraft_1_info: Dict[str, Any]
    aircraft_2_info: Dict[str, Any]
    time_to_conflict: float
    closest_approach_distance: float
    conflict_type: str  # 'horizontal', 'vertical', 'convergent'
    urgency_level: str  # 'low', 'medium', 'high', 'critical'
    environmental_conditions: Dict[str, Any]


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

    def __init__(self, model: str = 'llama3.1:8b', enable_function_calls: bool = True):
        """
        Initialize the LLM prompt engine.
        
        Args:
            model: LLM model to use for queries
            enable_function_calls: Whether to enable function calling capabilities
        """
        self.llm_client = LLMClient(model=model)
        self.enable_function_calls = enable_function_calls
        self.logger = logging.getLogger(__name__)
        
        # Standard separation requirements
        self.min_horizontal_separation_nm = 5.0
        self.min_vertical_separation_ft = 1000.0
        
        # Prompt templates
        self._init_prompt_templates()

    def _init_prompt_templates(self):
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

CRITICAL: Respond with EXACTLY ONE BlueSky command in this exact format:
HDG [AIRCRAFT_ID] [HEADING_DEGREES]
or
ALT [AIRCRAFT_ID] [ALTITUDE_FEET]
or
SPD [AIRCRAFT_ID] [SPEED_KNOTS]

Example valid responses:
HDG {aircraft_1_id} 270
ALT {aircraft_2_id} 36000
SPD {aircraft_1_id} 420

Choose ONE aircraft to modify and provide the command immediately:
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

RESPONSE FORMAT:
Conflict Detected: [YES/NO]
Aircraft Pairs at Risk: [List callsigns if conflicts found]
Time to Loss of Separation: [Seconds for each pair]
Confidence: [0.0-1.0 confidence in assessment]
Priority: [low/medium/high/critical for most urgent conflict]

Analysis:
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

RESPONSE:
Safety Rating: [SAFE/MARGINAL/UNSAFE]
Separation Achieved: [Distance in NM or ft]
Compliance: [ICAO compliant: YES/NO]
Risk Assessment: [Brief risk analysis]
Recommendation: [APPROVE/MODIFY/REJECT]
"""

    def format_conflict_prompt(self, conflict_info: Dict[str, Any]) -> str:
        """
        Create a descriptive natural-language prompt for conflict resolution.
        
        Args:
            conflict_info: Dictionary containing conflict and aircraft information
            
        Returns:
            Formatted prompt string for LLM query
        """
        try:
            # Extract aircraft information
            ac1_id = conflict_info.get('aircraft_1_id', 'AC001')
            ac2_id = conflict_info.get('aircraft_2_id', 'AC002')
            
            ac1_info = conflict_info.get('aircraft_1', {})
            ac2_info = conflict_info.get('aircraft_2', {})
            
            # Environmental conditions with defaults
            env_conditions = conflict_info.get('environmental_conditions', {})
            
            prompt = self.conflict_resolution_template.format(
                aircraft_1_id=ac1_id,
                aircraft_2_id=ac2_id,
                time_to_conflict=conflict_info.get('time_to_conflict', 120.0),
                closest_approach_distance=conflict_info.get('closest_approach_distance', 3.5),
                conflict_type=conflict_info.get('conflict_type', 'convergent'),
                urgency_level=conflict_info.get('urgency_level', 'medium'),
                
                # Aircraft 1 details
                ac1_lat=ac1_info.get('lat', 52.3676),
                ac1_lon=ac1_info.get('lon', 4.9041),
                ac1_alt=ac1_info.get('alt', 35000),
                ac1_hdg=ac1_info.get('hdg', 90),
                ac1_spd=ac1_info.get('spd', 450),
                ac1_type=ac1_info.get('type', 'B738'),
                
                # Aircraft 2 details
                ac2_lat=ac2_info.get('lat', 52.3700),
                ac2_lon=ac2_info.get('lon', 4.9100),
                ac2_alt=ac2_info.get('alt', 35000),
                ac2_hdg=ac2_info.get('hdg', 270),
                ac2_spd=ac2_info.get('spd', 460),
                ac2_type=ac2_info.get('type', 'A320'),
                
                # Environmental conditions
                wind_direction=env_conditions.get('wind_direction_deg', 270),
                wind_speed=env_conditions.get('wind_speed_kts', 15),
                visibility=env_conditions.get('visibility_km', '10+ km'),
                weather_conditions=env_conditions.get('conditions', 'Clear')
            )
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error formatting conflict prompt: {e}")
            return self._get_fallback_conflict_prompt(conflict_info)

    def format_detector_prompt(self, aircraft_states: List[Dict[str, Any]], 
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
            aircraft_list='\n'.join(aircraft_list),
            time_horizon=time_horizon
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
            if isinstance(response_text, dict) and response_text.get('type') == 'function_call':
                return self._parse_function_call_response(response_text)
            
            # Parse structured text response
            command_match = re.search(r'Command:\s*([^\n]+)', response_text, re.IGNORECASE)
            aircraft_match = re.search(r'Aircraft:\s*([^\n]+)', response_text, re.IGNORECASE)
            maneuver_match = re.search(r'Maneuver:\s*([^\n]+)', response_text, re.IGNORECASE)
            rationale_match = re.search(r'Rationale:\s*([^\n]+)', response_text, re.IGNORECASE)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', response_text, re.IGNORECASE)
            
            command = None
            if command_match:
                command = command_match.group(1).strip()
            else:
                # Try to extract BlueSky command patterns directly
                command = self._extract_bluesky_command(response_text)
            
            if not command:
                # Try even more flexible patterns for commands like "Turn TEST002 to heading 180"
                flexible_patterns = [
                    r'turn\s+([A-Z]{2,4}\d{2,4}[A-Z]?)\s+to\s+heading\s+(\d+)',
                    r'([A-Z]{2,4}\d{2,4}[A-Z]?)\s+turn\s+(?:to\s+)?(?:heading\s+)?(\d+)',
                    r'heading\s+(\d+)\s+for\s+([A-Z]{2,4}\d{2,4}[A-Z]?)',
                    r'altitude\s+(\d+)\s+for\s+([A-Z]{2,4}\d{2,4}[A-Z]?)',
                ]
                
                for pattern in flexible_patterns:
                    match = re.search(pattern, response_text, re.IGNORECASE)
                    if match:
                        if 'heading' in pattern or 'turn' in pattern:
                            if len(match.groups()) >= 2:
                                if match.group(1).isdigit():
                                    command = f"HDG {match.group(2).upper()} {match.group(1)}"
                                else:
                                    command = f"HDG {match.group(1).upper()} {match.group(2)}"
                                break
                        elif 'altitude' in pattern:
                            command = f"ALT {match.group(2).upper()} {match.group(1)}"
                            break
            
            if not command:
                self.logger.warning(f"Could not extract command from response: {response_text[:200]}...")
                return None
            
            # Validate and normalize command
            normalized_command = self._normalize_bluesky_command(command)
            if not normalized_command:
                self.logger.warning(f"Could not normalize command: {command}")
                return None
            
            # Extract aircraft ID from command
            aircraft_id = self._extract_aircraft_id(normalized_command)
            
            # Determine maneuver type
            maneuver_type = self._determine_maneuver_type(normalized_command)
            
            return ResolutionResponse(
                command=normalized_command,
                aircraft_id=aircraft_id,
                maneuver_type=maneuver_type,
                rationale=rationale_match.group(1).strip() if rationale_match else "No rationale provided",
                confidence=float(confidence_match.group(1)) if confidence_match else 0.5,
                safety_assessment="Pending verification"
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing resolution response: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return None

    def parse_detector_response(self, response_text: str) -> Dict[str, Any]:
        """
        Interpret conflict detection response from LLM.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Default response structure
            result = {
                'conflict_detected': False,
                'aircraft_pairs': [],
                'time_to_conflict': [],
                'confidence': 0.5,
                'priority': 'low'
            }
            
            # Parse conflict detection
            conflict_match = re.search(r'Conflict Detected:\s*(YES|NO)', response_text, re.IGNORECASE)
            if conflict_match:
                result['conflict_detected'] = conflict_match.group(1).upper() == 'YES'
            
            # Parse aircraft pairs
            pairs_match = re.search(r'Aircraft Pairs at Risk:\s*([^\n]+)', response_text, re.IGNORECASE)
            if pairs_match:
                pairs_text = pairs_match.group(1).strip()
                if pairs_text.lower() not in ['none', 'n/a', '']:
                    result['aircraft_pairs'] = self._parse_aircraft_pairs(pairs_text)
            
            # Parse time to conflict
            time_match = re.search(r'Time to Loss of Separation:\s*([^\n]+)', response_text, re.IGNORECASE)
            if time_match:
                result['time_to_conflict'] = self._parse_time_values(time_match.group(1))
            
            # Parse confidence
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', response_text, re.IGNORECASE)
            if confidence_match:
                result['confidence'] = float(confidence_match.group(1))
            
            # Parse priority
            priority_match = re.search(r'Priority:\s*(\w+)', response_text, re.IGNORECASE)
            if priority_match:
                result['priority'] = priority_match.group(1).lower()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing detector response: {e}")
            return {'conflict_detected': False, 'error': str(e)}

    def get_conflict_resolution(self, conflict_info: Dict[str, Any], 
                               use_function_calls: bool = None) -> Optional[str]:
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
            else:
                self.logger.warning("Failed to parse resolution from LLM response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting conflict resolution: {e}")
            return None

    def detect_conflict_via_llm(self, aircraft_states: List[Dict[str, Any]], 
                               time_horizon: float = 5.0) -> Dict[str, Any]:
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
            self.logger.error(f"Error in LLM-based conflict detection: {e}")
            return {'conflict_detected': False, 'error': str(e)}

    def assess_resolution_safety(self, command: str, conflict_info: Dict[str, Any]) -> Dict[str, Any]:
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
                conflict_description=conflict_desc
            )
            
            # Query LLM for safety assessment
            response = self.llm_client.ask(prompt, enable_function_calls=False)
            
            # Parse safety response
            return self._parse_safety_response(response)
            
        except Exception as e:
            self.logger.error(f"Error in safety assessment: {e}")
            return {'safety_rating': 'UNKNOWN', 'error': str(e)}

    # Helper methods
    
    def _get_fallback_conflict_prompt(self, conflict_info: Dict[str, Any]) -> str:
        """Generate a simple fallback prompt when main formatting fails"""
        return f"""
Aircraft conflict detected between {conflict_info.get('aircraft_1_id', 'AC1')} 
and {conflict_info.get('aircraft_2_id', 'AC2')}.

Please provide a single BlueSky command to resolve this conflict safely.
Maintain minimum separation of 5 NM horizontal or 1000 ft vertical.

Command:
"""

    def _parse_function_call_response(self, response_dict: Dict[str, Any]) -> Optional[ResolutionResponse]:
        """Parse function call response into ResolutionResponse"""
        try:
            function_name = response_dict.get('function_name', '')
            result = response_dict.get('result', {})
            
            if function_name == 'SendCommand' and result.get('success'):
                command = result.get('command', '')
                return ResolutionResponse(
                    command=command,
                    aircraft_id=self._extract_aircraft_id(command),
                    maneuver_type=self._determine_maneuver_type(command),
                    rationale="Generated via function call",
                    confidence=0.8,
                    safety_assessment="Function call successful"
                )
            return None
        except Exception:
            return None

    def _extract_bluesky_command(self, text: str) -> Optional[str]:
        """Extract BlueSky command patterns from text"""
        # Common BlueSky command patterns - more flexible
        patterns = [
            r'\b(HDG|ALT|SPD|VS)\s+([A-Z]{2,4}\d{2,4}[A-Z]?)\s+(\d+)\b',  # HDG AC001 270, HDG KLM492 270
            r'\b([A-Z]{2,4}\d{2,4}[A-Z]?)\s+(HDG|ALT|SPD|VS)\s+(\d+)\b',   # AC001 HDG 270
            r'(HDG|ALT|SPD|VS)\s+([A-Z]{2,4}\d{2,4}[A-Z]?)\s+(\d+)',      # More lenient matching
            r'Command:\s*(HDG|ALT|SPD|VS)\s+([A-Z]{2,4}\d{2,4}[A-Z]?)\s+(\d+)',  # Command: HDG ...
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the command components
                if len(match.groups()) >= 3:
                    if match.group(1).upper() in ['HDG', 'ALT', 'SPD', 'VS']:
                        # Pattern: CMD AIRCRAFT VALUE
                        return f"{match.group(1).upper()} {match.group(2).upper()} {match.group(3)}"
                    else:
                        # Pattern: AIRCRAFT CMD VALUE
                        return f"{match.group(2).upper()} {match.group(1).upper()} {match.group(3)}"
        
        return None

    def _normalize_bluesky_command(self, command: str) -> Optional[str]:
        """Normalize and validate BlueSky command format"""
        if not command:
            return None
        
        # Remove extra whitespace and convert to uppercase
        command = ' '.join(command.upper().split())
        
        # Validate basic command structure
        parts = command.split()
        if len(parts) < 3:
            return None
        
        # Ensure proper command format: CMD AIRCRAFT VALUE or AIRCRAFT CMD VALUE
        valid_commands = ['HDG', 'ALT', 'SPD', 'VS']
        
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
        
        # Validate aircraft ID pattern (more flexible)
        if not re.match(r'^[A-Z]{2,4}\d{2,4}[A-Z]?$', aircraft):
            return None
        
        # Validate value is numeric
        if not value.isdigit():
            return None
            
        return f"{cmd} {aircraft} {value}"

    def _extract_aircraft_id(self, command: str) -> str:
        """Extract aircraft ID from BlueSky command"""
        if not command:
            return ""
        
        parts = command.split()
        for part in parts:
            # More flexible aircraft ID pattern
            if re.match(r'^[A-Z]{2,4}\d{2,4}[A-Z]?$', part):
                return part
        
        return parts[1] if len(parts) > 1 else ""

    def _determine_maneuver_type(self, command: str) -> str:
        """Determine maneuver type from BlueSky command"""
        if not command:
            return "unknown"
        
        command_upper = command.upper()
        if 'HDG' in command_upper:
            return "heading"
        elif 'ALT' in command_upper:
            return "altitude"
        elif 'SPD' in command_upper:
            return "speed"
        elif 'VS' in command_upper:
            return "vertical_speed"
        
        return "unknown"

    def _parse_aircraft_pairs(self, pairs_text: str) -> List[Tuple[str, str]]:
        """Parse aircraft pairs from text"""
        pairs = []
        # Look for patterns like "AC001-AC002" or "AC001 and AC002"
        pair_patterns = [
            r'([A-Z]{2,3}\d{3,4})-([A-Z]{2,3}\d{3,4})',
            r'([A-Z]{2,3}\d{3,4})\s+and\s+([A-Z]{2,3}\d{3,4})',
        ]
        
        for pattern in pair_patterns:
            matches = re.findall(pattern, pairs_text)
            pairs.extend(matches)
        
        return pairs

    def _parse_time_values(self, time_text: str) -> List[float]:
        """Parse time values from text"""
        times = []
        # Extract numeric values that could be times
        time_matches = re.findall(r'(\d+(?:\.\d+)?)', time_text)
        for match in time_matches:
            try:
                times.append(float(match))
            except ValueError:
                continue
        return times

    def _parse_safety_response(self, response_text: str) -> Dict[str, Any]:
        """Parse safety assessment response"""
        result = {
            'safety_rating': 'UNKNOWN',
            'separation_achieved': 'Unknown',
            'icao_compliant': False,
            'risk_assessment': 'No assessment provided',
            'recommendation': 'UNKNOWN'
        }
        
        try:
            # Parse safety rating
            rating_match = re.search(r'Safety Rating:\s*(SAFE|MARGINAL|UNSAFE)', response_text, re.IGNORECASE)
            if rating_match:
                result['safety_rating'] = rating_match.group(1).upper()
            
            # Parse separation
            sep_match = re.search(r'Separation Achieved:\s*([^\n]+)', response_text, re.IGNORECASE)
            if sep_match:
                result['separation_achieved'] = sep_match.group(1).strip()
            
            # Parse compliance
            compliance_match = re.search(r'ICAO compliant:\s*(YES|NO)', response_text, re.IGNORECASE)
            if compliance_match:
                result['icao_compliant'] = compliance_match.group(1).upper() == 'YES'
            
            # Parse risk assessment
            risk_match = re.search(r'Risk Assessment:\s*([^\n]+)', response_text, re.IGNORECASE)
            if risk_match:
                result['risk_assessment'] = risk_match.group(1).strip()
            
            # Parse recommendation
            rec_match = re.search(r'Recommendation:\s*(APPROVE|MODIFY|REJECT)', response_text, re.IGNORECASE)
            if rec_match:
                result['recommendation'] = rec_match.group(1).upper()
            
        except Exception as e:
            self.logger.error(f"Error parsing safety response: {e}")
        
        return result
