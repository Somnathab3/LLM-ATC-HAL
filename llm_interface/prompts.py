#!/usr/bin/env python3
"""
ATC Prompt Engineering Templates for LLM-ATC-HAL
===============================================
Specialized prompts for air traffic control conflict resolution,
safety assessment, and technical compliance checking.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ConflictScenario:
    """Data structure for conflict scenario information"""

    aircraft_data: list[dict[str, Any]]
    conflict_type: str
    urgency_level: str
    environmental_conditions: dict[str, Any]
    airspace_constraints: Optional[dict[str, Any]] = None


class ATCPromptGenerator:
    """
    Generate specialized prompts for ATC conflict resolution using LLMs
    """

    def __init__(self) -> None:
        self.base_context = """
You are an expert Air Traffic Controller AI assistant with deep knowledge of:
- ICAO standards and regulations
- Air traffic separation requirements (5 NM horizontal, 1000 ft vertical)
- Aircraft performance characteristics
- Weather impact on flight operations
- Emergency procedures and safety protocols

Always prioritize safety, provide clear rationale, and suggest ICAO-compliant solutions.
"""

    def generate_conflict_prompt(
        self, aircraft_data: list[dict], conflict_type: str, urgency_level: str,
    ) -> str:
        """Generate conflict resolution prompt"""

        aircraft_info = self._format_aircraft_data(aircraft_data)

        return f"""{self.base_context}

CONFLICT RESOLUTION REQUEST
==========================
Conflict Type: {conflict_type}
Urgency Level: {urgency_level}

Aircraft Information:
{aircraft_info}

Please provide:
1. Immediate safety assessment
2. Recommended resolution actions for each aircraft
3. Alternative solutions if primary fails
4. Expected separation after resolution
5. Safety margin analysis

Ensure all recommendations comply with ICAO standards and consider:
- Minimum separation requirements
- Aircraft performance limitations
- Pilot workload
- Fuel efficiency
- Weather conditions
"""

    def generate_safety_assessment_prompt(self, scenario: ConflictScenario) -> str:
        """Generate safety assessment prompt"""

        return f"""{self.base_context}

SAFETY ASSESSMENT REQUEST
========================
Scenario Type: {scenario.conflict_type}
Urgency: {scenario.urgency_level}

Aircraft Data:
{self._format_aircraft_data(scenario.aircraft_data)}

Environmental Conditions:
{self._format_environmental_data(scenario.environmental_conditions)}

Please evaluate:
1. Current safety margin analysis
2. Risk factors and probability assessment
3. ICAO compliance verification
4. Potential safety violations
5. Recommended safety improvements

Provide quantitative safety scores where possible.
"""

    def generate_validation_prompt(self, proposed_solution: str, original_conflict: dict) -> str:
        """Generate validation prompt for proposed solutions"""

        return f"""{self.base_context}

SOLUTION VALIDATION REQUEST
===========================
Original Conflict: {original_conflict.get('description', 'Unspecified conflict')}

Proposed Solution:
{proposed_solution}

Please validate this solution by checking:
1. ICAO regulation compliance
2. Separation standards adherence
3. Feasibility given aircraft performance
4. Safety margin adequacy
5. Potential secondary conflicts

Provide:
- PASS/FAIL assessment
- Specific compliance issues (if any)
- Improvement suggestions
- Confidence score (0-100)
"""

    def generate_technical_compliance_prompt(self, actions: list[str]) -> str:
        """Generate technical compliance checking prompt"""

        actions_list = "\n".join([f"- {action}" for action in actions])

        return f"""{self.base_context}

TECHNICAL COMPLIANCE CHECK
=========================
Proposed ATC Actions:
{actions_list}

Please verify compliance with:
1. ICAO Annex 2 (Rules of the Air)
2. ICAO Doc 4444 (Air Traffic Management)
3. Standard phraseology requirements
4. Separation minima standards
5. Aircraft performance limitations

For each action, provide:
- Compliance status (COMPLIANT/NON-COMPLIANT)
- Regulation reference
- Risk assessment
- Alternative if non-compliant
"""

    def _format_aircraft_data(self, aircraft_data: list[dict]) -> str:
        """Format aircraft data for prompt inclusion"""
        formatted = []
        for i, aircraft in enumerate(aircraft_data, 1):
            info = f"""
Aircraft {i}: {aircraft.get('callsign', f'AC{i:03d}')}
- Type: {aircraft.get('aircraft_type', 'Unknown')}
- Position: {aircraft.get('latitude', 'N/A')}°N, {aircraft.get('longitude', 'N/A')}°E
- Altitude: {aircraft.get('altitude', 'N/A')} ft
- Heading: {aircraft.get('heading', 'N/A')}°
- Ground Speed: {aircraft.get('ground_speed', 'N/A')} kts
- Vertical Speed: {aircraft.get('vertical_speed', 0)} ft/min
"""
            formatted.append(info)

        return "\n".join(formatted)

    def _format_environmental_data(self, env_data: dict) -> str:
        """Format environmental conditions for prompt inclusion"""
        return f"""
- Weather: {env_data.get('weather', 'Clear')}
- Wind: {env_data.get('wind_speed', 0)} kts from {env_data.get('wind_direction', 0)}°
- Visibility: {env_data.get('visibility', 15)} km
- Turbulence: {env_data.get('turbulence', 'None')}
- Temperature: {env_data.get('temperature', 'Standard')}
"""
