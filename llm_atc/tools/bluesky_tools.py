# tools/bluesky_tools.py
"""
BlueSky Integration Tools - Function stubs for embodied agent system
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class AircraftInfo:
    """Aircraft information structure"""
    id: str
    lat: float
    lon: float
    alt: float
    hdg: float
    spd: float
    vs: float  # vertical speed
    type: str
    callsign: str


@dataclass
class ConflictInfo:
    """Conflict information structure"""
    conflict_id: str
    aircraft_1: str
    aircraft_2: str
    horizontal_separation: float
    vertical_separation: float
    time_to_cpa: float  # closest point of approach
    severity: str


class BlueSkyToolsError(Exception):
    """Custom exception for BlueSky tools"""


def get_all_aircraft_info() -> dict[str, Any]:
    """
    Get information about all aircraft in the simulation

    Returns:
        Dictionary containing aircraft information
    """
    try:
        logging.info("Getting all aircraft information")

        # Stub implementation - return simulated aircraft data
        aircraft_data = {
            "aircraft": {
                "AAL123": {
                    "id": "AAL123",
                    "lat": 52.3676,
                    "lon": 4.9041,
                    "alt": 35000,
                    "hdg": 90,
                    "spd": 450,
                    "vs": 0,
                    "type": "B738",
                    "callsign": "AAL123",
                },
                "DLH456": {
                    "id": "DLH456",
                    "lat": 52.3676,
                    "lon": 4.9141,
                    "alt": 35000,
                    "hdg": 270,
                    "spd": 460,
                    "vs": 0,
                    "type": "A320",
                    "callsign": "DLH456",
                },
            },
            "timestamp": time.time(),
            "total_aircraft": 2,
            "simulation_time": time.time(),
        }

        logging.info("Retrieved information for %d aircraft", aircraft_data["total_aircraft"])
        return aircraft_data

    except Exception as e:
        logging.exception("Error getting aircraft information")
        msg = f"Failed to get aircraft info: {e}"
        raise BlueSkyToolsError(msg) from e


def get_conflict_info() -> dict[str, Any]:
    """
    Get information about current conflicts in the simulation

    Returns:
        Dictionary containing conflict information
    """
    try:
        logging.info("Getting conflict information")

        # Stub implementation - return simulated conflict data
        conflict_data = {
            "conflicts": [
                {
                    "conflict_id": "CONF_001",
                    "aircraft_1": "AAL123",
                    "aircraft_2": "DLH456",
                    "horizontal_separation": 4.2,  # nautical miles
                    "vertical_separation": 0,  # feet
                    "time_to_cpa": 120,  # seconds
                    "severity": "medium",
                    "predicted_cpa_lat": 52.3676,
                    "predicted_cpa_lon": 4.9091,
                    "predicted_cpa_time": time.time() + 120,
                },
            ],
            "total_conflicts": 1,
            "timestamp": time.time(),
            "high_priority_conflicts": 0,
            "medium_priority_conflicts": 1,
            "low_priority_conflicts": 0,
        }

        logging.info("Retrieved %d conflicts", conflict_data["total_conflicts"])
        return conflict_data

    except Exception as e:
        logging.exception("Error getting conflict information")
        msg = f"Failed to get conflict info: {e}"
        raise BlueSkyToolsError(msg) from e


def continue_monitoring() -> dict[str, Any]:
    """
    Continue monitoring aircraft without taking action

    Returns:
        Status information about monitoring continuation
    """
    try:
        logging.info("Continuing monitoring")

        result = {
            "action": "continue_monitoring",
            "status": "active",
            "timestamp": time.time(),
            "next_check_interval": 30,  # seconds
            "monitoring_mode": "automatic",
            "alerts_enabled": True,
        }

        logging.info("Monitoring continuation confirmed")
        return result

    except Exception as e:
        logging.exception("Error continuing monitoring")
        msg = f"Failed to continue monitoring: {e}"
        raise BlueSkyToolsError(msg) from e


def send_command(command: str) -> dict[str, Any]:
    """
    Send a command to the BlueSky simulator

    Args:
        command: BlueSky command string (e.g., "ALT AAL123 FL350")

    Returns:
        Command execution result
    """
    try:
        logging.info("Sending command: %s", command)

        # Parse command for validation
        command_parts = command.strip().split()

        if not command_parts:
            msg = "Empty command"
            raise BlueSkyToolsError(msg)

        command_type = command_parts[0].upper()

        # Validate command format
        valid_commands = [
            "ALT", "HDG", "SPD", "CRE", "DEL", "DEST", "DIRECT", "LNAV",
            "DT", "DTMULT", "VS", "GO", "RESET", "AREA", "CDMETHOD", "CDSEP",
            "WIND", "TURB", "PAUSE", "UNPAUSE", "FF", "IC"
        ]

        if command_type not in valid_commands:
            logging.warning("Unknown command type: %s", command_type)

        # Stub implementation - simulate command execution
        result = {
            "command": command,
            "command_type": command_type,
            "status": "executed",
            "success": True,
            "timestamp": time.time(),
            "execution_time": 0.05,  # seconds
            "response": f"{command_type} command acknowledged",
            "simulation": True,  # Indicates this is a simulated response
            "affected_aircraft": command_parts[1] if len(command_parts) > 1 else None,
        }

        # Simulate occasional failures for testing
        if command_type == "UNKNOWN_COMMAND":
            result.update({
                "status": "failed",
                "success": False,
                "error": f"Unknown command: {command_type}",
                "response": "Command not recognized",
            })

        logging.info("Command executed: %s -> %s", command, result["status"])
        return result

    except Exception as e:
        logging.exception("Error sending command '%s'", command)
        return {
            "command": command,
            "status": "failed",
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


def search_experience_library(
    scenario_type: str, similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Search the experience library for similar scenarios

    Args:
        scenario_type: Type of scenario to search for
        similarity_threshold: Minimum similarity score for matches

    Returns:
        Dictionary containing matching experiences
    """
    try:
        logging.info("Searching experience library for: %s", scenario_type)

        # Stub implementation - return simulated experience data
        experience_data = {
            "query": {
                "scenario_type": scenario_type,
                "similarity_threshold": similarity_threshold,
                "timestamp": time.time(),
            },
            "matches": [
                {
                    "experience_id": "EXP_001",
                    "scenario_type": scenario_type,
                    "similarity_score": 0.92,
                    "conflict_description": "Similar altitude conflict between medium aircraft",
                    "resolution_used": "altitude_change",
                    "commands_executed": ["ALT AIRCRAFT1 FL370"],
                    "outcome": "successful",
                    "safety_margin_achieved": 5.2,  # nautical miles
                    "resolution_time": 180,  # seconds
                    "lessons_learned": (
                        "Early altitude change more effective than late heading change"
                    ),
                    "success_rate": 0.95,
                    "stored_at": time.time() - 86400,  # 1 day ago
                },
                {
                    "experience_id": "EXP_002",
                    "scenario_type": scenario_type,
                    "similarity_score": 0.87,
                    "conflict_description": "Parallel aircraft conflict scenario",
                    "resolution_used": "vector_change",
                    "commands_executed": ["HDG AIRCRAFT1 090", "HDG AIRCRAFT2 270"],
                    "outcome": "successful",
                    "safety_margin_achieved": 6.1,
                    "resolution_time": 240,
                    "lessons_learned": "Symmetric heading changes provide better separation",
                    "success_rate": 0.88,
                    "stored_at": time.time() - 172800,  # 2 days ago
                },
            ],
            "total_matches": 2,
            "search_time": 0.03,  # seconds
            "library_size": 150,  # total experiences in library
            "recommendations": [
                "Consider altitude change as primary resolution method",
                "Monitor vertical separation closely",
                "Early intervention generally more effective",
            ],
        }

        logging.info("Found %d matching experiences", experience_data["total_matches"])
        return experience_data

    except Exception as e:
        logging.exception("Error searching experience library")
        msg = f"Failed to search experience library: {e}"
        raise BlueSkyToolsError(msg) from e


def get_weather_info(lat: float | None = None, lon: float | None = None) -> dict[str, Any]:
    """
    Get weather information for specified location or current area

    Args:
        lat: Latitude (optional)
        lon: Longitude (optional)

    Returns:
        Weather information dictionary
    """
    try:
        logging.info("Getting weather info for lat: %s, lon: %s", lat, lon)

        # Stub implementation - return simulated weather data
        weather_data = {
            "location": {
                "lat": lat or 52.3676,
                "lon": lon or 4.9041,
                "name": "Amsterdam Area",
            },
            "current_conditions": {
                "wind_direction": 270,  # degrees
                "wind_speed": 15,  # knots
                "visibility": 10,  # kilometers
                "cloud_base": 2500,  # feet
                "cloud_coverage": "scattered",
                "temperature": 18,  # celsius
                "pressure": 1013.25,  # hPa
                "humidity": 65,  # percent
            },
            "forecast": {
                "wind_change_expected": False,
                "weather_trend": "stable",
                "turbulence_level": "light",
                "icing_conditions": False,
            },
            "aviation_impact": {
                "visibility_impact": "none",
                "wind_impact": "minimal",
                "turbulence_impact": "light",
                "overall_impact": "minimal",
            },
            "timestamp": time.time(),
        }

        logging.info("Weather information retrieved")
        return weather_data

    except Exception as e:
        logging.exception("Error getting weather info")
        msg = f"Failed to get weather info: {e}"
        raise BlueSkyToolsError(msg) from e


def get_airspace_info() -> dict[str, Any]:
    """
    Get information about current airspace restrictions and constraints

    Returns:
        Airspace information dictionary
    """
    try:
        logging.info("Getting airspace information")

        # Stub implementation - return simulated airspace data
        airspace_data = {
            "active_restrictions": [
                {
                    "restriction_id": "TFR_001",
                    "type": "temporary_flight_restriction",
                    "area": {
                        "center_lat": 52.4,
                        "center_lon": 4.9,
                        "radius": 10,  # nautical miles
                    },
                    "altitude_range": {
                        "floor": 0,
                        "ceiling": 5000,  # feet
                    },
                    "effective_time": time.time() - 3600,  # Started 1 hour ago
                    "expiry_time": time.time() + 7200,  # Expires in 2 hours
                    "reason": "VIP movement",
                },
            ],
            "airways": [
                {
                    "airway_id": "UL607",
                    "status": "active",
                    "restrictions": "none",
                    "traffic_density": "moderate",
                },
                {
                    "airway_id": "UM605",
                    "status": "active",
                    "restrictions": "speed_limited_280kts",
                    "traffic_density": "high",
                },
            ],
            "controlled_airspace": {
                "sectors_active": 12,
                "traffic_flow_status": "normal",
                "capacity_utilization": 0.75,
            },
            "timestamp": time.time(),
        }

        logging.info("Airspace information retrieved")
        return airspace_data

    except Exception as e:
        logging.exception("Error getting airspace info")
        msg = f"Failed to get airspace info: {e}"
        raise BlueSkyToolsError(msg) from e


def get_distance(aircraft_id1: str, aircraft_id2: str) -> dict[str, float]:
    """
    Compute current horizontal and vertical separation between two aircraft.
    
    Args:
        aircraft_id1: ID of first aircraft
        aircraft_id2: ID of second aircraft
    
    Returns:
        Dictionary with separation distances:
        - horizontal_nm: Horizontal separation in nautical miles
        - vertical_ft: Vertical separation in feet
        - total_3d_nm: Total 3D separation in nautical miles
    """
    try:
        logging.info("Computing distance between %s and %s", aircraft_id1, aircraft_id2)
        
        # Get aircraft information
        aircraft_data = get_all_aircraft_info()
        aircraft_dict = aircraft_data.get("aircraft", {})
        
        if aircraft_id1 not in aircraft_dict:
            msg = f"Aircraft {aircraft_id1} not found"
            raise BlueSkyToolsError(msg)
            
        if aircraft_id2 not in aircraft_dict:
            msg = f"Aircraft {aircraft_id2} not found"
            raise BlueSkyToolsError(msg)
        
        ac1 = aircraft_dict[aircraft_id1]
        ac2 = aircraft_dict[aircraft_id2]
        
        # Calculate horizontal distance using haversine formula
        horizontal_nm = _haversine_distance(
            ac1["lat"], ac1["lon"], ac2["lat"], ac2["lon"]
        )
        
        # Calculate vertical separation
        vertical_ft = abs(ac1["alt"] - ac2["alt"])
        
        # Calculate 3D separation
        # Convert vertical separation to nautical miles (1 ft = 1/6076 nm approximately)
        vertical_nm = vertical_ft / 6076.0
        total_3d_nm = math.sqrt(horizontal_nm**2 + vertical_nm**2)
        
        result = {
            "horizontal_nm": horizontal_nm,
            "vertical_ft": vertical_ft,
            "total_3d_nm": total_3d_nm
        }
        
        logging.info(
            "Distance computed: %.2f nm horizontal, %.0f ft vertical, %.2f nm 3D",
            horizontal_nm, vertical_ft, total_3d_nm
        )
        
        return result
        
    except Exception as e:
        logging.exception("Error computing distance between aircraft")
        msg = f"Failed to compute distance: {e}"
        raise BlueSkyToolsError(msg) from e


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in nautical miles.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Distance in nautical miles
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in nautical miles
    earth_radius_nm = 3440.065
    
    return c * earth_radius_nm


def step_simulation(minutes: float, dtmult: float = 1.0) -> dict[str, Any]:
    """
    Advance the BlueSky simulation by a number of minutes.
    
    Args:
        minutes: Number of minutes to advance the simulation
        dtmult: Time multiplier (simulation speed factor)
    
    Returns:
        Status dictionary with simulation step information
    """
    try:
        logging.info("Stepping simulation forward by %.2f minutes (dtmult=%.1f)", 
                    minutes, dtmult)
        
        # Calculate real-time delay based on simulation speed
        real_time_delay = (minutes * 60) / dtmult
        
        # In stub implementation, just sleep for the calculated time
        # In real implementation, this would send DT command to BlueSky
        time.sleep(min(real_time_delay, 5.0))  # Cap sleep time for testing
        
        # Send DT command to advance simulation
        dt_seconds = minutes * 60
        dt_command = f"DT {dt_seconds:.0f}"
        command_result = send_command(dt_command)
        
        result = {
            "action": "step_simulation",
            "minutes_advanced": minutes,
            "seconds_advanced": dt_seconds,
            "dtmult": dtmult,
            "real_time_elapsed": real_time_delay,
            "simulation_time": time.time(),
            "command_sent": dt_command,
            "command_result": command_result,
            "status": "completed",
            "success": True
        }
        
        logging.info("Simulation stepped forward successfully")
        return result
        
    except Exception as e:
        logging.exception("Error stepping simulation")
        return {
            "action": "step_simulation",
            "status": "failed",
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }


def reset_simulation() -> dict[str, Any]:
    """
    Reset the BlueSky simulation to initial state.
    
    Returns:
        Status dictionary with reset information
    """
    try:
        logging.info("Resetting BlueSky simulation")
        
        # Send RESET command
        reset_result = send_command("RESET")
        
        # Additional setup that might be needed after reset
        setup_commands = [
            "DTMULT 1",
            "CDMETHOD SWARM", 
            "CDSEP 5.0 1000"
        ]
        
        setup_results = []
        for cmd in setup_commands:
            setup_results.append(send_command(cmd))
        
        result = {
            "action": "reset_simulation",
            "reset_command": reset_result,
            "setup_commands": setup_results,
            "simulation_state": "initialized",
            "aircraft_count": 0,
            "timestamp": time.time(),
            "success": True,
            "status": "completed"
        }
        
        logging.info("Simulation reset completed")
        return result
        
    except Exception as e:
        logging.exception("Error resetting simulation")
        return {
            "action": "reset_simulation",
            "status": "failed",
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }


def get_minimum_separation() -> dict[str, float]:
    """
    Get the current minimum separation standards.
    
    Returns:
        Dictionary with minimum separation requirements
    """
    return {
        "horizontal_nm": 5.0,    # Standard horizontal separation
        "vertical_ft": 1000.0,   # Standard vertical separation
        "approach_horizontal_nm": 3.0,  # Approach phase horizontal
        "approach_vertical_ft": 500.0,  # Approach phase vertical
        "terminal_horizontal_nm": 3.0,  # Terminal area horizontal
        "oceanic_horizontal_nm": 10.0,  # Oceanic separation
        "rvsm_vertical_ft": 1000.0,     # RVSM vertical separation
    }


def check_separation_violation(aircraft_id1: str, aircraft_id2: str) -> dict[str, Any]:
    """
    Check if two aircraft are violating separation standards.
    
    Args:
        aircraft_id1: ID of first aircraft
        aircraft_id2: ID of second aircraft
    
    Returns:
        Dictionary with violation status and details
    """
    try:
        # Get current separation
        distances = get_distance(aircraft_id1, aircraft_id2)
        min_sep = get_minimum_separation()
        
        # Check violations
        horizontal_violation = distances["horizontal_nm"] < min_sep["horizontal_nm"]
        vertical_violation = distances["vertical_ft"] < min_sep["vertical_ft"]
        
        # Separation violation occurs when BOTH horizontal AND vertical are violated
        # (aircraft need to maintain EITHER horizontal OR vertical separation)
        separation_violation = horizontal_violation and vertical_violation
        
        result = {
            "aircraft_pair": [aircraft_id1, aircraft_id2],
            "current_separation": distances,
            "minimum_required": {
                "horizontal_nm": min_sep["horizontal_nm"],
                "vertical_ft": min_sep["vertical_ft"]
            },
            "violations": {
                "horizontal": horizontal_violation,
                "vertical": vertical_violation,
                "separation_loss": separation_violation
            },
            "safety_margins": {
                "horizontal_nm": distances["horizontal_nm"] - min_sep["horizontal_nm"],
                "vertical_ft": distances["vertical_ft"] - min_sep["vertical_ft"]
            },
            "timestamp": time.time()
        }
        
        if separation_violation:
            logging.warning(
                "SEPARATION VIOLATION: %s and %s - %.2f nm horizontal, %.0f ft vertical",
                aircraft_id1, aircraft_id2, 
                distances["horizontal_nm"], distances["vertical_ft"]
            )
        
        return result
        
    except Exception as e:
        logging.exception("Error checking separation violation")
        return {
            "aircraft_pair": [aircraft_id1, aircraft_id2],
            "error": str(e),
            "timestamp": time.time()
        }


# Tool registry for function calling
TOOL_REGISTRY = {
    "GetAllAircraftInfo": get_all_aircraft_info,
    "GetConflictInfo": get_conflict_info,
    "ContinueMonitoring": continue_monitoring,
    "SendCommand": send_command,
    "SearchExperienceLibrary": search_experience_library,
    "GetWeatherInfo": get_weather_info,
    "GetAirspaceInfo": get_airspace_info,
    "GetDistance": get_distance,
    "StepSimulation": step_simulation,
    "ResetSimulation": reset_simulation,
    "GetMinimumSeparation": get_minimum_separation,
    "CheckSeparationViolation": check_separation_violation,
}


def execute_tool(tool_name: str, **kwargs) -> dict[str, Any]:
    """
    Execute a tool by name with provided arguments

    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool

    Returns:
        Tool execution result
    """
    try:
        if tool_name not in TOOL_REGISTRY:
            msg = f"Unknown tool: {tool_name}"
            raise BlueSkyToolsError(msg)

        tool_function = TOOL_REGISTRY[tool_name]
        result = tool_function(**kwargs)

        return {
            "tool_name": tool_name,
            "success": True,
            "result": result,
            "timestamp": time.time(),
        }

    except Exception as e:
        logging.exception("Error executing tool %s", tool_name)
        return {
            "tool_name": tool_name,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


def get_available_tools() -> list[str]:
    """Get list of available tools"""
    return list(TOOL_REGISTRY.keys())
