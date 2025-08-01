# tools/bluesky_tools.py
"""
BlueSky Integration Tools - Real BlueSky simulator integration
"""

import logging
import math
import os
import socket
import time

try:
    import yaml
except ImportError:
    logging.warning("PyYAML not available, using default configuration")
    yaml = None
from dataclasses import dataclass
from typing import Any, Optional

# BlueSky imports - try to import the actual BlueSky simulator
try:
    import bluesky as bs
    from bluesky import sim, stack, traf

    BLUESKY_AVAILABLE = True
    logging.info("BlueSky simulator successfully imported")
except ImportError as e:
    BLUESKY_AVAILABLE = False
    logging.warning(f"BlueSky not available - using mock simulation: {e}")
    # Create dummy references
    bs = None
    stack = None
    sim = None
    traf = None


# Configuration management
class BlueSkyConfig:
    """Configuration manager for BlueSky integration"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()

    def _find_config_file(self) -> str:
        """Find the BlueSky configuration file"""
        possible_paths = [
            "bluesky_config.yaml",
            "config/bluesky_config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "bluesky_config.yaml"),
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "bluesky_config.yaml"
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Create default config if none found
        default_path = "bluesky_config.yaml"
        self._create_default_config(default_path)
        return default_path

    def _create_default_config(self, path: str) -> None:
        """Create a default configuration file"""
        default_config = {
            "bluesky": {
                "connection_type": "local",
                "network": {
                    "host": "localhost",
                    "port": 8080,
                    "timeout": 10.0,
                },
                "simulation": {
                    "default_dt_mult": 1.0,
                    "max_simulation_time": 3600,
                    "conflict_detection_method": "SWARM",
                    "separation_standards": {
                        "horizontal_nm": 5.0,
                        "vertical_ft": 1000.0,
                    },
                },
                "mock_data": {
                    "use_realistic_aircraft_count": True,
                    "default_aircraft_count": 10,
                    "airspace_bounds": {
                        "lat_min": 51.0,
                        "lat_max": 53.0,
                        "lon_min": 3.0,
                        "lon_max": 6.0,
                    },
                    "altitude_range": {
                        "min_fl": 200,
                        "max_fl": 400,
                    },
                },
            },
            "logging": {
                "level": "INFO",
                "log_bluesky_commands": True,
                "log_aircraft_states": False,
            },
        }

        with open(path, "w") as f:
            if yaml:
                yaml.dump(default_config, f, default_flow_style=False)
            else:
                # Fallback to JSON if yaml not available
                import json

                json.dump(default_config, f, indent=2)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path) as f:
                if yaml and self.config_path.endswith(".yaml"):
                    return yaml.safe_load(f)
                import json

                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration"""
        return {
            "bluesky": {
                "connection_type": "local",
                "simulation": {
                    "separation_standards": {
                        "horizontal_nm": 5.0,
                        "vertical_ft": 1000.0,
                    },
                },
                "mock_data": {
                    "default_aircraft_count": 10,
                },
            },
        }

    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated path"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value


# Global configuration instance
_config = BlueSkyConfig()


class BlueSkyInterface:
    """Interface for interacting with BlueSky simulator"""

    def __init__(self, strict_mode: bool = False) -> None:
        self.bluesky_available = BLUESKY_AVAILABLE
        self.connection_type = _config.get("bluesky.connection_type", "local")
        self.network_config = _config.get("bluesky.network", {})
        self.simulation_initialized = False
        self.strict_mode = strict_mode  # If True, fail instead of using mock data

        # Initialize connection if BlueSky is available
        if self.bluesky_available:
            self._initialize_bluesky()

    def _initialize_bluesky(self) -> None:
        """Initialize BlueSky simulator"""
        try:
            if self.connection_type == "local":
                # Initialize BlueSky for local use
                if hasattr(bs, "init"):
                    logging.info("Initializing BlueSky with bs.init()...")
                    bs.init()
                    logging.info("BlueSky core initialization completed")

                    # Verify that simulation modules are now available
                    from bluesky import sim, traf

                    if not hasattr(sim, "simt"):
                        msg = "Simulation module not properly initialized"
                        raise Exception(msg)
                    if not hasattr(traf, "id"):
                        msg = "Traffic module not properly initialized"
                        raise Exception(msg)

                    logging.info("BlueSky simulation and traffic modules verified")
                else:
                    msg = "BlueSky init method not available"
                    raise Exception(msg)

                # Set up simulation parameters
                self._setup_simulation()
                self.simulation_initialized = True
                logging.info("BlueSky simulator initialized successfully")

            elif self.connection_type == "network":
                # For network connections, we'd implement socket communication here
                self._test_network_connection()
                self.simulation_initialized = True
                logging.info("BlueSky network connection established")

        except Exception as e:
            logging.exception(f"Failed to initialize BlueSky: {e}")
            self.bluesky_available = False

    def _setup_simulation(self) -> None:
        """Setup simulation parameters"""
        try:
            # Initialize BlueSky simulation properly
            if hasattr(bs, "stack"):
                # 1. Initial condition - resets and initializes simulation
                logging.info("Setting up BlueSky simulation with IC command...")
                bs.stack.stack("IC")

                # 2. Set simulation area (default to Amsterdam area)
                area_cmd = "AREA EHAM"  # Amsterdam area
                logging.info(f"Setting simulation area: {area_cmd}")
                bs.stack.stack(area_cmd)

                # 3. Set conflict detection method
                cd_method = _config.get(
                    "bluesky.simulation.conflict_detection_method", "SWARM"
                )
                logging.info(f"Setting conflict detection method: {cd_method}")
                bs.stack.stack(f"CDMETHOD {cd_method}")

                # 4. Set separation standards
                h_sep = _config.get(
                    "bluesky.simulation.separation_standards.horizontal_nm", 5.0
                )
                v_sep = _config.get(
                    "bluesky.simulation.separation_standards.vertical_ft", 1000.0
                )
                logging.info(
                    f"Setting separation standards: {h_sep}nm horizontal, {v_sep}ft vertical",
                )
                bs.stack.stack(f"CDSEP {h_sep} {v_sep}")

                # 5. Start simulation
                logging.info("Starting BlueSky simulation with OP command...")
                bs.stack.stack("OP")

                # 6. Step simulation once to ensure proper initialization
                bs.sim.step()
                logging.info("BlueSky simulation stepped for initialization")

                logging.info("BlueSky simulation setup completed successfully")

        except Exception as e:
            logging.warning(f"Failed to setup simulation parameters: {e}")

    def _test_network_connection(self) -> None:
        """Test network connection to BlueSky"""
        host = self.network_config.get("host", "localhost")
        port = self.network_config.get("port", 8080)
        timeout = self.network_config.get("timeout", 10.0)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            if result != 0:
                msg = f"Cannot connect to BlueSky at {host}:{port}"
                raise ConnectionError(msg)

        except Exception as e:
            msg = f"Network connection test failed: {e}"
            raise ConnectionError(msg)

    def is_available(self) -> bool:
        """Check if BlueSky is available and initialized"""
        return self.bluesky_available and self.simulation_initialized

    def get_aircraft_data(self) -> dict[str, Any]:
        """Get real aircraft data from BlueSky"""
        if not self.is_available():
            if self.strict_mode:
                msg = "BlueSky not available and strict mode is enabled - cannot use mock data"
                raise BlueSkyToolsError(msg)
            return self._get_mock_aircraft_data()

        try:
            aircraft_dict = {}

            # Use bs.traf (the actual simulation traffic) instead of the imported traf module
            if hasattr(bs, "traf") and bs.traf is not None:
                traffic = bs.traf
                logging.debug(f"Checking bs.traf - has id: {hasattr(traffic, 'id')}")
                if hasattr(traffic, "id"):
                    logging.debug(f"bs.traf.id length: {len(traffic.id)}")
                    logging.debug(
                        f"bs.traf.id contents: {list(traffic.id) if len(traffic.id) > 0 else 'empty'}",
                    )

                # Get aircraft data from BlueSky traffic module
                if hasattr(traffic, "id") and hasattr(traffic, "lat"):
                    for i, acid in enumerate(traffic.id):
                        if i < len(traffic.lat) and i < len(traffic.lon):
                            aircraft_dict[acid] = {
                                "id": acid,
                                "lat": float(traffic.lat[i]),
                                "lon": float(traffic.lon[i]),
                                "alt": (
                                    float(traffic.alt[i])
                                    if i < len(traffic.alt)
                                    else 35000.0
                                ),
                                "hdg": (
                                    float(traffic.hdg[i])
                                    if i < len(traffic.hdg)
                                    else 0.0
                                ),
                                "spd": (
                                    float(traffic.tas[i])
                                    if i < len(traffic.tas)
                                    else 250.0
                                ),
                                "vs": (
                                    float(traffic.vs[i]) if i < len(traffic.vs) else 0.0
                                ),
                                "type": (
                                    traffic.type[i] if i < len(traffic.type) else "B738"
                                ),
                                "callsign": acid,
                            }

            logging.debug(f"Found {len(aircraft_dict)} aircraft in BlueSky")

            # If no aircraft in BlueSky
            if len(aircraft_dict) == 0:
                if self.strict_mode:
                    msg = "No aircraft in BlueSky simulation and strict mode is enabled"
                    raise BlueSkyToolsError(msg)
                logging.info("No aircraft in BlueSky simulation traffic module")
                return self._get_mock_aircraft_data()

            return {
                "aircraft": aircraft_dict,
                "timestamp": time.time(),
                "total_aircraft": len(aircraft_dict),
                "simulation_time": (
                    getattr(sim, "simt", time.time())
                    if hasattr(sim, "simt")
                    else time.time()
                ),
                "source": "bluesky_real",
            }

        except BlueSkyToolsError:
            raise  # Re-raise strict mode errors
        except Exception as e:
            logging.exception(f"Error getting real aircraft data: {e}")
            if self.strict_mode:
                msg = f"Failed to get real aircraft data: {e}"
                raise BlueSkyToolsError(msg)
            return self._get_mock_aircraft_data()

    def get_conflict_data(self) -> dict[str, Any]:
        """Get real conflict data from BlueSky"""
        if not self.is_available():
            return self._get_mock_conflict_data()

        try:
            conflicts = []

            # Access BlueSky's conflict detection system
            if hasattr(traf, "cd") and hasattr(traf.cd, "conflicts"):
                # Get conflicts from BlueSky's conflict detection
                cd_conflicts = traf.cd.conflicts

                for i, (ac1_idx, ac2_idx) in enumerate(cd_conflicts):
                    if ac1_idx < len(traf.id) and ac2_idx < len(traf.id):
                        ac1_id = traf.id[ac1_idx]
                        ac2_id = traf.id[ac2_idx]

                        # Calculate separation
                        h_sep = self._calculate_horizontal_separation(ac1_idx, ac2_idx)
                        v_sep = (
                            abs(traf.alt[ac1_idx] - traf.alt[ac2_idx])
                            if ac1_idx < len(traf.alt) and ac2_idx < len(traf.alt)
                            else 0
                        )

                        conflicts.append(
                            {
                                "conflict_id": f"CONF_{i + 1:03d}",
                                "aircraft_1": ac1_id,
                                "aircraft_2": ac2_id,
                                "horizontal_separation": h_sep,
                                "vertical_separation": v_sep,
                                "time_to_cpa": 120,  # Would need to calculate from BlueSky data
                                "severity": self._assess_conflict_severity(
                                    h_sep, v_sep
                                ),
                                "predicted_cpa_lat": (
                                    traf.lat[ac1_idx] + traf.lat[ac2_idx]
                                )
                                / 2,
                                "predicted_cpa_lon": (
                                    traf.lon[ac1_idx] + traf.lon[ac2_idx]
                                )
                                / 2,
                                "predicted_cpa_time": time.time() + 120,
                            },
                        )

            return {
                "conflicts": conflicts,
                "total_conflicts": len(conflicts),
                "timestamp": time.time(),
                "high_priority_conflicts": len(
                    [c for c in conflicts if c["severity"] == "high"]
                ),
                "medium_priority_conflicts": len(
                    [c for c in conflicts if c["severity"] == "medium"],
                ),
                "low_priority_conflicts": len(
                    [c for c in conflicts if c["severity"] == "low"]
                ),
                "source": "bluesky_real",
            }

        except Exception as e:
            logging.exception(f"Error getting real conflict data: {e}")
            return self._get_mock_conflict_data()

    def _calculate_horizontal_separation(self, ac1_idx: int, ac2_idx: int) -> float:
        """Calculate horizontal separation between two aircraft"""
        try:
            if (
                ac1_idx < len(traf.lat)
                and ac2_idx < len(traf.lat)
                and ac1_idx < len(traf.lon)
                and ac2_idx < len(traf.lon)
            ):
                return haversine_distance(
                    traf.lat[ac1_idx],
                    traf.lon[ac1_idx],
                    traf.lat[ac2_idx],
                    traf.lon[ac2_idx],
                )
        except:
            pass
        return 5.0  # Default safe separation

    def _assess_conflict_severity(self, h_sep: float, v_sep: float) -> str:
        """Assess conflict severity based on separation"""
        h_min = _config.get(
            "bluesky.simulation.separation_standards.horizontal_nm", 5.0
        )
        v_min = _config.get(
            "bluesky.simulation.separation_standards.vertical_ft", 1000.0
        )

        if h_sep < h_min * 0.6 and v_sep < v_min * 0.6:
            return "high"
        if h_sep < h_min * 0.8 and v_sep < v_min * 0.8:
            return "medium"
        return "low"

    def send_bluesky_command(self, command: str) -> dict[str, Any]:
        """Send command to BlueSky simulator"""
        if not self.is_available():
            return self._simulate_command_execution(command)

        try:
            # Send command through BlueSky's stack
            if hasattr(bs, "stack"):
                result = bs.stack.stack(command)
                success = True
            elif hasattr(stack, "stack"):
                result = stack.stack(command)
                success = True
            else:
                msg = "BlueSky stack not available"
                raise Exception(msg)

            # If this was a CRE (create aircraft) command, step simulation to make aircraft visible
            if command.strip().upper().startswith("CRE"):
                try:
                    bs.sim.step()
                    logging.debug(f"Stepped simulation after CRE command: {command}")
                except Exception as e:
                    logging.warning(f"Failed to step simulation after CRE command: {e}")

            return {
                "command": command,
                "status": "executed",
                "success": success,
                "timestamp": time.time(),
                "response": (
                    result if result else f"{command.split()[0]} command acknowledged"
                ),
                "source": "bluesky_real",
            }

        except Exception as e:
            logging.exception(f"Error sending BlueSky command '{command}': {e}")
            return {
                "command": command,
                "status": "failed",
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "source": "bluesky_real",
            }

    def step_simulation_real(
        self, minutes: float, dtmult: float = 1.0
    ) -> dict[str, Any]:
        """Step the real BlueSky simulation forward"""
        if not self.is_available():
            return self._simulate_step(minutes, dtmult)

        try:
            # Set time multiplier if different
            if dtmult != 1.0:
                logging.info(f"Setting time multiplier to {dtmult}")
                self.send_bluesky_command(f"DTMULT {dtmult}")

            # Step simulation forward using FF (fast-forward) instead of DT
            # FF advances simulation by given amount of time
            ff_minutes = minutes * dtmult
            logging.info(f"Advancing simulation by {ff_minutes} minutes")
            cmd_result = self.send_bluesky_command(f"FF {ff_minutes:.2f}")

            return {
                "action": "step_simulation",
                "minutes_advanced": minutes,
                "seconds_advanced": minutes * 60,
                "dtmult": dtmult,
                "simulation_time": (
                    getattr(sim, "simt", time.time())
                    if hasattr(sim, "simt")
                    else time.time()
                ),
                "command_result": cmd_result,
                "status": "completed" if cmd_result.get("success") else "failed",
                "success": cmd_result.get("success", False),
                "source": "bluesky_real",
            }

        except Exception as e:
            logging.exception(f"Error stepping real simulation: {e}")
            return {
                "action": "step_simulation",
                "status": "failed",
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "source": "bluesky_real",
            }

    def reset_simulation_real(self) -> dict[str, Any]:
        """Reset the real BlueSky simulation"""
        if not self.is_available():
            return self._simulate_reset()

        try:
            # Send reset command first
            logging.info("Resetting BlueSky simulation...")
            reset_result = self.send_bluesky_command("RESET")

            # Re-setup simulation parameters with proper sequence
            self._setup_simulation()

            # Get current aircraft count after reset
            aircraft_count = len(traf.id) if hasattr(traf, "id") else 0

            return {
                "action": "reset_simulation",
                "reset_command": reset_result,
                "simulation_state": "initialized_and_running",
                "aircraft_count": aircraft_count,
                "timestamp": time.time(),
                "success": reset_result.get("success", False),
                "status": "completed" if reset_result.get("success") else "failed",
                "source": "bluesky_real",
                "setup_commands": [
                    "IC",
                    "AREA EHAM",
                    "CDMETHOD SWARM",
                    "CDSEP 5.0 1000",
                    "OP",
                ],
            }

        except Exception as e:
            logging.exception(f"Error resetting real simulation: {e}")
            return {
                "action": "reset_simulation",
                "status": "failed",
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "source": "bluesky_real",
            }

    def _get_mock_aircraft_data(self) -> dict[str, Any]:
        """Generate mock aircraft data when BlueSky unavailable"""
        aircraft_count = _config.get("bluesky.mock_data.default_aircraft_count", 10)
        bounds = _config.get("bluesky.mock_data.airspace_bounds", {})
        alt_range = _config.get("bluesky.mock_data.altitude_range", {})

        aircraft_dict = {}

        # Add default aircraft for backward compatibility with tests
        default_aircraft = [
            {
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
            {
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
        ]

        # Add default aircraft
        for ac in default_aircraft:
            aircraft_dict[ac["id"]] = ac

        # Add additional aircraft based on configuration
        for i in range(max(0, aircraft_count - len(default_aircraft))):
            acid = f"AC{i + 1:03d}"
            aircraft_dict[acid] = {
                "id": acid,
                "lat": bounds.get("lat_min", 51.0)
                + (bounds.get("lat_max", 53.0) - bounds.get("lat_min", 51.0))
                * (i / max(1, aircraft_count)),
                "lon": bounds.get("lon_min", 3.0)
                + (bounds.get("lon_max", 6.0) - bounds.get("lon_min", 3.0))
                * (i / max(1, aircraft_count)),
                "alt": (
                    alt_range.get("min_fl", 200)
                    + (alt_range.get("max_fl", 400) - alt_range.get("min_fl", 200))
                    * (i / max(1, aircraft_count))
                )
                * 100,
                "hdg": (i * 36) % 360,  # Spread headings
                "spd": 250 + (i * 10) % 200,  # Vary speeds
                "vs": 0,
                "type": ["B738", "A320", "B777", "A330"][i % 4],
                "callsign": acid,
            }

        return {
            "aircraft": aircraft_dict,
            "timestamp": time.time(),
            "total_aircraft": len(aircraft_dict),
            "simulation_time": time.time(),
            "source": "mock_data",
        }

    def _get_mock_conflict_data(self) -> dict[str, Any]:
        """Generate mock conflict data when BlueSky unavailable"""
        return {
            "conflicts": [
                {
                    "conflict_id": "MOCK_CONF_001",
                    "aircraft_1": "AAL123",
                    "aircraft_2": "DLH456",
                    "horizontal_separation": 4.2,
                    "vertical_separation": 0,
                    "time_to_cpa": 120,
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
            "source": "mock_data",
        }

    def _simulate_command_execution(self, command: str) -> dict[str, Any]:
        """Simulate command execution when BlueSky unavailable"""
        command_parts = command.strip().split()
        command_type = command_parts[0].upper() if command_parts else "UNKNOWN"

        return {
            "command": command,
            "command_type": command_type,
            "status": "simulated",
            "success": True,
            "timestamp": time.time(),
            "response": f"{command_type} command simulated",
            "source": "mock_simulation",
        }

    def _simulate_step(self, minutes: float, dtmult: float) -> dict[str, Any]:
        """Simulate stepping when BlueSky unavailable"""
        return {
            "action": "step_simulation",
            "minutes_advanced": minutes,
            "seconds_advanced": minutes * 60,
            "dtmult": dtmult,
            "simulation_time": time.time(),
            "status": "simulated",
            "success": True,
            "source": "mock_simulation",
        }

    def _simulate_reset(self) -> dict[str, Any]:
        """Simulate reset when BlueSky unavailable"""
        return {
            "action": "reset_simulation",
            "simulation_state": "simulated_reset",
            "aircraft_count": 0,
            "timestamp": time.time(),
            "success": True,
            "status": "simulated",
            "source": "mock_simulation",
        }


# Global BlueSky interface instance
_bluesky_interface = BlueSkyInterface()


def set_strict_mode(enabled: bool = True) -> None:
    """Enable or disable strict mode for BlueSky operations"""
    global _bluesky_interface
    _bluesky_interface.strict_mode = enabled
    logging.info(f"BlueSky strict mode {'enabled' if enabled else 'disabled'}")


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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula

    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees

    Returns:
        Distance in nautical miles
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Earth's radius in nautical miles
    earth_radius_nm = 3440.065

    return c * earth_radius_nm


def get_all_aircraft_info() -> dict[str, Any]:
    """
    Get information about all aircraft in the simulation

    Returns:
        Dictionary containing aircraft information
    """
    try:
        logging.info("Getting all aircraft information")

        # Use the BlueSky interface to get real or mock data
        aircraft_data = _bluesky_interface.get_aircraft_data()

        logging.info(
            "Retrieved information for %d aircraft (source: %s)",
            aircraft_data.get("total_aircraft", 0),
            aircraft_data.get("source", "unknown"),
        )
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

        # Use the BlueSky interface to get real or mock conflict data
        conflict_data = _bluesky_interface.get_conflict_data()

        logging.info(
            "Retrieved %d conflicts (source: %s)",
            conflict_data.get("total_conflicts", 0),
            conflict_data.get("source", "unknown"),
        )
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
            "ALT",
            "HDG",
            "SPD",
            "CRE",
            "DEL",
            "DEST",
            "DIRECT",
            "LNAV",
            "DT",
            "DTMULT",
            "VS",
            "GO",
            "RESET",
            "AREA",
            "CDMETHOD",
            "CDSEP",
            "WIND",
            "TURB",
            "PAUSE",
            "UNPAUSE",
            "FF",
            "IC",
            "OP",
        ]

        if command_type not in valid_commands:
            logging.warning("Unknown command type: %s", command_type)

        # Use BlueSky interface to send the command
        result = _bluesky_interface.send_bluesky_command(command)

        # Add additional metadata for compatibility
        if "command_type" not in result:
            result["command_type"] = command_type
        if "affected_aircraft" not in result and len(command_parts) > 1:
            result["affected_aircraft"] = command_parts[1]

        logging.info(
            "Command executed: %s -> %s", command, result.get("status", "unknown")
        )
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
    scenario_type: str,
    similarity_threshold: float = 0.8,
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


def get_weather_info(
    lat: float | None = None, lon: float | None = None
) -> dict[str, Any]:
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
        horizontal_nm = haversine_distance(
            ac1["lat"],
            ac1["lon"],
            ac2["lat"],
            ac2["lon"],
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
            "total_3d_nm": total_3d_nm,
        }

        logging.info(
            "Distance computed: %.2f nm horizontal, %.0f ft vertical, %.2f nm 3D",
            horizontal_nm,
            vertical_ft,
            total_3d_nm,
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
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
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
        logging.info(
            "Stepping simulation forward by %.2f minutes (dtmult=%.1f)", minutes, dtmult
        )

        # Use BlueSky interface to step the simulation
        result = _bluesky_interface.step_simulation_real(minutes, dtmult)

        # Add backward compatibility fields
        if result.get("success"):
            result["command_sent"] = f"DT {minutes * 60:.0f}"

        logging.info(
            "Simulation stepped forward successfully (source: %s)",
            result.get("source", "unknown"),
        )
        return result

    except Exception as e:
        logging.exception("Error stepping simulation")
        return {
            "action": "step_simulation",
            "status": "failed",
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


def reset_simulation() -> dict[str, Any]:
    """
    Reset the BlueSky simulation to initial state.

    Returns:
        Status dictionary with reset information
    """
    try:
        logging.info("Resetting BlueSky simulation")

        # Use BlueSky interface to reset the simulation
        result = _bluesky_interface.reset_simulation_real()

        # Add backward compatibility fields
        if result.get("success"):
            result["setup_commands"] = [
                "IC",
                "DTMULT 1",
                "AREA EHAM",
                "CDMETHOD SWARM",
                "CDSEP 5.0 1000",
                "OP",
            ]

        logging.info(
            "Simulation reset completed (source: %s)", result.get("source", "unknown")
        )
        return result

    except Exception as e:
        logging.exception("Error resetting simulation")
        return {
            "action": "reset_simulation",
            "status": "failed",
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        }


def get_minimum_separation() -> dict[str, float]:
    """
    Get the current minimum separation standards.

    Returns:
        Dictionary with minimum separation requirements
    """
    return {
        "horizontal_nm": 5.0,  # Standard horizontal separation
        "vertical_ft": 1000.0,  # Standard vertical separation
        "approach_horizontal_nm": 3.0,  # Approach phase horizontal
        "approach_vertical_ft": 500.0,  # Approach phase vertical
        "terminal_horizontal_nm": 3.0,  # Terminal area horizontal
        "oceanic_horizontal_nm": 10.0,  # Oceanic separation
        "rvsm_vertical_ft": 1000.0,  # RVSM vertical separation
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
                "vertical_ft": min_sep["vertical_ft"],
            },
            "violations": {
                "horizontal": horizontal_violation,
                "vertical": vertical_violation,
                "separation_loss": separation_violation,
            },
            "safety_margins": {
                "horizontal_nm": distances["horizontal_nm"] - min_sep["horizontal_nm"],
                "vertical_ft": distances["vertical_ft"] - min_sep["vertical_ft"],
            },
            "timestamp": time.time(),
        }

        if separation_violation:
            logging.warning(
                "SEPARATION VIOLATION: %s and %s - %.2f nm horizontal, %.0f ft vertical",
                aircraft_id1,
                aircraft_id2,
                distances["horizontal_nm"],
                distances["vertical_ft"],
            )

        return result

    except Exception as e:
        logging.exception("Error checking separation violation")
        return {
            "aircraft_pair": [aircraft_id1, aircraft_id2],
            "error": str(e),
            "timestamp": time.time(),
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
