# scenarios/monte_carlo_framework.py
"""
BlueSky-integrated Monte Carlo Framework for ATC Scenario Generation
===================================================================
This module replaces all hard-coded scenario generation with BlueSky-based
sampling using only ranges defined in scenario_ranges.yaml.

Key Features:
- Range-based parameter specification
- BlueSky command generation for realistic scenarios
- Monte Carlo sampling with proper statistics
- Command logging for validation
"""

import yaml
import logging
import random
import math
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os
import json

# BlueSky imports
try:
    from bluesky import stack, sim, traf
    from bluesky.stack import stack as bs_stack
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False
    logging.warning("BlueSky not available - using mock generation")


class ComplexityTier(Enum):
    """Scenario complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"


@dataclass
class ScenarioConfiguration:
    """Generated scenario configuration"""
    aircraft_count: int
    aircraft_types: List[str]
    positions: List[Dict[str, float]]  # lat, lon, alt
    speeds: List[int]  # knots
    headings: List[int]  # degrees
    environmental_conditions: Dict[str, Any]
    bluesky_commands: List[str]
    complexity_tier: ComplexityTier
    duration_minutes: float
    generated_timestamp: float
    airspace_region: str = "EHAM_TMA"  # Default region
    distribution_shift_tier: str = "in_distribution"  # Track shift tier applied
    
    @property
    def aircraft_list(self) -> List[Dict[str, Any]]:
        """Generate aircraft_list for backward compatibility"""
        aircraft_list = []
        for i in range(self.aircraft_count):
            callsign = f'AC{i+1:03d}'
            aircraft = {
                'id': callsign,  # Add id field for compatibility
                'aircraft_type': self.aircraft_types[i] if i < len(self.aircraft_types) else 'B737',
                'latitude': self.positions[i]['lat'] if i < len(self.positions) else 52.3,
                'longitude': self.positions[i]['lon'] if i < len(self.positions) else 4.8,
                'altitude': self.positions[i]['alt'] if i < len(self.positions) else 35000,
                'heading': self.headings[i] if i < len(self.headings) else 90,
                'ground_speed': self.speeds[i] if i < len(self.speeds) else 350
                # Remove callsign and vertical_speed as they're not expected by validator
            }
            aircraft_list.append(aircraft)
        return aircraft_list

    @property
    def environmental(self) -> Dict[str, Any]:
        """Generate environmental data for backward compatibility"""
        # Extract weather conditions from environmental_conditions
        env_conditions = self.environmental_conditions.copy()
        
        # Add weather field based on wind conditions
        wind_speed = env_conditions.get('wind_speed_kts', 0)
        visibility = env_conditions.get('visibility_nm', 10)
        
        if wind_speed > 30 or visibility < 3:
            weather = 'STORM'
        elif wind_speed > 15 or visibility < 6:
            weather = 'FOG'  # Use FOG instead of CLOUDY
        else:
            weather = 'CLEAR'
        
        # Return with standardized field names expected by validator
        return {
            'weather': weather,
            'wind_speed': wind_speed,  # Use standard field name
            'visibility': visibility,  # Use standard field name
            'wind_direction': env_conditions.get('wind_direction_deg', 0),  # Use standard field name
            'turbulence_intensity': env_conditions.get('turbulence_intensity', 0.0)
        }


class BlueSkyScenarioGenerator:
    """
    Generates ATC scenarios using BlueSky commands based on ranges from YAML config.
    Replaces all hard-coded and random generators with range-based sampling.
    Supports distribution shift scenarios via apply_distribution_shift helper.
    """
    
    def __init__(self, ranges_file: str = "scenario_ranges.yaml", 
                 distribution_shift_file: str = "distribution_shift_levels.yaml",
                 ranges_dict: Optional[Dict[str, Any]] = None):
        """Initialize generator with range configuration"""
        self.ranges_file = ranges_file
        self.distribution_shift_file = distribution_shift_file
        self.command_log = []
        self.logger = logging.getLogger(__name__)
        
        # Validate BlueSky availability
        if not BLUESKY_AVAILABLE:
            self.logger.warning("BlueSky not available - generator will use mock commands")
            
        # Use provided ranges_dict or load from file
        if ranges_dict is not None:
            self.ranges = ranges_dict
            self.logger.info("Using provided ranges dictionary")
        else:
            self.ranges = self._load_ranges()
        
        self.shift_config = self._load_distribution_shift_config()
    
    def _load_ranges(self) -> Dict[str, Any]:
        """Load scenario ranges from YAML configuration"""
        try:
            with open(self.ranges_file, 'r') as f:
                ranges = yaml.safe_load(f)
            self.logger.info(f"Loaded scenario ranges from {self.ranges_file}")
            return ranges
        except Exception as e:
            self.logger.error(f"Failed to load ranges: {e}")
            return self._get_default_ranges()
    
    def _load_distribution_shift_config(self) -> Dict[str, Any]:
        """Load distribution shift configuration from YAML"""
        try:
            with open(self.distribution_shift_file, 'r') as f:
                shift_config = yaml.safe_load(f)
            self.logger.info(f"Loaded distribution shift config from {self.distribution_shift_file}")
            return shift_config
        except Exception as e:
            self.logger.warning(f"Failed to load distribution shift config: {e}")
            return {'distribution_shift_tiers': {}}
    
    def _get_default_ranges(self) -> Dict[str, Any]:
        """Fallback ranges if YAML file is not available"""
        return {
            'aircraft': {
                'count': {'simple': [2, 3], 'moderate': [4, 6], 'complex': [8, 12], 'extreme': [18, 25]},
                'types': {'pool': ['B737', 'A320', 'A321', 'B777'], 'weights': [0.3, 0.3, 0.2, 0.2]}
            },
            'geography': {
                'airspace_regions': {
                    'EHAM_TMA': {'center': [52.3086, 4.7639], 'radius_nm': [40, 60]}
                }
            },
            'altitude': {'min_fl': 100, 'max_fl': 410, 'step_fl': 10},
            'speed': {'cas_range_kts': [250, 480]},
            'heading': {'range_degrees': [0, 360]},
            'simulation': {'duration_minutes': [5, 15]},
            'weather': {'wind': {'speed_kts': [0, 80], 'direction_deg': [0, 360]}},
            'traffic': {'density_multiplier': [0.3, 2.0]}
        }
    
    def sample_from_range(self, range_spec: Any) -> Any:
        """Sample a value from a range specification"""
        if isinstance(range_spec, list) and len(range_spec) == 2:
            if isinstance(range_spec[0], int) and isinstance(range_spec[1], int):
                # Ensure min <= max to avoid empty range error
                min_val = min(range_spec[0], range_spec[1])
                max_val = max(range_spec[0], range_spec[1])
                return random.randint(min_val, max_val)
            else:
                # Ensure min <= max for float ranges too
                min_val = min(range_spec[0], range_spec[1])
                max_val = max(range_spec[0], range_spec[1])
                return random.uniform(min_val, max_val)
        elif isinstance(range_spec, (int, float, str)):
            return range_spec
        else:
            return range_spec
    
    def weighted_choice(self, choices: List[Any], weights: Optional[List[float]] = None) -> Any:
        """Make a weighted choice from options"""
        if weights and len(weights) == len(choices):
            return random.choices(choices, weights=weights)[0]
        return random.choice(choices)
    
    def apply_distribution_shift(self, base_ranges: Dict[str, Any], shift_tier: str) -> Dict[str, Any]:
        """
        Apply distribution shift to base ranges according to specified tier.
        
        This function warps the YAML ranges based on the shift configuration,
        ensuring all concrete values still come from BlueSky command sampling.
        
        Args:
            base_ranges: Original ranges from scenario_ranges.yaml
            shift_tier: Tier name from distribution_shift_levels.yaml
                       ('in_distribution', 'moderate_shift', 'extreme_shift')
        
        Returns:
            Modified ranges with shifts applied
        """
        if shift_tier == 'in_distribution' or shift_tier not in self.shift_config.get('distribution_shift_tiers', {}):
            # No shift - return base ranges unchanged
            return base_ranges.copy()
        
        shifted_ranges = base_ranges.copy()
        shift_config = self.shift_config['distribution_shift_tiers'][shift_tier]
        
        self.logger.info(f"Applying {shift_tier} distribution shift")
        
        # 1. Apply traffic density multiplier to aircraft counts
        if 'traffic_density_multiplier' in shift_config:
            multiplier = shift_config['traffic_density_multiplier']
            for complexity in shifted_ranges['aircraft']['count']:
                base_range = shifted_ranges['aircraft']['count'][complexity]
                new_min = max(1, int(base_range[0] * multiplier))
                new_max = max(new_min, min(25, int(base_range[1] * multiplier)))  # Ensure max >= min
                shifted_ranges['aircraft']['count'][complexity] = [new_min, new_max]
        
        # 2. Apply aircraft pool shifts
        aircraft_config = shift_config.get('aircraft', {})
        if not aircraft_config.get('use_baseline_pool', True):
            if 'alternative_pool' in aircraft_config:
                alt_pool = aircraft_config['alternative_pool']
                shifted_ranges['aircraft']['types'] = {
                    'pool': alt_pool['types'],
                    'weights': alt_pool['weights']
                }
        
        # 3. Apply weather shifts
        weather_config = shift_config.get('weather', {})
        
        # Wind speed shifts
        if 'wind' in weather_config and 'speed_shift_kts' in weather_config['wind']:
            wind_shift = weather_config['wind']['speed_shift_kts']
            base_wind = shifted_ranges['weather']['wind']['speed_kts']
            new_min = max(0, base_wind[0] + wind_shift[0])
            new_max = max(new_min, min(100, base_wind[1] + wind_shift[1]))  # Ensure max >= min
            shifted_ranges['weather']['wind']['speed_kts'] = [new_min, new_max]
        
        # Wind direction shifts
        if 'wind' in weather_config and 'direction_shift_deg' in weather_config['wind']:
            dir_shift = weather_config['wind']['direction_shift_deg']
            base_dir = shifted_ranges['weather']['wind']['direction_deg']
            
            # Apply shift and ensure valid range
            new_min = (base_dir[0] + dir_shift[0]) % 360
            new_max = (base_dir[1] + dir_shift[1]) % 360
            
            # Ensure min <= max, if not, wrap around or expand range
            if new_min > new_max:
                # If range wraps around 360°, use full range to avoid issues
                shifted_ranges['weather']['wind']['direction_deg'] = [0, 360]
            else:
                shifted_ranges['weather']['wind']['direction_deg'] = [new_min, new_max]
        
        # Turbulence intensity shifts
        if 'turbulence' in weather_config and 'intensity_shift' in weather_config['turbulence']:
            turb_shift = weather_config['turbulence']['intensity_shift']
            base_turb = shifted_ranges['weather'].get('turbulence_factor', [0.0, 0.3])
            new_min = max(0.0, base_turb[0] + turb_shift[0])
            new_max = max(new_min, min(1.0, base_turb[1] + turb_shift[1]))  # Ensure max >= min
            shifted_ranges['weather']['turbulence_factor'] = [new_min, new_max]
        
        # Visibility degradation
        if 'visibility' in weather_config and 'degradation_factor' in weather_config['visibility']:
            degradation = weather_config['visibility']['degradation_factor']
            for vis_type in ['clear_nm', 'reduced_nm']:
                if vis_type in shifted_ranges['weather'].get('visibility', {}):
                    base_vis = shifted_ranges['weather']['visibility'][vis_type]
                    new_min = max(0.5, base_vis[0] * degradation)
                    new_max = max(new_min, max(1.0, base_vis[1] * degradation))  # Ensure max >= min
                    shifted_ranges['weather']['visibility'][vis_type] = [new_min, new_max]
        
        # 4. Apply airspace complexity shifts
        airspace_config = shift_config.get('airspace', {})
        if 'sector_density_multiplier' in airspace_config:
            density_mult = airspace_config['sector_density_multiplier']
            if 'traffic' not in shifted_ranges:
                shifted_ranges['traffic'] = {}
            
            base_density = shifted_ranges['traffic'].get('density_multiplier', [0.5, 1.5])
            new_min = max(0.1, base_density[0] * density_mult)
            new_max = max(new_min, min(3.0, base_density[1] * density_mult))  # Ensure max >= min
            shifted_ranges['traffic']['density_multiplier'] = [new_min, new_max]
        
        # 5. Apply navigation error parameters (new ranges for error injection)
        nav_config = shift_config.get('navigation', {})
        if nav_config.get('error_injection_rate', 0) > 0:
            if 'navigation' not in shifted_ranges:
                shifted_ranges['navigation'] = {}
            
            shifted_ranges['navigation']['error_injection_rate'] = nav_config['error_injection_rate']
            shifted_ranges['navigation']['error_magnitude_nm'] = nav_config.get('error_magnitude_nm', [0.1, 1.0])
            shifted_ranges['navigation']['system_reliability'] = nav_config.get('system_reliability', 0.95)
        
        # 6. Apply geography shifts (if any region-specific modifications)
        geography_config = shift_config.get('geography', {})
        if 'radius_expansion_factor' in geography_config:
            expansion = geography_config['radius_expansion_factor']
            for region in shifted_ranges['geography']['airspace_regions']:
                base_radius = shifted_ranges['geography']['airspace_regions'][region]['radius_nm']
                new_min = base_radius[0] * expansion
                new_max = max(new_min, base_radius[1] * expansion)  # Ensure max >= min
                shifted_ranges['geography']['airspace_regions'][region]['radius_nm'] = [
                    new_min, new_max
                ]
        
        self.logger.info(f"Distribution shift applied: {shift_tier}")
        return shifted_ranges
    
    def generate_scenario(self, 
                         complexity_tier: ComplexityTier = ComplexityTier.MODERATE,
                         force_conflicts: bool = True,
                         airspace_region: Optional[str] = None,
                         distribution_shift_tier: str = 'in_distribution') -> ScenarioConfiguration:
        """
        Generate a complete scenario using BlueSky commands.
        
        Args:
            complexity_tier: Scenario complexity level
            force_conflicts: Whether to force conflict situations
            airspace_region: Specific airspace region to use
            distribution_shift_tier: Distribution shift tier to apply
                                   ('in_distribution', 'moderate_shift', 'extreme_shift')
            
        Returns:
            ScenarioConfiguration with all parameters and BlueSky commands
        """
        
        self.logger.info(f"Generating {complexity_tier.value} scenario with {distribution_shift_tier} shift")
        start_time = time.time()
        
        # Reset command log for this scenario
        self.command_log = []
        
        # Apply distribution shift to ranges
        shifted_ranges = self.apply_distribution_shift(self.ranges, distribution_shift_tier)
        
        # 1. Sample aircraft count based on complexity (using shifted ranges)
        aircraft_count = self.sample_from_range(
            shifted_ranges['aircraft']['count'][complexity_tier.value]
        )
        
        # 2. Sample aircraft types (using shifted ranges)
        aircraft_types = []
        type_pool = shifted_ranges['aircraft']['types']['pool']
        type_weights = shifted_ranges['aircraft']['types'].get('weights', None)
        
        for _ in range(aircraft_count):
            aircraft_type = self.weighted_choice(type_pool, type_weights)
            aircraft_types.append(aircraft_type)
        
        # 3. Select airspace region (using shifted ranges)
        if not airspace_region:
            airspace_region = random.choice(list(shifted_ranges['geography']['airspace_regions'].keys()))
        
        region_config = shifted_ranges['geography']['airspace_regions'][airspace_region]
        center_lat, center_lon = region_config['center']
        radius_range = region_config['radius_nm']
        radius_nm = self.sample_from_range(radius_range)
        
        # 4. Generate aircraft positions within region
        positions = []
        speeds = []
        headings = []
        
        for i in range(aircraft_count):
            # Sample position within circular region
            angle = random.uniform(0, 2 * math.pi)
            distance_nm = random.uniform(5, radius_nm)  # Minimum 5nm from center
            
            # Convert to lat/lon offset
            lat_offset = (distance_nm * math.cos(angle)) / 60.0  # 1 degree ≈ 60 nm
            lon_offset = (distance_nm * math.sin(angle)) / (60.0 * math.cos(math.radians(center_lat)))
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            
            # Sample altitude (using shifted ranges)
            min_fl = shifted_ranges['altitude']['min_fl']
            max_fl = shifted_ranges['altitude']['max_fl']
            step_fl = shifted_ranges['altitude']['step_fl']
            fl = random.randrange(min_fl, max_fl + 1, step_fl)
            alt_ft = fl * 100
            
            positions.append({'lat': lat, 'lon': lon, 'alt': alt_ft})
            
            # Sample speed (using shifted ranges)
            speed_range = shifted_ranges['speed']['cas_range_kts']
            speed = self.sample_from_range(speed_range)
            speeds.append(speed)
            
            # Sample heading (using shifted ranges)
            if force_conflicts and i > 0:
                # Create convergent headings for conflicts
                target_pos = positions[i-1]
                bearing = self._calculate_bearing(
                    lat, lon, target_pos['lat'], target_pos['lon']
                )
                # Add some variation
                heading = (bearing + random.uniform(-30, 30)) % 360
            else:
                heading_range = shifted_ranges['heading']['range_degrees']
                heading = self.sample_from_range(heading_range)
            
            headings.append(int(heading))
        
        # 5. Generate environmental conditions (using shifted ranges)
        environmental_conditions = self._generate_environmental_conditions(shifted_ranges)
        
        # 6. Generate BlueSky commands (with shift-aware parameters)
        bluesky_commands = self._generate_bluesky_commands(
            aircraft_count, aircraft_types, positions, speeds, headings,
            environmental_conditions, force_conflicts, shifted_ranges, distribution_shift_tier
        )
        
        # 7. Sample simulation duration (using shifted ranges)
        duration_minutes = self.sample_from_range(
            shifted_ranges['simulation']['duration_minutes']
        )
        
        scenario = ScenarioConfiguration(
            aircraft_count=aircraft_count,
            aircraft_types=aircraft_types,
            positions=positions,
            speeds=speeds,
            headings=headings,
            environmental_conditions=environmental_conditions,
            bluesky_commands=bluesky_commands,
            complexity_tier=complexity_tier,
            duration_minutes=duration_minutes,
            generated_timestamp=time.time(),
            airspace_region=airspace_region,
            distribution_shift_tier=distribution_shift_tier
        )
        
        generation_time = time.time() - start_time
        self.logger.info(f"Generated {distribution_shift_tier} scenario in {generation_time:.3f}s: "
                        f"{aircraft_count} aircraft, {len(bluesky_commands)} commands")
        
        return scenario
    
    def _generate_environmental_conditions(self, ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Generate environmental conditions from ranges (shift-aware)"""
        wind_config = ranges['weather']['wind']
        
        conditions = {
            'wind_speed_kts': self.sample_from_range(wind_config['speed_kts']),
            'wind_direction_deg': self.sample_from_range(wind_config['direction_deg']),
            'turbulence_intensity': self.sample_from_range(
                ranges['weather'].get('turbulence_factor', [0.0, 1.0])
            ),
            'visibility_nm': self.sample_from_range(
                ranges['weather'].get('visibility', {}).get('clear_nm', [8, 15])
            )
        }
        
        # Add navigation error injection if configured in shifted ranges
        if 'navigation' in ranges and 'error_injection_rate' in ranges['navigation']:
            if random.random() < ranges['navigation']['error_injection_rate']:
                conditions['navigation_error_nm'] = self.sample_from_range(
                    ranges['navigation']['error_magnitude_nm']
                )
                conditions['system_reliability'] = ranges['navigation']['system_reliability']
        
        return conditions
    
    def _generate_bluesky_commands(self, 
                                  aircraft_count: int,
                                  aircraft_types: List[str],
                                  positions: List[Dict[str, float]],
                                  speeds: List[int],
                                  headings: List[int],
                                  environmental_conditions: Dict[str, Any],
                                  force_conflicts: bool,
                                  ranges: Dict[str, Any],
                                  distribution_shift_tier: str) -> List[str]:
        """Generate BlueSky commands for scenario setup (shift-aware)"""
        commands = []
        
        # Reset simulation
        commands.append("RESET")
        commands.append("DTMULT 1")  # Real-time simulation
        
        # Set area (use first aircraft position as reference)
        ref_pos = positions[0]
        area_name = f"AREA {ref_pos['lat']:.2f},{ref_pos['lon']:.2f}"
        commands.append(area_name)
        
        # Set conflict detection
        commands.append("CDMETHOD SWARM")
        commands.append("CDSEP 5.0 1000")  # 5NM horizontal, 1000ft vertical
        
        # Set wind conditions (from shifted ranges)
        wind_cmd = f"WIND {ref_pos['lat']:.2f},{ref_pos['lon']:.2f},ALL," \
                  f"{environmental_conditions['wind_direction_deg']:.0f}," \
                  f"{environmental_conditions['wind_speed_kts']:.0f}"
        commands.append(wind_cmd)
        
        # Add turbulence if significant (extreme shift scenarios)
        if environmental_conditions.get('turbulence_intensity', 0) > 0.5:
            turb_cmd = f"TURB {environmental_conditions['turbulence_intensity']:.2f}"
            commands.append(turb_cmd)
        
        # Create aircraft using CRE commands
        for i in range(aircraft_count):
            callsign = f"AC{i+1:03d}"
            pos = positions[i]
            
            # CRE command: callsign, type, lat, lon, hdg, alt, spd
            cre_cmd = f"CRE {callsign} {aircraft_types[i]} " \
                     f"{pos['lat']:.4f} {pos['lon']:.4f} " \
                     f"{headings[i]} {pos['alt']:.0f} {speeds[i]}"
            
            commands.append(cre_cmd)
            
            # Apply navigation errors if configured in shift
            if 'navigation_error_nm' in environmental_conditions:
                error_nm = environmental_conditions['navigation_error_nm']
                # Inject position error via slight offset in CRE command
                error_lat = random.uniform(-error_nm/60, error_nm/60)  # Convert nm to degrees
                error_lon = random.uniform(-error_nm/60, error_nm/60)
                error_cmd = f"{callsign} MOVE {pos['lat']+error_lat:.4f} {pos['lon']+error_lon:.4f}"
                commands.append(f"DT 5")  # Small delay before error injection
                commands.append(error_cmd)
            
            # Add some variation in timing for realism
            if i > 0 and random.random() < 0.3:
                delay = random.randint(10, 60)  # 10-60 second delay
                commands.append(f"DT {delay}")
        
        # Add conflict-inducing maneuvers if requested
        if force_conflicts and aircraft_count >= 2:
            commands.extend(self._generate_conflict_commands(aircraft_count))
        
        # Set simulation time acceleration (from shifted traffic density)
        traffic_density = self.sample_from_range(ranges['traffic']['density_multiplier'])
        commands.append(f"DTMULT {traffic_density:.1f}")
        
        # Add distribution shift specific commands
        if distribution_shift_tier == 'extreme_shift':
            # Add emergency scenarios for extreme shift
            if random.random() < 0.1:  # 10% chance
                emergency_ac = f"AC{random.randint(1, aircraft_count):03d}"
                commands.append(f"DT 120")  # Wait 2 minutes
                commands.append(f"{emergency_ac} EMERGENCY FUEL")
        
        # Log commands for validation
        self.command_log.extend(commands)
        
        return commands
    
    def _generate_conflict_commands(self, aircraft_count: int) -> List[str]:
        """Generate commands to create conflict situations"""
        commands = []
        
        # Create converging paths between random aircraft pairs
        num_conflicts = min(aircraft_count // 2, 3)  # Max 3 conflicts
        
        for i in range(num_conflicts):
            ac1 = f"AC{i*2+1:03d}"
            ac2 = f"AC{i*2+2:03d}"
            
            # Add delayed heading changes to create conflicts
            delay = 60 + i * 30  # Stagger conflicts
            
            commands.append(f"DT {delay}")
            
            # Turn aircraft towards each other
            hdg_change1 = random.randint(15, 45)
            hdg_change2 = 180 + random.randint(-30, 30)  # Opposite direction
            
            commands.append(f"{ac1} HDG {hdg_change1}")
            commands.append(f"{ac2} HDG {hdg_change2}")
            
            # Optional altitude changes for 3D conflicts
            if random.random() < 0.4:
                alt_change = random.choice([-1000, -500, 500, 1000])
                target_ac = random.choice([ac1, ac2])
                commands.append(f"{target_ac} ALT +{alt_change}")
        
        return commands
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two lat/lon points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def execute_scenario(self, scenario: ScenarioConfiguration) -> Dict[str, Any]:
        """
        Execute scenario in BlueSky and return results.
        
        Args:
            scenario: Generated scenario configuration
            
        Returns:
            Execution results with conflicts detected and command log
        """
        
        if not BLUESKY_AVAILABLE:
            return self._mock_execution(scenario)
        
        try:
            # Execute BlueSky commands
            for cmd in scenario.bluesky_commands:
                stack.stack(cmd)
                self.logger.debug(f"Executed: {cmd}")
            
            # Run simulation for specified duration
            sim_time = scenario.duration_minutes * 60  # Convert to seconds
            end_time = sim.simt + sim_time
            
            conflicts_detected = []
            
            while sim.simt < end_time:
                sim.step()
                
                # Check for conflicts
                if hasattr(traf, 'cd') and traf.cd is not None:
                    if hasattr(traf.cd, 'confpairs_all') and traf.cd.confpairs_all:
                        for pair in traf.cd.confpairs_all:
                            ac1_idx, ac2_idx = pair
                            if ac1_idx < len(traf.id) and ac2_idx < len(traf.id):
                                conflict = {
                                    'id1': traf.id[ac1_idx],
                                    'id2': traf.id[ac2_idx],
                                    'time': sim.simt,
                                    'distance': getattr(traf.cd, 'dcpa', [0])[0] if hasattr(traf.cd, 'dcpa') else 0
                                }
                                conflicts_detected.append(conflict)
            
            return {
                'success': True,
                'conflicts_detected': conflicts_detected,
                'simulation_time': sim_time,
                'commands_executed': len(scenario.bluesky_commands),
                'aircraft_count': scenario.aircraft_count
            }
            
        except Exception as e:
            self.logger.error(f"BlueSky execution failed: {e}")
            return self._mock_execution(scenario)
    
    def _mock_execution(self, scenario: ScenarioConfiguration) -> Dict[str, Any]:
        """Mock execution when BlueSky is not available"""
        # Generate mock conflicts based on scenario complexity
        num_conflicts = {
            ComplexityTier.SIMPLE: 1,
            ComplexityTier.MODERATE: 2,
            ComplexityTier.COMPLEX: 3,
            ComplexityTier.EXTREME: 5
        }.get(scenario.complexity_tier, 2)
        
        conflicts_detected = []
        for i in range(min(num_conflicts, scenario.aircraft_count // 2)):
            conflict = {
                'id1': f'AC{i*2+1:03d}',
                'id2': f'AC{i*2+2:03d}',
                'time': random.uniform(60, scenario.duration_minutes * 60),
                'distance': random.uniform(3.0, 4.9)  # Close but not violation
            }
            conflicts_detected.append(conflict)
        
        return {
            'success': True,
            'conflicts_detected': conflicts_detected,
            'simulation_time': scenario.duration_minutes * 60,
            'commands_executed': len(scenario.bluesky_commands),
            'aircraft_count': scenario.aircraft_count,
            'mock': True
        }
    
    def generate_scenario_batch(self, 
                               count: int,
                               complexity_distribution: Optional[Dict[str, float]] = None,
                               distribution_shift_distribution: Optional[Dict[str, float]] = None) -> List[ScenarioConfiguration]:
        """
        Generate multiple scenarios for Monte Carlo testing.
        
        Args:
            count: Number of scenarios to generate
            complexity_distribution: Distribution of complexity levels
            distribution_shift_distribution: Distribution of shift tiers
            
        Returns:
            List of generated scenarios
        """
        
        if complexity_distribution is None:
            complexity_distribution = {
                'simple': 0.3,
                'moderate': 0.4,
                'complex': 0.2,
                'extreme': 0.1
            }
        
        if distribution_shift_distribution is None:
            distribution_shift_distribution = {
                'in_distribution': 0.5,
                'moderate_shift': 0.3,
                'extreme_shift': 0.2
            }
        
        scenarios = []
        complexity_tiers = list(ComplexityTier)
        tier_names = [tier.value for tier in complexity_tiers]
        tier_weights = [complexity_distribution.get(name, 0) for name in tier_names]
        
        shift_tiers = list(distribution_shift_distribution.keys())
        shift_weights = list(distribution_shift_distribution.values())
        
        self.logger.info(f"Generating {count} scenarios with complexity distribution: {complexity_distribution}")
        self.logger.info(f"Distribution shift distribution: {distribution_shift_distribution}")
        
        for i in range(count):
            if i % 100 == 0:
                self.logger.info(f"Generated {i}/{count} scenarios")
            
            # Sample complexity tier
            tier_name = self.weighted_choice(tier_names, tier_weights)
            complexity_tier = ComplexityTier(tier_name)
            
            # Sample distribution shift tier
            shift_tier = self.weighted_choice(shift_tiers, shift_weights)
            
            # Generate scenario with both complexity and shift
            scenario = self.generate_scenario(
                complexity_tier=complexity_tier,
                force_conflicts=True,
                distribution_shift_tier=shift_tier
            )
            
            scenarios.append(scenario)
        
        self.logger.info(f"Generated {count} scenarios successfully")
        return scenarios
    
    def get_command_log(self) -> List[str]:
        """Get the complete command log for validation"""
        return self.command_log.copy()
    
    def validate_ranges(self, scenario: ScenarioConfiguration) -> Dict[str, bool]:
        """
        Validate that generated scenario parameters are within specified ranges.
        
        Args:
            scenario: Generated scenario to validate
            
        Returns:
            Dictionary of validation results
        """
        
        validation = {
            'aircraft_count': True,
            'aircraft_types': True,
            'altitudes': True,
            'speeds': True,
            'headings': True,
            'environmental': True
        }
        
        # Validate aircraft count
        complexity_range = self.ranges['aircraft']['count'][scenario.complexity_tier.value]
        if not (complexity_range[0] <= scenario.aircraft_count <= complexity_range[1]):
            validation['aircraft_count'] = False
        
        # Validate aircraft types
        valid_types = set(self.ranges['aircraft']['types']['pool'])
        if not all(ac_type in valid_types for ac_type in scenario.aircraft_types):
            validation['aircraft_types'] = False
        
        # Validate altitudes
        min_alt = self.ranges['altitude']['min_fl'] * 100
        max_alt = self.ranges['altitude']['max_fl'] * 100
        if not all(min_alt <= pos['alt'] <= max_alt for pos in scenario.positions):
            validation['altitudes'] = False
        
        # Validate speeds
        speed_range = self.ranges['speed']['cas_range_kts']
        if not all(speed_range[0] <= speed <= speed_range[1] for speed in scenario.speeds):
            validation['speeds'] = False
        
        # Validate headings
        if not all(0 <= heading <= 360 for heading in scenario.headings):
            validation['headings'] = False
        
        # Validate environmental conditions
        wind_range = self.ranges['weather']['wind']['speed_kts']
        wind_speed = scenario.environmental_conditions.get('wind_speed_kts', 0)
        if not (wind_range[0] <= wind_speed <= wind_range[1]):
            validation['environmental'] = False
        
        return validation


# Convenience functions for backward compatibility
def generate_scenario(complexity_tier: str = "moderate", 
                     force_conflicts: bool = True,
                     distribution_shift_tier: str = 'in_distribution') -> ScenarioConfiguration:
    """Generate a single scenario - convenience function"""
    generator = BlueSkyScenarioGenerator()
    tier = ComplexityTier(complexity_tier)
    return generator.generate_scenario(tier, force_conflicts, distribution_shift_tier=distribution_shift_tier)


def generate_monte_carlo_scenarios(count: int, 
                                  complexity_distribution: Optional[Dict[str, float]] = None,
                                  distribution_shift_distribution: Optional[Dict[str, float]] = None) -> List[ScenarioConfiguration]:
    """Generate multiple scenarios for Monte Carlo testing - convenience function"""
    generator = BlueSkyScenarioGenerator()
    return generator.generate_scenario_batch(count, complexity_distribution, distribution_shift_distribution)


# Backward compatibility alias
MonteCarloScenarioGenerator = BlueSkyScenarioGenerator


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create generator
    generator = BlueSkyScenarioGenerator()
    
    # Generate scenarios with different distribution shifts
    scenarios = []
    
    # In-distribution scenario
    scenario_in_dist = generator.generate_scenario(
        ComplexityTier.MODERATE, 
        force_conflicts=True,
        distribution_shift_tier='in_distribution'
    )
    scenarios.append(('in_distribution', scenario_in_dist))
    
    # Moderate shift scenario
    scenario_mod_shift = generator.generate_scenario(
        ComplexityTier.MODERATE, 
        force_conflicts=True,
        distribution_shift_tier='moderate_shift'
    )
    scenarios.append(('moderate_shift', scenario_mod_shift))
    
    # Extreme shift scenario
    scenario_ext_shift = generator.generate_scenario(
        ComplexityTier.MODERATE, 
        force_conflicts=True,
        distribution_shift_tier='extreme_shift'
    )
    scenarios.append(('extreme_shift', scenario_ext_shift))
    
    # Display results
    for shift_type, scenario in scenarios:
        print(f"\n=== {shift_type.upper()} SCENARIO ===")
        print(f"Aircraft count: {scenario.aircraft_count}")
        print(f"Aircraft types: {scenario.aircraft_types[:3]}...")  # Show first 3
        print(f"Wind: {scenario.environmental_conditions['wind_speed_kts']:.1f} kts")
        print(f"Turbulence: {scenario.environmental_conditions['turbulence_intensity']:.2f}")
        print(f"BlueSky commands ({len(scenario.bluesky_commands)}):")
        for cmd in scenario.bluesky_commands[:5]:  # Show first 5 commands
            print(f"  {cmd}")
        
        # Validate ranges
        validation = generator.validate_ranges(scenario)
        print(f"Validation: {all(validation.values())} - {validation}")
    
    # Generate batch with distribution shifts
    print(f"\n=== BATCH GENERATION WITH SHIFTS ===")
    batch_scenarios = generator.generate_scenario_batch(
        count=10,
        complexity_distribution={'moderate': 0.6, 'complex': 0.4},
        distribution_shift_distribution={
            'in_distribution': 0.4,
            'moderate_shift': 0.4, 
            'extreme_shift': 0.2
        }
    )
    
    print(f"Generated {len(batch_scenarios)} scenarios with mixed shifts")
    
    # Count shift types in batch
    shift_counts = {}
    for scenario in batch_scenarios:
        shift_type = scenario.distribution_shift_tier
        shift_counts[shift_type] = shift_counts.get(shift_type, 0) + 1
    
    print(f"Shift distribution in batch: {shift_counts}")
