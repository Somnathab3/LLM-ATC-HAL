#!/usr/bin/env python3
"""
Enhanced Conflict-Based ATC Training Data Generator
==================================================

Generates diverse, conflict-rich training scenarios with varied outputs to address
the duplication issues in the current training data.

Key Improvements:
- Guaranteed conflict scenarios with varied severity levels
- Diverse action commands (turns, altitude changes, speed adjustments)
- Rich contextual variations (weather, aircraft types, airspace)
- Multiple instruction templates and reasoning patterns
- Parameterized conflict generation for controllable diversity

Author: Enhanced ATC Training Data Team
"""

import json
import random
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ConflictScenario:
    """Represents a structured conflict scenario with all parameters"""
    # Aircraft information
    ownship_callsign: str
    ownship_altitude: int
    ownship_heading: int
    ownship_speed: int
    ownship_type: str
    
    # Conflict information
    conflicting_aircraft: List[Dict[str, Any]]
    conflict_type: str  # "head_on", "overtaking", "crossing", "converging"
    separation_distance: float  # NM
    time_to_closest_approach: float  # minutes
    conflict_severity: str  # "low", "medium", "high", "critical"
    
    # Environmental factors
    weather_conditions: str
    airspace_type: str
    traffic_density: str
    time_of_day: str
    
    # Resolution parameters
    required_action: str
    action_urgency: str
    alternative_actions: List[str]
    
    def __post_init__(self):
        """Validate scenario parameters"""
        if self.separation_distance < 0:
            raise ValueError("Separation distance cannot be negative")
        if not self.conflicting_aircraft:
            raise ValueError("Must have at least one conflicting aircraft")


class ConflictScenarioGenerator:
    """Generates diverse conflict scenarios with guaranteed variety"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        # Aircraft types with different characteristics
        self.aircraft_types = {
            "B737": {"category": "medium", "speed_range": (240, 280), "typical_alt": (25000, 39000)},
            "A320": {"category": "medium", "speed_range": (235, 275), "typical_alt": (25000, 39000)},
            "B777": {"category": "heavy", "speed_range": (250, 290), "typical_alt": (30000, 43000)},
            "A350": {"category": "heavy", "speed_range": (255, 295), "typical_alt": (30000, 43000)},
            "B787": {"category": "heavy", "speed_range": (245, 285), "typical_alt": (30000, 43000)},
            "CRJ2": {"category": "light", "speed_range": (220, 260), "typical_alt": (20000, 35000)},
            "E190": {"category": "medium", "speed_range": (225, 265), "typical_alt": (22000, 37000)},
            "C172": {"category": "light", "speed_range": (90, 120), "typical_alt": (2000, 8000)},
            "BE20": {"category": "light", "speed_range": (180, 220), "typical_alt": (15000, 25000)},
        }
        
        # Callsign patterns for diversity
        self.airline_prefixes = ["AAL", "UAL", "DAL", "SWA", "JBU", "ASA", "FFT", "SKW", "RPA", "PDT"]
        self.general_aviation = ["N", "C-G", "D-E", "G-B", "F-G"]
        
        # Weather conditions affecting operations
        self.weather_conditions = [
            "clear skies, unlimited visibility",
            "scattered clouds at 8000 feet",
            "overcast layer at 12000 feet", 
            "light rain, visibility 6 miles",
            "moderate turbulence reported",
            "wind shear advisory active",
            "thunderstorms 20 miles northeast",
            "fog reducing visibility to 2 miles",
            "strong crosswinds 15G25 knots",
            "icing conditions above 8000 feet"
        ]
        
        # Airspace types
        self.airspace_types = [
            "terminal control area",
            "approach control airspace", 
            "departure control airspace",
            "enroute center airspace",
            "class B airspace",
            "class C airspace",
            "terminal radar service area",
            "special use airspace transition"
        ]
        
        # Traffic density levels
        self.traffic_densities = [
            "light traffic conditions",
            "moderate traffic volume",
            "heavy traffic saturation",
            "peak hour operations",
            "complex traffic pattern",
            "multiple simultaneous approaches"
        ]
        
        # Time periods affecting operations
        self.time_periods = [
            "morning rush hour",
            "midday operations",
            "evening departure push",
            "late night operations",
            "weekend leisure traffic",
            "holiday travel period"
        ]
    
    def generate_callsign(self, aircraft_type: str) -> str:
        """Generate realistic callsign based on aircraft type"""
        if aircraft_type in ["C172", "BE20"]:
            prefix = random.choice(self.general_aviation)
            if prefix == "N":
                return f"N{random.randint(100, 999)}{random.choice(['AB', 'CD', 'EF', 'GH'])}"
            else:
                return f"{prefix}{random.randint(100, 999)}"
        else:
            prefix = random.choice(self.airline_prefixes)
            return f"{prefix}{random.randint(100, 9999)}"
    
    def generate_conflict_scenario(self, conflict_type: str = None, severity: str = None) -> ConflictScenario:
        """Generate a single conflict scenario with specified parameters"""
        
        # Select conflict type
        if not conflict_type:
            conflict_type = random.choice(["head_on", "overtaking", "crossing", "converging"])
        
        # Select severity
        if not severity:
            severity = random.choices(
                ["low", "medium", "high", "critical"],
                weights=[0.2, 0.4, 0.3, 0.1]  # Bias toward more serious conflicts
            )[0]
        
        # Generate ownship
        ownship_type = random.choice(list(self.aircraft_types.keys()))
        ownship_specs = self.aircraft_types[ownship_type]
        
        ownship_callsign = self.generate_callsign(ownship_type)
        ownship_altitude = random.randint(*ownship_specs["typical_alt"]) // 100 * 100
        ownship_heading = random.randint(0, 359)
        ownship_speed = random.randint(*ownship_specs["speed_range"])
        
        # Generate conflicting aircraft based on conflict type and severity
        conflicting_aircraft = self._generate_conflicting_aircraft(
            conflict_type, severity, ownship_altitude, ownship_heading, ownship_speed
        )
        
        # Calculate separation and timing based on severity
        separation_distance, time_to_closest = self._calculate_conflict_geometry(
            conflict_type, severity, conflicting_aircraft
        )
        
        # Generate environmental factors
        weather = random.choice(self.weather_conditions)
        airspace = random.choice(self.airspace_types)
        traffic = random.choice(self.traffic_densities)
        time_period = random.choice(self.time_periods)
        
        # Determine required actions based on conflict characteristics
        required_action, urgency, alternatives = self._determine_resolution_actions(
            conflict_type, severity, separation_distance, conflicting_aircraft
        )
        
        return ConflictScenario(
            ownship_callsign=ownship_callsign,
            ownship_altitude=ownship_altitude,
            ownship_heading=ownship_heading,
            ownship_speed=ownship_speed,
            ownship_type=ownship_type,
            conflicting_aircraft=conflicting_aircraft,
            conflict_type=conflict_type,
            separation_distance=separation_distance,
            time_to_closest_approach=time_to_closest,
            conflict_severity=severity,
            weather_conditions=weather,
            airspace_type=airspace,
            traffic_density=traffic,
            time_of_day=time_period,
            required_action=required_action,
            action_urgency=urgency,
            alternative_actions=alternatives
        )
    
    def _generate_conflicting_aircraft(self, conflict_type: str, severity: str, 
                                     ownship_alt: int, ownship_hdg: int, ownship_spd: int) -> List[Dict[str, Any]]:
        """Generate conflicting aircraft appropriate for the scenario"""
        num_conflicts = 1
        if severity in ["high", "critical"]:
            num_conflicts = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        elif severity == "medium":
            num_conflicts = random.choices([1, 2], weights=[0.8, 0.2])[0]
        
        conflicts = []
        for i in range(num_conflicts):
            aircraft_type = random.choice(list(self.aircraft_types.keys()))
            specs = self.aircraft_types[aircraft_type]
            
            callsign = self.generate_callsign(aircraft_type)
            
            # Generate conflict geometry
            if conflict_type == "head_on":
                # Aircraft approaching from opposite direction
                conflict_heading = (ownship_hdg + 180 + random.randint(-20, 20)) % 360
                altitude = ownship_alt + random.choice([-500, 0, 500])
            elif conflict_type == "overtaking":
                # Faster aircraft from behind
                conflict_heading = ownship_hdg + random.randint(-30, 30)
                altitude = ownship_alt + random.choice([-1000, 0, 1000])
            elif conflict_type == "crossing":
                # Aircraft on crossing path
                conflict_heading = (ownship_hdg + random.choice([-90, 90]) + random.randint(-30, 30)) % 360
                altitude = ownship_alt + random.choice([-1000, -500, 0, 500, 1000])
            else:  # converging
                # Aircraft converging on similar track
                conflict_heading = ownship_hdg + random.randint(-60, 60)
                altitude = ownship_alt + random.choice([-500, 0, 500])
            
            # Adjust speed based on conflict type
            base_speed = random.randint(*specs["speed_range"])
            if conflict_type == "overtaking":
                speed = base_speed + random.randint(20, 50)  # Faster
            else:
                speed = base_speed + random.randint(-20, 20)
            
            # Calculate relative position
            distance = self._get_initial_distance(severity)
            bearing = random.randint(0, 359)
            
            conflicts.append({
                "callsign": callsign,
                "aircraft_type": aircraft_type,
                "altitude": altitude,
                "heading": conflict_heading,
                "speed": speed,
                "distance": distance,
                "bearing": bearing,
                "category": specs["category"]
            })
        
        return conflicts
    
    def _get_initial_distance(self, severity: str) -> float:
        """Get initial distance based on conflict severity"""
        if severity == "critical":
            return random.uniform(2.0, 4.0)  # Very close
        elif severity == "high":
            return random.uniform(3.0, 6.0)  # Close
        elif severity == "medium":
            return random.uniform(5.0, 10.0)  # Moderate
        else:  # low
            return random.uniform(8.0, 15.0)  # Distant but developing
    
    def _calculate_conflict_geometry(self, conflict_type: str, severity: str, 
                                   conflicting_aircraft: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate separation distance and time to closest approach"""
        
        # Get minimum separation from conflicting aircraft
        min_distance = min(ac["distance"] for ac in conflicting_aircraft)
        
        # Adjust based on severity
        severity_factor = {
            "critical": 0.5,
            "high": 0.7,
            "medium": 0.85,
            "low": 1.0
        }
        
        separation = min_distance * severity_factor[severity]
        
        # Calculate time to closest approach (simplified)
        if conflict_type == "head_on":
            time_to_closest = random.uniform(1.0, 4.0)  # Quick approach
        elif conflict_type == "overtaking":
            time_to_closest = random.uniform(3.0, 8.0)  # Gradual overtake
        else:
            time_to_closest = random.uniform(2.0, 6.0)  # Variable timing
        
        return separation, time_to_closest
    
    def _determine_resolution_actions(self, conflict_type: str, severity: str, 
                                    separation: float, conflicting_aircraft: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
        """Determine appropriate resolution actions based on conflict characteristics"""
        
        # Action urgency based on severity
        urgency_map = {
            "critical": "immediate action required",
            "high": "prompt action needed", 
            "medium": "coordinated response required",
            "low": "monitor and prepare for action"
        }
        urgency = urgency_map[severity]
        
        # Primary action based on conflict type and severity
        if conflict_type == "head_on":
            if severity in ["critical", "high"]:
                primary_actions = [
                    f"turn right 30 degrees immediately",
                    f"turn left 30 degrees immediately", 
                    f"climb to FL{random.randint(250, 400)}",
                    f"descend to FL{random.randint(180, 300)}"
                ]
            else:
                primary_actions = [
                    f"turn right 15 degrees for separation",
                    f"turn left 15 degrees for separation",
                    f"adjust altitude by 1000 feet",
                    f"reduce speed to {random.randint(200, 240)} knots"
                ]
        
        elif conflict_type == "overtaking":
            if severity in ["critical", "high"]:
                primary_actions = [
                    f"reduce speed to {random.randint(180, 220)} knots immediately",
                    f"climb to FL{random.randint(280, 400)}",
                    f"turn right 20 degrees for spacing",
                    f"descend to FL{random.randint(200, 320)}"
                ]
            else:
                primary_actions = [
                    f"adjust speed to {random.randint(200, 240)} knots",
                    f"minor heading adjustment left 10 degrees",
                    f"request altitude change to FL{random.randint(250, 350)}"
                ]
        
        elif conflict_type == "crossing":
            if severity in ["critical", "high"]:
                primary_actions = [
                    f"turn left 45 degrees for immediate separation",
                    f"turn right 45 degrees for immediate separation",
                    f"climb expedite to FL{random.randint(300, 400)}",
                    f"descend expedite to FL{random.randint(180, 280)}"
                ]
            else:
                primary_actions = [
                    f"turn left 20 degrees",
                    f"turn right 20 degrees", 
                    f"adjust speed to {random.randint(200, 270)} knots",
                    f"climb to FL{random.randint(260, 380)}"
                ]
        
        else:  # converging
            if severity in ["critical", "high"]:
                primary_actions = [
                    f"vector heading {random.randint(10, 359)} degrees",
                    f"reduce speed to {random.randint(180, 210)} knots",
                    f"climb to FL{random.randint(290, 410)}",
                    f"turn right 30 degrees immediately"
                ]
            else:
                primary_actions = [
                    f"slight left turn to heading {random.randint(10, 359)}",
                    f"adjust speed to {random.randint(220, 260)} knots",
                    f"maintain current altitude, monitor traffic"
                ]
        
        required_action = random.choice(primary_actions)
        
        # Generate alternative actions
        all_alternatives = [
            f"vector heading {random.randint(10, 359)} degrees",
            f"climb to FL{random.randint(250, 400)}",
            f"descend to FL{random.randint(180, 300)}",
            f"reduce speed to {random.randint(180, 230)} knots",
            f"increase speed to {random.randint(260, 300)} knots",
            f"turn left {random.randint(10, 45)} degrees",
            f"turn right {random.randint(10, 45)} degrees",
            f"hold current heading and monitor",
            f"request priority handling",
            f"coordinate with adjacent sectors"
        ]
        
        # Remove the chosen action from alternatives
        alternatives = [alt for alt in all_alternatives if alt != required_action]
        selected_alternatives = random.sample(alternatives, min(3, len(alternatives)))
        
        return required_action, urgency, selected_alternatives


class DiverseInstructionGenerator:
    """Generates diverse instruction templates to reduce duplication"""
    
    def __init__(self):
        self.instruction_templates = [
            "As an air traffic controller, analyze the current situation and provide the appropriate action with explanation.",
            "You are managing air traffic. Based on the scenario, determine the best course of action and explain your reasoning.",
            "Given the following air traffic situation, what action should be taken and why?",
            "As the controlling authority, assess this aircraft conflict scenario and provide your resolution with rationale.",
            "Analyze this air traffic control scenario and determine the optimal resolution strategy with detailed explanation.",
            "You are responsible for aircraft separation. Based on the current situation, what is your recommended action?",
            "As an experienced air traffic controller, how would you resolve this conflict situation? Provide action and reasoning.",
            "Given the traffic scenario below, determine the appropriate control action and explain the safety considerations.",
            "You are the radar controller. Based on the current traffic situation, what immediate action is required?",
            "As the sector controller, analyze this conflict scenario and provide your resolution with tactical explanation."
        ]
        
        self.scenario_intro_templates = [
            "Current traffic situation:",
            "Radar contact shows:",
            "Traffic scenario:",
            "Sector status:",
            "Aircraft conflict situation:",
            "Active traffic pattern:",
            "Current airspace situation:",
            "Developing traffic scenario:"
        ]
    
    def get_instruction(self) -> str:
        """Get a diverse instruction template"""
        return random.choice(self.instruction_templates)
    
    def get_scenario_intro(self) -> str:
        """Get a diverse scenario introduction"""
        return random.choice(self.scenario_intro_templates)


class ConflictTrainingDataGenerator:
    """Main class for generating diverse conflict-based training data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.scenario_generator = ConflictScenarioGenerator(self.config.get("seed"))
        self.instruction_generator = DiverseInstructionGenerator()
        self.samples = []
        self.stats = defaultdict(int)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "seed": 42,
            "samples_per_environment": 5000,
            "conflict_types": ["head_on", "overtaking", "crossing", "converging"],
            "severity_distribution": {
                "critical": 0.15,
                "high": 0.35,
                "medium": 0.35,
                "low": 0.15
            },
            "environments": ["HorizontalCREnv-v0", "VerticalCREnv-v0", "SectorCREnv-v0", "MergeEnv-v0"]
        }
    
    def generate_scenario_description(self, scenario: ConflictScenario) -> str:
        """Convert conflict scenario to natural language description"""
        
        intro = self.instruction_generator.get_scenario_intro()
        
        # Aircraft information
        aircraft_info = f"Aircraft {scenario.ownship_callsign} ({scenario.ownship_type}), FL{scenario.ownship_altitude//100}, heading {scenario.ownship_heading}°, {scenario.ownship_speed} knots"
        
        # Conflict information
        conflict_desc = f"Traffic conflict: {scenario.conflict_type} situation with {len(scenario.conflicting_aircraft)} aircraft"
        
        # Add details about conflicting aircraft
        conflict_details = []
        for i, ac in enumerate(scenario.conflicting_aircraft):
            detail = f"{ac['callsign']} ({ac['aircraft_type']}) at FL{ac['altitude']//100}, {ac['distance']:.1f} NM {ac['bearing']:.0f}° bearing, heading {ac['heading']}°"
            conflict_details.append(detail)
        
        # Separation and timing
        separation_info = f"Current separation: {scenario.separation_distance:.1f} NM, time to closest approach: {scenario.time_to_closest_approach:.1f} minutes"
        
        # Environmental factors
        env_info = f"Conditions: {scenario.weather_conditions}, {scenario.airspace_type}, {scenario.traffic_density}, {scenario.time_of_day}"
        
        # Combine all elements
        full_description = f"{intro} {aircraft_info}. {conflict_desc}. {'; '.join(conflict_details)}. {separation_info}. {env_info}. Conflict severity: {scenario.conflict_severity}."
        
        return full_description
    
    def generate_action_output(self, scenario: ConflictScenario) -> str:
        """Generate diverse action output with detailed reasoning"""
        
        # Action statement
        action_output = f"Action: {scenario.required_action}"
        
        # Detailed reasoning based on scenario characteristics
        reasoning_parts = []
        
        # Conflict analysis
        reasoning_parts.append(f"Analysis: {scenario.conflict_type} conflict with {scenario.conflict_severity} severity")
        
        # Separation assessment
        if scenario.separation_distance < 3.0:
            reasoning_parts.append(f"Critical separation of {scenario.separation_distance:.1f} NM requires immediate intervention")
        elif scenario.separation_distance < 5.0:
            reasoning_parts.append(f"Reduced separation of {scenario.separation_distance:.1f} NM necessitates prompt action")
        else:
            reasoning_parts.append(f"Developing conflict with {scenario.separation_distance:.1f} NM separation requires proactive management")
        
        # Timing considerations
        if scenario.time_to_closest_approach < 2.0:
            reasoning_parts.append(f"Immediate action required with {scenario.time_to_closest_approach:.1f} minutes to closest approach")
        else:
            reasoning_parts.append(f"Coordinated response needed with {scenario.time_to_closest_approach:.1f} minutes available for resolution")
        
        # Environmental considerations
        if "turbulence" in scenario.weather_conditions or "wind" in scenario.weather_conditions:
            reasoning_parts.append("Weather conditions factor into maneuvering constraints")
        
        if "heavy" in scenario.traffic_density:
            reasoning_parts.append("High traffic density requires careful coordination with adjacent aircraft")
        
        # Alternative considerations
        if scenario.alternative_actions:
            alt_text = random.choice([
                f"Alternative: {random.choice(scenario.alternative_actions)}",
                f"Backup option: {random.choice(scenario.alternative_actions)}",
                f"Secondary action if needed: {random.choice(scenario.alternative_actions)}"
            ])
            reasoning_parts.append(alt_text)
        
        # Combine action and reasoning
        full_output = f"{action_output}\n\nExplanation: {'. '.join(reasoning_parts)}."
        
        return full_output
    
    def generate_training_sample(self, environment: str) -> Dict[str, Any]:
        """Generate a single diverse training sample"""
        
        # Generate scenario appropriate for environment
        if environment == "HorizontalCREnv-v0":
            conflict_types = ["head_on", "overtaking", "crossing"]
        elif environment == "VerticalCREnv-v0":
            conflict_types = ["converging"]  # Focus on altitude conflicts
        elif environment == "SectorCREnv-v0":
            conflict_types = ["head_on", "crossing", "converging"]
        elif environment == "MergeEnv-v0":
            conflict_types = ["converging", "overtaking"]
        else:
            conflict_types = self.config["conflict_types"]
        
        conflict_type = random.choice(conflict_types)
        severity = random.choices(
            list(self.config["severity_distribution"].keys()),
            weights=list(self.config["severity_distribution"].values())
        )[0]
        
        # Generate scenario
        scenario = self.scenario_generator.generate_conflict_scenario(conflict_type, severity)
        
        # Generate instruction and input
        instruction = self.instruction_generator.get_instruction()
        input_text = f"Environment: {environment}\n\n{self.generate_scenario_description(scenario)}"
        
        # Generate output
        output = self.generate_action_output(scenario)
        
        # Create training sample
        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "metadata": {
                "environment": environment,
                "conflict_type": conflict_type,
                "severity": severity,
                "scenario_id": f"{environment}_{conflict_type}_{severity}_{random.randint(1000, 9999)}",
                "generated_timestamp": datetime.now().isoformat(),
                "separation_distance": scenario.separation_distance,
                "num_conflicts": len(scenario.conflicting_aircraft),
                "action_type": self._classify_action(scenario.required_action)
            }
        }
        
        # Update statistics
        self.stats[f"{environment}_samples"] += 1
        self.stats[f"conflict_type_{conflict_type}"] += 1
        self.stats[f"severity_{severity}"] += 1
        self.stats[f"action_type_{sample['metadata']['action_type']}"] += 1
        
        return sample
    
    def _classify_action(self, action: str) -> str:
        """Classify action type for statistics"""
        action_lower = action.lower()
        if "turn" in action_lower or "heading" in action_lower:
            return "heading_change"
        elif "climb" in action_lower:
            return "climb"
        elif "descend" in action_lower:
            return "descend"
        elif "speed" in action_lower and ("reduce" in action_lower or "slow" in action_lower):
            return "speed_reduction"
        elif "speed" in action_lower and ("increase" in action_lower or "fast" in action_lower):
            return "speed_increase"
        elif "vector" in action_lower:
            return "vector"
        elif "maintain" in action_lower:
            return "maintain"
        else:
            return "other"
    
    def generate_dataset(self, num_samples_per_env: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate complete diverse dataset"""
        
        if num_samples_per_env is None:
            num_samples_per_env = self.config["samples_per_environment"]
        
        logger.info(f"Generating {num_samples_per_env} samples per environment")
        
        all_samples = []
        
        for environment in self.config["environments"]:
            logger.info(f"Generating data for {environment}")
            
            env_samples = []
            for i in range(num_samples_per_env):
                try:
                    sample = self.generate_training_sample(environment)
                    env_samples.append(sample)
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"  Generated {i + 1}/{num_samples_per_env} samples for {environment}")
                        
                except Exception as e:
                    logger.error(f"Error generating sample {i} for {environment}: {e}")
                    continue
            
            all_samples.extend(env_samples)
            logger.info(f"Completed {environment}: {len(env_samples)} samples generated")
        
        self.samples = all_samples
        logger.info(f"Total samples generated: {len(all_samples)}")
        
        return all_samples
    
    def save_dataset(self, output_path: str, samples: List[Dict[str, Any]] = None) -> None:
        """Save dataset to JSONL format"""
        
        if samples is None:
            samples = self.samples
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(samples)} samples to {output_path}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset saved successfully to {output_path}")
    
    def save_statistics(self, stats_path: str) -> None:
        """Save generation statistics"""
        
        stats_data = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_samples": len(self.samples),
            "samples_per_environment": self.config["samples_per_environment"],
            "environments": self.config["environments"],
            "statistics": dict(self.stats),
            "diversity_metrics": self._calculate_diversity_metrics()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")
    
    def _calculate_diversity_metrics(self) -> Dict[str, Any]:
        """Calculate diversity metrics for the generated dataset"""
        
        if not self.samples:
            return {}
        
        # Instruction diversity
        instructions = [sample["instruction"] for sample in self.samples]
        unique_instructions = len(set(instructions))
        
        # Output diversity
        outputs = [sample["output"] for sample in self.samples]
        unique_outputs = len(set(outputs))
        
        # Input diversity
        inputs = [sample["input"] for sample in self.samples]
        unique_inputs = len(set(inputs))
        
        # Action diversity
        action_types = [sample["metadata"]["action_type"] for sample in self.samples]
        unique_action_types = len(set(action_types))
        
        return {
            "instruction_diversity": {
                "total": len(instructions),
                "unique": unique_instructions,
                "duplication_rate": 1 - (unique_instructions / len(instructions))
            },
            "output_diversity": {
                "total": len(outputs),
                "unique": unique_outputs,
                "duplication_rate": 1 - (unique_outputs / len(outputs))
            },
            "input_diversity": {
                "total": len(inputs),
                "unique": unique_inputs,
                "duplication_rate": 1 - (unique_inputs / len(inputs))
            },
            "action_type_distribution": dict(Counter(action_types)),
            "unique_action_types": unique_action_types
        }


def main():
    """Main function with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description="Generate diverse, conflict-rich ATC training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate default dataset
    python enhanced_conflict_data_generator.py --output enhanced_training_data.jsonl
    
    # Generate with specific parameters
    python enhanced_conflict_data_generator.py --samples-per-env 3000 --output custom_data.jsonl
    
    # Generate with custom seed for reproducibility
    python enhanced_conflict_data_generator.py --seed 123 --output reproducible_data.jsonl
        """
    )
    
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--samples-per-env", type=int, default=5000, 
                       help="Number of samples per environment (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--stats-file", default=None,
                       help="Statistics output file (default: <output>_stats.json)")
    parser.add_argument("--environments", nargs="+", 
                       default=["HorizontalCREnv-v0", "VerticalCREnv-v0", "SectorCREnv-v0", "MergeEnv-v0"],
                       help="Environments to generate data for")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare configuration
    config = {
        "seed": args.seed,
        "samples_per_environment": args.samples_per_env,
        "environments": args.environments,
        "conflict_types": ["head_on", "overtaking", "crossing", "converging"],
        "severity_distribution": {
            "critical": 0.15,
            "high": 0.35,
            "medium": 0.35,
            "low": 0.15
        }
    }
    
    # Generate dataset
    logger.info("Starting enhanced conflict-based training data generation")
    logger.info(f"Configuration: {config}")
    
    generator = ConflictTrainingDataGenerator(config)
    samples = generator.generate_dataset()
    
    # Save results
    generator.save_dataset(args.output)
    
    if args.stats_file:
        stats_path = args.stats_file
    else:
        stats_path = str(Path(args.output).with_suffix('')) + '_stats.json'
    
    generator.save_statistics(stats_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples generated: {len(samples):,}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Statistics file: {stats_path}")
    
    # Print diversity metrics
    diversity_metrics = generator._calculate_diversity_metrics()
    if diversity_metrics:
        logger.info(f"\nDiversity Metrics:")
        logger.info(f"  Instruction duplication rate: {diversity_metrics['instruction_diversity']['duplication_rate']:.1%}")
        logger.info(f"  Output duplication rate: {diversity_metrics['output_diversity']['duplication_rate']:.1%}")
        logger.info(f"  Input duplication rate: {diversity_metrics['input_diversity']['duplication_rate']:.1%}")
        logger.info(f"  Unique action types: {diversity_metrics['unique_action_types']}")
    
    logger.info("Generation completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
