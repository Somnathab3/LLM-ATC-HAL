#!/usr/bin/env python3
"""
BlueSky-Gym ATC Decision-Making Fine-tuning Data Generator
=========================================================

This script generates training data for LLM fine-tuning by running expert SAC policies
in BlueSky-Gym environments and converting the experience to natural language format.

Features:
- Load trained SAC models from multiple ATC environments
- Generate expert demonstrations with natural language descriptions
- Convert observations and actions to aviation terminology
- Create structured training datasets for LLM fine-tuning
- Support for multiple environment types (Horizontal, Vertical, Sector, Merge)
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import SAC
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Structure for a single training sample"""
    environment: str
    scenario_id: str
    scenario_description: str
    observation_summary: str
    expert_action: str
    reasoning: str
    reward_components: Dict[str, float]
    safety_metrics: Dict[str, float]
    step_number: int
    episode_id: str


@dataclass
class EnvironmentConfig:
    """Configuration for environment-specific data generation"""
    name: str
    model_path: str
    config_path: str
    output_path: str
    num_episodes: int
    samples_target: int


class ObservationInterpreter:
    """Converts numerical observations to natural language descriptions"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load environment-specific interpretation templates"""
        templates = {
            "HorizontalCREnv-v0": {
                "aircraft_state": "Aircraft at altitude {altitude}ft, heading {heading}°, speed {speed}kts",
                "waypoint_info": "Waypoint at bearing {bearing}° distance {distance}nm",
                "drift_info": "Current drift from optimal path: {drift}°",
                "conflict_aircraft": "Conflicting aircraft at bearing {bearing}°, distance {distance}nm, relative speed {rel_speed}kts"
            },
            "VerticalCREnv-v0": {
                "aircraft_state": "Aircraft at {current_alt}ft climbing/descending at {vz}fpm toward {target_alt}ft",
                "runway_info": "Runway distance: {runway_dist}nm",
                "altitude_status": "Altitude deviation from target: {alt_dev}ft",
                "conflict_aircraft": "Conflicting aircraft at {alt}ft, vertical separation {v_sep}ft"
            },
            "SectorCREnv-v0": {
                "aircraft_state": "Aircraft at sector position ({x}, {y}), heading {heading}°, speed {speed}kts",
                "sector_info": "Drift from optimal sector exit: {drift}°",
                "traffic_density": "Traffic density: {count} aircraft in sector",
                "conflict_aircraft": "Conflicting aircraft at relative position ({rel_x}, {rel_y}), closure rate {closure}kts"
            },
            "MergeEnv-v0": {
                "aircraft_state": "Aircraft at merge position ({x}, {y}), heading {heading}°, speed {speed}kts",
                "waypoint_info": "Merge waypoint at bearing {bearing}° distance {distance}nm",
                "traffic_flow": "Position in merge sequence: {position}, lead aircraft spacing {spacing}nm",
                "conflict_aircraft": "Conflicting aircraft in merge area at relative position ({rel_x}, {rel_y})"
            }
        }
        return templates.get(self.environment_name, {})
    
    def interpret_observation(self, obs: np.ndarray, env_info: Dict[str, Any]) -> str:
        """Convert observation array to natural language description"""
        if self.environment_name == "HorizontalCREnv-v0":
            return self._interpret_horizontal_obs(obs, env_info)
        elif self.environment_name == "VerticalCREnv-v0":
            return self._interpret_vertical_obs(obs, env_info)
        elif self.environment_name == "SectorCREnv-v0":
            return self._interpret_sector_obs(obs, env_info)
        elif self.environment_name == "MergeEnv-v0":
            return self._interpret_merge_obs(obs, env_info)
        else:
            return f"Unknown environment observation: {obs.tolist()}"
    
    def _interpret_horizontal_obs(self, obs: np.ndarray, env_info: Dict[str, Any]) -> str:
        """Interpret horizontal conflict resolution observations"""
        # Assuming observation structure: [drift, airspeed, relative_positions...]
        drift = obs[0]
        airspeed = obs[1] if len(obs) > 1 else 250  # default speed
        heading = obs[2] if len(obs) > 2 else 90   # default heading
        
        description = f"Aircraft heading {heading:.0f}°, speed {airspeed:.0f}kts. "
        description += f"Current drift from waypoint: {drift:.1f}°. "
        
        # Process conflicting aircraft (assuming 5 values per aircraft)
        if len(obs) > 3:
            num_conflicts = (len(obs) - 3) // 5
            if num_conflicts > 0:
                description += f"Detected {num_conflicts} potential conflict(s): "
                for i in range(num_conflicts):
                    base_idx = 3 + i * 5
                    if base_idx + 4 < len(obs):
                        rel_x = obs[base_idx]
                        rel_y = obs[base_idx + 1]
                        distance = np.sqrt(rel_x**2 + rel_y**2)
                        bearing = np.degrees(np.arctan2(rel_y, rel_x))
                        description += f"Aircraft {i+1} at {distance:.1f}nm, bearing {bearing:.0f}°; "
        
        return description.strip()
    
    def _interpret_vertical_obs(self, obs: np.ndarray, env_info: Dict[str, Any]) -> str:
        """Interpret vertical conflict resolution observations"""
        current_alt = obs[0] if len(obs) > 0 else 10000
        vz = obs[1] if len(obs) > 1 else 0
        target_alt = obs[2] if len(obs) > 2 else 10000
        runway_dist = obs[3] if len(obs) > 3 else 50
        
        description = f"Aircraft at {current_alt:.0f}ft, "
        if vz > 0:
            description += f"climbing at {vz:.0f}fpm "
        elif vz < 0:
            description += f"descending at {abs(vz):.0f}fpm "
        else:
            description += "maintaining level flight "
        
        description += f"toward target {target_alt:.0f}ft. "
        description += f"Runway distance: {runway_dist:.1f}nm. "
        
        alt_dev = abs(current_alt - target_alt)
        if alt_dev > 100:
            description += f"Altitude deviation: {alt_dev:.0f}ft. "
        
        # Process conflicting aircraft
        if len(obs) > 4:
            num_conflicts = (len(obs) - 4) // 7
            if num_conflicts > 0:
                description += f"Vertical conflicts with {num_conflicts} aircraft: "
                for i in range(num_conflicts):
                    base_idx = 4 + i * 7
                    if base_idx + 2 < len(obs):
                        conflict_alt = obs[base_idx]
                        v_separation = abs(current_alt - conflict_alt)
                        description += f"Aircraft {i+1} at {conflict_alt:.0f}ft ({v_separation:.0f}ft separation); "
        
        return description.strip()
    
    def _interpret_sector_obs(self, obs: np.ndarray, env_info: Dict[str, Any]) -> str:
        """Interpret sector conflict resolution observations"""
        drift = obs[0] if len(obs) > 0 else 0
        airspeed = obs[1] if len(obs) > 1 else 250
        relative_pos = obs[2] if len(obs) > 2 else 0
        
        description = f"Aircraft in sector with {drift:.1f}° drift from optimal exit path. "
        description += f"Current speed: {airspeed:.0f}kts. "
        
        # Estimate traffic density from observation length
        if len(obs) > 3:
            traffic_count = (len(obs) - 3) // 7
            if traffic_count > 0:
                description += f"Traffic environment: {traffic_count} aircraft in sector. "
                
                conflicts = 0
                for i in range(traffic_count):
                    base_idx = 3 + i * 7
                    if base_idx + 1 < len(obs):
                        rel_x = obs[base_idx]
                        rel_y = obs[base_idx + 1]
                        distance = np.sqrt(rel_x**2 + rel_y**2)
                        if distance < 10:  # potential conflict within 10nm
                            conflicts += 1
                
                if conflicts > 0:
                    description += f"Potential conflicts with {conflicts} aircraft requiring immediate attention. "
        
        return description.strip()
    
    def _interpret_merge_obs(self, obs: np.ndarray, env_info: Dict[str, Any]) -> str:
        """Interpret merge environment observations"""
        drift = obs[0] if len(obs) > 0 else 0
        airspeed = obs[1] if len(obs) > 1 else 250
        rel_pos_x = obs[2] if len(obs) > 2 else 0
        rel_pos_y = obs[3] if len(obs) > 3 else 0
        waypoint_dist = obs[4] if len(obs) > 4 else 10
        
        description = f"Aircraft approaching merge point, "
        description += f"speed {airspeed:.0f}kts, "
        description += f"waypoint distance {waypoint_dist:.1f}nm. "
        description += f"Drift from optimal merge path: {drift:.1f}°. "
        
        # Process traffic flow information
        if len(obs) > 5:
            traffic_count = (len(obs) - 5) // 7
            if traffic_count > 0:
                description += f"Merge sequence involves {traffic_count} aircraft. "
                
                # Check for lead/trail aircraft spacing
                for i in range(min(2, traffic_count)):  # Focus on closest aircraft
                    base_idx = 5 + i * 7
                    if base_idx + 1 < len(obs):
                        rel_x = obs[base_idx]
                        rel_y = obs[base_idx + 1]
                        spacing = np.sqrt(rel_x**2 + rel_y**2)
                        if spacing < 8:  # Close spacing requiring attention
                            description += f"Aircraft {i+1} spacing: {spacing:.1f}nm. "
        
        return description.strip()


class ActionInterpreter:
    """Converts numerical actions to aviation commands"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
    
    def interpret_action(self, action: np.ndarray, obs: np.ndarray) -> str:
        """Convert action array to aviation command description"""
        if self.environment_name == "HorizontalCREnv-v0":
            return self._interpret_horizontal_action(action, obs)
        elif self.environment_name == "VerticalCREnv-v0":
            return self._interpret_vertical_action(action, obs)
        elif self.environment_name == "SectorCREnv-v0":
            return self._interpret_sector_action(action, obs)
        elif self.environment_name == "MergeEnv-v0":
            return self._interpret_merge_action(action, obs)
        else:
            return f"Unknown action: {action.tolist()}"
    
    def _interpret_horizontal_action(self, action: np.ndarray, obs: np.ndarray) -> str:
        """Interpret heading change action"""
        heading_change = action[0] if len(action) > 0 else 0
        
        if abs(heading_change) < 1:
            return "Maintain current heading"
        elif heading_change > 0:
            return f"Turn right {heading_change:.0f}°"
        else:
            return f"Turn left {abs(heading_change):.0f}°"
    
    def _interpret_vertical_action(self, action: np.ndarray, obs: np.ndarray) -> str:
        """Interpret vertical speed change action"""
        vz_change = action[0] if len(action) > 0 else 0
        
        if abs(vz_change) < 50:
            return "Maintain current vertical speed"
        elif vz_change > 0:
            return f"Increase climb rate by {vz_change:.0f} feet per minute"
        else:
            return f"Increase descent rate by {abs(vz_change):.0f} feet per minute"
    
    def _interpret_sector_action(self, action: np.ndarray, obs: np.ndarray) -> str:
        """Interpret heading and speed change actions"""
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        commands = []
        
        if abs(heading_change) >= 1:
            if heading_change > 0:
                commands.append(f"Turn right {heading_change:.0f}°")
            else:
                commands.append(f"Turn left {abs(heading_change):.0f}°")
        
        if abs(speed_change) >= 5:
            if speed_change > 0:
                commands.append(f"Increase speed by {speed_change:.0f} knots")
            else:
                commands.append(f"Reduce speed by {abs(speed_change):.0f} knots")
        
        if not commands:
            return "Maintain current heading and speed"
        
        return " and ".join(commands)
    
    def _interpret_merge_action(self, action: np.ndarray, obs: np.ndarray) -> str:
        """Interpret merge heading and speed actions"""
        return self._interpret_sector_action(action, obs)  # Same format


class ReasoningGenerator:
    """Generates reasoning explanations for expert actions"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
    
    def generate_reasoning(self, obs: np.ndarray, action: np.ndarray, 
                         reward_components: Dict[str, float]) -> str:
        """Generate reasoning explanation for the action taken"""
        if self.environment_name == "HorizontalCREnv-v0":
            return self._generate_horizontal_reasoning(obs, action, reward_components)
        elif self.environment_name == "VerticalCREnv-v0":
            return self._generate_vertical_reasoning(obs, action, reward_components)
        elif self.environment_name == "SectorCREnv-v0":
            return self._generate_sector_reasoning(obs, action, reward_components)
        elif self.environment_name == "MergeEnv-v0":
            return self._generate_merge_reasoning(obs, action, reward_components)
        else:
            return "Action taken based on expert policy"
    
    def _generate_horizontal_reasoning(self, obs: np.ndarray, action: np.ndarray, 
                                     reward_components: Dict[str, float]) -> str:
        """Generate reasoning for horizontal conflict resolution"""
        drift = obs[0] if len(obs) > 0 else 0
        heading_change = action[0] if len(action) > 0 else 0
        
        reasoning_parts = []
        
        # Analyze drift correction
        if abs(drift) > 5:
            if (drift > 0 and heading_change < 0) or (drift < 0 and heading_change > 0):
                reasoning_parts.append("Correcting drift toward waypoint")
        
        # Analyze conflict avoidance
        if len(obs) > 3:
            num_conflicts = (len(obs) - 3) // 5
            if num_conflicts > 0:
                reasoning_parts.append("Avoiding conflicts while maintaining efficient path")
        
        # Analyze reward components
        if "intrusion_penalty" in reward_components and reward_components["intrusion_penalty"] < 0:
            reasoning_parts.append("Prioritizing separation maintenance")
        
        if "drift_penalty" in reward_components and abs(reward_components["drift_penalty"]) > 0.1:
            reasoning_parts.append("Balancing separation with path efficiency")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining optimal flight path")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_vertical_reasoning(self, obs: np.ndarray, action: np.ndarray, 
                                   reward_components: Dict[str, float]) -> str:
        """Generate reasoning for vertical conflict resolution"""
        current_alt = obs[0] if len(obs) > 0 else 10000
        target_alt = obs[2] if len(obs) > 2 else 10000
        vz_change = action[0] if len(action) > 0 else 0
        
        reasoning_parts = []
        
        # Analyze altitude targeting
        alt_diff = target_alt - current_alt
        if abs(alt_diff) > 200:
            if (alt_diff > 0 and vz_change > 0) or (alt_diff < 0 and vz_change < 0):
                reasoning_parts.append("Adjusting vertical profile toward target altitude")
        
        # Analyze conflict considerations
        if len(obs) > 4:
            reasoning_parts.append("Coordinating vertical separation with traffic")
        
        # Analyze approach timing
        if "runway_distance" in str(reward_components) or any("approach" in str(k) for k in reward_components.keys()):
            reasoning_parts.append("Optimizing approach timing and profile")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining safe vertical profile")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_sector_reasoning(self, obs: np.ndarray, action: np.ndarray, 
                                 reward_components: Dict[str, float]) -> str:
        """Generate reasoning for sector conflict resolution"""
        drift = obs[0] if len(obs) > 0 else 0
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        reasoning_parts = []
        
        # Analyze sector exit efficiency
        if abs(drift) > 3:
            reasoning_parts.append("Optimizing sector exit trajectory")
        
        # Analyze traffic coordination
        if len(obs) > 3:
            traffic_count = (len(obs) - 3) // 7
            if traffic_count > 2:
                reasoning_parts.append(f"Coordinating movement with {traffic_count} aircraft in sector")
        
        # Analyze action choices
        if abs(heading_change) > abs(speed_change):
            reasoning_parts.append("Prioritizing directional changes for separation")
        elif abs(speed_change) > abs(heading_change):
            reasoning_parts.append("Using speed control for traffic flow optimization")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining safe and efficient sector transit")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_merge_reasoning(self, obs: np.ndarray, action: np.ndarray, 
                                reward_components: Dict[str, float]) -> str:
        """Generate reasoning for merge operations"""
        drift = obs[0] if len(obs) > 0 else 0
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        reasoning_parts = []
        
        # Analyze merge positioning
        if abs(heading_change) > 2:
            reasoning_parts.append("Adjusting merge approach angle")
        
        if abs(speed_change) > 5:
            reasoning_parts.append("Optimizing merge timing through speed control")
        
        # Analyze traffic flow integration
        if len(obs) > 5:
            reasoning_parts.append("Coordinating with existing traffic flow")
        
        # Analyze merge efficiency
        if "merge_reward" in reward_components and reward_components["merge_reward"] > 0:
            reasoning_parts.append("Executing efficient merge sequence")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining safe merge operations")
        
        return ". ".join(reasoning_parts) + "."


class TrainingDataGenerator:
    """Main class for generating LLM training data from SAC expert demonstrations"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.env_name = self.config["environment"]["name"]
        
        # Initialize interpreters
        self.obs_interpreter = ObservationInterpreter(self.env_name)
        self.action_interpreter = ActionInterpreter(self.env_name)
        self.reasoning_generator = ReasoningGenerator(self.env_name)
        
        # Data storage
        self.training_samples: List[TrainingSample] = []
        
        logger.info(f"Initialized data generator for {self.env_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load environment configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_sac_model(self) -> SAC:
        """Load trained SAC model"""
        model_path = self.config["sac_model"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAC model not found at {model_path}")
        
        logger.info(f"Loading SAC model from {model_path}")
        return SAC.load(model_path)
    
    def _create_environment(self) -> gym.Env:
        """Create BlueSky-Gym environment"""
        try:
            env = gym.make(self.env_name)
            logger.info(f"Created environment {self.env_name}")
            return env
        except Exception as e:
            logger.error(f"Failed to create environment {self.env_name}: {e}")
            raise
    
    def _decompose_reward(self, total_reward: float, obs: np.ndarray, 
                         action: np.ndarray) -> Dict[str, float]:
        """Decompose total reward into components (environment-specific)"""
        # This is a simplified decomposition - in practice, you'd need access
        # to the environment's internal reward calculation
        components = {
            "total_reward": total_reward
        }
        
        if self.env_name == "HorizontalCREnv-v0":
            # Estimate drift penalty from observation
            drift = obs[0] if len(obs) > 0 else 0
            components["drift_penalty"] = -abs(drift) * 0.01
            
            # Estimate intrusion penalty (simplified)
            if len(obs) > 3:
                min_distance = float('inf')
                num_conflicts = (len(obs) - 3) // 5
                for i in range(num_conflicts):
                    base_idx = 3 + i * 5
                    if base_idx + 1 < len(obs):
                        rel_x = obs[base_idx]
                        rel_y = obs[base_idx + 1]
                        distance = np.sqrt(rel_x**2 + rel_y**2)
                        min_distance = min(min_distance, distance)
                
                if min_distance < 5:  # Within minimum separation
                    components["intrusion_penalty"] = -(5 - min_distance) * 0.1
                else:
                    components["intrusion_penalty"] = 0.0
        
        elif self.env_name == "VerticalCREnv-v0":
            current_alt = obs[0] if len(obs) > 0 else 0
            target_alt = obs[2] if len(obs) > 2 else 0
            components["altitude_deviation_penalty"] = -abs(current_alt - target_alt) * 0.001
        
        # Add more environment-specific decompositions as needed
        
        return components
    
    def _calculate_safety_metrics(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """Calculate safety-related metrics"""
        metrics = {}
        
        if self.env_name == "HorizontalCREnv-v0":
            if len(obs) > 3:
                min_separation = float('inf')
                num_conflicts = (len(obs) - 3) // 5
                for i in range(num_conflicts):
                    base_idx = 3 + i * 5
                    if base_idx + 1 < len(obs):
                        rel_x = obs[base_idx]
                        rel_y = obs[base_idx + 1]
                        distance = np.sqrt(rel_x**2 + rel_y**2)
                        min_separation = min(min_separation, distance)
                
                metrics["minimum_separation_nm"] = min_separation if min_separation != float('inf') else 10.0
                metrics["separation_violation"] = 1.0 if min_separation < 5.0 else 0.0
        
        elif self.env_name == "VerticalCREnv-v0":
            if len(obs) > 4:
                current_alt = obs[0]
                min_v_separation = float('inf')
                num_conflicts = (len(obs) - 4) // 7
                for i in range(num_conflicts):
                    base_idx = 4 + i * 7
                    if base_idx < len(obs):
                        conflict_alt = obs[base_idx]
                        v_separation = abs(current_alt - conflict_alt)
                        min_v_separation = min(min_v_separation, v_separation)
                
                metrics["minimum_vertical_separation_ft"] = min_v_separation if min_v_separation != float('inf') else 2000.0
                metrics["vertical_separation_violation"] = 1.0 if min_v_separation < 1000.0 else 0.0
        
        return metrics
    
    def generate_episode_data(self, model: SAC, env: gym.Env, episode_id: str) -> List[TrainingSample]:
        """Generate training samples from a single episode"""
        samples = []
        obs, _ = env.reset()
        step_count = 0
        max_steps = self.config["data_generation"]["max_steps_per_episode"]
        
        while step_count < max_steps:
            # Get expert action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Create training sample
            scenario_description = self._create_scenario_description(obs, info)
            observation_summary = self.obs_interpreter.interpret_observation(obs, info)
            expert_action = self.action_interpreter.interpret_action(action, obs)
            reward_components = self._decompose_reward(reward, obs, action)
            reasoning = self.reasoning_generator.generate_reasoning(obs, action, reward_components)
            safety_metrics = self._calculate_safety_metrics(obs, action)
            
            sample = TrainingSample(
                environment=self.env_name,
                scenario_id=f"{episode_id}_step_{step_count}",
                scenario_description=scenario_description,
                observation_summary=observation_summary,
                expert_action=expert_action,
                reasoning=reasoning,
                reward_components=reward_components,
                safety_metrics=safety_metrics,
                step_number=step_count,
                episode_id=episode_id
            )
            
            samples.append(sample)
            
            obs = next_obs
            step_count += 1
            
            if terminated or truncated:
                break
        
        return samples
    
    def _create_scenario_description(self, obs: np.ndarray, info: Dict[str, Any]) -> str:
        """Create high-level scenario description"""
        descriptions = {
            "HorizontalCREnv-v0": "Horizontal conflict resolution scenario with aircraft navigating to waypoint while avoiding conflicting traffic",
            "VerticalCREnv-v0": "Vertical conflict resolution scenario with aircraft managing altitude changes during approach",
            "SectorCREnv-v0": "Sector management scenario with multiple aircraft requiring coordinated traffic flow control",
            "MergeEnv-v0": "Traffic merge scenario with aircraft integrating into existing traffic flow pattern"
        }
        
        base_description = descriptions.get(self.env_name, "Air traffic control scenario")
        
        # Add specific details based on observation
        if len(obs) > 3:
            traffic_count = self._estimate_traffic_count(obs)
            if traffic_count > 0:
                base_description += f" involving {traffic_count} aircraft"
        
        return base_description
    
    def _estimate_traffic_count(self, obs: np.ndarray) -> int:
        """Estimate number of aircraft from observation length"""
        if self.env_name == "HorizontalCREnv-v0":
            return (len(obs) - 3) // 5 if len(obs) > 3 else 0
        elif self.env_name == "VerticalCREnv-v0":
            return (len(obs) - 4) // 7 if len(obs) > 4 else 0
        elif self.env_name in ["SectorCREnv-v0", "MergeEnv-v0"]:
            base_obs = 3 if self.env_name == "SectorCREnv-v0" else 5
            return (len(obs) - base_obs) // 7 if len(obs) > base_obs else 0
        return 0
    
    def generate_training_data(self) -> None:
        """Generate complete training dataset"""
        logger.info(f"Starting training data generation for {self.env_name}")
        
        # Load model and create environment
        model = self._load_sac_model()
        env = self._create_environment()
        
        num_episodes = self.config["data_generation"]["num_episodes"]
        target_samples = self.config["data_generation"]["samples_per_environment"]
        expert_threshold = self.config["data_generation"]["expert_threshold"]
        
        episode_rewards = []
        
        # Generate episodes
        for episode in tqdm(range(num_episodes), desc=f"Generating {self.env_name} episodes"):
            episode_id = f"{self.env_name}_episode_{episode:04d}"
            
            try:
                episode_samples = self.generate_episode_data(model, env, episode_id)
                
                # Calculate episode performance
                episode_reward = sum(s.reward_components["total_reward"] for s in episode_samples)
                episode_rewards.append(episode_reward)
                
                # Only include high-performing episodes
                if len(episode_rewards) < 10:  # Include first 10 episodes regardless
                    self.training_samples.extend(episode_samples)
                else:
                    # Calculate performance threshold
                    reward_threshold = np.percentile(episode_rewards, expert_threshold * 100)
                    if episode_reward >= reward_threshold:
                        self.training_samples.extend(episode_samples)
                        logger.debug(f"Episode {episode}: reward {episode_reward:.3f} (included)")
                    else:
                        logger.debug(f"Episode {episode}: reward {episode_reward:.3f} (excluded)")
                
                # Check if we have enough samples
                if len(self.training_samples) >= target_samples:
                    logger.info(f"Reached target of {target_samples} samples")
                    break
                    
            except Exception as e:
                logger.error(f"Error in episode {episode}: {e}")
                continue
        
        env.close()
        
        # Shuffle and limit samples
        np.random.shuffle(self.training_samples)
        self.training_samples = self.training_samples[:target_samples]
        
        logger.info(f"Generated {len(self.training_samples)} training samples for {self.env_name}")
    
    def save_training_data(self, output_path: str) -> None:
        """Save training data to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = [asdict(sample) for sample in self.training_samples]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} training samples to {output_file}")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the training data"""
        if not self.training_samples:
            return {}
        
        stats = {
            "environment": self.env_name,
            "total_samples": len(self.training_samples),
            "unique_episodes": len(set(s.episode_id for s in self.training_samples)),
            "average_episode_length": len(self.training_samples) / len(set(s.episode_id for s in self.training_samples)),
        }
        
        # Reward statistics
        rewards = [s.reward_components["total_reward"] for s in self.training_samples]
        stats["reward_stats"] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards)
        }
        
        # Safety statistics
        if self.training_samples[0].safety_metrics:
            safety_violations = sum(1 for s in self.training_samples 
                                  if any(v > 0 for k, v in s.safety_metrics.items() 
                                        if "violation" in k))
            stats["safety_violation_rate"] = safety_violations / len(self.training_samples)
        
        return stats


def main():
    """Main function to generate training data for all environments"""
    environments = [
        ("horizontal_config.yaml", "../training_data/horizontal_cr_samples.json"),
        ("vertical_config.yaml", "../training_data/vertical_cr_samples.json"),
        ("sector_config.yaml", "../training_data/sector_cr_samples.json"),
        ("merge_config.yaml", "../training_data/merge_samples.json")
    ]
    
    base_path = Path(__file__).parent.parent
    
    all_stats = {}
    
    for config_file, output_file in environments:
        config_path = base_path / "configs" / config_file
        output_path = base_path / output_file
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            continue
        
        try:
            logger.info(f"Processing {config_file}")
            generator = TrainingDataGenerator(config_path)
            generator.generate_training_data()
            generator.save_training_data(output_path)
            
            stats = generator.generate_statistics()
            all_stats[stats["environment"]] = stats
            
            logger.info(f"Completed {config_file}: {stats['total_samples']} samples generated")
            
        except Exception as e:
            logger.error(f"Failed to process {config_file}: {e}")
            continue
    
    # Save overall statistics
    stats_file = base_path / "training_data" / "generation_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"Training data generation completed. Statistics saved to {stats_file}")


if __name__ == "__main__":
    main()
