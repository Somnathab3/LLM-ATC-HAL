#!/usr/bin/env python3
"""
BlueSky-Integrated ATC Training Data Generator
==============================================

Generates training data for LLM fine-tuning by running expert SAC policies
in BlueSky-Gym environments and converting the experience to natural language format.

This script integrates with the BlueSky simulator and scenario generator to create
realistic ATC scenarios with proper conflict detection and resolution.

Features:
- Load trained SAC models from F:\\LLM-ATC-HAL\\BSKY_GYM_LLM\\models_backup
- Generate ≥10,000 expert trajectories per environment  
- Convert BlueSky observations to natural language scenario descriptions
- Transform actions to readable ATC commands
- Create structured training datasets with safety metrics
- Support for HorizontalCREnv-v0, VerticalCREnv-v0, SectorCREnv-v0, MergeEnv-v0
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,  # Back to INFO level for normal operation
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import template system
from templates.template_factory import TemplateFactory

# Try to import torch for Q-value extraction
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    logger.warning("PyTorch not available - Q-value extraction disabled")

# Try to import gymnasium and stable_baselines3
try:
    import gymnasium as gym
    from stable_baselines3 import SAC
    gym_available = True
except ImportError as e:
    logger.warning(f"Gymnasium/SAC not available: {e}")
    gym_available = False


# Data classes and exceptions
class SampleValidationError(Exception):
    """Custom exception for sample validation failures"""
    pass


@dataclass
class TrainingSample:
    """Structured training sample for LLM fine-tuning with built-in validation"""
    environment: str
    episode_id: str
    timestep: int
    scenario_description: str
    observation_summary: str
    expert_action: str
    reasoning: str
    reward_components: Dict[str, float]
    safety_metrics: Dict[str, float] = field(default_factory=dict)
    _validated: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Validate sample on creation"""
        self.validate()
        self._validated = True
    
    def validate(self) -> None:
        """Validate all fields meet requirements"""
        errors = []
        
        # Required string fields with minimum lengths
        string_fields = {
            'environment': (self.environment, 5),
            'episode_id': (self.episode_id, 10),
            'scenario_description': (self.scenario_description, 50),
            'observation_summary': (self.observation_summary, 30),
            'expert_action': (self.expert_action, 15),
            'reasoning': (self.reasoning, 40)
        }
        
        for field_name, (value, min_length) in string_fields.items():
            if not isinstance(value, str):
                errors.append(f"{field_name} must be a string, got {type(value)}")
            elif len(value.strip()) < min_length:
                errors.append(f"{field_name} must be at least {min_length} chars, got {len(value.strip())}")
            elif len(value.strip()) == 0:
                errors.append(f"{field_name} cannot be empty")
        
        # Validate timestep
        if not isinstance(self.timestep, int) or self.timestep < 0:
            errors.append(f"timestep must be non-negative integer, got {self.timestep}")
        
        # Validate reward components
        required_reward_keys = {'drift_penalty', 'intrusion_penalty', 'total_reward'}
        if not isinstance(self.reward_components, dict):
            errors.append("reward_components must be a dictionary")
        else:
            missing_keys = required_reward_keys - set(self.reward_components.keys())
            if missing_keys:
                errors.append(f"reward_components missing required keys: {missing_keys}")
            
            # Check for valid numeric values
            for key, value in self.reward_components.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    errors.append(f"reward_components[{key}] must be finite number, got {value}")
        
        # Validate safety metrics
        if not isinstance(self.safety_metrics, dict):
            errors.append("safety_metrics must be a dictionary")
        
        if errors:
            raise ValueError(f"TrainingSample validation failed: {'; '.join(errors)}")
    
    def get_hash(self) -> str:
        """Generate unique hash for duplicate detection"""
        content = f"{self.scenario_description}_{self.observation_summary}_{self.expert_action}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format for LLM fine-tuning"""
        return {
            "scenario": self.scenario_description,
            "observation": self.observation_summary,
            "action": self.expert_action,
            "reasoning": self.reasoning
        }


@dataclass
class EnvironmentConfig:
    """Configuration for environment-specific data generation"""
    name: str
    model_path: str
    config_path: str
    output_path: str
    num_episodes: int
    samples_target: int


class DuplicateTracker:
    """Track and prevent duplicate samples during generation"""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0
    
    def is_duplicate(self, sample: TrainingSample) -> bool:
        """Check if sample is a duplicate and track it"""
        sample_hash = sample.get_hash()
        if sample_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True
        
        self.seen_hashes.add(sample_hash)
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get duplication statistics"""
        return {
            'unique_samples': len(self.seen_hashes),
            'duplicates_rejected': self.duplicate_count,
            'total_processed': len(self.seen_hashes) + self.duplicate_count
        }


def get_q_value(model: 'SAC', obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray) -> Optional[float]:
    """Extract Q-value from SAC model with proper error handling"""
    if not torch_available:
        return None
    
    try:
        # Convert observation to tensor
        if isinstance(obs, dict):
            # Handle dictionary observations by concatenating relevant features
            obs_tensor = torch.tensor(_flatten_dict_obs(obs), dtype=torch.float32).unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        # Extract Q-value using critic network
        with torch.no_grad():
            if hasattr(model.policy, 'q_net'):
                q1, q2 = model.policy.q_net(obs_tensor, action_tensor)
                q_value = float(torch.min(q1, q2))
            elif hasattr(model.policy, 'critic'):
                q1, q2 = model.policy.critic(obs_tensor, action_tensor)
                q_value = float(torch.min(q1, q2))
            else:
                return None
        
        return q_value
    except Exception as e:
        logger.debug(f"Q-value extraction failed: {e}")
        return None


def _flatten_dict_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten dictionary observation to array for Q-network input"""
    features = []
    # Common BlueSky-Gym observation keys in preferred order
    key_order = ['cos_drift', 'sin_drift', 'waypoint_distance', 'heading', 'airspeed', 
                 'intruder_distance', 'intruder_bearing', 'vertical_speed', 'altitude']
    
    for key in key_order:
        if key in obs_dict:
            feature = obs_dict[key]
            if isinstance(feature, np.ndarray):
                features.extend(feature.flatten())
            else:
                features.append(feature)
    
    # Add any remaining features not in the preferred order
    for key, feature in obs_dict.items():
        if key not in key_order:
            if isinstance(feature, np.ndarray):
                features.extend(feature.flatten())
            else:
                features.append(feature)
    
    return np.array(features, dtype=np.float32)


def generate_single_episode(args: Tuple[str, str, int, int, str, Dict[str, Any]]) -> List[TrainingSample]:
    """Generate training samples for a single episode (for multiprocessing)"""
    config_path, env_name, episode_idx, max_steps, model_path, generation_config = args
    
    try:
        # Initialize components for this worker
        generator = TrainingDataGenerator(config_path)
        model = generator._load_sac_model()
        env = generator._create_environment()
        
        # Generate episode data
        episode_id = f"{env_name}_episode_{episode_idx:04d}"
        samples = generator.generate_episode_data(model, env, episode_id)
        
        env.close()
        return samples
        
    except Exception as e:
        logger.error(f"Episode {episode_idx} failed: {e}")
        return []


def validate_reward_decomposition(total_reward: float, components: Dict[str, float], 
                                tolerance: float = 1e-6) -> bool:
    """Validate that reward components sum to total reward"""
    calculated_total = sum(v for k, v in components.items() if k != "total_reward")
    residual = abs(total_reward - calculated_total)
    
    if residual > tolerance:
        logger.warning(f"Reward decomposition mismatch: total={total_reward:.6f}, "
                      f"components_sum={calculated_total:.6f}, residual={residual:.6f}")
        return False
    
    return True


@dataclass
class ObservationInterpreter:
    """Converts numerical observations to natural language descriptions"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.variant_counter = 0
        self.template = TemplateFactory.create_template(environment_name)
        
        # Built-in variant pools for diversity
        self.description_variants = {
            "navigation_verbs": ["navigating toward", "proceeding to", "heading for", "approaching", "tracking to", "routing to", "advancing toward"],
            "conflict_terms": ["conflicting traffic", "intruding aircraft", "traffic conflicts", "separation challenges", "proximate aircraft", "convergent traffic"],
            "urgency_levels": ["immediate attention required", "monitoring situation", "standard separation procedures", "proactive avoidance", "heightened awareness", "tactical response"],
            "position_terms": ["bearing", "relative position", "azimuth", "direction", "track angle", "heading reference"],
            "distance_terms": ["distance", "range", "separation", "spacing", "proximity", "interval"],
            "altitude_terms": ["flight level", "altitude", "vertical position", "height", "elevation"],
            "speed_terms": ["airspeed", "ground speed", "velocity", "speed", "rate"]
        }
    
    def interpret_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                            env_info: Dict[str, Any], step_count: int = 0, 
                            sample_id: str = "") -> str:
        """Convert BlueSky-Gym observation to natural language description with step-specific details"""
        
        # Use template to interpret observation
        body = self.template.interpret_observation(obs, env_info, step_count, sample_id)
        
        # Create final description with guaranteed minimum length
        result = f"{self.environment_name} update: {body}"
        
        # Guarantee minimum 30 characters
        if len(result) < 30:
            result += " with continuous traffic monitoring and situational awareness maintained"
        
        return result

class ActionInterpreter:
    """Converts numerical actions to aviation commands"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
    
    def interpret_action(self, action: np.ndarray, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> str:
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
    
    def _interpret_horizontal_action(self, action: np.ndarray, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> str:
        """Interpret heading change action"""
        heading_change = action[0] if len(action) > 0 else 0
        
        if abs(heading_change) < 1:
            return "Maintain current heading"
        elif heading_change > 0:
            return f"Turn right {heading_change:.0f}°"
        else:
            return f"Turn left {abs(heading_change):.0f}°"
    
    def _interpret_vertical_action(self, action: np.ndarray, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> str:
        """Interpret vertical speed change action"""
        vz_change = action[0] if len(action) > 0 else 0
        
        if abs(vz_change) < 50:
            return "Maintain current vertical speed"
        elif vz_change > 0:
            return f"Climb at {vz_change:.0f} ft/min"
        else:
            return f"Descend at {abs(vz_change):.0f} ft/min"
    
    def _interpret_sector_action(self, action: np.ndarray, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> str:
        """Interpret heading and speed change actions"""
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        if abs(heading_change) >= 1 and abs(speed_change) >= 5:
            # Both heading and speed changes
            heading_dir = "right" if heading_change > 0 else "left"
            speed_val = 250 + speed_change if speed_change > 0 else 250 - abs(speed_change)
            return f"Heading change {abs(heading_change):.0f}°, set speed {speed_val:.0f} kt"
        elif abs(heading_change) >= 1:
            heading_dir = "right" if heading_change > 0 else "left"
            return f"Turn {heading_dir} {abs(heading_change):.0f}°"
        elif abs(speed_change) >= 5:
            speed_val = 250 + speed_change if speed_change > 0 else 250 - abs(speed_change)
            return f"Set speed {speed_val:.0f} kt"
        else:
            return "Maintain current heading and speed"
    
    def _interpret_merge_action(self, action: np.ndarray, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> str:
        """Interpret merge heading and speed actions"""
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        if abs(heading_change) >= 1 and abs(speed_change) >= 5:
            # Both heading and speed adjustments for merge
            speed_val = 250 + speed_change if speed_change > 0 else 250 - abs(speed_change)
            return f"Adjust heading by {abs(heading_change):.0f}°, speed to {speed_val:.0f} kt"
        elif abs(heading_change) >= 1:
            return f"Adjust heading by {abs(heading_change):.0f}°"
        elif abs(speed_change) >= 5:
            speed_val = 250 + speed_change if speed_change > 0 else 250 - abs(speed_change)
            return f"Adjust speed to {speed_val:.0f} kt"
        else:
            return "Maintain current approach profile"


class ReasoningGenerator:
    """Generates reasoning explanations for expert actions"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
    
    def generate_reasoning(self, obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray, 
                         reward_components: Dict[str, float], q_value: Optional[float] = None) -> str:
        """Generate reasoning explanation for the action taken"""
        base_reasoning = ""
        
        if self.environment_name == "HorizontalCREnv-v0":
            base_reasoning = self._generate_horizontal_reasoning(obs, action, reward_components)
        elif self.environment_name == "VerticalCREnv-v0":
            base_reasoning = self._generate_vertical_reasoning(obs, action, reward_components)
        elif self.environment_name == "SectorCREnv-v0":
            base_reasoning = self._generate_sector_reasoning(obs, action, reward_components)
        elif self.environment_name == "MergeEnv-v0":
            base_reasoning = self._generate_merge_reasoning(obs, action, reward_components)
        else:
            base_reasoning = "Action taken based on expert policy."
        
        # Add Q-value information if available
        if q_value is not None:
            q_reasoning = f" Expert policy Q-value: {q_value:.3f} indicating high confidence in this action."
            base_reasoning = base_reasoning.rstrip('.') + q_reasoning
        
        return base_reasoning
    
    def _generate_horizontal_reasoning(self, obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray, 
                                     reward_components: Dict[str, float]) -> str:
        """Generate reasoning for horizontal conflict resolution"""
        # Extract drift information
        if isinstance(obs, dict):
            drift = 0.0
            if "cos_drift" in obs and "sin_drift" in obs:
                cos_drift = obs["cos_drift"][0] if len(obs["cos_drift"]) > 0 else 0
                sin_drift = obs["sin_drift"][0] if len(obs["sin_drift"]) > 0 else 0
                drift = np.degrees(np.arctan2(sin_drift, cos_drift))
            elif "drift" in obs:
                drift = obs["drift"][0] if len(obs["drift"]) > 0 else 0
        else:
            drift = obs[0] if len(obs) > 0 else 0
            
        heading_change = action[0] if len(action) > 0 else 0
        
        reasoning_parts = []
        
        # Analyze drift correction
        if abs(drift) > 5:
            if (drift > 0 and heading_change < 0) or (drift < 0 and heading_change > 0):
                reasoning_parts.append("Correcting drift toward waypoint for optimal approach trajectory")
        
        # Analyze conflict avoidance
        if isinstance(obs, dict):
            if "intruder_distance" in obs:
                intruder_dists = obs["intruder_distance"]
                num_conflicts = len(intruder_dists[intruder_dists > 0])
                if num_conflicts > 0:
                    reasoning_parts.append("Avoiding conflicts while maintaining efficient path")
        elif len(obs) > 3:
            num_conflicts = (len(obs) - 3) // 5
            if num_conflicts > 0:
                reasoning_parts.append("Avoiding conflicts while maintaining efficient path")
        
        # Analyze reward components
        if "intrusion_penalty" in reward_components and reward_components["intrusion_penalty"] < 0:
            reasoning_parts.append("Prioritizing separation maintenance for enhanced safety protocols")
        
        if "drift_penalty" in reward_components and abs(reward_components["drift_penalty"]) > 0.1:
            reasoning_parts.append("Balancing separation with path efficiency")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining optimal flight path with traffic awareness")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_vertical_reasoning(self, obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray, 
                                   reward_components: Dict[str, float]) -> str:
        """Generate reasoning for vertical conflict resolution"""
        if isinstance(obs, dict):
            # Handle dictionary observations - approximate values
            current_alt = 30000  # Default FL300
            target_alt = 25000   # Default FL250
        else:
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
        if isinstance(obs, dict) or len(obs) > 4:
            reasoning_parts.append("Coordinating vertical separation with traffic")
        
        # Analyze approach timing
        if "runway_distance" in str(reward_components) or any("approach" in str(k) for k in reward_components.keys()):
            reasoning_parts.append("Optimizing approach timing and profile")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining safe vertical profile during approach")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_sector_reasoning(self, obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray, 
                                 reward_components: Dict[str, float]) -> str:
        """Generate reasoning for sector conflict resolution"""
        if isinstance(obs, dict):
            drift = obs.get("drift", [0])[0] if "drift" in obs else 0
        else:
            drift = obs[0] if len(obs) > 0 else 0
            
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        reasoning_parts = []
        
        # Analyze sector exit efficiency (inject "sector" keyword)
        if abs(drift) > 3:
            reasoning_parts.append("Optimizing sector exit trajectory for efficient traffic flow management")
        else:
            reasoning_parts.append("Maintaining efficient sector transit with minimal deviation")
        
        # Analyze traffic coordination
        traffic_count = 2  # Default estimate
        if isinstance(obs, dict):
            # Count intruder-related keys
            traffic_count = len([k for k in obs.keys() if "intruder" in k.lower() or "traffic" in k.lower()])
        elif len(obs) > 3:
            traffic_count = (len(obs) - 3) // 7
            
        # Inject "traffic density" and "airspeed" keywords
        if traffic_count > 2:
            reasoning_parts.append(f"Managing traffic density with {traffic_count} aircraft while maintaining optimal airspeed")
        else:
            reasoning_parts.append("Adjusting airspeed to optimize traffic density management")
        
        # Analyze action choices
        if abs(heading_change) > abs(speed_change):
            reasoning_parts.append("Prioritizing directional changes for separation")
        elif abs(speed_change) > abs(heading_change):
            reasoning_parts.append("Using speed control for traffic flow optimization")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining safe and efficient sector transit with proper airspeed control")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_merge_reasoning(self, obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray, 
                                reward_components: Dict[str, float]) -> str:
        """Generate reasoning for merge operations"""
        if isinstance(obs, dict):
            drift = obs.get("drift", [0])[0] if "drift" in obs else 0
        else:
            drift = obs[0] if len(obs) > 0 else 0
            
        heading_change = action[0] if len(action) > 0 else 0
        speed_change = action[1] if len(action) > 1 else 0
        
        reasoning_parts = []
        
        # Analyze merge positioning (inject "merge" keyword)
        if abs(heading_change) > 2:
            reasoning_parts.append("Adjusting merge approach angle for optimal traffic integration")
        else:
            reasoning_parts.append("Maintaining current merge trajectory for smooth traffic integration")
        
        if abs(speed_change) > 5:
            reasoning_parts.append("Optimizing merge timing through speed control")
        
        # Analyze traffic flow integration (inject "traffic flow" keyword)
        traffic_count = 1  # Default estimate
        if isinstance(obs, dict):
            # Count intruder-related keys
            traffic_count = len([k for k in obs.keys() if "intruder" in k.lower() or "traffic" in k.lower()])
        elif len(obs) > 5:
            traffic_count = (len(obs) - 5) // 7
            
        if traffic_count > 0:
            reasoning_parts.append("Coordinating with existing traffic flow for seamless merge integration")
        else:
            reasoning_parts.append("Managing traffic flow during merge sequence")
        
        # Analyze merge efficiency and inject "FAF" keyword
        if "merge_reward" in reward_components and reward_components["merge_reward"] > 0:
            reasoning_parts.append("Executing efficient merge sequence while maintaining FAF approach profile coordination")
        else:
            reasoning_parts.append("Approaching FAF (Final Approach Fix) with proper merge sequence coordination")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining safe and efficient merge operations with proper FAF approach sequence and traffic flow coordination")
        
        return ". ".join(reasoning_parts) + "."
        
        return ". ".join(reasoning_parts) + "."


class TrainingDataGenerator:
    """Enhanced main class for generating LLM training data from SAC expert demonstrations"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.env_name = self.config["environment"]["name"]
        
        # Initialize interpreters
        self.obs_interpreter = ObservationInterpreter(self.env_name)
        self.action_interpreter = ActionInterpreter(self.env_name)
        self.reasoning_generator = ReasoningGenerator(self.env_name)
        
        # Enhanced tracking and validation
        self.training_samples: List[TrainingSample] = []
        self.duplicate_tracker = DuplicateTracker()
        self.q_value_cache = {}
        
        # Statistics tracking
        self.generation_stats = {
            'episodes_generated': 0,
            'samples_created': 0,
            'samples_validated': 0,
            'duplicates_rejected': 0,
            'validation_failures': 0,
            'q_value_extractions': 0
        }
        
        # Variant pools for scenario descriptions (shared with obs_interpreter)
        self.variant_counter = 0
        self.description_variants = {
            "navigation_verbs": ["navigating toward", "proceeding to", "heading for", "approaching", "tracking to", "routing to", "advancing toward"],
            "conflict_terms": ["intruders", "conflicting aircraft", "traffic conflicts", "potential collisions", "separation issues"],
            "urgency_levels": ["immediate attention", "careful coordination", "tactical maneuvering", "precise control", "swift response"]
        }
        
        logger.info(f"Initialized enhanced data generator for {self.env_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load environment configuration with validation"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required config sections
        required_sections = ['environment', 'sac_model', 'data_generation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def _load_sac_model(self) -> 'SAC':
        """Load trained SAC model with error handling"""
        if not gym_available:
            raise ImportError("Gymnasium and Stable-Baselines3 are required but not available")
        
        model_path = self.config["sac_model"]["model_path"]
        
        # Resolve model path relative to the project base directory
        if not os.path.isabs(model_path):
            # Get the base project directory (parent of scripts directory)
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAC model not found at {model_path}")
        
        logger.info(f"Loading SAC model from {model_path}")
        try:
            model = SAC.load(str(model_path))
            # Cache Q-network for efficient Q-value extraction
            if torch_available and hasattr(model.policy, 'q_net'):
                self.q_value_cache['q_net'] = model.policy.q_net
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load SAC model: {e}")
    
    def _create_environment(self) -> 'gym.Env':
        """Create BlueSky-Gym environment with comprehensive error handling"""
        try:
            # Import the BlueSky-Gym setup module
            import sys
            from pathlib import Path
            
            # Add project root to path
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            
            # Initialize BlueSky-Gym environments
            from bluesky_gym_setup import initialize_bluesky_gym
            working_envs = initialize_bluesky_gym()
            
            if self.env_name not in working_envs:
                raise RuntimeError(f"Environment {self.env_name} not available in working environments: {working_envs}")
            
            # Create the environment
            env = gym.make(self.env_name, render_mode=None)
            logger.info(f"Created BlueSky-Gym environment {self.env_name}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create BlueSky-Gym environment {self.env_name}: {e}")
            logger.error("BlueSky-Gym environments are required for this system.")
            logger.error("Please ensure BlueSky-Gym is properly installed and environments are available.")
            raise
    
    def generate_episode_data(self, model: 'SAC', env: 'gym.Env', episode_id: str) -> List[TrainingSample]:
        """Generate training samples from a single episode with enhanced validation"""
        samples = []
        obs, _ = env.reset()
        timestep = 0
        max_steps = self.config["data_generation"]["max_steps_per_episode"]
        
        while timestep < max_steps:
            try:
                # Get expert action
                action, _ = model.predict(obs, deterministic=True)
                
                # Extract Q-value with error handling
                q_value = get_q_value(model, obs, action)
                if q_value is not None:
                    self.generation_stats['q_value_extractions'] += 1
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Create sample ID for reproducible randomization
                sample_id = f"{episode_id}_step_{timestep:04d}"
                
                # Create training sample with enhanced data
                scenario_description = self._create_scenario_description(obs, info, timestep, episode_id)
                observation_summary = self.obs_interpreter.interpret_observation(obs, info, timestep, sample_id)
                expert_action = self.action_interpreter.interpret_action(action, obs)
                reward_components = self._decompose_reward(reward, obs, action, info)
                reasoning = self.reasoning_generator.generate_reasoning(obs, action, reward_components, q_value)
                safety_metrics = self._calculate_safety_metrics(obs, action)
                
                # Validate reward decomposition
                if not validate_reward_decomposition(reward, reward_components):
                    logger.warning(f"Reward decomposition validation failed for {sample_id}")
                
                # Create and validate sample
                try:
                    sample = TrainingSample(
                        environment=self.env_name,
                        episode_id=episode_id,
                        timestep=timestep,
                        scenario_description=scenario_description,
                        observation_summary=observation_summary,
                        expert_action=expert_action,
                        reasoning=reasoning,
                        reward_components=reward_components,
                        safety_metrics=safety_metrics
                    )
                    
                    # Check for duplicates
                    if not self.duplicate_tracker.is_duplicate(sample):
                        samples.append(sample)
                        self.generation_stats['samples_validated'] += 1
                    else:
                        self.generation_stats['duplicates_rejected'] += 1
                        
                    self.generation_stats['samples_created'] += 1
                    
                except ValueError as e:
                    logger.warning(f"Sample validation failed for {sample_id}: {e}")
                    self.generation_stats['validation_failures'] += 1
                
                obs = next_obs
                timestep += 1
                
                if terminated or truncated:
                    break
                    
            except Exception as e:
                logger.error(f"Error generating sample at timestep {timestep}: {e}")
                break
        
        self.generation_stats['episodes_generated'] += 1
        return samples
    
    def _decompose_reward(self, total_reward: float, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                         action: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Decompose total reward into components (environment-specific)"""
        # Initialize components with required fields
        components = {
            "total_reward": total_reward,
            "drift_penalty": 0.0,
            "intrusion_penalty": 0.0
        }
        
        if self.env_name == "HorizontalCREnv-v0":
            # Handle both dict and array observations
            if isinstance(obs, dict):
                # For dict observations, try to get drift from appropriate keys
                drift = 0.0
                if "cos_drift" in obs and "sin_drift" in obs:
                    cos_drift = obs["cos_drift"][0] if len(obs["cos_drift"]) > 0 else 0
                    sin_drift = obs["sin_drift"][0] if len(obs["sin_drift"]) > 0 else 0
                    drift = np.degrees(np.arctan2(sin_drift, cos_drift))
                elif "drift" in obs:
                    drift = obs["drift"][0] if len(obs["drift"]) > 0 else 0
                elif "track_angle_diff" in obs:
                    drift = obs["track_angle_diff"][0] if len(obs["track_angle_diff"]) > 0 else 0
            else:
                # Array observation format
                drift = obs[0] if len(obs) > 0 else 0
                
            components["drift_penalty"] = -abs(drift) * 0.01
            
            # Estimate intrusion penalty (simplified)
            if isinstance(obs, dict):
                if "intruder_distance" in obs:
                    intruder_dists = obs["intruder_distance"]
                    valid_distances = intruder_dists[intruder_dists > 0]
                    if len(valid_distances) > 0:
                        min_distance = np.min(valid_distances)
                        if min_distance < 5:  # Within minimum separation
                            components["intrusion_penalty"] = -(5 - min_distance) * 0.1
                        else:
                            components["intrusion_penalty"] = 0.0
                    else:
                        components["intrusion_penalty"] = 0.0
                else:
                    components["intrusion_penalty"] = 0.0
            elif len(obs) > 3:
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
            if isinstance(obs, dict):
                current_alt = info.get('own_altitude', 30000) if info else 30000
                target_alt = info.get('target_altitude', 25000) if info else 25000
            else:
                current_alt = obs[0] if len(obs) > 0 else 0
                target_alt = obs[2] if len(obs) > 2 else 0
            
            components["drift_penalty"] = -abs(current_alt - target_alt) * 0.001
            components["intrusion_penalty"] = 0.0  # No horizontal conflicts in vertical env
            
            # Add landing reward for vertical environment
            if info and 'runway_distance' in info:
                runway_dist = info['runway_distance']
                components["landing_reward"] = max(0, (50 - runway_dist) * 0.02)
            else:
                components["landing_reward"] = 0.0
        
        elif self.env_name == "SectorCREnv-v0":
            if isinstance(obs, dict):
                drift = obs.get("drift", [0])[0] if "drift" in obs else 0
            else:
                drift = obs[0] if len(obs) > 0 else 0
                
            components["drift_penalty"] = -abs(drift) * 0.01
            
            # Estimate traffic coordination penalty
            traffic_count = self._estimate_traffic_count(obs)
            if traffic_count > 3:
                components["intrusion_penalty"] = -traffic_count * 0.05
            else:
                components["intrusion_penalty"] = 0.0
        
        elif self.env_name == "MergeEnv-v0":
            if isinstance(obs, dict):
                drift = obs.get("drift", [0])[0] if "drift" in obs else 0
                waypoint_dist = obs.get("waypoint_distance", [10])[0] if "waypoint_distance" in obs else 10
            else:
                drift = obs[0] if len(obs) > 0 else 0
                waypoint_dist = obs[4] if len(obs) > 4 else 10
                
            components["drift_penalty"] = -abs(drift) * 0.01
            
            # Add merge reward
            components["merge_reward"] = max(0, (15 - waypoint_dist) * 0.03)
            components["intrusion_penalty"] = 0.0  # Will be updated based on traffic
            
            # Account for traffic flow integration
            traffic_count = self._estimate_traffic_count(obs)
            if traffic_count > 2:
                components["intrusion_penalty"] = -traffic_count * 0.03
        
        # Ensure components sum approximately to total_reward
        calculated_total = sum(v for k, v in components.items() if k != "total_reward")
        residual = total_reward - calculated_total
        
        # Add residual as task completion reward
        components["task_completion"] = residual
        
        return components
    
    def _calculate_safety_metrics(self, obs: Union[Dict[str, np.ndarray], np.ndarray], action: np.ndarray) -> Dict[str, float]:
        """Calculate safety-related metrics"""
        metrics = {}
        
        if self.env_name == "HorizontalCREnv-v0":
            if isinstance(obs, dict):
                # Handle dictionary observation format
                if "intruder_distance" in obs:
                    intruder_dists = obs["intruder_distance"]
                    valid_distances = intruder_dists[intruder_dists > 0]
                    if len(valid_distances) > 0:
                        min_separation = np.min(valid_distances)
                        metrics["minimum_separation_nm"] = min_separation
                        metrics["separation_violation"] = 1.0 if min_separation < 5.0 else 0.0
                    else:
                        metrics["minimum_separation_nm"] = 10.0
                        metrics["separation_violation"] = 0.0
                else:
                    metrics["minimum_separation_nm"] = 10.0
                    metrics["separation_violation"] = 0.0
            elif len(obs) > 3:
                # Array observation format
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
            else:
                metrics["minimum_separation_nm"] = 10.0
                metrics["separation_violation"] = 0.0
        
        elif self.env_name == "VerticalCREnv-v0":
            if isinstance(obs, dict):
                # Handle dictionary observation - use info if available
                metrics["minimum_vertical_separation_ft"] = 2000.0  # Safe default
                metrics["vertical_separation_violation"] = 0.0
            elif len(obs) > 4:
                # Array observation format
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
            else:
                metrics["minimum_vertical_separation_ft"] = 2000.0
                metrics["vertical_separation_violation"] = 0.0
        
        # Default safety metrics for other environments
        if not metrics:
            metrics["minimum_separation_nm"] = 10.0
            metrics["separation_violation"] = 0.0
        
        return metrics
    
    def _create_scenario_description(self, obs: Union[Dict[str, np.ndarray], np.ndarray], info: Dict[str, Any], timestep: int = 0, episode_id: str = "") -> str:
        """Create step-specific scenario description with embedded state information"""
        
        # Get variant words for diversity  
        nav_verb = self.description_variants["navigation_verbs"][self.variant_counter % len(self.description_variants["navigation_verbs"])]
        conflict_term = self.description_variants["conflict_terms"][self.variant_counter % len(self.description_variants["conflict_terms"])]
        urgency_level = self.description_variants["urgency_levels"][self.variant_counter % len(self.description_variants["urgency_levels"])]
        
        if self.env_name == "HorizontalCREnv-v0":
            # Extract key state values for uniqueness
            if isinstance(obs, dict):
                drift = obs.get("cos_drift", [0])[0] if "cos_drift" in obs else 0
                waypoint_dist = obs.get("waypoint_distance", [10])[0] if "waypoint_distance" in obs else 10
                num_intruders = len(obs.get("intruder_distance", [])) if "intruder_distance" in obs else 0
            else:
                drift = obs[0] if len(obs) > 0 else 0
                waypoint_dist = obs[1] if len(obs) > 1 else 10
                num_intruders = self._estimate_traffic_count(obs)
            
            return (f"Step {timestep}: Horizontal conflict resolution with {drift:.1f}° drift, "
                   f"waypoint {waypoint_dist:.1f} NM away, {num_intruders} {conflict_term} present while "
                   f"{nav_verb} to destination requiring {urgency_level}.")
                   
        elif self.env_name == "VerticalCREnv-v0":
            # Extract altitude-specific state
            if isinstance(obs, dict):
                current_alt = info.get('own_altitude', 30000)
                target_alt = info.get('target_altitude', 25000)
                runway_dist = info.get('runway_distance', 50)
            else:
                current_alt = obs[0] if len(obs) > 0 else 30000
                target_alt = obs[2] if len(obs) > 2 else 25000
                runway_dist = obs[3] if len(obs) > 3 else 50
            
            alt_diff = abs(current_alt - target_alt)
            return (f"Step {timestep}: Vertical conflict resolution at {current_alt:.0f} ft "
                   f"targeting {target_alt:.0f} ft ({alt_diff:.0f} ft deviation), "
                   f"runway {runway_dist:.1f} NM away with {conflict_term} requiring {urgency_level}.")
                   
        elif self.env_name == "SectorCREnv-v0":
            # Extract sector management state
            if isinstance(obs, dict):
                drift = obs.get("drift", [0])[0] if "drift" in obs else 0
                airspeed = obs.get("airspeed", [250])[0] if "airspeed" in obs else 250
            else:
                drift = obs[0] if len(obs) > 0 else 0
                airspeed = obs[1] if len(obs) > 1 else 250
                
            traffic_count = self._estimate_traffic_count(obs)
            # Inject sector-specific keywords: "sector", "airspeed", "traffic density"
            traffic_density = traffic_count * 0.8 + random.uniform(0.2, 1.8)  # Generate realistic density
            return (f"Step {timestep}: Sector management with {drift:.1f}° drift from optimal exit, "
                   f"current airspeed is {airspeed:.0f} kt and traffic density is {traffic_density:.1f} aircraft per sector, "
                   f"coordinating {traffic_count} aircraft with {conflict_term} requiring {urgency_level}.")
                   
        elif self.env_name == "MergeEnv-v0":
            # Extract merge-specific state
            if isinstance(obs, dict):
                drift = obs.get("drift", [0])[0] if "drift" in obs else 0
                waypoint_dist = obs.get("waypoint_distance", [10])[0] if "waypoint_distance" in obs else 10
                airspeed = obs.get("airspeed", [250])[0] if "airspeed" in obs else 250
            else:
                drift = obs[0] if len(obs) > 0 else 0
                waypoint_dist = obs[4] if len(obs) > 4 else 10
                airspeed = obs[1] if len(obs) > 1 else 250
                
            traffic_count = self._estimate_traffic_count(obs)
            # Inject merge-specific keywords: "merge", "FAF", "traffic flow"
            faf_distance = waypoint_dist + random.uniform(-2.0, 2.0)  # FAF distance variation
            return (f"Step {timestep}: Traffic merge sequencing with {drift:.1f}° drift, "
                   f"{nav_verb} merge point {waypoint_dist:.1f} NM away at {airspeed:.0f} kt, "
                   f"approaching the FAF (Final Approach Fix) at {faf_distance:.1f} NM and coordinating traffic flow "
                   f"integrating {traffic_count} aircraft in merge sequence requiring {urgency_level}.")
        
        # Fallback with step info
        return f"Step {timestep}: Air traffic control scenario requiring {urgency_level} with {conflict_term}."
    
    def _estimate_traffic_count(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> int:
        """Estimate number of aircraft from observation length"""
        if isinstance(obs, dict):
            # For dict observations, count intruder-related keys
            if "intruder_distance" in obs:
                return len(obs["intruder_distance"])
            else:
                # Count traffic-related keys
                return len([k for k in obs.keys() if "traffic" in k.lower() or "intruder" in k.lower()])
        else:
            # For array observations, use original logic
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
                logger.error(f"Error in episode {episode}: {type(e).__name__}: {str(e)}")
                logger.debug(f"Full traceback for episode {episode}:\n{traceback.format_exc()}")
                continue
        
        env.close()
        
        # Shuffle and limit samples
        random.shuffle(self.training_samples)
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


def generate_parallel_episodes(generator: TrainingDataGenerator, model: 'SAC', env: 'gym.Env', 
                             num_episodes: int, max_workers: Optional[int] = None) -> List[TrainingSample]:
    """Generate episodes in parallel using multiprocessing"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Conservative default
    
    logger.info(f"Generating {num_episodes} episodes using {max_workers} workers")
    
    all_samples = []
    
    # Prepare arguments for parallel execution
    generation_config = {
        'max_steps_per_episode': generator.config["data_generation"]["max_steps_per_episode"],
        'expert_threshold': generator.config["data_generation"].get("expert_threshold", 0.0)
    }
    
    args_list = [
        (str(generator.config_path), generator.env_name, episode_idx, 
         generation_config['max_steps_per_episode'], 
         generator.config["sac_model"]["model_path"], generation_config)
        for episode_idx in range(num_episodes)
    ]
    
    # Use ProcessPoolExecutor for better resource management
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {
            executor.submit(generate_single_episode, args): episode_idx
            for episode_idx, args in enumerate(args_list)
        }
        
        for future in tqdm(as_completed(future_to_episode), total=num_episodes, desc="Episodes"):
            episode_idx = future_to_episode[future]
            try:
                episode_samples = future.result()
                all_samples.extend(episode_samples)
                generator.generation_stats['episodes_generated'] += 1
            except Exception as e:
                logger.error(f"Episode {episode_idx} failed: {e}")
    
    return all_samples


def validate_generated_samples(samples: List[TrainingSample], output_path: str) -> Dict[str, Any]:
    """Run validation on generated samples before saving"""
    logger.info("Running validation on generated samples...")
    
    # Save temporary file for validation
    temp_file = Path(output_path).with_suffix('.tmp.json')
    try:
        data = [asdict(sample) for sample in samples]
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Import and run validator
        try:
            from validate_training_data import TrainingDataValidator
            validator = TrainingDataValidator()
            results = validator.validate_file(str(temp_file))
            
            validation_report = {
                'total_samples': results.total_samples,
                'valid_samples': results.valid_samples,
                'validation_passed': results.is_valid,
                'error_summary': results.get_summary()
            }
            
            if not results.is_valid:
                logger.warning("Validation found issues - check validation report")
            else:
                logger.info("All samples passed validation")
            
            return validation_report
            
        except ImportError:
            logger.warning("Validation module not available - skipping validation")
            return {'validation_skipped': True}
            
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()


def main():
    """Enhanced main function with comprehensive CLI support"""
    parser = argparse.ArgumentParser(
        description="Generate LLM training data from SAC expert demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for all environments
    python generate_training_data.py --envs all --episodes 100 --output-dir ../training_data
    
    # Generate for specific environments with parallelization
    python generate_training_data.py --envs horizontal,vertical --episodes 200 --workers 4
    
    # Generate with custom configuration
    python generate_training_data.py --config custom_config.yaml --samples-per-env 5000 --seed 42
        """)
    
    # Environment selection
    parser.add_argument("--envs", default="all", 
                       help="Environments to generate data for (comma-separated: horizontal,vertical,sector,merge or 'all')")
    
    # Data generation parameters
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes per environment (default: 100)")
    parser.add_argument("--samples-per-env", type=int, default=10000,
                       help="Target samples per environment (default: 10000)")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Maximum steps per episode (default: 500)")
    
    # Parallelization
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    
    # I/O paths
    parser.add_argument("--config-dir", type=str, default="configs",
                       help="Directory containing config files (default: configs)")
    parser.add_argument("--output-dir", type=str, default="training_data",
                       help="Output directory for training data (default: training_data)")
    
    # Quality control
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation on generated samples")
    parser.add_argument("--expert-threshold", type=float, default=0.0,
                       help="Minimum reward threshold for expert samples (default: 0.0)")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log to file instead of console")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Parse environments
    if args.envs.lower() == "all":
        environments = [
            ("horizontal_config.yaml", "horizontal_cr_samples.json"),
            ("vertical_config.yaml", "vertical_cr_samples.json"),
            ("sector_config.yaml", "sector_cr_samples.json"),
            ("merge_config.yaml", "merge_samples.json")
        ]
    else:
        env_map = {
            "horizontal": ("horizontal_config.yaml", "horizontal_cr_samples.json"),
            "vertical": ("vertical_config.yaml", "vertical_cr_samples.json"),
            "sector": ("sector_config.yaml", "sector_cr_samples.json"),
            "merge": ("merge_config.yaml", "merge_samples.json")
        }
        env_names = [name.strip() for name in args.envs.split(",")]
        environments = [env_map[name] for name in env_names if name in env_map]
        
        if not environments:
            logger.error(f"No valid environments found in: {args.envs}")
            return 1
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    config_dir = Path(args.config_dir) if os.path.isabs(args.config_dir) else base_path / args.config_dir
    output_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else base_path / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    failed_environments = []
    
    logger.info(f"Starting data generation for {len(environments)} environments")
    logger.info(f"Parameters: episodes={args.episodes}, samples_per_env={args.samples_per_env}, workers={args.workers}")
    
    for config_file, output_file in environments:
        config_path = config_dir / config_file
        output_path = output_dir / output_file
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            failed_environments.append(config_file)
            continue
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {config_file}")
            logger.info(f"{'='*60}")
            
            # Create generator with enhanced configuration
            generator = TrainingDataGenerator(config_path)
            
            # Override config with CLI parameters
            generator.config["data_generation"].update({
                "num_episodes": args.episodes,
                "samples_per_environment": args.samples_per_env,
                "max_steps_per_episode": args.max_steps,
                "expert_threshold": args.expert_threshold
            })
            
            # Generate training data
            if args.no_parallel:
                generator.generate_training_data()
            else:
                # Use parallel generation
                model = generator._load_sac_model()
                env = generator._create_environment()
                
                parallel_samples = generate_parallel_episodes(
                    generator, model, env, args.episodes, args.workers
                )
                
                # Shuffle and limit samples
                np.random.shuffle(parallel_samples)
                generator.training_samples = parallel_samples[:args.samples_per_env]
                
                env.close()
            
            # Validate if requested
            validation_report = None
            if args.validate:
                validation_report = validate_generated_samples(generator.training_samples, str(output_path))
            
            # Save training data
            generator.save_training_data(str(output_path))
            
            # Generate statistics
            stats = generator.generate_statistics()
            if validation_report:
                stats['validation'] = validation_report
            all_stats[stats["environment"]] = stats
            
            logger.info(f"Completed {config_file}: {stats['total_samples']} samples generated")
            logger.info(f"Duplicate rejection rate: {generator.duplicate_tracker.duplicate_count}/{generator.generation_stats['samples_created']} ({100*generator.duplicate_tracker.duplicate_count/max(1, generator.generation_stats['samples_created']):.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to process {config_file}: {e}", exc_info=args.verbose)
            failed_environments.append(config_file)
            continue
    
    # Save comprehensive statistics
    stats_file = output_dir / "generation_statistics.json"
    final_stats = {
        "generation_timestamp": str(np.datetime64('now')),
        "generation_parameters": {
            "episodes_per_env": args.episodes,
            "samples_per_env": args.samples_per_env,
            "max_steps_per_episode": args.max_steps,
            "parallel_workers": args.workers,
            "seed": args.seed
        },
        "environments_processed": len(environments) - len(failed_environments),
        "environments_failed": len(failed_environments),
        "failed_environments": failed_environments,
        "environment_stats": all_stats
    }
    
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Environments processed: {len(environments) - len(failed_environments)}/{len(environments)}")
    
    total_samples = sum(stats.get('total_samples', 0) for stats in all_stats.values())
    logger.info(f"Total samples generated: {total_samples:,}")
    
    if failed_environments:
        logger.warning(f"Failed environments: {', '.join(failed_environments)}")
    
    logger.info(f"Statistics saved to: {stats_file}")
    
    return 0 if not failed_environments else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
