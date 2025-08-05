"""Base template class for environment-specific LLM training data generation"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import numpy as np


class BaseEnvironmentTemplate(ABC):
    """Abstract base class for environment-specific templates"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.variant_counter = 0
        
        # Common variant pools for diversity
        self.description_variants = {
            "navigation_verbs": ["navigating toward", "proceeding to", "heading for", "approaching", "tracking to", "routing to", "advancing toward"],
            "conflict_terms": ["intruders", "conflicting aircraft", "traffic conflicts", "potential collisions", "separation issues"],
            "urgency_levels": ["immediate attention", "careful coordination", "tactical maneuvering", "precise control", "swift response"],
            "safety_terms": ["maintaining separation", "ensuring clearance", "avoiding conflicts", "preserving safety margins", "monitoring closely"],
            "position_terms": ["ahead", "behind", "to the left", "to the right", "above", "below", "converging"],
            "distance_terms": ["distance", "range", "separation", "spacing", "proximity", "interval"],
            "speed_terms": ["airspeed", "ground speed", "velocity", "speed", "rate"]
        }
    
    @abstractmethod
    def interpret_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                            env_info: Dict[str, Any], step_count: int = 0, sample_id: str = "") -> str:
        """Convert numerical observation to natural language description"""
        pass
    
    @abstractmethod
    def create_scenario_description(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                                  info: Dict[str, Any], timestep: int = 0, episode_id: str = "") -> str:
        """Create step-specific scenario description with embedded state information"""
        pass
    
    def get_variant(self, variant_type: str) -> str:
        """Get cycling variant for text diversity"""
        if variant_type not in self.description_variants:
            return ""
        variants = self.description_variants[variant_type]
        variant = variants[self.variant_counter % len(variants)]
        self.variant_counter += 1
        return variant
    
    def estimate_traffic_count(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> int:
        """Estimate number of aircraft from observation"""
        if isinstance(obs, dict):
            if "intruder_distance" in obs:
                return len(obs["intruder_distance"])
            else:
                return len([k for k in obs.keys() if "traffic" in k.lower() or "intruder" in k.lower()])
        else:
            # Environment-specific logic will be implemented in subclasses
            return 0
