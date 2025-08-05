"""Merge environment template"""

from typing import Dict, Any, Union, List
import numpy as np
from .base_template import BaseEnvironmentTemplate


class MergeTemplate(BaseEnvironmentTemplate):
    """Template for MergeEnv-v0 environment"""
    
    def interpret_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                            env_info: Dict[str, Any], step_count: int = 0, sample_id: str = "") -> str:
        """Convert merge environment observation to natural language"""
        descriptions: List[str] = []
        
        # Handle both dict and array observations
        if isinstance(obs, dict):
            drift = obs.get("drift", [0])[0] if "drift" in obs else 0
            airspeed = obs.get("airspeed", [250])[0] if "airspeed" in obs else 250
            waypoint_dist = obs.get("waypoint_distance", [10])[0] if "waypoint_distance" in obs else 10
        else:
            drift = obs[0] if len(obs) > 0 else 0
            airspeed = obs[1] if len(obs) > 1 else 250
            waypoint_dist = obs[4] if len(obs) > 4 else 10
        
        nav_verb = self.get_variant("navigation_verbs")
        distance_term = self.get_variant("distance_terms")
        urgency_level = self.get_variant("urgency_levels")
        
        descriptions.append(f"aircraft {nav_verb} merge point at {distance_term} {waypoint_dist:.1f} NM")
        descriptions.append(f"speed {airspeed:.0f} kts with drift {drift:.1f}° from optimal merge path")
        
        # Process traffic flow information
        traffic_count = self.estimate_traffic_count(obs)
        if traffic_count > 0:
            descriptions.append(f"sequencing with {traffic_count} aircraft in arrival flow requiring {urgency_level}")
        
        descriptions.append(f"step {step_count} merge sequencing complete")
        
        if not descriptions:
            descriptions.append("aircraft maintaining stable approach profile with continuous arrival coordination")
        
        return "; ".join(descriptions) + "."
    
    def create_scenario_description(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                                  info: Dict[str, Any], timestep: int = 0, episode_id: str = "") -> str:
        """Create merge scenario description"""
        nav_verb = self.get_variant("navigation_verbs")
        urgency_level = self.get_variant("urgency_levels")
        
        # Extract merge-specific state
        if isinstance(obs, dict):
            drift = obs.get("drift", [0])[0] if "drift" in obs else 0
            waypoint_dist = obs.get("waypoint_distance", [10])[0] if "waypoint_distance" in obs else 10
            airspeed = obs.get("airspeed", [250])[0] if "airspeed" in obs else 250
        else:
            drift = obs[0] if len(obs) > 0 else 0
            waypoint_dist = obs[4] if len(obs) > 4 else 10
            airspeed = obs[1] if len(obs) > 1 else 250
            
        traffic_count = self.estimate_traffic_count(obs)
        return (f"Step {timestep}: Traffic merge sequencing with {drift:.1f}° drift, "
               f"{nav_verb} merge point {waypoint_dist:.1f} NM away at {airspeed:.0f} kt, "
               f"integrating {traffic_count} aircraft in flow requiring {urgency_level}.")
    
    def estimate_traffic_count(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> int:
        """Estimate traffic count for merge environment"""
        if isinstance(obs, dict):
            return super().estimate_traffic_count(obs)
        else:
            return (len(obs) - 5) // 7 if len(obs) > 5 else 0
