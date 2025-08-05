"""Sector conflict resolution environment template"""

from typing import Dict, Any, Union, List
import numpy as np
from .base_template import BaseEnvironmentTemplate


class SectorTemplate(BaseEnvironmentTemplate):
    """Template for SectorCREnv-v0 environment"""
    
    def interpret_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                            env_info: Dict[str, Any], step_count: int = 0, sample_id: str = "") -> str:
        """Convert sector conflict observation to natural language"""
        descriptions: List[str] = []
        
        # Handle both dict and array observations
        if isinstance(obs, dict):
            drift = obs.get("drift", [0])[0] if "drift" in obs else 0
            airspeed = obs.get("airspeed", [250])[0] if "airspeed" in obs else 250
        else:
            drift = obs[0] if len(obs) > 0 else 0
            airspeed = obs[1] if len(obs) > 1 else 250
        
        conflict_term = self.get_variant("conflict_terms")
        urgency_level = self.get_variant("urgency_levels")
        nav_verb = self.get_variant("navigation_verbs")
        speed_term = self.get_variant("speed_terms")
        
        descriptions.append(f"aircraft {nav_verb} sector exit point with {drift:.1f}° drift from optimal track")
        descriptions.append(f"maintaining {airspeed:.0f} kts through controlled airspace")
        
        # Estimate traffic coordination requirements
        traffic_count = self.estimate_traffic_count(obs)
        if traffic_count > 0:
            descriptions.append(f"coordinating with {traffic_count} aircraft in sector requiring {urgency_level}")
        
        descriptions.append(f"step {step_count} sector management complete")
        
        if not descriptions:
            descriptions.append("aircraft maintaining sector transit profile with continuous coordination")
        
        return "; ".join(descriptions) + "."
    
    def create_scenario_description(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                                  info: Dict[str, Any], timestep: int = 0, episode_id: str = "") -> str:
        """Create sector conflict scenario description"""
        nav_verb = self.get_variant("navigation_verbs")
        conflict_term = self.get_variant("conflict_terms")
        urgency_level = self.get_variant("urgency_levels")
        
        # Extract sector management state
        if isinstance(obs, dict):
            drift = obs.get("drift", [0])[0] if "drift" in obs else 0
            airspeed = obs.get("airspeed", [250])[0] if "airspeed" in obs else 250
        else:
            drift = obs[0] if len(obs) > 0 else 0
            airspeed = obs[1] if len(obs) > 1 else 250
            
        traffic_count = self.estimate_traffic_count(obs)
        return (f"Step {timestep}: Sector management with {drift:.1f}° drift from optimal exit, "
               f"speed {airspeed:.0f} kt, coordinating {traffic_count} aircraft with "
               f"{conflict_term} requiring {urgency_level}.")
    
    def estimate_traffic_count(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> int:
        """Estimate traffic count for sector environment"""
        if isinstance(obs, dict):
            return super().estimate_traffic_count(obs)
        else:
            return (len(obs) - 3) // 7 if len(obs) > 3 else 0
