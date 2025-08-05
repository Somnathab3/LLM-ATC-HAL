"""Vertical conflict resolution environment template"""

from typing import Dict, Any, Union, List
import numpy as np
from .base_template import BaseEnvironmentTemplate


class VerticalTemplate(BaseEnvironmentTemplate):
    """Template for VerticalCREnv-v0 environment"""
    
    def interpret_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                            env_info: Dict[str, Any], step_count: int = 0, sample_id: str = "") -> str:
        """Convert vertical conflict observation to natural language"""
        descriptions: List[str] = []
        
        # Handle both dict and array observations
        if isinstance(obs, dict):
            current_alt = env_info.get('own_altitude', 30000) if env_info else 30000
            target_alt = env_info.get('target_altitude', 25000) if env_info else 25000
            runway_dist = env_info.get('runway_distance', 50) if env_info else 50
            vz = obs.get("vertical_speed", [0])[0] if "vertical_speed" in obs else 0
        else:
            current_alt = obs[0] if len(obs) > 0 else 30000
            vz = obs[1] if len(obs) > 1 else 0
            target_alt = obs[2] if len(obs) > 2 else 25000
            runway_dist = obs[3] if len(obs) > 3 else 50
        
        # Build altitude description
        if vz > 0:
            alt_status = f"climbing at {abs(vz):.0f} ft/min"
        elif vz < 0:
            alt_status = f"descending at {abs(vz):.0f} ft/min"
        else:
            alt_status = "maintaining level flight"
        
        urgency_level = self.get_variant("urgency_levels")
        
        descriptions.append(f"aircraft at {current_alt:.0f} ft {alt_status} toward target {target_alt:.0f} ft")
        descriptions.append(f"runway {runway_dist:.1f} NM away with {urgency_level}")
        
        # Calculate altitude deviation
        alt_dev = abs(current_alt - target_alt)
        if alt_dev > 500:
            descriptions.append(f"altitude deviation {alt_dev:.0f} ft requiring correction")
        
        descriptions.append(f"step {step_count} vertical guidance complete")
        
        if not descriptions:
            descriptions.append("aircraft maintaining stable vertical profile with continuous altitude monitoring")
        
        return "; ".join(descriptions) + "."
    
    def create_scenario_description(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                                  info: Dict[str, Any], timestep: int = 0, episode_id: str = "") -> str:
        """Create vertical conflict scenario description"""
        conflict_term = self.get_variant("conflict_terms")
        urgency_level = self.get_variant("urgency_levels")
        
        # Extract altitude-specific state
        if isinstance(obs, dict):
            current_alt = info.get('own_altitude', 30000) if info else 30000
            target_alt = info.get('target_altitude', 25000) if info else 25000
            runway_dist = info.get('runway_distance', 50) if info else 50
        else:
            current_alt = obs[0] if len(obs) > 0 else 30000
            target_alt = obs[2] if len(obs) > 2 else 25000
            runway_dist = obs[3] if len(obs) > 3 else 50
        
        alt_diff = abs(current_alt - target_alt)
        return (f"Step {timestep}: Vertical conflict resolution at {current_alt:.0f} ft "
               f"targeting {target_alt:.0f} ft ({alt_diff:.0f} ft deviation), "
               f"runway {runway_dist:.1f} NM away with {conflict_term} requiring {urgency_level}.")
    
    def estimate_traffic_count(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> int:
        """Estimate traffic count for vertical environment"""
        if isinstance(obs, dict):
            return super().estimate_traffic_count(obs)
        else:
            return (len(obs) - 4) // 7 if len(obs) > 4 else 0
