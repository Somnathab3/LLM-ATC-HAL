"""Horizontal conflict resolution environment template"""

from typing import Dict, Any, Union, List
import numpy as np
from .base_template import BaseEnvironmentTemplate


class HorizontalTemplate(BaseEnvironmentTemplate):
    """Template for HorizontalCREnv-v0 environment"""
    
    def interpret_observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                            env_info: Dict[str, Any], step_count: int = 0, sample_id: str = "") -> str:
        """Convert horizontal conflict observation to natural language"""
        descriptions: List[str] = []
        speed_conflicts = 0
        
        # Extract state variables
        if isinstance(obs, dict):
            # Dictionary observation format
            drift_angle = 0.0
            if "cos_drift" in obs and "sin_drift" in obs:
                cos_drift = obs["cos_drift"][0] if len(obs["cos_drift"]) > 0 else 0
                sin_drift = obs["sin_drift"][0] if len(obs["sin_drift"]) > 0 else 0
                drift_angle = np.degrees(np.arctan2(sin_drift, cos_drift))
            elif "drift" in obs:
                drift_angle = obs["drift"][0] if len(obs["drift"]) > 0 else 0
                
            waypoint_dist = obs.get("waypoint_distance", [10.0])[0] if "waypoint_distance" in obs else 10.0
            heading = obs.get("heading", [0.0])[0] if "heading" in obs else 0.0
            airspeed = obs.get("airspeed", [250.0])[0] if "airspeed" in obs else 250.0
            
            # Handle intruders
            if "intruder_distance" in obs and len(obs["intruder_distance"]) > 0:
                intruder_distances = obs["intruder_distance"]
                valid_distances = intruder_distances[intruder_distances > 0]
                if len(valid_distances) > 0:
                    min_distance = np.min(valid_distances)
                    num_intruders = len(valid_distances)
                    position_term = self.get_variant("position_terms")
                    descriptions.append(f"Nearest of {num_intruders} intruders at {min_distance:.1f} NM {position_term}")
        else:
            # Array observation format
            drift_angle = obs[0] if len(obs) > 0 else 0
            waypoint_dist = obs[1] if len(obs) > 1 else 10
            heading = obs[2] if len(obs) > 2 else 0
            airspeed = 250.0  # Default
            
            # Count and analyze intruders
            num_intruders = (len(obs) - 3) // 5
            min_distance = float('inf')
            
            for i in range(num_intruders):
                base_idx = 3 + i * 5
                if base_idx + 4 < len(obs):
                    rel_x = obs[base_idx]
                    rel_y = obs[base_idx + 1]
                    distance = np.sqrt(rel_x**2 + rel_y**2)
                    min_distance = min(min_distance, distance)
                    
                    rel_speed = obs[base_idx + 4] if base_idx + 4 < len(obs) else 0
                    if abs(rel_speed) > 20:
                        speed_conflicts += 1
            
            if min_distance != float('inf'):
                position_term = self.get_variant("position_terms")
                descriptions.append(f"Nearest of {num_intruders} intruders at {min_distance:.1f} NM {position_term}")
        
        # Navigation status
        nav_verb = self.get_variant("navigation_verbs")
        distance_term = "approximately" if waypoint_dist > 5 else "just"
        descriptions.append(f"{nav_verb} waypoint at {distance_term} {waypoint_dist:.1f} NM")
        
        # Course deviation analysis
        if abs(drift_angle) > 5:
            descriptions.append(f"course deviation {drift_angle:.1f}° from optimal track")
        
        # Speed conflict analysis
        if speed_conflicts > 0:
            descriptions.append(f"{speed_conflicts} aircraft with significant speed convergence requiring attention")
        
        # Flight parameters
        descriptions.append(f"aircraft heading {heading:.0f}° at {airspeed:.0f} kts")
        descriptions.append(f"drift from optimal path {drift_angle:.1f}°")
        
        descriptions.append(f"step {step_count} evaluation complete")
        
        # Add contextual safety note
        if not descriptions or len(descriptions) < 3:
            descriptions.append("aircraft maintaining stable flight parameters with continuous traffic monitoring")
        
        return "; ".join(descriptions) + "."
    
    def create_scenario_description(self, obs: Union[Dict[str, np.ndarray], np.ndarray], 
                                  info: Dict[str, Any], timestep: int = 0, episode_id: str = "") -> str:
        """Create horizontal conflict scenario description"""
        nav_verb = self.get_variant("navigation_verbs") 
        conflict_term = self.get_variant("conflict_terms")
        urgency_level = self.get_variant("urgency_levels")
        
        # Extract key state values
        if isinstance(obs, dict):
            drift = obs.get("cos_drift", [0])[0] if "cos_drift" in obs else 0
            waypoint_dist = obs.get("waypoint_distance", [10])[0] if "waypoint_distance" in obs else 10
            num_intruders = len(obs.get("intruder_distance", [])) if "intruder_distance" in obs else 0
        else:
            drift = obs[0] if len(obs) > 0 else 0
            waypoint_dist = obs[1] if len(obs) > 1 else 10
            num_intruders = self.estimate_traffic_count(obs)
        
        return (f"Step {timestep}: Horizontal conflict resolution with {drift:.1f}° drift, "
               f"waypoint {waypoint_dist:.1f} NM away, {num_intruders} {conflict_term} present while "
               f"{nav_verb} to destination requiring {urgency_level}.")
    
    def estimate_traffic_count(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> int:
        """Estimate traffic count for horizontal environment"""
        if isinstance(obs, dict):
            return super().estimate_traffic_count(obs)
        else:
            return (len(obs) - 3) // 5 if len(obs) > 3 else 0
