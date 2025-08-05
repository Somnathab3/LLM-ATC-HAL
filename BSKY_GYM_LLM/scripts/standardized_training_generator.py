#!/usr/bin/env python3
"""
Fine-tuning Training Data Generator with Standardized Prompts
===========================================================
Generates training data using the same LLMPromptEngine templates
as the production system for consistent model comparison.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_atc.tools.standardized_prompt_manager import StandardizedPromptManager
from llm_atc.tools.llm_prompt_engine import LLMPromptEngine

logger = logging.getLogger(__name__)


class StandardizedTrainingDataGenerator:
    """
    Generates training data using standardized LLMPromptEngine templates.
    Ensures fine-tuned models learn the same prompt format as production.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the standardized training data generator.
        
        Args:
            config_path: Path to ATC configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.prompt_manager = StandardizedPromptManager(config_path)
        
        # Get standardized engine for training data generation
        # Use base model for template generation (prompts will be consistent)
        self.engine = self.prompt_manager.get_standardized_engine("llama3.1:8b")
        
        self.logger.info("✅ Initialized standardized training data generator")
        self.logger.info(f"   Separation standards: {self.engine.min_horizontal_separation_nm}NM / {self.engine.min_vertical_separation_ft}ft")
        self.logger.info(f"   Optimized prompts: {self.engine.enable_optimized_prompts}")
    
    def generate_conflict_detection_training_example(self, aircraft_states: List[Dict[str, Any]], 
                                                   ground_truth_conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a conflict detection training example using standardized prompts.
        
        Args:
            aircraft_states: List of aircraft state dictionaries
            ground_truth_conflicts: Ground truth conflict information
            
        Returns:
            Training example dictionary
        """
        try:
            # Generate prompt using standardized engine
            if self.engine.enable_optimized_prompts:
                system_prompt, user_prompt = self.engine.format_conflict_detection_prompt_optimized(aircraft_states)
                input_text = f"System: {system_prompt}\n\nUser: {user_prompt}"
            else:
                input_text = self.engine.format_detector_prompt(aircraft_states)
            
            # Generate expected output format
            has_conflicts = len(ground_truth_conflicts) > 0
            aircraft_pairs = []
            time_to_conflict = []
            
            for conflict in ground_truth_conflicts:
                ac1_id = conflict.get("aircraft_1_id", "AC1")
                ac2_id = conflict.get("aircraft_2_id", "AC2")
                aircraft_pairs.append(f"{ac1_id}-{ac2_id}")
                time_to_conflict.append(conflict.get("time_to_conflict", 120.0))
            
            # Create response in the exact format expected by the engine
            expected_response = {
                "conflict_detected": has_conflicts,
                "aircraft_pairs": aircraft_pairs,
                "time_to_conflict": time_to_conflict,
                "confidence": 0.85 if has_conflicts else 0.95,
                "priority": "high" if has_conflicts else "low",
                "analysis": self._generate_analysis_text(aircraft_states, ground_truth_conflicts),
                "calculation_details": self._generate_calculation_details(aircraft_states, ground_truth_conflicts)
            }
            
            return {
                "instruction": "Analyze the aircraft positions and detect any conflicts that violate separation standards.",
                "input": input_text,
                "output": json.dumps(expected_response, indent=2),
                "metadata": {
                    "type": "conflict_detection",
                    "aircraft_count": len(aircraft_states),
                    "conflicts_detected": len(ground_truth_conflicts),
                    "uses_standardized_prompts": True,
                    "separation_standards": {
                        "horizontal_nm": self.engine.min_horizontal_separation_nm,
                        "vertical_ft": self.engine.min_vertical_separation_ft
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating conflict detection example: {e}")
            return None
    
    def generate_conflict_resolution_training_example(self, conflict_info: Dict[str, Any],
                                                    resolution_command: str,
                                                    rationale: str) -> Dict[str, Any]:
        """
        Generate a conflict resolution training example using standardized prompts.
        
        Args:
            conflict_info: Conflict scenario information
            resolution_command: Expected resolution command
            rationale: Rationale for the resolution
            
        Returns:
            Training example dictionary
        """
        try:
            # Generate prompt using standardized engine
            if self.engine.enable_optimized_prompts:
                system_prompt, user_prompt = self.engine.format_conflict_resolution_prompt_optimized(conflict_info)
                input_text = f"System: {system_prompt}\n\nUser: {user_prompt}"
            else:
                input_text = self.engine.format_conflict_prompt(conflict_info)
            
            # Generate expected output in the exact format expected by the engine
            if self.engine.enable_optimized_prompts:
                expected_response = f"""COMMAND: {resolution_command}
RATIONALE: {rationale}
CONFIDENCE: 0.87"""
            else:
                # Extract aircraft ID from command
                import re
                aircraft_match = re.search(r'\b([A-Z0-9-]+)\b', resolution_command)
                aircraft_id = aircraft_match.group(1) if aircraft_match else "UNKNOWN"
                
                # Determine maneuver type
                if resolution_command.startswith("HDG"):
                    maneuver = "heading_change"
                elif resolution_command.startswith("ALT"):
                    maneuver = "altitude_change"
                elif resolution_command.startswith("SPD"):
                    maneuver = "speed_change"
                else:
                    maneuver = "heading_change"
                
                expected_response = f"""COMMAND: {resolution_command}
AIRCRAFT: {aircraft_id}
MANEUVER: {maneuver}
RATIONALE: {rationale}
CONFIDENCE: 0.87"""
            
            return {
                "instruction": "Analyze the conflict scenario and provide an appropriate resolution command.",
                "input": input_text,
                "output": expected_response,
                "metadata": {
                    "type": "conflict_resolution",
                    "command": resolution_command,
                    "uses_standardized_prompts": True,
                    "optimized_format": self.engine.enable_optimized_prompts,
                    "separation_standards": {
                        "horizontal_nm": self.engine.min_horizontal_separation_nm,
                        "vertical_ft": self.engine.min_vertical_separation_ft
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating conflict resolution example: {e}")
            return None
    
    def _generate_analysis_text(self, aircraft_states: List[Dict[str, Any]], 
                              conflicts: List[Dict[str, Any]]) -> str:
        """Generate analysis text for conflict detection."""
        if not conflicts:
            return "No conflicts detected. All aircraft maintain adequate separation."
        
        h_sep = self.engine.min_horizontal_separation_nm
        v_sep = self.engine.min_vertical_separation_ft
        
        analysis_parts = []
        for i, conflict in enumerate(conflicts):
            ac1_id = conflict.get("aircraft_1_id", f"AC{i*2+1}")
            ac2_id = conflict.get("aircraft_2_id", f"AC{i*2+2}")
            distance = conflict.get("horizontal_distance", 4.2)
            altitude_diff = conflict.get("vertical_separation", 500)
            
            analysis_parts.append(
                f"Conflict detected between {ac1_id} and {ac2_id}: "
                f"{distance:.1f} NM horizontal (< {h_sep} NM), "
                f"{altitude_diff} ft vertical (< {v_sep} ft)."
            )
        
        return " ".join(analysis_parts)
    
    def _generate_calculation_details(self, aircraft_states: List[Dict[str, Any]], 
                                    conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate calculation details for conflict detection."""
        if not conflicts:
            return {
                "current_horizontal_nm": [],
                "current_vertical_ft": [],
                "meets_separation_standards": True
            }
        
        horizontal_distances = []
        vertical_separations = []
        
        for conflict in conflicts:
            horizontal_distances.append(conflict.get("horizontal_distance", 4.2))
            vertical_separations.append(conflict.get("vertical_separation", 500))
        
        return {
            "current_horizontal_nm": horizontal_distances,
            "current_vertical_ft": vertical_separations,
            "meets_separation_standards": False
        }
    
    def convert_legacy_training_data(self, legacy_file: str, output_file: str) -> int:
        """
        Convert legacy training data to use standardized prompts.
        
        Args:
            legacy_file: Path to legacy training data file
            output_file: Path to output standardized training data
            
        Returns:
            Number of examples converted
        """
        converted_count = 0
        
        try:
            with open(legacy_file, 'r') as f_in, open(output_file, 'w') as f_out:
                for line_num, line in enumerate(f_in):
                    try:
                        legacy_example = json.loads(line.strip())
                        
                        # Convert to standardized format
                        standardized_example = self._convert_legacy_example(legacy_example)
                        
                        if standardized_example:
                            f_out.write(json.dumps(standardized_example) + '\n')
                            converted_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to convert line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error converting training data: {e}")
            return 0
        
        self.logger.info(f"✅ Converted {converted_count} training examples to standardized format")
        return converted_count
    
    def _convert_legacy_example(self, legacy_example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a single legacy example to standardized format."""
        try:
            # Extract information from legacy format
            input_text = legacy_example.get("input", "")
            output_text = legacy_example.get("output", "")
            
            # Parse aircraft information from input (simplified parsing)
            aircraft_states = self._parse_aircraft_from_legacy_input(input_text)
            
            if "conflict" in output_text.lower() or "turn" in output_text.lower():
                # This is likely a resolution example
                return self._convert_legacy_resolution_example(legacy_example, aircraft_states)
            else:
                # This is likely a detection example
                return self._convert_legacy_detection_example(legacy_example, aircraft_states)
                
        except Exception as e:
            self.logger.warning(f"Failed to convert legacy example: {e}")
            return None
    
    def _parse_aircraft_from_legacy_input(self, input_text: str) -> List[Dict[str, Any]]:
        """Parse aircraft information from legacy input text."""
        # Simplified parsing - in practice, this would be more sophisticated
        aircraft_states = [
            {
                "id": "AC001",
                "lat": 52.37,
                "lon": 4.90,
                "alt": 35000,
                "hdg": 90,
                "spd": 450
            },
            {
                "id": "AC002", 
                "lat": 52.37,
                "lon": 4.91,
                "alt": 35000,
                "hdg": 270,
                "spd": 460
            }
        ]
        return aircraft_states
    
    def _convert_legacy_resolution_example(self, legacy_example: Dict[str, Any], 
                                         aircraft_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert legacy resolution example."""
        # Create conflict info
        conflict_info = {
            "aircraft_1_id": aircraft_states[0]["id"],
            "aircraft_2_id": aircraft_states[1]["id"],
            "aircraft_1": aircraft_states[0],
            "aircraft_2": aircraft_states[1],
            "time_to_conflict": 120.0,
            "closest_approach_distance": 3.5,
        }
        
        # Extract command from legacy output
        output_text = legacy_example.get("output", "")
        command = "HDG AC001 045"  # Simplified - would parse from output
        rationale = "Turn to avoid conflict"  # Simplified
        
        return self.generate_conflict_resolution_training_example(
            conflict_info, command, rationale
        )
    
    def _convert_legacy_detection_example(self, legacy_example: Dict[str, Any],
                                        aircraft_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert legacy detection example."""
        # Create ground truth from legacy output
        output_text = legacy_example.get("output", "")
        has_conflict = "conflict" in output_text.lower()
        
        ground_truth_conflicts = []
        if has_conflict:
            ground_truth_conflicts = [{
                "aircraft_1_id": aircraft_states[0]["id"],
                "aircraft_2_id": aircraft_states[1]["id"],
                "time_to_conflict": 120.0,
                "horizontal_distance": 3.5,
                "vertical_separation": 0
            }]
        
        return self.generate_conflict_detection_training_example(
            aircraft_states, ground_truth_conflicts
        )


def main():
    """Main function to demonstrate standardized training data generation."""
    logging.basicConfig(level=logging.INFO)
    
    generator = StandardizedTrainingDataGenerator()
    
    # Example: Convert legacy training data
    legacy_file = "BSKY_GYM_LLM/data/training_examples.jsonl"
    standardized_file = "BSKY_GYM_LLM/data/standardized_training_examples.jsonl"
    
    if Path(legacy_file).exists():
        converted_count = generator.convert_legacy_training_data(legacy_file, standardized_file)
        print(f"✅ Converted {converted_count} examples to standardized format")
        print(f"📁 Output saved to: {standardized_file}")
    else:
        print(f"❌ Legacy file not found: {legacy_file}")


if __name__ == "__main__":
    main()
