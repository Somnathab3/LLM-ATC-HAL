# tools/baseline_resolution_strategy.py
"""
Baseline Resolution Strategy for ATC Conflict Resolution
Implements conventional ATC strategies for comparison with LLM approaches
"""

import logging
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ResolutionType(Enum):
    """Types of resolution maneuvers"""
    HORIZONTAL_TURN = "horizontal_turn"
    VERTICAL_CLIMB = "vertical_climb"
    VERTICAL_DESCEND = "vertical_descend"
    SPEED_INCREASE = "speed_increase"
    SPEED_DECREASE = "speed_decrease"
    COMBINED_TURN_CLIMB = "combined_turn_climb"
    COMBINED_TURN_DESCEND = "combined_turn_descend"
    NO_ACTION = "no_action"


@dataclass
class ResolutionCommand:
    """A resolution command with metadata"""
    command: str
    aircraft_id: str
    resolution_type: ResolutionType
    magnitude: float
    rationale: str
    priority: int
    estimated_effectiveness: float


@dataclass
class ConflictGeometry:
    """Conflict geometry information"""
    aircraft1_id: str
    aircraft2_id: str
    horizontal_distance_nm: float
    vertical_separation_ft: float
    time_to_closest_approach_min: float
    relative_bearing_deg: float
    aircraft1_heading: float
    aircraft2_heading: float
    aircraft1_altitude: float
    aircraft2_altitude: float
    aircraft1_speed: float
    aircraft2_speed: float
    closing_speed_kts: float


class BaselineResolutionStrategy:
    """Conventional ATC resolution strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard separation requirements
        self.min_horizontal_separation_nm = 5.0
        self.min_vertical_separation_ft = 1000.0
        
        # Resolution parameters
        self.standard_heading_change = 20  # degrees
        self.standard_altitude_change = 1000  # feet
        self.standard_speed_change = 20  # knots
        
        # Strategy weights
        self.strategy_weights = {
            ResolutionType.HORIZONTAL_TURN: 0.4,
            ResolutionType.VERTICAL_CLIMB: 0.25,
            ResolutionType.VERTICAL_DESCEND: 0.25,
            ResolutionType.SPEED_INCREASE: 0.05,
            ResolutionType.SPEED_DECREASE: 0.05,
        }
        
        # Initialize ASAS-like rules
        self.horizontal_methods = ["HDG", "SPD"]
        self.vertical_methods = ["ALT", "VS"]
        
    def generate_baseline_resolution(
        self, 
        conflict: ConflictGeometry,
        preferred_method: Optional[str] = None,
        asas_mode: bool = False
    ) -> List[ResolutionCommand]:
        """
        Generate baseline resolution commands using conventional ATC strategies
        
        Args:
            conflict: Conflict geometry information
            preferred_method: Preferred resolution method ("horizontal", "vertical", "speed")
            asas_mode: Use ASAS-like automated separation logic
            
        Returns:
            List of resolution commands
        """
        
        if asas_mode:
            return self._generate_asas_resolution(conflict)
        
        # Determine best resolution strategy based on conflict geometry
        resolution_type = self._select_resolution_strategy(conflict, preferred_method)
        
        # Generate specific commands based on strategy
        commands = self._generate_commands_for_strategy(conflict, resolution_type)
        
        # Validate and rank commands
        validated_commands = self._validate_and_rank_commands(commands, conflict)
        
        self.logger.info(
            f"Generated {len(validated_commands)} baseline resolution commands for conflict {conflict.aircraft1_id}-{conflict.aircraft2_id}"
        )
        
        return validated_commands
    
    def _select_resolution_strategy(
        self, 
        conflict: ConflictGeometry, 
        preferred_method: Optional[str] = None
    ) -> ResolutionType:
        """Select the most appropriate resolution strategy"""
        
        # If preferred method is specified
        if preferred_method:
            if preferred_method.lower() == "horizontal":
                return ResolutionType.HORIZONTAL_TURN
            elif preferred_method.lower() == "vertical":
                return random.choice([ResolutionType.VERTICAL_CLIMB, ResolutionType.VERTICAL_DESCEND])
            elif preferred_method.lower() == "speed":
                return random.choice([ResolutionType.SPEED_INCREASE, ResolutionType.SPEED_DECREASE])
        
        # Decision logic based on conflict characteristics
        time_factor = conflict.time_to_closest_approach_min
        altitude_diff = abs(conflict.aircraft1_altitude - conflict.aircraft2_altitude)
        horizontal_distance = conflict.horizontal_distance_nm
        
        # Prefer vertical separation if aircraft are at similar altitudes
        if altitude_diff < 500:  # Very close vertically
            if conflict.aircraft1_altitude > conflict.aircraft2_altitude:
                return ResolutionType.VERTICAL_CLIMB if random.random() > 0.5 else ResolutionType.VERTICAL_DESCEND
            else:
                return ResolutionType.VERTICAL_DESCEND if random.random() > 0.5 else ResolutionType.VERTICAL_CLIMB
        
        # Prefer horizontal separation for head-on or near head-on conflicts
        relative_angle = abs(conflict.aircraft1_heading - conflict.aircraft2_heading)
        if 150 <= relative_angle <= 210:  # Head-on conflict
            return ResolutionType.HORIZONTAL_TURN
        
        # For overtaking situations, prefer speed adjustments
        if relative_angle < 30 or relative_angle > 330:  # Same direction
            speed_diff = abs(conflict.aircraft1_speed - conflict.aircraft2_speed)
            if speed_diff > 50:  # Significant speed difference
                return ResolutionType.SPEED_DECREASE if conflict.aircraft1_speed > conflict.aircraft2_speed else ResolutionType.SPEED_INCREASE
        
        # Default to horizontal turn for most situations
        return ResolutionType.HORIZONTAL_TURN
    
    def _generate_commands_for_strategy(
        self, 
        conflict: ConflictGeometry, 
        resolution_type: ResolutionType
    ) -> List[ResolutionCommand]:
        """Generate specific commands for the chosen strategy"""
        
        commands: List[ResolutionCommand] = []
        
        if resolution_type == ResolutionType.HORIZONTAL_TURN:
            commands.extend(self._generate_heading_commands(conflict))
        
        elif resolution_type == ResolutionType.VERTICAL_CLIMB:
            commands.extend(self._generate_climb_commands(conflict))
        
        elif resolution_type == ResolutionType.VERTICAL_DESCEND:
            commands.extend(self._generate_descent_commands(conflict))
        
        elif resolution_type == ResolutionType.SPEED_INCREASE:
            commands.extend(self._generate_speed_increase_commands(conflict))
        
        elif resolution_type == ResolutionType.SPEED_DECREASE:
            commands.extend(self._generate_speed_decrease_commands(conflict))
        
        elif resolution_type in [ResolutionType.COMBINED_TURN_CLIMB, ResolutionType.COMBINED_TURN_DESCEND]:
            commands.extend(self._generate_combined_commands(conflict, resolution_type))
        
        return commands
    
    def _generate_heading_commands(self, conflict: ConflictGeometry) -> List[ResolutionCommand]:
        """Generate heading change commands"""
        commands: List[ResolutionCommand] = []
        
        # Standard right turn for first aircraft
        new_heading_1 = int((conflict.aircraft1_heading + self.standard_heading_change) % 360)
        command_1 = f"HDG {conflict.aircraft1_id} {new_heading_1:03d}"
        
        commands.append(ResolutionCommand(
            command=command_1,
            aircraft_id=conflict.aircraft1_id,
            resolution_type=ResolutionType.HORIZONTAL_TURN,
            magnitude=self.standard_heading_change,
            rationale=f"Turn right {self.standard_heading_change}° to avoid conflict",
            priority=1,
            estimated_effectiveness=0.85
        ))
        
        # Alternative: left turn for second aircraft
        new_heading_2 = int((conflict.aircraft2_heading - self.standard_heading_change) % 360)
        command_2 = f"HDG {conflict.aircraft2_id} {new_heading_2:03d}"
        
        commands.append(ResolutionCommand(
            command=command_2,
            aircraft_id=conflict.aircraft2_id,
            resolution_type=ResolutionType.HORIZONTAL_TURN,
            magnitude=self.standard_heading_change,
            rationale=f"Turn left {self.standard_heading_change}° to avoid conflict",
            priority=2,
            estimated_effectiveness=0.85
        ))
        
        return commands
    
    def _generate_climb_commands(self, conflict: ConflictGeometry) -> List[ResolutionCommand]:
        """Generate climb commands"""
        commands: List[ResolutionCommand] = []
        
        # Choose which aircraft should climb
        aircraft_to_climb = conflict.aircraft1_id if conflict.aircraft1_altitude <= conflict.aircraft2_altitude else conflict.aircraft2_id
        current_altitude = conflict.aircraft1_altitude if aircraft_to_climb == conflict.aircraft1_id else conflict.aircraft2_altitude
        
        new_altitude = current_altitude + self.standard_altitude_change
        command = f"ALT {aircraft_to_climb} {int(new_altitude)}"
        
        commands.append(ResolutionCommand(
            command=command,
            aircraft_id=aircraft_to_climb,
            resolution_type=ResolutionType.VERTICAL_CLIMB,
            magnitude=self.standard_altitude_change,
            rationale=f"Climb {self.standard_altitude_change} ft to establish vertical separation",
            priority=1,
            estimated_effectiveness=0.90
        ))
        
        return commands
    
    def _generate_descent_commands(self, conflict: ConflictGeometry) -> List[ResolutionCommand]:
        """Generate descent commands"""
        commands: List[ResolutionCommand] = []
        
        # Choose which aircraft should descend
        aircraft_to_descend = conflict.aircraft1_id if conflict.aircraft1_altitude >= conflict.aircraft2_altitude else conflict.aircraft2_id
        current_altitude = conflict.aircraft1_altitude if aircraft_to_descend == conflict.aircraft1_id else conflict.aircraft2_altitude
        
        new_altitude = max(10000, current_altitude - self.standard_altitude_change)  # Don't go below 10,000 ft
        command = f"ALT {aircraft_to_descend} {int(new_altitude)}"
        
        commands.append(ResolutionCommand(
            command=command,
            aircraft_id=aircraft_to_descend,
            resolution_type=ResolutionType.VERTICAL_DESCEND,
            magnitude=self.standard_altitude_change,
            rationale=f"Descend {self.standard_altitude_change} ft to establish vertical separation",
            priority=1,
            estimated_effectiveness=0.90
        ))
        
        return commands
    
    def _generate_speed_increase_commands(self, conflict: ConflictGeometry) -> List[ResolutionCommand]:
        """Generate speed increase commands"""
        commands: List[ResolutionCommand] = []
        
        # Choose faster aircraft to speed up more
        aircraft_to_speed_up = conflict.aircraft1_id if conflict.aircraft1_speed >= conflict.aircraft2_speed else conflict.aircraft2_id
        current_speed = conflict.aircraft1_speed if aircraft_to_speed_up == conflict.aircraft1_id else conflict.aircraft2_speed
        
        new_speed = min(350, current_speed + self.standard_speed_change)  # Don't exceed 350 kts
        command = f"SPD {aircraft_to_speed_up} {int(new_speed)}"
        
        commands.append(ResolutionCommand(
            command=command,
            aircraft_id=aircraft_to_speed_up,
            resolution_type=ResolutionType.SPEED_INCREASE,
            magnitude=self.standard_speed_change,
            rationale=f"Increase speed by {self.standard_speed_change} kts to resolve conflict timing",
            priority=1,
            estimated_effectiveness=0.70
        ))
        
        return commands
    
    def _generate_speed_decrease_commands(self, conflict: ConflictGeometry) -> List[ResolutionCommand]:
        """Generate speed decrease commands"""
        commands: List[ResolutionCommand] = []
        
        # Choose faster aircraft to slow down
        aircraft_to_slow_down = conflict.aircraft1_id if conflict.aircraft1_speed >= conflict.aircraft2_speed else conflict.aircraft2_id
        current_speed = conflict.aircraft1_speed if aircraft_to_slow_down == conflict.aircraft1_id else conflict.aircraft2_speed
        
        new_speed = max(180, current_speed - self.standard_speed_change)  # Don't go below 180 kts
        command = f"SPD {aircraft_to_slow_down} {int(new_speed)}"
        
        commands.append(ResolutionCommand(
            command=command,
            aircraft_id=aircraft_to_slow_down,
            resolution_type=ResolutionType.SPEED_DECREASE,
            magnitude=self.standard_speed_change,
            rationale=f"Reduce speed by {self.standard_speed_change} kts to resolve conflict timing",
            priority=1,
            estimated_effectiveness=0.70
        ))
        
        return commands
    
    def _generate_combined_commands(
        self, 
        conflict: ConflictGeometry, 
        resolution_type: ResolutionType
    ) -> List[ResolutionCommand]:
        """Generate combined maneuver commands"""
        commands: List[ResolutionCommand] = []
        
        # Generate both heading and altitude changes
        heading_commands = self._generate_heading_commands(conflict)
        
        if resolution_type == ResolutionType.COMBINED_TURN_CLIMB:
            altitude_commands = self._generate_climb_commands(conflict)
        else:
            altitude_commands = self._generate_descent_commands(conflict)
        
        # Combine for same aircraft if possible
        for heading_cmd in heading_commands:
            for alt_cmd in altitude_commands:
                if heading_cmd.aircraft_id == alt_cmd.aircraft_id:
                    combined_command = ResolutionCommand(
                        command=f"{heading_cmd.command}; {alt_cmd.command}",
                        aircraft_id=heading_cmd.aircraft_id,
                        resolution_type=resolution_type,
                        magnitude=(heading_cmd.magnitude + alt_cmd.magnitude) / 2,
                        rationale=f"Combined maneuver: {heading_cmd.rationale} and {alt_cmd.rationale}",
                        priority=1,
                        estimated_effectiveness=0.95
                    )
                    commands.append(combined_command)
                    break
        
        # If no combined command possible, return individual commands
        if not commands:
            commands.extend(heading_commands[:1])  # Take best heading command
            commands.extend(altitude_commands[:1])  # Take best altitude command
        
        return commands
    
    def _generate_asas_resolution(self, conflict: ConflictGeometry) -> List[ResolutionCommand]:
        """Generate ASAS-like automated resolution"""
        commands: List[ResolutionCommand] = []
        
        # ASAS logic: prefer least disruptive maneuver
        time_to_conflict = conflict.time_to_closest_approach_min
        
        # Urgent conflicts (< 2 minutes) - immediate action required
        if time_to_conflict < 2.0:
            # Use most effective resolution
            if conflict.vertical_separation_ft < 500:
                commands.extend(self._generate_climb_commands(conflict))
            else:
                commands.extend(self._generate_heading_commands(conflict))
        
        # Medium-term conflicts (2-5 minutes) - standard resolution
        elif time_to_conflict < 5.0:
            # Prefer horizontal resolution for medium-term conflicts
            commands.extend(self._generate_heading_commands(conflict))
        
        # Long-term conflicts (> 5 minutes) - gentle resolution
        else:
            # Use speed adjustments for long-term conflicts
            if conflict.closing_speed_kts > 100:
                commands.extend(self._generate_speed_decrease_commands(conflict))
            else:
                commands.extend(self._generate_heading_commands(conflict))
        
        # Mark as ASAS-generated
        for cmd in commands:
            cmd.rationale = f"ASAS: {cmd.rationale}"
        
        return commands
    
    def _validate_and_rank_commands(
        self, 
        commands: List[ResolutionCommand], 
        conflict: ConflictGeometry
    ) -> List[ResolutionCommand]:
        """Validate and rank resolution commands"""
        
        valid_commands: List[ResolutionCommand] = []
        
        for cmd in commands:
            # Basic validation
            if self._validate_command_safety(cmd, conflict):
                valid_commands.append(cmd)
            else:
                self.logger.warning(f"Command failed safety validation: {cmd.command}")
        
        # Sort by priority and effectiveness
        valid_commands.sort(key=lambda x: (x.priority, -x.estimated_effectiveness))
        
        return valid_commands
    
    def _validate_command_safety(self, command: ResolutionCommand, conflict: ConflictGeometry) -> bool:
        """Validate that a command is safe"""
        
        # Check altitude limits
        if "ALT" in command.command:
            parts = command.command.split()
            if len(parts) >= 3:
                try:
                    new_altitude = int(parts[2])
                    if new_altitude < 5000 or new_altitude > 45000:
                        return False
                except ValueError:
                    return False
        
        # Check speed limits
        if "SPD" in command.command:
            parts = command.command.split()
            if len(parts) >= 3:
                try:
                    new_speed = int(parts[2])
                    if new_speed < 150 or new_speed > 400:
                        return False
                except ValueError:
                    return False
        
        # Check heading validity
        if "HDG" in command.command:
            parts = command.command.split()
            if len(parts) >= 3:
                try:
                    new_heading = int(parts[2])
                    if new_heading < 0 or new_heading >= 360:
                        return False
                except ValueError:
                    return False
        
        return True
    
    def get_resolution_summary(self, commands: List[ResolutionCommand]) -> str:
        """Generate a summary of resolution commands"""
        if not commands:
            return "No resolution commands generated"
        
        summary = f"Baseline Resolution Strategy ({len(commands)} commands):\n"
        
        for i, cmd in enumerate(commands, 1):
            summary += f"{i}. {cmd.command}\n"
            summary += f"   Aircraft: {cmd.aircraft_id}\n"
            summary += f"   Type: {cmd.resolution_type.value}\n"
            summary += f"   Rationale: {cmd.rationale}\n"
            summary += f"   Effectiveness: {cmd.estimated_effectiveness:.2f}\n\n"
        
        return summary.strip()


# Global strategy instance
_strategy = None

def get_baseline_strategy() -> BaselineResolutionStrategy:
    """Get the global baseline strategy instance"""
    global _strategy
    if _strategy is None:
        _strategy = BaselineResolutionStrategy()
    return _strategy

def generate_baseline_resolution(
    conflict: ConflictGeometry, 
    preferred_method: Optional[str] = None,
    asas_mode: bool = False
) -> List[ResolutionCommand]:
    """Convenience function for baseline resolution generation"""
    return get_baseline_strategy().generate_baseline_resolution(conflict, preferred_method, asas_mode)
