# tools/bluesky_command_validator.py
"""
BlueSky Command Validation and Auto-Correction System
Provides comprehensive validation and intelligent correction of BlueSky commands
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CommandPattern:
    """Pattern definition for BlueSky commands"""
    
    command: str
    description: str
    pattern: str
    parameters: List[str]
    examples: List[str]
    category: str


class BlueSkyCommandValidator:
    """Validates and auto-corrects BlueSky commands"""
    
    def __init__(self):
        self.command_patterns = self._initialize_command_patterns()
        self.natural_language_mappings = self._initialize_nlp_mappings()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_command_patterns(self) -> Dict[str, CommandPattern]:
        """Initialize comprehensive BlueSky command patterns"""
        patterns = {
            # Aircraft Control Commands
            "HDG": CommandPattern(
                command="HDG",
                description="Change aircraft heading",
                pattern=r"^HDG\s+([A-Z0-9]+)\s+(\d{1,3})(?:\s+([LR]))?$",
                parameters=["aircraft_id", "heading_deg", "turn_direction"],
                examples=["HDG AC001 270", "HDG AC001 090 L"],
                category="control"
            ),
            "ALT": CommandPattern(
                command="ALT",
                description="Change aircraft altitude",
                pattern=r"^ALT\s+([A-Z0-9]+)\s+(\d+)(?:\s+(FL|FT))?$",
                parameters=["aircraft_id", "altitude", "unit"],
                examples=["ALT AC001 35000", "ALT AC001 FL350"],
                category="control"
            ),
            "SPD": CommandPattern(
                command="SPD",
                description="Change aircraft speed",
                pattern=r"^SPD\s+([A-Z0-9]+)\s+(\d+)(?:\s+(KT|MACH))?$",
                parameters=["aircraft_id", "speed", "unit"],
                examples=["SPD AC001 250", "SPD AC001 MACH 0.8"],
                category="control"
            ),
            "VS": CommandPattern(
                command="VS",
                description="Set vertical speed",
                pattern=r"^VS\s+([A-Z0-9]+)\s+([-+]?\d+)$",
                parameters=["aircraft_id", "vertical_speed_fpm"],
                examples=["VS AC001 1500", "VS AC001 -800"],
                category="control"
            ),
            
            # Route Commands
            "DIRECT": CommandPattern(
                command="DIRECT",
                description="Direct aircraft to waypoint",
                pattern=r"^DIRECT\s+([A-Z0-9]+)\s+([A-Z0-9]{3,5})$",
                parameters=["aircraft_id", "waypoint"],
                examples=["DIRECT AC001 EHAM", "DIRECT AC001 WPT01"],
                category="navigation"
            ),
            "LNAV": CommandPattern(
                command="LNAV",
                description="Enable lateral navigation",
                pattern=r"^LNAV\s+([A-Z0-9]+)\s+(ON|OFF)$",
                parameters=["aircraft_id", "state"],
                examples=["LNAV AC001 ON", "LNAV AC001 OFF"],
                category="navigation"
            ),
            "VNAV": CommandPattern(
                command="VNAV",
                description="Enable vertical navigation",
                pattern=r"^VNAV\s+([A-Z0-9]+)\s+(ON|OFF)$",
                parameters=["aircraft_id", "state"],
                examples=["VNAV AC001 ON", "VNAV AC001 OFF"],
                category="navigation"
            ),
            
            # Aircraft Management
            "CRE": CommandPattern(
                command="CRE",
                description="Create aircraft",
                pattern=r"^CRE\s+([A-Z0-9]+)\s+([A-Z0-9]+)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+(\d+)\s+(\d+)\s+(\d+)$",
                parameters=["aircraft_id", "type", "lat", "lon", "heading", "altitude", "speed"],
                examples=["CRE AC001 B738 50.0 4.0 90 35000 250"],
                category="management"
            ),
            "DEL": CommandPattern(
                command="DEL",
                description="Delete aircraft",
                pattern=r"^DEL\s+([A-Z0-9]+)$",
                parameters=["aircraft_id"],
                examples=["DEL AC001"],
                category="management"
            ),
            "MOVE": CommandPattern(
                command="MOVE",
                description="Move aircraft position",
                pattern=r"^MOVE\s+([A-Z0-9]+)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)(?:\s+(\d+))?$",
                parameters=["aircraft_id", "lat", "lon", "altitude"],
                examples=["MOVE AC001 51.0 4.5", "MOVE AC001 51.0 4.5 36000"],
                category="management"
            ),
            
            # Conflict Detection and Resolution
            "ASAS": CommandPattern(
                command="ASAS",
                description="Separation assurance system",
                pattern=r"^ASAS\s+(ON|OFF)(?:\s+([A-Z0-9]+))?$",
                parameters=["state", "aircraft_id"],
                examples=["ASAS ON", "ASAS OFF AC001"],
                category="separation"
            ),
            "RMETHH": CommandPattern(
                command="RMETHH",
                description="Set horizontal resolution method",
                pattern=r"^RMETHH\s+(NONE|HDG|SPD|BOTH)$",
                parameters=["method"],
                examples=["RMETHH HDG", "RMETHH BOTH"],
                category="separation"
            ),
            "RMETHV": CommandPattern(
                command="RMETHV",
                description="Set vertical resolution method",
                pattern=r"^RMETHV\s+(NONE|ALT|VS|BOTH)$",
                parameters=["method"],
                examples=["RMETHV ALT", "RMETHV BOTH"],
                category="separation"
            ),
            
            # Simulation Control
            "RESET": CommandPattern(
                command="RESET",
                description="Reset simulation",
                pattern=r"^RESET$",
                parameters=[],
                examples=["RESET"],
                category="simulation"
            ),
            "OP": CommandPattern(
                command="OP",
                description="Start/unpause simulation",
                pattern=r"^OP$",
                parameters=[],
                examples=["OP"],
                category="simulation"
            ),
            "HOLD": CommandPattern(
                command="HOLD",
                description="Pause simulation",
                pattern=r"^HOLD$",
                parameters=[],
                examples=["HOLD"],
                category="simulation"
            ),
            "DT": CommandPattern(
                command="DT",
                description="Set simulation time step",
                pattern=r"^DT\s+(\d+\.?\d*)$",
                parameters=["timestep_seconds"],
                examples=["DT 60.0", "DT 1.0"],
                category="simulation"
            ),
            "DTMULT": CommandPattern(
                command="DTMULT",
                description="Set simulation speed multiplier",
                pattern=r"^DTMULT\s+(\d+\.?\d*)$",
                parameters=["multiplier"],
                examples=["DTMULT 1.0", "DTMULT 10.0"],
                category="simulation"
            ),
            "FF": CommandPattern(
                command="FF",
                description="Fast forward simulation",
                pattern=r"^FF\s+(\d+\.?\d*)$",
                parameters=["minutes"],
                examples=["FF 10.0", "FF 5.5"],
                category="simulation"
            ),
            
            # Information Commands
            "POS": CommandPattern(
                command="POS",
                description="Show aircraft position",
                pattern=r"^POS\s+([A-Z0-9]+)$",
                parameters=["aircraft_id"],
                examples=["POS AC001"],
                category="information"
            ),
            "DIST": CommandPattern(
                command="DIST",
                description="Show distance between aircraft",
                pattern=r"^DIST\s+([A-Z0-9]+)\s+([A-Z0-9]+)$",
                parameters=["aircraft_id1", "aircraft_id2"],
                examples=["DIST AC001 AC002"],
                category="information"
            ),
            
            # Area and Sector Commands
            "AREA": CommandPattern(
                command="AREA",
                description="Set simulation area",
                pattern=r"^AREA\s+([A-Z]{4})$",
                parameters=["airport_code"],
                examples=["AREA EHAM", "AREA KJFK"],
                category="simulation"
            ),
            "ZOOM": CommandPattern(
                command="ZOOM",
                description="Set view zoom level",
                pattern=r"^ZOOM\s+(\d+\.?\d*)$",
                parameters=["zoom_factor"],
                examples=["ZOOM 1.0", "ZOOM 0.5"],
                category="display"
            ),
        }
        
        return patterns
    
    def _initialize_nlp_mappings(self) -> Dict[str, str]:
        """Initialize natural language to BlueSky command mappings"""
        return {
            # Heading/Direction mappings
            "turn": "HDG",
            "heading": "HDG",
            "direction": "HDG",
            "steer": "HDG",
            "bearing": "HDG",
            
            # Altitude mappings
            "climb": "ALT",
            "ascend": "ALT",
            "descend": "ALT",
            "altitude": "ALT",
            "level": "ALT",
            "height": "ALT",
            
            # Speed mappings
            "speed": "SPD",
            "accelerate": "SPD",
            "decelerate": "SPD",
            "slow": "SPD",
            "fast": "SPD",
            
            # Vertical speed mappings
            "climb_rate": "VS",
            "descent_rate": "VS",
            "vertical_speed": "VS",
            
            # Navigation mappings
            "proceed": "DIRECT",
            "route": "DIRECT",
            "navigate": "DIRECT",
            "waypoint": "DIRECT",
            
            # Aircraft management
            "create": "CRE",
            "spawn": "CRE",
            "delete": "DEL",
            "remove": "DEL",
            "move": "MOVE",
            "relocate": "MOVE",
        }
    
    def validate_command(self, command: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a BlueSky command
        
        Returns:
            Tuple of (is_valid, error_message, suggestion)
        """
        if not command:
            return False, "Command cannot be empty", None
        
        command = command.strip().upper()
        if not command:
            return False, "Command cannot be empty", None
        
        # Extract command type
        parts = command.split()
        if not parts:
            return False, "Command cannot be empty", None
        
        command_type = parts[0]
        
        # Check if command exists
        if command_type not in self.command_patterns:
            suggestion = self._suggest_command_correction(command)
            return False, f"Unknown command: {command_type}", suggestion
        
        # Validate command pattern
        pattern = self.command_patterns[command_type]
        if not re.match(pattern.pattern, command):
            return False, f"Invalid format for {command_type}. Expected: {pattern.description}", pattern.examples[0]
        
        return True, None, None
    
    def auto_correct_command(self, command: str, strict_mode: bool = False) -> Tuple[Optional[str], List[str]]:
        """
        Attempt to auto-correct a command
        
        Returns:
            Tuple of (corrected_command, warnings)
        """
        warnings: List[str] = []
        
        if not command:
            return None, ["Invalid command input"]
        
        original_command = command
        command = command.strip()
        
        # Try natural language to BlueSky translation
        corrected = self._translate_natural_language(command)
        if corrected != command:
            command = corrected
            warnings.append(f"Translated '{original_command}' to BlueSky syntax")
        
        # Validate the (potentially corrected) command
        is_valid, error, suggestion = self.validate_command(command)
        
        if is_valid:
            return command, warnings
        
        if strict_mode:
            return None, [f"Command validation failed: {error}"]
        
        # Try to fix common issues
        fixed_command = self._fix_common_issues(command)
        if fixed_command != command:
            is_valid, _, _ = self.validate_command(fixed_command)
            if is_valid:
                warnings.append(f"Auto-corrected command format")
                return fixed_command, warnings
        
        # If we have a suggestion, try that
        if suggestion:
            warnings.append(f"Using suggested format: {suggestion}")
            return suggestion, warnings
        
        return None, [f"Could not correct command: {error}"]
    
    def _suggest_command_correction(self, command: str) -> Optional[str]:
        """Suggest a correction for an unknown command"""
        parts = command.split()
        if not parts:
            return None
        
        command_word = parts[0].lower()
        
        # Check natural language mappings
        if command_word in self.natural_language_mappings:
            bluesky_cmd = self.natural_language_mappings[command_word]
            pattern = self.command_patterns[bluesky_cmd]
            return pattern.examples[0]
        
        # Find similar commands using edit distance
        min_distance = float('inf')
        best_match = None
        
        for cmd in self.command_patterns:
            distance = self._edit_distance(command_word, cmd.lower())
            if distance < min_distance and distance <= 2:  # Allow up to 2 character differences
                min_distance = distance
                best_match = self.command_patterns[cmd].examples[0]
        
        return best_match
    
    def _translate_natural_language(self, command: str) -> str:
        """Translate natural language commands to BlueSky syntax"""
        command_lower = command.lower()
        
        # Common patterns for natural language commands
        patterns = [
            # "Turn AC001 to heading 270" -> "HDG AC001 270"
            (r"turn\s+([a-z0-9]+)\s+to\s+heading\s+(\d+)", r"HDG \1 \2"),
            (r"turn\s+([a-z0-9]+)\s+(\d+)", r"HDG \1 \2"),
            
            # "Climb AC001 to 35000" -> "ALT AC001 35000"
            (r"climb\s+([a-z0-9]+)\s+to\s+(\d+)", r"ALT \1 \2"),
            (r"ascend\s+([a-z0-9]+)\s+to\s+(\d+)", r"ALT \1 \2"),
            (r"descend\s+([a-z0-9]+)\s+to\s+(\d+)", r"ALT \1 \2"),
            
            # "Set AC001 speed to 250" -> "SPD AC001 250"
            (r"set\s+([a-z0-9]+)\s+speed\s+to\s+(\d+)", r"SPD \1 \2"),
            (r"speed\s+([a-z0-9]+)\s+(\d+)", r"SPD \1 \2"),
            
            # "Direct AC001 to EHAM" -> "DIRECT AC001 EHAM"
            (r"direct\s+([a-z0-9]+)\s+to\s+([a-z0-9]+)", r"DIRECT \1 \2"),
            (r"proceed\s+([a-z0-9]+)\s+to\s+([a-z0-9]+)", r"DIRECT \1 \2"),
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, command_lower)
            if match:
                result = re.sub(pattern, replacement, command_lower).upper()
                return result
        
        return command
    
    def _fix_common_issues(self, command: str) -> str:
        """Fix common formatting issues"""
        # Remove extra whitespace
        command = re.sub(r'\s+', ' ', command.strip())
        
        # Ensure uppercase
        command = command.upper()
        
        parts = command.split()
        if not parts:
            return command
        
        command_type = parts[0]
        
        # Command-specific fixes
        if command_type == "HDG" and len(parts) >= 3:
            # Ensure heading is 3 digits
            try:
                heading = int(parts[2])
                parts[2] = f"{heading:03d}"
                command = " ".join(parts)
            except ValueError:
                pass
        
        elif command_type == "ALT" and len(parts) >= 3:
            # Handle FL notation
            if parts[2].startswith("FL"):
                try:
                    fl = int(parts[2][2:])
                    parts[2] = str(fl * 100)
                    command = " ".join(parts)
                except ValueError:
                    pass
        
        elif command_type == "SPD" and len(parts) >= 3:
            # Remove units if present in wrong position
            if parts[2].endswith("KT"):
                parts[2] = parts[2][:-2]
                command = " ".join(parts)
        
        return command
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_command_help(self, command_type: Optional[str] = None) -> str:
        """Get help information for commands"""
        if command_type:
            command_type = command_type.upper()
            if command_type in self.command_patterns:
                pattern = self.command_patterns[command_type]
                help_text = f"""
Command: {pattern.command}
Description: {pattern.description}
Parameters: {', '.join(pattern.parameters)}
Examples:
"""
                for example in pattern.examples:
                    help_text += f"  {example}\n"
                return help_text.strip()
            else:
                return f"Unknown command: {command_type}"
        
        # Return overview of all commands
        categories: Dict[str, List[str]] = {}
        for cmd, pattern in self.command_patterns.items():
            if pattern.category not in categories:
                categories[pattern.category] = []
            categories[pattern.category].append(f"{cmd}: {pattern.description}")
        
        help_text = "BlueSky Commands by Category:\n\n"
        for category, commands in sorted(categories.items()):
            help_text += f"{category.title()}:\n"
            for cmd_desc in commands:
                help_text += f"  {cmd_desc}\n"
            help_text += "\n"
        
        return help_text.strip()
    
    def is_command_supported(self, command: str) -> bool:
        """Check if a command type is supported"""
        if not command:
            return False
        
        command_type = command.strip().upper().split()[0]
        return command_type in self.command_patterns


# Global validator instance
_validator = None

def get_validator() -> BlueSkyCommandValidator:
    """Get the global command validator instance"""
    global _validator
    if _validator is None:
        _validator = BlueSkyCommandValidator()
    return _validator

def validate_command(command: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Convenience function for command validation"""
    return get_validator().validate_command(command)

def auto_correct_command(command: str, strict_mode: bool = False) -> Tuple[Optional[str], List[str]]:
    """Convenience function for command auto-correction"""
    return get_validator().auto_correct_command(command, strict_mode)
