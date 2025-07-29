# validation/input_validator.py
"""
Input Validation Module for LLM-ATC-HAL Framework
Provides JSON schema validation and input sanitization for security hardening
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from jsonschema import validate, ValidationError, draft7_format_checker
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    warnings: List[str] = None

class InputValidator:
    """Comprehensive input validator with JSON schema validation and sanitization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_schemas()
    
    def _init_schemas(self):
        """Initialize JSON schemas for different input types"""
        
        # Aircraft schema
        self.aircraft_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "pattern": "^[A-Z0-9]{1,10}$"},
                "aircraft_type": {"type": "string", "enum": ["A320", "B737", "A380", "B777", "CRJ900", "DHC8"]},
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180}, 
                "altitude": {"type": "number", "minimum": 0, "maximum": 60000},
                "heading": {"type": "number", "minimum": 0, "maximum": 360},
                "ground_speed": {"type": "number", "minimum": 0, "maximum": 1000},
                "vertical_rate": {"type": "number", "minimum": -6000, "maximum": 6000},
                "equipment_failure": {"type": "boolean"},
                "pilot_response_delay": {"type": "number", "minimum": 0, "maximum": 60}
            },
            "required": ["id", "aircraft_type", "latitude", "longitude", "altitude", "heading", "ground_speed"],
            "additionalProperties": False
        }
        
        # Scenario configuration schema
        self.scenario_schema = {
            "type": "object",
            "properties": {
                "scenario_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,50}$"},
                "test_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,50}$"},
                "complexity_level": {"type": "string", "enum": ["simple", "moderate", "complex", "extreme"]},
                "scenario_type": {"type": "string"},
                "generation_timestamp": {"type": "number"},
                "airspace_region": {"type": "string"},
                "aircraft_list": {
                    "type": "array",
                    "items": self.aircraft_schema,
                    "minItems": 1,
                    "maxItems": 20
                },
                "environmental_conditions": {
                    "type": "object",
                    "properties": {
                        "weather": {"type": "string", "enum": ["CLEAR", "RAIN", "TURBULENCE", "WIND_SHEAR", "STORM", "FOG", "clear", "turbulence", "wind_shear", "storm", "fog"]},
                        "wind_speed": {"type": "number", "minimum": 0, "maximum": 200},
                        "wind_direction": {"type": "number", "minimum": 0, "maximum": 360},
                        "visibility": {"type": "number", "minimum": 0, "maximum": 20},
                        "turbulence_intensity": {"type": "number", "minimum": 0, "maximum": 1},
                        "temperature": {"type": "number", "minimum": -60, "maximum": 60},
                        "pressure": {"type": "number", "minimum": 800, "maximum": 1100}
                    },
                    "required": ["weather", "wind_speed", "visibility"],
                    "additionalProperties": False
                },
                "traffic_density": {"type": "number", "minimum": 0, "maximum": 1},
                "conflict_probability": {"type": "number", "minimum": 0, "maximum": 1},
                "system_failures": {
                    "type": "object",
                    "properties": {
                        "transponder_failure": {"type": "boolean"},
                        "communication_degraded": {"type": "boolean"},
                        "navigation_accuracy_reduced": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "additionalProperties": True
                }
            },
            "required": ["scenario_id", "aircraft_list", "environmental_conditions"],
            "additionalProperties": True
        }
        
        # LLM prompt schema
        self.llm_prompt_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": 10000,
                    "pattern": "^[\\w\\s\\.,!?():;\\n\\r-]+$"  # Allow colons, newlines, semicolons
                },
                "context": {
                    "type": "object",
                    "additionalProperties": True
                },
                "model": {"type": "string", "enum": ["llama3.1:8b", "mistral:7b", "codellama:7b"]},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096}
            },
            "required": ["prompt", "model"],
            "additionalProperties": False
        }
    
    def validate_aircraft_data(self, aircraft_data: Dict[str, Any]) -> ValidationResult:
        """Validate aircraft data against schema"""
        return self._validate_against_schema(aircraft_data, self.aircraft_schema, "aircraft")
    
    def validate_scenario_data(self, scenario_data: Dict[str, Any]) -> ValidationResult:
        """Validate scenario configuration against schema"""
        return self._validate_against_schema(scenario_data, self.scenario_schema, "scenario")
    
    def validate_llm_prompt(self, prompt_data: Dict[str, Any]) -> ValidationResult:
        """Validate LLM prompt data against schema"""
        return self._validate_against_schema(prompt_data, self.llm_prompt_schema, "llm_prompt")
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any], data_type: str) -> ValidationResult:
        """Generic schema validation method"""
        errors = []
        warnings = []
        
        try:
            # JSON Schema validation
            validate(instance=data, schema=schema, format_checker=draft7_format_checker)
            
            # Additional security checks
            sanitized_data = self._sanitize_data(data, data_type)
            
            # Security-specific validations
            security_warnings = self._perform_security_checks(data, data_type)
            warnings.extend(security_warnings)
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                sanitized_data=sanitized_data,
                warnings=warnings
            )
            
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            self.logger.warning(f"Validation error for {data_type}: {e.message}")
            
        except Exception as e:
            errors.append(f"Validation failed: {str(e)}")
            self.logger.error(f"Unexpected validation error for {data_type}: {e}")
        
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings
        )
    
    def _sanitize_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Sanitize input data to prevent injection attacks"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potential SQL injection characters
                sanitized_value = re.sub(r"[';\"\\]", "", value)
                # Remove potential script tags
                sanitized_value = re.sub(r"<[^>]*>", "", sanitized_value)
                # Limit length
                sanitized_value = sanitized_value[:1000]
                sanitized[key] = sanitized_value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value, data_type)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_item(item) for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_item(self, item: Any) -> Any:
        """Sanitize individual list items"""
        if isinstance(item, str):
            sanitized = re.sub(r"[';\"\\]", "", item)
            sanitized = re.sub(r"<[^>]*>", "", sanitized)
            return sanitized[:1000]
        elif isinstance(item, dict):
            return self._sanitize_data(item, "list_item")
        else:
            return item
    
    def _perform_security_checks(self, data: Dict[str, Any], data_type: str) -> List[str]:
        """Perform additional security checks"""
        warnings = []
        
        # Check for suspicious patterns
        data_str = json.dumps(data).lower()
        
        suspicious_patterns = [
            r"script\s*:",
            r"javascript\s*:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"subprocess",
            r"os\.system",
            r"file\s*:",
            r"data\s*:",
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, data_str):
                warnings.append(f"Suspicious pattern detected: {pattern}")
                self.logger.warning(f"Security warning: suspicious pattern '{pattern}' in {data_type} data")
        
        # Check data size
        if len(data_str) > 100000:  # 100KB limit
            warnings.append("Input data size exceeds recommended limit")
        
        return warnings
    
    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate file paths to prevent directory traversal"""
        errors = []
        warnings = []
        
        # Check for directory traversal attempts
        if ".." in file_path or file_path.startswith("/"):
            errors.append("Directory traversal attempt detected")
        
        # Check for valid file extensions
        allowed_extensions = [".json", ".scn", ".log", ".txt", ".yaml", ".yml"]
        if not any(file_path.endswith(ext) for ext in allowed_extensions):
            warnings.append("File extension not in whitelist")
        
        # Check path length
        if len(file_path) > 255:
            errors.append("File path too long")
        
        # Sanitize path
        sanitized_path = re.sub(r"[<>:\"|?*]", "", file_path)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data={"file_path": sanitized_path},
            warnings=warnings
        )

# Global validator instance
validator = InputValidator()

def validate_input(data: Dict[str, Any], data_type: str) -> ValidationResult:
    """Convenience function for input validation"""
    if data_type == "aircraft":
        return validator.validate_aircraft_data(data)
    elif data_type == "scenario":
        return validator.validate_scenario_data(data)
    elif data_type == "llm_prompt":
        return validator.validate_llm_prompt(data)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# Example usage
if __name__ == "__main__":
    # Test aircraft validation
    test_aircraft = {
        "id": "TEST001",
        "aircraft_type": "A320",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "altitude": 35000,
        "heading": 180,
        "ground_speed": 450
    }
    
    result = validator.validate_aircraft_data(test_aircraft)
    print(f"Aircraft validation: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
