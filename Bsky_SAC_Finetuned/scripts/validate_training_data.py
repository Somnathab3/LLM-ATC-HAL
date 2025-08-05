#!/usr/bin/env python3
"""
Comprehensive Training Data Validation System
============================================

Validates training data JSON files against strict schema and quality requirements.
Performs comprehensive checks including:
- JSON schema validation
- Field-specific regex validation
- Reward component consistency checks
- Scenario uniqueness and coverage analysis
- Duplicate detection
- Perplexity smoke tests using DialoGPT

Usage:
    python validate_training_data.py --input training_data/ --environment all
    python validate_training_data.py --input horizontal_cr_samples.json --environment horizontal
"""

import json
import hashlib
import re
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import argparse

import jsonschema
import numpy as np
from tqdm import tqdm

# Optional dependencies for perplexity testing
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - skipping perplexity tests")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results from validation process"""
    total_samples: int = 0
    valid_samples: int = 0
    schema_errors: List[str] = None
    content_errors: List[str] = None
    regex_errors: List[str] = None
    consistency_errors: List[str] = None
    uniqueness_errors: List[str] = None
    coverage_errors: List[str] = None
    duplicate_errors: List[str] = None
    perplexity_errors: List[str] = None
    
    def __post_init__(self):
        """Initialize lists"""
        if self.schema_errors is None:
            self.schema_errors = []
        if self.content_errors is None:
            self.content_errors = []
        if self.regex_errors is None:
            self.regex_errors = []
        if self.consistency_errors is None:
            self.consistency_errors = []
        if self.uniqueness_errors is None:
            self.uniqueness_errors = []
        if self.coverage_errors is None:
            self.coverage_errors = []
        if self.duplicate_errors is None:
            self.duplicate_errors = []
        if self.perplexity_errors is None:
            self.perplexity_errors = []
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return (len(self.schema_errors) == 0 and 
                len(self.content_errors) == 0 and
                len(self.regex_errors) == 0 and
                len(self.consistency_errors) == 0 and
                len(self.uniqueness_errors) == 0 and
                len(self.coverage_errors) == 0 and
                len(self.duplicate_errors) == 0 and
                len(self.perplexity_errors) == 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "validation_passed": self.is_valid,
            "error_counts": {
                "schema_errors": len(self.schema_errors),
                "content_errors": len(self.content_errors),
                "regex_errors": len(self.regex_errors),
                "consistency_errors": len(self.consistency_errors),
                "uniqueness_errors": len(self.uniqueness_errors),
                "coverage_errors": len(self.coverage_errors),
                "duplicate_errors": len(self.duplicate_errors),
                "perplexity_errors": len(self.perplexity_errors)
            }
        }


class TrainingDataValidator:
    """Comprehensive training data validator"""
    
    def __init__(self):
        self.schema = self._create_json_schema()
        self.regex_patterns = self._create_regex_patterns()
        self.coverage_keywords = self._create_coverage_keywords()
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        
    def _create_json_schema(self) -> Dict[str, Any]:
        """Create JSON schema for validation"""
        # Schema for direct format (our main format)
        main_schema = {
            "type": "object",
            "required": [
                "environment", 
                "episode_id", 
                "timestep", 
                "scenario_description", 
                "observation_summary", 
                "expert_action", 
                "reasoning", 
                "reward_components"
            ],
            "properties": {
                "environment": {"type": "string"},
                "episode_id": {"type": "string"},
                "timestep": {"type": "integer"},
                "scenario_description": {"type": "string"},
                "observation_summary": {"type": "string"},
                "expert_action": {"type": "string"},
                "reasoning": {"type": "string"},
                "reward_components": {
                    "type": "object",
                    "required": ["drift_penalty", "intrusion_penalty", "total_reward"],
                    "properties": {
                        "drift_penalty": {"type": "number"},
                        "intrusion_penalty": {"type": "number"},
                        "total_reward": {"type": "number"},
                        "merge_reward": {"type": "number"},
                        "landing_reward": {"type": "number"},
                        "task_completion": {"type": "number"},
                        "conflict_penalty": {"type": "number"},
                        "action_efficiency": {"type": "number"}
                    },
                    "additionalProperties": True
                },
                "safety_metrics": {"type": "object"}
            },
            "additionalProperties": True
        }
        
        return {"main_format": main_schema}
    
    def _create_regex_patterns(self) -> Dict[str, List[str]]:
        """Create regex patterns for action validation"""
        return {
            "horizontal": [
                r"^(Turn|Heading)\s+(left|right)\s+\d+°",
                r"^Maintain\s+current\s+heading",
                r"^Turn\s+(left|right)\s+\d+°"
            ],
            "vertical": [
                r"^(Climb|Descend)\s+at\s+\d+\s*ft/min",
                r"^Maintain\s+current\s+vertical\s+speed",
                r"^(Increase|Decrease)\s+(climb|descent)\s+rate"
            ],
            "sector": [
                r"^Heading\s+change\s+\d+°,\s*set\s+speed\s+\d+\s*kt",
                r"^Turn\s+(left|right)\s+\d+°",
                r"^Set\s+speed\s+\d+\s*kt",
                r"^Maintain\s+current\s+(heading|speed)",
                r"^Maintain\s+current\s+heading\s+and\s+speed"
            ],
            "merge": [
                r"^Adjust\s+heading\s+by\s+\d+°,\s*speed\s+to\s+\d+\s*kt",
                r"^Adjust\s+heading\s+by\s+\d+°",
                r"^Adjust\s+speed\s+to\s+\d+\s*kt",
                r"^Maintain\s+current\s+approach\s+profile"
            ]
        }
    
    def _create_coverage_keywords(self) -> Dict[str, List[str]]:
        """Create coverage keywords for environment validation"""
        return {
            "horizontal": ["drift", "intruder", "waypoint"],
            "vertical": ["altitude", "vertical speed", "runway"],
            "sector": ["sector", "airspeed", "traffic density"],
            "merge": ["merge", "FAF", "traffic flow"]
        }
    
    def _initialize_perplexity_model(self):
        """Initialize DialoGPT model for perplexity testing"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - skipping perplexity model initialization")
            return
            
        try:
            model_name = "microsoft/DialoGPT-small"
            logger.info(f"Loading perplexity model: {model_name}")
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.perplexity_model.eval()
            
            # Add padding token if missing
            if self.perplexity_tokenizer.pad_token is None:
                self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
                
            logger.info("Perplexity model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load perplexity model: {e}")
            self.perplexity_model = None
            self.perplexity_tokenizer = None
    
    def validate_file(self, file_path: str, environment: Optional[str] = None) -> ValidationResults:
        """Validate a single training data file"""
        logger.info(f"Validating file: {file_path}")
        
        results = ValidationResults()
        
        try:
            # Load data
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    data = []
                    for line_num, line in enumerate(f, 1):
                        try:
                            data.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            results.schema_errors.append(f"Line {line_num}: Invalid JSON - {e}")
                else:
                    data = json.load(f)
            
            if not isinstance(data, list):
                results.schema_errors.append("Data must be a list of samples")
                return results
                
            results.total_samples = len(data)
            logger.info(f"Loaded {results.total_samples} samples")
            
            # Extract environment from filename if not provided
            if environment is None:
                if "horizontal" in file_path.lower():
                    environment = "horizontal"
                elif "vertical" in file_path.lower():
                    environment = "vertical"
                elif "sector" in file_path.lower():
                    environment = "sector"
                elif "merge" in file_path.lower():
                    environment = "merge"
                else:
                    environment = "unknown"
            
            # Run validation checks
            self._validate_schema(data, results)
            self._validate_content(data, results)
            self._validate_regex(data, environment, results)
            self._validate_consistency(data, results)
            self._validate_uniqueness(data, results)
            self._validate_coverage(data, environment, results)
            self._validate_duplicates(data, results)
            
            # Run perplexity tests (optional)
            if TRANSFORMERS_AVAILABLE:
                self._validate_perplexity(data, results)
            
            results.valid_samples = results.total_samples - len(results.schema_errors)
            
        except Exception as e:
            results.schema_errors.append(f"Failed to load file: {e}")
        
        return results
    
    def _validate_schema(self, data: List[Dict[str, Any]], results: ValidationResults):
        """Validate JSON schema"""
        logger.info("Validating JSON schema...")
        
        for i, sample in enumerate(tqdm(data, desc="Schema validation")):
            try:
                # We expect our main format with required fields
                jsonschema.validate(sample, self.schema["main_format"])
                
            except jsonschema.exceptions.ValidationError as e:
                results.schema_errors.append(f"Sample {i}: Schema validation failed - {e.message}")
            except Exception as e:
                results.schema_errors.append(f"Sample {i}: Unexpected error - {e}")
    
    def _validate_content(self, data: List[Dict[str, Any]], results: ValidationResults):
        """Validate content requirements with improved length checking"""
        logger.info("Validating content requirements...")
        
        for i, sample in enumerate(tqdm(data, desc="Content validation")):
            try:
                # Extract fields from our direct format
                scenario_desc = sample.get("scenario_description", "")
                obs_summary = sample.get("observation_summary", "")
                expert_action = sample.get("expert_action", "")
                reasoning = sample.get("reasoning", "")
                
                # Check minimum lengths
                if len(scenario_desc.strip()) < 20:
                    results.content_errors.append(f"Sample {i}: scenario_description too short ({len(scenario_desc.strip())} chars)")
                
                if len(obs_summary.strip()) < 20:
                    results.content_errors.append(f"Sample {i}: observation_summary too short ({len(obs_summary.strip())} chars)")
                
                if len(expert_action.strip()) < 5:
                    results.content_errors.append(f"Sample {i}: expert_action too short ({len(expert_action.strip())} chars)")
                
                if len(reasoning.strip()) < 5:
                    results.content_errors.append(f"Sample {i}: reasoning too short ({len(reasoning.strip())} chars)")
                    
            except Exception as e:
                results.content_errors.append(f"Sample {i}: Content validation error - {e}")
    
    def _validate_regex(self, data: List[Dict[str, Any]], environment: str, results: ValidationResults):
        """Validate expert actions against regex patterns"""
        logger.info(f"Validating regex patterns for {environment}...")
        
        if environment not in self.regex_patterns:
            logger.warning(f"No regex patterns defined for environment: {environment}")
            return
        
        patterns = self.regex_patterns[environment]
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for i, sample in enumerate(tqdm(data, desc="Regex validation")):
            try:
                # Extract expert action from our direct format
                expert_action = sample.get("expert_action", "")
                
                # Check if action matches any pattern
                matches_pattern = any(pattern.search(expert_action) for pattern in compiled_patterns)
                
                if not matches_pattern:
                    results.regex_errors.append(
                        f"Sample {i}: expert_action doesn't match {environment} patterns: '{expert_action[:100]}...'"
                    )
                    
            except Exception as e:
                results.regex_errors.append(f"Sample {i}: Regex validation error - {e}")
    
    def _validate_consistency(self, data: List[Dict[str, Any]], results: ValidationResults):
        """Validate reward component consistency"""
        logger.info("Validating reward component consistency...")
        
        for i, sample in enumerate(tqdm(data, desc="Consistency validation")):
            try:
                # Extract reward components from our direct format
                reward_components = sample.get("reward_components", {})
                
                if not reward_components:
                    results.consistency_errors.append(f"Sample {i}: Missing reward_components")
                    continue
                
                # Get required components
                drift_penalty = reward_components.get("drift_penalty", 0.0)
                intrusion_penalty = reward_components.get("intrusion_penalty", 0.0)
                total_reward = reward_components.get("total_reward", 0.0)
                
                # Get optional components
                merge_reward = reward_components.get("merge_reward", 0.0)
                landing_reward = reward_components.get("landing_reward", 0.0)
                task_completion = reward_components.get("task_completion", 0.0)
                conflict_penalty = reward_components.get("conflict_penalty", 0.0)
                action_efficiency = reward_components.get("action_efficiency", 0.0)
                
                # Calculate expected total (different approaches based on available components)
                if "task_completion" in reward_components:
                    # New format with task_completion - includes all penalty components
                    expected_total = (task_completion + drift_penalty + intrusion_penalty + 
                                    conflict_penalty + action_efficiency + merge_reward + landing_reward)
                else:
                    # Original format
                    expected_total = drift_penalty + intrusion_penalty + merge_reward + landing_reward
                
                # Check consistency with tolerance
                tolerance = 1e-6
                if abs(expected_total - total_reward) > tolerance:
                    results.consistency_errors.append(
                        f"Sample {i}: Reward components inconsistent - "
                        f"expected {expected_total:.6f}, got {total_reward:.6f}"
                    )
                    
            except Exception as e:
                results.consistency_errors.append(f"Sample {i}: Consistency validation error - {e}")
    
    def _validate_uniqueness(self, data: List[Dict[str, Any]], results: ValidationResults):
        """Validate scenario uniqueness with all four text fields hashed"""
        logger.info("Validating scenario uniqueness...")
        
        # Hash complete samples using all four text fields
        sample_hashes = set()
        
        for i, sample in enumerate(tqdm(data, desc="Uniqueness validation")):
            try:
                # Extract all four text fields for uniqueness check
                scenario_description = sample.get("scenario_description", "")
                observation_summary = sample.get("observation_summary", "")
                expert_action = sample.get("expert_action", "")
                reasoning = sample.get("reasoning", "")
                
                # Concatenate all four text fields for uniqueness hash
                combined_text = scenario_description + observation_summary + expert_action + reasoning
                
                # Create hash of all four text fields
                sample_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest()
                
                if sample_hash in sample_hashes:
                    results.uniqueness_errors.append(f"Sample {i}: Duplicate content detected in combined text fields")
                else:
                    sample_hashes.add(sample_hash)
                    
            except Exception as e:
                results.uniqueness_errors.append(f"Sample {i}: Uniqueness validation error - {e}")
        
        # Check overall uniqueness rate
        unique_samples = len(sample_hashes)
        uniqueness_rate = unique_samples / len(data) if len(data) > 0 else 0
        
        if uniqueness_rate < 0.95:  # Require 95% unique samples
            results.uniqueness_errors.append(
                f"Low uniqueness rate: {uniqueness_rate:.2%} unique samples (< 95% required)"
            )
        
        logger.info(f"Found {unique_samples} unique samples out of {len(data)} total ({uniqueness_rate:.2%})")
    
    def _validate_coverage(self, data: List[Dict[str, Any]], environment: str, results: ValidationResults):
        """Validate environment-specific keyword coverage"""
        logger.info(f"Validating coverage for {environment}...")
        
        if environment not in self.coverage_keywords:
            logger.warning(f"No coverage keywords defined for environment: {environment}")
            return
        
        keywords = self.coverage_keywords[environment]
        keyword_counts = defaultdict(int)
        samples_with_keywords = 0
        
        for i, sample in enumerate(tqdm(data, desc="Coverage validation")):
            try:
                # Extract text to check
                if "instruction" in sample:
                    text_to_check = sample.get("instruction", "") + " " + sample.get("output", "")
                else:
                    text_to_check = (sample.get("scenario_description", "") + " " + 
                                   sample.get("observation_summary", "") + " " +
                                   sample.get("expert_action", "") + " " +
                                   sample.get("reasoning", ""))
                
                text_lower = text_to_check.lower()
                sample_has_keywords = False
                
                # Check for each keyword
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        keyword_counts[keyword] += 1
                        sample_has_keywords = True
                
                if sample_has_keywords:
                    samples_with_keywords += 1
                    
            except Exception as e:
                results.coverage_errors.append(f"Sample {i}: Coverage validation error - {e}")
        
        # Check coverage requirements
        coverage_percentage = (samples_with_keywords / len(data)) * 100 if data else 0
        min_coverage = 80.0  # At least 80% of samples should contain environment keywords
        
        if coverage_percentage < min_coverage:
            results.coverage_errors.append(
                f"Insufficient keyword coverage: {coverage_percentage:.1f}% < {min_coverage}% required"
            )
        
        # Check that each keyword appears at least once
        for keyword in keywords:
            if keyword_counts[keyword] == 0:
                results.coverage_errors.append(f"Keyword '{keyword}' never appears in samples")
    
    def _validate_duplicates(self, data: List[Dict[str, Any]], results: ValidationResults):
        """Validate duplicate record detection"""
        logger.info("Validating duplicate records...")
        
        serialized_samples = set()
        duplicate_count = 0
        
        for i, sample in enumerate(tqdm(data, desc="Duplicate validation")):
            try:
                # Serialize the entire sample
                serialized = json.dumps(sample, sort_keys=True)
                
                if serialized in serialized_samples:
                    duplicate_count += 1
                    if duplicate_count <= 10:  # Only report first 10 duplicates
                        results.duplicate_errors.append(f"Sample {i}: Exact duplicate detected")
                else:
                    serialized_samples.add(serialized)
                    
            except Exception as e:
                results.duplicate_errors.append(f"Sample {i}: Duplicate validation error - {e}")
        
        # Check duplicate rate
        duplicate_rate = (duplicate_count / len(data)) * 100 if data else 0
        max_duplicate_rate = 1.0  # Maximum 1% duplicates allowed
        
        if duplicate_rate > max_duplicate_rate:
            results.duplicate_errors.append(
                f"Excessive duplicate rate: {duplicate_rate:.2f}% > {max_duplicate_rate}% allowed"
            )
    
    def _validate_perplexity(self, data: List[Dict[str, Any]], results: ValidationResults):
        """Validate using perplexity smoke test"""
        logger.info("Running perplexity smoke test...")
        
        if self.perplexity_model is None:
            self._initialize_perplexity_model()
        
        if self.perplexity_model is None:
            logger.warning("Perplexity model not available - skipping perplexity validation")
            return
        
        # Sample random scenarios for testing
        sample_size = min(500, len(data))
        sampled_indices = np.random.choice(len(data), sample_size, replace=False)
        
        perplexities = []
        
        for idx in tqdm(sampled_indices, desc="Perplexity validation"):
            try:
                sample = data[idx]
                
                # Extract scenario and action
                if "instruction" in sample:
                    scenario = sample.get("instruction", "")
                    action = sample.get("output", "")
                else:
                    scenario = sample.get("scenario_description", "")
                    action = sample.get("expert_action", "")
                
                # Combine for perplexity calculation
                text = f"{scenario} {action}"
                
                # Tokenize
                inputs = self.perplexity_tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                
                # Calculate perplexity
                with torch.no_grad():
                    outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
                    
            except Exception as e:
                results.perplexity_errors.append(f"Sample {idx}: Perplexity calculation error - {e}")
        
        if perplexities:
            avg_perplexity = np.mean(perplexities)
            max_perplexity_threshold = 50000.0  # Higher threshold for technical aviation text
            
            logger.info(f"Average perplexity: {avg_perplexity:.2f}")
            
            if avg_perplexity > max_perplexity_threshold:
                results.perplexity_errors.append(
                    f"High average perplexity: {avg_perplexity:.2f} > {max_perplexity_threshold} threshold"
                )


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--environment", help="Environment type (horizontal/vertical/sector/merge/all)")
    parser.add_argument("--output", help="Output validation report file")
    parser.add_argument("--strict", action="store_true", help="Fail on any validation error")
    
    args = parser.parse_args()
    
    validator = TrainingDataValidator()
    
    # Determine files to validate
    input_path = Path(args.input)
    if input_path.is_file():
        files_to_validate = [str(input_path)]
    elif input_path.is_dir():
        files_to_validate = []
        for pattern in ["*.json", "*.jsonl"]:
            files_to_validate.extend(str(f) for f in input_path.glob(pattern))
    else:
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Validate files
    all_results = {}
    overall_valid = True
    
    for file_path in files_to_validate:
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating: {file_path}")
        logger.info(f"{'='*60}")
        
        results = validator.validate_file(file_path, args.environment)
        all_results[file_path] = results
        
        # Print results
        logger.info(f"\nValidation Results for {Path(file_path).name}:")
        logger.info(f"Total samples: {results.total_samples}")
        logger.info(f"Valid samples: {results.valid_samples}")
        logger.info(f"Validation passed: {results.is_valid}")
        
        if not results.is_valid:
            overall_valid = False
            logger.error(f"Validation failed for {file_path}")
            
            # Print error details
            for error_type, errors in [
                ("Schema errors", results.schema_errors),
                ("Content errors", results.content_errors),
                ("Regex errors", results.regex_errors),
                ("Consistency errors", results.consistency_errors),
                ("Uniqueness errors", results.uniqueness_errors),
                ("Coverage errors", results.coverage_errors),
                ("Duplicate errors", results.duplicate_errors),
                ("Perplexity errors", results.perplexity_errors)
            ]:
                if errors:
                    logger.error(f"\n{error_type}: {len(errors)}")
                    for error in errors[:5]:  # Show first 5 errors
                        logger.error(f"  - {error}")
                    if len(errors) > 5:
                        logger.error(f"  ... and {len(errors) - 5} more")
    
    # Generate summary report
    summary = {
        "validation_timestamp": str(np.datetime64('now')),
        "overall_validation_passed": overall_valid,
        "files_validated": len(files_to_validate),
        "results": {file_path: results.get_summary() for file_path, results in all_results.items()}
    }
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Validation report saved to: {args.output}")
    
    # Print overall summary
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Files validated: {len(files_to_validate)}")
    logger.info(f"Overall validation passed: {overall_valid}")
    
    total_samples = sum(results.total_samples for results in all_results.values())
    total_valid = sum(results.valid_samples for results in all_results.values())
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total valid samples: {total_valid}")
    
    if args.strict and not overall_valid:
        logger.error("Validation failed in strict mode")
        sys.exit(1)
    
    logger.info("Validation completed!")


if __name__ == "__main__":
    main()
