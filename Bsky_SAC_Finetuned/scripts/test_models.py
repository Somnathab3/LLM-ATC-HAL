#!/usr/bin/env python3
"""
ATC Model Testing and Evaluation Framework
=========================================

This script evaluates fine-tuned LLM models against SAC expert policies
and provides comprehensive performance analysis for ATC decision-making.

Features:
- Load and test environment-specific fine-tuned models
- Compare LLM decisions with SAC expert actions
- Safety evaluation and separation analysis
- Performance metrics and efficiency assessment
- Detailed reporting and visualization
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """Test scenario for model evaluation"""
    scenario_id: str
    environment: str
    observation: np.ndarray
    expert_action: np.ndarray
    scenario_description: str
    observation_summary: str
    expected_action_description: str
    safety_metrics: Dict[str, float]
    

@dataclass
class ModelPrediction:
    """Model prediction result"""
    scenario_id: str
    llm_response: str
    parsed_action: Optional[np.ndarray]
    confidence_score: float
    response_time: float
    safety_assessment: Dict[str, float]


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result"""
    environment: str
    model_path: str
    total_scenarios: int
    agreement_rate: float
    safety_score: float
    efficiency_score: float
    response_time_avg: float
    detailed_results: List[Dict[str, Any]]


class ActionParser:
    """Parses LLM text responses into numerical actions"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self._setup_parsing_patterns()
    
    def _setup_parsing_patterns(self):
        """Setup regex patterns for action parsing"""
        import re
        self.patterns = {
            "heading_change": [
                r"turn\s+(?:left|right)\s+(\d+)(?:\s*degrees?|\s*°)?",
                r"heading\s+change\s+(?:of\s+)?([+-]?\d+)(?:\s*degrees?|\s*°)?",
                r"(?:left|right)\s+(\d+)(?:\s*degrees?|\s*°)?",
            ],
            "speed_change": [
                r"(?:increase|reduce|decrease)\s+speed\s+(?:by\s+)?(\d+)(?:\s*knots?|\s*kts?)?",
                r"speed\s+change\s+(?:of\s+)?([+-]?\d+)(?:\s*knots?|\s*kts?)?",
            ],
            "vertical_speed": [
                r"(?:climb|descent)\s+(?:rate\s+)?(?:by\s+)?(\d+)(?:\s*(?:feet\s*per\s*minute|fpm))?",
                r"vertical\s+speed\s+(?:change\s+)?(?:of\s+)?([+-]?\d+)(?:\s*(?:feet\s*per\s*minute|fpm))?",
            ],
            "maintain": [
                r"maintain\s+(?:current\s+)?(?:heading|speed|altitude)",
                r"no\s+change",
                r"continue\s+(?:current\s+)?(?:heading|speed)",
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in pattern_list]
    
    def parse_response(self, response: str) -> Optional[np.ndarray]:
        """Parse LLM response into action array"""
        response = response.lower().strip()
        
        if self.environment_name == "HorizontalCREnv-v0":
            return self._parse_horizontal_action(response)
        elif self.environment_name == "VerticalCREnv-v0":
            return self._parse_vertical_action(response)
        elif self.environment_name in ["SectorCREnv-v0", "MergeEnv-v0"]:
            return self._parse_sector_merge_action(response)
        
        return None
    
    def _parse_horizontal_action(self, response: str) -> Optional[np.ndarray]:
        """Parse horizontal (heading-only) action"""
        # Check for maintain patterns first
        for pattern in self.compiled_patterns["maintain"]:
            if pattern.search(response):
                return np.array([0.0])
        
        # Look for heading changes
        for pattern in self.compiled_patterns["heading_change"]:
            match = pattern.search(response)
            if match:
                degrees = float(match.group(1))
                
                # Determine direction
                if "left" in response:
                    return np.array([-degrees])
                elif "right" in response:
                    return np.array([degrees])
                else:
                    # Check for explicit sign
                    sign_match = re.search(r'([+-])', match.group(0))
                    if sign_match:
                        sign = 1 if sign_match.group(1) == '+' else -1
                        return np.array([sign * degrees])
                    return np.array([degrees])  # Default to right turn
        
        return np.array([0.0])  # Default to no change
    
    def _parse_vertical_action(self, response: str) -> Optional[np.ndarray]:
        """Parse vertical (vertical speed) action"""
        # Check for maintain patterns
        for pattern in self.compiled_patterns["maintain"]:
            if pattern.search(response):
                return np.array([0.0])
        
        # Look for vertical speed changes
        for pattern in self.compiled_patterns["vertical_speed"]:
            match = pattern.search(response)
            if match:
                rate = float(match.group(1))
                
                # Determine direction
                if any(word in response for word in ["climb", "increase", "up"]):
                    return np.array([rate])
                elif any(word in response for word in ["descent", "descend", "decrease", "down"]):
                    return np.array([-rate])
                else:
                    # Check for explicit sign
                    sign_match = re.search(r'([+-])', match.group(0))
                    if sign_match:
                        sign = 1 if sign_match.group(1) == '+' else -1
                        return np.array([sign * rate])
                    return np.array([rate])  # Default positive
        
        return np.array([0.0])  # Default to no change
    
    def _parse_sector_merge_action(self, response: str) -> Optional[np.ndarray]:
        """Parse sector/merge (heading and speed) actions"""
        heading_change = 0.0
        speed_change = 0.0
        
        # Parse heading change
        for pattern in self.compiled_patterns["heading_change"]:
            match = pattern.search(response)
            if match:
                degrees = float(match.group(1))
                if "left" in response:
                    heading_change = -degrees
                elif "right" in response:
                    heading_change = degrees
                break
        
        # Parse speed change
        for pattern in self.compiled_patterns["speed_change"]:
            match = pattern.search(response)
            if match:
                speed = float(match.group(1))
                if any(word in response for word in ["increase", "accelerate", "faster"]):
                    speed_change = speed
                elif any(word in response for word in ["reduce", "decrease", "slower", "decelerate"]):
                    speed_change = -speed
                break
        
        return np.array([heading_change, speed_change])


class SafetyEvaluator:
    """Evaluates safety aspects of model decisions"""
    
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.safety_thresholds = self._get_safety_thresholds()
    
    def _get_safety_thresholds(self) -> Dict[str, float]:
        """Get safety thresholds for environment"""
        thresholds = {
            "HorizontalCREnv-v0": {
                "min_horizontal_separation": 5.0,  # nautical miles
                "max_heading_change": 30.0,        # degrees
            },
            "VerticalCREnv-v0": {
                "min_vertical_separation": 1000.0,  # feet
                "max_vertical_rate": 2000.0,        # fpm
            },
            "SectorCREnv-v0": {
                "min_horizontal_separation": 5.0,
                "max_heading_change": 45.0,
                "max_speed_change": 50.0,           # knots
            },
            "MergeEnv-v0": {
                "min_horizontal_separation": 3.0,   # closer for merge
                "max_heading_change": 30.0,
                "max_speed_change": 30.0,
            }
        }
        return thresholds.get(self.environment_name, {})
    
    def evaluate_action_safety(self, action: np.ndarray, observation: np.ndarray) -> Dict[str, float]:
        """Evaluate safety of a specific action"""
        safety_metrics = {
            "overall_safety_score": 1.0,
            "separation_maintained": 1.0,
            "action_magnitude_safe": 1.0,
        }
        
        # Check action magnitude
        if self.environment_name == "HorizontalCREnv-v0":
            if len(action) > 0:
                heading_change = abs(action[0])
                if heading_change > self.safety_thresholds.get("max_heading_change", 30):
                    safety_metrics["action_magnitude_safe"] = 0.0
        
        elif self.environment_name == "VerticalCREnv-v0":
            if len(action) > 0:
                vz_change = abs(action[0])
                if vz_change > self.safety_thresholds.get("max_vertical_rate", 2000):
                    safety_metrics["action_magnitude_safe"] = 0.0
        
        elif self.environment_name in ["SectorCREnv-v0", "MergeEnv-v0"]:
            if len(action) >= 2:
                heading_change = abs(action[0])
                speed_change = abs(action[1])
                max_heading = self.safety_thresholds.get("max_heading_change", 45)
                max_speed = self.safety_thresholds.get("max_speed_change", 50)
                
                if heading_change > max_heading or speed_change > max_speed:
                    safety_metrics["action_magnitude_safe"] = 0.0
        
        # Estimate separation impact (simplified)
        min_separation = self._estimate_minimum_separation(observation, action)
        min_threshold = self.safety_thresholds.get("min_horizontal_separation", 5.0)
        
        if min_separation < min_threshold:
            safety_metrics["separation_maintained"] = min_separation / min_threshold
        
        # Calculate overall score
        safety_metrics["overall_safety_score"] = min(
            safety_metrics["separation_maintained"],
            safety_metrics["action_magnitude_safe"]
        )
        
        return safety_metrics
    
    def _estimate_minimum_separation(self, observation: np.ndarray, action: np.ndarray) -> float:
        """Estimate minimum separation after action (simplified)"""
        # This is a simplified estimation - in practice, you'd need full trajectory prediction
        
        if self.environment_name == "HorizontalCREnv-v0" and len(observation) > 3:
            # Find closest aircraft
            min_distance = float('inf')
            num_conflicts = (len(observation) - 3) // 5
            for i in range(num_conflicts):
                base_idx = 3 + i * 5
                if base_idx + 1 < len(observation):
                    rel_x = observation[base_idx]
                    rel_y = observation[base_idx + 1]
                    distance = np.sqrt(rel_x**2 + rel_y**2)
                    min_distance = min(min_distance, distance)
            
            return min_distance if min_distance != float('inf') else 10.0
        
        return 10.0  # Default safe distance


class ModelEvaluator:
    """Main class for evaluating fine-tuned models"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.environment_name = self.config["environment"]["name"]
        
        # Initialize components
        self.action_parser = ActionParser(self.environment_name)
        self.safety_evaluator = SafetyEvaluator(self.environment_name)
        
        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        
        logger.info(f"Initialized evaluator for {self.environment_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self) -> Tuple[Any, Any]:
        """Load fine-tuned model and tokenizer"""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load base model
            base_model_name = self.config["llm_training"]["base_model"]
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            # Load LoRA weights
            model = PeftModel.from_pretrained(model, self.model_path)
            model.eval()
            
            logger.info(f"Loaded model from {self.model_path}")
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_test_scenarios(self) -> List[TestScenario]:
        """Load test scenarios from training data"""
        # For now, use a subset of training data as test scenarios
        # In practice, you'd want separate test data
        training_data_path = self.model_path.parent.parent.parent / "training_data"
        
        data_file_map = {
            "HorizontalCREnv-v0": "horizontal_cr_samples.json",
            "VerticalCREnv-v0": "vertical_cr_samples.json", 
            "SectorCREnv-v0": "sector_cr_samples.json",
            "MergeEnv-v0": "merge_samples.json"
        }
        
        data_file = training_data_path / data_file_map[self.environment_name]
        
        with open(data_file, 'r') as f:
            training_data = json.load(f)
        
        # Use every 10th sample as test scenario (10% of data)
        test_samples = training_data[::10][:100]  # Maximum 100 test scenarios
        
        scenarios = []
        for i, sample in enumerate(test_samples):
            # Create dummy observation and action arrays
            # In practice, these would come from the actual environment
            obs_size = {
                "HorizontalCREnv-v0": 8,
                "VerticalCREnv-v0": 11,
                "SectorCREnv-v0": 10,
                "MergeEnv-v0": 12
            }
            
            action_size = {
                "HorizontalCREnv-v0": 1,
                "VerticalCREnv-v0": 1,
                "SectorCREnv-v0": 2,
                "MergeEnv-v0": 2
            }
            
            # Create scenario
            scenario = TestScenario(
                scenario_id=sample["scenario_id"],
                environment=self.environment_name,
                observation=np.random.randn(obs_size[self.environment_name]),  # Dummy
                expert_action=np.random.randn(action_size[self.environment_name]),  # Dummy
                scenario_description=sample["scenario_description"],
                observation_summary=sample["observation_summary"],
                expected_action_description=sample["expert_action"],
                safety_metrics=sample.get("safety_metrics", {})
            )
            
            scenarios.append(scenario)
        
        logger.info(f"Loaded {len(scenarios)} test scenarios")
        return scenarios
    
    def _generate_llm_response(self, scenario: TestScenario) -> ModelPrediction:
        """Generate LLM response for a scenario"""
        import time
        
        # Create prompt
        system_prompt = self.config["prompts"]["system_prompt"]
        user_prompt = f"""Scenario: {scenario.scenario_description}

Current Situation: {scenario.observation_summary}

Based on this air traffic control situation, what action should be taken?"""

        # Format conversation
        conversation = f"""<|system|>
{system_prompt}

<|user|>
{user_prompt}

<|assistant|>
"""

        # Tokenize
        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        
        # Parse action
        parsed_action = self.action_parser.parse_response(response)
        
        # Assess safety
        if parsed_action is not None:
            safety_assessment = self.safety_evaluator.evaluate_action_safety(
                parsed_action, scenario.observation
            )
        else:
            safety_assessment = {"overall_safety_score": 0.0}
        
        return ModelPrediction(
            scenario_id=scenario.scenario_id,
            llm_response=response,
            parsed_action=parsed_action,
            confidence_score=1.0,  # Simplified
            response_time=response_time,
            safety_assessment=safety_assessment
        )
    
    def evaluate_model(self) -> EvaluationResult:
        """Evaluate model performance"""
        logger.info(f"Starting model evaluation for {self.environment_name}")
        
        # Load test scenarios
        scenarios = self._load_test_scenarios()
        
        # Generate predictions
        predictions = []
        for scenario in scenarios:
            try:
                prediction = self._generate_llm_response(scenario)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to generate prediction for {scenario.scenario_id}: {e}")
                continue
        
        # Calculate metrics
        agreement_rate = self._calculate_agreement_rate(scenarios, predictions)
        safety_score = self._calculate_safety_score(predictions)
        efficiency_score = self._calculate_efficiency_score(predictions)
        response_time_avg = np.mean([p.response_time for p in predictions])
        
        # Create detailed results
        detailed_results = []
        for scenario, prediction in zip(scenarios, predictions):
            detailed_results.append({
                "scenario_id": scenario.scenario_id,
                "expected_action": scenario.expected_action_description,
                "llm_response": prediction.llm_response,
                "parsed_action": prediction.parsed_action.tolist() if prediction.parsed_action is not None else None,
                "safety_score": prediction.safety_assessment.get("overall_safety_score", 0.0),
                "response_time": prediction.response_time
            })
        
        result = EvaluationResult(
            environment=self.environment_name,
            model_path=str(self.model_path),
            total_scenarios=len(scenarios),
            agreement_rate=agreement_rate,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            response_time_avg=response_time_avg,
            detailed_results=detailed_results
        )
        
        logger.info(f"Evaluation completed: {agreement_rate:.2%} agreement, {safety_score:.3f} safety score")
        return result
    
    def _calculate_agreement_rate(self, scenarios: List[TestScenario], 
                                predictions: List[ModelPrediction]) -> float:
        """Calculate agreement rate with expert actions"""
        # Simplified agreement calculation based on text similarity
        # In practice, you'd compare numerical actions
        
        agreements = 0
        total = 0
        
        for scenario, prediction in zip(scenarios, predictions):
            if prediction.parsed_action is not None:
                # Simple text-based comparison for now
                expected_lower = scenario.expected_action_description.lower()
                response_lower = prediction.llm_response.lower()
                
                # Check for key action words
                if "maintain" in expected_lower and "maintain" in response_lower:
                    agreements += 1
                elif "turn left" in expected_lower and "turn left" in response_lower:
                    agreements += 1
                elif "turn right" in expected_lower and "turn right" in response_lower:
                    agreements += 1
                elif "climb" in expected_lower and "climb" in response_lower:
                    agreements += 1
                elif "descend" in expected_lower and "descend" in response_lower:
                    agreements += 1
            
            total += 1
        
        return agreements / total if total > 0 else 0.0
    
    def _calculate_safety_score(self, predictions: List[ModelPrediction]) -> float:
        """Calculate overall safety score"""
        if not predictions:
            return 0.0
        
        safety_scores = [p.safety_assessment.get("overall_safety_score", 0.0) for p in predictions]
        return np.mean(safety_scores)
    
    def _calculate_efficiency_score(self, predictions: List[ModelPrediction]) -> float:
        """Calculate efficiency score based on response times and action appropriateness"""
        if not predictions:
            return 0.0
        
        # Simple efficiency metric based on response time
        response_times = [p.response_time for p in predictions]
        avg_time = np.mean(response_times)
        
        # Efficiency score: faster is better (normalized to 0-1)
        # Assume 2 seconds is ideal, 10 seconds is poor
        efficiency = max(0, 1 - (avg_time - 2) / 8)
        return min(1.0, efficiency)


def evaluate_all_models():
    """Evaluate all fine-tuned models"""
    base_path = Path(__file__).parent.parent
    
    # Model and config mappings
    model_configs = [
        ("models/horizontal_cr_llama", "configs/horizontal_config.yaml"),
        ("models/vertical_cr_llama", "configs/vertical_config.yaml"),
        ("models/sector_cr_llama", "configs/sector_config.yaml"),
        ("models/merge_llama", "configs/merge_config.yaml")
    ]
    
    all_results = {}
    
    for model_dir, config_file in model_configs:
        model_path = base_path / model_dir
        config_path = base_path / config_file
        
        if not model_path.exists() or not config_path.exists():
            logger.warning(f"Model or config not found: {model_path}, {config_path}")
            continue
        
        try:
            logger.info(f"Evaluating {model_dir}")
            evaluator = ModelEvaluator(str(model_path), str(config_path))
            result = evaluator.evaluate_model()
            
            all_results[result.environment] = asdict(result)
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_dir}: {e}")
            continue
    
    # Save results
    results_file = base_path / "models" / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    for env, result in all_results.items():
        print(f"\n{env}:")
        print(f"  Agreement Rate: {result['agreement_rate']:.2%}")
        print(f"  Safety Score:   {result['safety_score']:.3f}")
        print(f"  Efficiency:     {result['efficiency_score']:.3f}")
        print(f"  Avg Response:   {result['response_time_avg']:.2f}s")
        print(f"  Test Scenarios: {result['total_scenarios']}")
    
    logger.info(f"Evaluation results saved to {results_file}")


def main():
    """Main function"""
    evaluate_all_models()


if __name__ == "__main__":
    main()
