# llm_interface/ensemble.py
"""
LLM Ensemble System for Enhanced ATC Decision Making
Integrates multiple models with self-consistency and consensus checking
"""

import contextlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import ollama


class ModelRole(Enum):
    PRIMARY = "primary"
    VALIDATOR = "validator"
    TECHNICAL = "technical"
    SAFETY = "safety"


@dataclass
class ModelConfig:
    """Configuration for individual model in ensemble"""

    name: str
    model_id: str
    role: ModelRole
    weight: float
    temperature: float
    max_tokens: int
    timeout: float


@dataclass
class EnsembleResponse:
    """Response from ensemble of models"""

    consensus_response: dict
    individual_responses: dict[str, dict]
    confidence: float
    consensus_score: float
    uncertainty: float
    response_time: float
    safety_flags: list[str]
    uncertainty_metrics: dict[str, float]


class OllamaEnsembleClient:
    """Ensemble client for multiple Ollama models"""

    def __init__(self) -> None:
        self.client = ollama.Client()
        self.models = self._initialize_models()
        self.response_history = []

    def _initialize_models(self) -> dict[str, ModelConfig]:
        """Initialize model ensemble configuration"""

        # Check available models
        available_models = self._get_available_models()

        models = {}

        # Fine-tuned BlueSky Gym model (if available)
        if (
            "llama3.1-bsky:latest" in available_models
            or "llama3.1-bsky" in available_models
        ):
            model_id = (
                "llama3.1-bsky:latest"
                if "llama3.1-bsky:latest" in available_models
                else "llama3.1-bsky"
            )
            models["fine_tuned_bsky"] = ModelConfig(
                name="fine_tuned_bsky",
                model_id=model_id,
                role=ModelRole.PRIMARY,
                weight=0.5,  # High weight for fine-tuned model
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                timeout=15.0,
            )

        # Primary model - Main decision maker
        if "llama3.1:8b" in available_models:
            # Adjust weight if fine-tuned model is available
            has_fine_tuned = (
                "llama3.1-bsky:latest" in available_models
                or "llama3.1-bsky" in available_models
            )
            weight = 0.3 if has_fine_tuned else 0.4
            models["primary"] = ModelConfig(
                name="primary",
                model_id="llama3.1:8b",
                role=ModelRole.PRIMARY,
                weight=weight,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                timeout=10.0,
            )

        # Validator model - Cross-checks decisions
        if "mistral:7b" in available_models:
            models["validator"] = ModelConfig(
                name="validator",
                model_id="mistral:7b",
                role=ModelRole.VALIDATOR,
                weight=0.3,
                temperature=0.2,
                max_tokens=300,
                timeout=8.0,
            )
        elif "llama3.1:8b" in available_models:
            # Use same model with different temperature as fallback
            models["validator"] = ModelConfig(
                name="validator",
                model_id="llama3.1:8b",
                role=ModelRole.VALIDATOR,
                weight=0.3,
                temperature=0.3,  # Higher temperature for diversity
                max_tokens=300,
                timeout=8.0,
            )

        # Technical model - Focus on technical accuracy
        if "codellama:7b" in available_models:
            models["technical"] = ModelConfig(
                name="technical",
                model_id="codellama:7b",
                role=ModelRole.TECHNICAL,
                weight=0.2,
                temperature=0.1,
                max_tokens=400,
                timeout=10.0,
            )
        elif "llama3.1:8b" in available_models:
            models["technical"] = ModelConfig(
                name="technical",
                model_id="llama3.1:8b",
                role=ModelRole.TECHNICAL,
                weight=0.2,
                temperature=0.05,  # Very low temperature for technical precision
                max_tokens=400,
                timeout=10.0,
            )

        # Safety model - Focus on safety assessment
        models["safety"] = ModelConfig(
            name="safety",
            model_id="llama3.1:8b",  # Use primary model with safety-focused prompts
            role=ModelRole.SAFETY,
            weight=0.1,
            temperature=0.1,
            max_tokens=200,
            timeout=5.0,
        )

        logging.info(
            f"Initialized ensemble with {len(models)} models: {list(models.keys())}"
        )
        return models

    def _get_available_models(self) -> list[str]:
        """Get list of available Ollama models"""
        try:
            # Try the newer list() method first
            models_response = self.client.list()

            # Handle different response formats
            if hasattr(models_response, "models"):
                # Ollama ListResponse object with models attribute
                available = [model.model for model in models_response.models]
            elif isinstance(models_response, dict):
                if "models" in models_response:
                    # Standard response format: {'models': [{'name': '...', ...}, ...]}
                    available = [model["name"] for model in models_response["models"]]
                elif "data" in models_response:
                    # Alternative response format: {'data': [{'name': '...', ...}, ...]}
                    available = [model["name"] for model in models_response["data"]]
                else:
                    # Try to find any list of models in the response
                    for _key, value in models_response.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and "name" in value[0]:
                                available = [model["name"] for model in value]
                                break
                    else:
                        msg = f"Unexpected response format: {models_response}"
                        raise ValueError(msg)
            elif isinstance(models_response, list):
                # Direct list response: [{'name': '...', ...}, ...]
                available = [model["name"] for model in models_response]
            else:
                msg = f"Unexpected response type: {type(models_response)}"
                raise ValueError(msg)

            logging.info(f"Available Ollama models: {available}")
            return available

        except Exception as e:
            logging.warning(f"Failed to get available models: {e}")
            logging.debug(f"Exception details: {type(e).__name__}: {e!s}")

            # Try alternative method with raw API call if available
            try:
                import requests

                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    tags_data = response.json()
                    if "models" in tags_data:
                        available = [model["name"] for model in tags_data["models"]]
                        logging.info(f"Available models via /api/tags: {available}")
                        return available
            except Exception as alt_e:
                logging.debug(f"Alternative API call also failed: {alt_e}")

            # Final fallback to known models
            fallback_models = ["llama3.1:8b", "mistral:7b", "codellama:7b"]
            logging.warning(f"Using fallback models: {fallback_models}")
            return fallback_models

    def query_ensemble(
        self,
        prompt: str,
        context: dict,
        require_json: bool = True,
        timeout: float = 30.0,
    ) -> EnsembleResponse:
        """Query ensemble of models and return consensus response"""

        start_time = time.time()
        individual_responses = {}
        safety_flags = []

        try:
            # Create role-specific prompts
            role_prompts = self._create_role_specific_prompts(prompt, context)

            # Query models in parallel
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                future_to_model = {}

                for model_name, model_config in self.models.items():
                    role_prompt = role_prompts.get(model_config.role, prompt)
                    future = executor.submit(
                        self._query_single_model,
                        model_config,
                        role_prompt,
                        require_json,
                    )
                    future_to_model[future] = model_name

                # Collect responses with timeout
                for future in as_completed(future_to_model, timeout=timeout):
                    model_name = future_to_model[future]
                    try:
                        response = future.result()
                        individual_responses[model_name] = response
                    except Exception as e:
                        logging.warning(f"Model {model_name} failed: {e}")
                        individual_responses[model_name] = {"error": str(e)}

            # Analyze responses for safety flags
            safety_flags = self._analyze_safety_flags(individual_responses)

            # Calculate consensus
            consensus_decision, confidence_score, agreement_level = (
                self._calculate_consensus(
                    individual_responses,
                )
            )

            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(
                individual_responses
            )

            response_time = time.time() - start_time

            ensemble_response = EnsembleResponse(
                consensus_response=consensus_decision,
                individual_responses=individual_responses,
                confidence=confidence_score,
                consensus_score=agreement_level,
                uncertainty=0.0,  # Calculate from uncertainty_metrics
                response_time=response_time,
                safety_flags=safety_flags,
                uncertainty_metrics=uncertainty_metrics,
            )

            # Store in history for learning
            self.response_history.append(ensemble_response)

            return ensemble_response

        except Exception as e:
            logging.exception(f"Ensemble query failed: {e}")
            return self._create_error_response(str(e), time.time() - start_time)

    def _create_role_specific_prompts(
        self,
        base_prompt: str,
        context: dict,
    ) -> dict[ModelRole, str]:
        """Create role-specific prompts for different models"""

        role_prompts = {}

        # Primary model - General decision making
        role_prompts[
            ModelRole.PRIMARY
        ] = f"""You are the primary ATC conflict resolution assistant.
        Analyze the following conflict and recommend the best resolution maneuver.

        Context: {json.dumps(context)}

        {base_prompt}

        Provide a JSON response with: action, type, safety_score, reasoning."""

        # Validator model - Cross-validation
        role_prompts[
            ModelRole.VALIDATOR
        ] = f"""You are a validation specialist for ATC decisions.
        Review the conflict scenario and independently determine the optimal resolution.

        Context: {json.dumps(context)}

        {base_prompt}

        Focus on validating safety and operational compliance.
        Provide JSON response with: action, type, safety_score, validation_notes."""

        # Technical model - Technical accuracy
        role_prompts[
            ModelRole.TECHNICAL
        ] = f"""You are a technical aviation systems specialist.
        Analyze the conflict from a technical perspective, considering aircraft performance and flight dynamics.

        Context: {json.dumps(context)}

        {base_prompt}

        Focus on technical feasibility and aircraft capability constraints.
        Provide JSON response with: action, type, safety_score, technical_analysis."""

        # Safety model - Safety assessment
        role_prompts[
            ModelRole.SAFETY
        ] = f"""You are a safety assessment specialist for aviation.
        Evaluate the conflict scenario specifically for safety risks and mitigation strategies.

        Context: {json.dumps(context)}

        {base_prompt}

        Focus exclusively on safety implications and risk assessment.
        Provide JSON response with: safety_level, risk_factors, recommended_action."""

        return role_prompts

    def _query_single_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        require_json: bool,
    ) -> dict:
        """Query a single model in the ensemble"""

        try:
            response = self.client.chat(
                model=model_config.model_id,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": model_config.temperature,
                    "num_predict": model_config.max_tokens,
                    "top_p": 0.9,
                },
            )

            content = response["message"]["content"].strip()

            if require_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to clean and repair JSON
                    cleaned_json = self._clean_json_response(content)
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re

                        json_match = re.search(r"\{.*\}", content, re.DOTALL)
                        if json_match:
                            try:
                                cleaned_match = self._clean_json_response(
                                    json_match.group()
                                )
                                return json.loads(cleaned_match)
                            except json.JSONDecodeError:
                                pass
                        # Return partial response structure if all parsing fails
                        return self._create_valid_response_structure(content)
            else:
                return {"content": content}

        except Exception as e:
            logging.exception(f"Model {model_config.name} query failed: {e}")
            return {"error": str(e)}

    def _analyze_safety_flags(self, responses: dict[str, dict]) -> list[str]:
        """Analyze responses for safety flags and concerns"""

        safety_flags = []

        for model_name, response in responses.items():
            if "error" in response:
                safety_flags.append(f"Model {model_name} error: {response['error']}")
                continue

            # Check for safety indicators
            safety_score = response.get("safety_score", 0.5)
            # Ensure safety_score is a number
            try:
                safety_score = float(safety_score) if safety_score is not None else 0.5
            except (ValueError, TypeError):
                safety_score = 0.5

            if safety_score < 0.3:
                safety_flags.append(
                    f"Low safety score from {model_name}: {safety_score}"
                )

            # Check for concerning content
            content_str = json.dumps(response).lower()
            concerning_terms = [
                "emergency",
                "critical",
                "unsafe",
                "violation",
                "danger",
            ]

            for term in concerning_terms:
                if term in content_str:
                    safety_flags.append(
                        f"Safety concern from {model_name}: {term} mentioned"
                    )

            # Check for invalid recommendations
            action = response.get("action", "")
            # Ensure action is a string
            if isinstance(action, dict):
                action = str(action)
            elif action is None:
                action = ""
            if "invalid" in str(action).lower() or "error" in str(action).lower():
                safety_flags.append(f"Invalid action from {model_name}: {action}")

        return safety_flags

    def _calculate_consensus(
        self, responses: dict[str, dict]
    ) -> tuple[dict, float, float]:
        """Calculate consensus decision from ensemble responses"""

        valid_responses = {k: v for k, v in responses.items() if "error" not in v}

        if not valid_responses:
            return {"error": "No valid responses"}, 0.0, 0.0

        # Extract key decision elements
        actions = []
        types = []
        safety_scores = []
        weights = []

        for model_name, response in valid_responses.items():
            model_config = self.models.get(model_name)
            if not model_config:
                continue

            # Handle action field properly
            action = response.get("action", "")
            if isinstance(action, dict):
                action = str(action)
            elif action is None:
                action = ""

            # Handle type field properly
            action_type = response.get("type", "")
            if isinstance(action_type, dict):
                action_type = str(action_type)
            elif action_type is None:
                action_type = ""

            actions.append(str(action))
            types.append(str(action_type))
            safety_scores.append(response.get("safety_score", 0.5))
            weights.append(model_config.weight)

        if not actions:
            return {"error": "No valid actions"}, 0.0, 0.0

        # Calculate weighted consensus
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights

        # Safety score consensus (weighted average)
        consensus_safety_score = float(np.average(safety_scores, weights=weights))

        # Action and type consensus (majority vote weighted by model weights)
        action_scores = {}
        type_scores = {}

        for i, (action, action_type) in enumerate(zip(actions, types)):
            action_scores[action] = action_scores.get(action, 0) + weights[i]
            type_scores[action_type] = type_scores.get(action_type, 0) + weights[i]

        # Select consensus action and type
        consensus_action = (
            max(action_scores, key=action_scores.get) if action_scores else ""
        )
        consensus_type = max(type_scores, key=type_scores.get) if type_scores else ""

        # Calculate agreement level
        max_action_weight = max(action_scores.values()) if action_scores else 0
        max_type_weight = max(type_scores.values()) if type_scores else 0
        agreement_level = float((max_action_weight + max_type_weight) / 2)

        # Calculate confidence based on agreement and safety scores
        safety_variance = float(np.var(safety_scores))
        confidence_score = (
            agreement_level * (1 - safety_variance) * consensus_safety_score
        )

        consensus_decision = {
            "action": consensus_action,
            "type": consensus_type,
            "safety_score": consensus_safety_score,
            "consensus_method": "weighted_voting",
            "participating_models": list(valid_responses.keys()),
        }

        return consensus_decision, confidence_score, agreement_level

    def _calculate_uncertainty_metrics(
        self, responses: dict[str, dict]
    ) -> dict[str, float]:
        """Calculate uncertainty metrics from ensemble responses"""

        valid_responses = {k: v for k, v in responses.items() if "error" not in v}

        if len(valid_responses) < 2:
            return {"epistemic_uncertainty": 1.0, "response_diversity": 0.0}

        # Extract safety scores for uncertainty calculation
        safety_scores = [r.get("safety_score", 0.5) for r in valid_responses.values()]

        # Epistemic uncertainty (variance across models)
        epistemic_uncertainty = float(np.var(safety_scores))

        # Response diversity (how different are the responses)
        actions = [r.get("action", "") for r in valid_responses.values()]
        unique_actions = len(set(actions))
        response_diversity = float(unique_actions / len(actions))

        # Model agreement (how often models agree)
        types = [r.get("type", "") for r in valid_responses.values()]
        unique_types = len(set(types))
        type_agreement = float(1.0 - (unique_types - 1) / max(len(types) - 1, 1))

        return {
            "epistemic_uncertainty": epistemic_uncertainty,
            "response_diversity": response_diversity,
            "type_agreement": type_agreement,
            "model_count": len(valid_responses),
        }

    def _create_error_response(
        self, error_msg: str, response_time: float
    ) -> EnsembleResponse:
        """Create error response when ensemble fails"""

        return EnsembleResponse(
            consensus_response={"error": error_msg},
            individual_responses={},
            confidence=0.0,
            consensus_score=0.0,
            uncertainty=1.0,
            response_time=response_time,
            safety_flags=[f"Ensemble error: {error_msg}"],
            uncertainty_metrics={"epistemic_uncertainty": 1.0},
        )

    def get_ensemble_statistics(self) -> dict:
        """Get statistics about ensemble performance"""

        if not self.response_history:
            return {"error": "No response history available"}

        response_times = [r.response_time for r in self.response_history]
        confidence_scores = [r.confidence_score for r in self.response_history]
        agreement_levels = [r.agreement_level for r in self.response_history]
        safety_flag_counts = [len(r.safety_flags) for r in self.response_history]

        stats = {
            "total_queries": len(self.response_history),
            "average_response_time": float(np.mean(response_times)),
            "average_confidence": float(np.mean(confidence_scores)),
            "average_agreement": float(np.mean(agreement_levels)),
            "average_safety_flags": float(np.mean(safety_flag_counts)),
            "response_time_percentiles": {
                "p50": float(np.percentile(response_times, 50)),
                "p95": float(np.percentile(response_times, 95)),
                "p99": float(np.percentile(response_times, 99)),
            },
            "model_performance": {},
        }

        # Model-specific performance
        for model_name in self.models:
            model_errors = sum(
                1
                for r in self.response_history
                if "error" in r.individual_responses.get(model_name, {})
            )
            model_success_rate = 1.0 - (model_errors / len(self.response_history))

            stats["model_performance"][model_name] = {
                "success_rate": float(model_success_rate),
                "total_queries": len(self.response_history),
                "errors": model_errors,
            }

        return stats

    def _clean_json_response(self, json_str: str) -> str:
        """Clean and repair common JSON formatting issues"""
        import re

        # Remove any leading/trailing non-JSON content
        json_str = json_str.strip()

        # Find JSON object boundaries
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx : end_idx + 1]

        # Fix missing commas between fields
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
        json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)

        # Remove trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix unescaped quotes in strings
        return re.sub(r':\s*"([^"]*)"([^,}\]]*)"', r': "\1\2"', json_str)

    def _create_valid_response_structure(self, raw_content: str) -> dict[str, Any]:
        """Create a valid response structure from failed JSON parsing"""
        return {
            "action": "maintain_heading",
            "reasoning": f"JSON parsing failed, using fallback response: {raw_content[:100]}...",
            "confidence": 0.1,
            "safety_check": "passed",
            "alternatives": [],
            "raw_content": raw_content,
            "parsing_error": True,
        }

    def _extract_partial_response_data(self, raw_content: str) -> dict[str, Any]:
        """Extract partial response data from malformed JSON"""
        import re

        result = self._create_valid_response_structure(raw_content)

        # Try to extract common fields
        action_match = re.search(r'"action":\s*"([^"]*)"', raw_content, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1)

        reasoning_match = re.search(
            r'"reasoning":\s*"([^"]*)"', raw_content, re.IGNORECASE
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)

        confidence_match = re.search(
            r'"confidence":\s*([0-9.]+)', raw_content, re.IGNORECASE
        )
        if confidence_match:
            with contextlib.suppress(ValueError):
                result["confidence"] = float(confidence_match.group(1))

        return result


class RAGValidator:
    """Retrieval-Augmented Generation validator for aviation knowledge"""

    def __init__(self) -> None:
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> dict[str, Any]:
        """Initialize aviation knowledge base"""

        return {
            "separation_standards": {
                "horizontal_minimum": 5.0,  # nautical miles
                "vertical_minimum": 1000,  # feet
                "time_minimum": 60,  # seconds
            },
            "maneuver_types": {
                "heading_change": {
                    "typical_range": (-30, 30),  # degrees
                    "execution_time": 15,  # seconds
                    "fuel_impact": "minimal",
                },
                "altitude_change": {
                    "typical_range": (-2000, 2000),  # feet
                    "execution_time": 60,  # seconds
                    "fuel_impact": "moderate",
                },
                "speed_change": {
                    "typical_range": (-50, 50),  # knots
                    "execution_time": 30,  # seconds
                    "fuel_impact": "moderate",
                },
            },
            "aircraft_constraints": {
                "B737": {
                    "max_climb_rate": 2000,  # ft/min
                    "max_descent_rate": 2500,  # ft/min
                    "max_bank_angle": 30,  # degrees
                    "service_ceiling": 41000,  # feet
                },
                "A320": {
                    "max_climb_rate": 2200,
                    "max_descent_rate": 2800,
                    "max_bank_angle": 30,
                    "service_ceiling": 39000,
                },
            },
            "emergency_procedures": {
                "minimum_separation_loss": {
                    "action": "immediate_vector",
                    "priority": "critical",
                },
                "equipment_failure": {
                    "action": "altitude_separation",
                    "priority": "high",
                },
            },
        }

    def validate_response(self, response: dict, context: dict) -> dict[str, Any]:
        """Validate response against aviation knowledge base"""

        validation_result = {
            "valid": True,
            "violations": [],
            "warnings": [],
            "knowledge_score": 1.0,
        }

        try:
            # Validate maneuver type
            maneuver_type = response.get("type", "")
            if maneuver_type not in self.knowledge_base["maneuver_types"]:
                validation_result["violations"].append(
                    f"Invalid maneuver type: {maneuver_type}"
                )
                validation_result["valid"] = False
                validation_result["knowledge_score"] -= 0.3

            # Validate maneuver parameters
            if maneuver_type in self.knowledge_base["maneuver_types"]:
                maneuver_config = self.knowledge_base["maneuver_types"][maneuver_type]

                if maneuver_type == "heading_change":
                    heading_change = response.get("heading_change", 0)
                    min_hdg, max_hdg = maneuver_config["typical_range"]
                    if not (min_hdg <= heading_change <= max_hdg):
                        validation_result["warnings"].append(
                            f"Heading change {heading_change}° outside typical range {min_hdg}-{max_hdg}°",
                        )
                        validation_result["knowledge_score"] -= 0.1

                elif maneuver_type == "altitude_change":
                    alt_change = response.get("altitude_change", 0)
                    min_alt, max_alt = maneuver_config["typical_range"]
                    if not (min_alt <= alt_change <= max_alt):
                        validation_result["warnings"].append(
                            f"Altitude change {alt_change}ft outside typical range {min_alt}-{max_alt}ft",
                        )
                        validation_result["knowledge_score"] -= 0.1

            # Validate aircraft constraints
            aircraft_type = context.get("aircraft_type", "B737")
            if aircraft_type in self.knowledge_base["aircraft_constraints"]:
                constraints = self.knowledge_base["aircraft_constraints"][aircraft_type]

                # Check altitude constraints
                target_altitude = context.get("altitude", 35000) + response.get(
                    "altitude_change",
                    0,
                )
                if target_altitude > constraints["service_ceiling"]:
                    validation_result["violations"].append(
                        f"Target altitude {target_altitude}ft exceeds service ceiling {constraints['service_ceiling']}ft",
                    )
                    validation_result["valid"] = False
                    validation_result["knowledge_score"] -= 0.4

            # Validate safety margins
            safety_score = response.get("safety_score", 0.5)
            if safety_score < 0.3:
                validation_result["warnings"].append(
                    f"Low safety score: {safety_score}"
                )
                validation_result["knowledge_score"] -= 0.2

        except Exception as e:
            logging.exception(f"Knowledge validation failed: {e}")
            validation_result["violations"].append(f"Validation error: {e!s}")
            validation_result["valid"] = False
            validation_result["knowledge_score"] = 0.0

        return validation_result


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create ensemble client
    ensemble = OllamaEnsembleClient()

    # Test ensemble query
    test_prompt = """Analyze this conflict resolution scenario:
    - Aircraft AC001 and AC002 are converging
    - Current separation: 6 nautical miles
    - Time to conflict: 120 seconds
    - Both at FL350

    Recommend the best resolution maneuver."""

    test_context = {
        "id1": "AC001",
        "id2": "AC002",
        "distance": 6.0,
        "time": 120,
        "altitude": 35000,
    }

    response = ensemble.query_ensemble(test_prompt, test_context)

    # Test RAG validator
    rag_validator = RAGValidator()
    validation = rag_validator.validate_response(
        response.consensus_response, test_context
    )

    if validation["violations"]:
        pass
    if validation["warnings"]:
        pass

    # Get ensemble statistics
    stats = ensemble.get_ensemble_statistics()
