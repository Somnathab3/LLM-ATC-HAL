# llm_interface/ensemble.py
"""
LLM Ensemble System for Enhanced ATC Decision Making
Integrates multiple models with self-consistency and consensus checking
"""

import asyncio
import json
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    consensus_response: Dict
    individual_responses: Dict[str, Dict]
    confidence: float
    consensus_score: float
    uncertainty: float
    response_time: float
    safety_flags: List[str]
    uncertainty_metrics: Dict[str, float]

class OllamaEnsembleClient:
    """Ensemble client for multiple Ollama models"""
    
    def __init__(self):
        self.client = ollama.Client()
        self.models = self._initialize_models()
        self.response_history = []
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize model ensemble configuration"""
        
        # Check available models
        available_models = self._get_available_models()
        
        models = {}
        
        # Primary model - Main decision maker
        if 'llama3.1:8b' in available_models:
            models['primary'] = ModelConfig(
                name='primary',
                model_id='llama3.1:8b',
                role=ModelRole.PRIMARY,
                weight=0.4,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                timeout=10.0
            )
        
        # Validator model - Cross-checks decisions
        if 'mistral:7b' in available_models:
            models['validator'] = ModelConfig(
                name='validator',
                model_id='mistral:7b',
                role=ModelRole.VALIDATOR,
                weight=0.3,
                temperature=0.2,
                max_tokens=300,
                timeout=8.0
            )
        elif 'llama3.1:8b' in available_models:
            # Use same model with different temperature as fallback
            models['validator'] = ModelConfig(
                name='validator',
                model_id='llama3.1:8b',
                role=ModelRole.VALIDATOR,
                weight=0.3,
                temperature=0.3,  # Higher temperature for diversity
                max_tokens=300,
                timeout=8.0
            )
        
        # Technical model - Focus on technical accuracy
        if 'codellama:7b' in available_models:
            models['technical'] = ModelConfig(
                name='technical',
                model_id='codellama:7b',
                role=ModelRole.TECHNICAL,
                weight=0.2,
                temperature=0.1,
                max_tokens=400,
                timeout=10.0
            )
        elif 'llama3.1:8b' in available_models:
            models['technical'] = ModelConfig(
                name='technical',
                model_id='llama3.1:8b',
                role=ModelRole.TECHNICAL,
                weight=0.2,
                temperature=0.05,  # Very low temperature for technical precision
                max_tokens=400,
                timeout=10.0
            )
        
        # Safety model - Focus on safety assessment
        models['safety'] = ModelConfig(
            name='safety',
            model_id='llama3.1:8b',  # Use primary model with safety-focused prompts
            role=ModelRole.SAFETY,
            weight=0.1,
            temperature=0.1,
            max_tokens=200,
            timeout=5.0
        )
        
        logging.info(f"Initialized ensemble with {len(models)} models: {list(models.keys())}")
        return models
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models_response = self.client.list()
            available = [model['name'] for model in models_response['models']]
            logging.info(f"Available Ollama models: {available}")
            return available
        except Exception as e:
            logging.warning(f"Failed to get available models: {e}")
            return ['llama3.1:8b']  # Fallback to known model
    
    def query_ensemble(self, 
                      prompt: str, 
                      context: Dict,
                      require_json: bool = True,
                      timeout: float = 30.0) -> EnsembleResponse:
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
                        require_json
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
                        individual_responses[model_name] = {'error': str(e)}
            
            # Analyze responses for safety flags
            safety_flags = self._analyze_safety_flags(individual_responses)
            
            # Calculate consensus
            consensus_decision, confidence_score, agreement_level = self._calculate_consensus(
                individual_responses
            )
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(individual_responses)
            
            response_time = time.time() - start_time
            
            ensemble_response = EnsembleResponse(
                consensus_response=consensus_decision,
                individual_responses=individual_responses,
                confidence=confidence_score,
                consensus_score=agreement_level,
                uncertainty=0.0,  # Calculate from uncertainty_metrics
                response_time=response_time,
                safety_flags=safety_flags,
                uncertainty_metrics=uncertainty_metrics
            )
            
            # Store in history for learning
            self.response_history.append(ensemble_response)
            
            return ensemble_response
            
        except Exception as e:
            logging.error(f"Ensemble query failed: {e}")
            return self._create_error_response(str(e), time.time() - start_time)
    
    def _create_role_specific_prompts(self, base_prompt: str, context: Dict) -> Dict[ModelRole, str]:
        """Create role-specific prompts for different models"""
        
        role_prompts = {}
        
        # Primary model - General decision making
        role_prompts[ModelRole.PRIMARY] = f"""You are the primary ATC conflict resolution assistant. 
        Analyze the following conflict and recommend the best resolution maneuver.
        
        Context: {json.dumps(context)}
        
        {base_prompt}
        
        Provide a JSON response with: action, type, safety_score, reasoning."""
        
        # Validator model - Cross-validation
        role_prompts[ModelRole.VALIDATOR] = f"""You are a validation specialist for ATC decisions.
        Review the conflict scenario and independently determine the optimal resolution.
        
        Context: {json.dumps(context)}
        
        {base_prompt}
        
        Focus on validating safety and operational compliance. 
        Provide JSON response with: action, type, safety_score, validation_notes."""
        
        # Technical model - Technical accuracy
        role_prompts[ModelRole.TECHNICAL] = f"""You are a technical aviation systems specialist.
        Analyze the conflict from a technical perspective, considering aircraft performance and flight dynamics.
        
        Context: {json.dumps(context)}
        
        {base_prompt}
        
        Focus on technical feasibility and aircraft capability constraints.
        Provide JSON response with: action, type, safety_score, technical_analysis."""
        
        # Safety model - Safety assessment
        role_prompts[ModelRole.SAFETY] = f"""You are a safety assessment specialist for aviation.
        Evaluate the conflict scenario specifically for safety risks and mitigation strategies.
        
        Context: {json.dumps(context)}
        
        {base_prompt}
        
        Focus exclusively on safety implications and risk assessment.
        Provide JSON response with: safety_level, risk_factors, recommended_action."""
        
        return role_prompts
    
    def _query_single_model(self, 
                           model_config: ModelConfig, 
                           prompt: str, 
                           require_json: bool) -> Dict:
        """Query a single model in the ensemble"""
        
        try:
            response = self.client.chat(
                model=model_config.model_id,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': model_config.temperature,
                    'num_predict': model_config.max_tokens,
                    'top_p': 0.9
                }
            )
            
            content = response['message']['content'].strip()
            
            if require_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        return {'error': 'Invalid JSON response', 'raw_content': content}
            else:
                return {'content': content}
                
        except Exception as e:
            logging.error(f"Model {model_config.name} query failed: {e}")
            return {'error': str(e)}
    
    def _analyze_safety_flags(self, responses: Dict[str, Dict]) -> List[str]:
        """Analyze responses for safety flags and concerns"""
        
        safety_flags = []
        
        for model_name, response in responses.items():
            if 'error' in response:
                safety_flags.append(f"Model {model_name} error: {response['error']}")
                continue
            
            # Check for safety indicators
            safety_score = response.get('safety_score', 0.5)
            if safety_score < 0.3:
                safety_flags.append(f"Low safety score from {model_name}: {safety_score}")
            
            # Check for concerning content
            content_str = json.dumps(response).lower()
            concerning_terms = ['emergency', 'critical', 'unsafe', 'violation', 'danger']
            
            for term in concerning_terms:
                if term in content_str:
                    safety_flags.append(f"Safety concern from {model_name}: {term} mentioned")
            
            # Check for invalid recommendations
            action = response.get('action', '')
            if 'invalid' in action.lower() or 'error' in action.lower():
                safety_flags.append(f"Invalid action from {model_name}: {action}")
        
        return safety_flags
    
    def _calculate_consensus(self, responses: Dict[str, Dict]) -> Tuple[Dict, float, float]:
        """Calculate consensus decision from ensemble responses"""
        
        valid_responses = {k: v for k, v in responses.items() if 'error' not in v}
        
        if not valid_responses:
            return {'error': 'No valid responses'}, 0.0, 0.0
        
        # Extract key decision elements
        actions = []
        types = []
        safety_scores = []
        weights = []
        
        for model_name, response in valid_responses.items():
            model_config = self.models.get(model_name)
            if not model_config:
                continue
                
            actions.append(response.get('action', ''))
            types.append(response.get('type', ''))
            safety_scores.append(response.get('safety_score', 0.5))
            weights.append(model_config.weight)
        
        if not actions:
            return {'error': 'No valid actions'}, 0.0, 0.0
        
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
        consensus_action = max(action_scores, key=action_scores.get) if action_scores else ''
        consensus_type = max(type_scores, key=type_scores.get) if type_scores else ''
        
        # Calculate agreement level
        max_action_weight = max(action_scores.values()) if action_scores else 0
        max_type_weight = max(type_scores.values()) if type_scores else 0
        agreement_level = float((max_action_weight + max_type_weight) / 2)
        
        # Calculate confidence based on agreement and safety scores
        safety_variance = float(np.var(safety_scores))
        confidence_score = agreement_level * (1 - safety_variance) * consensus_safety_score
        
        consensus_decision = {
            'action': consensus_action,
            'type': consensus_type,
            'safety_score': consensus_safety_score,
            'consensus_method': 'weighted_voting',
            'participating_models': list(valid_responses.keys())
        }
        
        return consensus_decision, confidence_score, agreement_level
    
    def _calculate_uncertainty_metrics(self, responses: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate uncertainty metrics from ensemble responses"""
        
        valid_responses = {k: v for k, v in responses.items() if 'error' not in v}
        
        if len(valid_responses) < 2:
            return {'epistemic_uncertainty': 1.0, 'response_diversity': 0.0}
        
        # Extract safety scores for uncertainty calculation
        safety_scores = [r.get('safety_score', 0.5) for r in valid_responses.values()]
        
        # Epistemic uncertainty (variance across models)
        epistemic_uncertainty = float(np.var(safety_scores))
        
        # Response diversity (how different are the responses)
        actions = [r.get('action', '') for r in valid_responses.values()]
        unique_actions = len(set(actions))
        response_diversity = float(unique_actions / len(actions))
        
        # Model agreement (how often models agree)
        types = [r.get('type', '') for r in valid_responses.values()]
        unique_types = len(set(types))
        type_agreement = float(1.0 - (unique_types - 1) / max(len(types) - 1, 1))
        
        return {
            'epistemic_uncertainty': epistemic_uncertainty,
            'response_diversity': response_diversity,
            'type_agreement': type_agreement,
            'model_count': len(valid_responses)
        }
    
    def _create_error_response(self, error_msg: str, response_time: float) -> EnsembleResponse:
        """Create error response when ensemble fails"""
        
        return EnsembleResponse(
            consensus_response={'error': error_msg},
            individual_responses={},
            confidence=0.0,
            consensus_score=0.0,
            uncertainty=1.0,
            response_time=response_time,
            safety_flags=[f"Ensemble error: {error_msg}"],
            uncertainty_metrics={'epistemic_uncertainty': 1.0}
        )
    
    def get_ensemble_statistics(self) -> Dict:
        """Get statistics about ensemble performance"""
        
        if not self.response_history:
            return {'error': 'No response history available'}
        
        response_times = [r.response_time for r in self.response_history]
        confidence_scores = [r.confidence_score for r in self.response_history]
        agreement_levels = [r.agreement_level for r in self.response_history]
        safety_flag_counts = [len(r.safety_flags) for r in self.response_history]
        
        stats = {
            'total_queries': len(self.response_history),
            'average_response_time': float(np.mean(response_times)),
            'average_confidence': float(np.mean(confidence_scores)),
            'average_agreement': float(np.mean(agreement_levels)),
            'average_safety_flags': float(np.mean(safety_flag_counts)),
            'response_time_percentiles': {
                'p50': float(np.percentile(response_times, 50)),
                'p95': float(np.percentile(response_times, 95)),
                'p99': float(np.percentile(response_times, 99))
            },
            'model_performance': {}
        }
        
        # Model-specific performance
        for model_name in self.models.keys():
            model_errors = sum(1 for r in self.response_history 
                             if 'error' in r.individual_responses.get(model_name, {}))
            model_success_rate = 1.0 - (model_errors / len(self.response_history))
            
            stats['model_performance'][model_name] = {
                'success_rate': float(model_success_rate),
                'total_queries': len(self.response_history),
                'errors': model_errors
            }
        
        return stats

class RAGValidator:
    """Retrieval-Augmented Generation validator for aviation knowledge"""
    
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize aviation knowledge base"""
        
        return {
            'separation_standards': {
                'horizontal_minimum': 5.0,  # nautical miles
                'vertical_minimum': 1000,  # feet
                'time_minimum': 60        # seconds
            },
            'maneuver_types': {
                'heading_change': {
                    'typical_range': (-30, 30),  # degrees
                    'execution_time': 15,        # seconds
                    'fuel_impact': 'minimal'
                },
                'altitude_change': {
                    'typical_range': (-2000, 2000),  # feet
                    'execution_time': 60,             # seconds
                    'fuel_impact': 'moderate'
                },
                'speed_change': {
                    'typical_range': (-50, 50),  # knots
                    'execution_time': 30,        # seconds
                    'fuel_impact': 'moderate'
                }
            },
            'aircraft_constraints': {
                'B737': {
                    'max_climb_rate': 2000,    # ft/min
                    'max_descent_rate': 2500,  # ft/min
                    'max_bank_angle': 30,      # degrees
                    'service_ceiling': 41000   # feet
                },
                'A320': {
                    'max_climb_rate': 2200,
                    'max_descent_rate': 2800,
                    'max_bank_angle': 30,
                    'service_ceiling': 39000
                }
            },
            'emergency_procedures': {
                'minimum_separation_loss': {
                    'action': 'immediate_vector',
                    'priority': 'critical'
                },
                'equipment_failure': {
                    'action': 'altitude_separation',
                    'priority': 'high'
                }
            }
        }
    
    def validate_response(self, response: Dict, context: Dict) -> Dict[str, Any]:
        """Validate response against aviation knowledge base"""
        
        validation_result = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'knowledge_score': 1.0
        }
        
        try:
            # Validate maneuver type
            maneuver_type = response.get('type', '')
            if maneuver_type not in self.knowledge_base['maneuver_types']:
                validation_result['violations'].append(f"Invalid maneuver type: {maneuver_type}")
                validation_result['valid'] = False
                validation_result['knowledge_score'] -= 0.3
            
            # Validate maneuver parameters
            if maneuver_type in self.knowledge_base['maneuver_types']:
                maneuver_config = self.knowledge_base['maneuver_types'][maneuver_type]
                
                if maneuver_type == 'heading_change':
                    heading_change = response.get('heading_change', 0)
                    min_hdg, max_hdg = maneuver_config['typical_range']
                    if not (min_hdg <= heading_change <= max_hdg):
                        validation_result['warnings'].append(
                            f"Heading change {heading_change}° outside typical range {min_hdg}-{max_hdg}°"
                        )
                        validation_result['knowledge_score'] -= 0.1
                
                elif maneuver_type == 'altitude_change':
                    alt_change = response.get('altitude_change', 0)
                    min_alt, max_alt = maneuver_config['typical_range']
                    if not (min_alt <= alt_change <= max_alt):
                        validation_result['warnings'].append(
                            f"Altitude change {alt_change}ft outside typical range {min_alt}-{max_alt}ft"
                        )
                        validation_result['knowledge_score'] -= 0.1
            
            # Validate aircraft constraints
            aircraft_type = context.get('aircraft_type', 'B737')
            if aircraft_type in self.knowledge_base['aircraft_constraints']:
                constraints = self.knowledge_base['aircraft_constraints'][aircraft_type]
                
                # Check altitude constraints
                target_altitude = context.get('altitude', 35000) + response.get('altitude_change', 0)
                if target_altitude > constraints['service_ceiling']:
                    validation_result['violations'].append(
                        f"Target altitude {target_altitude}ft exceeds service ceiling {constraints['service_ceiling']}ft"
                    )
                    validation_result['valid'] = False
                    validation_result['knowledge_score'] -= 0.4
            
            # Validate safety margins
            safety_score = response.get('safety_score', 0.5)
            if safety_score < 0.3:
                validation_result['warnings'].append(f"Low safety score: {safety_score}")
                validation_result['knowledge_score'] -= 0.2
            
        except Exception as e:
            logging.error(f"Knowledge validation failed: {e}")
            validation_result['violations'].append(f"Validation error: {str(e)}")
            validation_result['valid'] = False
            validation_result['knowledge_score'] = 0.0
        
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
        'id1': 'AC001',
        'id2': 'AC002',
        'distance': 6.0,
        'time': 120,
        'altitude': 35000
    }
    
    print("Testing ensemble query...")
    response = ensemble.query_ensemble(test_prompt, test_context)
    
    print(f"\nConsensus Decision: {json.dumps(response.consensus_response, indent=2)}")
    print(f"Confidence Score: {response.confidence:.3f}")
    print(f"Consensus Score: {response.consensus_score:.3f}")
    print(f"Response Time: {response.response_time:.3f}s")
    print(f"Safety Flags: {response.safety_flags}")
    
    # Test RAG validator
    rag_validator = RAGValidator()
    validation = rag_validator.validate_response(response.consensus_response, test_context)
    
    print(f"\nKnowledge Validation:")
    print(f"Valid: {validation['valid']}")
    print(f"Knowledge Score: {validation['knowledge_score']:.3f}")
    if validation['violations']:
        print(f"Violations: {validation['violations']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Get ensemble statistics
    stats = ensemble.get_ensemble_statistics()
    print(f"\nEnsemble Statistics:")
    print(json.dumps(stats, indent=2))
