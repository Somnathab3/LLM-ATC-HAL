# analysis/enhanced_hallucination_detection.py
"""
Enhanced Multi-Layer Hallucination Detection Framework for ATC LLMs
Based on latest research in LLM hallucination detection and safety-critical systems

Implementation of Six-Layer Detection System:
Layer 1: MIND Framework, Attention Pattern Detection, Eigenvalue Analysis
Layer 2: Semantic Entropy, Predictive Uncertainty, Convex Hull Dispersion
Layer 3: Self-Consistency, Multi-Model Consensus, RAG Validation
"""

import numpy as np
import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Fallback for when sentence-transformers is not available
    SentenceTransformer = None
import time
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class HallucinationType(Enum):
    FABRICATION = "fabrication"
    OMISSION = "omission"
    IRRELEVANCY = "irrelevancy"
    CONTRADICTION = "contradiction"
    SEMANTIC_DRIFT = "semantic_drift"
    UNCERTAINTY_COLLAPSE = "uncertainty_collapse"

@dataclass
class HallucinationResult:
    """Result of hallucination detection analysis"""
    detected: bool
    types: List[HallucinationType]
    confidence: float
    evidence: Dict[str, Any]
    safety_impact: str  # 'critical', 'moderate', 'minimal'
    layer_scores: Dict[str, float]

# ==================== LAYER 1 DETECTORS ====================

class MINDFramework:
    """Mutual Information Neural Decomposition for hallucination detection"""
    
    def __init__(self):
        self.threshold = 0.15
        self.baseline_entropy = None
        
    def analyze_information_flow(self, response_data: Dict, context_data: Dict) -> Tuple[float, Dict]:
        """Compute mutual information score between response and context"""
        try:
            # Convert text to numerical features for analysis
            response_features = self._extract_numerical_features(response_data)
            context_features = self._extract_numerical_features(context_data)
            
            # Calculate correlation-based mutual information
            if len(response_features) > 0 and len(context_features) > 0:
                correlation = np.corrcoef(response_features, context_features)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Convert correlation to information score
                mutual_info_score = abs(correlation)
                
                evidence = {
                    'correlation': float(correlation),
                    'response_variance': float(np.var(response_features)),
                    'context_variance': float(np.var(context_features)),
                    'feature_count': len(response_features)
                }
                
                return mutual_info_score, evidence
            else:
                return 0.0, {'error': 'insufficient_features'}
                
        except Exception as e:
            logging.error(f"MIND framework analysis failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _extract_numerical_features(self, data: Dict) -> np.ndarray:
        """Extract numerical features from response/context data"""
        features = []
        
        # Extract numerical values
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Convert string to features (length, word count, etc.)
                features.extend([
                    len(value),
                    len(value.split()),
                    value.count(' '),
                    sum(1 for c in value if c.isdigit())
                ])
        
        return np.array(features) if features else np.array([0])

class AttentionPatternDetector:
    """Attention pattern analysis for hallucination detection"""
    
    def __init__(self):
        self.attention_threshold = 0.3
        
    def analyze_attention_patterns(self, response: Dict, context: Dict) -> Tuple[float, Dict]:
        """Analyze attention patterns for inconsistencies"""
        try:
            # Simulate attention analysis based on response relevance
            attention_score = self._calculate_attention_score(response, context)
            
            # Check for attention drift indicators
            drift_indicators = self._detect_attention_drift(response, context)
            
            evidence = {
                'attention_score': attention_score,
                'drift_indicators': drift_indicators,
                'focus_consistency': self._measure_focus_consistency(response)
            }
            
            return attention_score, evidence
            
        except Exception as e:
            logging.error(f"Attention pattern analysis failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_attention_score(self, response: Dict, context: Dict) -> float:
        """Calculate attention alignment score"""
        # Check if response addresses context elements
        context_elements = set(context.keys())
        response_elements = set(response.keys())
        
        overlap = len(context_elements.intersection(response_elements))
        total_context = len(context_elements)
        
        return overlap / max(total_context, 1)
    
    def _detect_attention_drift(self, response: Dict, context: Dict) -> List[str]:
        """Detect attention drift patterns"""
        indicators = []
        
        # Check for missing critical context elements
        critical_keys = ['id1', 'id2', 'distance', 'time']
        for key in critical_keys:
            if key in context and key not in str(response):
                indicators.append(f'missing_context_{key}')
        
        return indicators
    
    def _measure_focus_consistency(self, response: Dict) -> float:
        """Measure internal focus consistency"""
        # Simple consistency measure based on response structure
        if not isinstance(response, dict):
            return 0.0
        
        required_fields = ['action', 'type', 'safety_score']
        present_fields = sum(1 for field in required_fields if field in response)
        
        return present_fields / len(required_fields)

class EigenvalueAnalyzer:
    """Eigenvalue-based stability analysis for hallucination detection"""
    
    def __init__(self):
        self.stability_threshold = 0.7
        
    def analyze_stability(self, response_sequence: List[Dict]) -> Tuple[float, Dict]:
        """Analyze eigenvalue stability across response sequence"""
        try:
            if len(response_sequence) < 2:
                return 1.0, {'note': 'insufficient_sequence'}
            
            # Extract feature matrices from responses
            feature_matrices = []
            for response in response_sequence:
                features = self._response_to_features(response)
                feature_matrices.append(features)
            
            # Calculate stability metrics
            stability_score = self._calculate_eigenvalue_stability(feature_matrices)
            
            evidence = {
                'stability_score': stability_score,
                'sequence_length': len(response_sequence),
                'variance_analysis': self._analyze_variance(feature_matrices)
            }
            
            return stability_score, evidence
            
        except Exception as e:
            logging.error(f"Eigenvalue analysis failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _response_to_features(self, response: Dict) -> np.ndarray:
        """Convert response to feature vector"""
        features = []
        
        # Safety score
        features.append(response.get('safety_score', 0.5))
        
        # Action type encoding
        action_types = ['heading', 'altitude', 'speed', 'vector', 'hold']
        action_type = response.get('type', '').lower()
        for atype in action_types:
            features.append(1.0 if atype in action_type else 0.0)
        
        # Numerical parameters
        for param in ['heading_change', 'altitude_change', 'speed_change']:
            features.append(response.get(param, 0))
        
        return np.array(features)
    
    def _calculate_eigenvalue_stability(self, feature_matrices: List[np.ndarray]) -> float:
        """Calculate stability based on eigenvalue analysis"""
        try:
            # Stack features and compute covariance
            stacked = np.vstack(feature_matrices)
            cov_matrix = np.cov(stacked.T)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)  # Take real part
            
            # Stability metric: ratio of largest to smallest eigenvalue
            if len(eigenvalues) > 1 and np.min(eigenvalues) > 1e-10:
                condition_number = np.max(eigenvalues) / np.min(eigenvalues)
                stability = 1.0 / (1.0 + np.log(max(condition_number, 1)))
            else:
                stability = 0.5
            
            return float(stability)
            
        except Exception as e:
            logging.error(f"Eigenvalue stability calculation failed: {e}")
            return 0.5
    
    def _analyze_variance(self, feature_matrices: List[np.ndarray]) -> Dict:
        """Analyze variance patterns"""
        try:
            stacked = np.vstack(feature_matrices)
            variances = np.var(stacked, axis=0)
            
            return {
                'total_variance': float(np.sum(variances)),
                'max_variance': float(np.max(variances)),
                'variance_distribution': variances.tolist()
            }
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            logging.warning(f"Eigenvalue variance analysis failed: {e}")
            return {'error': 'variance_analysis_failed', 'details': str(e)}

# ==================== LAYER 2 DETECTORS ====================

class SemanticEntropyCalculator:
    """Semantic entropy-based uncertainty quantification"""
    
    def __init__(self):
        self.entropy_threshold = 2.5
        
    def calculate_semantic_entropy(self, response: Dict, context: Dict) -> Tuple[float, Dict]:
        """Calculate semantic entropy of response given context"""
        try:
            # Calculate information content
            response_info = self._calculate_information_content(response)
            context_info = self._calculate_information_content(context)
            
            # Semantic entropy based on information distribution
            entropy = self._compute_entropy(response_info, context_info)
            
            evidence = {
                'entropy_value': entropy,
                'response_complexity': response_info,
                'context_complexity': context_info,
                'normalized_entropy': entropy / max(context_info, 1e-6)
            }
            
            return entropy, evidence
            
        except Exception as e:
            logging.error(f"Semantic entropy calculation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_information_content(self, data: Dict) -> float:
        """Calculate information content of data structure"""
        if not isinstance(data, dict):
            return 0.0
        
        info_content = 0.0
        
        for key, value in data.items():
            if isinstance(value, str):
                # String complexity based on length and uniqueness
                info_content += len(set(value.lower())) / max(len(value), 1)
            elif isinstance(value, (int, float)):
                # Numerical information based on magnitude
                info_content += np.log(max(abs(value), 1e-6))
            elif isinstance(value, (list, dict)):
                # Nested structure complexity
                info_content += len(str(value)) / 100
        
        return info_content
    
    def _compute_entropy(self, response_info: float, context_info: float) -> float:
        """Compute entropy measure"""
        if context_info == 0:
            return float('inf')
        
        # Entropy as ratio of information contents
        entropy = response_info / context_info
        
        # Apply logarithmic transformation
        return float(np.log(max(entropy, 1e-6)))

class PredictiveUncertaintyAnalyzer:
    """Predictive uncertainty analysis for hallucination detection"""
    
    def __init__(self):
        self.uncertainty_threshold = 0.6
        
    def analyze_predictive_uncertainty(self, response: Dict, baseline: Dict) -> Tuple[float, Dict]:
        """Analyze predictive uncertainty in response"""
        try:
            # Calculate prediction divergence
            divergence = self._calculate_divergence(response, baseline)
            
            # Estimate uncertainty metrics
            uncertainty_metrics = self._estimate_uncertainty(response, baseline)
            
            # Overall uncertainty score
            uncertainty_score = self._compute_uncertainty_score(divergence, uncertainty_metrics)
            
            evidence = {
                'divergence': divergence,
                'uncertainty_metrics': uncertainty_metrics,
                'uncertainty_score': uncertainty_score
            }
            
            return uncertainty_score, evidence
            
        except Exception as e:
            logging.error(f"Predictive uncertainty analysis failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_divergence(self, response: Dict, baseline: Dict) -> float:
        """Calculate divergence between response and baseline"""
        if not isinstance(response, dict) or not isinstance(baseline, dict):
            return 1.0
        
        # Compare safety scores
        resp_safety = response.get('safety_score', 0.5)
        base_safety = baseline.get('safety_score', 0.5)
        safety_divergence = abs(resp_safety - base_safety)
        
        # Compare action types
        resp_type = response.get('type', '')
        base_type = baseline.get('type', '')
        type_divergence = 1.0 if resp_type != base_type else 0.0
        
        return (safety_divergence + type_divergence) / 2
    
    def _estimate_uncertainty(self, response: Dict, baseline: Dict) -> Dict:
        """Estimate various uncertainty metrics"""
        return {
            'aleatoric': self._estimate_aleatoric_uncertainty(response),
            'epistemic': self._estimate_epistemic_uncertainty(response, baseline),
            'total': self._estimate_total_uncertainty(response, baseline)
        }
    
    def _estimate_aleatoric_uncertainty(self, response: Dict) -> float:
        """Estimate aleatoric (data) uncertainty"""
        # Based on response consistency indicators
        safety_score = response.get('safety_score', 0.5)
        return 1.0 - abs(safety_score - 0.5) * 2  # Higher when safety_score is ~0.5
    
    def _estimate_epistemic_uncertainty(self, response: Dict, baseline: Dict) -> float:
        """Estimate epistemic (model) uncertainty"""
        # Based on divergence from baseline
        return self._calculate_divergence(response, baseline)
    
    def _estimate_total_uncertainty(self, response: Dict, baseline: Dict) -> float:
        """Estimate total uncertainty"""
        aleatoric = self._estimate_aleatoric_uncertainty(response)
        epistemic = self._estimate_epistemic_uncertainty(response, baseline)
        return np.sqrt(aleatoric**2 + epistemic**2)
    
    def _compute_uncertainty_score(self, divergence: float, metrics: Dict) -> float:
        """Compute overall uncertainty score"""
        total_uncertainty = metrics.get('total', 0.5)
        weighted_score = 0.6 * total_uncertainty + 0.4 * divergence
        return float(weighted_score)

class ConvexHullDispersionMeasure:
    """Convex hull-based dispersion analysis for response consistency"""
    
    def __init__(self):
        self.dispersion_threshold = 0.8
        
    def analyze_dispersion(self, response_ensemble: List[Dict]) -> Tuple[float, Dict]:
        """Analyze dispersion of ensemble responses using convex hull"""
        try:
            if len(response_ensemble) < 3:
                return 0.0, {'note': 'insufficient_ensemble_size'}
            
            # Convert responses to points in feature space
            points = []
            for response in response_ensemble:
                point = self._response_to_point(response)
                points.append(point)
            
            points = np.array(points)
            
            # Calculate convex hull and dispersion metrics
            dispersion_score = self._calculate_dispersion(points)
            
            evidence = {
                'dispersion_score': dispersion_score,
                'ensemble_size': len(response_ensemble),
                'feature_dimensions': points.shape[1] if len(points) > 0 else 0,
                'point_spread': self._calculate_point_spread(points)
            }
            
            return dispersion_score, evidence
            
        except Exception as e:
            logging.error(f"Convex hull dispersion analysis failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _response_to_point(self, response: Dict) -> np.ndarray:
        """Convert response to point in feature space"""
        features = []
        
        # Safety score
        features.append(response.get('safety_score', 0.5))
        
        # Action type encoding (one-hot)
        action_types = ['heading', 'altitude', 'speed', 'vector', 'hold']
        action = response.get('type', '').lower()
        for atype in action_types:
            features.append(1.0 if atype in action else 0.0)
        
        # Numerical parameters
        features.append(response.get('heading_change', 0) / 180.0)  # Normalize
        features.append(response.get('altitude_change', 0) / 2000.0)  # Normalize
        features.append(response.get('speed_change', 0) / 100.0)  # Normalize
        
        return np.array(features)
    
    def _calculate_dispersion(self, points: np.ndarray) -> float:
        """Calculate dispersion score using convex hull volume"""
        try:
            if points.shape[0] < points.shape[1] + 1:
                # Not enough points for convex hull in this dimension
                return self._calculate_alternative_dispersion(points)
            
            # Calculate convex hull
            hull = ConvexHull(points)
            
            # Dispersion based on hull volume
            volume = hull.volume
            
            # Normalize volume (approximate)
            max_possible_volume = 1.0  # Rough estimate for normalized features
            normalized_volume = min(volume / max_possible_volume, 1.0)
            
            return float(normalized_volume)
            
        except (ValueError, np.linalg.LinAlgError, AttributeError) as e:
            logging.warning(f"Convex hull dispersion calculation failed: {e}")
            return self._calculate_alternative_dispersion(points)
    
    def _calculate_alternative_dispersion(self, points: np.ndarray) -> float:
        """Alternative dispersion calculation when convex hull fails"""
        if len(points) < 2:
            return 0.0
        
        # Use pairwise distances
        distances = cdist(points, points)
        mean_distance = np.mean(distances)
        
        # Normalize by maximum possible distance in unit hypercube
        max_distance = np.sqrt(points.shape[1])
        normalized_dispersion = mean_distance / max_distance
        
        return float(min(normalized_dispersion, 1.0))
    
    def _calculate_point_spread(self, points: np.ndarray) -> Dict:
        """Calculate point spread statistics"""
        if len(points) == 0:
            return {'error': 'no_points'}
        
        return {
            'mean': points.mean(axis=0).tolist(),
            'std': points.std(axis=0).tolist(),
            'range': (points.max(axis=0) - points.min(axis=0)).tolist()
        }

# ==================== LAYER 3 DETECTORS ====================

class SelfConsistencyValidator:
    """Self-consistency validation across multiple model calls"""
    
    def __init__(self):
        self.consistency_threshold = 0.8
        
    def validate_consistency(self, responses: List[Dict]) -> Tuple[float, Dict]:
        """Validate self-consistency across multiple responses"""
        try:
            if len(responses) < 2:
                return 1.0, {'note': 'single_response'}
            
            # Calculate consistency metrics
            consistency_score = self._calculate_consistency_score(responses)
            consistency_details = self._analyze_consistency_details(responses)
            
            evidence = {
                'consistency_score': consistency_score,
                'response_count': len(responses),
                'consistency_details': consistency_details
            }
            
            return consistency_score, evidence
            
        except Exception as e:
            logging.error(f"Self-consistency validation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_consistency_score(self, responses: List[Dict]) -> float:
        """Calculate overall consistency score"""
        if len(responses) < 2:
            return 1.0
        
        # Compare all pairs of responses
        consistency_scores = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pair_consistency = self._compare_responses(responses[i], responses[j])
                consistency_scores.append(pair_consistency)
        
        return float(np.mean(consistency_scores))
    
    def _compare_responses(self, resp1: Dict, resp2: Dict) -> float:
        """Compare two responses for consistency"""
        score = 0.0
        comparisons = 0
        
        # Safety score consistency
        if 'safety_score' in resp1 and 'safety_score' in resp2:
            safety_diff = abs(resp1['safety_score'] - resp2['safety_score'])
            score += 1.0 - safety_diff  # Higher score for smaller difference
            comparisons += 1
        
        # Action type consistency
        if 'type' in resp1 and 'type' in resp2:
            type_match = 1.0 if resp1['type'] == resp2['type'] else 0.0
            score += type_match
            comparisons += 1
        
        # Action consistency
        if 'action' in resp1 and 'action' in resp2:
            action_similarity = self._calculate_action_similarity(resp1['action'], resp2['action'])
            score += action_similarity
            comparisons += 1
        
        return score / max(comparisons, 1)
    
    def _calculate_action_similarity(self, action1: str, action2: str) -> float:
        """Calculate similarity between action descriptions"""
        if action1 == action2:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(action1.lower().split())
        words2 = set(action2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / max(union, 1)
    
    def _analyze_consistency_details(self, responses: List[Dict]) -> Dict:
        """Analyze detailed consistency patterns"""
        details = {
            'safety_score_variance': 0.0,
            'action_type_agreement': 0.0,
            'response_diversity': 0.0
        }
        
        # Safety score variance
        safety_scores = [r.get('safety_score', 0.5) for r in responses]
        details['safety_score_variance'] = float(np.var(safety_scores))
        
        # Action type agreement
        action_types = [r.get('type', '') for r in responses]
        unique_types = len(set(action_types))
        details['action_type_agreement'] = 1.0 - (unique_types - 1) / max(len(action_types) - 1, 1)
        
        # Response diversity
        unique_responses = len(set(str(r) for r in responses))
        details['response_diversity'] = unique_responses / len(responses)
        
        return details

class MultiModelConsensusChecker:
    """Multi-model consensus analysis for hallucination detection"""
    
    def __init__(self):
        self.consensus_threshold = 0.75
        
    def check_consensus(self, model_responses: Dict[str, Dict]) -> Tuple[float, Dict]:
        """Check consensus across multiple model responses"""
        try:
            if len(model_responses) < 2:
                return 1.0, {'note': 'insufficient_models'}
            
            # Calculate consensus metrics
            consensus_score = self._calculate_consensus_score(model_responses)
            consensus_details = self._analyze_consensus_details(model_responses)
            
            evidence = {
                'consensus_score': consensus_score,
                'model_count': len(model_responses),
                'consensus_details': consensus_details,
                'participating_models': list(model_responses.keys())
            }
            
            return consensus_score, evidence
            
        except Exception as e:
            logging.error(f"Multi-model consensus check failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_consensus_score(self, model_responses: Dict[str, Dict]) -> float:
        """Calculate consensus score across models"""
        responses = list(model_responses.values())
        
        # Filter out error responses
        valid_responses = [r for r in responses if 'error' not in r]
        
        if len(valid_responses) < 2:
            return 0.0
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(valid_responses)):
            for j in range(i + 1, len(valid_responses)):
                agreement = self._calculate_agreement(valid_responses[i], valid_responses[j])
                agreements.append(agreement)
        
        return float(np.mean(agreements))
    
    def _calculate_agreement(self, resp1: Dict, resp2: Dict) -> float:
        """Calculate agreement between two model responses"""
        agreement_score = 0.0
        metrics = 0
        
        # Safety score agreement
        if 'safety_score' in resp1 and 'safety_score' in resp2:
            safety_diff = abs(resp1['safety_score'] - resp2['safety_score'])
            safety_agreement = max(0, 1.0 - safety_diff)
            agreement_score += safety_agreement
            metrics += 1
        
        # Action type agreement
        if 'type' in resp1 and 'type' in resp2:
            type_agreement = 1.0 if resp1['type'] == resp2['type'] else 0.0
            agreement_score += type_agreement
            metrics += 1
        
        # Numerical parameter agreement
        for param in ['heading_change', 'altitude_change', 'speed_change']:
            if param in resp1 and param in resp2:
                param_diff = abs(resp1[param] - resp2[param])
                # Normalize by expected range
                ranges = {'heading_change': 180, 'altitude_change': 2000, 'speed_change': 100}
                normalized_diff = param_diff / ranges.get(param, 100)
                param_agreement = max(0, 1.0 - normalized_diff)
                agreement_score += param_agreement
                metrics += 1
        
        return agreement_score / max(metrics, 1)
    
    def _analyze_consensus_details(self, model_responses: Dict[str, Dict]) -> Dict:
        """Analyze detailed consensus patterns"""
        valid_responses = {k: v for k, v in model_responses.items() if 'error' not in v}
        
        details = {
            'valid_model_count': len(valid_responses),
            'error_model_count': len(model_responses) - len(valid_responses),
            'safety_score_consensus': self._analyze_safety_consensus(valid_responses),
            'action_type_consensus': self._analyze_action_consensus(valid_responses)
        }
        
        return details
    
    def _analyze_safety_consensus(self, responses: Dict[str, Dict]) -> Dict:
        """Analyze safety score consensus"""
        safety_scores = [r.get('safety_score', 0.5) for r in responses.values()]
        
        return {
            'mean': float(np.mean(safety_scores)),
            'std': float(np.std(safety_scores)),
            'range': float(np.max(safety_scores) - np.min(safety_scores))
        }
    
    def _analyze_action_consensus(self, responses: Dict[str, Dict]) -> Dict:
        """Analyze action type consensus"""
        action_types = [r.get('type', '') for r in responses.values()]
        type_counts = {}
        
        for action_type in action_types:
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
        
        majority_type = max(type_counts, key=type_counts.get) if type_counts else ''
        majority_count = type_counts.get(majority_type, 0)
        
        return {
            'type_distribution': type_counts,
            'majority_type': majority_type,
            'majority_ratio': majority_count / max(len(action_types), 1)
        }

class RetrievalAugmentedValidator:
    """RAG-based validation against aviation knowledge base"""
    
    def __init__(self):
        self.knowledge_base = self._initialize_aviation_knowledge()
        
    def validate_response(self, response: Dict, context: Dict) -> Tuple[float, Dict]:
        """Validate response against aviation knowledge base"""
        try:
            validation_score = 1.0
            violations = []
            warnings = []
            
            # Validate against aviation rules
            rule_validation = self._validate_aviation_rules(response, context)
            validation_score *= rule_validation['score']
            violations.extend(rule_validation['violations'])
            warnings.extend(rule_validation['warnings'])
            
            # Validate technical constraints
            tech_validation = self._validate_technical_constraints(response, context)
            validation_score *= tech_validation['score']
            violations.extend(tech_validation['violations'])
            warnings.extend(tech_validation['warnings'])
            
            # Validate safety principles
            safety_validation = self._validate_safety_principles(response, context)
            validation_score *= safety_validation['score']
            violations.extend(safety_validation['violations'])
            warnings.extend(safety_validation['warnings'])
            
            evidence = {
                'validation_score': validation_score,
                'violations': violations,
                'warnings': warnings,
                'rule_validation': rule_validation,
                'tech_validation': tech_validation,
                'safety_validation': safety_validation
            }
            
            return validation_score, evidence
            
        except Exception as e:
            logging.error(f"RAG validation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _initialize_aviation_knowledge(self) -> Dict:
        """Initialize aviation knowledge base"""
        return {
            'separation_standards': {
                'horizontal_min': 5.0,  # nautical miles
                'vertical_min': 1000,   # feet
                'time_min': 60         # seconds
            },
            'maneuver_constraints': {
                'heading_change': {'min': -30, 'max': 30, 'typical': 15},
                'altitude_change': {'min': -2000, 'max': 2000, 'typical': 1000},
                'speed_change': {'min': -50, 'max': 50, 'typical': 20}
            },
            'safety_principles': [
                'maintain_separation',
                'minimize_disruption',
                'ensure_feasibility',
                'prioritize_safety'
            ]
        }
    
    def _validate_aviation_rules(self, response: Dict, context: Dict) -> Dict:
        """Validate against aviation rules"""
        score = 1.0
        violations = []
        warnings = []
        
        # Check separation maintenance
        if context.get('distance', 10) < self.knowledge_base['separation_standards']['horizontal_min']:
            if response.get('safety_score', 1.0) < 0.8:
                violations.append("Insufficient safety score for close proximity conflict")
                score *= 0.5
        
        return {'score': score, 'violations': violations, 'warnings': warnings}
    
    def _validate_technical_constraints(self, response: Dict, context: Dict) -> Dict:
        """Validate technical feasibility"""
        score = 1.0
        violations = []
        warnings = []
        
        maneuver_type = response.get('type', '')
        constraints = self.knowledge_base['maneuver_constraints']
        
        for param, limits in constraints.items():
            if param in response:
                value = response[param]
                if not (limits['min'] <= value <= limits['max']):
                    violations.append(f"{param} {value} outside limits {limits['min']}-{limits['max']}")
                    score *= 0.7
        
        return {'score': score, 'violations': violations, 'warnings': warnings}
    
    def _validate_safety_principles(self, response: Dict, context: Dict) -> Dict:
        """Validate against safety principles"""
        score = 1.0
        violations = []
        warnings = []
        
        # Check safety score adequacy
        safety_score = response.get('safety_score', 0.5)
        if safety_score < 0.3:
            violations.append("Safety score below acceptable threshold")
            score *= 0.3
        elif safety_score < 0.5:
            warnings.append("Safety score below recommended threshold")
            score *= 0.8
        
        return {'score': score, 'violations': violations, 'warnings': warnings}

class AttentionPatternDetector:
    """Detects anomalous attention patterns indicative of hallucination"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        
    def analyze_attention_coherence(self, response_tokens: List[str], 
                                  context_tokens: List[str]) -> Tuple[float, Dict]:
        """Analyze attention pattern coherence"""
        try:
            # Simulate attention weights based on token overlap and similarity
            attention_scores = []
            
            for resp_token in response_tokens:
                token_attention = []
                for ctx_token in context_tokens:
                    # Simple similarity metric (in practice, use actual attention weights)
                    similarity = 1.0 if resp_token.lower() == ctx_token.lower() else 0.0
                    token_attention.append(similarity)
                attention_scores.append(token_attention)
            
            attention_matrix = np.array(attention_scores) if attention_scores else np.array([[0]])
            
            # Compute attention statistics
            attention_variance = np.var(attention_matrix)
            attention_entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-10))
            attention_sparsity = np.count_nonzero(attention_matrix) / attention_matrix.size
            
            # Detect anomalies
            features = np.array([[attention_variance, attention_entropy, attention_sparsity]])
            
            if not self.is_fitted and len(features) > 1:
                self.anomaly_detector.fit(features)
                self.is_fitted = True
                anomaly_score = 0.0
            elif self.is_fitted:
                anomaly_score = self.anomaly_detector.decision_function(features)[0]
            else:
                anomaly_score = 0.0
            
            return float(anomaly_score), {
                'variance': float(attention_variance),
                'entropy': float(attention_entropy),
                'sparsity': float(attention_sparsity)
            }
            
        except Exception as e:
            logging.warning(f"Attention analysis failed: {e}")
            return 0.0, {}

class EigenvalueAnalyzer:
    """Analyzes eigenvalue spectrum for hallucination detection"""
    
    def __init__(self):
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
    def analyze_representation_space(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Analyze the eigenvalue spectrum of embeddings"""
        try:
            if embeddings.shape[0] < 2:
                return {'spectral_anomaly': 0.0, 'rank_deficit': 0.0}
                
            # Compute PCA
            self.pca.fit(embeddings)
            eigenvalues = self.pca.explained_variance_
            
            # Analyze eigenvalue distribution
            eigenvalue_ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
            effective_rank = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
            
            # Detect anomalies
            spectral_anomaly = min(eigenvalue_ratio / 100.0, 1.0)  # Normalize
            rank_deficit = 1.0 - (effective_rank / len(eigenvalues))
            
            return {
                'spectral_anomaly': float(spectral_anomaly),
                'rank_deficit': float(rank_deficit),
                'spectral_entropy': float(spectral_entropy),
                'effective_rank': float(effective_rank)
            }
            
        except Exception as e:
            logging.warning(f"Eigenvalue analysis failed: {e}")
            return {'spectral_anomaly': 0.0, 'rank_deficit': 0.0}

class SemanticEntropyCalculator:
    """Calculates semantic entropy for uncertainty quantification"""
    
    def __init__(self):
        self.sentence_transformer = None
        self._load_model()
        
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.warning(f"Failed to load sentence transformer: {e}")
            
    def calculate_semantic_entropy(self, response_variants: List[str]) -> float:
        """Calculate semantic entropy across response variants"""
        try:
            if not self.sentence_transformer or len(response_variants) < 2:
                return 0.5  # Default moderate entropy
                
            # Get embeddings
            embeddings = self.sentence_transformer.encode(response_variants)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            # Convert similarities to probabilities and calculate entropy
            if similarities:
                similarities = np.array(similarities)
                probs = (similarities + 1) / 2  # Normalize to [0,1]
                probs = probs / np.sum(probs)  # Normalize to probability distribution
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                return float(entropy)
            else:
                return 0.5
                
        except Exception as e:
            logging.warning(f"Semantic entropy calculation failed: {e}")
            return 0.5

class PredictiveUncertaintyAnalyzer:
    """Analyzes predictive uncertainty for hallucination detection"""
    
    def __init__(self):
        self.uncertainty_threshold = 0.3
        
    def analyze_prediction_confidence(self, response_data: Dict) -> Dict[str, float]:
        """Analyze confidence metrics in model predictions"""
        try:
            # Extract confidence indicators from response
            safety_score = response_data.get('safety_score', 0.5)
            
            # Simulate confidence metrics (in practice, extract from model)
            epistemic_uncertainty = abs(0.5 - safety_score) * 2  # Distance from neutral
            aleatoric_uncertainty = 0.1  # Assume low data uncertainty for ATC
            
            # Combined uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # Confidence metrics
            confidence = 1.0 - total_uncertainty
            uncertainty_ratio = epistemic_uncertainty / (aleatoric_uncertainty + 1e-10)
            
            return {
                'epistemic_uncertainty': float(epistemic_uncertainty),
                'aleatoric_uncertainty': float(aleatoric_uncertainty),
                'total_uncertainty': float(total_uncertainty),
                'confidence': float(confidence),
                'uncertainty_ratio': float(uncertainty_ratio)
            }
            
        except Exception as e:
            logging.warning(f"Uncertainty analysis failed: {e}")
            return {'confidence': 0.5, 'total_uncertainty': 0.5}

class ConvexHullDispersionMeasure:
    """Measures dispersion in embedding space using convex hull analysis"""
    
    def compute_dispersion(self, embeddings: np.ndarray) -> float:
        """Compute embedding dispersion using convex hull volume"""
        try:
            if embeddings.shape[0] < 3 or embeddings.shape[1] < 2:
                return 0.5  # Default moderate dispersion
                
            from scipy.spatial import ConvexHull
            
            # Reduce dimensionality for hull computation
            if embeddings.shape[1] > 10:
                pca = PCA(n_components=10)
                embeddings = pca.fit_transform(embeddings)
            
            # Compute convex hull
            hull = ConvexHull(embeddings)
            
            # Normalize volume by number of points and dimensions
            normalized_volume = hull.volume / (embeddings.shape[0] * embeddings.shape[1])
            
            # Convert to dispersion score [0,1]
            dispersion = min(normalized_volume * 100, 1.0)
            
            return float(dispersion)
            
        except Exception as e:
            logging.warning(f"Convex hull analysis failed: {e}")
            return 0.5

class SelfConsistencyValidator:
    """Validates self-consistency across multiple model calls"""
    
    def __init__(self):
        self.consistency_threshold = 0.8
        
    def validate_consistency(self, responses: List[Dict]) -> Dict[str, float]:
        """Validate consistency across multiple responses"""
        try:
            if len(responses) < 2:
                return {'consistency_score': 1.0}
                
            # Check safety score consistency
            safety_scores = [r.get('safety_score', 0.5) for r in responses]
            safety_variance = np.var(safety_scores)
            
            # Check action type consistency
            action_types = [r.get('type', 'unknown') for r in responses]
            type_consistency = len(set(action_types)) / len(action_types)
            
            # Combined consistency score
            consistency_score = 1.0 - min(safety_variance * 4, 1.0)  # Scale variance
            type_score = 1.0 - type_consistency + (1.0 / len(action_types))
            
            overall_consistency = (consistency_score + type_score) / 2
            
            return {
                'consistency_score': float(overall_consistency),
                'safety_variance': float(safety_variance),
                'type_consistency': float(type_score)
            }
            
        except Exception as e:
            logging.warning(f"Consistency validation failed: {e}")
            return {'consistency_score': 0.5}

class MultiModelConsensusChecker:
    """Checks consensus across multiple LLM models"""
    
    def __init__(self):
        self.consensus_threshold = 0.7
        
    def check_consensus(self, model_responses: Dict[str, Dict]) -> Dict[str, float]:
        """Check consensus across different model responses"""
        try:
            if len(model_responses) < 2:
                return {'consensus_score': 1.0}
                
            # Extract key decisions
            safety_scores = [r.get('safety_score', 0.5) for r in model_responses.values()]
            action_types = [r.get('type', 'unknown') for r in model_responses.values()]
            
            # Calculate consensus metrics
            safety_consensus = 1.0 - np.std(safety_scores)
            
            # Action type consensus (fraction agreeing with majority)
            type_counts = {}
            for action_type in action_types:
                type_counts[action_type] = type_counts.get(action_type, 0) + 1
            
            majority_type = max(type_counts, key=type_counts.get)
            type_consensus = type_counts[majority_type] / len(action_types)
            
            overall_consensus = (safety_consensus + type_consensus) / 2
            
            return {
                'consensus_score': float(overall_consensus),
                'safety_consensus': float(safety_consensus),
                'type_consensus': float(type_consensus)
            }
            
        except Exception as e:
            logging.warning(f"Consensus checking failed: {e}")
            return {'consensus_score': 0.5}

class RetrievalAugmentedValidator:
    """Validates responses against retrieved aviation knowledge"""
    
    def __init__(self):
        self.knowledge_base = self._load_aviation_knowledge()
        
    def _load_aviation_knowledge(self) -> Dict[str, Any]:
        """Load aviation knowledge base"""
        # Simplified knowledge base
        return {
            'separation_minimums': {
                'horizontal': 5.0,  # nautical miles
                'vertical': 1000,   # feet
            },
            'valid_maneuvers': [
                'heading_change', 'altitude_change', 'speed_change',
                'vector', 'hold', 'climb', 'descend'
            ],
            'altitude_limits': {'min': 1000, 'max': 50000},
            'speed_limits': {'min': 100, 'max': 600}
        }
        
    def validate_against_knowledge(self, response: Dict) -> Dict[str, float]:
        """Validate response against aviation knowledge base"""
        try:
            validation_score = 1.0
            violations = []
            
            # Check maneuver validity
            maneuver_type = response.get('type', '')
            if maneuver_type not in self.knowledge_base['valid_maneuvers']:
                validation_score -= 0.3
                violations.append(f"Invalid maneuver type: {maneuver_type}")
            
            # Check altitude constraints
            if 'altitude' in response:
                alt = response['altitude']
                limits = self.knowledge_base['altitude_limits']
                if alt < limits['min'] or alt > limits['max']:
                    validation_score -= 0.4
                    violations.append(f"Altitude {alt} outside limits")
            
            # Check speed constraints
            if 'speed' in response:
                speed = response['speed']
                limits = self.knowledge_base['speed_limits']
                if speed < limits['min'] or speed > limits['max']:
                    validation_score -= 0.3
                    violations.append(f"Speed {speed} outside limits")
            
            validation_score = max(validation_score, 0.0)
            
            return {
                'validation_score': float(validation_score),
                'violations': violations,
                'knowledge_compliance': float(validation_score)
            }
            
        except Exception as e:
            logging.warning(f"Knowledge validation failed: {e}")
            return {'validation_score': 0.5, 'violations': []}

class EnhancedHallucinationDetector:
    """Enhanced multi-layer hallucination detector integrating all detection methods"""
    
    def __init__(self):
        # Initialize all detection layers
        self.mind_framework = MINDFramework()
        self.attention_detector = AttentionPatternDetector()
        self.eigenvalue_analyzer = EigenvalueAnalyzer()
        self.semantic_entropy_calc = SemanticEntropyCalculator()
        self.uncertainty_analyzer = PredictiveUncertaintyAnalyzer()
        self.dispersion_measure = ConvexHullDispersionMeasure()
        self.consistency_validator = SelfConsistencyValidator()
        self.consensus_checker = MultiModelConsensusChecker()
        self.rag_validator = RetrievalAugmentedValidator()
        
        # Detection thresholds
        self.thresholds = {
            'fabrication': 0.6,
            'omission': 0.5,
            'irrelevancy': 0.7,
            'contradiction': 0.5,
            'semantic_drift': 0.4,
            'uncertainty_collapse': 0.8
        }
        
    def detect_hallucinations(self, 
                            llm_response: Dict,
                            baseline_response: Dict,
                            conflict_context: Dict,
                            response_variants: Optional[List[str]] = None,
                            model_responses: Optional[Dict[str, Dict]] = None) -> HallucinationResult:
        """
        Comprehensive hallucination detection using all available methods
        """
        start_time = time.time()
        
        try:
            # Initialize detection results
            detected_types = []
            evidence = {}
            confidence_scores = []
            
            # Layer 1: Basic structural detection
            basic_issues = self._detect_basic_hallucinations(llm_response, baseline_response, conflict_context)
            if basic_issues['detected']:
                detected_types.extend(basic_issues['types'])
                evidence.update(basic_issues['evidence'])
                confidence_scores.append(basic_issues['confidence'])
            
            # Layer 2: Advanced semantic analysis
            if response_variants:
                semantic_issues = self._detect_semantic_issues(response_variants)
                if semantic_issues['detected']:
                    detected_types.extend(semantic_issues['types'])
                    evidence.update(semantic_issues['evidence'])
                    confidence_scores.append(semantic_issues['confidence'])
            
            # Layer 3: Multi-model consensus
            if model_responses:
                consensus_issues = self._detect_consensus_issues(model_responses)
                if consensus_issues['detected']:
                    detected_types.extend(consensus_issues['types'])
                    evidence.update(consensus_issues['evidence'])
                    confidence_scores.append(consensus_issues['confidence'])
            
            # Layer 4: Knowledge validation
            knowledge_issues = self._validate_knowledge_compliance(llm_response)
            if knowledge_issues['detected']:
                detected_types.extend(knowledge_issues['types'])
                evidence.update(knowledge_issues['evidence'])
                confidence_scores.append(knowledge_issues['confidence'])
            
            # Combine results
            overall_detected = len(detected_types) > 0
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            safety_impact = self._assess_safety_impact(detected_types, evidence)
            
            # Calculate layer scores
            layer_scores = {
                'layer1_basic': basic_issues.get('confidence', 0.0) if basic_issues else 0.0,
                'layer2_semantic': semantic_issues.get('confidence', 0.0) if 'semantic_issues' in locals() else 0.0,
                'layer3_consensus': consensus_issues.get('confidence', 0.0) if 'consensus_issues' in locals() else 0.0,
                'layer4_knowledge': knowledge_issues.get('confidence', 0.0) if knowledge_issues else 0.0
            }
            
            # Remove duplicates
            unique_types = list(set(detected_types))
            
            detection_time = time.time() - start_time
            evidence['detection_time_ms'] = detection_time * 1000
            
            return HallucinationResult(
                detected=overall_detected,
                types=unique_types,
                confidence=float(overall_confidence),
                evidence=evidence,
                safety_impact=safety_impact,
                layer_scores=layer_scores
            )
            
        except Exception as e:
            logging.error(f"Hallucination detection failed: {e}")
            return HallucinationResult(
                detected=False,
                types=[],
                confidence=0.0,
                evidence={'error': str(e)},
                safety_impact='minimal',
                layer_scores={}
            )
    
    def _detect_basic_hallucinations(self, llm_response: Dict, baseline_response: Dict, 
                                   conflict_context: Dict) -> Dict[str, Any]:
        """Basic hallucination detection (fabrication, omission, irrelevancy, contradiction)"""
        issues = []
        evidence = {}
        
        # Check fabrication
        if self._check_fabrication(llm_response, conflict_context):
            issues.append(HallucinationType.FABRICATION)
            evidence['fabrication_details'] = "Invalid parameters or maneuver types detected"
        
        # Check omission
        if self._check_omission(llm_response):
            issues.append(HallucinationType.OMISSION)
            evidence['omission_details'] = "Required safety parameters missing"
        
        # Check irrelevancy
        if self._check_irrelevancy(llm_response, conflict_context):
            issues.append(HallucinationType.IRRELEVANCY)
            evidence['irrelevancy_details'] = "Response doesn't address conflict scenario"
        
        # Check contradiction
        if self._check_contradiction(llm_response, baseline_response):
            issues.append(HallucinationType.CONTRADICTION)
            evidence['contradiction_details'] = "Response contradicts baseline or physics"
        
        return {
            'detected': len(issues) > 0,
            'types': issues,
            'confidence': 0.8 if issues else 0.0,
            'evidence': evidence
        }
    
    def _detect_semantic_issues(self, response_variants: List[str]) -> Dict[str, Any]:
        """Detect semantic drift and consistency issues"""
        issues = []
        evidence = {}
        
        # Calculate semantic entropy
        semantic_entropy = self.semantic_entropy_calc.calculate_semantic_entropy(response_variants)
        
        if semantic_entropy > 0.7:  # High entropy indicates semantic drift
            issues.append(HallucinationType.SEMANTIC_DRIFT)
            evidence['semantic_entropy'] = semantic_entropy
        
        return {
            'detected': len(issues) > 0,
            'types': issues,
            'confidence': min(semantic_entropy, 1.0),
            'evidence': evidence
        }
    
    def _detect_consensus_issues(self, model_responses: Dict[str, Dict]) -> Dict[str, Any]:
        """Detect issues based on multi-model consensus"""
        issues = []
        evidence = {}
        
        consensus_results = self.consensus_checker.check_consensus(model_responses)
        consensus_score = consensus_results['consensus_score']
        
        if consensus_score < 0.5:  # Low consensus indicates potential hallucination
            issues.append(HallucinationType.UNCERTAINTY_COLLAPSE)
            evidence['consensus_analysis'] = consensus_results
        
        return {
            'detected': len(issues) > 0,
            'types': issues,
            'confidence': 1.0 - consensus_score,
            'evidence': evidence
        }
    
    def _validate_knowledge_compliance(self, llm_response: Dict) -> Dict[str, Any]:
        """Validate response against aviation knowledge base"""
        issues = []
        evidence = {}
        
        validation_results = self.rag_validator.validate_against_knowledge(llm_response)
        validation_score = validation_results['validation_score']
        
        if validation_score < 0.7:  # Low validation score indicates fabrication
            issues.append(HallucinationType.FABRICATION)
            evidence['knowledge_validation'] = validation_results
        
        return {
            'detected': len(issues) > 0,
            'types': issues,
            'confidence': 1.0 - validation_score,
            'evidence': evidence
        }
    
    def _check_fabrication(self, response: Dict, context: Dict) -> bool:
        """Basic fabrication check"""
        if not isinstance(response, dict):
            return True
        
        # Check basic validity
        maneuver_type = response.get('type', '')
        valid_types = ['heading', 'altitude', 'speed', 'vector', 'hold']
        
        return maneuver_type not in valid_types
    
    def _check_omission(self, response: Dict) -> bool:
        """Basic omission check"""
        required_fields = ['action', 'type', 'safety_score']
        return not all(field in response for field in required_fields)
    
    def _check_irrelevancy(self, response: Dict, context: Dict) -> bool:
        """Basic irrelevancy check"""
        # Simple check: if response targets wrong aircraft
        target_aircraft = response.get('aircraft_id', '')
        valid_aircraft = [context.get('id1', ''), context.get('id2', '')]
        
        return target_aircraft and target_aircraft not in valid_aircraft
    
    def _check_contradiction(self, llm_response: Dict, baseline_response: Dict) -> bool:
        """Basic contradiction check"""
        if not isinstance(llm_response, dict) or not isinstance(baseline_response, dict):
            return False
        
        llm_safety = llm_response.get('safety_score', 0.5)
        baseline_safety = baseline_response.get('safety_score', 0.5)
        
        # Significant safety score difference indicates contradiction
        return abs(llm_safety - baseline_safety) > 0.4
    
    def _assess_safety_impact(self, hallucination_types: List[HallucinationType], 
                            evidence: Dict[str, Any]) -> str:
        """Assess safety impact of detected hallucinations"""
        if not hallucination_types:
            return 'minimal'
        
        critical_types = [HallucinationType.FABRICATION, HallucinationType.CONTRADICTION]
        moderate_types = [HallucinationType.OMISSION, HallucinationType.UNCERTAINTY_COLLAPSE]
        
        if any(h_type in critical_types for h_type in hallucination_types):
            return 'critical'
        elif any(h_type in moderate_types for h_type in hallucination_types):
            return 'moderate'
        else:
            return 'minimal'

# Factory function for easy instantiation
def create_enhanced_detector() -> EnhancedHallucinationDetector:
    """Create and return an enhanced hallucination detector"""
    return EnhancedHallucinationDetector()

if __name__ == "__main__":
    # Test the enhanced detection system
    detector = create_enhanced_detector()
    
    # Example test case
    llm_response = {
        'action': 'turn left 10 degrees',
        'type': 'heading',
        'safety_score': 0.8,
        'aircraft_id': 'AC001'
    }
    
    baseline_response = {
        'action': 'climb 1000 ft',
        'type': 'altitude',
        'safety_score': 0.9,
        'aircraft_id': 'AC001'
    }
    
    conflict_context = {
        'id1': 'AC001',
        'id2': 'AC002',
        'distance': 4.5,
        'time': 120
    }
    
    result = detector.detect_hallucinations(llm_response, baseline_response, conflict_context)
    
    print("Enhanced Hallucination Detection Test:")
    print(f"Detected: {result.detected}")
    print(f"Types: {[t.value for t in result.types]}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Safety Impact: {result.safety_impact}")
    print(f"Evidence: {json.dumps(result.evidence, indent=2)}")
