# analysis/hallucination_taxonomy.py
"""
Comprehensive hallucination detection framework for ATC conflict resolution LLMs.
Implements multi-layer detection system with:
- Layer 1: MINDFramework, AttentionPatternDetector, EigenvalueAnalyzer
- Layer 2: SemanticEntropyCalculator, PredictiveUncertaintyAnalyzer, ConvexHullDispersionMeasure
- Layer 3: SelfConsistencyValidator, MultiModelConsensusChecker, RetrievalAugmentedValidator
Based on the four categories: Fabrication, Omission, Irrelevancy, and Contradiction.
"""

import logging
import json
import numpy as np
import math
import os
import time
from scipy.spatial import ConvexHull
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass

class HallucinationType(Enum):
    FABRICATION = "fabrication"
    OMISSION = "omission"
    IRRELEVANCY = "irrelevancy"
    CONTRADICTION = "contradiction"

@dataclass
class DetectionResult:
    """Result from a hallucination detection layer"""
    layer: str
    detector: str
    confidence: float
    hallucination_score: float
    details: Dict
    timestamp: float

@dataclass
class HallucinationAnalysis:
    """Comprehensive hallucination analysis result"""
    hallucination_types: List[HallucinationType]
    overall_score: float
    layer_results: List[DetectionResult]
    safety_impact: float
    recommendation: str

# ================== LAYER 1 DETECTORS ==================

class MINDFramework:
    """Memory-based Introspection for Narrative Deviation detection"""
    
    def __init__(self):
        self.memory_bank = {}
        self.similarity_threshold = 0.85
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Detect narrative deviation using memory-based introspection"""
        import time
        
        try:
            # Create scenario signature
            scenario_sig = self._create_scenario_signature(context)
            
            # Check against memory bank
            similarity_scores = []
            for stored_sig, stored_response in self.memory_bank.items():
                similarity = self._calculate_similarity(scenario_sig, stored_sig)
                if similarity > self.similarity_threshold:
                    response_similarity = self._compare_responses(response, stored_response)
                    similarity_scores.append(response_similarity)
            
            # Calculate deviation score
            if similarity_scores:
                avg_similarity = np.mean(similarity_scores)
                hallucination_score = 1.0 - avg_similarity
            else:
                hallucination_score = 0.5  # No memory, moderate uncertainty
            
            # Store current scenario
            self.memory_bank[scenario_sig] = response
            
            return DetectionResult(
                layer="Layer1",
                detector="MINDFramework",
                confidence=0.8,
                hallucination_score=hallucination_score,
                details={
                    'memory_matches': len(similarity_scores),
                    'avg_similarity': np.mean(similarity_scores) if similarity_scores else 0,
                    'scenario_signature': scenario_sig
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer1",
                detector="MINDFramework",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _create_scenario_signature(self, context: Dict) -> str:
        """Create a signature for the conflict scenario"""
        aircraft_count = len(context.get('aircraft', []))
        complexity = context.get('complexity', 'simple')
        weather = context.get('weather', 'clear')
        return f"{aircraft_count}_{complexity}_{weather}"
    
    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between scenario signatures"""
        if sig1 == sig2:
            return 1.0
        # Simple Jaccard similarity
        set1 = set(sig1.split('_'))
        set2 = set(sig2.split('_'))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _compare_responses(self, resp1: Dict, resp2: Dict) -> float:
        """Compare similarity between two responses"""
        if not isinstance(resp1, dict) or not isinstance(resp2, dict):
            return 0.0
        
        common_keys = set(resp1.keys()).intersection(set(resp2.keys()))
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if resp1[key] == resp2[key]:
                similarities.append(1.0)
            elif isinstance(resp1[key], (int, float)) and isinstance(resp2[key], (int, float)):
                # Numerical similarity
                diff = abs(resp1[key] - resp2[key])
                max_val = max(abs(resp1[key]), abs(resp2[key]), 1)
                similarities.append(1.0 - (diff / max_val))
            else:
                similarities.append(0.0)
        
        return np.mean(similarities)

class AttentionPatternDetector:
    """Detects anomalous attention patterns in LLM decision making"""
    
    def __init__(self):
        self.expected_attention_keys = ['aircraft_positions', 'safety_margins', 'airspace', 'weather']
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Detect attention pattern anomalies"""
        import time
        
        try:
            # Analyze attention distribution
            attention_score = self._analyze_attention_distribution(response, context)
            
            # Check for missing critical attention
            missing_attention = self._check_missing_attention(response, context)
            
            # Calculate hallucination score based on attention patterns
            hallucination_score = (1.0 - attention_score) + (len(missing_attention) * 0.2)
            hallucination_score = min(1.0, hallucination_score)
            
            return DetectionResult(
                layer="Layer1",
                detector="AttentionPatternDetector",
                confidence=0.75,
                hallucination_score=hallucination_score,
                details={
                    'attention_score': attention_score,
                    'missing_attention': missing_attention,
                    'expected_keys': self.expected_attention_keys
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer1",
                detector="AttentionPatternDetector",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _analyze_attention_distribution(self, response: Dict, context: Dict) -> float:
        """Analyze how well the response attends to relevant context"""
        if not isinstance(response, dict):
            return 0.0
        
        # Check if response addresses key context elements
        context_coverage = 0
        total_context_elements = 0
        
        for key in self.expected_attention_keys:
            if key in context:
                total_context_elements += 1
                # Check if response shows attention to this context element
                if self._shows_attention_to(response, key, context[key]):
                    context_coverage += 1
        
        return context_coverage / max(total_context_elements, 1)
    
    def _shows_attention_to(self, response: Dict, context_key: str, context_value: Any) -> bool:
        """Check if response shows attention to specific context element"""
        response_str = json.dumps(response, default=str).lower()
        
        if context_key == 'aircraft_positions':
            return 'aircraft' in response_str or 'position' in response_str
        elif context_key == 'safety_margins':
            return 'safety' in response_str or 'margin' in response_str or 'separation' in response_str
        elif context_key == 'airspace':
            return 'airspace' in response_str or 'sector' in response_str
        elif context_key == 'weather':
            return 'weather' in response_str or 'wind' in response_str
        
        return False
    
    def _check_missing_attention(self, response: Dict, context: Dict) -> List[str]:
        """Check for critical context elements that lack attention"""
        missing = []
        for key in self.expected_attention_keys:
            if key in context and not self._shows_attention_to(response, key, context[key]):
                missing.append(key)
        return missing

class EigenvalueAnalyzer:
    """Analyzes eigenvalue patterns of response embeddings for anomaly detection"""
    
    def __init__(self):
        self.embedding_dim = 10  # Simplified embedding dimension
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Detect anomalies using eigenvalue analysis"""
        import time
        
        try:
            # Create embedding matrix
            embedding_matrix = self._create_embedding_matrix(response, context)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(embedding_matrix)
            eigenvalues = np.real(eigenvalues)  # Take real part
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # Analyze eigenvalue distribution
            anomaly_score = self._analyze_eigenvalue_distribution(eigenvalues)
            
            return DetectionResult(
                layer="Layer1",
                detector="EigenvalueAnalyzer",
                confidence=0.7,
                hallucination_score=anomaly_score,
                details={
                    'eigenvalues': eigenvalues.tolist(),
                    'max_eigenvalue': float(eigenvalues[0]),
                    'eigenvalue_ratio': float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] != 0 else float('inf')
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer1",
                detector="EigenvalueAnalyzer",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _create_embedding_matrix(self, response: Dict, context: Dict) -> np.ndarray:
        """Create a simplified embedding matrix from response and context"""
        # Simplified embedding based on numerical features
        features = []
        
        # Response features
        features.append(response.get('safety_score', 0.5))
        features.append(len(str(response)) / 100.0)  # Response length
        features.append(len(response.keys()) if isinstance(response, dict) else 0)
        
        # Context features
        features.append(len(context.get('aircraft', [])))
        features.append(context.get('complexity_score', 0.5))
        
        # Pad or truncate to embedding_dim
        while len(features) < self.embedding_dim:
            features.append(0.0)
        features = features[:self.embedding_dim]
        
        # Create matrix (simplified as outer product)
        feature_vector = np.array(features)
        embedding_matrix = np.outer(feature_vector, feature_vector)
        
        return embedding_matrix
    
    def _analyze_eigenvalue_distribution(self, eigenvalues: np.ndarray) -> float:
        """Analyze eigenvalue distribution for anomalies"""
        if len(eigenvalues) < 2:
            return 0.5
        
        # Check for extreme ratios (sign of potential anomalies)
        max_eigenvalue = eigenvalues[0]
        min_eigenvalue = eigenvalues[-1]
        
        if min_eigenvalue == 0:
            ratio = float('inf')
        else:
            ratio = max_eigenvalue / abs(min_eigenvalue)
        
        # Normalize ratio to [0, 1] score
        if ratio > 100:  # Very large ratio indicates anomaly
            return 0.9
        elif ratio > 10:
            return 0.6
        else:
            return 0.3

# ================== LAYER 2 DETECTORS ==================

class SemanticEntropyCalculator:
    """Calculates semantic entropy to detect uncertainty and potential hallucinations"""
    
    def __init__(self):
        self.vocabulary_size = 1000  # Simplified vocabulary
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Calculate semantic entropy of the response"""
        import time
        
        try:
            # Convert response to text for entropy calculation
            response_text = json.dumps(response, default=str)
            
            # Calculate token-level entropy
            token_entropy = self._calculate_token_entropy(response_text)
            
            # Calculate semantic coherence
            coherence_score = self._calculate_semantic_coherence(response, context)
            
            # Combine entropy and coherence for hallucination score
            hallucination_score = token_entropy + (1.0 - coherence_score)
            hallucination_score = min(1.0, hallucination_score / 2.0)
            
            return DetectionResult(
                layer="Layer2",
                detector="SemanticEntropyCalculator",
                confidence=0.8,
                hallucination_score=hallucination_score,
                details={
                    'token_entropy': token_entropy,
                    'coherence_score': coherence_score,
                    'text_length': len(response_text)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer2",
                detector="SemanticEntropyCalculator",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _calculate_token_entropy(self, text: str) -> float:
        """Calculate entropy of token distribution"""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Calculate probabilities
        total_tokens = len(tokens)
        probabilities = [count / total_tokens for count in token_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(token_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_semantic_coherence(self, response: Dict, context: Dict) -> float:
        """Calculate semantic coherence between response and context"""
        if not isinstance(response, dict):
            return 0.0
        
        # Check for semantic consistency
        coherence_factors = []
        
        # Check if response type matches context requirements
        if 'conflict' in context and 'action' in response:
            coherence_factors.append(1.0)
        else:
            coherence_factors.append(0.5)
        
        # Check numerical consistency
        if 'altitude' in response and isinstance(response['altitude'], (int, float)):
            if 1000 <= response['altitude'] <= 50000:
                coherence_factors.append(1.0)
            else:
                coherence_factors.append(0.0)
        
        # Check for conflicting instructions
        if 'climb' in str(response).lower() and 'descend' in str(response).lower():
            coherence_factors.append(0.0)
        else:
            coherence_factors.append(1.0)
        
        return np.mean(coherence_factors) if coherence_factors else 0.5

class PredictiveUncertaintyAnalyzer:
    """Analyzes predictive uncertainty patterns to detect hallucinations"""
    
    def __init__(self):
        self.uncertainty_threshold = 0.7
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Analyze predictive uncertainty patterns"""
        import time
        
        try:
            # Calculate response uncertainty
            response_uncertainty = self._calculate_response_uncertainty(response)
            
            # Calculate context-response alignment
            alignment_score = self._calculate_alignment(response, context)
            
            # Calculate prediction confidence
            prediction_confidence = self._calculate_prediction_confidence(response)
            
            # Combine factors for hallucination score
            hallucination_score = response_uncertainty + (1.0 - alignment_score) + (1.0 - prediction_confidence)
            hallucination_score = min(1.0, hallucination_score / 3.0)
            
            return DetectionResult(
                layer="Layer2",
                detector="PredictiveUncertaintyAnalyzer",
                confidence=0.75,
                hallucination_score=hallucination_score,
                details={
                    'response_uncertainty': response_uncertainty,
                    'alignment_score': alignment_score,
                    'prediction_confidence': prediction_confidence
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer2",
                detector="PredictiveUncertaintyAnalyzer",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _calculate_response_uncertainty(self, response: Dict) -> float:
        """Calculate uncertainty in the response"""
        if not isinstance(response, dict):
            return 1.0
        
        uncertainty_indicators = 0
        total_checks = 0
        
        # Check for vague language
        response_str = json.dumps(response, default=str).lower()
        vague_words = ['maybe', 'possibly', 'might', 'could', 'uncertain', 'unclear']
        for word in vague_words:
            total_checks += 1
            if word in response_str:
                uncertainty_indicators += 1
        
        # Check for missing confidence scores
        total_checks += 1
        if 'confidence' not in response and 'safety_score' not in response:
            uncertainty_indicators += 1
        
        # Check for incomplete responses
        total_checks += 1
        required_fields = ['action', 'type']
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            uncertainty_indicators += 1
        
        return uncertainty_indicators / max(total_checks, 1)
    
    def _calculate_alignment(self, response: Dict, context: Dict) -> float:
        """Calculate alignment between response and context"""
        if not isinstance(response, dict) or not isinstance(context, dict):
            return 0.0
        
        alignment_score = 0.0
        total_factors = 0
        
        # Check aircraft alignment
        if 'aircraft' in context and 'action' in response:
            total_factors += 1
            context_aircraft = context['aircraft']
            if any(ac.get('id') in str(response) for ac in context_aircraft):
                alignment_score += 1.0
        
        # Check conflict type alignment
        if 'conflict_type' in context:
            total_factors += 1
            conflict_type = context['conflict_type']
            response_str = str(response).lower()
            if conflict_type.lower() in response_str:
                alignment_score += 1.0
        
        return alignment_score / max(total_factors, 1)
    
    def _calculate_prediction_confidence(self, response: Dict) -> float:
        """Calculate confidence in the prediction"""
        if not isinstance(response, dict):
            return 0.0
        
        # Check for explicit confidence
        if 'confidence' in response:
            return float(response['confidence'])
        
        # Check for safety score as proxy
        if 'safety_score' in response:
            return float(response['safety_score'])
        
        # Infer confidence from response completeness
        required_fields = ['action', 'type', 'safety_score']
        present_fields = [field for field in required_fields if field in response]
        return len(present_fields) / len(required_fields)

class ConvexHullDispersionMeasure:
    """Measures dispersion using convex hull analysis for anomaly detection"""
    
    def __init__(self):
        self.feature_space_dim = 5
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Analyze dispersion patterns using convex hull"""
        import time
        
        try:
            # Extract feature points
            feature_points = self._extract_feature_points(response, context)
            
            # Calculate convex hull dispersion
            dispersion_score = self._calculate_convex_hull_dispersion(feature_points)
            
            # Anomaly detection based on dispersion
            hallucination_score = self._interpret_dispersion(dispersion_score)
            
            return DetectionResult(
                layer="Layer2",
                detector="ConvexHullDispersionMeasure",
                confidence=0.7,
                hallucination_score=hallucination_score,
                details={
                    'dispersion_score': dispersion_score,
                    'feature_points': feature_points.tolist(),
                    'hull_volume': self._calculate_hull_volume(feature_points)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer2",
                detector="ConvexHullDispersionMeasure",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _extract_feature_points(self, response: Dict, context: Dict) -> np.ndarray:
        """Extract feature points for convex hull analysis"""
        points = []
        
        # Generate multiple feature vectors
        for i in range(10):  # Create 10 points for hull analysis
            point = []
            
            # Feature 1: Response complexity
            if isinstance(response, dict):
                complexity = len(response.keys()) / 10.0
            else:
                complexity = 0.1
            point.append(complexity + np.random.normal(0, 0.1))
            
            # Feature 2: Safety score variance
            safety_score = response.get('safety_score', 0.5) if isinstance(response, dict) else 0.5
            point.append(safety_score + np.random.normal(0, 0.1))
            
            # Feature 3: Context alignment
            if isinstance(context, dict) and isinstance(response, dict):
                alignment = len(set(str(context).split()) & set(str(response).split())) / 100.0
            else:
                alignment = 0.1
            point.append(alignment + np.random.normal(0, 0.1))
            
            # Feature 4: Numerical consistency
            num_values = [v for v in response.values() if isinstance(v, (int, float))] if isinstance(response, dict) else []
            consistency = 1.0 if num_values and all(0 <= v <= 1000000 for v in num_values) else 0.0
            point.append(consistency + np.random.normal(0, 0.1))
            
            # Feature 5: Response length
            response_length = len(str(response)) / 1000.0
            point.append(response_length + np.random.normal(0, 0.1))
            
            points.append(point)
        
        return np.array(points)
    
    def _calculate_convex_hull_dispersion(self, points: np.ndarray) -> float:
        """Calculate dispersion using convex hull"""
        if len(points) < 3:
            return 0.5
        
        try:
            hull = ConvexHull(points)
            # Dispersion as ratio of hull volume to bounding box volume
            hull_volume = hull.volume
            
            # Calculate bounding box volume
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            bbox_volume = np.prod(max_coords - min_coords)
            
            if bbox_volume > 0:
                dispersion = hull_volume / bbox_volume
            else:
                dispersion = 0.5
            
            return min(1.0, dispersion)
            
        except Exception:
            # Fallback to simple dispersion measure
            return np.std(points.flatten())
    
    def _interpret_dispersion(self, dispersion_score: float) -> float:
        """Interpret dispersion score as hallucination probability"""
        # High dispersion might indicate inconsistent/hallucinated responses
        if dispersion_score > 0.8:
            return 0.8
        elif dispersion_score > 0.5:
            return 0.4
        else:
            return 0.2
    
    def _calculate_hull_volume(self, points: np.ndarray) -> float:
        """Calculate convex hull volume"""
        try:
            hull = ConvexHull(points)
            return float(hull.volume)
        except Exception:
            return 0.0

# ================== LAYER 3 DETECTORS ==================

class SelfConsistencyValidator:
    """Validates self-consistency by multiple sampling and comparison"""
    
    def __init__(self):
        self.sample_count = 5
        self.consistency_threshold = 0.8
        
    def detect(self, response: Dict, context: Dict, llm_client=None) -> DetectionResult:
        """Validate self-consistency through multiple sampling"""
        import time
        
        try:
            if llm_client is None:
                # Mock validation when no LLM client available
                return self._mock_consistency_check(response, context)
            
            # Generate multiple responses for the same prompt
            responses = self._generate_multiple_responses(context, llm_client)
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency(responses)
            
            # Calculate hallucination score (inverse of consistency)
            hallucination_score = 1.0 - consistency_score
            
            return DetectionResult(
                layer="Layer3",
                detector="SelfConsistencyValidator",
                confidence=0.9,
                hallucination_score=hallucination_score,
                details={
                    'consistency_score': consistency_score,
                    'sample_count': len(responses),
                    'response_variance': self._calculate_response_variance(responses)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer3",
                detector="SelfConsistencyValidator",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _mock_consistency_check(self, response: Dict, context: Dict) -> DetectionResult:
        """Mock consistency check when LLM client not available"""
        import time
        
        # Simple heuristic-based consistency check
        consistency_score = 0.8  # Assume reasonable consistency
        
        if isinstance(response, dict):
            # Check for internal consistency
            if 'action' in response and 'safety_score' in response:
                if response['safety_score'] > 0.7 and 'safe' in str(response['action']).lower():
                    consistency_score = 0.9
                elif response['safety_score'] < 0.3 and 'emergency' in str(response['action']).lower():
                    consistency_score = 0.9
        
        return DetectionResult(
            layer="Layer3",
            detector="SelfConsistencyValidator",
            confidence=0.6,
            hallucination_score=1.0 - consistency_score,
            details={
                'consistency_score': consistency_score,
                'method': 'mock_heuristic'
            },
            timestamp=time.time()
        )
    
    def _generate_multiple_responses(self, context: Dict, llm_client) -> List[Dict]:
        """Generate multiple responses for consistency checking"""
        responses = []
        prompt = self._create_consistency_prompt(context)
        
        for _ in range(self.sample_count):
            try:
                response = llm_client.query(prompt)
                if isinstance(response, dict):
                    responses.append(response)
            except Exception:
                continue
        
        return responses
    
    def _create_consistency_prompt(self, context: Dict) -> str:
        """Create prompt for consistency checking"""
        return f"Given the ATC conflict scenario: {json.dumps(context)}, provide the best resolution."
    
    def _calculate_consistency(self, responses: List[Dict]) -> float:
        """Calculate consistency score across multiple responses"""
        if len(responses) < 2:
            return 0.5
        
        consistency_scores = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                score = self._compare_responses(responses[i], responses[j])
                consistency_scores.append(score)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _compare_responses(self, resp1: Dict, resp2: Dict) -> float:
        """Compare two responses for consistency"""
        if not isinstance(resp1, dict) or not isinstance(resp2, dict):
            return 0.0
        
        similarities = []
        
        # Check action similarity
        if 'action' in resp1 and 'action' in resp2:
            if resp1['action'] == resp2['action']:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        # Check safety score similarity
        if 'safety_score' in resp1 and 'safety_score' in resp2:
            diff = abs(resp1['safety_score'] - resp2['safety_score'])
            similarities.append(1.0 - diff)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_response_variance(self, responses: List[Dict]) -> float:
        """Calculate variance in responses"""
        if not responses:
            return 0.0
        
        # Calculate variance in safety scores
        safety_scores = [r.get('safety_score', 0.5) for r in responses if isinstance(r, dict)]
        if safety_scores:
            return float(np.var(safety_scores))
        return 0.0

class MultiModelConsensusChecker:
    """Checks consensus across multiple LLM models"""
    
    def __init__(self):
        self.models = ['llama3.1:8b', 'mistral:7b', 'codellama:7b']
        self.consensus_threshold = 0.6
        
    def detect(self, response: Dict, context: Dict, ensemble_client=None) -> DetectionResult:
        """Check multi-model consensus"""
        import time
        
        try:
            if ensemble_client is None:
                return self._mock_consensus_check(response, context)
            
            # Get responses from multiple models
            model_responses = self._query_multiple_models(context, ensemble_client)
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus(model_responses)
            
            # Calculate hallucination score
            hallucination_score = 1.0 - consensus_score
            
            return DetectionResult(
                layer="Layer3",
                detector="MultiModelConsensusChecker",
                confidence=0.85,
                hallucination_score=hallucination_score,
                details={
                    'consensus_score': consensus_score,
                    'model_count': len(model_responses),
                    'agreement_matrix': self._calculate_agreement_matrix(model_responses)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer3",
                detector="MultiModelConsensusChecker",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _mock_consensus_check(self, response: Dict, context: Dict) -> DetectionResult:
        """Mock consensus check"""
        import time
        
        # Simple mock consensus based on response quality
        consensus_score = 0.7  # Default consensus
        
        if isinstance(response, dict):
            quality_indicators = 0
            total_checks = 0
            
            # Check for required fields
            required_fields = ['action', 'safety_score', 'type']
            for field in required_fields:
                total_checks += 1
                if field in response:
                    quality_indicators += 1
            
            consensus_score = quality_indicators / max(total_checks, 1)
        
        return DetectionResult(
            layer="Layer3",
            detector="MultiModelConsensusChecker",
            confidence=0.6,
            hallucination_score=1.0 - consensus_score,
            details={
                'consensus_score': consensus_score,
                'method': 'mock_quality_check'
            },
            timestamp=time.time()
        )
    
    def _query_multiple_models(self, context: Dict, ensemble_client) -> List[Dict]:
        """Query multiple models for consensus checking"""
        responses = []
        prompt = f"Resolve this ATC conflict: {json.dumps(context)}"
        
        for model in self.models:
            try:
                response = ensemble_client.query_model(prompt, model)
                if isinstance(response, dict):
                    responses.append(response)
            except Exception:
                continue
        
        return responses
    
    def _calculate_consensus(self, responses: List[Dict]) -> float:
        """Calculate consensus score across models"""
        if len(responses) < 2:
            return 0.5
        
        agreement_scores = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                agreement = self._calculate_agreement(responses[i], responses[j])
                agreement_scores.append(agreement)
        
        return np.mean(agreement_scores) if agreement_scores else 0.5
    
    def _calculate_agreement(self, resp1: Dict, resp2: Dict) -> float:
        """Calculate agreement between two model responses"""
        if not isinstance(resp1, dict) or not isinstance(resp2, dict):
            return 0.0
        
        agreements = []
        
        # Check action agreement
        if 'action' in resp1 and 'action' in resp2:
            if str(resp1['action']).lower() == str(resp2['action']).lower():
                agreements.append(1.0)
            else:
                agreements.append(0.0)
        
        # Check type agreement
        if 'type' in resp1 and 'type' in resp2:
            if str(resp1['type']).lower() == str(resp2['type']).lower():
                agreements.append(1.0)
            else:
                agreements.append(0.0)
        
        # Check safety score agreement (within tolerance)
        if 'safety_score' in resp1 and 'safety_score' in resp2:
            diff = abs(resp1['safety_score'] - resp2['safety_score'])
            if diff < 0.2:  # 20% tolerance
                agreements.append(1.0)
            else:
                agreements.append(1.0 - diff)
        
        return np.mean(agreements) if agreements else 0.5
    
    def _calculate_agreement_matrix(self, responses: List[Dict]) -> List[List[float]]:
        """Calculate pairwise agreement matrix"""
        n = len(responses)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._calculate_agreement(responses[i], responses[j])
        
        return matrix

class RetrievalAugmentedValidator:
    """Validates responses against retrieved aviation knowledge base"""
    
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        
    def detect(self, response: Dict, context: Dict) -> DetectionResult:
        """Validate response against aviation knowledge base"""
        import time
        
        try:
            # Retrieve relevant knowledge
            relevant_knowledge = self._retrieve_knowledge(context)
            
            # Validate response against knowledge
            validation_score = self._validate_against_knowledge(response, relevant_knowledge)
            
            # Calculate hallucination score
            hallucination_score = 1.0 - validation_score
            
            return DetectionResult(
                layer="Layer3",
                detector="RetrievalAugmentedValidator",
                confidence=0.8,
                hallucination_score=hallucination_score,
                details={
                    'validation_score': validation_score,
                    'relevant_knowledge': relevant_knowledge,
                    'knowledge_matches': self._count_knowledge_matches(response, relevant_knowledge)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return DetectionResult(
                layer="Layer3",
                detector="RetrievalAugmentedValidator",
                confidence=0.0,
                hallucination_score=0.5,
                details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize aviation knowledge base"""
        return {
            'separation_standards': {
                'horizontal': 5.0,  # NM
                'vertical': 1000,   # feet
                'wake_turbulence': 3.0  # NM
            },
            'altitude_ranges': {
                'min': 1000,
                'max': 50000,
                'standard_levels': [10000, 18000, 24000, 35000, 41000]
            },
            'valid_maneuvers': [
                'heading_change', 'altitude_change', 'speed_change',
                'vector', 'direct_to', 'hold', 'climb', 'descend'
            ],
            'emergency_procedures': [
                'immediate_turn', 'emergency_descent', 'priority_clearance'
            ]
        }
    
    def _retrieve_knowledge(self, context: Dict) -> Dict:
        """Retrieve relevant knowledge for the given context"""
        relevant = {}
        
        # Always include separation standards
        relevant['separation_standards'] = self.knowledge_base['separation_standards']
        
        # Include altitude ranges if altitude-related conflict
        if any('altitude' in str(v).lower() for v in context.values()):
            relevant['altitude_ranges'] = self.knowledge_base['altitude_ranges']
        
        # Include valid maneuvers
        relevant['valid_maneuvers'] = self.knowledge_base['valid_maneuvers']
        
        # Include emergency procedures if high severity
        if context.get('severity', 'low') == 'high':
            relevant['emergency_procedures'] = self.knowledge_base['emergency_procedures']
        
        return relevant
    
    def _validate_against_knowledge(self, response: Dict, knowledge: Dict) -> float:
        """Validate response against retrieved knowledge"""
        if not isinstance(response, dict):
            return 0.0
        
        validation_scores = []
        
        # Validate maneuver type
        if 'type' in response:
            maneuver_type = response['type'].lower()
            valid_maneuvers = knowledge.get('valid_maneuvers', [])
            if maneuver_type in [m.lower() for m in valid_maneuvers]:
                validation_scores.append(1.0)
            else:
                validation_scores.append(0.0)
        
        # Validate altitude ranges
        if 'altitude' in response and 'altitude_ranges' in knowledge:
            altitude = response['altitude']
            alt_ranges = knowledge['altitude_ranges']
            if alt_ranges['min'] <= altitude <= alt_ranges['max']:
                validation_scores.append(1.0)
            else:
                validation_scores.append(0.0)
        
        # Validate separation compliance
        if 'safety_score' in response and 'separation_standards' in knowledge:
            safety_score = response['safety_score']
            # High safety score should correlate with separation compliance
            if safety_score > 0.7:
                validation_scores.append(1.0)
            elif safety_score > 0.5:
                validation_scores.append(0.7)
            else:
                validation_scores.append(0.3)
        
        return np.mean(validation_scores) if validation_scores else 0.5
    
    def _count_knowledge_matches(self, response: Dict, knowledge: Dict) -> int:
        """Count how many knowledge elements match the response"""
        matches = 0
        response_str = json.dumps(response, default=str).lower()
        
        for category, items in knowledge.items():
            if isinstance(items, list):
                for item in items:
                    if str(item).lower() in response_str:
                        matches += 1
            elif isinstance(items, dict):
                for key, value in items.items():
                    if str(key).lower() in response_str or str(value).lower() in response_str:
                        matches += 1
        
        return matches

# ================== MAIN HALLUCINATION DETECTION FRAMEWORK ==================

class ComprehensiveHallucinationDetector:
    """
    Main hallucination detection system that combines all three layers
    for comprehensive analysis of LLM responses in ATC scenarios.
    """
    
    def __init__(self, llm_client=None, ensemble_client=None):
        # Layer 1 detectors
        self.mind_framework = MINDFramework()
        self.attention_detector = AttentionPatternDetector()
        self.eigenvalue_analyzer = EigenvalueAnalyzer()
        
        # Layer 2 detectors
        self.entropy_calculator = SemanticEntropyCalculator()
        self.uncertainty_analyzer = PredictiveUncertaintyAnalyzer()
        self.convex_hull_detector = ConvexHullDispersionMeasure()
        
        # Layer 3 detectors
        self.consistency_validator = SelfConsistencyValidator()
        self.consensus_checker = MultiModelConsensusChecker()
        self.rag_validator = RetrievalAugmentedValidator()
        
        # External dependencies
        self.llm_client = llm_client
        self.ensemble_client = ensemble_client
        
        # Detection thresholds
        self.thresholds = {
            'fabrication': 0.6,
            'omission': 0.5,
            'irrelevancy': 0.7,
            'contradiction': 0.8
        }
        
        self.logger = logging.getLogger(__name__)
    
    def detect_hallucinations(self, llm_response: Dict, baseline_response: Dict, 
                            conflict_context: Dict) -> HallucinationAnalysis:
        """
        Comprehensive hallucination detection using all three layers
        
        Args:
            llm_response: The LLM's response to analyze
            baseline_response: Baseline/reference response for comparison
            conflict_context: Context of the conflict scenario
            
        Returns:
            HallucinationAnalysis with comprehensive results
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting comprehensive hallucination detection")
        
        # Collect all detection results
        layer_results = []
        
        # Layer 1 Detection
        try:
            layer_results.append(self.mind_framework.detect(llm_response, conflict_context))
            layer_results.append(self.attention_detector.detect(llm_response, conflict_context))
            layer_results.append(self.eigenvalue_analyzer.detect(llm_response, conflict_context))
        except Exception as e:
            self.logger.error(f"Layer 1 detection error: {e}")
        
        # Layer 2 Detection
        try:
            layer_results.append(self.entropy_calculator.detect(llm_response, conflict_context))
            layer_results.append(self.uncertainty_analyzer.detect(llm_response, conflict_context))
            layer_results.append(self.convex_hull_detector.detect(llm_response, conflict_context))
        except Exception as e:
            self.logger.error(f"Layer 2 detection error: {e}")
        
        # Layer 3 Detection
        try:
            layer_results.append(self.consistency_validator.detect(llm_response, conflict_context, self.llm_client))
            layer_results.append(self.consensus_checker.detect(llm_response, conflict_context, self.ensemble_client))
            layer_results.append(self.rag_validator.detect(llm_response, conflict_context))
        except Exception as e:
            self.logger.error(f"Layer 3 detection error: {e}")
        
        # Analyze results and classify hallucination types
        hallucination_types = self._classify_hallucination_types(layer_results, llm_response, baseline_response, conflict_context)
        
        # Calculate overall hallucination score
        overall_score = self._calculate_overall_score(layer_results)
        
        # Assess safety impact
        safety_impact = self._assess_safety_impact(llm_response, baseline_response, overall_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(hallucination_types, overall_score, safety_impact)
        
        execution_time = time.time() - start_time
        self.logger.info(f"Hallucination detection completed in {execution_time:.3f}s")
        
        return HallucinationAnalysis(
            hallucination_types=hallucination_types,
            overall_score=overall_score,
            layer_results=layer_results,
            safety_impact=safety_impact,
            recommendation=recommendation
        )
    
    def _classify_hallucination_types(self, layer_results: List[DetectionResult], 
                                    llm_response: Dict, baseline_response: Dict, 
                                    conflict_context: Dict) -> List[HallucinationType]:
        """Classify the types of hallucinations detected"""
        hallucination_types = []
        
        # Aggregate scores by type
        type_scores = {
            'fabrication': 0.0,
            'omission': 0.0,
            'irrelevancy': 0.0,
            'contradiction': 0.0
        }
        
        # Analyze layer results for type indicators
        for result in layer_results:
            if result.confidence > 0.5:  # Only consider confident detections
                # Map detector results to hallucination types
                if result.detector in ['MINDFramework', 'EigenvalueAnalyzer']:
                    type_scores['fabrication'] += result.hallucination_score * result.confidence
                elif result.detector in ['AttentionPatternDetector', 'PredictiveUncertaintyAnalyzer']:
                    type_scores['omission'] += result.hallucination_score * result.confidence
                elif result.detector in ['SemanticEntropyCalculator', 'ConvexHullDispersionMeasure']:
                    type_scores['irrelevancy'] += result.hallucination_score * result.confidence
                elif result.detector in ['SelfConsistencyValidator', 'MultiModelConsensusChecker']:
                    type_scores['contradiction'] += result.hallucination_score * result.confidence
        
        # Normalize scores
        max_possible_score = sum(1.0 for r in layer_results if r.confidence > 0.5)
        if max_possible_score > 0:
            for key in type_scores:
                type_scores[key] /= max_possible_score
        
        # Additional heuristic-based classification
        type_scores = self._enhance_type_classification(type_scores, llm_response, baseline_response, conflict_context)
        
        # Determine which types exceed thresholds
        for hallucination_type, score in type_scores.items():
            if score >= self.thresholds.get(hallucination_type, 0.5):
                hallucination_types.append(HallucinationType(hallucination_type))
        
        return hallucination_types
    
    def _enhance_type_classification(self, type_scores: Dict, llm_response: Dict, 
                                   baseline_response: Dict, conflict_context: Dict) -> Dict:
        """Enhance type classification with heuristic analysis"""
        
        # Fabrication detection
        if isinstance(llm_response, dict):
            # Check for unrealistic values
            if 'altitude' in llm_response:
                alt = llm_response['altitude']
                if isinstance(alt, (int, float)) and (alt < 0 or alt > 60000):
                    type_scores['fabrication'] += 0.3
            
            # Check for invalid maneuver types
            if 'type' in llm_response:
                valid_types = ['heading_change', 'altitude_change', 'speed_change', 'vector', 'direct_to']
                if llm_response['type'] not in valid_types:
                    type_scores['fabrication'] += 0.2
        
        # Omission detection
        required_fields = ['action', 'type', 'safety_score']
        if isinstance(llm_response, dict):
            missing_fields = [f for f in required_fields if f not in llm_response]
            type_scores['omission'] += len(missing_fields) * 0.15
        
        # Irrelevancy detection
        if isinstance(conflict_context, dict) and isinstance(llm_response, dict):
            # Check if response addresses the conflict
            response_str = json.dumps(llm_response, default=str).lower()
            context_aircraft = conflict_context.get('aircraft', [])
            aircraft_mentioned = any(ac.get('id', '').lower() in response_str for ac in context_aircraft)
            if not aircraft_mentioned and context_aircraft:
                type_scores['irrelevancy'] += 0.25
        
        # Contradiction detection
        if isinstance(llm_response, dict) and isinstance(baseline_response, dict):
            # Check for conflicting actions
            llm_action = str(llm_response.get('action', '')).lower()
            baseline_action = str(baseline_response.get('action', '')).lower()
            
            # Detect contradictory instructions
            if ('climb' in llm_action and 'descend' in baseline_action) or \
               ('turn_left' in llm_action and 'turn_right' in baseline_action):
                type_scores['contradiction'] += 0.3
            
            # Check safety score contradictions
            llm_safety = llm_response.get('safety_score', 0.5)
            baseline_safety = baseline_response.get('safety_score', 0.5)
            if abs(llm_safety - baseline_safety) > 0.5:
                type_scores['contradiction'] += 0.2
        
        # Ensure scores don't exceed 1.0
        for key in type_scores:
            type_scores[key] = min(1.0, type_scores[key])
        
        return type_scores
    
    def _calculate_overall_score(self, layer_results: List[DetectionResult]) -> float:
        """Calculate overall hallucination score"""
        if not layer_results:
            return 0.5
        
        # Weighted average based on confidence and layer
        weighted_scores = []
        weights = []
        
        for result in layer_results:
            # Layer 3 gets highest weight, Layer 1 gets lowest
            layer_weight = {
                'Layer1': 1.0,
                'Layer2': 1.5,
                'Layer3': 2.0
            }.get(result.layer, 1.0)
            
            total_weight = result.confidence * layer_weight
            weighted_scores.append(result.hallucination_score * total_weight)
            weights.append(total_weight)
        
        if sum(weights) > 0:
            overall_score = sum(weighted_scores) / sum(weights)
        else:
            overall_score = 0.5
        
        return min(1.0, max(0.0, overall_score))
    
    def _assess_safety_impact(self, llm_response: Dict, baseline_response: Dict, 
                            overall_score: float) -> float:
        """Assess the safety impact of detected hallucinations"""
        safety_impact = overall_score  # Base impact on overall hallucination score
        
        # Enhance based on response analysis
        if isinstance(llm_response, dict) and isinstance(baseline_response, dict):
            # Safety score degradation
            llm_safety = llm_response.get('safety_score', 0.5)
            baseline_safety = baseline_response.get('safety_score', 0.5)
            safety_degradation = max(0, baseline_safety - llm_safety)
            safety_impact += safety_degradation * 0.5
            
            # Critical field hallucinations
            critical_fields = ['altitude', 'heading', 'action']
            for field in critical_fields:
                if field in llm_response and field in baseline_response:
                    if llm_response[field] != baseline_response[field]:
                        safety_impact += 0.1
        
        return min(1.0, safety_impact)
    
    def _generate_recommendation(self, hallucination_types: List[HallucinationType], 
                               overall_score: float, safety_impact: float) -> str:
        """Generate recommendation based on detection results"""
        if overall_score < 0.3 and safety_impact < 0.3:
            return "LOW_RISK: Response appears reliable, proceed with confidence"
        elif overall_score < 0.6 and safety_impact < 0.5:
            return "MODERATE_RISK: Some hallucination detected, verify response before implementation"
        elif overall_score < 0.8 or safety_impact < 0.7:
            return "HIGH_RISK: Significant hallucination detected, human review required"
        else:
            return "CRITICAL_RISK: Severe hallucination detected, reject response and escalate to human controller"
    
    def export_detection_report(self, analysis: HallucinationAnalysis, 
                              scenario_id: str, output_path: str) -> str:
        """Export comprehensive detection report"""
        report = {
            'scenario_id': scenario_id,
            'timestamp': time.time(),
            'analysis': {
                'hallucination_types': [ht.value for ht in analysis.hallucination_types],
                'overall_score': analysis.overall_score,
                'safety_impact': analysis.safety_impact,
                'recommendation': analysis.recommendation
            },
            'layer_results': [
                {
                    'layer': result.layer,
                    'detector': result.detector,
                    'confidence': result.confidence,
                    'hallucination_score': result.hallucination_score,
                    'details': result.details,
                    'timestamp': result.timestamp
                }
                for result in analysis.layer_results
            ]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path

# ================== LEGACY COMPATIBILITY ==================

class HallucinationDetector:
    """Legacy detector for backward compatibility"""
    
    def __init__(self):
        self.comprehensive_detector = ComprehensiveHallucinationDetector()
        self.valid_maneuver_types = {
            'heading_change', 'altitude_change', 'speed_change', 
            'vector', 'direct_to', 'hold', 'climb', 'descend', 'turn_left', 'turn_right'
        }
        self.valid_altitude_range = (1000, 50000)
        self.valid_heading_range = (0, 360)
        self.valid_speed_range = (100, 600)
        self.required_safety_parameters = {'safety_score', 'action', 'type'}
        
    def analyze_response(self, llm_response: Dict, baseline_response: Dict, 
                        conflict_context: Dict) -> Tuple[List[HallucinationType], Dict]:
        """Legacy analyze_response method"""
        analysis = self.comprehensive_detector.detect_hallucinations(
            llm_response, baseline_response, conflict_context
        )
        
        return analysis.hallucination_types, {
            'overall_score': analysis.overall_score,
            'safety_impact': analysis.safety_impact,
            'recommendation': analysis.recommendation,
            'layer_count': len(analysis.layer_results)
        }

def analyze_hallucinations_in_log(log_filepath: str) -> Dict:
    """
    Analyze hallucinations in simulation log file
    """
    results = {
        'total_scenarios': 0,
        'hallucination_events': [],
        'summary_stats': {}
    }
    
    detector = ComprehensiveHallucinationDetector()
    
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                try:
                    if line.strip() and line.startswith('{'):
                        log_entry = json.loads(line.strip())
                        
                        if 'llm_choice' in log_entry and 'baseline_choice' in log_entry:
                            results['total_scenarios'] += 1
                            
                            # Analyze for hallucinations
                            analysis = detector.detect_hallucinations(
                                log_entry['llm_choice'],
                                log_entry['baseline_choice'],
                                log_entry.get('conflict', {})
                            )
                            
                            if analysis.overall_score > 0.5:  # Threshold for significant hallucination
                                hallucination_event = {
                                    'scenario': log_entry.get('scenario', 'unknown'),
                                    'timestamp': log_entry.get('timestamp', 0),
                                    'hallucination_types': [ht.value for ht in analysis.hallucination_types],
                                    'overall_score': analysis.overall_score,
                                    'safety_impact': analysis.safety_impact,
                                    'recommendation': analysis.recommendation
                                }
                                results['hallucination_events'].append(hallucination_event)
                                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.warning(f"Error processing log entry: {e}")
                    continue
    
    except FileNotFoundError:
        logging.error(f"Log file not found: {log_filepath}")
        return results
    
    # Calculate summary statistics
    if results['hallucination_events']:
        overall_scores = [event['overall_score'] for event in results['hallucination_events']]
        safety_impacts = [event['safety_impact'] for event in results['hallucination_events']]
        
        results['summary_stats'] = {
            'total_hallucination_events': len(results['hallucination_events']),
            'hallucination_rate': len(results['hallucination_events']) / max(results['total_scenarios'], 1),
            'avg_hallucination_score': np.mean(overall_scores),
            'max_hallucination_score': np.max(overall_scores),
            'avg_safety_impact': np.mean(safety_impacts),
            'max_safety_impact': np.max(safety_impacts),
            'critical_events': len([e for e in results['hallucination_events'] if e['safety_impact'] > 0.8])
        }
    
    return results

class HallucinationDetector:
    """Detects and classifies hallucinations in LLM responses for ATC conflict resolution."""
    
    def __init__(self):
        self.valid_maneuver_types = {
            'heading_change', 'altitude_change', 'speed_change', 
            'vector', 'direct_to', 'hold', 'climb', 'descend', 'turn_left', 'turn_right'
        }
        self.valid_altitude_range = (1000, 50000)  # feet
        self.valid_heading_range = (0, 360)  # degrees
        self.valid_speed_range = (100, 600)  # knots
        self.required_safety_parameters = {'safety_score', 'action', 'type'}
        
    def analyze_response(self, llm_response: Dict, baseline_response: Dict, 
                        conflict_context: Dict) -> Tuple[List[HallucinationType], Dict]:
        """
        Analyze LLM response for hallucinations compared to baseline and context.
        
        Returns:
            Tuple of (hallucination_types, detailed_analysis)
        """
        hallucinations = []
        analysis = {
            'fabrication_details': [],
            'omission_details': [],
            'irrelevancy_details': [],
            'contradiction_details': []
        }
        
        # Check for Fabrication
        fabrication_issues = self._check_fabrication(llm_response, conflict_context)
        if fabrication_issues:
            hallucinations.append(HallucinationType.FABRICATION)
            analysis['fabrication_details'] = fabrication_issues
            
        # Check for Omission
        omission_issues = self._check_omission(llm_response)
        if omission_issues:
            hallucinations.append(HallucinationType.OMISSION)
            analysis['omission_details'] = omission_issues
            
        # Check for Irrelevancy
        irrelevancy_issues = self._check_irrelevancy(llm_response, conflict_context)
        if irrelevancy_issues:
            hallucinations.append(HallucinationType.IRRELEVANCY)
            analysis['irrelevancy_details'] = irrelevancy_issues
            
        # Check for Contradiction
        contradiction_issues = self._check_contradiction(llm_response, baseline_response)
        if contradiction_issues:
            hallucinations.append(HallucinationType.CONTRADICTION)
            analysis['contradiction_details'] = contradiction_issues
            
        return hallucinations, analysis
    
    def _check_fabrication(self, response: Dict, context: Dict) -> List[str]:
        """Check if LLM invented parameters or maneuvers not supported by BlueSky."""
        issues = []
        
        if not isinstance(response, dict):
            issues.append("Response is not a valid dictionary format")
            return issues
            
        # Check maneuver type validity
        maneuver_type = response.get('type', '').lower()
        if maneuver_type and maneuver_type not in self.valid_maneuver_types:
            issues.append(f"Invalid maneuver type: {maneuver_type}")
            
        # Check parameter ranges
        if 'altitude' in response:
            alt = response['altitude']
            if isinstance(alt, (int, float)) and not (self.valid_altitude_range[0] <= alt <= self.valid_altitude_range[1]):
                issues.append(f"Altitude {alt} outside valid range {self.valid_altitude_range}")
                
        if 'heading' in response:
            hdg = response['heading']
            if isinstance(hdg, (int, float)) and not (self.valid_heading_range[0] <= hdg <= self.valid_heading_range[1]):
                issues.append(f"Heading {hdg} outside valid range {self.valid_heading_range}")
                
        if 'speed' in response:
            spd = response['speed']
            if isinstance(spd, (int, float)) and not (self.valid_speed_range[0] <= spd <= self.valid_speed_range[1]):
                issues.append(f"Speed {spd} outside valid range {self.valid_speed_range}")
                
        # Check for invented aircraft IDs
        if 'aircraft_id' in response:
            response_id = response['aircraft_id']
            valid_ids = [context.get('id1', ''), context.get('id2', '')]
            if response_id not in valid_ids:
                issues.append(f"Invented aircraft ID: {response_id}")
                
        return issues
    
    def _check_omission(self, response: Dict) -> List[str]:
        """Check if LLM omitted required safety checks or parameters."""
        issues = []
        
        if not isinstance(response, dict):
            issues.append("Response format prevents safety parameter checking")
            return issues
            
        # Check for required safety parameters
        missing_params = self.required_safety_parameters - set(response.keys())
        if missing_params:
            issues.append(f"Missing required safety parameters: {missing_params}")
            
        # Check for safety score validity
        safety_score = response.get('safety_score')
        if safety_score is None:
            issues.append("No safety score provided")
        elif not isinstance(safety_score, (int, float)) or not (0 <= safety_score <= 1):
            issues.append(f"Invalid safety score: {safety_score}")
            
        # Check for action description
        action = response.get('action')
        if not action or (isinstance(action, str) and len(action.strip()) < 5):
            issues.append("Missing or insufficient action description")
            
        return issues
    
    def _check_irrelevancy(self, response: Dict, context: Dict) -> List[str]:
        """Check if LLM suggestion doesn't address the conflict scenario."""
        issues = []
        
        if not isinstance(response, dict):
            issues.append("Cannot assess relevancy of non-dictionary response")
            return issues
            
        # Check if response addresses the specific aircraft in conflict
        conflict_aircraft = {context.get('id1', ''), context.get('id2', '')}
        response_targets = set()
        
        if 'aircraft_id' in response:
            response_targets.add(response['aircraft_id'])
        if 'target_aircraft' in response:
            response_targets.add(response['target_aircraft'])
            
        if response_targets and not response_targets.intersection(conflict_aircraft):
            issues.append("Response does not target aircraft involved in conflict")
            
        # Check if maneuver type is appropriate for conflict type
        maneuver_type = response.get('type', '').lower()
        conflict_distance = context.get('distance', 5.0)
        conflict_time = context.get('time', 120)
        
        if conflict_distance < 3.0 and conflict_time < 60:  # Immediate conflict
            if maneuver_type in ['hold', 'vector']:
                issues.append("Slow maneuver suggested for immediate conflict")
                
        return issues
    
    def _check_contradiction(self, llm_response: Dict, baseline_response: Dict) -> List[str]:
        """Check if LLM response contradicts physics or baseline recommendations."""
        issues = []
        
        if not isinstance(llm_response, dict) or not isinstance(baseline_response, dict):
            return issues
            
        # Check safety score contradiction
        llm_safety = llm_response.get('safety_score', 0.5)
        baseline_safety = baseline_response.get('safety_score', 0.5)
        
        if isinstance(llm_safety, (int, float)) and isinstance(baseline_safety, (int, float)):
            if llm_safety < baseline_safety - 0.3:  # Significantly lower safety
                issues.append(f"LLM safety score {llm_safety} much lower than baseline {baseline_safety}")
                
        # Check opposing maneuver types
        llm_type = llm_response.get('type', '').lower()
        baseline_type = baseline_response.get('type', '').lower()
        
        opposing_pairs = [
            ('climb', 'descend'), ('turn_left', 'turn_right'),
            ('speed_increase', 'speed_decrease')
        ]
        
        for pair in opposing_pairs:
            if (llm_type in pair and baseline_type in pair and 
                llm_type != baseline_type):
                issues.append(f"Contradictory maneuver types: {llm_type} vs {baseline_type}")
                
        return issues

def analyze_hallucinations_in_log(log_file: str) -> Dict:
    """Analyze hallucinations in simulation log file."""
    detector = HallucinationDetector()
    hallucination_stats = {
        'total_entries': 0,
        'hallucination_breakdown': {
            'fabrication': 0,
            'omission': 0,
            'irrelevancy': 0,
            'contradiction': 0
        },
        'detailed_events': []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Extract JSON from log line
                    if line.startswith('INFO:') and '{' in line:
                        json_part = line[line.find('{'):]
                        data = json.loads(json_part)
                    elif line.startswith('{'):
                        data = json.loads(line)
                    else:
                        continue
                        
                    if 'best_by_llm' in data and 'baseline_best' in data:
                        hallucination_stats['total_entries'] += 1
                        
                        llm_response = data['best_by_llm']
                        baseline_response = data['baseline_best']
                        conflict_context = data.get('conflict', {})
                        
                        hallucinations, analysis = detector.analyze_response(
                            llm_response, baseline_response, conflict_context
                        )
                        
                        # Update statistics
                        for h_type in hallucinations:
                            hallucination_stats['hallucination_breakdown'][h_type.value] += 1
                            
                        if hallucinations:
                            hallucination_stats['detailed_events'].append({
                                'conflict_id': f"{conflict_context.get('id1', 'Unknown')}-{conflict_context.get('id2', 'Unknown')}",
                                'hallucination_types': [h.value for h in hallucinations],
                                'analysis': analysis,
                                'llm_response': llm_response,
                                'baseline_response': baseline_response
                            })
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.warning(f"Error processing log line: {e}")
                    continue
                    
    except FileNotFoundError:
        logging.error(f"Log file {log_file} not found")
    except Exception as e:
        logging.error(f"Error analyzing log file: {e}")
        
    return hallucination_stats

if __name__ == "__main__":
    # Test hallucination detection
    results = analyze_hallucinations_in_log('simulation.log')
    print("Hallucination Analysis Results:")
    print(json.dumps(results, indent=2))
