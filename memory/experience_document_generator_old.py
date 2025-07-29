# memory/experience_document_generator.py
"""
Experience Document Generator for LLM-ATC-HAL
Generates structured experience documents and creates 1024-dimensional embeddings
using Hugging Face's intfloat/e5-large-v2 model with local Chroma HNSW storage.
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os


@dataclass
class ExperienceDocument:
    """Structured experience document for storage and retrieval"""
    experience_id: str
    timestamp: float
    conflict_type: str  # 'convergent', 'parallel', 'crossing', 'overtaking'
    num_aircraft: int
    scenario_text: str  # Natural language description
    conflict_geometry_text: str  # Geometry description
    environmental_text: str  # Environmental conditions
    llm_decision_text: str  # LLM decision description
    baseline_decision_text: str  # Baseline decision description
    outcome_text: str  # Outcome description
    lessons_learned: str
    safety_margin: float
    icao_compliant: bool
    hallucination_detected: bool
    hallucination_types: List[str]
    metadata: Dict[str, Any]


class ExperienceDocumentGenerator:
    """Generates structured experience documents from raw conflict data"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            # Use environment variable
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not os.getenv('OPENAI_API_KEY') and not openai_api_key:
            logging.warning("OpenAI API key not provided. Embedding functionality will be limited.")
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path="memory/chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection with 3072-dim embeddings
        self.collection_name = "atc_experiences_3072"
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv('OPENAI_API_KEY') or openai_api_key,
                    model_name="text-embedding-3-large"
                )
            )
            logging.info(f"Connected to existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv('OPENAI_API_KEY') or openai_api_key,
                    model_name="text-embedding-3-large"
                ),
                metadata={"hnsw:space": "cosine"}  # HNSW with cosine similarity
            )
            logging.info(f"Created new collection: {self.collection_name}")
    
    def generate_experience(self, 
                          scenario_context: Dict[str, Any],
                          conflict_geometry: Dict[str, float],
                          environmental_conditions: Dict[str, Any],
                          llm_decision: Dict[str, Any],
                          baseline_decision: Dict[str, Any],
                          actual_outcome: Dict[str, Any],
                          safety_metrics: Dict[str, float],
                          hallucination_detected: bool = False,
                          hallucination_types: List[str] = None,
                          lessons_learned: str = "",
                          experience_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured experience document from raw conflict data
        
        Returns:
            Dict containing the experience document and metadata
        """
        
        if hallucination_types is None:
            hallucination_types = []
        
        if not experience_id:
            experience_id = str(uuid.uuid4())
        
        try:
            # Extract basic metadata
            aircraft_list = scenario_context.get('aircraft_list', [])
            num_aircraft = len(aircraft_list)
            
            # Determine conflict type based on geometry and aircraft movements
            conflict_type = self._determine_conflict_type(conflict_geometry, aircraft_list)
            
            # Generate natural language descriptions
            scenario_text = self._generate_scenario_description(scenario_context)
            conflict_geometry_text = self._generate_geometry_description(conflict_geometry)
            environmental_text = self._generate_environmental_description(environmental_conditions)
            llm_decision_text = self._generate_decision_description(llm_decision, "LLM")
            baseline_decision_text = self._generate_decision_description(baseline_decision, "Baseline")
            outcome_text = self._generate_outcome_description(actual_outcome, safety_metrics)
            
            # Create experience document
            experience_doc = ExperienceDocument(
                experience_id=experience_id,
                timestamp=time.time(),
                conflict_type=conflict_type,
                num_aircraft=num_aircraft,
                scenario_text=scenario_text,
                conflict_geometry_text=conflict_geometry_text,
                environmental_text=environmental_text,
                llm_decision_text=llm_decision_text,
                baseline_decision_text=baseline_decision_text,
                outcome_text=outcome_text,
                lessons_learned=lessons_learned,
                safety_margin=safety_metrics.get('effective_margin', 0.0),
                icao_compliant=safety_metrics.get('icao_compliant', False),
                hallucination_detected=hallucination_detected,
                hallucination_types=hallucination_types,
                metadata={
                    'aircraft_types': [ac.get('aircraft_type', 'unknown') for ac in aircraft_list],
                    'altitude_levels': list(set(ac.get('altitude', 0) for ac in aircraft_list)),
                    'weather': environmental_conditions.get('weather', 'unknown'),
                    'time_to_conflict': conflict_geometry.get('time_to_closest_approach', 0),
                    'closest_distance': conflict_geometry.get('closest_approach_distance', 0),
                    'resolution_success': actual_outcome.get('resolution_success', False)
                }
            )
            
            return asdict(experience_doc)
            
        except Exception as e:
            logging.error(f"Failed to generate experience document: {e}")
            # Return minimal document
            return {
                'experience_id': experience_id or str(uuid.uuid4()),
                'timestamp': time.time(),
                'conflict_type': 'unknown',
                'num_aircraft': 0,
                'scenario_text': 'Error generating document',
                'error': str(e)
            }
    
    def _determine_conflict_type(self, conflict_geometry: Dict[str, float], aircraft_list: List[Dict[str, Any]]) -> str:
        """Determine the type of conflict based on geometry and aircraft data"""
        
        try:
            # Default to convergent if we can't determine
            if len(aircraft_list) < 2:
                return 'unknown'
            
            # Check altitude differences
            altitudes = [ac.get('altitude', 0) for ac in aircraft_list]
            altitude_diff = abs(max(altitudes) - min(altitudes))
            
            # Check if aircraft are at similar altitudes (within 1000 ft)
            same_level = altitude_diff < 1000
            
            # Check approach angles and speeds to classify conflict type
            closest_distance = conflict_geometry.get('closest_approach_distance', 0)
            time_to_conflict = conflict_geometry.get('time_to_closest_approach', 0)
            
            if not same_level:
                return 'crossing'  # Different altitude levels
            elif closest_distance < 2.0:  # Very close approach
                return 'convergent'
            elif time_to_conflict > 300:  # Long time to conflict
                return 'overtaking'
            else:
                return 'parallel'
                
        except Exception:
            return 'convergent'  # Safe default
    
    def _generate_scenario_description(self, scenario_context: Dict[str, Any]) -> str:
        """Generate natural language description of the scenario"""
        
        aircraft_list = scenario_context.get('aircraft_list', [])
        if not aircraft_list:
            return "Unknown aircraft scenario"
        
        descriptions = []
        for i, aircraft in enumerate(aircraft_list):
            ac_type = aircraft.get('aircraft_type', 'unknown')
            callsign = aircraft.get('callsign', f'Aircraft {i+1}')
            altitude = aircraft.get('altitude', 0)
            speed = aircraft.get('speed', 0)
            heading = aircraft.get('heading', 0)
            
            desc = f"{callsign} ({ac_type}) at {altitude:,.0f} ft, {speed:.0f} kts, heading {heading:.0f}°"
            descriptions.append(desc)
        
        return f"Conflict scenario involving {len(aircraft_list)} aircraft: " + "; ".join(descriptions)
    
    def _generate_geometry_description(self, conflict_geometry: Dict[str, float]) -> str:
        """Generate natural language description of conflict geometry"""
        
        time_to_conflict = conflict_geometry.get('time_to_closest_approach', 0)
        closest_distance = conflict_geometry.get('closest_approach_distance', 0)
        altitude_diff = conflict_geometry.get('closest_approach_altitude_diff', 0)
        
        desc = f"Time to closest approach: {time_to_conflict:.0f} seconds, "
        desc += f"minimum separation: {closest_distance:.1f} nautical miles"
        
        if altitude_diff > 0:
            desc += f", vertical separation: {altitude_diff:.0f} feet"
        
        return desc
    
    def _generate_environmental_description(self, environmental_conditions: Dict[str, Any]) -> str:
        """Generate natural language description of environmental conditions"""
        
        weather = environmental_conditions.get('weather', 'unknown')
        wind_speed = environmental_conditions.get('wind_speed', 0)
        visibility = environmental_conditions.get('visibility', 0)
        turbulence = environmental_conditions.get('turbulence_intensity', 0)
        
        desc = f"Weather: {weather}, wind: {wind_speed:.0f} kts, visibility: {visibility:.1f} NM"
        
        if turbulence > 0:
            desc += f", turbulence intensity: {turbulence:.1f}"
        
        return desc
    
    def _generate_decision_description(self, decision: Dict[str, Any], decision_type: str) -> str:
        """Generate natural language description of a decision"""
        
        action = decision.get('action', 'no action')
        decision_kind = decision.get('type', 'unknown')
        confidence = decision.get('confidence', 0)
        safety_score = decision.get('safety_score', 0)
        
        desc = f"{decision_type} decision: {action} (type: {decision_kind})"
        if confidence > 0:
            desc += f", confidence: {confidence:.2f}"
        if safety_score > 0:
            desc += f", safety score: {safety_score:.2f}"
        
        return desc
    
    def _generate_outcome_description(self, actual_outcome: Dict[str, Any], safety_metrics: Dict[str, float]) -> str:
        """Generate natural language description of the outcome"""
        
        success = actual_outcome.get('resolution_success', False)
        separation_achieved = actual_outcome.get('separation_achieved', 0)
        time_to_resolution = actual_outcome.get('time_to_resolution', 0)
        safety_margin = safety_metrics.get('effective_margin', 0)
        
        desc = f"Resolution {'successful' if success else 'failed'}"
        
        if separation_achieved > 0:
            desc += f", final separation: {separation_achieved:.1f} NM"
        
        if time_to_resolution > 0:
            desc += f", resolution time: {time_to_resolution:.0f} seconds"
        
        desc += f", safety margin: {safety_margin:.2f}"
        
        return desc
    
    def embed_and_store(self, exp_doc: Dict[str, Any]) -> None:
        """
        Create embeddings using text-embedding-3-large and store in Chroma HNSW
        
        Args:
            exp_doc: Experience document dictionary
        """
        
        try:
            # Create comprehensive text for embedding
            text_for_embedding = self._create_embedding_text(exp_doc)
            
            # Prepare metadata for filtering
            metadata = {
                'conflict_type': exp_doc.get('conflict_type', 'unknown'),
                'num_aircraft': exp_doc.get('num_aircraft', 0),
                'timestamp': exp_doc.get('timestamp', time.time()),
                'safety_margin': exp_doc.get('safety_margin', 0.0),
                'icao_compliant': exp_doc.get('icao_compliant', False),
                'hallucination_detected': exp_doc.get('hallucination_detected', False),
                'resolution_success': exp_doc.get('metadata', {}).get('resolution_success', False)
            }
            
            # Add additional metadata from document metadata
            doc_metadata = exp_doc.get('metadata', {})
            if doc_metadata:
                metadata.update({
                    'weather': doc_metadata.get('weather', 'unknown'),
                    'time_to_conflict': doc_metadata.get('time_to_conflict', 0),
                    'closest_distance': doc_metadata.get('closest_distance', 0)
                })
            
            # Store in Chroma with automatic 3072-dim embedding
            self.collection.add(
                documents=[text_for_embedding],
                metadatas=[metadata],
                ids=[exp_doc['experience_id']]
            )
            
            logging.info(f"Stored experience {exp_doc['experience_id']} with 3072-dim embedding")
            
        except Exception as e:
            logging.error(f"Failed to embed and store experience: {e}")
            raise
    
    def _create_embedding_text(self, exp_doc: Dict[str, Any]) -> str:
        """Create comprehensive text for embedding generation"""
        
        text_parts = []
        
        # Add all textual components
        text_parts.append(f"Conflict Type: {exp_doc.get('conflict_type', 'unknown')}")
        text_parts.append(f"Aircraft Count: {exp_doc.get('num_aircraft', 0)}")
        text_parts.append(exp_doc.get('scenario_text', ''))
        text_parts.append(exp_doc.get('conflict_geometry_text', ''))
        text_parts.append(exp_doc.get('environmental_text', ''))
        text_parts.append(exp_doc.get('llm_decision_text', ''))
        text_parts.append(exp_doc.get('baseline_decision_text', ''))
        text_parts.append(exp_doc.get('outcome_text', ''))
        
        if exp_doc.get('lessons_learned'):
            text_parts.append(f"Lessons Learned: {exp_doc['lessons_learned']}")
        
        if exp_doc.get('hallucination_detected'):
            text_parts.append(f"Hallucination detected: {', '.join(exp_doc.get('hallucination_types', []))}")
        
        # Join all parts
        return " | ".join(filter(None, text_parts))
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored experiences"""
        
        try:
            count = self.collection.count()
            
            # Get some sample metadata to understand the distribution
            if count > 0:
                sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])
                
                conflict_types = {}
                aircraft_counts = {}
                hallucination_count = 0
                
                for metadata in sample_results.get('metadatas', []):
                    conflict_type = metadata.get('conflict_type', 'unknown')
                    conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1
                    
                    num_aircraft = metadata.get('num_aircraft', 0)
                    aircraft_counts[num_aircraft] = aircraft_counts.get(num_aircraft, 0) + 1
                    
                    if metadata.get('hallucination_detected', False):
                        hallucination_count += 1
                
                return {
                    'total_experiences': count,
                    'conflict_type_distribution': conflict_types,
                    'aircraft_count_distribution': aircraft_counts,
                    'hallucination_rate': hallucination_count / len(sample_results.get('metadatas', [])) if sample_results.get('metadatas') else 0,
                    'embedding_dimension': 3072  # text-embedding-3-large dimension
                }
            else:
                return {
                    'total_experiences': 0,
                    'embedding_dimension': 3072
                }
                
        except Exception as e:
            logging.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create document generator
    doc_generator = ExperienceDocumentGenerator()
    
    # Sample experience data
    scenario_context = {
        'aircraft_list': [
            {
                'aircraft_type': 'B737',
                'callsign': 'UAL123',
                'altitude': 35000,
                'speed': 450,
                'heading': 90
            },
            {
                'aircraft_type': 'A320',
                'callsign': 'DAL456',
                'altitude': 35000,
                'speed': 420,
                'heading': 270
            }
        ]
    }
    
    conflict_geometry = {
        'time_to_closest_approach': 120,
        'closest_approach_distance': 4.2,
        'closest_approach_altitude_diff': 0
    }
    
    environmental_conditions = {
        'weather': 'clear',
        'wind_speed': 15,
        'visibility': 10.0,
        'turbulence_intensity': 0.1
    }
    
    llm_decision = {
        'action': 'turn left 15 degrees',
        'type': 'heading',
        'confidence': 0.8,
        'safety_score': 0.85
    }
    
    baseline_decision = {
        'action': 'climb 1000 feet',
        'type': 'altitude',
        'confidence': 0.9,
        'safety_score': 0.9
    }
    
    actual_outcome = {
        'resolution_success': True,
        'separation_achieved': 6.5,
        'time_to_resolution': 95
    }
    
    safety_metrics = {
        'effective_margin': 0.82,
        'icao_compliant': True
    }
    
    # Generate experience document
    exp_doc = doc_generator.generate_experience(
        scenario_context, conflict_geometry, environmental_conditions,
        llm_decision, baseline_decision, actual_outcome, safety_metrics,
        lessons_learned="Heading changes are effective for convergent conflicts at same altitude"
    )
    
    print("Generated Experience Document:")
    print(json.dumps(exp_doc, indent=2))
    
    # Store with embedding
    doc_generator.embed_and_store(exp_doc)
    
    # Get statistics
    stats = doc_generator.get_collection_stats()
    print(f"\nCollection Statistics:")
    print(json.dumps(stats, indent=2))
