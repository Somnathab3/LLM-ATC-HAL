# memory/replay_store.py
"""
Vector-based Experience Replay System for ATC Conflict Resolution
Stores and retrieves similar scenarios using embeddings and similarity search
"""

import numpy as np
import json
import logging
import pickle
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import uuid
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility, MilvusClient
)
from sklearn.metrics.pairwise import cosine_similarity

# Configure Milvus logging
milvus_logger = logging.getLogger('pymilvus')
milvus_logger.setLevel(logging.WARNING)

@dataclass
class ConflictExperience:
    """Experience record for conflict resolution"""
    experience_id: str
    timestamp: float
    scenario_context: Dict[str, Any]
    conflict_geometry: Dict[str, float]
    environmental_conditions: Dict[str, Any]
    llm_decision: Dict[str, Any]
    baseline_decision: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    safety_metrics: Dict[str, float]
    hallucination_detected: bool
    hallucination_types: List[str]
    controller_override: Optional[Dict[str, Any]]
    lessons_learned: str
    embedding: Optional[np.ndarray] = None

@dataclass
class SimilarityResult:
    """Result of similarity search"""
    experience: ConflictExperience
    similarity_score: float
    relevant_aspects: List[str]

class ExperienceEmbedder:
    """Creates embeddings for conflict resolution experiences"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.sentence_transformer = SentenceTransformer(model_name)
        self.context_weights = {
            'aircraft_types': 0.15,
            'conflict_geometry': 0.25,
            'environmental': 0.15,
            'decisions': 0.25,
            'outcomes': 0.20
        }
        
    def create_embedding(self, experience: ConflictExperience) -> np.ndarray:
        """Create vector embedding for an experience"""
        
        try:
            # Create textual representation of experience
            text_components = self._extract_text_features(experience)
            
            # Create separate embeddings for different aspects
            aspect_embeddings = {}
            
            for aspect, text in text_components.items():
                if text.strip():
                    embedding = self.sentence_transformer.encode(text)
                    aspect_embeddings[aspect] = embedding
                else:
                    # Use zero vector for missing aspects
                    aspect_embeddings[aspect] = np.zeros(self.sentence_transformer.get_sentence_embedding_dimension())
            
            # Combine embeddings with weights
            combined_embedding = np.zeros_like(list(aspect_embeddings.values())[0])
            
            for aspect, weight in self.context_weights.items():
                if aspect in aspect_embeddings:
                    combined_embedding += weight * aspect_embeddings[aspect]
            
            # Normalize the combined embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            return combined_embedding
            
        except Exception as e:
            logging.error(f"Failed to create embedding: {e}")
            # Return random embedding as fallback
            dim = self.sentence_transformer.get_sentence_embedding_dimension()
            return np.random.normal(0, 0.1, dim)
    
    def _extract_text_features(self, experience: ConflictExperience) -> Dict[str, str]:
        """Extract textual features from experience for embedding"""
        
        features = {}
        
        # Aircraft types and configuration
        aircraft_info = []
        scenario = experience.scenario_context
        if 'aircraft_list' in scenario:
            for aircraft in scenario['aircraft_list']:
                aircraft_info.append(f"{aircraft.get('aircraft_type', 'unknown')} at {aircraft.get('altitude', 0)}ft")
        
        features['aircraft_types'] = f"Aircraft configuration: {', '.join(aircraft_info)}"
        
        # Conflict geometry
        geometry = experience.conflict_geometry
        features['conflict_geometry'] = (
            f"Conflict geometry: {geometry.get('time_to_closest_approach', 0):.0f} seconds to conflict, "
            f"{geometry.get('closest_approach_distance', 0):.1f} nautical miles separation, "
            f"{geometry.get('closest_approach_altitude_diff', 0):.0f} feet vertical separation"
        )
        
        # Environmental conditions
        env = experience.environmental_conditions
        features['environmental'] = (
            f"Environment: {env.get('weather', 'unknown')} weather, "
            f"{env.get('wind_speed', 0):.0f} knot winds, "
            f"visibility {env.get('visibility', 0):.1f} nautical miles, "
            f"turbulence level {env.get('turbulence_intensity', 0):.1f}"
        )
        
        # Decisions made
        llm_action = experience.llm_decision.get('action', 'unknown')
        baseline_action = experience.baseline_decision.get('action', 'unknown')
        features['decisions'] = (
            f"LLM recommended: {llm_action}, "
            f"Baseline recommended: {baseline_action}"
        )
        
        # Outcomes and safety
        outcome = experience.actual_outcome
        safety = experience.safety_metrics
        features['outcomes'] = (
            f"Outcome: {outcome.get('resolution_success', 'unknown')}, "
            f"safety margin {safety.get('effective_margin', 0):.2f}, "
            f"hallucination detected: {experience.hallucination_detected}"
        )
        
        return features

class VectorReplayStore:
    """Vector-based experience replay store using Milvus for GPU acceleration"""
    
    def __init__(self, storage_dir: str = "memory/replay_data", 
                 milvus_host: str = "localhost", milvus_port: int = 19530):
        self.storage_dir = storage_dir
        self.embedder = ExperienceEmbedder()
        self.experiences = {}
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "conflict_experiences"
        self.collection = None
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize Milvus connection and collection
        self._initialize_milvus()
        
        # Load existing experiences
        self._load_experiences()
    
    def _initialize_milvus(self):
        """Initialize Milvus connection and collection"""
        
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            logging.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            
            # Get embedding dimension
            embedding_dim = self.embedder.sentence_transformer.get_sentence_embedding_dimension()
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logging.info(f"Using existing collection '{self.collection_name}'")
            else:
                # Create collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, 
                               auto_id=False, max_length=100),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                    FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
                    FieldSchema(name="hallucination_detected", dtype=DataType.BOOL),
                    FieldSchema(name="conflict_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="safety_score", dtype=DataType.FLOAT)
                ]
                
                schema = CollectionSchema(
                    fields=fields,
                    description="ATC conflict resolution experiences"
                )
                
                # Create collection
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using='default'
                )
                
                logging.info(f"Created new collection '{self.collection_name}' with {embedding_dim}D vectors")
            
            # Create index for vector search (GPU optimized)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # Load collection into memory for search
            self.collection.load()
            
            logging.info(f"Milvus collection '{self.collection_name}' initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Milvus: {e}")
            raise RuntimeError(f"Milvus initialization failed: {e}")
    
    def _load_experiences(self):
        """Load existing experiences from storage"""
        
        experiences_file = os.path.join(self.storage_dir, "experiences.json")
        
        try:
            # Load experiences from JSON file
            if os.path.exists(experiences_file):
                with open(experiences_file, 'r') as f:
                    experiences_data = json.load(f)
                
                for exp_data in experiences_data:
                    # Convert back to ConflictExperience object
                    experience = ConflictExperience(**exp_data)
                    self.experiences[experience.experience_id] = experience
                
                logging.info(f"Loaded {len(self.experiences)} experiences from JSON")
                
                # Sync with Milvus collection
                self._sync_with_milvus()
            
        except Exception as e:
            logging.error(f"Failed to load experiences: {e}")
            self.experiences = {}
    
    def _sync_with_milvus(self):
        """Sync local experiences with Milvus collection"""
        
        try:
            # Get existing IDs from Milvus
            if self.collection.num_entities > 0:
                # Query all existing IDs
                existing_ids = set()
                for batch in self.collection.query(
                    expr="id != ''",
                    output_fields=["id"],
                    limit=16384  # Milvus default limit
                ):
                    existing_ids.add(batch["id"])
                
                # Add missing experiences to Milvus
                missing_experiences = []
                for exp_id, experience in self.experiences.items():
                    if exp_id not in existing_ids:
                        missing_experiences.append(experience)
                
                if missing_experiences:
                    self._batch_insert_to_milvus(missing_experiences)
                    logging.info(f"Synced {len(missing_experiences)} missing experiences to Milvus")
            else:
                # Collection is empty, insert all experiences
                if self.experiences:
                    self._batch_insert_to_milvus(list(self.experiences.values()))
                    logging.info(f"Inserted all {len(self.experiences)} experiences to Milvus")
                    
        except Exception as e:
            logging.error(f"Failed to sync with Milvus: {e}")
    
    def _batch_insert_to_milvus(self, experiences: List[ConflictExperience]):
        """Insert a batch of experiences to Milvus"""
        
        if not experiences:
            return
            
        try:
            # Prepare data for insertion
            ids = []
            embeddings = []
            timestamps = []
            hallucination_flags = []
            conflict_types = []
            safety_scores = []
            
            for experience in experiences:
                # Create embedding if not exists
                if experience.embedding is None:
                    embedding = self.embedder.create_embedding(experience)
                    experience.embedding = embedding
                else:
                    embedding = experience.embedding
                
                ids.append(experience.experience_id)
                embeddings.append(embedding.tolist())
                timestamps.append(experience.timestamp)
                hallucination_flags.append(experience.hallucination_detected)
                
                # Extract conflict type from LLM decision
                conflict_type = experience.llm_decision.get('type', 'unknown')[:49]  # Limit to 49 chars
                conflict_types.append(conflict_type)
                
                # Extract safety score
                safety_score = experience.safety_metrics.get('safety_level', 0.5)
                if isinstance(safety_score, str):
                    # Convert string safety levels to numeric
                    safety_map = {'critical': 0.1, 'low': 0.3, 'adequate': 0.7, 'high': 0.9}
                    safety_score = safety_map.get(safety_score, 0.5)
                safety_scores.append(float(safety_score))
            
            # Insert data
            data = [
                ids,
                embeddings,
                timestamps,
                hallucination_flags,
                conflict_types,
                safety_scores
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logging.info(f"Inserted {len(experiences)} experiences to Milvus")
            
        except Exception as e:
            logging.error(f"Failed to batch insert to Milvus: {e}")
            raise
    
    def store_experience(self, experience: ConflictExperience) -> str:
        """Store a new conflict resolution experience"""
        
        try:
            # Generate unique ID if not provided
            if not experience.experience_id:
                experience.experience_id = str(uuid.uuid4())
            
            # Create embedding
            embedding = self.embedder.create_embedding(experience)
            experience.embedding = embedding
            
            # Store in memory
            self.experiences[experience.experience_id] = experience
            
            # Insert into Milvus
            self._insert_single_to_milvus(experience)
            
            # Persist to disk
            self._save_experiences()
            
            logging.info(f"Stored experience {experience.experience_id}")
            return experience.experience_id
            
        except Exception as e:
            logging.error(f"Failed to store experience: {e}")
            return ""
    
    def _insert_single_to_milvus(self, experience: ConflictExperience):
        """Insert a single experience to Milvus"""
        
        try:
            # Extract conflict type and safety score
            conflict_type = experience.llm_decision.get('type', 'unknown')[:49]
            safety_score = experience.safety_metrics.get('safety_level', 0.5)
            if isinstance(safety_score, str):
                safety_map = {'critical': 0.1, 'low': 0.3, 'adequate': 0.7, 'high': 0.9}
                safety_score = safety_map.get(safety_score, 0.5)
            
            # Prepare data
            data = [
                [experience.experience_id],
                [experience.embedding.tolist()],
                [experience.timestamp],
                [experience.hallucination_detected],
                [conflict_type],
                [float(safety_score)]
            ]
            
            # Insert to Milvus
            self.collection.insert(data)
            self.collection.flush()
            
        except Exception as e:
            logging.error(f"Failed to insert experience to Milvus: {e}")
            raise
    
    def find_similar_experiences(self, 
                                query_experience: ConflictExperience,
                                top_k: int = 5,
                                similarity_threshold: float = 0.7) -> List[SimilarityResult]:
        """Find similar past experiences using Milvus vector search"""
        
        try:
            if self.collection.num_entities == 0:
                logging.info("No experiences in store")
                return []
            
            # Create embedding for query
            query_embedding = self.embedder.create_embedding(query_experience)
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Perform vector search in Milvus
            search_results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "timestamp", "hallucination_detected", "conflict_type", "safety_score"],
                expr=None
            )
            
            results = []
            
            for hit in search_results[0]:
                exp_id = hit.entity.get("id")
                distance = hit.distance
                
                # Get experience from memory
                experience = self.experiences.get(exp_id)
                if not experience:
                    continue
                
                # Convert L2 distance to similarity score
                similarity_score = 1.0 / (1.0 + distance)
                
                if similarity_score >= similarity_threshold:
                    # Analyze what aspects are similar
                    relevant_aspects = self._analyze_similarity_aspects(
                        query_experience, experience
                    )
                    
                    result = SimilarityResult(
                        experience=experience,
                        similarity_score=similarity_score,
                        relevant_aspects=relevant_aspects
                    )
                    
                    results.append(result)
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logging.info(f"Found {len(results)} similar experiences")
            return results
            
        except Exception as e:
            logging.error(f"Failed to find similar experiences: {e}")
            return []
    
    def _analyze_similarity_aspects(self, 
                                  query: ConflictExperience, 
                                  candidate: ConflictExperience) -> List[str]:
        """Analyze which aspects make two experiences similar"""
        
        aspects = []
        
        # Check aircraft type similarity
        query_aircraft = set(ac.get('aircraft_type', '') for ac in query.scenario_context.get('aircraft_list', []))
        candidate_aircraft = set(ac.get('aircraft_type', '') for ac in candidate.scenario_context.get('aircraft_list', []))
        
        if query_aircraft.intersection(candidate_aircraft):
            aspects.append("similar_aircraft_types")
        
        # Check conflict geometry similarity
        query_geom = query.conflict_geometry
        candidate_geom = candidate.conflict_geometry
        
        time_diff = abs(query_geom.get('time_to_closest_approach', 0) - 
                       candidate_geom.get('time_to_closest_approach', 0))
        if time_diff < 60:  # Within 60 seconds
            aspects.append("similar_conflict_timing")
        
        distance_diff = abs(query_geom.get('closest_approach_distance', 0) - 
                           candidate_geom.get('closest_approach_distance', 0))
        if distance_diff < 2.0:  # Within 2 nautical miles
            aspects.append("similar_separation_distance")
        
        # Check environmental similarity
        query_env = query.environmental_conditions
        candidate_env = candidate.environmental_conditions
        
        if query_env.get('weather') == candidate_env.get('weather'):
            aspects.append("similar_weather_conditions")
        
        # Check decision similarity
        if (query.llm_decision.get('type') == candidate.llm_decision.get('type')):
            aspects.append("similar_llm_decision_type")
        
        # Check outcome similarity
        if (query.hallucination_detected == candidate.hallucination_detected):
            aspects.append("similar_hallucination_outcome")
        
        return aspects
    
    def get_lessons_for_scenario(self, scenario_context: Dict[str, Any]) -> List[str]:
        """Get lessons learned from similar scenarios"""
        
        try:
            # Create a mock experience for similarity search
            mock_experience = ConflictExperience(
                experience_id="",
                timestamp=time.time(),
                scenario_context=scenario_context,
                conflict_geometry={'time_to_closest_approach': 120, 'closest_approach_distance': 5.0},
                environmental_conditions={'weather': 'clear'},
                llm_decision={},
                baseline_decision={},
                actual_outcome={},
                safety_metrics={},
                hallucination_detected=False,
                hallucination_types=[],
                controller_override=None,
                lessons_learned=""
            )
            
            similar_experiences = self.find_similar_experiences(mock_experience, top_k=10)
            
            lessons = []
            for result in similar_experiences:
                if result.experience.lessons_learned:
                    lessons.append(result.experience.lessons_learned)
            
            return lessons[:5]  # Return top 5 lessons
            
        except Exception as e:
            logging.error(f"Failed to get lessons: {e}")
            return []
    
    def get_hallucination_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in hallucination events"""
        
        try:
            hallucination_experiences = [
                exp for exp in self.experiences.values() 
                if exp.hallucination_detected
            ]
            
            if not hallucination_experiences:
                return {'no_data': True}
            
            # Analyze patterns
            patterns = {
                'total_hallucinations': len(hallucination_experiences),
                'hallucination_types': {},
                'environmental_correlations': {},
                'aircraft_type_correlations': {},
                'geometric_factors': {}
            }
            
            # Count hallucination types
            for exp in hallucination_experiences:
                for h_type in exp.hallucination_types:
                    patterns['hallucination_types'][h_type] = \
                        patterns['hallucination_types'].get(h_type, 0) + 1
            
            # Environmental correlations
            for exp in hallucination_experiences:
                weather = exp.environmental_conditions.get('weather', 'unknown')
                patterns['environmental_correlations'][weather] = \
                    patterns['environmental_correlations'].get(weather, 0) + 1
            
            # Aircraft type correlations
            for exp in hallucination_experiences:
                for aircraft in exp.scenario_context.get('aircraft_list', []):
                    ac_type = aircraft.get('aircraft_type', 'unknown')
                    patterns['aircraft_type_correlations'][ac_type] = \
                        patterns['aircraft_type_correlations'].get(ac_type, 0) + 1
            
            # Geometric factors
            close_approaches = [
                exp.conflict_geometry.get('closest_approach_distance', 10) 
                for exp in hallucination_experiences
            ]
            
            if close_approaches:
                patterns['geometric_factors'] = {
                    'avg_separation': float(np.mean(close_approaches)),
                    'min_separation': float(np.min(close_approaches)),
                    'max_separation': float(np.max(close_approaches))
                }
            
            return patterns
            
        except Exception as e:
            logging.error(f"Failed to analyze hallucination patterns: {e}")
            return {'error': str(e)}
    
    def _save_experiences(self):
        """Save experiences to disk (Milvus handles vector storage)"""
        
        try:
            # Save experiences as JSON (metadata only, vectors are in Milvus)
            experiences_file = os.path.join(self.storage_dir, "experiences.json")
            experiences_data = []
            
            for experience in self.experiences.values():
                exp_dict = asdict(experience)
                # Remove embedding (stored in Milvus)
                exp_dict.pop('embedding', None)
                experiences_data.append(exp_dict)
            
            with open(experiences_file, 'w') as f:
                json.dump(experiences_data, f, indent=2)
            
            logging.info("Saved experiences metadata to disk")
            
        except Exception as e:
            logging.error(f"Failed to save experiences: {e}")
    
    def export_dataset(self, filepath: str):
        """Export experiences as dataset for analysis"""
        
        try:
            export_data = {
                'metadata': {
                    'total_experiences': len(self.experiences),
                    'export_timestamp': time.time(),
                    'version': '1.0'
                },
                'experiences': []
            }
            
            for experience in self.experiences.values():
                exp_dict = asdict(experience)
                exp_dict.pop('embedding', None)  # Remove embedding
                export_data['experiences'].append(exp_dict)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logging.info(f"Exported {len(self.experiences)} experiences to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to export dataset: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        total_experiences = len(self.experiences)
        hallucination_count = sum(1 for exp in self.experiences.values() if exp.hallucination_detected)
        override_count = sum(1 for exp in self.experiences.values() if exp.controller_override)
        
        if total_experiences > 0:
            recent_experiences = [
                exp for exp in self.experiences.values() 
                if time.time() - exp.timestamp < 86400  # Last 24 hours
            ]
            
            stats = {
                'total_experiences': total_experiences,
                'hallucination_rate': hallucination_count / total_experiences,
                'override_rate': override_count / total_experiences,
                'recent_experiences_24h': len(recent_experiences),
                'milvus_collection_size': self.collection.num_entities if self.collection else 0,
                'storage_directory': self.storage_dir,
                'milvus_host': f"{self.milvus_host}:{self.milvus_port}"
            }
        else:
            stats = {
                'total_experiences': 0,
                'hallucination_rate': 0.0,
                'override_rate': 0.0,
                'recent_experiences_24h': 0,
                'milvus_collection_size': 0,
                'storage_directory': self.storage_dir,
                'milvus_host': f"{self.milvus_host}:{self.milvus_port}"
            }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create replay store
    replay_store = VectorReplayStore()
    
    # Create sample experience
    sample_experience = ConflictExperience(
        experience_id="",
        timestamp=time.time(),
        scenario_context={
            'aircraft_list': [
                {'aircraft_type': 'B737', 'altitude': 35000},
                {'aircraft_type': 'A320', 'altitude': 35000}
            ]
        },
        conflict_geometry={
            'time_to_closest_approach': 120,
            'closest_approach_distance': 4.5,
            'closest_approach_altitude_diff': 0
        },
        environmental_conditions={
            'weather': 'clear',
            'wind_speed': 15,
            'visibility': 10
        },
        llm_decision={
            'action': 'turn left 10 degrees',
            'type': 'heading',
            'safety_score': 0.8
        },
        baseline_decision={
            'action': 'climb 1000 ft',
            'type': 'altitude',
            'safety_score': 0.9
        },
        actual_outcome={
            'resolution_success': True,
            'separation_achieved': 6.2
        },
        safety_metrics={
            'effective_margin': 0.75,
            'safety_level': 'adequate'
        },
        hallucination_detected=False,
        hallucination_types=[],
        controller_override=None,
        lessons_learned="Heading changes are effective for convergent conflicts at same altitude"
    )
    
    # Store experience
    exp_id = replay_store.store_experience(sample_experience)
    print(f"Stored experience: {exp_id}")
    
    # Find similar experiences
    similar = replay_store.find_similar_experiences(sample_experience, top_k=3)
    print(f"Found {len(similar)} similar experiences")
    
    # Get statistics
    stats = replay_store.get_statistics()
    print("Storage Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Export dataset
    replay_store.export_dataset("sample_experiences.json")
    print("Dataset exported")
