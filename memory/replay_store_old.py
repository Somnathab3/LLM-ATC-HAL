# memory/replay_store.py
"""
Vector-Based Experience Replay Store for LLM-ATC-HAL
Uses 3072-dimensional text-embedding-3-large embeddings with Chroma HNSW storage
and metadata filtering for efficient experience retrieval.
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievedExperience:
    """Container for retrieved experience with similarity score"""
    experience_id: str
    experience_data: Dict[str, Any]
    similarity_score: float
    metadata: Dict[str, Any]


class VectorReplayStore:
    """
    Vector-based experience replay store using 3072-dim text-embedding-3-large
    with Chroma HNSW and metadata filtering
    """
    
    def __init__(self, storage_dir: str = "memory/chroma_db", openai_api_key: Optional[str] = None):
        self.storage_dir = storage_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=storage_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection with 3072-dim embeddings
        self.collection_name = "atc_experiences_3072"
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            self.logger.warning("OpenAI API key not found. Using default embedding function.")
            embedding_function = None
        else:
            embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-large"
            )
        
        try:
            if embedding_function:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
                self.logger.info(f"Connected to existing collection: {self.collection_name}")
            else:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                self.logger.info(f"Connected to existing collection without embedding function")
        except Exception as e:
            self.logger.info(f"Collection doesn't exist, creating new one: {e}")
            if embedding_function:
                try:
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=embedding_function,
                        metadata={"hnsw:space": "cosine"}  # HNSW with cosine similarity
                    )
                    self.logger.info(f"Created new collection: {self.collection_name}")
                except Exception as create_error:
                    self.logger.warning(f"Failed to create collection with embedding function: {create_error}")
                    # Fall back to creating without embedding function
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"Created new collection without embedding function: {self.collection_name}")
            else:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info(f"Created new collection without embedding function: {self.collection_name}")
    
    def store_experience(self, experience_data: Dict[str, Any]) -> bool:
        """
        Store an experience with automatic 3072-dim embedding generation
        
        Args:
            experience_data: Dictionary containing experience information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract required fields
            experience_id = experience_data.get('experience_id')
            if not experience_id:
                experience_id = f"exp_{int(time.time()*1000)}"
                experience_data['experience_id'] = experience_id
            
            # Create text for embedding
            embedding_text = self._create_embedding_text(experience_data)
            
            # Prepare metadata for filtering
            metadata = self._extract_metadata(experience_data)
            
            # Store in Chroma with automatic embedding
            self.collection.add(
                documents=[embedding_text],
                metadatas=[metadata],
                ids=[experience_id]
            )
            
            self.logger.debug(f"Stored experience {experience_id} with 3072-dim embedding")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store experience: {e}")
            return False
    
    def retrieve_experience(self, 
                          conflict_desc: str, 
                          conflict_type: str, 
                          num_ac: int, 
                          k: int = 5,
                          similarity_threshold: float = 0.7) -> List[RetrievedExperience]:
        """
        Retrieve similar experiences using metadata filtering then cosine search
        
        Args:
            conflict_desc: Natural language description of the conflict
            conflict_type: Type of conflict ('convergent', 'parallel', 'crossing', 'overtaking')
            num_ac: Number of aircraft involved
            k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of RetrievedExperience objects
        """
        try:
            # Build metadata filter
            where_filter = {
                "conflict_type": {"$eq": conflict_type},
                "num_aircraft": {"$eq": num_ac}
            }
            
            # Perform search with metadata filtering
            search_results = self.collection.query(
                query_texts=[conflict_desc],
                n_results=k,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to RetrievedExperience objects
            retrieved_experiences = []
            
            if search_results['ids']:
                ids = search_results['ids'][0]
                documents = search_results.get('documents', [[]])[0]
                metadatas = search_results.get('metadatas', [[]])[0]
                distances = search_results.get('distances', [[]])[0]
                
                for i, exp_id in enumerate(ids):
                    # Convert distance to similarity score (cosine distance to similarity)
                    similarity_score = 1.0 - distances[i] if i < len(distances) else 0.0
                    
                    # Apply similarity threshold
                    if similarity_score >= similarity_threshold:
                        # Reconstruct experience data from metadata and document
                        experience_data = self._reconstruct_experience_data(
                            exp_id, 
                            documents[i] if i < len(documents) else "",
                            metadatas[i] if i < len(metadatas) else {}
                        )
                        
                        retrieved_exp = RetrievedExperience(
                            experience_id=exp_id,
                            experience_data=experience_data,
                            similarity_score=similarity_score,
                            metadata=metadatas[i] if i < len(metadatas) else {}
                        )
                        
                        retrieved_experiences.append(retrieved_exp)
            
            self.logger.info(f"Retrieved {len(retrieved_experiences)} experiences for {conflict_type} with {num_ac} aircraft")
            return retrieved_experiences
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    def get_all_experiences_by_filter(self, 
                                    conflict_type: Optional[str] = None, 
                                    num_aircraft: Optional[int] = None,
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all experiences matching the given filters
        
        Args:
            conflict_type: Filter by conflict type
            num_aircraft: Filter by number of aircraft
            limit: Maximum number of results
            
        Returns:
            List of experience dictionaries
        """
        try:
            where_filter = {}
            
            if conflict_type:
                where_filter["conflict_type"] = {"$eq": conflict_type}
            if num_aircraft:
                where_filter["num_aircraft"] = {"$eq": num_aircraft}
            
            if where_filter:
                results = self.collection.get(
                    where=where_filter,
                    limit=limit,
                    include=['documents', 'metadatas']
                )
            else:
                results = self.collection.get(
                    limit=limit,
                    include=['documents', 'metadatas']
                )
            
            experiences = []
            if results['ids']:
                for i, exp_id in enumerate(results['ids']):
                    experience_data = self._reconstruct_experience_data(
                        exp_id,
                        results['documents'][i] if i < len(results['documents']) else "",
                        results['metadatas'][i] if i < len(results['metadatas']) else {}
                    )
                    experiences.append(experience_data)
            
            return experiences
            
        except Exception as e:
            self.logger.error(f"Failed to get filtered experiences: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored experiences"""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    'total_experiences': 0,
                    'embedding_dimension': 3072,
                    'storage_backend': 'Chroma HNSW'
                }
            
            # Get sample of experiences for statistics
            sample_results = self.collection.get(
                limit=min(100, count),
                include=['metadatas']
            )
            
            stats = {
                'total_experiences': count,
                'embedding_dimension': 3072,
                'storage_backend': 'Chroma HNSW',
                'conflict_type_distribution': {},
                'aircraft_count_distribution': {},
                'hallucination_rate': 0
            }
            
            if sample_results['metadatas']:
                hallucination_count = 0
                for metadata in sample_results['metadatas']:
                    # Count conflict types
                    conflict_type = metadata.get('conflict_type', 'unknown')
                    stats['conflict_type_distribution'][conflict_type] = \
                        stats['conflict_type_distribution'].get(conflict_type, 0) + 1
                    
                    # Count aircraft numbers
                    num_aircraft = metadata.get('num_aircraft', 0)
                    stats['aircraft_count_distribution'][num_aircraft] = \
                        stats['aircraft_count_distribution'].get(num_aircraft, 0) + 1
                    
                    # Count hallucinations
                    if metadata.get('hallucination_detected', False):
                        hallucination_count += 1
                
                stats['hallucination_rate'] = hallucination_count / len(sample_results['metadatas'])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def _create_embedding_text(self, experience_data: Dict[str, Any]) -> str:
        """Create comprehensive text for embedding from experience data"""
        text_parts = []
        
        # Add conflict information
        text_parts.append(f"Conflict: {experience_data.get('conflict_type', 'unknown')}")
        text_parts.append(f"Aircraft: {experience_data.get('num_aircraft', 0)}")
        
        # Add scenario description if available
        scenario = experience_data.get('scenario', {})
        if isinstance(scenario, dict):
            for key, value in scenario.items():
                if isinstance(value, str) and value:
                    text_parts.append(f"{key}: {value}")
        
        # Add action/decision information
        action = experience_data.get('action', '')
        if action:
            text_parts.append(f"Action: {action}")
        
        # Add outcome information
        outcome = experience_data.get('outcome', {})
        if isinstance(outcome, dict):
            for key, value in outcome.items():
                if isinstance(value, (str, bool)) and value:
                    text_parts.append(f"{key}: {value}")
        
        # Add any additional text fields
        for field in ['description', 'lessons_learned', 'notes']:
            if field in experience_data and experience_data[field]:
                text_parts.append(str(experience_data[field]))
        
        return " | ".join(filter(None, text_parts))
    
    def _extract_metadata(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for filtering from experience data"""
        metadata = {}
        
        # Required fields for filtering
        metadata['conflict_type'] = experience_data.get('conflict_type', 'unknown')
        metadata['num_aircraft'] = experience_data.get('num_aircraft', 0)
        metadata['timestamp'] = experience_data.get('timestamp', time.time())
        
        # Outcome fields
        outcome = experience_data.get('outcome', {})
        if isinstance(outcome, dict):
            metadata['safety_margin'] = outcome.get('safety_margin', 0.0)
            metadata['icao_compliant'] = outcome.get('icao_compliant', False)
            metadata['hallucination_detected'] = outcome.get('hallucination_detected', False)
        
        # Scenario metadata
        scenario_metadata = experience_data.get('metadata', {})
        if isinstance(scenario_metadata, dict):
            metadata.update({
                'weather': scenario_metadata.get('weather', 'unknown'),
                'resolution_success': scenario_metadata.get('resolution_success', False)
            })
        
        return metadata
    
    def _reconstruct_experience_data(self, exp_id: str, document: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct experience data from stored document and metadata"""
        return {
            'experience_id': exp_id,
            'conflict_type': metadata.get('conflict_type', 'unknown'),
            'num_aircraft': metadata.get('num_aircraft', 0),
            'timestamp': metadata.get('timestamp', 0),
            'description': document,
            'outcome': {
                'safety_margin': metadata.get('safety_margin', 0.0),
                'icao_compliant': metadata.get('icao_compliant', False),
                'hallucination_detected': metadata.get('hallucination_detected', False),
            },
            'metadata': {
                'weather': metadata.get('weather', 'unknown'),
                'resolution_success': metadata.get('resolution_success', False)
            }
        }
    
    def delete_experience(self, experience_id: str) -> bool:
        """Delete an experience by ID"""
        try:
            self.collection.delete(ids=[experience_id])
            self.logger.info(f"Deleted experience {experience_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete experience {experience_id}: {e}")
            return False
    
    def clear_all_experiences(self) -> bool:
        """Clear all stored experiences (use with caution)"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(self.collection_name)
            
            # Recreate collection
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-large"
                )
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            self.logger.info("Cleared all experiences and recreated collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear experiences: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create replay store
    store = VectorReplayStore()
    
    # Sample experience data
    sample_experience = {
        'experience_id': 'test_001',
        'conflict_type': 'convergent',
        'num_aircraft': 2,
        'scenario': {
            'description': 'Two aircraft approaching at same altitude',
            'aircraft_1': 'B737 at FL350',
            'aircraft_2': 'A320 at FL350'
        },
        'action': 'Turn left 15 degrees',
        'outcome': {
            'safety_margin': 0.85,
            'icao_compliant': True,
            'hallucination_detected': False
        },
        'metadata': {
            'weather': 'clear',
            'resolution_success': True
        }
    }
    
    # Store experience
    success = store.store_experience(sample_experience)
    print(f"Store success: {success}")
    
    # Retrieve similar experiences
    similar = store.retrieve_experience(
        conflict_desc="Two aircraft on collision course at same altitude",
        conflict_type="convergent",
        num_ac=2,
        k=5
    )
    
    print(f"Found {len(similar)} similar experiences")
    for exp in similar:
        print(f"  - {exp.experience_id}: similarity {exp.similarity_score:.3f}")
    
    # Get statistics
    stats = store.get_statistics()
    print(f"Store statistics: {stats}")
