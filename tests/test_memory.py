# tests/test_memory.py
"""
Test suite for memory system with 3072-dim embeddings and Chroma storage
"""

import os
import tempfile
import time
import pytest
from typing import Dict, Any

# Add parent directory for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.experience_document_generator import ExperienceDocumentGenerator, ExperienceDocument
from memory.replay_store import VectorReplayStore, RetrievedExperience


class TestExperienceDocumentGenerator:
    """Test experience document generation and embedding"""
    
    @pytest.fixture
    def doc_generator(self):
        """Create document generator for testing"""
        # Use temp directory for testing
        return ExperienceDocumentGenerator()
    
    @pytest.fixture
    def sample_experience_data(self):
        """Sample experience data for testing"""
        return {
            'scenario_context': {
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
            },
            'conflict_geometry': {
                'time_to_closest_approach': 120,
                'closest_approach_distance': 4.2,
                'closest_approach_altitude_diff': 0
            },
            'environmental_conditions': {
                'weather': 'clear',
                'wind_speed': 15,
                'visibility': 10.0,
                'turbulence_intensity': 0.1
            },
            'llm_decision': {
                'action': 'turn left 15 degrees',
                'type': 'heading',
                'confidence': 0.8,
                'safety_score': 0.85
            },
            'baseline_decision': {
                'action': 'climb 1000 feet',
                'type': 'altitude',
                'confidence': 0.9,
                'safety_score': 0.9
            },
            'actual_outcome': {
                'resolution_success': True,
                'separation_achieved': 6.5,
                'time_to_resolution': 95
            },
            'safety_metrics': {
                'effective_margin': 0.82,
                'icao_compliant': True
            }
        }
    
    def test_document_generation(self, doc_generator, sample_experience_data):
        """Test basic document generation"""
        exp_doc = doc_generator.generate_experience(
            **sample_experience_data,
            lessons_learned="Heading changes effective for convergent conflicts"
        )
        
        # Check required fields
        assert 'experience_id' in exp_doc
        assert 'conflict_type' in exp_doc
        assert 'num_aircraft' in exp_doc
        assert 'scenario_text' in exp_doc
        assert 'timestamp' in exp_doc
        
        # Check values
        assert exp_doc['num_aircraft'] == 2
        assert exp_doc['conflict_type'] in ['convergent', 'parallel', 'crossing', 'overtaking']
        assert exp_doc['safety_margin'] == 0.82
        assert exp_doc['icao_compliant'] is True
        assert exp_doc['lessons_learned'] == "Heading changes effective for convergent conflicts"
    
    def test_conflict_type_determination(self, doc_generator):
        """Test conflict type determination logic"""
        # Same altitude, close approach -> convergent
        conflict_geometry = {
            'closest_approach_distance': 1.5,
            'time_to_closest_approach': 60
        }
        aircraft_list = [
            {'altitude': 35000}, {'altitude': 35000}
        ]
        
        conflict_type = doc_generator._determine_conflict_type(conflict_geometry, aircraft_list)
        assert conflict_type == 'convergent'
        
        # Different altitudes -> crossing
        aircraft_list = [
            {'altitude': 35000}, {'altitude': 37000}
        ]
        
        conflict_type = doc_generator._determine_conflict_type(conflict_geometry, aircraft_list)
        assert conflict_type == 'crossing'
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key required")
    def test_embedding_and_storage(self, sample_experience_data):
        """Test embedding generation and storage (requires OpenAI API key)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_generator = ExperienceDocumentGenerator()
            doc_generator.chroma_client.path = temp_dir
            
            # Generate document
            exp_doc = doc_generator.generate_experience(**sample_experience_data)
            
            # Test embedding and storage
            doc_generator.embed_and_store(exp_doc)
            
            # Verify storage
            stats = doc_generator.get_collection_stats()
            assert stats['total_experiences'] == 1
            assert stats['embedding_dimension'] == 3072


class TestVectorReplayStore:
    """Test vector replay store functionality"""
    
    @pytest.fixture
    def temp_store(self):
        """Create temporary store for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorReplayStore(storage_dir=temp_dir)
            yield store
    
    @pytest.fixture
    def sample_experiences(self):
        """Sample experiences for testing"""
        return [
            {
                'experience_id': 'test_001',
                'conflict_type': 'convergent',
                'num_aircraft': 2,
                'scenario': {
                    'description': 'Two aircraft approaching at same altitude'
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
            },
            {
                'experience_id': 'test_002',
                'conflict_type': 'parallel',
                'num_aircraft': 2,
                'scenario': {
                    'description': 'Two aircraft flying parallel tracks'
                },
                'action': 'Maintain course',
                'outcome': {
                    'safety_margin': 0.90,
                    'icao_compliant': True,
                    'hallucination_detected': False
                },
                'metadata': {
                    'weather': 'cloudy',
                    'resolution_success': True
                }
            }
        ]
    
    def test_store_experience(self, temp_store, sample_experiences):
        """Test storing experiences"""
        for exp in sample_experiences:
            success = temp_store.store_experience(exp)
            assert success is True
        
        # Check statistics
        stats = temp_store.get_statistics()
        assert stats['total_experiences'] == 2
        assert stats['embedding_dimension'] == 3072
        assert stats['storage_backend'] == 'Chroma HNSW'
    
    def test_retrieve_experience_metadata_filtering(self, temp_store, sample_experiences):
        """Test experience retrieval with metadata filtering"""
        # Store experiences
        for exp in sample_experiences:
            temp_store.store_experience(exp)
        
        # Retrieve convergent conflicts
        results = temp_store.retrieve_experience(
            conflict_desc="Aircraft approaching collision",
            conflict_type="convergent", 
            num_ac=2,
            k=5
        )
        
        assert len(results) >= 1
        assert all(isinstance(r, RetrievedExperience) for r in results)
        assert all(r.metadata.get('conflict_type') == 'convergent' for r in results)
        assert all(r.metadata.get('num_aircraft') == 2 for r in results)
    
    def test_get_filtered_experiences(self, temp_store, sample_experiences):
        """Test getting experiences with filters"""
        # Store experiences
        for exp in sample_experiences:
            temp_store.store_experience(exp)
        
        # Get all convergent conflicts
        convergent_exps = temp_store.get_all_experiences_by_filter(
            conflict_type="convergent"
        )
        
        assert len(convergent_exps) == 1
        assert convergent_exps[0]['conflict_type'] == 'convergent'
        
        # Get all 2-aircraft conflicts
        two_ac_exps = temp_store.get_all_experiences_by_filter(
            num_aircraft=2
        )
        
        assert len(two_ac_exps) == 2
    
    def test_embedding_text_generation(self, temp_store):
        """Test embedding text generation"""
        experience = {
            'conflict_type': 'convergent',
            'num_aircraft': 2,
            'action': 'Turn left',
            'outcome': {
                'safety_margin': 0.8,
                'icao_compliant': True
            }
        }
        
        embedding_text = temp_store._create_embedding_text(experience)
        
        assert 'convergent' in embedding_text
        assert '2' in embedding_text
        assert 'Turn left' in embedding_text
        assert len(embedding_text) > 0
    
    def test_metadata_extraction(self, temp_store):
        """Test metadata extraction for filtering"""
        experience = {
            'conflict_type': 'parallel',
            'num_aircraft': 3,
            'outcome': {
                'safety_margin': 0.9,
                'icao_compliant': True,
                'hallucination_detected': False
            },
            'metadata': {
                'weather': 'stormy',
                'resolution_success': True
            }
        }
        
        metadata = temp_store._extract_metadata(experience)
        
        assert metadata['conflict_type'] == 'parallel'
        assert metadata['num_aircraft'] == 3
        assert metadata['safety_margin'] == 0.9
        assert metadata['icao_compliant'] is True
        assert metadata['weather'] == 'stormy'
    
    def test_delete_experience(self, temp_store, sample_experiences):
        """Test experience deletion"""
        # Store experience
        temp_store.store_experience(sample_experiences[0])
        
        # Verify it exists
        stats = temp_store.get_statistics()
        assert stats['total_experiences'] == 1
        
        # Delete it
        success = temp_store.delete_experience('test_001')
        assert success is True
        
        # Verify deletion
        stats = temp_store.get_statistics()
        assert stats['total_experiences'] == 0


class TestEmbeddingDimensions:
    """Test that embeddings are 3072-dimensional"""
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key required")
    def test_embedding_dimensions(self):
        """Test that embeddings are indeed 3072-dimensional"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create store
            store = VectorReplayStore(storage_dir=temp_dir)
            
            # Store a dummy experience
            dummy_exp = {
                'experience_id': 'dim_test',
                'conflict_type': 'convergent',
                'num_aircraft': 2,
                'description': 'Test experience for dimension verification'
            }
            
            success = store.store_experience(dummy_exp)
            assert success is True
            
            # Verify statistics report 3072 dimensions
            stats = store.get_statistics()
            assert stats['embedding_dimension'] == 3072


def test_memory_system_integration():
    """Test the complete memory system workflow"""
    # Store two dummy experiences
    experiences = [
        {
            'experience_id': 'test_1',
            'conflict_type': 'convergent',
            'num_aircraft': 2,
            'description': 'Two aircraft on collision course',
            'outcome': {
                'safety_margin': 0.8,
                'icao_compliant': True,
                'hallucination_detected': False
            }
        },
        {
            'experience_id': 'test_2', 
            'conflict_type': 'parallel',
            'num_aircraft': 2,
            'description': 'Two aircraft flying parallel paths',
            'outcome': {
                'safety_margin': 0.9,
                'icao_compliant': True,
                'hallucination_detected': False
            }
        }
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorReplayStore(storage_dir=temp_dir)
        
        # Store experiences
        for exp in experiences:
            success = store.store_experience(exp)
            assert success is True
        
        # Retrieve them
        results = store.retrieve_experience(
            conflict_desc="Aircraft collision scenario",
            conflict_type="convergent",
            num_ac=2,
            k=5
        )
        
        # Assert retrieval works
        assert len(results) >= 1
        assert all(hasattr(r, 'similarity_score') for r in results)
        
        # Verify 3072-dim vectors
        stats = store.get_statistics()
        assert stats['embedding_dimension'] == 3072
        assert stats['total_experiences'] == 2


# Run basic tests if executed directly
if __name__ == "__main__":
    print("Running basic memory system tests...")
    
    # Test basic functionality without OpenAI API
    test_memory_system_integration()
    print("✅ Basic integration test passed")
    
    # Test document generation
    doc_gen = ExperienceDocumentGenerator()
    sample_data = {
        'scenario_context': {
            'aircraft_list': [
                {'aircraft_type': 'B737', 'altitude': 35000},
                {'aircraft_type': 'A320', 'altitude': 35000}
            ]
        },
        'conflict_geometry': {'closest_approach_distance': 2.0},
        'environmental_conditions': {'weather': 'clear'},
        'llm_decision': {'action': 'turn left'},
        'baseline_decision': {'action': 'climb'},
        'actual_outcome': {'resolution_success': True},
        'safety_metrics': {'effective_margin': 0.8, 'icao_compliant': True}
    }
    
    exp_doc = doc_gen.generate_experience(**sample_data)
    assert 'experience_id' in exp_doc
    assert exp_doc['num_aircraft'] == 2
    print("✅ Document generation test passed")
    
    print("\n🎉 All basic memory tests passed!")
    print("Note: Full embedding tests require OPENAI_API_KEY environment variable")
    """Test suite for the memory system"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_experience_data(self):
        """Sample experience data for testing"""
        return {
            'scenario_context': {
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
            },
            'conflict_geometry': {
                'time_to_closest_approach': 120,
                'closest_approach_distance': 4.2,
                'closest_approach_altitude_diff': 0
            },
            'environmental_conditions': {
                'weather': 'clear',
                'wind_speed': 15,
                'visibility': 10.0,
                'turbulence_intensity': 0.1
            },
            'llm_decision': {
                'action': 'turn left 15 degrees',
                'type': 'heading',
                'confidence': 0.8,
                'safety_score': 0.85
            },
            'baseline_decision': {
                'action': 'climb 1000 feet',
                'type': 'altitude',
                'confidence': 0.9,
                'safety_score': 0.9
            },
            'actual_outcome': {
                'resolution_success': True,
                'separation_achieved': 6.5,
                'time_to_resolution': 95
            },
            'safety_metrics': {
                'effective_margin': 0.82,
                'icao_compliant': True
            }
        }
    
    @pytest.fixture
    def second_experience_data(self):
        """Second sample experience data for testing"""
        return {
            'scenario_context': {
                'aircraft_list': [
                    {
                        'aircraft_type': 'A380',
                        'callsign': 'EK001',
                        'altitude': 37000,
                        'speed': 480,
                        'heading': 180
                    },
                    {
                        'aircraft_type': 'B777',
                        'callsign': 'AF123',
                        'altitude': 37000,
                        'speed': 460,
                        'heading': 000
                    }
                ]
            },
            'conflict_geometry': {
                'time_to_closest_approach': 90,
                'closest_approach_distance': 3.8,
                'closest_approach_altitude_diff': 0
            },
            'environmental_conditions': {
                'weather': 'cloudy',
                'wind_speed': 25,
                'visibility': 8.0,
                'turbulence_intensity': 0.3
            },
            'llm_decision': {
                'action': 'descend 2000 feet',
                'type': 'altitude',
                'confidence': 0.9,
                'safety_score': 0.88
            },
            'baseline_decision': {
                'action': 'turn right 20 degrees',
                'type': 'heading',
                'confidence': 0.85,
                'safety_score': 0.82
            },
            'actual_outcome': {
                'resolution_success': True,
                'separation_achieved': 7.2,
                'time_to_resolution': 75
            },
            'safety_metrics': {
                'effective_margin': 0.91,
                'icao_compliant': True
            }
        }
    
    def test_experience_document_generator_creation(self, sample_experience_data):
        """Test experience document generation"""
        
        # Create document generator (without API key for testing)
        doc_generator = ExperienceDocumentGenerator(openai_api_key="test_key")
        
        # Generate experience document
        exp_doc = doc_generator.generate_experience(**sample_experience_data)
        
        # Verify document structure
        assert isinstance(exp_doc, dict)
        assert 'experience_id' in exp_doc
        assert 'timestamp' in exp_doc
        assert 'conflict_type' in exp_doc
        assert 'num_aircraft' in exp_doc
        assert exp_doc['num_aircraft'] == 2
        assert exp_doc['conflict_type'] in ['convergent', 'parallel', 'crossing', 'overtaking']
        
        # Verify text descriptions are generated
        assert 'scenario_text' in exp_doc
        assert 'conflict_geometry_text' in exp_doc
        assert 'environmental_text' in exp_doc
        assert 'llm_decision_text' in exp_doc
        assert 'baseline_decision_text' in exp_doc
        assert 'outcome_text' in exp_doc
        
        # Verify metadata
        assert 'metadata' in exp_doc
        assert 'aircraft_types' in exp_doc['metadata']
        assert 'altitude_levels' in exp_doc['metadata']
        assert 'weather' in exp_doc['metadata']
        
        print(f"✅ Experience document generated successfully: {exp_doc['experience_id']}")
    
    def test_vector_replay_store_initialization(self, temp_storage_dir):
        """Test vector replay store initialization"""
        
        # Create replay store with temporary directory
        replay_store = VectorReplayStore(storage_dir=temp_storage_dir, openai_api_key="test_key")
        
        # Verify initialization
        assert replay_store.storage_dir == temp_storage_dir
        assert replay_store.collection_name == "atc_experiences"
        assert hasattr(replay_store, 'collection')
        assert hasattr(replay_store, 'experiences')
        
        # Verify storage directory exists
        assert os.path.exists(temp_storage_dir)
        
        print(f"✅ Vector replay store initialized successfully")
    
    def test_store_two_dummy_experiences(self, temp_storage_dir, sample_experience_data, second_experience_data):
        """Test storing two dummy experiences and verifying 3072-dim vectors"""
        
        # Create replay store
        replay_store = VectorReplayStore(storage_dir=temp_storage_dir, openai_api_key="test_key")
        
        # Create ConflictExperience objects
        experience1 = ConflictExperience(
            experience_id="test_exp_1",
            timestamp=time.time(),
            scenario_context=sample_experience_data['scenario_context'],
            conflict_geometry=sample_experience_data['conflict_geometry'],
            environmental_conditions=sample_experience_data['environmental_conditions'],
            llm_decision=sample_experience_data['llm_decision'],
            baseline_decision=sample_experience_data['baseline_decision'],
            actual_outcome=sample_experience_data['actual_outcome'],
            safety_metrics=sample_experience_data['safety_metrics'],
            hallucination_detected=False,
            hallucination_types=[],
            controller_override=None,
            lessons_learned="Test experience 1 - heading changes effective"
        )
        
        experience2 = ConflictExperience(
            experience_id="test_exp_2",
            timestamp=time.time(),
            scenario_context=second_experience_data['scenario_context'],
            conflict_geometry=second_experience_data['conflict_geometry'],
            environmental_conditions=second_experience_data['environmental_conditions'],
            llm_decision=second_experience_data['llm_decision'],
            baseline_decision=second_experience_data['baseline_decision'],
            actual_outcome=second_experience_data['actual_outcome'],
            safety_metrics=second_experience_data['safety_metrics'],
            hallucination_detected=False,
            hallucination_types=[],
            controller_override=None,
            lessons_learned="Test experience 2 - altitude changes preferred"
        )
        
        # Store both experiences
        exp_id_1 = replay_store.store_experience(experience1)
        exp_id_2 = replay_store.store_experience(experience2)
        
        # Verify storage
        assert exp_id_1 == "test_exp_1"
        assert exp_id_2 == "test_exp_2"
        assert len(replay_store.experiences) == 2
        
        print(f"✅ Stored two experiences: {exp_id_1}, {exp_id_2}")
        
        # Get statistics to verify embedding dimension
        stats = replay_store.get_statistics()
        
        # Verify 3072-dimensional embeddings
        assert stats['embedding_dimension'] == 3072
        assert stats['storage_backend'] == 'Chroma HNSW'
        assert stats['total_experiences'] == 2
        
        print(f"✅ Verified 3072-dimensional embeddings: {stats['embedding_dimension']}")
        print(f"✅ Storage backend confirmed: {stats['storage_backend']}")
        
        return replay_store, exp_id_1, exp_id_2
    
    def test_retrieve_experiences(self, temp_storage_dir, sample_experience_data, second_experience_data):
        """Test retrieving stored experiences"""
        
        # Store experiences first
        replay_store, exp_id_1, exp_id_2 = self.test_store_two_dummy_experiences(
            temp_storage_dir, sample_experience_data, second_experience_data
        )
        
        # Test retrieval with conflict description
        retrieved = replay_store.retrieve_experience(
            conflict_desc="Aircraft on collision course at same altitude level",
            k=2
        )
        
        # Verify retrieval works
        assert isinstance(retrieved, list)
        assert len(retrieved) <= 2  # Should return at most 2 experiences
        
        # Check retrieved experience structure
        if retrieved:
            exp = retrieved[0]
            assert 'experience_id' in exp
            assert 'similarity_score' in exp
            assert 'metadata' in exp
            assert 'full_experience' in exp
            
            # Verify similarity score is reasonable
            assert 0.0 <= exp['similarity_score'] <= 1.0
        
        print(f"✅ Retrieved {len(retrieved)} experiences successfully")
        
        # Test retrieval with metadata filtering
        filtered_retrieved = replay_store.retrieve_experience(
            conflict_desc="Aircraft conflict scenario",
            conflict_type="convergent",
            num_ac=2,
            k=5
        )
        
        assert isinstance(filtered_retrieved, list)
        print(f"✅ Filtered retrieval returned {len(filtered_retrieved)} experiences")
    
    def test_find_similar_experiences_compatibility(self, temp_storage_dir, sample_experience_data, second_experience_data):
        """Test find_similar_experiences method for compatibility"""
        
        # Store experiences first
        replay_store, exp_id_1, exp_id_2 = self.test_store_two_dummy_experiences(
            temp_storage_dir, sample_experience_data, second_experience_data
        )
        
        # Create query experience
        query_experience = ConflictExperience(
            experience_id="query_exp",
            timestamp=time.time(),
            scenario_context={
                'aircraft_list': [
                    {'aircraft_type': 'B787', 'altitude': 35000},
                    {'aircraft_type': 'A350', 'altitude': 35000}
                ]
            },
            conflict_geometry={
                'time_to_closest_approach': 100,
                'closest_approach_distance': 4.0,
                'closest_approach_altitude_diff': 0
            },
            environmental_conditions={'weather': 'clear'},
            llm_decision={'action': 'turn left', 'type': 'heading'},
            baseline_decision={'action': 'climb', 'type': 'altitude'},
            actual_outcome={'resolution_success': True},
            safety_metrics={'effective_margin': 0.8},
            hallucination_detected=False,
            hallucination_types=[],
            controller_override=None,
            lessons_learned=""
        )
        
        # Find similar experiences
        similar = replay_store.find_similar_experiences(
            query_experience=query_experience,
            top_k=3,
            similarity_threshold=0.0  # Low threshold for testing
        )
        
        # Verify results
        assert isinstance(similar, list)
        print(f"✅ Found {len(similar)} similar experiences using compatibility method")
        
        # Check structure of similarity results
        if similar:
            result = similar[0]
            assert hasattr(result, 'experience')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'relevant_aspects')
            assert isinstance(result.experience, ConflictExperience)
    
    def test_memory_system_integration(self, temp_storage_dir):
        """Test integration between document generator and replay store"""
        
        # Create both components
        doc_generator = ExperienceDocumentGenerator(openai_api_key="test_key")
        replay_store = VectorReplayStore(storage_dir=temp_storage_dir, openai_api_key="test_key")
        
        # Sample experience data
        experience_data = {
            'scenario_context': {
                'aircraft_list': [
                    {'aircraft_type': 'B737', 'callsign': 'TEST001', 'altitude': 36000},
                    {'aircraft_type': 'A320', 'callsign': 'TEST002', 'altitude': 36000}
                ]
            },
            'conflict_geometry': {
                'time_to_closest_approach': 150,
                'closest_approach_distance': 5.0,
                'closest_approach_altitude_diff': 0
            },
            'environmental_conditions': {
                'weather': 'storm',
                'wind_speed': 40,
                'visibility': 3.0
            },
            'llm_decision': {'action': 'emergency descent', 'type': 'altitude'},
            'baseline_decision': {'action': 'vector away', 'type': 'heading'},
            'actual_outcome': {'resolution_success': False},
            'safety_metrics': {'effective_margin': 0.3, 'icao_compliant': False},
            'hallucination_detected': True,
            'hallucination_types': ['invalid_command', 'unrealistic_parameters'],
            'lessons_learned': "Severe weather conditions require conservative approaches"
        }
        
        # Generate document
        exp_doc = doc_generator.generate_experience(**experience_data)
        
        # Convert to ConflictExperience and store
        conflict_exp = ConflictExperience(
            experience_id=exp_doc['experience_id'],
            timestamp=exp_doc['timestamp'],
            scenario_context=experience_data['scenario_context'],
            conflict_geometry=experience_data['conflict_geometry'],
            environmental_conditions=experience_data['environmental_conditions'],
            llm_decision=experience_data['llm_decision'],
            baseline_decision=experience_data['baseline_decision'],
            actual_outcome=experience_data['actual_outcome'],
            safety_metrics=experience_data['safety_metrics'],
            hallucination_detected=experience_data['hallucination_detected'],
            hallucination_types=experience_data['hallucination_types'],
            controller_override=None,
            lessons_learned=experience_data['lessons_learned']
        )
        
        # Store in replay store
        stored_id = replay_store.store_experience(conflict_exp)
        
        # Verify integration
        assert stored_id == exp_doc['experience_id']
        
        # Test retrieval
        retrieved = replay_store.retrieve_experience(
            conflict_desc="Emergency situation with severe weather",
            k=1
        )
        
        assert len(retrieved) > 0
        assert retrieved[0]['experience_id'] == stored_id
        
        print(f"✅ Memory system integration test passed: {stored_id}")


def run_memory_tests():
    """Run all memory system tests"""
    
    print("Running Memory System Tests")
    print("=" * 50)
    
    # Create test instance
    test_instance = TestMemorySystem()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Sample data
        sample_data = {
            'scenario_context': {
                'aircraft_list': [
                    {'aircraft_type': 'B737', 'callsign': 'UAL123', 'altitude': 35000},
                    {'aircraft_type': 'A320', 'callsign': 'DAL456', 'altitude': 35000}
                ]
            },
            'conflict_geometry': {'time_to_closest_approach': 120, 'closest_approach_distance': 4.2},
            'environmental_conditions': {'weather': 'clear', 'wind_speed': 15},
            'llm_decision': {'action': 'turn left 15 degrees', 'type': 'heading'},
            'baseline_decision': {'action': 'climb 1000 feet', 'type': 'altitude'},
            'actual_outcome': {'resolution_success': True},
            'safety_metrics': {'effective_margin': 0.82, 'icao_compliant': True}
        }
        
        second_data = {
            'scenario_context': {
                'aircraft_list': [
                    {'aircraft_type': 'A380', 'callsign': 'EK001', 'altitude': 37000},
                    {'aircraft_type': 'B777', 'callsign': 'AF123', 'altitude': 37000}
                ]
            },
            'conflict_geometry': {'time_to_closest_approach': 90, 'closest_approach_distance': 3.8},
            'environmental_conditions': {'weather': 'cloudy', 'wind_speed': 25},
            'llm_decision': {'action': 'descend 2000 feet', 'type': 'altitude'},
            'baseline_decision': {'action': 'turn right 20 degrees', 'type': 'heading'},
            'actual_outcome': {'resolution_success': True},
            'safety_metrics': {'effective_margin': 0.91, 'icao_compliant': True}
        }
        
        # Run tests
        print("\n1. Testing Experience Document Generation...")
        test_instance.test_experience_document_generator_creation(sample_data)
        
        print("\n2. Testing Vector Replay Store Initialization...")
        test_instance.test_vector_replay_store_initialization(temp_dir)
        
        print("\n3. Testing Storage of Two Dummy Experiences...")
        test_instance.test_store_two_dummy_experiences(temp_dir, sample_data, second_data)
        
        print("\n4. Testing Experience Retrieval...")
        test_instance.test_retrieve_experiences(temp_dir, sample_data, second_data)
        
        print("\n5. Testing Compatibility Methods...")
        test_instance.test_find_similar_experiences_compatibility(temp_dir, sample_data, second_data)
        
        print("\n6. Testing Memory System Integration...")
        test_instance.test_memory_system_integration(temp_dir)
        
        print("\n" + "=" * 50)
        print("✅ ALL MEMORY TESTS PASSED!")
        print("✅ 3072-dimensional embeddings verified")
        print("✅ Chroma HNSW storage confirmed")
        print("✅ Experience retrieval working")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    run_memory_tests()
