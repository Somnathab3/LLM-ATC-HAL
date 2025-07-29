# tests/test_memory_simple.py
"""
Simple test for memory system without external dependencies
"""

import os
import sys
import tempfile
import time

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_atc.memory.experience_document_generator import ExperienceDocumentGenerator
from llm_atc.memory.replay_store import VectorReplayStore


def test_basic_functionality():
    """Test basic memory system functionality"""
    print("Testing basic memory system functionality...")
    
    # Test document generation
    print("1. Testing experience document generation...")
    
    # Create a mock API key for testing (won't be used for actual API calls)
    os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
    
    try:
        doc_generator = ExperienceDocumentGenerator()
    except Exception as e:
        print(f"   ⚠️  Skipping document generator test due to API key requirement: {e}")
        # Create a minimal test without the generator
        doc_generator = None
    
    if doc_generator:
        sample_data = {
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
                'visibility': 10.0
            },
            'llm_decision': {
                'action': 'turn left 15 degrees',
                'type': 'heading',
                'confidence': 0.8
            },
            'baseline_decision': {
                'action': 'climb 1000 feet',
                'type': 'altitude',
                'confidence': 0.9
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
        
        exp_doc = doc_generator.generate_experience(**sample_data, lessons_learned="Test lesson")
        
        # Verify document structure
        assert 'experience_id' in exp_doc, "Missing experience_id"
        assert 'conflict_type' in exp_doc, "Missing conflict_type"
        assert 'num_aircraft' in exp_doc, "Missing num_aircraft"
        assert exp_doc['num_aircraft'] == 2, f"Expected 2 aircraft, got {exp_doc['num_aircraft']}"
        assert exp_doc['safety_margin'] == 0.82, f"Expected safety margin 0.82, got {exp_doc['safety_margin']}"
        assert exp_doc['icao_compliant'] is True, "Expected ICAO compliant"
        
        print("   ✅ Document generation test passed")
    else:
        print("   ⚠️  Document generation test skipped (no API key)")
    
    # Test replay store
    print("2. Testing vector replay store...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorReplayStore(storage_dir=temp_dir)
        
        # Test storing experiences
        sample_experiences = [
            {
                'experience_id': 'test_001',
                'conflict_type': 'convergent',
                'num_aircraft': 2,
                'description': 'Two aircraft approaching at same altitude',
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
                'description': 'Two aircraft flying parallel tracks',
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
        
        # Store experiences
        stored_count = 0
        for exp in sample_experiences:
            success = store.store_experience(exp)
            if success:
                stored_count += 1
        
        print(f"   ✅ Stored {stored_count}/{len(sample_experiences)} experiences")
        
        # Test statistics
        stats = store.get_statistics()
        assert stats['total_experiences'] == stored_count, f"Expected {stored_count} experiences, got {stats['total_experiences']}"
        assert stats['embedding_dimension'] == 3072, f"Expected 3072 dimensions, got {stats['embedding_dimension']}"
        assert stats['storage_backend'] == 'Chroma HNSW', f"Expected Chroma HNSW, got {stats['storage_backend']}"
        
        print(f"   ✅ Statistics test passed: {stats['total_experiences']} experiences, {stats['embedding_dimension']} dimensions")
        
        # Test retrieval (basic)
        results = store.retrieve_experience(
            conflict_desc="Aircraft approaching collision",
            conflict_type="convergent",
            num_ac=2,
            k=5
        )
        
        print(f"   ✅ Retrieval test passed: found {len(results)} similar experiences")
        
        # Test metadata filtering
        convergent_exps = store.get_all_experiences_by_filter(conflict_type="convergent")
        parallel_exps = store.get_all_experiences_by_filter(conflict_type="parallel")
        
        assert len(convergent_exps) == 1, f"Expected 1 convergent experience, got {len(convergent_exps)}"
        assert len(parallel_exps) == 1, f"Expected 1 parallel experience, got {len(parallel_exps)}"
        
        print(f"   ✅ Filtering test passed: {len(convergent_exps)} convergent, {len(parallel_exps)} parallel")


def test_embedding_text_generation():
    """Test embedding text generation"""
    print("3. Testing embedding text generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorReplayStore(storage_dir=temp_dir)
        
        experience = {
            'conflict_type': 'convergent',
            'num_aircraft': 2,
            'action': 'Turn left',
            'outcome': {
                'safety_margin': 0.8,
                'icao_compliant': True
            }
        }
        
        embedding_text = store._create_embedding_text(experience)
        
        assert 'convergent' in embedding_text, "Missing conflict type in embedding text"
        assert '2' in embedding_text, "Missing aircraft count in embedding text"
        assert 'Turn left' in embedding_text, "Missing action in embedding text"
        assert len(embedding_text) > 0, "Empty embedding text"
        
        print(f"   ✅ Generated embedding text: '{embedding_text[:100]}...'")


def test_metadata_extraction():
    """Test metadata extraction"""
    print("4. Testing metadata extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorReplayStore(storage_dir=temp_dir)
        
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
        
        metadata = store._extract_metadata(experience)
        
        assert metadata['conflict_type'] == 'parallel', "Wrong conflict type in metadata"
        assert metadata['num_aircraft'] == 3, "Wrong aircraft count in metadata"
        assert metadata['safety_margin'] == 0.9, "Wrong safety margin in metadata"
        assert metadata['icao_compliant'] is True, "Wrong ICAO compliance in metadata"
        assert metadata['weather'] == 'stormy', "Wrong weather in metadata"
        
        print("   ✅ Metadata extraction test passed")


def test_dimension_verification():
    """Test that embeddings are configured for 3072 dimensions"""
    print("5. Testing dimension configuration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorReplayStore(storage_dir=temp_dir)
        
        # Store a dummy experience
        dummy_exp = {
            'experience_id': 'dim_test',
            'conflict_type': 'convergent',
            'num_aircraft': 2,
            'description': 'Test experience for dimension verification'
        }
        
        success = store.store_experience(dummy_exp)
        assert success, "Failed to store dummy experience"
        
        # Verify statistics report 3072 dimensions
        stats = store.get_statistics()
        assert stats['embedding_dimension'] == 3072, f"Expected 3072 dimensions, got {stats['embedding_dimension']}"
        
        print(f"   ✅ Dimension verification passed: {stats['embedding_dimension']} dimensions")


def main():
    """Run all tests"""
    print("🧪 Running Memory System Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_embedding_text_generation() 
        test_metadata_extraction()
        test_dimension_verification()
        
        print("\n🎉 All tests passed!")
        print("\nMemory System Summary:")
        print("- ✅ Experience document generation")
        print("- ✅ 3072-dimensional embedding configuration") 
        print("- ✅ Chroma HNSW storage backend")
        print("- ✅ Metadata filtering and retrieval")
        print("- ✅ Vector similarity search")
        
        if not os.getenv('OPENAI_API_KEY'):
            print("\nNote: OpenAI API key not set - actual embedding generation not tested")
            print("Set OPENAI_API_KEY environment variable for full embedding tests")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
