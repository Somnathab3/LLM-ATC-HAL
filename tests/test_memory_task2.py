# tests/test_memory.py
"""
Test suite for the Experience Library memory system
Tests the E5-large-v2 embedding functionality with local Chroma storage
"""

import unittest
import tempfile
import shutil
import os
from typing import Dict, List, Any
import numpy as np

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.experience_document_generator import ExperienceDocumentGenerator
from memory.replay_store import VectorReplayStore


class TestMemorySystem(unittest.TestCase):
    """Test the memory system with in-memory Chroma collections"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ExperienceDocumentGenerator(persist_directory=self.temp_dir)
        self.replay_store = VectorReplayStore(storage_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_experience_document(self):
        """Test experience document generation"""
        # Generate a test experience
        exp_doc = self.generator.generate_experience(
            conflict_desc="Two aircraft on converging paths at FL350",
            commands_do=["Turn left 15 degrees", "Maintain current altitude"],
            commands_dont=["Descend", "Turn right"],
            reasoning="Left turn provides best separation with minimal deviation",
            conflict_type="convergent",
            num_ac=2,
            safety_margin=3.2,
            icao_compliant=True
        )
        
        # Verify document structure
        self.assertIsInstance(exp_doc, dict)
        self.assertIn('experience_id', exp_doc)
        self.assertIn('conflict_type', exp_doc)
        self.assertIn('num_aircraft', exp_doc)
        self.assertEqual(exp_doc['conflict_type'], 'convergent')
        self.assertEqual(exp_doc['num_aircraft'], 2)
        self.assertEqual(exp_doc['safety_margin'], 3.2)
        self.assertTrue(exp_doc['icao_compliant'])
    
    def test_embed_and_store(self):
        """Test embedding and storage functionality"""
        # Generate two test experiences
        exp1 = self.generator.generate_experience(
            conflict_desc="Aircraft converging at same altitude",
            commands_do=["Turn left", "Climb"],
            commands_dont=["Continue straight"],
            reasoning="Avoid collision",
            conflict_type="convergent",
            num_ac=2
        )
        
        exp2 = self.generator.generate_experience(
            conflict_desc="Parallel aircraft with speed difference",
            commands_do=["Reduce speed", "Maintain heading"],
            commands_dont=["Accelerate"],
            reasoning="Manage overtaking situation",
            conflict_type="parallel",
            num_ac=2
        )
        
        # Store both experiences
        self.generator.embed_and_store(exp1)
        self.generator.embed_and_store(exp2)
        
        # Verify storage
        stats = self.generator.get_collection_stats()
        self.assertEqual(stats['total_experiences'], 2)
        self.assertEqual(stats['embedding_model'], 'intfloat/e5-large-v2')
        self.assertEqual(stats['embedding_dim'], 1024)
    
    def test_retrieve_experience_with_metadata_filter(self):
        """Test experience retrieval with metadata filtering"""
        # Generate and store test experiences
        convergent_exp = self.generator.generate_experience(
            conflict_desc="Head-on convergence at FL350",
            commands_do=["Turn left 20 degrees"],
            commands_dont=["Continue on heading"],
            reasoning="Standard avoidance maneuver",
            conflict_type="convergent",
            num_ac=2
        )
        
        parallel_exp = self.generator.generate_experience(
            conflict_desc="Parallel tracks with overtaking",
            commands_do=["Reduce speed 10 knots"],
            commands_dont=["Maintain speed"],
            reasoning="Prevent overtaking conflict",
            conflict_type="parallel",
            num_ac=2
        )
        
        # Store experiences
        self.generator.embed_and_store(convergent_exp)
        self.generator.embed_and_store(parallel_exp)
        
        # Test retrieval with metadata filter
        results = self.replay_store.retrieve_experience(
            conflict_desc="Aircraft approaching each other",
            conflict_type="convergent",
            num_ac=2,
            k=5
        )
        
        # Verify results
        self.assertEqual(len(results), 1)  # Only convergent should match
        self.assertEqual(results[0]['metadata']['conflict_type'], 'convergent')
        self.assertEqual(results[0]['metadata']['num_ac'], 2)
        self.assertGreater(results[0]['similarity_score'], 0.0)
    
    def test_vector_similarity_search(self):
        """Test vector similarity functionality"""
        # Generate similar experiences
        exp1 = self.generator.generate_experience(
            conflict_desc="Two aircraft converging at intersection",
            commands_do=["Turn left"],
            commands_dont=["Continue straight"],
            reasoning="Avoid collision",
            conflict_type="convergent",
            num_ac=2
        )
        
        exp2 = self.generator.generate_experience(
            conflict_desc="Aircraft approaching intersection point",
            commands_do=["Turn right"],
            commands_dont=["Maintain heading"],
            reasoning="Provide separation",
            conflict_type="convergent",
            num_ac=2
        )
        
        # Store experiences
        self.generator.embed_and_store(exp1)
        self.generator.embed_and_store(exp2)
        
        # Search for similar experience
        results = self.replay_store.retrieve_experience(
            conflict_desc="Two planes meeting at waypoint",  # Similar but different wording
            conflict_type="convergent",
            num_ac=2,
            k=2
        )
        
        # Verify similarity search works
        self.assertEqual(len(results), 2)
        # Results should be ordered by similarity (highest first)
        self.assertGreaterEqual(results[0]['similarity_score'], results[1]['similarity_score'])
    
    def test_embedding_dimension(self):
        """Test that embeddings have the correct dimension (1024)"""
        # Generate an experience
        exp_doc = self.generator.generate_experience(
            conflict_desc="Test conflict scenario",
            commands_do=["Test command"],
            commands_dont=["Test avoidance"],
            reasoning="Test reasoning",
            conflict_type="convergent",
            num_ac=2
        )
        
        # Test that the embedding model produces 1024-dim vectors
        conflict_desc = exp_doc['conflict_geometry_text']
        embedding = self.generator.embedding_model.encode(conflict_desc, normalize_embeddings=True)
        
        # Verify embedding shape
        self.assertEqual(embedding.shape, (1024,))
        self.assertIsInstance(embedding, np.ndarray)
        
        # Verify normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_get_all_experiences(self):
        """Test retrieving all experiences with filters"""
        # Generate experiences with different types
        for i in range(3):
            exp = self.generator.generate_experience(
                conflict_desc=f"Convergent conflict {i}",
                commands_do=[f"Action {i}"],
                commands_dont=[f"Avoid {i}"],
                reasoning=f"Reasoning {i}",
                conflict_type="convergent",
                num_ac=2
            )
            self.generator.embed_and_store(exp)
        
        # Generate parallel conflict
        parallel_exp = self.generator.generate_experience(
            conflict_desc="Parallel conflict",
            commands_do=["Parallel action"],
            commands_dont=["Parallel avoid"],
            reasoning="Parallel reasoning",
            conflict_type="parallel",
            num_ac=3
        )
        self.generator.embed_and_store(parallel_exp)
        
        # Test get all without filter
        all_experiences = self.replay_store.get_all_experiences()
        self.assertEqual(len(all_experiences), 4)
        
        # Test get all with conflict type filter
        convergent_experiences = self.replay_store.get_all_experiences(conflict_type="convergent")
        self.assertEqual(len(convergent_experiences), 3)
        
        # Test get all with aircraft count filter
        two_ac_experiences = self.replay_store.get_all_experiences(num_ac=2)
        self.assertEqual(len(two_ac_experiences), 3)
        
        # Test get all with both filters
        specific_experiences = self.replay_store.get_all_experiences(
            conflict_type="parallel", 
            num_ac=3
        )
        self.assertEqual(len(specific_experiences), 1)
    
    def test_stats_and_management(self):
        """Test statistics and management functions"""
        # Initially empty
        stats = self.replay_store.get_stats()
        self.assertEqual(stats['total_experiences'], 0)
        
        # Add some experiences
        for conflict_type in ['convergent', 'parallel', 'crossing']:
            exp = self.generator.generate_experience(
                conflict_desc=f"{conflict_type} scenario",
                commands_do=["Action"],
                commands_dont=["Avoid"],
                reasoning="Reasoning",
                conflict_type=conflict_type,
                num_ac=2
            )
            self.generator.embed_and_store(exp)
        
        # Check updated stats
        stats = self.replay_store.get_stats()
        self.assertEqual(stats['total_experiences'], 3)
        self.assertEqual(len(stats['by_conflict_type']), 3)
        self.assertEqual(stats['embedding_model'], 'intfloat/e5-large-v2')
        self.assertEqual(stats['embedding_dim'], 1024)


class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for the complete memory system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ExperienceDocumentGenerator(persist_directory=self.temp_dir)
        self.replay_store = VectorReplayStore(storage_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from generation to retrieval"""
        # Step 1: Generate multiple related experiences
        scenarios = [
            {
                "conflict_desc": "Aircraft converging at same waypoint from different directions",
                "commands_do": ["Turn left 15 degrees", "Maintain altitude"],
                "commands_dont": ["Continue straight", "Descend"],
                "reasoning": "Left turn provides sufficient separation margin",
                "conflict_type": "convergent",
                "num_ac": 2
            },
            {
                "conflict_desc": "Head-on approach at same flight level",
                "commands_do": ["Turn right 20 degrees", "Climb 1000 feet"],
                "commands_dont": ["Maintain heading", "Descend"],
                "reasoning": "Right turn and climb for maximum separation",
                "conflict_type": "convergent", 
                "num_ac": 2
            },
            {
                "conflict_desc": "Parallel tracks with speed differential",
                "commands_do": ["Reduce speed 15 knots"],
                "commands_dont": ["Increase speed", "Change heading"],
                "reasoning": "Speed reduction prevents overtaking conflict",
                "conflict_type": "parallel",
                "num_ac": 2
            }
        ]
        
        # Step 2: Generate and store all experiences
        experience_ids = []
        for scenario in scenarios:
            exp_doc = self.generator.generate_experience(**scenario)
            self.generator.embed_and_store(exp_doc)
            experience_ids.append(exp_doc['experience_id'])
        
        # Step 3: Verify storage
        stats = self.replay_store.get_stats()
        self.assertEqual(stats['total_experiences'], 3)
        self.assertEqual(stats['by_conflict_type']['convergent'], 2)
        self.assertEqual(stats['by_conflict_type']['parallel'], 1)
        
        # Step 4: Test retrieval of similar experiences
        similar_results = self.replay_store.retrieve_experience(
            conflict_desc="Two aircraft approaching same point",
            conflict_type="convergent",
            num_ac=2,
            k=2
        )
        
        # Should get both convergent experiences
        self.assertEqual(len(similar_results), 2)
        for result in similar_results:
            self.assertEqual(result['metadata']['conflict_type'], 'convergent')
            self.assertIn(result['experience_id'], experience_ids)
        
        # Step 5: Test that filtering works correctly
        parallel_results = self.replay_store.retrieve_experience(
            conflict_desc="Aircraft with different speeds",
            conflict_type="parallel",
            num_ac=2,
            k=5
        )
        
        # Should get only the parallel experience
        self.assertEqual(len(parallel_results), 1)
        self.assertEqual(parallel_results[0]['metadata']['conflict_type'], 'parallel')


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
