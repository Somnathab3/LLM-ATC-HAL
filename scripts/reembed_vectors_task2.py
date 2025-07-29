#!/usr/bin/env python3
"""
Re-embedding Script for LLM-ATC-HAL Experience Library
Migrates existing experience vectors to use intfloat/e5-large-v2 (1024-dim) embeddings
instead of legacy 384-dim or 3072-dim embeddings, storing in local Chroma.

Usage:
    python scripts/reembed_vectors.py [--source-dir memory/chroma_db] [--target-dir memory/chroma_experience_library]
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_atc.memory.experience_document_generator import ExperienceDocumentGenerator
from llm_atc.memory.replay_store import VectorReplayStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperienceReembedder:
    """Re-embeds existing experience data with E5-large-v2 model"""
    
    def __init__(self, 
                 source_dir: str = "memory/chroma_db",
                 target_dir: str = "memory/chroma_experience_library"):
        """
        Initialize the re-embedder
        
        Args:
            source_dir: Directory containing old embeddings
            target_dir: Directory for new E5-large-v2 embeddings
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        
        # Initialize new system components
        self.generator = ExperienceDocumentGenerator(persist_directory=target_dir)
        self.replay_store = VectorReplayStore(storage_dir=target_dir)
        
        # Initialize source Chroma client
        if os.path.exists(source_dir):
            self.source_client = chromadb.PersistentClient(
                path=source_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            self.source_client = None
            logger.warning(f"Source directory {source_dir} does not exist")
    
    def find_source_collections(self) -> List[str]:
        """Find all collections in the source directory"""
        if not self.source_client:
            return []
        
        try:
            collections = self.source_client.list_collections()
            collection_names = [col.name for col in collections]
            logger.info(f"Found source collections: {collection_names}")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list source collections: {e}")
            return []
    
    def migrate_collection(self, collection_name: str) -> int:
        """
        Migrate a single collection to the new E5-large-v2 format
        
        Args:
            collection_name: Name of the source collection
            
        Returns:
            Number of experiences migrated
        """
        try:
            # Get source collection
            source_collection = self.source_client.get_collection(collection_name)
            
            # Get all documents from source
            results = source_collection.get(include=['documents', 'metadatas'])
            
            if not results['ids']:
                logger.info(f"No documents found in collection {collection_name}")
                return 0
            
            migrated_count = 0
            
            for i, doc_id in enumerate(results['ids']):
                try:
                    # Get document and metadata
                    document = results['documents'][i] if results['documents'] else ""
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    
                    # Extract key information for the new format
                    conflict_desc = document or metadata.get('conflict_desc', 'Unknown conflict')
                    conflict_type = metadata.get('conflict_type', 'convergent')
                    num_ac = metadata.get('num_ac', metadata.get('num_aircraft', 2))
                    
                    # Create commands and reasoning from available data
                    commands_do = metadata.get('commands_do', ['Maintain separation'])
                    commands_dont = metadata.get('commands_dont', ['Do not ignore conflicts'])
                    reasoning = metadata.get('reasoning', metadata.get('lessons_learned', 'Maintain safe separation'))
                    
                    # Generate new experience document
                    exp_doc = self.generator.generate_experience(
                        conflict_desc=conflict_desc,
                        commands_do=commands_do if isinstance(commands_do, list) else [str(commands_do)],
                        commands_dont=commands_dont if isinstance(commands_dont, list) else [str(commands_dont)],
                        reasoning=reasoning,
                        conflict_type=conflict_type,
                        num_ac=int(num_ac),
                        experience_id=doc_id,
                        safety_margin=metadata.get('safety_margin', 0.0),
                        icao_compliant=metadata.get('icao_compliant', True),
                        hallucination_detected=metadata.get('hallucination_detected', False),
                        hallucination_types=metadata.get('hallucination_types', []),
                        additional_metadata=metadata
                    )
                    
                    # Embed and store in new format
                    self.generator.embed_and_store(exp_doc)
                    migrated_count += 1
                    
                    if migrated_count % 10 == 0:
                        logger.info(f"Migrated {migrated_count} experiences...")
                    
                except Exception as e:
                    logger.error(f"Failed to migrate experience {doc_id}: {e}")
                    continue
            
            logger.info(f"Successfully migrated {migrated_count} experiences from {collection_name}")
            return migrated_count
            
        except Exception as e:
            logger.error(f"Failed to migrate collection {collection_name}: {e}")
            return 0
    
    def migrate_legacy_json_data(self, json_file: str = "memory/replay_data/experiences.json") -> int:
        """
        Migrate legacy JSON experience data
        
        Args:
            json_file: Path to legacy JSON file
            
        Returns:
            Number of experiences migrated
        """
        if not os.path.exists(json_file):
            logger.info(f"No legacy JSON file found at {json_file}")
            return 0
        
        try:
            with open(json_file, 'r') as f:
                legacy_data = json.load(f)
            
            migrated_count = 0
            
            for exp_id, exp_data in legacy_data.items():
                try:
                    # Extract information for new format
                    conflict_desc = exp_data.get('scenario_text', exp_data.get('conflict_desc', 'Unknown conflict'))
                    conflict_type = exp_data.get('conflict_type', 'convergent')
                    num_ac = exp_data.get('num_aircraft', 2)
                    
                    # Generate new experience document
                    exp_doc = self.generator.generate_experience(
                        conflict_desc=conflict_desc,
                        commands_do=['Maintain separation'],  # Default values
                        commands_dont=['Ignore conflicts'],
                        reasoning=exp_data.get('lessons_learned', 'Maintain safe separation'),
                        conflict_type=conflict_type,
                        num_ac=num_ac,
                        experience_id=exp_id,
                        additional_metadata=exp_data
                    )
                    
                    # Embed and store
                    self.generator.embed_and_store(exp_doc)
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate JSON experience {exp_id}: {e}")
                    continue
            
            logger.info(f"Successfully migrated {migrated_count} experiences from JSON file")
            return migrated_count
            
        except Exception as e:
            logger.error(f"Failed to migrate JSON data: {e}")
            return 0
    
    def run_migration(self) -> Dict[str, int]:
        """
        Run the complete migration process
        
        Returns:
            Dictionary with migration statistics
        """
        logger.info("Starting experience re-embedding migration...")
        
        results = {
            'collections_migrated': 0,
            'total_experiences': 0,
            'json_experiences': 0
        }
        
        # Migrate Chroma collections
        source_collections = self.find_source_collections()
        for collection_name in source_collections:
            count = self.migrate_collection(collection_name)
            if count > 0:
                results['collections_migrated'] += 1
                results['total_experiences'] += count
        
        # Migrate legacy JSON data
        json_count = self.migrate_legacy_json_data()
        results['json_experiences'] = json_count
        results['total_experiences'] += json_count
        
        # Print final statistics
        stats = self.replay_store.get_stats()
        logger.info(f"Migration complete! Final statistics: {stats}")
        
        return results


def main():
    """Main migration script"""
    parser = argparse.ArgumentParser(description='Re-embed experience vectors with E5-large-v2')
    parser.add_argument('--source-dir', default='memory/chroma_db',
                       help='Source directory with old embeddings')
    parser.add_argument('--target-dir', default='memory/chroma_experience_library',
                       help='Target directory for new embeddings')
    parser.add_argument('--clear-target', action='store_true',
                       help='Clear target collection before migration')
    
    args = parser.parse_args()
    
    # Create re-embedder
    embedder = ExperienceReembedder(
        source_dir=args.source_dir,
        target_dir=args.target_dir
    )
    
    # Clear target if requested
    if args.clear_target:
        logger.info("Clearing target collection...")
        embedder.replay_store.clear_all()
    
    # Run migration
    results = embedder.run_migration()
    
    # Print results
    print("\n" + "="*50)
    print("MIGRATION RESULTS")
    print("="*50)
    print(f"Collections migrated: {results['collections_migrated']}")
    print(f"Total experiences migrated: {results['total_experiences']}")
    print(f"JSON experiences migrated: {results['json_experiences']}")
    print("="*50)
    
    return results['total_experiences'] > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
