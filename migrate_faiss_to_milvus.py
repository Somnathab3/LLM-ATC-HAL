#!/usr/bin/env python3
"""
FAISS to Milvus Migration Script
Migrates existing FAISS-based experience replay data to Milvus GPU vector database.

Usage:
    python migrate_faiss_to_milvus.py [--storage-dir memory/replay_data] [--milvus-host localhost] [--milvus-port 19530]
"""

import os
import sys
import json
import logging
import argparse
import pickle
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.replay_store import VectorReplayStore, ConflictExperience

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSToMilvusMigrator:
    """Migrates FAISS-based experience data to Milvus"""
    
    def __init__(self, storage_dir: str, milvus_host: str = "localhost", milvus_port: int = 19530):
        self.storage_dir = storage_dir
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.legacy_files = {
            'experiences': os.path.join(storage_dir, "experiences.json"),
            'faiss_index': os.path.join(storage_dir, "faiss_index.idx"),
            'experience_ids': os.path.join(storage_dir, "experience_ids.pkl")
        }
        
    def check_legacy_data(self) -> bool:
        """Check if legacy FAISS data exists"""
        
        logger.info("Checking for legacy FAISS data...")
        
        if not os.path.exists(self.storage_dir):
            logger.warning(f"Storage directory {self.storage_dir} does not exist")
            return False
            
        has_experiences = os.path.exists(self.legacy_files['experiences'])
        has_index = os.path.exists(self.legacy_files['faiss_index'])
        has_ids = os.path.exists(self.legacy_files['experience_ids'])
        
        logger.info(f"Found experiences.json: {has_experiences}")
        logger.info(f"Found faiss_index.idx: {has_index}")
        logger.info(f"Found experience_ids.pkl: {has_ids}")
        
        return has_experiences
    
    def backup_legacy_data(self) -> str:
        """Create backup of legacy FAISS data"""
        
        import shutil
        from datetime import datetime
        
        backup_dir = os.path.join(self.storage_dir, f"faiss_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)
        
        logger.info(f"Creating backup in {backup_dir}")
        
        for file_type, file_path in self.legacy_files.items():
            if os.path.exists(file_path):
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up {file_type}: {file_path} -> {backup_path}")
        
        return backup_dir
    
    def load_legacy_experiences(self) -> List[ConflictExperience]:
        """Load experiences from legacy JSON format"""
        
        logger.info("Loading legacy experiences...")
        
        if not os.path.exists(self.legacy_files['experiences']):
            logger.warning("No legacy experiences.json found")
            return []
        
        try:
            with open(self.legacy_files['experiences'], 'r') as f:
                experiences_data = json.load(f)
            
            experiences = []
            for exp_data in experiences_data:
                # Convert back to ConflictExperience object
                experience = ConflictExperience(**exp_data)
                experiences.append(experience)
            
            logger.info(f"Loaded {len(experiences)} legacy experiences")
            return experiences
            
        except Exception as e:
            logger.error(f"Failed to load legacy experiences: {e}")
            return []
    
    def test_milvus_connection(self) -> bool:
        """Test connection to Milvus server"""
        
        logger.info(f"Testing Milvus connection to {self.milvus_host}:{self.milvus_port}")
        
        try:
            from pymilvus import connections, utility
            
            # Test connection
            connections.connect(
                alias="migration_test",
                host=self.milvus_host,
                port=self.milvus_port
            )
            
            # Get server info
            version = utility.get_server_version()
            logger.info(f"Connected to Milvus version: {version}")
            
            # Disconnect test connection
            connections.disconnect("migration_test")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            logger.error("Please ensure Milvus is running and accessible")
            return False
    
    def migrate_to_milvus(self, experiences: List[ConflictExperience]) -> bool:
        """Migrate experiences to Milvus"""
        
        if not experiences:
            logger.warning("No experiences to migrate")
            return True
        
        try:
            logger.info(f"Migrating {len(experiences)} experiences to Milvus...")
            
            # Create new Milvus-based vector store
            vector_store = VectorReplayStore(
                storage_dir=self.storage_dir,
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port
            )
            
            # Clear existing experiences to avoid duplicates
            vector_store.experiences.clear()
            
            # Batch insert experiences
            vector_store._batch_insert_to_milvus(experiences)
            
            # Update local experience cache
            for experience in experiences:
                vector_store.experiences[experience.experience_id] = experience
            
            # Save updated experiences
            vector_store._save_experiences()
            
            # Verify migration
            milvus_count = vector_store.collection.num_entities
            logger.info(f"Migration complete: {milvus_count} experiences in Milvus")
            
            return milvus_count == len(experiences)
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def cleanup_legacy_files(self, backup_dir: str):
        """Remove legacy FAISS files after successful migration"""
        
        logger.info("Cleaning up legacy FAISS files...")
        
        for file_type, file_path in self.legacy_files.items():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed legacy {file_type}: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Legacy files backed up to: {backup_dir}")
    
    def verify_migration(self) -> bool:
        """Verify migration was successful"""
        
        logger.info("Verifying migration...")
        
        try:
            # Test new Milvus store
            vector_store = VectorReplayStore(
                storage_dir=self.storage_dir,
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port
            )
            
            # Get statistics
            stats = vector_store.get_statistics()
            
            logger.info("Migration verification:")
            logger.info(f"  Total experiences: {stats['total_experiences']}")
            logger.info(f"  Milvus collection size: {stats['milvus_collection_size']}")
            logger.info(f"  Milvus host: {stats['milvus_host']}")
            
            # Test similarity search (if we have experiences)
            if stats['total_experiences'] > 0:
                logger.info("Testing similarity search...")
                
                # Get a sample experience
                sample_id = next(iter(vector_store.experiences.keys()))
                sample_experience = vector_store.experiences[sample_id]
                
                # Perform similarity search
                similar = vector_store.find_similar_experiences(sample_experience, top_k=3)
                logger.info(f"  Found {len(similar)} similar experiences")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False

def main():
    """Main migration script"""
    
    parser = argparse.ArgumentParser(description='Migrate FAISS experience data to Milvus')
    parser.add_argument('--storage-dir', default='memory/replay_data',
                       help='Storage directory containing FAISS data (default: memory/replay_data)')
    parser.add_argument('--milvus-host', default='localhost',
                       help='Milvus server host (default: localhost)')
    parser.add_argument('--milvus-port', type=int, default=19530,
                       help='Milvus server port (default: 19530)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Check migration requirements without performing migration')
    parser.add_argument('--force', action='store_true',
                       help='Force migration even if no legacy data found')
    
    args = parser.parse_args()
    
    logger.info("=== FAISS to Milvus Migration Tool ===")
    logger.info(f"Storage directory: {args.storage_dir}")
    logger.info(f"Milvus server: {args.milvus_host}:{args.milvus_port}")
    
    # Initialize migrator
    migrator = FAISSToMilvusMigrator(
        storage_dir=args.storage_dir,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port
    )
    
    # Check for legacy data
    has_legacy_data = migrator.check_legacy_data()
    
    if not has_legacy_data and not args.force:
        logger.info("No legacy FAISS data found. Migration not needed.")
        logger.info("Use --force to create empty Milvus collection anyway.")
        return 0
    
    # Test Milvus connection
    if not migrator.test_milvus_connection():
        logger.error("Cannot connect to Milvus. Please ensure it's running:")
        logger.error("  docker-compose -f docker-compose-milvus.yml up -d")
        return 1
    
    if args.dry_run:
        logger.info("Dry run complete - migration requirements satisfied")
        return 0
    
    # Load legacy experiences
    experiences = migrator.load_legacy_experiences()
    
    if experiences or args.force:
        # Create backup
        backup_dir = migrator.backup_legacy_data()
        
        # Perform migration
        success = migrator.migrate_to_milvus(experiences)
        
        if success:
            logger.info("✅ Migration successful!")
            
            # Verify migration
            if migrator.verify_migration():
                logger.info("✅ Migration verification passed!")
                
                # Cleanup legacy files
                migrator.cleanup_legacy_files(backup_dir)
                
                logger.info("🎉 FAISS to Milvus migration complete!")
                logger.info("Your experience replay system now uses GPU-accelerated Milvus!")
                
                return 0
            else:
                logger.error("❌ Migration verification failed")
                return 1
        else:
            logger.error("❌ Migration failed")
            return 1
    else:
        logger.info("No experiences to migrate")
        return 0

if __name__ == "__main__":
    exit(main())
