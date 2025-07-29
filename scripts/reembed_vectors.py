# scripts/reembed_vectors.py
"""
One-time script to re-embed existing vectors from 384-dim to 3072-dim
using text-embedding-3-large model and migrate to new Chroma storage.
"""

import logging
import os
import json
import time
import pickle
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.experience_document_generator import ExperienceDocumentGenerator
from memory.replay_store import VectorReplayStore


class VectorReembedder:
    """Handles migration from old embedding system to new 3072-dim system"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize new components
        self.doc_generator = ExperienceDocumentGenerator(openai_api_key)
        self.new_store = VectorReplayStore(openai_api_key=openai_api_key)
        
        # Migration statistics
        self.migration_stats = {
            'total_found': 0,
            'successfully_migrated': 0,
            'errors': 0,
            'start_time': time.time()
        }
    
    def find_existing_vector_data(self) -> List[Dict[str, Any]]:
        """Find existing vector data from various sources"""
        existing_data = []
        
        # Look for JSON backup files
        self.logger.info("Searching for JSON backup files...")
        json_data = self._find_json_backup_data()
        if json_data:
            existing_data.extend(json_data)
            self.logger.info(f"Found {len(json_data)} JSON experiences")
        
        # Look for FAISS data (if any)
        self.logger.info("Searching for existing FAISS data...")
        faiss_data = self._find_faiss_data()
        if faiss_data:
            existing_data.extend(faiss_data)
            self.logger.info(f"Found {len(faiss_data)} FAISS experiences")
        
        self.migration_stats['total_found'] = len(existing_data)
        self.logger.info(f"Total experiences found: {len(existing_data)}")
        
        return existing_data
    
    def _find_faiss_data(self) -> List[Dict[str, Any]]:
        """Try to extract data from existing FAISS files"""
        experiences = []
        
        # Look for common FAISS file locations
        faiss_paths = [
            "memory/replay_data",
            "memory",
            "data"
        ]
        
        for base_path in faiss_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith('.pkl') or file.endswith('.json') or 'faiss' in file.lower():
                            file_path = os.path.join(root, file)
                            try:
                                file_data = self._load_file_data(file_path)
                                if file_data:
                                    experiences.extend(file_data)
                                    self.logger.info(f"Loaded {len(file_data)} experiences from {file_path}")
                            except Exception as e:
                                self.logger.warning(f"Could not load {file_path}: {e}")
        
        return experiences
    
    def _find_json_backup_data(self) -> List[Dict[str, Any]]:
        """Look for JSON backup files with experience data"""
        experiences = []
        
        # Look for JSON files in common locations
        json_paths = [
            "memory/replay_data",
            "data/simulated",
            "test_results",
            "output"
        ]
        
        for base_path in json_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith('.json') and ('experience' in file.lower() or 'conflict' in file.lower()):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                
                                if isinstance(data, list):
                                    experiences.extend(data)
                                elif isinstance(data, dict):
                                    experiences.append(data)
                                
                                self.logger.info(f"Loaded data from {file_path}")
                                
                            except Exception as e:
                                self.logger.warning(f"Could not load JSON {file_path}: {e}")
        
        return experiences
    
    def _load_file_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from various file formats"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
            
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
        
        except Exception as e:
            self.logger.warning(f"Error loading {file_path}: {e}")
        
        return []
    
    def migrate_experiences(self, experiences: List[Dict[str, Any]]) -> bool:
        """Migrate experiences to new 3072-dim embedding system"""
        self.logger.info(f"Starting migration of {len(experiences)} experiences...")
        
        success_count = 0
        error_count = 0
        
        for i, exp_data in enumerate(experiences):
            try:
                # Validate required fields
                if not self._validate_experience_data(exp_data):
                    self.logger.warning(f"Skipping invalid experience {i}")
                    error_count += 1
                    continue
                
                # Enhance experience for embedding
                enhanced_exp = self._enhance_experience_for_embedding(exp_data)
                
                # Store with new 3072-dim embedding
                success = self.new_store.store_experience(enhanced_exp)
                
                if success:
                    success_count += 1
                    if success_count % 10 == 0:
                        self.logger.info(f"Migrated {success_count}/{len(experiences)} experiences")
                else:
                    error_count += 1
                    self.logger.warning(f"Failed to store experience {exp_data.get('experience_id', i)}")
                
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error migrating experience {i}: {e}")
        
        self.migration_stats['successfully_migrated'] = success_count
        self.migration_stats['errors'] = error_count
        
        self.logger.info(f"Migration completed: {success_count} successful, {error_count} errors")
        return error_count == 0
    
    def _validate_experience_data(self, exp_data: Dict[str, Any]) -> bool:
        """Validate that experience data has minimum required fields"""
        required_fields = ['conflict_type']
        
        for field in required_fields:
            if field not in exp_data:
                # Try to infer conflict type
                if 'scenario_type' in exp_data:
                    exp_data['conflict_type'] = exp_data['scenario_type']
                else:
                    exp_data['conflict_type'] = 'convergent'  # Default
        
        # Set defaults for missing fields
        if 'num_aircraft' not in exp_data:
            exp_data['num_aircraft'] = 2
        
        if 'experience_id' not in exp_data:
            exp_data['experience_id'] = f"migrated_{int(time.time()*1000)}_{hash(str(exp_data)) % 10000}"
        
        return True
    
    def _enhance_experience_for_embedding(self, exp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance experience data for better embedding generation"""
        enhanced = exp_data.copy()
        
        # Add descriptive text if missing
        if 'description' not in enhanced and 'scenario' not in enhanced:
            enhanced['description'] = f"Conflict scenario with {enhanced.get('num_aircraft', 2)} aircraft of type {enhanced.get('conflict_type', 'unknown')}"
        
        # Ensure outcome structure
        if 'outcome' not in enhanced:
            enhanced['outcome'] = {
                'safety_margin': enhanced.get('safety_margin', 0.5),
                'icao_compliant': enhanced.get('icao_compliant', True),
                'hallucination_detected': enhanced.get('hallucination_detected', False)
            }
        
        # Add migration metadata
        if 'metadata' not in enhanced:
            enhanced['metadata'] = {}
        
        enhanced['metadata'].update({
            'migration_source': 'reembedding_script',
            'migration_timestamp': time.time(),
            'new_embedding_dimension': 3072
        })
        
        return enhanced
    
    def generate_migration_report(self) -> str:
        """Generate a comprehensive migration report"""
        end_time = time.time()
        duration = end_time - self.migration_stats['start_time']
        
        report = f"""
Vector Re-embedding Migration Report
===================================

Migration Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.2f} seconds

Results:
- Total experiences found: {self.migration_stats['total_found']}
- Successfully migrated: {self.migration_stats['successfully_migrated']}
- Errors encountered: {self.migration_stats['errors']}
- Success rate: {(self.migration_stats['successfully_migrated'] / max(1, self.migration_stats['total_found'])) * 100:.1f}%

New System Configuration:
- Embedding model: text-embedding-3-large
- Embedding dimension: 3072
- Vector database: Chroma HNSW
- Similarity metric: Cosine

Post-migration Statistics:
"""
        
        # Get new store statistics
        try:
            stats = self.new_store.get_statistics()
            for key, value in stats.items():
                report += f"- {key}: {value}\n"
        except Exception as e:
            report += f"- Error getting statistics: {e}\n"
        
        return report
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        self.logger.info("Starting vector re-embedding migration...")
        
        try:
            # Find existing data
            existing_experiences = self.find_existing_vector_data()
            
            if not existing_experiences:
                self.logger.info("No existing vector data found. Migration not needed.")
                return True
            
            # Migrate to new system
            success = self.migrate_experiences(existing_experiences)
            
            # Generate report
            report = self.generate_migration_report()
            
            # Save report
            report_path = f"migration_report_{int(time.time())}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Migration report saved to: {report_path}")
            print(report)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False


def main():
    """Main migration function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set. Migration may fail.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Create and run migrator
    migrator = VectorReembedder()
    success = migrator.run_migration()
    
    if success:
        logger.info("✅ Migration completed successfully!")
    else:
        logger.error("❌ Migration failed or had errors.")
    
    return success


if __name__ == "__main__":
    main()
