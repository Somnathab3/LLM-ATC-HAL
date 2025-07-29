#!/usr/bin/env python3
"""
Quick Test Runner for LLM-ATC-HAL
Runs a small test to verify the complete system works
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_hallucination_tester_v2 import ComprehensiveHallucinationTesterV2, TestConfiguration

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quick_test.log')
        ]
    )

async def run_quick_test():
    """Run a quick test with 5 scenarios"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Quick Test of LLM-ATC-HAL System")
    logger.info("=" * 50)
    
    # Create test configuration with all required parameters
    test_config = TestConfiguration(
        # Model Configuration
        models_to_test=["llama3.1:8b", "mistral:7b"],
        ensemble_weights={"primary": 0.4, "validator": 0.3, "technical": 0.3},
        
        # Scenario Configuration
        num_scenarios=5,
        complexity_distribution={'simple': 1.0},
        
        # Testing Parameters
        parallel_workers=2,
        timeout_per_test=15.0,
        
        # Performance Thresholds
        target_accuracy=0.85,
        target_response_time=2.0,
        target_safety_compliance=0.95,
        
        # GPU/Hardware Configuration
        use_gpu_acceleration=False,
        batch_size=5,
        
        # Output Configuration
        output_directory='test_results',
        generate_visualizations=False,
        detailed_logging=True,
        stream_results_to_disk=True
    )
    
    logger.info(f"Configuration: {test_config.num_scenarios} scenarios, {test_config.parallel_workers} workers")
    
    try:
        # Initialize the tester
        tester = ComprehensiveHallucinationTesterV2(test_config)
        
        # Run the testing campaign
        await tester.run_comprehensive_testing_campaign()
        
        logger.info("Quick test completed successfully!")
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_quick_test())
