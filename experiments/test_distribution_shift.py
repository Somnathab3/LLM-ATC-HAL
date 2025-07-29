# experiments/test_distribution_shift.py
"""
Test script for distribution shift runner
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.distribution_shift_runner import run_distribution_shift_experiment
from analysis.metrics import aggregate_thesis_metrics
import pandas as pd

def test_distribution_shift_runner():
    """Test the distribution shift runner with a small sample"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Distribution Shift Runner...")
    
    try:
        # Run small test experiment
        results_file = run_distribution_shift_experiment(
            n_sims_per_tier=5  # Very small test
        )
        
        print(f"✅ Experiment completed successfully!")
        print(f"Results file: {results_file}")
        
        # Load and analyze results
        df = pd.read_parquet(results_file)
        print(f"✅ Loaded {len(df)} results")
        
        # Test aggregation functions
        aggregated_metrics = aggregate_thesis_metrics(df)
        print(f"✅ Aggregation completed")
        
        # Display summary
        print("\n" + "="*50)
        print("DISTRIBUTION SHIFT TEST RESULTS")
        print("="*50)
        
        overview = aggregated_metrics.get('experiment_overview', {})
        print(f"Total experiments: {overview.get('total_experiments', 0)}")
        print(f"Tiers tested: {overview.get('distribution_shift_tiers', [])}")
        
        hallucination = aggregated_metrics.get('hallucination_analysis', {})
        print(f"Overall hallucination detection rate: {hallucination.get('overall_detection_rate', 0):.3f}")
        
        safety = aggregated_metrics.get('safety_performance', {})
        print(f"Overall safety score: {safety.get('overall_safety_score', 0):.3f}")
        print(f"ICAO compliance rate: {safety.get('icao_compliance_rate', 0):.3f}")
        
        print("\nPer-tier performance:")
        for tier, rate in hallucination.get('detection_rate_by_tier', {}).items():
            safety_score = safety.get('safety_score_by_tier', {}).get(tier, 0)
            print(f"  {tier}: Detection={rate:.3f}, Safety={safety_score:.3f}")
        
        print("="*50)
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distribution_shift_runner()
    sys.exit(0 if success else 1)
