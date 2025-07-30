# experiments/test_distribution_shift.py
"""
Test script for distribution shift runner
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional

import pandas as pd
from experiments.distribution_shift_runner import run_distribution_shift_experiment

from analysis.metrics import aggregate_thesis_metrics


def test_distribution_shift_runner() -> Optional[bool]:
    """Test the distribution shift runner with a small sample"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


    try:
        # Run small test experiment
        results_file = run_distribution_shift_experiment(
            n_sims_per_tier=5,  # Very small test
        )


        # Load and analyze results
        df = pd.read_parquet(results_file)

        # Test aggregation functions
        aggregated_metrics = aggregate_thesis_metrics(df)

        # Display summary

        aggregated_metrics.get("experiment_overview", {})

        hallucination = aggregated_metrics.get("hallucination_analysis", {})

        safety = aggregated_metrics.get("safety_performance", {})

        for tier, _rate in hallucination.get("detection_rate_by_tier", {}).items():
            safety.get("safety_score_by_tier", {}).get(tier, 0)


        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_distribution_shift_runner()
    sys.exit(0 if success else 1)
