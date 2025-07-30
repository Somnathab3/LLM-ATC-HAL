#!/usr/bin/env python3
"""
Simple validation test for Monte Carlo analysis functionality.
Tests basic operations without complex dependencies.
"""

import json
import tempfile
import os
import sys
from pathlib import Path

# Add the metrics module to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'llm_atc', 'metrics'))

def test_monte_carlo_analysis():
    """Test the Monte Carlo analysis with sample data."""
    print("Testing Monte Carlo Analysis...")
    
    try:
        import monte_carlo_analysis
        print("✓ Module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Create sample data
    sample_data = [
        {
            'scenario_type': 'horizontal',
            'success': True,
            'predicted_conflicts': [],
            'actual_conflicts': [],
            'horizontal_margin': 6.2,
            'vertical_margin': 1200,
            'efficiency_penalty': 2.1,
            'distribution_shift_level': 'in_distribution'
        },
        {
            'scenario_type': 'vertical',
            'success': False,
            'predicted_conflicts': [{'aircraft_1': 'AC001', 'aircraft_2': 'AC002'}],
            'actual_conflicts': [{'aircraft_1': 'AC001', 'aircraft_2': 'AC002'}],
            'horizontal_margin': 3.8,
            'vertical_margin': 800,
            'efficiency_penalty': 5.3,
            'distribution_shift_level': 'moderate_shift'
        },
        {
            'scenario_type': 'sector',
            'success': True,
            'predicted_conflicts': [{'aircraft_1': 'AC003', 'aircraft_2': 'AC004'}],
            'actual_conflicts': [],
            'horizontal_margin': 7.1,
            'vertical_margin': 1500,
            'efficiency_penalty': 1.8,
            'distribution_shift_level': 'in_distribution'
        }
    ]
    
    # Test analyzer
    try:
        analyzer = monte_carlo_analysis.MonteCarloResultsAnalyzer()
        print("✓ Analyzer initialized")
    except Exception as e:
        print(f"✗ Analyzer initialization failed: {e}")
        return False
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        
        # Test reading results
        df = analyzer.read_results_file(temp_file)
        print(f"✓ Read {len(df)} scenarios from file")
        
        # Test metrics computation
        fp_fn_rates = analyzer.compute_false_positive_negative_rates(df)
        print(f"✓ FP/FN rates: FP={fp_fn_rates['false_positive_rate']:.3f}, FN={fp_fn_rates['false_negative_rate']:.3f}")
        
        success_rates = analyzer.compute_success_rates_by_scenario(df)
        print(f"✓ Success rates computed for {len(success_rates)} scenario types")
        
        margins = analyzer.compute_average_separation_margins(df)
        print(f"✓ Avg margins: H={margins['avg_horizontal_margin']:.1f}nm, V={margins['avg_vertical_margin']:.0f}ft")
        
        penalties = analyzer.compute_efficiency_penalties(df)
        print(f"✓ Avg efficiency penalty: {penalties['avg_efficiency_penalty']:.1f}nm")
        
        # Test comprehensive aggregation
        aggregated = analyzer.aggregate_monte_carlo_metrics(df)
        print(f"✓ Aggregated metrics for {aggregated['summary']['total_scenarios']} scenarios")
        
        # Test visualizer (without actual plotting)
        visualizer = monte_carlo_analysis.MonteCarloVisualizer()
        print("✓ Visualizer initialized")
        
        # Test complete pipeline
        with tempfile.TemporaryDirectory() as temp_dir:
            results = monte_carlo_analysis.analyze_monte_carlo_results(temp_file, temp_dir)
            print(f"✓ Complete analysis pipeline successful")
            print(f"  - Metrics file: {results['metrics_file']}")
            print(f"  - Summary plots: {len(results['summary_plots'])}")
            print(f"  - Output dir: {results['output_directory']}")
        
        print("\n✅ All Monte Carlo analysis tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if 'temp_file' in locals():
            Path(temp_file).unlink(missing_ok=True)

if __name__ == "__main__":
    success = test_monte_carlo_analysis()
    sys.exit(0 if success else 1)