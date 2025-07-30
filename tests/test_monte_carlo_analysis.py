#!/usr/bin/env python3
"""
Simple unit tests for Monte Carlo analysis functions.
Tests the core functionality of monte_carlo_analysis.py.
"""

import unittest
import tempfile
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Direct import to avoid dependency chain issues
sys.path.insert(0, os.path.join(project_root, 'llm_atc', 'metrics'))

try:
    import monte_carlo_analysis
    MONTE_CARLO_AVAILABLE = True
except ImportError as e:
    MONTE_CARLO_AVAILABLE = False
    print(f"Monte Carlo analysis module not available: {e}")

# Mock dependencies that may not be available in test environment
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class TestMonteCarloAnalysis(unittest.TestCase):
    """Test cases for Monte Carlo analysis functions"""
    
    def setUp(self):
        """Setup test fixtures"""
        if not PANDAS_AVAILABLE:
            self.skipTest("pandas/numpy not available for testing")
        
        # Import the modules we're testing (direct import to avoid dependency issues)
        sys.path.insert(0, os.path.join(project_root, 'llm_atc', 'metrics'))
        import monte_carlo_analysis
        
        self.MonteCarloResultsAnalyzer = monte_carlo_analysis.MonteCarloResultsAnalyzer
        self.MonteCarloVisualizer = monte_carlo_analysis.MonteCarloVisualizer
        self.analyze_function = monte_carlo_analysis.analyze_monte_carlo_results
        
        
        self.analyzer = self.MonteCarloResultsAnalyzer()
        self.visualizer = self.MonteCarloVisualizer()
        
        # Create sample test data
        self.sample_results = [
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
    
    def test_analyzer_initialization(self):
        """Test that MonteCarloResultsAnalyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.logger)
    
    def test_read_json_results(self):
        """Test reading results from JSON file"""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_results, f)
            temp_file = f.name
        
        try:
            # Test reading the file
            df = self.analyzer.read_results_file(temp_file)
            self.assertEqual(len(df), 3)
            self.assertIn('scenario_type', df.columns)
            self.assertIn('success', df.columns)
            
        finally:
            os.unlink(temp_file)
    
    def test_read_nonexistent_file(self):
        """Test error handling for nonexistent files"""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.read_results_file("nonexistent_file.json")
    
    def test_compute_false_positive_negative_rates(self):
        """Test false positive/negative rate calculation"""
        df = pd.DataFrame(self.sample_results)
        rates = self.analyzer.compute_false_positive_negative_rates(df)
        
        # Check that we get expected structure
        self.assertIn('false_positive_rate', rates)
        self.assertIn('false_negative_rate', rates)
        self.assertIn('total_false_positives', rates)
        self.assertIn('total_false_negatives', rates)
        
        # Check that rates are between 0 and 1
        self.assertGreaterEqual(rates['false_positive_rate'], 0)
        self.assertLessEqual(rates['false_positive_rate'], 1)
        self.assertGreaterEqual(rates['false_negative_rate'], 0)
        self.assertLessEqual(rates['false_negative_rate'], 1)
        
        # Based on our sample data: 1 FP (scenario 3), 0 FN
        self.assertEqual(rates['total_false_positives'], 1)
        self.assertEqual(rates['total_false_negatives'], 0)
    
    def test_compute_success_rates_by_scenario(self):
        """Test success rate calculation by scenario type"""
        df = pd.DataFrame(self.sample_results)
        success_rates = self.analyzer.compute_success_rates_by_scenario(df)
        
        # Check structure
        self.assertIn('horizontal', success_rates)
        self.assertIn('vertical', success_rates)
        self.assertIn('sector', success_rates)
        
        # Check horizontal scenario (1 success out of 1)
        horizontal_stats = success_rates['horizontal']
        self.assertEqual(horizontal_stats['success_rate'], 1.0)
        self.assertEqual(horizontal_stats['total_scenarios'], 1)
        
        # Check vertical scenario (0 success out of 1)
        vertical_stats = success_rates['vertical']
        self.assertEqual(vertical_stats['success_rate'], 0.0)
        self.assertEqual(vertical_stats['total_scenarios'], 1)
    
    def test_compute_average_separation_margins(self):
        """Test separation margin calculation"""
        df = pd.DataFrame(self.sample_results)
        margins = self.analyzer.compute_average_separation_margins(df)
        
        # Check structure
        self.assertIn('avg_horizontal_margin', margins)
        self.assertIn('avg_vertical_margin', margins)
        self.assertIn('std_horizontal_margin', margins)
        self.assertIn('std_vertical_margin', margins)
        
        # Check that averages are reasonable
        expected_h_avg = (6.2 + 3.8 + 7.1) / 3
        expected_v_avg = (1200 + 800 + 1500) / 3
        
        self.assertAlmostEqual(margins['avg_horizontal_margin'], expected_h_avg, places=2)
        self.assertAlmostEqual(margins['avg_vertical_margin'], expected_v_avg, places=2)
    
    def test_compute_efficiency_penalties(self):
        """Test efficiency penalty calculation"""
        df = pd.DataFrame(self.sample_results)
        penalties = self.analyzer.compute_efficiency_penalties(df)
        
        # Check structure
        self.assertIn('avg_efficiency_penalty', penalties)
        self.assertIn('std_efficiency_penalty', penalties)
        self.assertIn('max_efficiency_penalty', penalties)
        
        # Check values are reasonable
        expected_avg = (2.1 + 5.3 + 1.8) / 3
        self.assertAlmostEqual(penalties['avg_efficiency_penalty'], expected_avg, places=2)
        self.assertEqual(penalties['max_efficiency_penalty'], 5.3)
    
    def test_aggregate_monte_carlo_metrics(self):
        """Test comprehensive metrics aggregation"""
        df = pd.DataFrame(self.sample_results)
        metrics = self.analyzer.aggregate_monte_carlo_metrics(df)
        
        # Check top-level structure
        self.assertIn('summary', metrics)
        self.assertIn('detection_performance', metrics)
        self.assertIn('success_rates_by_scenario', metrics)
        self.assertIn('separation_margins', metrics)
        self.assertIn('efficiency_metrics', metrics)
        self.assertIn('distribution_shift_analysis', metrics)
        
        # Check summary
        self.assertEqual(metrics['summary']['total_scenarios'], 3)
        self.assertIn('horizontal', metrics['summary']['scenario_types'])
        
        # Check that distribution shift analysis is present
        shift_analysis = metrics['distribution_shift_analysis']
        self.assertIn('in_distribution', shift_analysis)
        self.assertIn('moderate_shift', shift_analysis)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()
        
        # Should not crash and return reasonable defaults
        rates = self.analyzer.compute_false_positive_negative_rates(empty_df)
        self.assertEqual(rates['false_positive_rate'], 0.0)
        self.assertEqual(rates['false_negative_rate'], 0.0)
        
        success_rates = self.analyzer.compute_success_rates_by_scenario(empty_df)
        self.assertEqual(success_rates, {})
        
        margins = self.analyzer.compute_average_separation_margins(empty_df)
        self.assertEqual(margins['avg_horizontal_margin'], 0.0)
        
        penalties = self.analyzer.compute_efficiency_penalties(empty_df)
        self.assertEqual(penalties['avg_efficiency_penalty'], 0.0)
    
    def test_visualizer_initialization(self):
        """Test that MonteCarloVisualizer initializes correctly"""
        self.assertIsNotNone(self.visualizer)
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_full_analysis_pipeline(self):
        """Test the complete analysis pipeline"""
        # Create temporary file with sample data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_results, f)
            temp_file = f.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Run full analysis
                results = self.analyze_function(temp_file, temp_dir)
                
                # Check results structure
                self.assertIn('metrics', results)
                self.assertIn('metrics_file', results)
                self.assertIn('summary_plots', results)
                self.assertIn('distribution_shift_plots', results)
                self.assertIn('output_directory', results)
                
                # Check that metrics file was created
                metrics_file = Path(results['metrics_file'])
                self.assertTrue(metrics_file.exists())
                
                # Check that some plots were created (if matplotlib available)
                summary_plots = results['summary_plots']
                if summary_plots:  # Only check if plotting is available
                    self.assertGreater(len(summary_plots), 0)
                    # Check that plot files exist
                    for plot_path in summary_plots:
                        self.assertTrue(Path(plot_path).exists())
                
            finally:
                os.unlink(temp_file)
    
    def test_conflicts_to_set_conversion(self):
        """Test conversion of conflict lists to sets"""
        # Test with various conflict formats
        conflicts = [
            {'aircraft_1': 'AC001', 'aircraft_2': 'AC002'},
            {'aircraft1': 'AC003', 'aircraft2': 'AC004'},  # Alternative format
            {'aircraft_1': 'AC002', 'aircraft_2': 'AC001'}  # Same pair, different order
        ]
        
        conflict_set = self.analyzer._conflicts_to_set(conflicts)
        
        # Should have 2 unique pairs (AC001-AC002 is same as AC002-AC001)
        self.assertEqual(len(conflict_set), 2)
        self.assertIn(('AC001', 'AC002'), conflict_set)
        self.assertIn(('AC003', 'AC004'), conflict_set)
    
    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"some text")
            temp_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.analyzer.read_results_file(temp_file)
        finally:
            os.unlink(temp_file)


class TestMonteCarloAnalysisSimple(unittest.TestCase):
    """Simplified tests that don't require pandas/numpy"""
    
    def test_module_import(self):
        """Test that the module can be imported"""
        try:
            # Direct import to avoid dependency issues
            sys.path.insert(0, os.path.join(project_root, 'llm_atc', 'metrics'))
            import monte_carlo_analysis
            self.assertIsNotNone(monte_carlo_analysis)
        except ImportError as e:
            self.fail(f"Failed to import monte_carlo_analysis: {e}")
    
    def test_class_definitions_exist(self):
        """Test that required classes are defined"""
        try:
            # Direct import to avoid dependency issues
            sys.path.insert(0, os.path.join(project_root, 'llm_atc', 'metrics'))
            import monte_carlo_analysis
            
            # Test that classes exist
            self.assertTrue(hasattr(monte_carlo_analysis, 'MonteCarloResultsAnalyzer'))
            self.assertTrue(hasattr(monte_carlo_analysis, 'MonteCarloVisualizer'))
            
            # Test that classes can be instantiated (may fail if pandas not available)
            try:
                analyzer = monte_carlo_analysis.MonteCarloResultsAnalyzer()
                visualizer = monte_carlo_analysis.MonteCarloVisualizer()
                self.assertIsNotNone(analyzer)
                self.assertIsNotNone(visualizer)
            except Exception:
                # Expected if dependencies not available
                pass
                
        except ImportError as e:
            self.fail(f"Required classes not defined: {e}")
    
    def test_convenience_function_exists(self):
        """Test that the convenience function exists"""
        try:
            # Direct import to avoid dependency issues
            sys.path.insert(0, os.path.join(project_root, 'llm_atc', 'metrics'))
            import monte_carlo_analysis
            
            self.assertTrue(hasattr(monte_carlo_analysis, 'analyze_monte_carlo_results'))
            self.assertTrue(callable(monte_carlo_analysis.analyze_monte_carlo_results))
        except ImportError as e:
            self.fail(f"Convenience function not available: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)