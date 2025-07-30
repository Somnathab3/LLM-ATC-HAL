#!/usr/bin/env python3
"""
Unit tests for the Monte Carlo benchmark CLI command.
Tests argument parsing, validation, and error handling.
"""

import unittest
import tempfile
import unittest.mock as mock
from pathlib import Path
from click.testing import CliRunner

# Mock all potentially missing dependencies before importing the CLI
mock_modules = [
    'numpy', 'pandas', 'matplotlib', 'matplotlib.pyplot', 'ollama',
    'sentence_transformers', 'chromadb', 'scipy', 'sklearn', 'seaborn',
    'scenarios.monte_carlo_runner', 'scenarios.monte_carlo_framework', 
    'scenarios.scenario_generator', 'llm_atc.tools.llm_prompt_engine',
    'llm_atc.tools.bluesky_tools'
]

import sys
for module in mock_modules:
    sys.modules[module] = mock.Mock()

from llm_atc.cli import cli


class TestMonteCarloBenchmarkCLI(unittest.TestCase):
    """Test cases for the monte-carlo-benchmark CLI command"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    def test_command_exists(self):
        """Test that the monte-carlo-benchmark command is registered"""
        commands = [cmd.name for cmd in cli.commands.values()]
        self.assertIn('monte-carlo-benchmark', commands)
    
    def test_command_help(self):
        """Test that the command help displays correctly"""
        result = self.runner.invoke(cli, ['monte-carlo-benchmark', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Run the Monte Carlo safety benchmark', result.output)
        self.assertIn('--num-horizontal', result.output)
        self.assertIn('--num-vertical', result.output)
        self.assertIn('--num-sector', result.output)
        self.assertIn('--complexities', result.output)
        self.assertIn('--shift-levels', result.output)
        self.assertIn('--horizon', result.output)
        self.assertIn('--output-dir', result.output)
    
    def test_default_arguments(self):
        """Test that default arguments are properly set"""
        # This test checks the argument parsing by mocking the benchmark execution
        with mock.patch('llm_atc.cli.MonteCarloBenchmark') as mock_benchmark, \
             mock.patch('llm_atc.cli.BenchmarkConfiguration') as mock_config, \
             mock.patch('llm_atc.cli.ComplexityTier') as mock_complexity, \
             mock.patch('llm_atc.cli.ScenarioType') as mock_scenario_type:
            
            # Setup mocks
            mock_complexity.SIMPLE = 'SIMPLE'
            mock_complexity.MODERATE = 'MODERATE'
            mock_complexity.COMPLEX = 'COMPLEX'
            
            mock_scenario_type.HORIZONTAL = 'HORIZONTAL'
            mock_scenario_type.VERTICAL = 'VERTICAL'
            mock_scenario_type.SECTOR = 'SECTOR'
            
            mock_benchmark_instance = mock.Mock()
            mock_benchmark_instance.run.return_value = {
                'successful_scenarios': 5,
                'total_scenarios': 5
            }
            mock_benchmark.return_value = mock_benchmark_instance
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = self.runner.invoke(cli, [
                    'monte-carlo-benchmark',
                    '--output-dir', temp_dir
                ])
                
                # Should succeed with mocked dependencies
                self.assertEqual(result.exit_code, 0)
                self.assertIn('Starting Monte Carlo Safety Benchmark', result.output)
                self.assertIn('Configuration Summary', result.output)
                
                # Verify BenchmarkConfiguration was called with expected defaults
                self.assertTrue(mock_config.called)
                config_call = mock_config.call_args
                self.assertEqual(config_call[1]['num_scenarios_per_type'], 50)  # max of defaults
                self.assertEqual(config_call[1]['time_horizon_minutes'], 5.0)
                self.assertEqual(config_call[1]['output_directory'], temp_dir)
    
    def test_invalid_complexity_tier(self):
        """Test handling of invalid complexity tier"""
        with mock.patch('llm_atc.cli.ComplexityTier') as mock_complexity, \
             mock.patch('llm_atc.cli.ScenarioType') as mock_scenario_type:
            
            # Setup mocks
            mock_complexity.SIMPLE = 'SIMPLE'
            mock_scenario_type.HORIZONTAL = 'HORIZONTAL'
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = self.runner.invoke(cli, [
                    'monte-carlo-benchmark',
                    '--complexities', 'invalid,simple',
                    '--output-dir', temp_dir
                ])
                
                # Should continue with valid complexities and warn about invalid ones
                self.assertEqual(result.exit_code, 0)
                self.assertIn('Warning: Unknown complexity tier', result.output)
    
    def test_no_valid_complexity_tiers(self):
        """Test error when no valid complexity tiers are provided"""
        with mock.patch('llm_atc.cli.ComplexityTier') as mock_complexity:
            mock_complexity.SIMPLE = 'SIMPLE'
            
            result = self.runner.invoke(cli, [
                'monte-carlo-benchmark',
                '--complexities', 'invalid,unknown'
            ])
            
            self.assertEqual(result.exit_code, 1)
            self.assertIn('No valid complexity tiers specified', result.output)
    
    def test_no_scenario_types(self):
        """Test error when all scenario counts are zero"""
        with mock.patch('llm_atc.cli.ComplexityTier') as mock_complexity:
            mock_complexity.SIMPLE = 'SIMPLE'
            
            result = self.runner.invoke(cli, [
                'monte-carlo-benchmark',
                '--num-horizontal', '0',
                '--num-vertical', '0',
                '--num-sector', '0'
            ])
            
            self.assertEqual(result.exit_code, 1)
            self.assertIn('At least one scenario type must have count > 0', result.output)
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        with mock.patch('llm_atc.cli.MonteCarloBenchmark') as mock_benchmark, \
             mock.patch('llm_atc.cli.BenchmarkConfiguration'), \
             mock.patch('llm_atc.cli.ComplexityTier') as mock_complexity, \
             mock.patch('llm_atc.cli.ScenarioType') as mock_scenario_type:
            
            mock_complexity.SIMPLE = 'SIMPLE'
            mock_scenario_type.HORIZONTAL = 'HORIZONTAL'
            
            mock_benchmark_instance = mock.Mock()
            mock_benchmark_instance.run.return_value = {
                'successful_scenarios': 1,
                'total_scenarios': 1
            }
            mock_benchmark.return_value = mock_benchmark_instance
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "new_subdir" / "monte_carlo"
                
                result = self.runner.invoke(cli, [
                    'monte-carlo-benchmark',
                    '--output-dir', str(output_dir)
                ])
                
                self.assertEqual(result.exit_code, 0)
                self.assertTrue(output_dir.exists())
                self.assertTrue(output_dir.is_dir())
    
    def test_import_error_handling(self):
        """Test handling of import errors"""
        # Clear the mock for monte_carlo_runner to simulate import error
        if 'scenarios.monte_carlo_runner' in sys.modules:
            del sys.modules['scenarios.monte_carlo_runner']
        
        result = self.runner.invoke(cli, ['monte-carlo-benchmark'])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn('Import error', result.output)
        self.assertIn('Make sure all required modules are installed', result.output)


def run_basic_tests():
    """Run basic tests that should always pass"""
    print("🧪 Running basic CLI tests...")
    
    # Test 1: Command exists
    try:
        commands = [cmd.name for cmd in cli.commands.values()]
        if 'monte-carlo-benchmark' in commands:
            print("✅ Command registration test passed")
        else:
            print("❌ Command registration test failed")
            return False
    except Exception as e:
        print(f"❌ Command registration test failed: {e}")
        return False
    
    # Test 2: Help works
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ['monte-carlo-benchmark', '--help'])
        if result.exit_code == 0 and 'Run the Monte Carlo safety benchmark' in result.output:
            print("✅ Help command test passed")
        else:
            print("❌ Help command test failed")
            return False
    except Exception as e:
        print(f"❌ Help command test failed: {e}")
        return False
    
    # Test 3: Error handling for missing dependencies
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ['monte-carlo-benchmark'])
        if result.exit_code == 1 and 'Import error' in result.output:
            print("✅ Error handling test passed")
        else:
            print("❌ Error handling test failed")
            return False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    print("✅ All basic tests passed!")
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--basic':
        # Run basic tests that don't require full unittest framework
        success = run_basic_tests()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        unittest.main()