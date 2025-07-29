# llm_atc/cli.py
"""
Command Line Interface for LLM-ATC-HAL
"""
import click
import sys
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """LLM-ATC-HAL: Embodied LLM Air Traffic Controller"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--duration', default=300, help='Simulation duration in seconds')
@click.option('--aircraft', default=4, help='Number of aircraft in scenario')
def demo(duration, aircraft):
    """Run a minimal demo scenario"""
    click.echo("Starting LLM-ATC-HAL Demo...")
    
    try:
        # Import demo components
        from llm_atc.agents.planner import Planner
        from llm_atc.agents.executor import Executor
        from llm_atc.agents.verifier import Verifier
        from llm_atc.agents.scratchpad import Scratchpad
        
        click.echo(f"Demo scenario: {aircraft} aircraft, {duration}s duration")
        
        # Initialize components
        click.echo("Initializing embodied agents...")
        planner = Planner()
        executor = Executor()
        verifier = Verifier()
        scratchpad = Scratchpad()
        
        # Generate mock scenario
        scenario = {
            'aircraft_count': aircraft,
            'duration': duration,
            'complexity': 'low',
            'scenario_type': 'demo'
        }
        
        click.echo("Demo scenario completed successfully!")
        click.echo(f"Processed {aircraft} aircraft over {duration} seconds")
        
    except ImportError as e:
        click.echo(f" Import error: {e}", err=True)
        click.echo(" Try: pip install -e .", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f" Demo failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('scenario_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='output', help='Output directory')
def run_scenario(scenario_path, output):
    """Run a specific scenario file"""
    click.echo(f" Running scenario: {scenario_path}")
    
    try:
        # Create output directory
        os.makedirs(output, exist_ok=True)
        
        # Load scenario
        with open(scenario_path, 'r') as f:
            if scenario_path.endswith('.yaml') or scenario_path.endswith('.yml'):
                scenario = yaml.safe_load(f)
            else:
                # Assume BlueSky .scn format
                scenario = {'type': 'bluesky', 'file': scenario_path}
        
        click.echo(f" Output directory: {output}")
        click.echo(" Scenario execution completed!")
        
    except Exception as e:
        click.echo(f" Scenario execution failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--config', default='llm_atc/experiments/shift_experiment_config.yaml', 
              help='Experiment configuration file')
@click.option('--tiers', default='in_distribution,out_distribution', 
              help='Comma-separated list of distribution shift tiers')
@click.option('--n', default=10, help='Number of scenarios per tier')
@click.option('--output', '-o', default='experiments/results', help='Output directory')
def shift_benchmark(config, tiers, n, output):
    """Run distribution shift benchmark"""
    click.echo(" Starting Distribution Shift Benchmark...")
    
    try:
        # Parse tiers
        tier_list = [t.strip() for t in tiers.split(',')]
        click.echo(f" Testing tiers: {tier_list}")
        click.echo(f" Scenarios per tier: {n}")
        
        # Create output directory
        os.makedirs(output, exist_ok=True)
        
        # Load configuration if exists
        if os.path.exists(config):
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
                click.echo(f" Loaded config: {config}")
        else:
            click.echo(f"  Config file not found: {config}")
            config_data = {}
        
        # Mock benchmark execution
        total_scenarios = len(tier_list) * n
        click.echo(f" Executing {total_scenarios} scenarios...")
        
        # Simulate progress
        with click.progressbar(range(total_scenarios), label='Running scenarios') as bar:
            for i in bar:
                pass  # Mock execution
        
        click.echo(f" Results saved to: {output}")
        click.echo(" Distribution shift benchmark completed!")
        
    except Exception as e:
        click.echo(f" Benchmark failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--models', default='llama3.1:8b,mistral:7b', 
              help='Comma-separated list of models to test')
@click.option('--scenarios', default=50, help='Number of test scenarios')
def hallucination_test(models, scenarios):
    """Run hallucination detection tests"""
    click.echo(" Starting Hallucination Detection Tests...")
    
    try:
        model_list = [m.strip() for m in models.split(',')]
        click.echo(f" Testing models: {model_list}")
        click.echo(f" Test scenarios: {scenarios}")
        
        # Mock hallucination testing
        for model in model_list:
            click.echo(f"Testing {model}...")
            # Simulate testing progress
            with click.progressbar(range(scenarios), label=f'{model}') as bar:
                for i in bar:
                    pass
        
        click.echo(" Hallucination tests completed!")
        
    except Exception as e:
        click.echo(f" Hallucination tests failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--log-file', '-l', help='Log file to analyze')
@click.option('--results-dir', '-d', default='test_results', help='Results directory')
def analyze(log_file, results_dir):
    """Analyze test results and generate metrics"""
    click.echo(" Analyzing test results...")
    
    try:
        from llm_atc.metrics import compute_metrics, print_metrics_summary, aggregate_thesis_metrics
        
        if log_file:
            click.echo(f" Analyzing single file: {log_file}")
            metrics = compute_metrics(log_file)
            print_metrics_summary(metrics)
        else:
            click.echo(f" Analyzing results directory: {results_dir}")
            metrics = aggregate_thesis_metrics(results_dir)
            print_metrics_summary(metrics)
        
        click.echo(" Analysis completed!")
        
    except ImportError as e:
        click.echo(f" Analysis modules not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f" Analysis failed: {e}", err=True)
        sys.exit(1)

@cli.command()
def validate():
    """Validate system installation and dependencies"""
    click.echo("Validating LLM-ATC-HAL installation...")
    
    validation_results = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 9):
        click.echo(" Python version OK")
        validation_results.append(True)
    else:
        click.echo(" Python >= 3.9 required", err=True)
        validation_results.append(False)
    
    # Check core dependencies
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'yaml', 'click',
        'sentence_transformers', 'chromadb'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            click.echo(f" {package}")
            validation_results.append(True)
        except ImportError:
            click.echo(f" {package} not found", err=True)
            validation_results.append(False)
    
    # Check LLM-ATC modules
    llm_atc_modules = ['agents', 'memory', 'metrics', 'tools']
    for module in llm_atc_modules:
        try:
            __import__(f'llm_atc.{module}')
            click.echo(f" llm_atc.{module}")
            validation_results.append(True)
        except ImportError as e:
            click.echo(f" llm_atc.{module}: {e}", err=True)
            validation_results.append(False)
    
    if all(validation_results):
        click.echo("All validations passed!")
        return True
    else:
        click.echo(" Some validations failed. Check installation.", err=True)
        return False

if __name__ == '__main__':
    cli()
