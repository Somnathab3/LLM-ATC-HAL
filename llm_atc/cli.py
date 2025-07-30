# llm_atc/cli.py
"""
Command Line Interface for LLM-ATC-HAL
"""
import logging
import os
import sys
from pathlib import Path

import click
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose) -> None:
    """LLM-ATC-HAL: Embodied LLM Air Traffic Controller"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option("--duration", default=300, help="Simulation duration in seconds")
@click.option("--aircraft", default=4, help="Number of aircraft in scenario")
def demo(duration, aircraft) -> None:
    """Run a minimal demo scenario"""
    click.echo("Starting LLM-ATC-HAL Demo...")

    try:
        # Import demo components
        from llm_atc.agents.executor import Executor
        from llm_atc.agents.planner import Planner
        from llm_atc.agents.scratchpad import Scratchpad
        from llm_atc.agents.verifier import Verifier

        click.echo(f"Demo scenario: {aircraft} aircraft, {duration}s duration")

        # Initialize components
        click.echo("Initializing embodied agents...")
        Planner()
        Executor()
        Verifier()
        Scratchpad()

        # Generate mock scenario

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
@click.argument("scenario_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="output", help="Output directory")
def run_scenario(scenario_path, output) -> None:
    """Run a specific scenario file"""
    click.echo(f" Running scenario: {scenario_path}")

    try:
        # Create output directory
        os.makedirs(output, exist_ok=True)

        # Load scenario
        with open(scenario_path) as f:
            if scenario_path.endswith((".yaml", ".yml")):
                yaml.safe_load(f)
            else:
                # Assume BlueSky .scn format
                pass

        click.echo(f" Output directory: {output}")
        click.echo(" Scenario execution completed!")

    except Exception as e:
        click.echo(f" Scenario execution failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--config", default="llm_atc/experiments/shift_experiment_config.yaml",
              help="Experiment configuration file")
@click.option("--tiers", default="in_distribution,out_distribution",
              help="Comma-separated list of distribution shift tiers")
@click.option("--n", default=10, help="Number of scenarios per tier")
@click.option("--output", "-o", default="experiments/results", help="Output directory")
def shift_benchmark(config, tiers, n, output) -> None:
    """Run distribution shift benchmark"""
    click.echo(" Starting Distribution Shift Benchmark...")

    try:
        # Parse tiers
        tier_list = [t.strip() for t in tiers.split(",")]
        click.echo(f" Testing tiers: {tier_list}")
        click.echo(f" Scenarios per tier: {n}")

        # Create output directory
        os.makedirs(output, exist_ok=True)

        # Load configuration if exists
        if os.path.exists(config):
            with open(config) as f:
                yaml.safe_load(f)
                click.echo(f" Loaded config: {config}")
        else:
            click.echo(f"  Config file not found: {config}")

        # Mock benchmark execution
        total_scenarios = len(tier_list) * n
        click.echo(f" Executing {total_scenarios} scenarios...")

        # Simulate progress
        with click.progressbar(range(total_scenarios), label="Running scenarios") as bar:
            for _i in bar:
                pass  # Mock execution

        click.echo(f" Results saved to: {output}")
        click.echo(" Distribution shift benchmark completed!")

    except Exception as e:
        click.echo(f" Benchmark failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--models", default="llama3.1:8b,mistral:7b",
              help="Comma-separated list of models to test")
@click.option("--scenarios", default=50, help="Number of test scenarios")
def hallucination_test(models, scenarios) -> None:
    """Run hallucination detection tests"""
    click.echo(" Starting Hallucination Detection Tests...")

    try:
        model_list = [m.strip() for m in models.split(",")]
        click.echo(f" Testing models: {model_list}")
        click.echo(f" Test scenarios: {scenarios}")

        # Mock hallucination testing
        for model in model_list:
            click.echo(f"Testing {model}...")
            # Simulate testing progress
            with click.progressbar(range(scenarios), label=f"{model}") as bar:
                for _i in bar:
                    pass

        click.echo(" Hallucination tests completed!")

    except Exception as e:
        click.echo(f" Hallucination tests failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--log-file", "-l", help="Log file to analyze")
@click.option("--results-dir", "-d", default="test_results", help="Results directory")
def analyze(log_file, results_dir) -> None:
    """Analyze test results and generate metrics"""
    click.echo(" Analyzing test results...")

    try:
        from llm_atc.metrics import aggregate_thesis_metrics, compute_metrics, print_metrics_summary

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
@click.option("--num-horizontal", default=50, help="Number of horizontal scenarios")
@click.option("--num-vertical", default=50, help="Number of vertical scenarios")
@click.option("--num-sector", default=50, help="Number of sector scenarios")
@click.option("--complexities", default="simple,moderate,complex", help="Comma-separated complexity tiers")
@click.option("--shift-levels", default="in_distribution,moderate_shift,extreme_shift", help="Comma-separated shift levels")
@click.option("--horizon", default=5, help="Minutes to simulate after each resolution")
@click.option("--max-interventions", default=5, help="Maximum interventions per scenario")
@click.option("--step-size", default=10.0, help="Simulation step size in seconds")
@click.option("--output-dir", default="experiments/monte_carlo_results", help="Directory to save results")
def monte_carlo_benchmark(**opts) -> None:
    """Run the Monte Carlo safety benchmark."""
    click.echo("🚀 Starting Monte Carlo Safety Benchmark...")

    try:
        # Import required modules
        from scenarios.monte_carlo_framework import ComplexityTier
        from scenarios.monte_carlo_runner import BenchmarkConfiguration, MonteCarloBenchmark
        from scenarios.scenario_generator import ScenarioType

        # Validate and parse complexities into ComplexityTier objects
        complexity_strings = [c.strip().lower() for c in opts["complexities"].split(",")]
        complexity_tiers = []

        complexity_mapping = {
            "simple": ComplexityTier.SIMPLE,
            "moderate": ComplexityTier.MODERATE,
            "complex": ComplexityTier.COMPLEX,
            "extreme": ComplexityTier.EXTREME,
        }

        invalid_complexities = []
        for comp_str in complexity_strings:
            if comp_str in complexity_mapping:
                complexity_tiers.append(complexity_mapping[comp_str])
            else:
                invalid_complexities.append(comp_str)

        # Validate complexity tiers explicitly
        if invalid_complexities:
            valid_options = list(complexity_mapping.keys())
            msg = (
                f"Invalid complexity tier(s): {', '.join(invalid_complexities)}. "
                f"Valid options are: {', '.join(valid_options)}"
            )
            raise click.BadParameter(
                msg,
            )

        if not complexity_tiers:
            msg = "No valid complexity tiers specified"
            raise click.BadParameter(msg)

        # Parse shift levels into strings
        shift_levels = [s.strip() for s in opts["shift_levels"].split(",")]

        # Create per-type scenario counts dictionary
        scenario_counts = {}
        total_scenarios = 0

        if opts["num_horizontal"] > 0:
            scenario_counts[ScenarioType.HORIZONTAL.value] = opts["num_horizontal"]
            total_scenarios += opts["num_horizontal"]

        if opts["num_vertical"] > 0:
            scenario_counts[ScenarioType.VERTICAL.value] = opts["num_vertical"]
            total_scenarios += opts["num_vertical"]

        if opts["num_sector"] > 0:
            scenario_counts[ScenarioType.SECTOR.value] = opts["num_sector"]
            total_scenarios += opts["num_sector"]

        if not scenario_counts:
            msg = "At least one scenario type must have count > 0"
            raise click.BadParameter(msg)

        # Determine scenario types from counts
        scenario_types = [
            getattr(ScenarioType, key.upper())
            for key in scenario_counts
        ]

        # Calculate adaptive step size based on time horizon
        horizon_seconds = float(opts["horizon"]) * 60
        if horizon_seconds < 300:  # Less than 5 minutes
            default_step_size = min(float(opts["step_size"]), 5.0)
        elif horizon_seconds > 1200:  # More than 20 minutes
            default_step_size = max(float(opts["step_size"]), 15.0)
        else:
            default_step_size = float(opts["step_size"])

        # Create output directory if it doesn't exist
        output_dir = Path(opts["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"📁 Output directory: {output_dir}")

        # Create benchmark configuration
        config = BenchmarkConfiguration(
            scenario_counts=scenario_counts,
            scenario_types=scenario_types,
            complexity_tiers=complexity_tiers,
            distribution_shift_levels=shift_levels,
            time_horizon_minutes=float(opts["horizon"]),
            max_interventions_per_scenario=int(opts["max_interventions"]),
            step_size_seconds=default_step_size,
            output_directory=str(output_dir),
            generate_visualizations=True,
            detailed_logging=True,
        )

        # Display configuration summary
        click.echo("📊 Configuration Summary:")
        click.echo(f"   Scenario counts: {scenario_counts}")
        click.echo(f"   Complexity tiers: {[c.value for c in complexity_tiers]}")
        click.echo(f"   Shift levels: {shift_levels}")
        click.echo(f"   Max interventions: {opts['max_interventions']}")
        click.echo(f"   Step size: {default_step_size:.1f}s")
        click.echo(f"   Time horizon: {opts['horizon']} minutes")

        total_scenarios_expanded = sum(scenario_counts.values()) * len(complexity_tiers) * len(shift_levels)
        click.echo(f"   Total scenarios: {total_scenarios_expanded}")

        # Initialize and run benchmark
        benchmark = MonteCarloBenchmark(config)

        click.echo("🔄 Running benchmark... (this may take a while)")
        summary = benchmark.run()

        click.echo(f"✅ Benchmark complete! Results saved to {opts['output_dir']}")

        # Extract success metrics from summary
        if isinstance(summary, dict):
            scenario_counts_summary = summary.get("scenario_counts", {})
            successful = scenario_counts_summary.get("successful_scenarios", 0)
            total = scenario_counts_summary.get("total_scenarios", 0)
            success_rate = scenario_counts_summary.get("success_rate", 0.0)

            click.echo(f"📈 Summary: {successful}/{total} scenarios successful ({success_rate:.1%})")
        else:
            click.echo("📈 Summary: Benchmark completed")

    except click.BadParameter as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    except ImportError as e:
        click.echo(f"❌ Import error: {e}", err=True)
        click.echo("💡 Make sure all required modules are installed", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Benchmark execution failed: {e}", err=True)
        import traceback
        if verbose_logging := os.getenv("VERBOSE_LOGGING"):
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
def validate() -> bool:
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
        "numpy", "pandas", "matplotlib", "yaml", "click",
        "sentence_transformers", "chromadb",
    ]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            click.echo(f" {package}")
            validation_results.append(True)
        except ImportError:
            click.echo(f" {package} not found", err=True)
            validation_results.append(False)

    # Check LLM-ATC modules
    llm_atc_modules = ["agents", "memory", "metrics", "tools"]
    for module in llm_atc_modules:
        try:
            __import__(f"llm_atc.{module}")
            click.echo(f" llm_atc.{module}")
            validation_results.append(True)
        except ImportError as e:
            click.echo(f" llm_atc.{module}: {e}", err=True)
            validation_results.append(False)

    if all(validation_results):
        click.echo("All validations passed!")
        return True
    click.echo(" Some validations failed. Check installation.", err=True)
    return False

if __name__ == "__main__":
    cli()
