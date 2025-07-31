# LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Hallucination Detection

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Advanced Large Language Model Air Traffic Controller with Comprehensive Hallucination Detection & Monte Carlo Safety Benchmarking**

## 🎯 Project Summary

LLM-ATC-HAL is a cutting-edge research platform for testing Large Language Model (LLM) integration into air traffic control systems. The project focuses on **embodied LLM air traffic controllers** with comprehensive **hallucination detection** and **Monte Carlo safety benchmarking**.

### Key Capabilities

- **🛩️ Scenario Generation**: Configurable ATC scenarios (horizontal, vertical, sector conflicts) with distribution shift testing
- **🤖 LLM Conflict Detection & Resolution**: Multi-model ensemble approach with confidence scoring
- **🌐 BlueSky Integration**: Industry-standard flight simulation platform integration
- **📊 Comprehensive Metrics**: False positive/negative analysis, safety margin quantification, ICAO compliance
- **📈 Visualization**: Real-time monitoring, results analysis, and performance dashboards
- **🔍 Hallucination Detection**: Multi-layer detection across different conflict resolution methods
- **⚡ Monte Carlo Benchmarking**: Statistical validation across thousands of scenarios

## � Quick Start

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai) for local LLM support
- Git

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull codellama:7b

# 5. Validate installation
python cli.py validate
```

### Example Commands

```bash
# Run system demo
python cli.py demo --duration 300 --aircraft 4

# Monte Carlo benchmark with enhanced output
python cli.py monte-carlo-benchmark --enhanced-output --num-horizontal 10 --num-vertical 10 --num-sector 10

# Distribution shift testing
python cli.py shift-benchmark --tiers in_distribution,moderate_shift,extreme_shift --n 25

# Hallucination detection testing
python cli.py hallucination-test --models llama3.1:8b,mistral:7b --scenarios 100

# Analyze results
python cli.py analyze --results-dir experiments/monte_carlo_results

# Quick validation test
python cli.py quick-test --quick
```

## 📁 Directory Structure

```
LLM-ATC-HAL/
├── cli.py                               # Main command-line interface
├── pyproject.toml                       # Project configuration and dependencies
├── requirements.txt                     # Python package requirements
├── *.yaml                              # Configuration files for scenarios, shifts, BlueSky
│
├── scenarios/                           # Core scenario generation and execution
│   ├── scenario_generator.py           # Environment-specific scenario creation (Horizontal/Vertical/Sector)
│   ├── monte_carlo_framework.py        # BlueSky-integrated scenario generation framework
│   └── monte_carlo_runner.py           # Main benchmark orchestration pipeline
│
├── llm_atc/                            # Core LLM-ATC components
│   ├── agents/                         # Multi-agent LLM coordination
│   ├── tools/                          # BlueSky tools and LLM prompt engines
│   ├── metrics/                        # Performance analysis and safety metrics
│   ├── experiments/                    # Experiment runners and configurations
│   └── baseline_models/                # Traditional ATC conflict detection
│
├── llm_interface/                       # LLM client and ensemble management
│   ├── llm_client.py                   # Ollama and external LLM integration
│   ├── ensemble.py                     # Multi-model ensemble coordination
│   └── filter_sort.py                  # Response filtering and ranking
│
├── bluesky_sim/                        # BlueSky simulator integration
│   ├── scenarios.py                    # BlueSky scenario loading
│   └── simulation_runner.py            # Simulation execution engine
│
├── analysis/                           # Results analysis and visualization
│   ├── enhanced_hallucination_detection.py  # Advanced hallucination detection algorithms
│   ├── metrics.py                      # Statistical analysis and performance metrics
│   └── visualisation.py                # Results plotting and dashboard generation
│
├── solver/                             # Conflict resolution algorithms
│   └── conflict_solver.py              # Mathematical conflict resolution methods
│
├── experiments/                        # Test results and experiment data
├── visualizations/                     # Generated plots and analysis outputs
├── logs/                               # System logs and debug information
└── data/                               # Scenario data and cached results
```

## 🔧 Core Components

### scenarios/
- **`scenario_generator.py`**: Environment-specific scenario creation with precise ground truth conflict labeling
- **`monte_carlo_framework.py`**: BlueSky-integrated range-based scenario generation with distribution shift support
- **`monte_carlo_runner.py`**: Three-stage pipeline orchestrator (detection → resolution → verification)

### llm_atc/
- **`agents/`**: Multi-agent LLM coordination for complex conflict resolution
- **`tools/`**: BlueSky command generation, LLM prompt engines, and simulation interfaces
- **`metrics/`**: Comprehensive performance analysis including false positive/negative detection
- **`experiments/`**: Distribution shift testing and specialized experiment runners
- **`baseline_models/`**: Traditional ATC conflict detection for comparison benchmarking

### llm_interface/
- **`llm_client.py`**: Unified interface for Ollama local models and external LLM APIs
- **`ensemble.py`**: Multi-model ensemble coordination with weighted voting
- **`filter_sort.py`**: Response quality filtering and confidence-based ranking

### bluesky_sim/
- **`scenarios.py`**: BlueSky scenario loading and validation
- **`simulation_runner.py`**: BlueSky simulation execution with real-time monitoring

### analysis/
- **`enhanced_hallucination_detection.py`**: Advanced multi-layer hallucination detection
- **`metrics.py`**: Statistical analysis, ICAO compliance checking, safety margin calculation
- **`visualisation.py`**: Performance dashboards, conflict visualization, trend analysis

### solver/
- **`conflict_solver.py`**: Mathematical conflict resolution algorithms and optimization

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| `scenario_ranges.yaml` | Defines parameter ranges for scenario generation (aircraft counts, speeds, altitudes) |
| `distribution_shift_levels.yaml` | Three-tier distribution shift configuration (in-distribution, moderate, extreme) |
| `bluesky_config.yaml` | BlueSky simulator connection settings and simulation parameters |
| `comprehensive_test_config.yaml` | Comprehensive testing campaign configuration with model and performance settings |

## 🎮 Usage Examples

### Basic Demo
```bash
# Run minimal demo scenario
python cli.py demo --duration 300 --aircraft 4
```

### Monte Carlo Benchmarking
```bash
# Standard benchmark with enhanced output
python cli.py monte-carlo-benchmark --enhanced-output

# Custom scenario counts with specific complexities
python cli.py monte-carlo-benchmark \
    --num-horizontal 25 \
    --num-vertical 25 \
    --num-sector 25 \
    --complexities simple,moderate,complex \
    --shift-levels in_distribution,moderate_shift \
    --horizon 5 \
    --max-interventions 5 \
    --step-size 10.0 \
    --enhanced-output

# Quick test for development
python cli.py quick-test --quick
```

### Distribution Shift Testing
```bash
# Test robustness across distribution shifts
python cli.py shift-benchmark \
    --tiers in_distribution,moderate_shift,extreme_shift \
    --n 50 \
    --output experiments/shift_analysis
```

### Hallucination Detection
```bash
# Multi-model hallucination testing
python cli.py hallucination-test \
    --models llama3.1:8b,mistral:7b,codellama:7b \
    --scenarios 200
```

### Analysis and Reporting
```bash
# Analyze specific experiment results
python cli.py analyze --results-dir experiments/monte_carlo_results

# Analyze specific log file
python cli.py analyze --log-file experiments/results/benchmark_abc123.json

# System validation
python cli.py validate
```

## 📊 Metrics & Reporting

### Output Locations
- **Detection Comparison**: `experiments/monte_carlo_results/detection_comparison.csv`
- **Results Summary**: `experiments/monte_carlo_results/results_summary.json`
- **Detailed Logs**: `experiments/monte_carlo_results/logs/`
- **Visualizations**: `visualizations/` and `experiments/*/plots/`

### Loading Results in Python
```python
import pandas as pd
import json

# Load detection comparison data
df = pd.read_csv('experiments/monte_carlo_results/detection_comparison.csv')
print(df.groupby(['scenario_type', 'complexity'])['accuracy'].mean())

# Load results summary
with open('experiments/monte_carlo_results/results_summary.json') as f:
    summary = json.load(f)
    print(f"Overall accuracy: {summary['overall_metrics']['detection_accuracy']:.3f}")
```

### Automated Reporting
```bash
# Generate comprehensive analysis report
python cli.py monte-carlo-benchmark --auto-analyze --analysis-format comprehensive

# Custom analysis on existing results
python analysis/metrics.py --input experiments/monte_carlo_results --output reports/
```

## 🧪 Testing & CI

### Running Tests
```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_atc --cov-report=html

# Run specific test categories
pytest tests/test_scenario_generation.py
pytest tests/test_llm_prompt_engine.py
pytest tests/test_metrics.py
```

### Code Quality
```bash
# Linting and formatting
ruff check .
black . --check

# Type checking
mypy llm_atc/

# System validation
python cli.py validate
```

### Test Coverage
Tests are located in `tests/` and cover:
- **Scenario Generation**: Parameter sampling, conflict injection, ground truth calculation
- **LLM Integration**: Prompt formatting, response parsing, ensemble coordination
- **BlueSky Tools**: Command generation, simulation execution, state monitoring
- **Metrics Analysis**: False positive/negative detection, safety margin calculation
- **End-to-End Pipeline**: Full workflow from scenario generation to results analysis

## 🚦 Performance Targets

- **Detection Accuracy**: >85% AUROC for hallucination detection
- **Response Time**: <2s mean response time for conflict resolution
- **Safety Compliance**: 95% ICAO standard compliance
- **Horizontal Separation**: 5.0 nautical miles minimum
- **Vertical Separation**: 1000 feet minimum

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run development test suite
pytest tests/ --disable-warnings
```

### Code Style Guidelines
- **Python 3.9+** with PEP 8 compliance
- **Type hints** on all public functions and dataclasses
- **Docstrings** in Google or NumPy style
- **Absolute imports** rooted at project package
- **Module-level logging** with `logger = logging.getLogger(__name__)`

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding guidelines
4. Add tests for new functionality
5. Run the test suite (`pytest tests/`)
6. Submit a pull request

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Homepage**: [GitHub Repository](https://github.com/Somnathab3/LLM-ATC-HAL)
- **Documentation**: [Project Wiki](https://github.com/Somnathab3/LLM-ATC-HAL/wiki)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/Somnathab3/LLM-ATC-HAL/issues)
- **PyPI Package**: `pip install llm-atc`

## � Research & Citation

This project supports academic research in LLM safety and aviation systems. If you use LLM-ATC-HAL in your research, please cite:

```bibtex
@software{llm_atc_hal,
  title={LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Hallucination Detection},
  author={Somnath},
  year={2025},
  url={https://github.com/Somnathab3/LLM-ATC-HAL}
}
```

---

**Status**: Production-ready alpha version with comprehensive testing and documentation.
