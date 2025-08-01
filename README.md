# LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Hallucination Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/Somnathab3/LLM-ATC-HAL/workflows/CI/badge.svg)](https://github.com/Somnathab3/LLM-ATC-HAL/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Advanced Large Language Model Air Traffic Controller with Comprehensive Hallucination Detection & Monte Carlo Safety Benchmarking**

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [CLI Reference](#cli-reference)
- [Fine-tuned Models](#fine-tuned-models)
- [Testing & Benchmarking](#testing--benchmarking)
- [Results Analysis](#results-analysis)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Performance Targets](#performance-targets)
- [Citation](#citation)
- [License](#license)

## Overview

LLM-ATC-HAL is a cutting-edge research platform for integrating Large Language Models (LLMs) into air traffic control systems. The project focuses on **embodied LLM air traffic controllers** with comprehensive **hallucination detection** and **Monte Carlo safety benchmarking**.

The system generates realistic air traffic conflict scenarios, feeds them to LLM-based controllers for conflict detection and resolution, then validates the responses through multi-layer hallucination detection and safety analysis.

### Research Focus

- **Safety-Critical AI**: Testing LLM reliability in life-critical aviation scenarios
- **Hallucination Detection**: Multi-layer detection of AI reasoning errors
- **Distribution Shift Robustness**: Testing performance across varied operational conditions
- **Monte Carlo Validation**: Statistical validation across thousands of scenarios
- **BlueSky Integration**: Industry-standard flight simulation platform integration

## Key Features

### 🛩️ Scenario Generation
- **Environment-Specific Scenarios**: Horizontal, vertical, and sector conflict scenarios
- **Configurable Complexity**: Simple to extreme difficulty levels with 2-20 aircraft
- **Distribution Shift Testing**: In-distribution, moderate, and extreme operational shifts
- **Ground Truth Labeling**: Precise conflict timing and separation calculations

### 🤖 LLM Integration
- **Multi-Model Support**: Ollama integration with Llama 3.1, Mistral, CodeLlama
- **Fine-tuned Models**: Specialized `llama3.1-bsky` model trained on BlueSky Gym scenarios
- **Ensemble Methods**: Weighted voting across multiple LLM models
- **Prompt Engineering**: Sophisticated ICAO-compliant prompt templates

### 🔍 Hallucination Detection
- **Multi-Layer Detection**: Aircraft existence, altitude confusion, protocol violations
- **Confidence Scoring**: Quantitative assessment of response reliability
- **Safety Validation**: ICAO compliance checking and separation standard validation
- **Real-time Monitoring**: Live detection during simulation execution

### 📊 Comprehensive Metrics
- **Safety Margins**: Horizontal (5nm) and vertical (1000ft) separation analysis
- **False Positive/Negative**: Detailed classification accuracy metrics
- **Performance Benchmarking**: Response time, success rate, and efficiency analysis
- **Statistical Validation**: Monte Carlo analysis across large scenario sets

### 🌐 BlueSky Simulator Integration
- **Industry Standard**: Integration with BlueSky open-source flight simulator
- **Real-time Simulation**: Step-by-step conflict evolution and resolution
- **Mock Mode**: Fallback simulation for development without BlueSky installation
- **Command Validation**: BlueSky command syntax verification and execution

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Scenario        │────│ LLM Prompt      │────│ BlueSky         │
│ Generator       │    │ Engine          │    │ Simulator       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Monte Carlo     │────│ Hallucination   │────│ Safety Metrics  │
│ Framework       │    │ Detection       │    │ Analysis        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Core Pipeline**: Scenario → LLM → BlueSky → Metrics
1. **Scenario Generation**: Create conflict scenarios with ground truth
2. **LLM Processing**: Detect conflicts and generate resolution commands
3. **Simulation**: Execute commands in BlueSky and monitor results
4. **Analysis**: Validate safety, detect hallucinations, compute metrics

## Installation

### Prerequisites

- **Python 3.9+** (tested with 3.9-3.12)
- **Git** for repository cloning
- **Ollama** for local LLM support
- **Optional**: BlueSky simulator for full integration

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/Somnathab3/LLM-ATC-HAL.git
cd LLM-ATC-HAL

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install BlueSky simulator (optional, for full integration)
pip install -e ".[bluesky]"
```

### Step 3: Install LLM Models

```bash
# Install Ollama (https://ollama.ai)
# Then pull required models:

ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull codellama:7b

# Fine-tuned model (if available)
ollama pull llama3.1-bsky
```

### Step 4: Validation

```bash
# Validate installation
python cli.py --help

# Run system check
python cli.py demo --duration 60 --aircraft 2
```

## Configuration

The system uses YAML configuration files for flexible scenario generation and testing:

### Core Configuration Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `scenario_ranges.yaml` | Scenario parameters | Aircraft counts, speeds, altitudes, airspace regions |
| `distribution_shift_levels.yaml` | Distribution shift testing | In-distribution, moderate, extreme operational conditions |
| `bluesky_config.yaml` | BlueSky integration | Connection settings, simulation parameters |

### Example: Scenario Configuration

```yaml
# scenario_ranges.yaml
aircraft:
  count:
    simple: [2, 3]
    moderate: [4, 6] 
    complex: [8, 12]
    extreme: [18, 20]

geography:
  airspace_regions:
    EHAM_TMA:  # Amsterdam
      center: [52.3086, 4.7639]
      radius_nm: [40, 60]
```

### Example: Distribution Shift Configuration

```yaml
# distribution_shift_levels.yaml
in_distribution:
  traffic_density_multiplier: 1.0
  weather:
    wind:
      speed_shift_kts: [0, 0]

extreme_shift:
  traffic_density_multiplier: 2.5
  weather:
    wind:
      speed_shift_kts: [30, 50]
```

## Usage Examples

### Basic Scenario Execution

```bash
# Run a simple demo
python cli.py demo --duration 300 --aircraft 4

# Generate and analyze a specific scenario
python cli.py run-scenario data/scenarios/example.yaml --output results/
```

### Monte Carlo Benchmarking

```bash
# Quick test (2 scenarios per type)
python cli.py quick-test --quick

# Medium test (15 scenarios per type)
python cli.py quick-test --medium

# Full benchmark with custom parameters
python cli.py monte-carlo-benchmark \
    --num-horizontal 25 \
    --num-vertical 25 \
    --num-sector 25 \
    --complexities simple,moderate,complex \
    --shift-levels in_distribution,moderate_shift \
    --enhanced-output
```

### Distribution Shift Testing

```bash
# Test robustness across operational conditions
python cli.py shift-benchmark \
    --tiers in_distribution,moderate_shift,extreme_shift \
    --n 50 \
    --output experiments/shift_analysis
```

### Hallucination Detection Testing

```bash
# Multi-model hallucination testing
python cli.py hallucination-test \
    --models llama3.1:8b,mistral:7b,llama3.1-bsky \
    --scenarios 200
```

### Model Performance Comparison

```bash
# Compare base vs fine-tuned model
python compare_model_performance.py \
    --base-model llama3.1:8b \
    --fine-tuned llama3.1-bsky \
    --scenarios 100
```

## CLI Reference

| Command | Description | Key Options |
|---------|-------------|-------------|
| `demo` | Run minimal demo scenario | `--duration`, `--aircraft` |
| `quick-test` | Fast validation tests | `--quick`, `--medium` |
| `monte-carlo-benchmark` | Full statistical benchmark | `--num-*`, `--complexities`, `--enhanced-output` |
| `shift-benchmark` | Distribution shift testing | `--tiers`, `--n`, `--output` |
| `hallucination-test` | Hallucination detection tests | `--models`, `--scenarios` |
| `run-scenario` | Execute specific scenario file | `--output` |
| `analyze` | Analyze test results | `--results-dir`, `--log-file` |

### Advanced CLI Examples

```bash
# Comprehensive benchmark with analysis
python cli.py monte-carlo-benchmark \
    --num-horizontal 100 \
    --num-vertical 100 \
    --num-sector 100 \
    --horizon 10 \
    --max-interventions 5 \
    --auto-analyze \
    --analysis-format comprehensive

# Strict BlueSky mode (require real simulator)
python cli.py monte-carlo-benchmark \
    --strict-bluesky \
    --num-horizontal 10

# Mock simulation mode (development)
python cli.py monte-carlo-benchmark \
    --mock-simulation \
    --enhanced-output
```

## Fine-tuned Models

The project includes fine-tuned models specialized for air traffic control:

### llama3.1-bsky Model

- **Base Model**: Llama 3.1 8B
- **Training Data**: 510 BlueSky Gym RL scenarios
- **Environments**: HorizontalCREnv, VerticalCREnv, SectorCREnv
- **Performance**: Enhanced accuracy on aviation-specific tasks

### Using Fine-tuned Models

```python
# Direct usage
import ollama
client = ollama.Client()
response = client.chat(
    model="llama3.1-bsky",
    messages=[{"role": "user", "content": "Analyze conflict scenario..."}]
)

# Ensemble integration
from llm_interface.ensemble import OllamaEnsembleClient
ensemble = OllamaEnsembleClient()
ensemble.add_model("llama3.1-bsky", weight=0.6, role="primary")
```

## Testing & Benchmarking

### Quick Validation Tests

```bash
# Fast development testing
python cli.py quick-test --quick          # 2 scenarios per type
python cli.py quick-test --medium         # 15 scenarios per type
```

### Comprehensive Benchmarking

```bash
# Statistical validation benchmark
python cli.py monte-carlo-benchmark \
    --num-horizontal 200 \
    --num-vertical 200 \
    --num-sector 200 \
    --complexities simple,moderate,complex \
    --shift-levels in_distribution,moderate_shift,extreme_shift
```

### Performance Analysis

```bash
# PowerShell script for comprehensive testing
.\run_comprehensive_benchmark_and_analyze.ps1

# Python analysis tools
python analysis/metrics.py --input experiments/results/
python analysis/visualisation.py --data experiments/results/detection_comparison.csv
```

## Results Analysis

### Output Locations

| Type | Location | Description |
|------|----------|-------------|
| Detection Results | `experiments/*/detection_comparison.csv` | Per-scenario detection accuracy |
| Summary Metrics | `experiments/*/results_summary.json` | Aggregated performance metrics |
| Detailed Logs | `experiments/*/logs/` | Full execution logs |
| Visualizations | `visualizations/` | Generated plots and charts |

### Loading Results in Python

```python
import pandas as pd
import json

# Load detection comparison data
df = pd.read_csv('experiments/monte_carlo_results/detection_comparison.csv')

# Analyze accuracy by scenario type
accuracy_by_type = df.groupby(['scenario_type', 'complexity'])['accuracy'].mean()
print(accuracy_by_type)

# Load summary metrics
with open('experiments/monte_carlo_results/results_summary.json') as f:
    summary = json.load(f)
    overall_accuracy = summary['overall_metrics']['detection_accuracy']
    print(f"Overall Detection Accuracy: {overall_accuracy:.3f}")
```

### Key Metrics

- **Detection Accuracy**: Conflict detection true positive rate
- **False Positive Rate**: Incorrect conflict alerts
- **False Negative Rate**: Missed actual conflicts
- **Safety Margins**: Separation distance analysis
- **Response Time**: LLM processing time per scenario
- **ICAO Compliance**: Standard adherence percentage

## Project Structure

```
LLM-ATC-HAL/
├── cli.py                                 # Main command-line interface
├── pyproject.toml                         # Project configuration
├── requirements.txt                       # Dependencies
├── *.yaml                                # Configuration files
│
├── scenarios/                            # Scenario generation & execution
│   ├── scenario_generator.py            # Environment-specific scenarios
│   ├── monte_carlo_framework.py         # BlueSky integration framework
│   └── monte_carlo_runner.py            # Main benchmark orchestration
│
├── llm_atc/                             # Core LLM-ATC components
│   ├── agents/                          # Multi-agent coordination
│   ├── tools/                           # BlueSky tools & prompt engines
│   ├── metrics/                         # Performance analysis
│   ├── baseline_models/                 # Traditional ATC methods
│   └── memory/                          # Experience replay system
│
├── llm_interface/                        # LLM client management
│   ├── llm_client.py                    # Ollama integration
│   ├── ensemble.py                      # Multi-model coordination
│   └── filter_sort.py                   # Response filtering
│
├── bluesky_sim/                         # BlueSky simulator integration
│   ├── scenarios.py                     # Scenario loading
│   └── simulation_runner.py             # Simulation execution
│
├── analysis/                            # Results analysis & visualization
│   ├── enhanced_hallucination_detection.py  # Hallucination detection
│   ├── metrics.py                       # Statistical analysis
│   └── visualisation.py                 # Plotting & dashboards
│
├── solver/                              # Conflict resolution algorithms
├── BSKY_GYM_LLM/                       # Fine-tuning pipeline
├── Bsky_gym_Trained/                    # Pre-trained RL models
├── experiments/                         # Test results & benchmarks
└── docs/                               # Documentation
```

### Core Modules

#### scenarios/
- **`scenario_generator.py`**: Environment-specific scenario creation (Horizontal/Vertical/Sector)
- **`monte_carlo_framework.py`**: BlueSky-integrated scenario generation with distribution shift
- **`monte_carlo_runner.py`**: Three-stage pipeline (detection → resolution → verification)

#### llm_atc/
- **`tools/llm_prompt_engine.py`**: Sophisticated ICAO-compliant prompt templates
- **`tools/bluesky_tools.py`**: BlueSky command generation and simulation interface
- **`metrics/monte_carlo_analysis.py`**: Comprehensive performance analysis
- **`agents/`**: Multi-agent LLM coordination (Planner, Executor, Verifier, Scratchpad)

#### analysis/
- **`enhanced_hallucination_detection.py`**: Multi-layer hallucination detection
- **`metrics.py`**: Statistical analysis and ICAO compliance checking
- **`visualisation.py`**: Performance dashboards and trend analysis

## Contributing

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting and formatting
ruff check .
black . --check

# Run type checking
mypy llm_atc/
```

### Code Style Guidelines

- **Python 3.9+** with PEP 8 compliance
- **Type hints** on all public functions
- **Docstrings** in Google/NumPy style
- **Absolute imports** rooted at project package
- **Module-level logging** with `logger = logging.getLogger(__name__)`

### Testing

Currently, the project focuses on integration testing through CLI commands. Unit test framework setup is planned for future releases.

```bash
# System validation
python cli.py demo --duration 60 --aircraft 2

# Quick integration test
python cli.py quick-test --quick
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow coding guidelines and add documentation
4. Test your changes with CLI validation
5. Submit a pull request

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Detection Accuracy | >85% AUROC | Benchmarking |
| Response Time | <2s mean | <1.5s |
| Safety Compliance | 95% ICAO standards | Monitoring |
| Horizontal Separation | 5.0 nm minimum | Validated |
| Vertical Separation | 1000 ft minimum | Validated |

## Citation

If you use LLM-ATC-HAL in your research, please cite:

```bibtex
@software{llm_atc_hal_2025,
  title={LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Hallucination Detection},
  author={Somnath, Abhishek},
  year={2025},
  url={https://github.com/Somnathab3/LLM-ATC-HAL},
  note={Advanced Large Language Model Air Traffic Controller with Monte Carlo Safety Benchmarking}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Links

- **Repository**: [GitHub](https://github.com/Somnathab3/LLM-ATC-HAL)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/Somnathab3/LLM-ATC-HAL/issues)
- **Documentation**: [Project Wiki](https://github.com/Somnathab3/LLM-ATC-HAL/wiki)
- **CI/CD**: [GitHub Actions](https://github.com/Somnathab3/LLM-ATC-HAL/actions)

**Status**: Active development - Production-ready alpha with comprehensive testing framework.
