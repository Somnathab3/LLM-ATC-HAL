# LLM-ATC-HAL

**Embodied LLM Air Traffic Controller with Safety Metrics and Hallucination Detection**

This repository provides a comprehensive framework for evaluating Large Language Model performance in safety-critical air traffic control scenarios with advanced hallucination detection and mitigation capabilities.

## Overview

LLM-ATC-HAL is a research framework designed to assess and improve the reliability of LLM-based decision making in air traffic control environments. The system integrates with the BlueSky air traffic control simulator to provide realistic scenario generation, conflict detection, and resolution testing with comprehensive safety metrics.

**Key Features:**
- **Sophisticated Prompt Engineering** with ICAO standards compliance
- **Mathematical Precision Validation** for conflict detection
- **Safety Margin Quantification** and risk assessment
- **Experience Replay** with ChromaDB vector storage
- **Monte Carlo Analysis** for statistical validation
- **Enhanced Hallucination Detection** algorithms
- **Multi-Model LLM Support** (OpenAI, Ollama, etc.)

## Quickstart

Get up and running with LLM-ATC-HAL in minutes:

```bash
# 1. Clone the repository
git clone https://github.com/Somnathab3/LLM-ATC-HAL.git
cd LLM-ATC-HAL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a demo scenario
python cli.py demo --aircraft 4 --duration 300

# 4. View help for all available commands
python cli.py --help
```

## Installation

### Prerequisites
- Python 3.9 or higher
- 8GB+ RAM recommended for optimal performance

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install BlueSky Simulator (Optional but Recommended)
The system can run with mock data, but BlueSky provides the most realistic simulation experience:

```bash
# Follow BlueSky installation instructions for your platform at:
# https://github.com/TUDelft-CNS-ATM/bluesky
```

### Step 3: Configure LLM Backend
The system supports multiple LLM providers. For Ollama (recommended for local development):

```bash
# Install Ollama
# Visit https://ollama.ai/ for installation instructions

# Pull recommended model
ollama pull llama3.1:8b
```

## Usage

### Basic Commands

```bash
# Run comprehensive help
python cli.py --help

# Quick demo scenario
python cli.py demo --aircraft 4 --duration 300

# Run hallucination detection tests
python cli.py hallucination-test --models "llama3.1:8b,mistral:7b" --scenarios 50

# Distribution shift analysis
python cli.py shift-benchmark --tier comprehensive --complexities "simple,moderate,complex"

# Analyze results with advanced metrics
python cli.py analyze --results-dir experiments/results --format comprehensive
```

### Advanced Usage

```bash
# Monte Carlo benchmark with custom parameters
python cli.py monte-carlo-benchmark --iterations 100 --complexities "moderate,complex" --model "llama3.1:8b"

# Run specific scenario files
python cli.py run-scenario path/to/scenario.json --output results/

# Batch scenario processing
python cli.py batch-scenarios --input-dir scenarios/ --output-dir results/
```

## Configuration

### Environment Variables
- `OLLAMA_HOST`: Ollama server host (default: localhost)
- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)

### Configuration Files
- `bluesky_config.yaml`: BlueSky simulator settings
- `scenario_ranges.yaml`: Scenario parameter ranges
- `distribution_shift_levels.yaml`: Distribution shift configurations

### Example Configuration
```yaml
# bluesky_config.yaml
bluesky:
  connection_type: local
  simulation:
    conflict_detection_method: SWARM
    separation_standards:
      horizontal_nm: 5.0
      vertical_ft: 1000.0
```

## Project Structure

```
llm_atc/           # Core simulation framework
├── agents/        # Decision-making agents (planner, executor, verifier, scratchpad)
├── tools/         # BlueSky integration, conflict detection, prompt engines
├── metrics/       # Performance measurement and safety quantification
└── memory/        # State tracking and experience replay

analysis/          # Post-simulation analysis and visualization
scenarios/         # Scenario generation and Monte Carlo framework
llm_interface/     # LLM client management and ensemble support
cli.py            # Main command-line interface
```

### Core Components

1. **`llm_atc/`** - Main ATC Simulation Framework
   - Entry point: `python cli.py`
   - Modular agent-based architecture

2. **`analysis/`** - Post-Simulation Analysis
   - Enhanced hallucination detection algorithms
   - Visualization tools and statistical reports

3. **`scenarios/`** - Scenario Generation & Monte Carlo
   - Automated scenario generation with configurable parameters
   - Statistical analysis framework

4. **`llm_interface/`** - LLM Client Management
   - Multi-provider support (OpenAI, Ollama, etc.)
   - Ensemble methods and response filtering

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with appropriate tests
4. Commit your changes (`git commit -am 'Add your feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Include unit tests for new functionality
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use LLM-ATC-HAL in your research, please cite:

```bibtex
@software{llm_atc_hal_2025,
  title={LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Safety Metrics},
  author={LLM-ATC-HAL Contributors},
  year={2025},
  url={https://github.com/Somnathab3/LLM-ATC-HAL}
}
```
