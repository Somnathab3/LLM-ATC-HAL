# LLM-ATC-HAL

**Embodied LLM Air Traffic Controller with Safety Metrics and Hallucination Detection**

This repository provides a comprehensive framework for LLM-based air traffic control simulation and analysis, featuring sophisticated prompt engineering, safety metrics, and hallucination detection capabilities.

## Core Components

### 1. **`llm_atc/`** - Main ATC Simulation Framework
   - **Entry point:** `python cli.py`
   - **Subpackages:**
     - **agents/** – decision-making agents (planner, executor, verifier, scratchpad)
     - **tools/** – BlueSky integration, conflict detection, prompt engines
     - **metrics/** – performance measurement and safety quantification
     - **memory/** – state tracking and experience replay

### 2. **`analysis/`** - Post-Simulation Analysis
   - Enhanced hallucination detection algorithms
   - Visualization tools and statistical reports
   - Performance metrics and safety analysis

### 3. **`scenarios/`** - Scenario Generation & Monte Carlo
   - Automated scenario generation with configurable parameters
   - Monte Carlo simulation framework for statistical analysis
   - Batch processing capabilities

### 4. **`llm_interface/`** - LLM Client Management
   - Multi-provider LLM client (OpenAI, Ollama, etc.)
   - Ensemble support and filtering mechanisms
   - Prompt templating and response processing

### 5. **Utilities**
   - `analyze_dependencies.py` – dependency analysis tool
   - `bluesky_config.yaml` – BlueSky simulator configuration
   - Various analysis and maintenance scripts

## Installation

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install BlueSky simulator** (required for simulation)
   ```bash
   # Follow BlueSky installation instructions for your platform
   ```

## Usage

### Quick Start
```bash
# Run a demo scenario
python cli.py demo --aircraft 4 --duration 300

# Get comprehensive help
python cli.py --help

# Run hallucination detection tests
python cli.py hallucination-test --models "llama3.1:8b,mistral:7b" --scenarios 50

# Analyze results
python cli.py analyze --results-dir experiments/results
```

### Available Commands
- `demo` – Run minimal demo scenarios
- `run-scenario` – Execute specific scenario files
- `hallucination-test` – Run hallucination detection benchmarks
- `shift-benchmark` – Test distribution shift scenarios
- `analyze` – Comprehensive result analysis with sophisticated metrics

## Key Features

- **Sophisticated Prompt Engineering** with ICAO standards compliance
- **Mathematical Precision Validation** for conflict detection
- **Safety Margin Quantification** and risk assessment
- **Experience Replay** with ChromaDB integration
- **Monte Carlo Analysis** for statistical validation
- **Enhanced Hallucination Detection** algorithms
- **Multi-Model LLM Support** (OpenAI, Ollama, etc.)

## Project Structure
```
llm_atc/           # Core simulation framework
├── agents/        # Decision-making agents
├── tools/         # BlueSky integration & utilities
├── metrics/       # Performance & safety metrics
└── memory/        # State tracking & replay

analysis/          # Post-simulation analysis
scenarios/         # Scenario generation & Monte Carlo
llm_interface/     # LLM client management
cli.py            # Main command-line interface
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
