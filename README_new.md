# LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Hallucination Detection

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/status-production--ready-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

> **Advanced Large Language Model Air Traffic Controller with Comprehensive Hallucination Detection & Clean Output**

A cutting-edge research platform that integrates Large Language Models (LLMs) into air traffic control systems with clean progress bars, comprehensive debug logging, and detailed hallucination analysis.

## 🎯 Overview

LLM-ATC-HAL provides a complete framework for testing LLM-based air traffic control with:

✅ **Clean progress bars** with tqdm showing "horizontal_simple_baseline: 0%" style tracking  
✅ **Comprehensive logging** - detailed debug info written to logs instead of console spam  
✅ **LLM interactions** captured in detail and kept in log files (not shown in output)  
✅ **Aircraft states comparison** - mathematical vs BlueSky vs LLM outputs logged  
✅ **Progress tracking** working with batch-by-batch updates  

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Install Ollama models
ollama pull llama3.1:8b
ollama pull mistral:7b

# Run system validation
python cli.py validate

# Execute short clean benchmark (recommended)
python cli.py monte-carlo-benchmark --enhanced-output --num-horizontal 5 --num-vertical 5 --num-sector 5
```

## 📊 Key Features

### Enhanced Output & Clean Interface
- **Clean Progress Bars**: Beautiful tqdm progress tracking with meaningful labels
- **Comprehensive Logging**: All debug details saved to files, clean console output
- **LLM Interaction Logs**: Complete capture of all LLM conversations and decisions
- **CSV Export**: Detection comparison data for analysis

### Advanced Hallucination Detection
- **Multi-layer Detection**: Comprehensive analysis across different detection methods
- **Ground Truth Comparison**: Mathematical vs BlueSky vs LLM conflict detection
- **Real-time Monitoring**: Live tracking of false positives and negatives
- **Confidence Scoring**: Detailed confidence analysis for each detection

### Safety-First Architecture
- **ICAO Compliance**: Full adherence to international aviation standards
- **BlueSky Integration**: Industry-standard flight simulation platform
- **Mathematical Validation**: Rigorous mathematical conflict resolution verification
- **Safety Margins**: Real-time monitoring of separation distances

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Ollama (for local LLM support)
- Git

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Ollama and models**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull codellama:7b
```

5. **Validate installation**
```bash
python cli.py validate
```

## 🎮 Usage Examples

### Quick Start - Short Clean Run (Recommended)
```bash
# Run a short, clean benchmark with enhanced output
python cli.py monte-carlo-benchmark \
    --enhanced-output \
    --num-horizontal 5 \
    --num-vertical 5 \
    --num-sector 5 \
    --complexities simple,moderate \
    --shift-levels in_distribution,moderate_shift

# Output includes:
# - Clean progress bars: "horizontal_simple_baseline: 0%"
# - Detailed logs in: experiments/monte_carlo_results/logs/
# - CSV analysis: experiments/monte_carlo_results/detection_comparison.csv
# - LLM interactions: experiments/monte_carlo_results/logs/llm_interactions.log
```

### Standard Benchmark Run
```bash
# Standard run with default parameters
python cli.py monte-carlo-benchmark --enhanced-output

# Medium-scale run
python cli.py monte-carlo-benchmark \
    --enhanced-output \
    --num-horizontal 25 \
    --num-vertical 25 \
    --num-sector 25
```

### Distribution Shift Analysis
```bash
# Test across different distribution shifts
python cli.py shift-benchmark \
    --tiers all \
    --n 20 \
    --output experiments/shift_analysis
```

### Manual Testing Interface
```bash
# Interactive testing with specific models
python manual_llama_test.py
```

### Analysis & Visualization
```bash
# Analyze results from previous runs
python cli.py analyze --results-dir experiments/monte_carlo_results

# Generate comprehensive metrics
python cli.py analyze --log-file path/to/specific/log.json
```

## 📁 Project Structure

```
LLM-ATC-HAL/
├── cli.py                           # Main command-line interface
├── enhanced_monte_carlo_runner.py   # Enhanced runner with clean output
├── scenarios/                       # Scenario generation and management
│   ├── monte_carlo_runner.py       # Standard Monte Carlo benchmark
│   ├── monte_carlo_framework.py    # Framework definitions
│   └── scenario_generator.py       # Scenario generation logic
├── llm_atc/                        # Core LLM-ATC framework
│   ├── agents/                     # LLM agents and controllers
│   ├── baseline_models/            # Baseline comparison models
│   ├── experiments/                # Experiment runners
│   ├── memory/                     # Memory and context management
│   ├── metrics/                    # Performance metrics
│   └── tools/                      # Utility tools and helpers
├── analysis/                       # Analysis and visualization
│   ├── enhanced_hallucination_detection.py
│   ├── metrics.py
│   └── visualisation.py
├── bluesky_sim/                    # BlueSky simulation integration
├── llm_interface/                  # LLM communication layer
├── solver/                         # Conflict resolution algorithms
├── data/                          # Data storage and scenarios
├── experiments/                   # Experiment results
├── logs/                         # System logs
├── notebooks/                    # Jupyter analysis notebooks
└── visualizations/              # Generated plots and charts
```

## 🔧 Configuration

### Key Configuration Files
- `comprehensive_test_config.yaml` - Main test configuration
- `bluesky_config.yaml` - BlueSky simulation settings
- `distribution_shift_levels.yaml` - Distribution shift parameters
- `scenario_ranges.yaml` - Scenario generation ranges

### Environment Variables
```bash
# Optional: Enable verbose logging
export VERBOSE_LOGGING=1

# Optional: Set custom models
export LLM_MODELS="llama3.1:8b,mistral:7b"
```

## 📊 Output & Results

### Enhanced Output Mode
When using `--enhanced-output`, you get:

1. **Clean Progress Bars**
   ```
   horizontal_simple_baseline: 0%|          | 0/25 [00:00<?, ?scenario/s]
   ```

2. **Structured Logging**
   - `logs/benchmark.log` - Main execution log
   - `logs/llm_interactions.log` - All LLM conversations
   - `logs/debug.log` - Detailed debug information

3. **Analysis Files**
   - `detection_comparison.csv` - Ground truth vs LLM detection comparison
   - `results_summary.json` - Comprehensive metrics summary
   - `visualizations/` - Generated charts and plots

### Key Metrics Tracked
- **Detection Accuracy**: True positive/negative rates
- **Hallucination Rates**: By scenario type and complexity
- **Safety Margins**: Minimum separation distances
- **Response Times**: LLM inference and processing times
- **ICAO Compliance**: Adherence to aviation standards

## 🧪 Testing & Validation

### Quick Validation
```bash
# System check
python cli.py validate

# Short test run (5 minutes)
python cli.py monte-carlo-benchmark --enhanced-output --num-horizontal 3 --num-vertical 3 --num-sector 3
```

### Comprehensive Testing
```bash
# Full benchmark suite (30+ minutes)
python cli.py monte-carlo-benchmark --enhanced-output

# With automatic analysis
python cli.py monte-carlo-benchmark --enhanced-output --auto-analyze
```

## 📈 Research & Analysis

### Jupyter Notebooks
```bash
# Start Jupyter for analysis
cd notebooks
jupyter notebook monte_carlo_visualization_demo.ipynb
```

### Key Research Areas
- **Hallucination Pattern Analysis**: Understanding when and why LLMs hallucinate
- **Safety Margin Optimization**: Balancing efficiency with safety
- **Distribution Shift Impact**: How changes affect model performance
- **Multi-Model Comparison**: Evaluating different LLM architectures

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black llm_atc/ analysis/ scenarios/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links & References

- **BlueSky Flight Simulator**: [https://github.com/TUDelft-CNS-ATM/bluesky](https://github.com/TUDelft-CNS-ATM/bluesky)
- **Ollama**: [https://ollama.ai](https://ollama.ai)
- **ICAO Standards**: [https://www.icao.int](https://www.icao.int)

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: [Your Contact Information]
- Documentation: [Link to detailed docs]

---

**⚡ Ready to get started?** Run `python cli.py validate` to check your setup!
