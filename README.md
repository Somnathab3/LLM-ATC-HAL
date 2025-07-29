# LLM-ATC-HAL: Embodied LLM Air‑Traffic Controller with Safety Metrics

![CI](https://github.com/Somnathab3/LLM-ATC-HAL/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-XX%25-green)

## ⚡ Quick Start

```bash
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL
pip install -e ".[gpu]"     # or just pip install -e .
llm-atc demo                # run a toy scenario
```

## System Architecture

```
llm_atc/
  agents/              # Embodied agent system (planner, executor, verifier)
  baseline_models/     # Traditional ATC conflict detection/resolution
  memory/              # Experience replay with vector storage
  metrics/             # Safety metrics and hallucination detection
  tools/               # BlueSky integration and utility functions
  experiments/         # Distribution shift testing framework
  data/                # Training and validation datasets
  cli.py               # Command-line interface
tests/                 # Comprehensive test suite
docs/                  # Documentation and notebooks
```

## Performance Metrics



## Features

- **🤖 Embodied LLM Agents**: Multi-agent system with planner, executor, and verifier
- **🛡️ Safety Metrics**: Real-time ICAO-compliant safety margin calculation
- **🧠 Memory System**: Vector-based experience replay for continuous learning
- **🔍 Hallucination Detection**: Multi-layer detection and mitigation framework
- **📊 Distribution Shift Testing**: Robustness evaluation across operational conditions
- **🎮 BlueSky Integration**: Realistic air traffic simulation environment

## Installation

### Prerequisites
- Python 3.9+
- Git LFS (for model files)
- Optional: NVIDIA GPU with CUDA support

### Step 1: Clone and Install
```bash
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL
pip install -e ".[gpu]"  # Full installation with GPU support
# OR
pip install -e .         # CPU-only installation
```

### Step 2: Verify Installation
```bash
llm-atc validate        # Check dependencies and system health
```

## Usage

### Run Demo Scenario
```bash
llm-atc demo --aircraft 6 --duration 600
```

### Execute Custom Scenario
```bash
llm-atc run-scenario scenarios/complex_weather.yaml
```

### Distribution Shift Benchmark
```bash
llm-atc shift-benchmark --tiers all --n 50
```

### Analyze Results
```bash
llm-atc analyze --results-dir test_results/
```

## CLI Commands

- `llm-atc demo` - Run a minimal demonstration scenario
- `llm-atc run-scenario <path>` - Execute a specific scenario file  
- `llm-atc shift-benchmark` - Run distribution shift robustness tests
- `llm-atc hallucination-test` - Test hallucination detection capabilities
- `llm-atc analyze` - Analyze test results and generate metrics
- `llm-atc validate` - Validate system installation and dependencies

## Configuration

The system uses YAML configuration files in `llm_atc/experiments/`:

```yaml
# Example: shift_experiment_config.yaml
experiment:
  name: "distribution_shift_study"
  tiers: ["in_distribution", "out_distribution", "adversarial"]
  scenarios_per_tier: 100
  
models:
  primary: "llama3.1:8b"
  ensemble: ["mistral:7b", "codellama:13b"]
  
safety:
  min_separation: 5.0  # nautical miles
  warning_threshold: 8.0
```

## Development

### Running Tests
```bash
pytest                          # All tests
pytest tests/test_agents.py     # Specific module
pytest -m "not slow"            # Skip slow tests
```

### Code Quality
```bash
ruff check llm_atc              # Linting
ruff format llm_atc             # Code formatting
mypy llm_atc                    # Type checking
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Research Applications

This framework supports research in:

- **Hallucination Detection**: Multi-modal detection mechanisms
- **Safety-Critical AI**: Real-time safety constraint validation  
- **Human-AI Collaboration**: Interactive decision support systems
- **Embodied AI**: Multi-agent reasoning and execution
- **Distribution Shift**: Robustness across operational conditions

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_atc_hal_2025,
  title={LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Safety Metrics},
  author={LLM-ATC-HAL Research Team},
  year={2025},
  url={https://github.com/Somnathab3/LLM-ATC-HAL},
  version={0.1.0}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/Somnathab3/LLM-ATC-HAL/issues)
- 💬 [Discussions](https://github.com/Somnathab3/LLM-ATC-HAL/discussions)

---

**Research Contact**: For academic collaboration and research inquiries, please open a discussion or issue.
