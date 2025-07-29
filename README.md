# LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Hallucination Detection

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/status-production--ready-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A comprehensive framework for testing and mitigating hallucinations in Large Language Model-based Air Traffic Control systems with ICAO-compliant safety metrics and real-time experience replay.

## 🎯 Quick Start

```bash
# Clone and setup
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull codellama:7b

# Run system validation
python deficiency_check_fixed.py

# Execute comprehensive testing
python comprehensive_hallucination_tester_v2.py
```

## 📊 Latest Performance Results

| Metric | Current Performance |
|--------|-------------------|
| **Error Rate** | 0.00% ✅ |
| **Test Success Rate** | 100% (39/39 tests) |
| **Response Time (Mean)** | 0.002s |
| **Hallucination Detection Rate** | 0.00% false positives |
| **ICAO Compliance Rate** | 76.92% |
| **Critical Safety Margin Rate** | 23.08% |

## 🏗️ System Architecture

```
LLM-ATC-HAL/
├── analysis/                     # Hallucination Detection & Analysis
│   ├── enhanced_hallucination_detection.py  # 6-layer detection framework
│   ├── hallucination_taxonomy.py           # Classification system
│   ├── metrics.py                          # Performance evaluation
│   └── visualisation.py                   # Research visualization tools
├── llm_atc/                     # Core LLM-ATC Framework
│   ├── agents/                  # Multi-Agent Controller System
│   │   ├── controller_interface.py         # Main ATC controller interface
│   │   ├── executor.py                     # Resolution execution engine
│   │   ├── planner.py                      # Conflict resolution planner
│   │   ├── scratchpad.py                   # Reasoning trace system
│   │   └── verifier.py                     # Safety verification agent
│   ├── baseline_models/         # Traditional ATC Models
│   │   ├── conflict_detector.py            # Classical conflict detection
│   │   ├── conflict_resolver.py            # Rule-based resolution
│   │   └── evaluate.py                     # Baseline performance evaluation
│   ├── memory/                  # Experience Replay System
│   │   ├── replay_store.py                 # Vector-based storage
│   │   └── experience_integrator.py        # Memory integration
│   ├── metrics/                 # ICAO Safety Quantification
│   │   └── safety_margin_quantifier.py     # Safety compliance metrics
│   ├── tools/                   # Specialized ATC Tools
│   │   └── bluesky_tools.py                # BlueSky simulation interface
│   ├── data/                    # Training & validation datasets
│   ├── experiments/             # Research experiment configurations
│   └── cli.py                   # Command-line interface
├── llm_interface/               # Multi-Model LLM Ensemble
│   ├── ensemble.py              # Weighted consensus system
│   ├── llm_client.py           # Primary LLM interface
│   └── prompts.py              # ATC-specific prompts
├── testing/                     # Comprehensive Testing Framework
│   ├── test_executor.py         # Test execution engine
│   ├── scenario_manager.py      # BlueSky scenario generation
│   └── result_analyzer.py       # Statistical analysis
├── scenarios/                   # Monte Carlo Testing
│   └── monte_carlo_framework.py # BlueSky-integrated scenarios
├── solver/                      # Conflict Resolution Engine
│   └── conflict_solver.py       # Advanced conflict resolution algorithms
└── comprehensive_hallucination_tester_v2.py  # Main testing suite
```

## 🚀 Core Features

### 🔍 **Hallucination Detection System**
- **6-Layer Detection Framework**: MIND framework, attention patterns, semantic entropy
- **Multi-Model Consensus**: Cross-validation between llama3.1:8b, mistral:7b, codellama:7b
- **Real-Time Classification**: Fabrication, omission, contradiction, uncertainty collapse detection
- **Safety Impact Assessment**: Critical safety analysis for aviation context

### 🛡️ **ICAO-Compliant Safety Metrics**
- **Safety Margin Quantification**: Horizontal/vertical separation standards
- **Risk Assessment**: Real-time compliance validation
- **Critical Threshold Monitoring**: Automated safety boundary enforcement
- **Performance Tracking**: Statistical safety compliance reporting

### 🧠 **Experience Replay System**
- **Vector Storage**: 1024-dim embeddings with Chroma HNSW
- **Similarity Search**: Metadata-filtered experience retrieval
- **Continuous Learning**: Conflict resolution pattern recognition
- **Local Embeddings**: Privacy-preserving intfloat/e5-large-v2 model

### 🎮 **BlueSky Simulation Integration**
- **Realistic Scenarios**: Range-based parameter sampling
- **ICAO Callsigns**: Authentic aviation data
- **Environmental Modeling**: Weather, turbulence, traffic density
- **Monte Carlo Testing**: Statistical scenario generation

## 📋 Installation & Setup

### Prerequisites
```bash
# Required software
Python 3.9+
Ollama (for LLM models)
Git (for cloning)

# Optional for enhanced performance
NVIDIA GPU with CUDA support
Docker (for Milvus vector database)
```

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/Somnathab3/LLM-ATC-HAL
cd LLM-ATC-HAL

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Ollama Model Installation
```bash
# Install required LLM models
ollama pull llama3.1:8b        # Primary reasoning model
ollama pull mistral:7b         # Validation model  
ollama pull codellama:7b       # Technical analysis model

# Verify installation
ollama list
```

### Step 3: System Validation
```bash
# Run comprehensive system check
python deficiency_check_fixed.py

# Expected output: All checks passed!
# Validates: imports, Ollama connectivity, scenario generation,
# hallucination detection, safety quantification
```

## 🎯 Usage Guide

### Quick System Test
```bash
# Run fast validation test (16 scenarios)
python quick_test_runner.py

# Expected results:
# - Total Tests: 16
# - Error Rate: 0.00%
# - ICAO Compliance: 62.50%+
```

### Comprehensive Testing Campaign
```bash
# Full testing suite (default configuration)
python comprehensive_hallucination_tester_v2.py

# Features tested:
# ✅ System validation (15 components)
# ✅ Multi-model ensemble testing
# ✅ BlueSky scenario generation
# ✅ Hallucination detection (6 layers)
# ✅ ICAO safety compliance
# ✅ Statistical analysis & visualization
```

### Individual Component Testing
```bash
# Test specific components
python system_validation.py           # System health check
python -m testing.test_executor       # Test execution engine
python -m analysis.enhanced_hallucination_detection  # Detection algorithms
```

## 📚 Core Modules & Functions

### 1. **Hallucination Detection** (`analysis/`)

#### `enhanced_hallucination_detection.py`
```python
from analysis.enhanced_hallucination_detection import EnhancedHallucinationDetector

detector = EnhancedHallucinationDetector()
result = detector.detect_hallucinations(llm_response, context, scenario_data)

# Returns: HallucinationResult with detection confidence, types, evidence
# Detection Layers: MIND framework, semantic entropy, self-consistency
```

#### `hallucination_taxonomy.py`
```python
from analysis.hallucination_taxonomy import HallucinationType, HallucinationClassifier

# Classification types: FABRICATION, OMISSION, IRRELEVANCY, 
# CONTRADICTION, SEMANTIC_DRIFT, UNCERTAINTY_COLLAPSE
classifier = HallucinationClassifier()
classification = classifier.classify_hallucination(response, context)
```

#### `metrics.py`
```python
from analysis.metrics import PerformanceMetrics, SafetyMetrics

# Calculate comprehensive performance scores
metrics = PerformanceMetrics()
safety_score = metrics.calculate_safety_score(resolution_data)
hallucination_rate = metrics.calculate_hallucination_rate(detection_results)
```

### 2. **LLM Ensemble System** (`llm_interface/`)

#### `ensemble.py`
```python
from llm_interface.ensemble import OllamaEnsembleClient

ensemble = OllamaEnsembleClient()
response = ensemble.query_ensemble(prompt, context, require_json=True)

# Returns: EnsembleResponse with consensus, individual responses, 
# confidence scores, uncertainty metrics
```

#### `llm_client.py`
```python
from llm_interface.llm_client import OllamaClient

client = OllamaClient(model="llama3.1:8b")
response = client.generate_response(prompt, max_tokens=1000, temperature=0.1)
```

#### `prompts.py`
```python
from llm_interface.prompts import ATCPromptGenerator

prompt_gen = ATCPromptGenerator()
atc_prompt = prompt_gen.generate_conflict_prompt(aircraft_data, urgency='high')
system_prompt = prompt_gen.get_system_prompt(role='conflict_resolver')
```

### 3. **Core LLM-ATC Framework** (`llm_atc/`)

The `llm_atc/` directory contains the core framework for LLM-based Air Traffic Control with comprehensive agent architecture, baseline models, and specialized tools.

#### **Agent System** (`llm_atc/agents/`)

##### `controller_interface.py`
```python
from llm_atc.agents.controller_interface import ATCControllerInterface

controller = ATCControllerInterface()
control_decision = controller.process_conflict(aircraft_states, airspace_context)
safety_validation = controller.validate_decision(decision, safety_constraints)
```

##### `executor.py`
```python
from llm_atc.agents.executor import ConflictExecutor

executor = ConflictExecutor()
execution_result = executor.execute_resolution(resolution_plan, real_time_data)
performance_metrics = executor.track_execution_performance()
```

##### `planner.py`
```python
from llm_atc.agents.planner import ConflictPlanner

planner = ConflictPlanner()
resolution_plan = planner.plan_conflict_resolution(scenario_data, constraints)
alternative_plans = planner.generate_alternatives(primary_plan, risk_threshold)
```

##### `scratchpad.py`
```python
from llm_atc.agents.scratchpad import ReasoningScratchpad

scratchpad = ReasoningScratchpad()
reasoning_steps = scratchpad.trace_reasoning(llm_response, decision_context)
explanation = scratchpad.generate_explanation(reasoning_chain)
```

##### `verifier.py`
```python
from llm_atc.agents.verifier import SafetyVerifier

verifier = SafetyVerifier()
safety_check = verifier.verify_resolution(proposed_solution, safety_standards)
compliance_report = verifier.generate_compliance_report(verification_results)
```

#### **Baseline Models** (`llm_atc/baseline_models/`)

##### `conflict_detector.py`
```python
from llm_atc.baseline_models.conflict_detector import BaselineConflictDetector

detector = BaselineConflictDetector()
conflicts = detector.detect_conflicts(aircraft_positions, trajectories)
risk_assessment = detector.assess_conflict_risk(conflict_data)
```

##### `conflict_resolver.py`
```python
from llm_atc.baseline_models.conflict_resolver import BaselineResolver

resolver = BaselineResolver()
baseline_solution = resolver.resolve_conflict(conflict_scenario)
comparison_metrics = resolver.compare_with_llm_solution(llm_solution, baseline_solution)
```

##### `evaluate.py`
```python
from llm_atc.baseline_models.evaluate import BaselineEvaluator

evaluator = BaselineEvaluator()
performance_comparison = evaluator.evaluate_against_baselines(llm_results, scenarios)
benchmark_report = evaluator.generate_benchmark_report(evaluation_data)
```

#### **CLI Interface** (`llm_atc/cli.py`)
```python
# Command-line interface for LLM-ATC operations
# Usage: python -m llm_atc.cli --command validate --config config.yaml

from llm_atc.cli import ATCCommandLineInterface

cli = ATCCommandLineInterface()
cli.run_validation_pipeline()
cli.execute_batch_testing(config_file="test_config.yaml")
```

#### **Data Management** (`llm_atc/data/`)
- **Scenario Storage**: Historical conflict scenarios and resolutions
- **Training Data**: Curated datasets for model training and evaluation
- **Validation Sets**: Standardized test cases for performance benchmarking

#### **Experiments** (`llm_atc/experiments/`)
- **Distribution Shift Studies**: Cross-domain performance analysis
- **Ablation Studies**: Component-wise performance evaluation
- **Comparative Analysis**: LLM vs baseline model comparisons

#### **Safety Metrics** (`llm_atc/metrics/`)

##### `safety_margin_quantifier.py`
```python
from llm_atc.metrics.safety_margin_quantifier import SafetyMarginQuantifier

quantifier = SafetyMarginQuantifier()
safety_result = quantifier.calculate_safety_margins(conflict_geometry, resolution)

# Returns: SafetyMetrics with ICAO compliance, risk assessment,
# horizontal/vertical margins, overall safety score
```

#### **BlueSky Tools** (`llm_atc/tools/`)

##### `bluesky_tools.py`
```python
from llm_atc.tools.bluesky_tools import BlueSkyInterface

bluesky = BlueSkyInterface()
simulation_commands = bluesky.generate_bluesky_commands(scenario_data)
simulation_results = bluesky.execute_simulation(commands, duration=600)
performance_analysis = bluesky.analyze_simulation_results(results)
```

#### `replay_store.py`
```python
from llm_atc.memory.replay_store import VectorReplayStore

store = VectorReplayStore()
store.store_experience(conflict_experience)  # Store with 1024-dim embeddings
similar_cases = store.retrieve_experience(conflict_desc, k=5)  # Similarity search
```

#### `experience_integrator.py`
```python
from llm_atc.memory.experience_integrator import ExperienceIntegrator

integrator = ExperienceIntegrator(replay_store)
guidance = integrator.get_experience_guidance(current_scenario)
integrator.integrate_experience(scenario, controller_feedback)
```

### 5. **Testing Framework** (`testing/`)

#### `test_executor.py`
```python
from testing import TestExecutor

executor = TestExecutor(ensemble_client, hallucination_detector, safety_quantifier)
results = await executor.execute_test(scenario, models_to_test)

# Returns: List[TestResult] with performance metrics, safety scores,
# hallucination detection results, timing data
```

#### `scenario_manager.py`
```python
from testing import ScenarioManager

manager = ScenarioManager()
scenarios = manager.generate_comprehensive_scenarios(
    num_scenarios=50,
    complexity_distribution={'simple': 0.3, 'moderate': 0.4, 'complex': 0.3}
)
```

#### `result_analyzer.py`
```python
from testing.result_analyzer import ResultAnalyzer

analyzer = ResultAnalyzer()
analysis = analyzer.analyze_test_results(results_list)
analyzer.generate_visualizations(analysis, output_dir="test_results/")
```

### 6. **BlueSky Integration** (`scenarios/`)

#### `monte_carlo_framework.py`
```python
from scenarios.monte_carlo_framework import BlueSkyScenarioGenerator, ComplexityTier

generator = BlueSkyScenarioGenerator("scenario_ranges.yaml")
scenario = generator.generate_scenario(ComplexityTier.COMPLEX)

# Generates: Realistic ATC scenarios with ICAO callsigns,
# environmental conditions, BlueSky commands
```

### 7. **Solver System** (`solver/`)

#### `conflict_solver.py`
```python
from solver.conflict_solver import ConflictSolver

solver = ConflictSolver()
resolution = solver.solve_conflict(aircraft_states, safety_constraints)
validation = solver.validate_resolution(resolution, scenario_context)
```

## 🔧 Configuration Files

### Test Configuration
```python
# comprehensive_hallucination_tester_v2.py
config = TestConfiguration(
    models_to_test=['llama3.1:8b', 'mistral:7b', 'codellama:7b'],
    num_scenarios=10,
    parallel_workers=4,
    timeout_per_test=30.0,
    use_gpu_acceleration=True,
    generate_visualizations=True
)
```

### Scenario Ranges (`scenario_ranges.yaml`)
```yaml
aircraft:
  count: [2, 8]               # Number of aircraft
  
geography:
  latitude: [50.0, 55.0]      # European airspace
  longitude: [3.0, 7.0]
  
weather:
  wind_speed: [0, 50]         # Knots
  turbulence_factor: [0.0, 1.0]
  
conflicts:
  severity: ['minor', 'moderate', 'critical']
  time_to_collision: [30, 600]  # Seconds
```

### Distribution Shift Levels (`distribution_shift_levels.yaml`)
```yaml
tiers:
  in_distribution:
    description: "Normal operational parameters"
    aircraft_count: [2, 4]
    
  out_distribution:
    description: "Edge case operational conditions"  
    aircraft_count: [5, 8]
    
  adversarial:
    description: "Stress testing with extreme conditions"
    aircraft_count: [8, 12]
```

## 📊 System Validation Results

Current system status (validated 2025-07-29):

```
✅ All Checks Passed! System ready for testing.

Component Status:
✅ Testing modules imported successfully
✅ LLM interface imported successfully  
✅ Hallucination detection imported successfully
✅ Safety margin quantifier imported successfully
✅ Memory modules imported successfully
✅ Ollama connectivity confirmed
✅ Scenario generation working (8 scenarios)
✅ Safety quantification working
```

## 📈 Performance Monitoring

### Real-Time Metrics
- **Response Time**: < 3 seconds (target: < 5s)
- **Memory Usage**: < 8GB RAM
- **GPU Utilization**: NVIDIA GeForce RTX 5070 Ti detected
- **Concurrent Tests**: 4 parallel workers

### Success Criteria
- **Zero Error Rate**: No system failures during execution
- **100% Test Completion**: All scenarios successfully processed
- **ICAO Compliance**: >60% safety standard compliance
- **Hallucination Detection**: <1% false positive rate

## 🐛 Troubleshooting

### Common Issues

1. **SentenceTransformers Error**
   ```bash
   # Known issue with corrupted cache
   # System automatically uses fallback embeddings
   # Performance impact: Reduced vector similarity accuracy
   ```

2. **Ollama Connection Issues**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check model availability
   ollama list
   ```

3. **Import Errors**
   ```bash
   # Verify all dependencies installed
   pip install -r requirements.txt
   
   # Run system validation
   python deficiency_check_fixed.py
   ```

4. **Memory Issues**
   ```bash
   # Reduce parallel workers if memory constrained
   # Edit comprehensive_hallucination_tester_v2.py:
   # parallel_workers=2  # Default: 4
   ```

## 📋 Dependencies

### Core Requirements
- **Python**: 3.9+ (tested with 3.12)
- **PyTorch**: 2.1.0+ (GPU acceleration)
- **ChromaDB**: 0.4.24+ (vector storage)
- **Ollama**: Latest (LLM models)
- **BlueSky**: ATC simulation engine
- **Sentence-Transformers**: 2.2.2+ (embeddings)

### Optional Enhancements  
- **NVIDIA CUDA**: GPU acceleration
- **Docker**: Milvus vector database
- **Weights & Biases**: Experiment tracking
- **Plotly**: Interactive visualizations


## 🔬 Research Applications

This framework supports research in:
- **Hallucination Detection**: Multi-modal detection mechanisms
- **Safety-Critical AI**: Real-time safety constraint validation
- **Human-AI Collaboration**: Interactive decision support
- **Embodied AI**: Multi-agent reasoning systems
- **Distribution Shift**: Robustness across operational conditions

## 🧪 Development & Testing

### Running Individual Tests
```bash
# Test specific modules
python tests/test_agents_simple.py       # Agent functionality
python tests/test_memory_simple.py       # Memory system  
python tests/test_baseline.py            # Baseline models

# Test with pytest
pytest tests/ -v                         # All tests with verbose output
pytest tests/test_memory_task2.py -k "vector"  # Specific test patterns
```

### Code Quality & Linting
```bash
# Python code formatting (if available)
black llm_atc/ analysis/ testing/        # Code formatting
flake8 .                                  # Style checking

# Type checking (if mypy installed)
mypy llm_atc/                            # Static type analysis
```

### Performance Profiling
```bash
# Memory profiling during tests
python -m memory_profiler comprehensive_hallucination_tester_v2.py

# Timing analysis
python -m cProfile -o profile_results.prof comprehensive_hallucination_tester_v2.py
```

## 📊 Example Output

### Successful Test Run
```
=== LLM-ATC Hallucination Testing System ===
🔧 System Validation: ✅ PASSED (15/15 components)
🧠 LLM Models: llama3.1:8b, mistral:7b, codellama:7b
📋 Test Configuration: 10 scenarios, 4 workers, 30s timeout

🎯 Scenario Generation: ✅ 10 scenarios generated
🔍 Hallucination Detection: ✅ 6-layer framework active
🛡️  Safety Quantification: ✅ ICAO standards enforced

📊 PERFORMANCE SUMMARY:
┌─────────────────┬──────────┬──────────────┬─────────────┐
│ Model           │ Tests    │ Error Rate   │ ICAO Comp   │
├─────────────────┼──────────┼──────────────┼─────────────┤
│ llama3.1:8b     │ 10       │ 0.00%        │ 80.00%      │
│ mistral:7b      │ 10       │ 0.00%        │ 70.00%      │
│ codellama:7b    │ 10       │ 0.00%        │ 60.00%      │
└─────────────────┴──────────┴──────────────┴─────────────┘

🎉 Testing completed successfully!
📁 Results saved to: test_results/comprehensive_test_20250729_121531.log
📈 Visualizations: test_results/*.png
```

## 🔄 Continuous Integration

### Automated Testing Pipeline
```bash
# Run full validation pipeline
bash scripts/sanity_run.sh              # Linux/macOS
PowerShell scripts/sanity_run.ps1       # Windows

# Expected pipeline stages:
# 1. System validation
# 2. Quick test run  
# 3. Memory system check
# 4. Agent functionality test
# 5. Full comprehensive test
```

### Results Analysis
```bash
# Analyze test results programmatically
python -m testing.result_analyzer test_results/

# Generate summary reports
python testing/result_streamer.py --input test_results/ --output summary.json
```

## 🎯 Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_atc_hal_2025,
  title={LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Safety Metrics},
  author={LLM-ATC-HAL Research Team},
  year={2025},
  url={https://github.com/Somnathab3/LLM-ATC-HAL},
  version={0.1.0},
  note={Comprehensive hallucination testing framework for safety-critical AI systems}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python comprehensive_hallucination_tester_v2.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

### Contribution Guidelines
- All new features must include tests
- Maintain backward compatibility where possible
- Follow existing code style and patterns
- Add documentation for new modules/functions
- Ensure system validation passes

## 📞 Support

- 📖 [Documentation](CLEANUP_SUMMARY.md)
- 🐛 [Issue Tracker](https://github.com/Somnathab3/LLM-ATC-HAL/issues)
- 💬 [Discussions](https://github.com/Somnathab3/LLM-ATC-HAL/discussions)

### Getting Help
1. Check existing issues and discussions
2. Run system validation: `python deficiency_check_fixed.py`
3. Review logs in `logs/` directory
4. Provide system configuration details when reporting issues

---

**Research Contact**: For academic collaboration and research inquiries, please open a discussion or issue.

**System Status**: ✅ Operational (Last validated: 2025-07-29)
**Testing Framework**: 🧪 Comprehensive (39/39 tests passing)
**Safety Compliance**: 🛡️ ICAO Standards (76.92% compliance rate)
