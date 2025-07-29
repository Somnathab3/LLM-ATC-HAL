# Distribution Shift Experiments

This module provides systematic testing of the LLM-ATC-HAL framework across distribution shift tiers to evaluate robustness and performance degradation under varying operational conditions.

## Overview

The distribution shift experiment runner loops over distribution shift tiers × N simulations, capturing comprehensive performance metrics including:

- **BlueSky command logs** and LLM outputs
- **Hallucination detection** results across all layers
- **Safety metrics** including ICAO compliance
- **Performance metrics** including response times
- **Environmental impact** analysis
- **Automated visualization generation** including conflict detection timelines, system flowcharts, and tier comparisons

Results are stored in parquet format for statistical analysis and thesis research. The system automatically generates research-grade visualizations saved to `thesis_results/` directory.

## Usage

### Quick Start

```python
from experiments.distribution_shift_runner import run_distribution_shift_experiment

# Run experiment with default settings (100 sims per tier)
results_file = run_distribution_shift_experiment()

# Run smaller test
results_file = run_distribution_shift_experiment(n_sims_per_tier=10)
```

### Advanced Configuration

```python
from experiments.distribution_shift_runner import DistributionShiftRunner

# Custom configuration
runner = DistributionShiftRunner(
    config_file="custom_config.yaml",
    output_dir="custom_results"
)

results_file = runner.run_experiment()
```

### Analyzing Results

```python
import pandas as pd
from analysis.metrics import aggregate_thesis_metrics

# Load results
df = pd.read_parquet(results_file)

# Aggregate metrics for thesis analysis
metrics = aggregate_thesis_metrics(df)

# Display key findings
print(f"Overall hallucination rate: {metrics['hallucination_analysis']['overall_detection_rate']:.3f}")
print(f"Safety performance by tier: {metrics['safety_performance']['safety_score_by_tier']}")
```

## File Structure

```
experiments/
├── distribution_shift_runner.py     # Main experiment runner
├── shift_experiment_config.yaml     # Configuration file
├── test_distribution_shift.py       # Test script
├── README.md                        # This file
└── results/                         # Output directory
    ├── distribution_shift_experiment_*.parquet  # Results data
    └── experiment_summary_*.json     # Summary statistics
```

## Configuration

Edit `shift_experiment_config.yaml` to customize:

### Experiment Parameters
```yaml
experiment:
  n_sims_per_tier: 100
  distribution_shift_tiers:
    - 'in_distribution'
    - 'moderate_shift' 
    - 'extreme_shift'
  complexity_distribution:
    simple: 0.2
    moderate: 0.4
    complex: 0.3
    extreme: 0.1
```

### Models and Detection
```yaml
models:
  primary: 'llama3.1:8b'
  validator: 'mistral:7b'
  technical: 'codellama:7b'

detection:
  enable_all_layers: true
  confidence_threshold: 0.7
```

### Output Options
```yaml
output:
  save_intermediate: true
  compress_parquet: true
  include_command_logs: true
```

## Output Data Schema

The parquet files contain the following columns:

### Experiment Metadata
- `tier`: Distribution shift tier ('in_distribution', 'moderate_shift', 'extreme_shift')
- `sim_id`: Simulation ID within tier
- `scenario_id`: Unique scenario identifier
- `complexity`: Scenario complexity level
- `aircraft_count`: Number of aircraft

### LLM Performance
- `hallucination_detected`: Boolean hallucination detection result
- `fp`: False positives in conflict detection
- `fn`: False negatives in conflict detection  
- `llm_confidence`: LLM confidence score
- `ensemble_consensus`: Ensemble consensus score

### Safety Metrics
- `horiz_margin_ft`: Horizontal safety margin (feet)
- `vert_margin_nm`: Vertical safety margin (nautical miles)
- `extra_nm`: Extra distance traveled due to resolution
- `n_interventions`: Number of controller interventions required
- `safety_score`: Overall safety score (0-1)
- `icao_compliant`: ICAO compliance status

### Performance Metrics
- `runtime_s`: Total scenario processing time
- `response_time_s`: LLM response time
- `detection_time_s`: Hallucination detection time

### Environmental Conditions
- `wind_speed_kts`: Wind speed
- `turbulence_intensity`: Turbulence intensity
- `visibility_nm`: Visibility
- `navigation_error_nm`: Navigation error magnitude

### Logs (JSON strings)
- `bluesky_commands`: BlueSky command log
- `llm_output`: LLM decision output
- `detection_evidence`: Hallucination detection evidence

## Analysis Functions

### `calc_fp_fn(pred_conflicts, gt_conflicts)`
Calculates false positives and false negatives in conflict detection with time tolerance.

```python
from analysis.metrics import calc_fp_fn

fp, fn = calc_fp_fn(
    pred_conflicts=[{'id1': 'AC001', 'id2': 'AC002', 'time': 120}],
    gt_conflicts=[{'id1': 'AC001', 'id2': 'AC002', 'time': 125}]
)
```

### `calc_path_extra(actual_traj, original_traj)`
Calculates extra distance traveled due to conflict resolution using great circle distance.

```python
from analysis.metrics import calc_path_extra

extra_nm = calc_path_extra(
    actual_traj=[{
        'aircraft_id': 'AC001',
        'path': [{'lat': 52.3, 'lon': 4.8, 'time': 0}, 
                {'lat': 52.4, 'lon': 4.9, 'time': 300}]
    }],
    original_traj=[{
        'aircraft_id': 'AC001', 
        'path': [{'lat': 52.3, 'lon': 4.8, 'time': 0},
                {'lat': 52.35, 'lon': 4.85, 'time': 300}]
    }]
)
```

### `aggregate_thesis_metrics(df)`
Comprehensive aggregation wrapper for thesis analysis.

```python
from analysis.metrics import aggregate_thesis_metrics

metrics = aggregate_thesis_metrics(df)

# Access aggregated results
hallucination_analysis = metrics['hallucination_analysis']
safety_performance = metrics['safety_performance']
statistical_tests = metrics['statistical_tests']
```

## Testing

Run the test script to verify functionality:

```bash
python experiments/test_distribution_shift.py
```

This will run a small experiment (5 simulations per tier) and verify all components work correctly.

## Integration with LLM-ATC-HAL

The experiment runner integrates with all major LLM-ATC-HAL components:

- **Scenario Generator**: Uses BlueSky-integrated Monte Carlo framework with distribution shifts
- **LLM Ensemble**: Tests all configured models (llama3.1:8b, mistral:7b, codellama:7b)
- **Hallucination Detection**: Applies 6-layer detection framework
- **Safety Quantification**: ICAO-compliant safety margin calculations
- **Experience Replay**: Stores results for learning and analysis

## Research Applications

This framework supports several research applications:

1. **Distribution Shift Robustness**: Quantify performance degradation under operational condition changes
2. **Hallucination Pattern Analysis**: Identify conditions that trigger different hallucination types
3. **Safety-Performance Tradeoffs**: Analyze relationships between safety margins and efficiency
4. **Environmental Impact Assessment**: Correlate weather conditions with system performance
5. **Model Comparison**: Compare ensemble models across distribution shifts

## Performance Considerations

- **Parallel Execution**: Can be enabled in config for faster runs
- **Memory Management**: Intermediate results saved periodically to prevent data loss
- **Resource Limits**: Configurable memory and CPU limits
- **Error Handling**: Continues execution on individual simulation failures

## Output Examples

### Experiment Summary
```json
{
  "experiment_info": {
    "total_simulations": 300,
    "tiers_tested": ["in_distribution", "moderate_shift", "extreme_shift"],
    "completion_time": "2025-07-29 14:30:00"
  },
  "performance_by_tier": {
    "in_distribution": {
      "hallucination_rate": 0.05,
      "safety_score": 0.92,
      "icao_compliance": 0.95
    },
    "extreme_shift": {
      "hallucination_rate": 0.18,
      "safety_score": 0.78,
      "icao_compliance": 0.72
    }
  }
}
```

## Automated Visualization Generation

The experiment runner now automatically generates research-grade visualizations during execution:

### Generated Visualizations

1. **Conflict Detection Timelines** (`cd_timeline_<sim_id>.png`)
   - Time-series plots showing aircraft separation evolution
   - CD trigger points and CR command execution markers
   - ICAO separation standards overlay
   - One random simulation per tier automatically selected

2. **Conflict Resolution Flowcharts** (`cr_flowchart_<tier>_<sim_id>.png`)
   - NetworkX-based system architecture diagrams
   - Complete LLM-ATC-HAL processing pipeline visualization
   - Generated once per tier for system documentation

3. **Tier Comparison Analysis** (`tier_comparison.png`)
   - Multi-panel performance comparison across distribution shift tiers
   - Safety score distributions, hallucination rates, runtime analysis
   - Statistical significance indicators between tiers

4. **Visualization Summary** (`visualization_summary.png`)
   - Overview of all generated plots for research presentations
   - Grid layout of key figures for thesis documentation

### Output Directory Structure

```
experiments/results/
├── distribution_shift_experiment_<timestamp>.parquet  # Raw data
├── experiment_summary_<timestamp>.json              # Aggregated metrics
└── thesis_results/                                  # Auto-generated visualizations
    ├── cd_timeline_<sim_id>.png                    # Timeline plots
    ├── cr_flowchart_<tier>_<sim_id>.png           # System flowcharts  
    ├── tier_comparison.png                         # Performance comparison
    ├── visualization_summary.png                   # Research overview
    └── README_visualisation.md                     # Interpretation guide
```

### Visualization Configuration

Control visualization generation in `shift_experiment_config.yaml`:

```yaml
visualization:
  enabled: true
  output_dir: "thesis_results"
  formats: ["png"]
  dpi: 300
  generate_timelines: true
  generate_flowcharts: true
  generate_comparisons: true
  save_summary: true
```

### Statistical Results
The aggregation function provides comprehensive statistical analysis including:
- ANOVA tests for tier significance
- Correlation matrices for key metrics
- Effect size calculations (Cohen's d)
- Environmental factor correlations

This enables rigorous scientific analysis of distribution shift impacts on LLM-based ATC systems.
