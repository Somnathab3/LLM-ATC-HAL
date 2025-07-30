# Scenario Generator Module

## Overview

The **Scenario Generator Module** (`scenarios/scenario_generator.py`) encapsulates environment-specific scenario creation logic for three types of Air Traffic Control conflict scenarios:

- **HorizontalCREnv**: Same-altitude conflict scenarios
- **VerticalCREnv**: Altitude-based conflict scenarios  
- **SectorCREnv**: Full-sector realistic scenarios

This module wraps the existing Monte Carlo framework (`scenarios/monte_carlo_framework.py`) and provides targeted scenario generation with precise ground truth conflict labeling for testing false positive/negative rates in conflict detection systems.

## Key Features

✅ **Environment-Specific Generation**: Three specialized environments for different conflict types
✅ **Ground Truth Labeling**: Precise conflict information for validation
✅ **BlueSky Integration**: Generates realistic BlueSky commands
✅ **Distribution Shift Support**: Test scenarios across varying operational conditions
✅ **Backward Compatibility**: Integrates seamlessly with existing codebase
✅ **Comprehensive Testing**: Full test suite validates functionality

## Installation

The module is part of the LLM-ATC-HAL framework. Ensure you have the required dependencies:

```bash
# Install required packages (if not already installed)
pip install numpy pyyaml

# No additional installation needed - module is ready to use
```

## Quick Start

### Basic Usage

```python
from scenarios.scenario_generator import (
    HorizontalCREnv, 
    VerticalCREnv, 
    SectorCREnv,
    ComplexityTier
)

# Horizontal conflict scenario (same altitude)
h_env = HorizontalCREnv()
h_scenario = h_env.generate_scenario(n_aircraft=3, conflict=True)

# Vertical conflict scenario (climb/descent conflicts)
v_env = VerticalCREnv()
v_scenario = v_env.generate_scenario(n_aircraft=2, conflict=True)

# Sector scenario (realistic multi-aircraft)
s_env = SectorCREnv()
s_scenario = s_env.generate_scenario(
    complexity=ComplexityTier.MODERATE,
    force_conflicts=False
)
```

### Convenience Functions

```python
from scenarios.scenario_generator import (
    generate_horizontal_scenario,
    generate_vertical_scenario,
    generate_sector_scenario
)

# Generate specific scenario types directly
horizontal = generate_horizontal_scenario(n_aircraft=2, conflict=True)
vertical = generate_vertical_scenario(n_aircraft=2, conflict=True)
sector = generate_sector_scenario(complexity=ComplexityTier.COMPLEX)
```

## Scenario Types

### 1. Horizontal Conflict Scenarios

**Purpose**: Test conflict detection when aircraft are at the same flight level.

**Key Features**:
- All aircraft at same altitude (eliminates vertical separation)
- Configurable convergent/divergent headings
- Ground truth horizontal separation violations

```python
h_env = HorizontalCREnv()

# Generate conflict scenario
conflict_scenario = h_env.generate_scenario(n_aircraft=3, conflict=True)
# → Aircraft will have convergent headings at FL350

# Generate safe scenario  
safe_scenario = h_env.generate_scenario(n_aircraft=3, conflict=False)
# → Aircraft will have divergent headings to avoid conflicts
```

**Verification**:
- All aircraft have identical altitudes
- Conflict scenarios have convergent headings
- Safe scenarios have divergent headings

### 2. Vertical Conflict Scenarios

**Purpose**: Test conflict detection for climb/descent conflicts.

**Key Features**:
- Aircraft at different initial altitudes
- Climb/descent commands create vertical conflicts
- Ground truth vertical separation violations

```python
v_env = VerticalCREnv()

# Generate vertical conflict scenario
scenario = v_env.generate_scenario(n_aircraft=2, conflict=True)
# → One aircraft climbs while another descends

# Check for vertical maneuvers
alt_commands = [cmd for cmd in scenario.commands if 'ALT' in cmd or 'VS' in cmd]
```

**Verification**:
- Aircraft have different initial altitudes
- Contains altitude (ALT) or vertical speed (VS) commands
- Ground truth includes vertical conflicts

### 3. Sector Scenarios

**Purpose**: Test realistic multi-aircraft sector operations.

**Key Features**:
- Uses full Monte Carlo scenario generation
- Configurable complexity levels (SIMPLE, MODERATE, COMPLEX, EXTREME)
- Distribution shift support
- Realistic environmental conditions

```python
s_env = SectorCREnv()

# Generate complex sector scenario
scenario = s_env.generate_scenario(
    complexity=ComplexityTier.COMPLEX,
    shift_level="moderate_shift",
    force_conflicts=True
)
```

**Verification**:
- Aircraft count matches complexity tier
- Environmental conditions vary realistically
- Distribution shift applied correctly

## Ground Truth Conflict Data

Each scenario provides detailed ground truth information:

```python
scenario = generate_horizontal_scenario(n_aircraft=2, conflict=True)

for conflict in scenario.ground_truth_conflicts:
    print(f"Aircraft pair: {conflict.aircraft_pair}")
    print(f"Conflict type: {conflict.conflict_type}")
    print(f"Time to conflict: {conflict.time_to_conflict} seconds")
    print(f"Min separation: {conflict.min_separation}")
    print(f"Severity: {conflict.severity}")
    print(f"Actual violation: {conflict.is_actual_conflict}")
```

### Ground Truth Fields

- **aircraft_pair**: Tuple of aircraft callsigns in conflict
- **conflict_type**: 'horizontal', 'vertical', 'convergent', 'parallel', 'overtaking'
- **time_to_conflict**: Estimated seconds until closest approach
- **min_separation**: Dict with 'horizontal_nm' and 'vertical_ft' values
- **severity**: 'low', 'medium', 'high', 'critical'
- **is_actual_conflict**: Boolean - true if separation standards violated

## Distribution Shift Testing

Test robustness across varying operational conditions:

```python
# Test across distribution shift levels
shift_levels = ["in_distribution", "moderate_shift", "extreme_shift"]

for shift_level in shift_levels:
    scenario = generate_sector_scenario(
        complexity=ComplexityTier.MODERATE,
        shift_level=shift_level,
        force_conflicts=True
    )
    print(f"Shift: {shift_level}, Aircraft: {scenario.aircraft_count}")
```

**Distribution Shift Effects**:
- **in_distribution**: Baseline operational conditions
- **moderate_shift**: +30% traffic density, wind variations, different aircraft types
- **extreme_shift**: +70% traffic density, severe weather, mixed aircraft performance

## Integration with Existing Code

### Scenario Data Structure

Scenarios provide all fields expected by existing LLM-ATC-HAL components:

```python
scenario = generate_horizontal_scenario(n_aircraft=2)

# Convert to dict for compatibility
scenario_dict = scenario.to_dict()

# Access standard fields
print(f"Scenario ID: {scenario.scenario_id}")
print(f"Aircraft count: {scenario.aircraft_count}")
print(f"BlueSky commands: {scenario.commands}")
print(f"Aircraft states: {scenario.initial_states}")
print(f"Environmental conditions: {scenario.environmental_conditions}")
```

### Aircraft State Format

Each aircraft state contains:

```python
{
    'callsign': 'AC001',
    'aircraft_type': 'B737',
    'latitude': 52.3,
    'longitude': 4.8,
    'altitude': 35000,
    'heading': 90,
    'ground_speed': 350,
    'vertical_rate': 0
}
```

## Testing and Validation

### Running Tests

```bash
# Run comprehensive test suite
python tests/test_scenario_generator_simple.py

# Run basic functionality test
python -c "
from scenarios.scenario_generator import ScenarioGenerator
generator = ScenarioGenerator()
scenario = generator.generate_horizontal_scenario(n_aircraft=2)
print(f'Generated scenario with {scenario.aircraft_count} aircraft')
"
```

### Test Coverage

- ✅ Horizontal scenarios have same altitudes
- ✅ Vertical scenarios have different altitudes
- ✅ Sector scenarios respect complexity tiers
- ✅ Ground truth conflicts properly generated
- ✅ Environment classes work correctly
- ✅ Distribution shift functionality
- ✅ Data structure compatibility

### Running Demonstration

```bash
# See all features in action
python scenarios/demo_scenario_generator.py
```

## Configuration

### Scenario Ranges

Edit `scenario_ranges.yaml` to customize:

```yaml
aircraft:
  count:
    simple: [2, 3]
    moderate: [4, 6]
    complex: [8, 12]

weather:
  wind:
    speed_kts: [0, 80]
    direction_deg: [0, 360]
```

### Distribution Shift

Edit `distribution_shift_levels.yaml` to customize shift behavior:

```yaml
moderate_shift:
  traffic_density_multiplier: 1.3
  weather:
    wind:
      speed_shift_kts: [-20, 20]
```

## API Reference

### ScenarioGenerator

Main generator class with methods for all scenario types.

```python
generator = ScenarioGenerator()

# Generate by type
scenario = generator.generate_scenario(ScenarioType.HORIZONTAL, **kwargs)

# Type-specific methods
horizontal = generator.generate_horizontal_scenario(n_aircraft=2, conflict=True)
vertical = generator.generate_vertical_scenario(n_aircraft=2, conflict=True)
sector = generator.generate_sector_scenario(complexity=ComplexityTier.MODERATE)
```

### Environment Classes

Specialized classes for each environment type.

```python
# Horizontal conflicts
h_env = HorizontalCREnv()
h_scenario = h_env.generate_scenario(n_aircraft=3, conflict=True)

# Vertical conflicts
v_env = VerticalCREnv()
v_scenario = v_env.generate_scenario(n_aircraft=2, conflict=True)

# Sector scenarios
s_env = SectorCREnv()
s_scenario = s_env.generate_scenario(complexity=ComplexityTier.COMPLEX)
```

### Data Classes

#### Scenario

Main scenario data structure:

```python
@dataclass
class Scenario:
    scenario_id: str
    scenario_type: ScenarioType
    aircraft_count: int
    commands: List[str]  # BlueSky commands
    initial_states: List[Dict[str, Any]]  # Aircraft states
    ground_truth_conflicts: List[GroundTruthConflict]
    expected_conflict_count: int
    has_conflicts: bool
    complexity_tier: ComplexityTier
    generation_timestamp: float
    environmental_conditions: Dict[str, Any]
    airspace_region: str
```

#### GroundTruthConflict

Conflict information for validation:

```python
@dataclass  
class GroundTruthConflict:
    aircraft_pair: Tuple[str, str]
    conflict_type: str
    time_to_conflict: float
    min_separation: Dict[str, float]
    severity: str
    is_actual_conflict: bool
```

## Examples

### False Positive/Negative Testing

```python
from scenarios.scenario_generator import generate_horizontal_scenario

# Generate test cases for FP/FN analysis
test_cases = []

# True positives: scenarios with real conflicts
for i in range(10):
    scenario = generate_horizontal_scenario(n_aircraft=3, conflict=True)
    test_cases.append({
        'scenario': scenario,
        'expected_conflicts': len(scenario.ground_truth_conflicts),
        'label': 'positive'
    })

# True negatives: scenarios without conflicts  
for i in range(10):
    scenario = generate_horizontal_scenario(n_aircraft=3, conflict=False)
    test_cases.append({
        'scenario': scenario,
        'expected_conflicts': 0,
        'label': 'negative'
    })

# Test your conflict detection system
for test_case in test_cases:
    scenario = test_case['scenario']
    expected = test_case['expected_conflicts']
    
    # Run your detection system
    detected = your_conflict_detector(scenario.commands)
    
    # Calculate metrics
    if expected > 0 and detected > 0:
        result = "TP"  # True Positive
    elif expected == 0 and detected == 0:
        result = "TN"  # True Negative
    elif expected > 0 and detected == 0:
        result = "FN"  # False Negative
    else:
        result = "FP"  # False Positive
    
    print(f"Scenario {scenario.scenario_id}: {result}")
```

### Batch Generation

```python
from scenarios.scenario_generator import ScenarioGenerator, ComplexityTier

generator = ScenarioGenerator()

# Generate test suite
test_scenarios = []

# Horizontal conflicts
for n_aircraft in [2, 3, 4]:
    for conflict in [True, False]:
        scenario = generator.generate_horizontal_scenario(
            n_aircraft=n_aircraft,
            conflict=conflict
        )
        test_scenarios.append(scenario)

# Vertical conflicts  
for n_aircraft in [2, 3]:
    scenario = generator.generate_vertical_scenario(
        n_aircraft=n_aircraft,
        conflict=True
    )
    test_scenarios.append(scenario)

# Sector scenarios across complexity
for complexity in [ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX]:
    scenario = generator.generate_sector_scenario(complexity=complexity)
    test_scenarios.append(scenario)

print(f"Generated {len(test_scenarios)} test scenarios")
```

## Troubleshooting

### Common Issues

**Import Error**: Ensure you're running from the project root directory.

```bash
cd f:\LLM-ATC-HAL
python your_script.py
```

**Range Errors**: Check `scenario_ranges.yaml` and `distribution_shift_levels.yaml` for invalid ranges.

**No Conflicts Generated**: Increase `force_conflicts=True` or check aircraft positioning.

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

from scenarios.scenario_generator import ScenarioGenerator
generator = ScenarioGenerator()
# Will show detailed generation process
```

## Contributing

To extend the scenario generator:

1. **Add New Scenario Types**: Extend `ScenarioType` enum and add generation methods
2. **Modify Conflict Detection**: Update `_calculate_*_ground_truth` methods
3. **Add New Environments**: Create new environment classes following the pattern
4. **Extend Distribution Shifts**: Add new shift configurations in YAML files

### Code Style

- Follow existing naming conventions
- Add comprehensive docstrings
- Include type hints
- Write unit tests for new functionality

## License

Part of the LLM-ATC-HAL framework. See main project license.
