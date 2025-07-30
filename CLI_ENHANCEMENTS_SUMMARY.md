## CLI Configuration Enhancements Summary

### Overview
Successfully implemented CLI configuration improvements for the Monte Carlo benchmark runner, addressing inconsistencies and adding new configuration options.

### 1. Separate Scenario Counts ✅

**Problem**: Previous CLI used a single `num_scenarios_per_type` for all scenario types.

**Solution**: Added distinct CLI options for each scenario type:
```bash
--num-horizontal  # Number of horizontal scenarios (default: 50)
--num-vertical    # Number of vertical scenarios (default: 50) 
--num-sector      # Number of sector scenarios (default: 50)
```

**Implementation**:
- Modified CLI command signature to accept three distinct counts
- Created `scenario_counts` dictionary mapping scenario types to counts
- Updated `BenchmarkConfiguration` to use per-type counts
- Modified `_calculate_total_scenarios()` and `_run_scenario_batch()` to read from dictionary

### 2. Explicit Complexity Tier Validation ✅

**Problem**: Previous implementation silently skipped invalid complexity tiers with warnings.

**Solution**: Added explicit validation with clear error messages:
```python
# Before mapping strings to ComplexityTier enums, check each provided value
invalid_complexities = []
for comp_str in complexity_strings:
    if comp_str in complexity_mapping:
        complexity_tiers.append(complexity_mapping[comp_str])
    else:
        invalid_complexities.append(comp_str)

# Validate complexity tiers explicitly
if invalid_complexities:
    valid_options = list(complexity_mapping.keys())
    raise click.BadParameter(
        f"Invalid complexity tier(s): {', '.join(invalid_complexities)}. "
        f"Valid options are: {', '.join(valid_options)}"
    )
```

**Benefits**:
- Immediate feedback on configuration errors
- Lists valid options for user reference
- Prevents silent configuration issues

### 3. Exposed Simulation Parameters ✅

**Problem**: Key simulation parameters were hardcoded and not configurable via CLI.

**Solution**: Added new CLI options:
```bash
--max-interventions  # Maximum interventions per scenario (default: 5)
--step-size         # Simulation step size in seconds (default: 10.0)
```

**Features**:
- Adaptive step size calculation based on time horizon:
  - < 5 minutes: min(step_size, 5.0) seconds
  - > 20 minutes: max(step_size, 15.0) seconds  
  - Otherwise: use provided step_size
- Configurable maximum interventions per scenario
- Propagated to `BenchmarkConfiguration`

### 4. Enhanced Error Handling ✅

**Improvements**:
- Use `click.BadParameter` for configuration errors instead of `sys.exit(1)`
- Clear error messages with actionable guidance
- Validation before execution starts
- Comprehensive input validation

### 5. Enhanced Configuration Display ✅

**New Summary Output**:
```
📊 Configuration Summary:
   Scenario counts: {'horizontal': 10, 'vertical': 5, 'sector': 0}
   Complexity tiers: ['simple', 'moderate']
   Shift levels: ['in_distribution', 'moderate_shift']
   Max interventions: 3
   Step size: 5.0s
   Time horizon: 1 minutes
   Total scenarios: 30
```

### 6. Data Structure Improvements ✅

**BenchmarkConfiguration Updates**:
```python
@dataclass 
class BenchmarkConfiguration:
    # NEW: per-type scenario counts
    scenario_counts: Optional[Dict[str, int]] = None
    
    # NEW: exposed simulation parameters
    max_interventions_per_scenario: int = 5
    step_size_seconds: float = 10.0
    
    # Backward compatibility maintained
    num_scenarios_per_type: int = 50
```

**ScenarioResult Fixes**:
- Fixed dataclass field ordering issues
- Added proper default values for mutable fields
- Added `__post_init__` method for list initialization

### 7. Example Usage

**Basic Usage**:
```bash
python -m llm_atc.cli monte-carlo-benchmark \
    --num-horizontal 20 \
    --num-vertical 10 \
    --num-sector 5 \
    --complexities simple,moderate \
    --max-interventions 3 \
    --step-size 5.0
```

**Custom Configuration**:
```bash
python -m llm_atc.cli monte-carlo-benchmark \
    --num-horizontal 50 \
    --num-vertical 0 \
    --num-sector 30 \
    --complexities simple,complex,extreme \
    --shift-levels in_distribution,extreme_shift \
    --horizon 10 \
    --max-interventions 7 \
    --step-size 15.0 \
    --output-dir custom_results
```

### 8. Validation Tests ✅

Created comprehensive test suite verifying:
- ✅ Help output includes all new options
- ✅ Invalid complexity tiers are properly rejected
- ✅ Per-type scenario counts work correctly
- ✅ Zero scenario counts are validated
- ✅ BenchmarkConfiguration handles new parameters
- ✅ Error handling with clear messages

### 9. Backward Compatibility ✅

All changes maintain backward compatibility:
- Existing `num_scenarios_per_type` field preserved
- Default values provided for new fields
- Automatic initialization of `scenario_counts` from `num_scenarios_per_type`

### 10. Files Modified

1. **`llm_atc/cli.py`**:
   - Added new CLI options
   - Enhanced validation and error handling
   - Improved configuration display

2. **`scenarios/monte_carlo_runner.py`**:
   - Fixed ScenarioResult dataclass field ordering
   - Enhanced BenchmarkConfiguration with new fields
   - Already supported scenario_counts functionality

3. **`test_cli_enhancements.py`** (new):
   - Comprehensive test suite for CLI improvements

### Benefits

1. **Flexibility**: Users can now specify different scenario counts per type
2. **Validation**: Clear error messages prevent configuration mistakes
3. **Configurability**: Key simulation parameters are now adjustable
4. **Usability**: Better error handling and configuration summary
5. **Maintainability**: Clean validation logic and backward compatibility
6. **Testing**: Comprehensive test coverage ensures reliability

All requested CLI configuration enhancements have been successfully implemented and tested.
