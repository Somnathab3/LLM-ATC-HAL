# Fix Summary: Distribution Shift Range Errors

## Issue Description
The scenario generator was failing with "empty range in randrange" errors when using distribution shifts, particularly for `moderate_shift` and `extreme_shift` scenarios. The errors occurred when traffic density multipliers created invalid ranges where the minimum value was larger than the maximum value.

## Root Cause
The distribution shift logic in `monte_carlo_framework.py` was applying multipliers to ranges without ensuring that the resulting minimum and maximum values maintained the correct ordering (min ≤ max). This happened because:

1. Traffic density multipliers (1.3 for moderate_shift, 1.7 for extreme_shift) were applied to both min and max values
2. When a safety cap was applied (e.g., `min(25, max_value)`), it could create situations where the new minimum exceeded the capped maximum
3. The `sample_from_range` function used `random.randint(min, max)` which requires min ≤ max

## Example of the Problem
```python
# Original range: [18, 20] for extreme complexity
# After 1.7x multiplier: [30.6, 34] → [30, 34] 
# After safety cap: [30, min(25, 34)] → [30, 25]
# Result: randrange(30, 25) → ERROR: empty range
```

## Fixes Applied

### 1. Fixed `sample_from_range` function
```python
def sample_from_range(self, range_spec: Any) -> Any:
    """Sample a value from a range specification"""
    if isinstance(range_spec, list) and len(range_spec) == 2:
        if isinstance(range_spec[0], int) and isinstance(range_spec[1], int):
            # Ensure min <= max to avoid empty range error
            min_val = min(range_spec[0], range_spec[1])
            max_val = max(range_spec[0], range_spec[1])
            return random.randint(min_val, max_val)
        else:
            # Ensure min <= max for float ranges too
            min_val = min(range_spec[0], range_spec[1])
            max_val = max(range_spec[0], range_spec[1])
            return random.uniform(min_val, max_val)
```

### 2. Fixed aircraft count range calculations
```python
# Apply traffic density multiplier to aircraft counts
if 'traffic_density_multiplier' in shift_config:
    multiplier = shift_config['traffic_density_multiplier']
    for complexity in shifted_ranges['aircraft']['count']:
        base_range = shifted_ranges['aircraft']['count'][complexity]
        new_min = max(1, int(base_range[0] * multiplier))
        new_max = max(new_min, min(25, int(base_range[1] * multiplier)))  # Ensure max >= min
        shifted_ranges['aircraft']['count'][complexity] = [new_min, new_max]
```

### 3. Fixed wind speed range calculations
```python
# Wind speed shifts
if 'wind' in weather_config and 'speed_shift_kts' in weather_config['wind']:
    wind_shift = weather_config['wind']['speed_shift_kts']
    base_wind = shifted_ranges['weather']['wind']['speed_kts']
    new_min = max(0, base_wind[0] + wind_shift[0])
    new_max = max(new_min, min(100, base_wind[1] + wind_shift[1]))  # Ensure max >= min
    shifted_ranges['weather']['wind']['speed_kts'] = [new_min, new_max]
```

### 4. Applied similar fixes to all range calculations
- Turbulence intensity ranges
- Visibility degradation ranges
- Traffic density multiplier ranges
- Geography radius expansion ranges

## Verification
Created test script `test_scenario_fixes.py` that verifies:
- All distribution shift levels work correctly
- All complexity tiers work correctly  
- All scenario environment types work correctly
- No more "empty range" errors occur

### Test Results
```
Success rate: 9/9 (100.0%)
🎉 All tests PASSED! Distribution shift issues have been fixed.

Environment types: 3/3 working
🎉 ALL TESTS PASSED! Scenario generator is working correctly.
```

## Files Modified
1. `scenarios/monte_carlo_framework.py` - Fixed range calculation logic
2. `test_scenario_fixes.py` - Created comprehensive test suite

## Impact
- All distribution shift scenarios now generate successfully
- No more "empty range in randrange" errors
- Scenario generator is fully functional across all complexity and shift levels
- Maintains backward compatibility with existing code
