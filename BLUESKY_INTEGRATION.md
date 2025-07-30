# BlueSky Integration - Real Simulator Integration

This document describes the new real BlueSky simulator integration that replaces the previous hardcoded aircraft data in `bluesky_tools.py`.

## Overview

The BlueSky integration now provides:
- **Real BlueSky Simulator Integration**: Direct connection to BlueSky when available
- **Enhanced Mock Simulation**: Realistic fallback when BlueSky is not available
- **Graceful Degradation**: Seamless switching between real and mock data
- **Configuration Management**: YAML-based configuration for all settings
- **Source Tracking**: All data includes source information for debugging

## Key Features

### Real BlueSky Integration (when available)
- ✅ Live aircraft data from `bluesky.traf` module
- ✅ Real conflict detection from BlueSky's CD system
- ✅ Direct command execution through BlueSky stack
- ✅ Actual simulation time control and physics
- ✅ Realistic flight dynamics and performance models

### Enhanced Mock Simulation (fallback)
- 🎭 Configurable aircraft distributions and counts
- 🌍 Realistic geographical bounds (configurable airspace)
- 📊 Altitude ranges and aircraft types
- ⚙️ Consistent API regardless of BlueSky availability

## Configuration

### Configuration File: `bluesky_config.yaml`

```yaml
bluesky:
  # Connection method: 'local' for direct import, 'network' for TCP connection
  connection_type: "local"
  
  # Network connection settings (for remote BlueSky instances)
  network:
    host: "localhost"
    port: 8080
    timeout: 10.0
    
  # Simulation settings
  simulation:
    default_dt_mult: 1.0
    max_simulation_time: 3600  # seconds
    conflict_detection_method: "SWARM"
    separation_standards:
      horizontal_nm: 5.0
      vertical_ft: 1000.0
      
  # Mock data settings (fallback when BlueSky unavailable)
  mock_data:
    use_realistic_aircraft_count: true
    default_aircraft_count: 10
    airspace_bounds:
      lat_min: 51.0
      lat_max: 53.0
      lon_min: 3.0
      lon_max: 6.0
    altitude_range:
      min_fl: 200  # FL200
      max_fl: 400  # FL400
```

## API Usage

### Basic Usage
```python
from llm_atc.tools.bluesky_tools import (
    get_all_aircraft_info,
    get_conflict_info,
    send_command,
    step_simulation,
    reset_simulation,
    get_distance,
    BLUESKY_AVAILABLE
)

# Check if real BlueSky is available
if BLUESKY_AVAILABLE:
    print("Using real BlueSky simulator")
else:
    print("Using enhanced mock simulation")

# Get aircraft data (real or mock)
aircraft_data = get_all_aircraft_info()
print(f"Data source: {aircraft_data['source']}")  # 'bluesky_real' or 'mock_data'
print(f"Aircraft count: {aircraft_data['total_aircraft']}")

# Get conflicts (real or mock)
conflicts = get_conflict_info()
print(f"Conflicts found: {conflicts['total_conflicts']}")
print(f"Data source: {conflicts['source']}")

# Send commands to simulation
result = send_command("CRE KLM123,B738,52.3,4.8,90,35000,250")
print(f"Command result: {result['status']} (source: {result['source']})")

# Step simulation forward
step_result = step_simulation(minutes=2.0, dtmult=1.0)
print(f"Simulation advanced: {step_result['minutes_advanced']} minutes")
```

### Distance Calculation
```python
# Calculate distance between aircraft
distance = get_distance("KLM123", "AFR456")
print(f"Horizontal: {distance['horizontal_nm']:.2f} nm")
print(f"Vertical: {distance['vertical_ft']:.0f} ft")
print(f"3D: {distance['total_3d_nm']:.2f} nm")
```

## Data Sources

All functions now return a `source` field indicating the data origin:

- `"bluesky_real"`: Data from actual BlueSky simulator
- `"mock_data"`: Enhanced mock data (BlueSky unavailable)
- `"mock_simulation"`: Simulated command execution

## Error Handling

The integration includes comprehensive error handling:

```python
try:
    aircraft_data = get_all_aircraft_info()
    # Process data...
except BlueSkyToolsError as e:
    print(f"BlueSky tools error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Installation Requirements

### For Real BlueSky Integration
```bash
# Install BlueSky simulator
pip install bluesky-simulator

# Install optional dependencies for enhanced features
pip install PyYAML  # For YAML configuration support
```

### Minimum Requirements (Mock Mode)
```bash
# Only Python standard library required for mock mode
# PyYAML is optional (will use JSON fallback)
```

## Testing

### Integration Test
```bash
python test_bluesky_integration.py
```

### Demo Script
```bash
python bluesky_integration_demo.py
```

## Migration from Previous Version

The new integration is **backward compatible**. Existing code will continue to work, but will now benefit from:

1. **Real BlueSky data** when the simulator is available
2. **Enhanced mock data** with realistic distributions
3. **Source tracking** for debugging and validation
4. **Configuration management** for customization

## Monte Carlo Benchmark Impact

The new integration significantly improves Monte Carlo benchmark capabilities:

### Before (Hardcoded Data)
- ❌ Static aircraft positions
- ❌ Fake conflict scenarios
- ❌ No realistic dynamics
- ❌ Limited test scenarios

### After (Real Integration)
- ✅ Dynamic aircraft movement with physics
- ✅ Real conflict detection algorithms
- ✅ Realistic flight performance models
- ✅ Unlimited scenario generation
- ✅ True simulation time progression
- ✅ Environmental effects (weather, winds, etc.)

## Configuration Examples

### High-Density Testing
```yaml
mock_data:
  default_aircraft_count: 50
  airspace_bounds:
    lat_min: 51.0
    lat_max: 52.0  # Smaller area = higher density
    lon_min: 4.0
    lon_max: 5.0
```

### Oceanic Testing
```yaml
mock_data:
  airspace_bounds:
    lat_min: 40.0
    lat_max: 50.0
    lon_min: -40.0
    lon_max: -10.0  # Atlantic Ocean
  separation_standards:
    horizontal_nm: 10.0  # Oceanic separation
```

## Troubleshooting

### BlueSky Not Detected
- Ensure BlueSky is installed: `pip install bluesky-simulator`
- Check Python path and import statements
- Verify BlueSky can be imported: `python -c "import bluesky; print('OK')"`

### Configuration Issues
- Check `bluesky_config.yaml` syntax
- Ensure file is in project root or config/ directory
- Check file permissions

### Network Connection (Future)
- Verify BlueSky server is running
- Check host/port configuration
- Test connection: `telnet <host> <port>`

## Future Enhancements

- 🚀 Network-based BlueSky connections
- 📊 Real-time visualization integration
- 🌦️ Weather simulation integration
- 🛰️ ADS-B data integration
- 📈 Performance optimization for large scenarios
