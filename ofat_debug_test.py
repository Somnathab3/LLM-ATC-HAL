#!/usr/bin/env python3
"""
OFAT Sweep Debug Test - Simplified Example
==========================================
This script demonstrates the One Factor At a Time (OFAT) parameter sweep 
with detailed debugging output to show exactly what happens during the process.
"""

import yaml
import numpy as np
import json
import os
import copy
from typing import Dict, List, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file with debug info"""
    print(f"🔍 Loading YAML file: {file_path}")
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"✅ YAML loaded successfully. Keys: {list(data.keys())}")
    return data

def flatten_ranges_dict(ranges_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, List]:
    """Flatten nested ranges dictionary with detailed debugging"""
    print(f"\n🔍 Flattening ranges dictionary...")
    print(f"   Parent key: '{parent_key}'")
    
    items = []
    for k, v in ranges_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        print(f"   Processing key: '{k}' -> '{new_key}'")
        
        if isinstance(v, dict):
            if 'pool' in v and 'weights' in v:
                print(f"     ⏭️ Skipping non-numeric parameter (pool/weights): {new_key}")
                continue
            elif all(isinstance(val, list) and len(val) == 2 for val in v.values() if isinstance(val, list)):
                print(f"     📋 Found leaf node with range specifications")
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, list) and len(sub_v) == 2 and all(isinstance(x, (int, float)) for x in sub_v):
                        param_name = f"{new_key}{sep}{sub_k}"
                        items.append((param_name, sub_v))
                        print(f"       ✅ Added parameter: {param_name} = {sub_v}")
            else:
                print(f"     🔄 Recursing into nested dictionary")
                nested_items = flatten_ranges_dict(v, new_key, sep=sep)
                items.extend(nested_items.items())
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            items.append((new_key, v))
            print(f"     ✅ Added numeric range: {new_key} = {v}")
        else:
            print(f"     ⏭️ Skipping non-range value: {new_key} = {type(v).__name__}")
    
    result = dict(items)
    print(f"\n✅ Flattening complete. Found {len(result)} numeric parameters:")
    for param, range_val in result.items():
        print(f"   • {param}: {range_val}")
    
    return result

def demonstrate_parameter_override(base_ranges: Dict[str, Any], param_path: str, new_value: float) -> Dict[str, Any]:
    """Demonstrate how parameter override works with detailed debugging"""
    print(f"\n🎯 PARAMETER OVERRIDE DEMONSTRATION")
    print(f"   Parameter: {param_path}")
    print(f"   New value: {new_value}")
    
    # Create a deep copy to avoid modifying original
    temp_ranges = copy.deepcopy(base_ranges)
    print(f"   📋 Created deep copy of base ranges")
    
    # Navigate to the parameter location
    keys = param_path.split('.')
    print(f"   🗂️ Navigation path: {' -> '.join(keys)}")
    
    current = temp_ranges
    for i, key in enumerate(keys[:-1]):
        print(f"   📁 Level {i+1}: Entering '{key}'")
        if key in current:
            current = current[key]
            print(f"      ✅ Found key, type: {type(current).__name__}")
        else:
            print(f"      ❌ Key not found!")
            return temp_ranges
    
    # Set the final parameter
    final_key = keys[-1]
    print(f"   🎯 Setting final parameter: '{final_key}'")
    
    if final_key in current:
        original_value = current[final_key]
        print(f"      📊 Original value: {original_value}")
        
        if isinstance(original_value, list) and len(original_value) == 2:
            current[final_key] = [new_value, new_value]  # Fixed range
            print(f"      ✅ Updated to fixed range: [{new_value}, {new_value}]")
        else:
            print(f"      ⚠️ Cannot override non-range value")
    else:
        print(f"      ❌ Final key '{final_key}' not found!")
    
    return temp_ranges

def generate_mock_scenario(param_name: str, param_value: float, scenario_id: int) -> Dict[str, Any]:
    """Generate a mock scenario with detailed debugging"""
    print(f"   🎭 Generating scenario {scenario_id} for {param_name}={param_value}")
    
    scenario = {
        'id': f"scenario_{scenario_id}",
        'complexity': 'complex',
        'parameter_override': {param_name: param_value},
        'aircraft_count': np.random.randint(8, 12),
        'conflicts_expected': np.random.randint(1, 4),
        'duration_minutes': np.random.uniform(5, 15),
        'airspace_region': 'EHAM_TMA',
        'generated_at': f"step_{scenario_id}"
    }
    
    print(f"      ✅ Generated: {scenario['aircraft_count']} aircraft, {scenario['conflicts_expected']} conflicts")
    return scenario

def run_ofat_debug_test():
    """Run the complete OFAT debug test"""
    print("🚀 STARTING OFAT SWEEP DEBUG TEST")
    print("=" * 60)
    
    # Configuration
    k = 3  # Reduced for debugging (3 points per parameter)
    M = 5  # Reduced scenarios per parameter-value pair
    
    print(f"\n📋 CONFIGURATION:")
    print(f"   Grid resolution (k): {k}")
    print(f"   Scenarios per parameter-value: {M}")
    
    # Step 1: Load base ranges
    print(f"\n📖 STEP 1: LOADING BASE RANGES")
    base_ranges = load_yaml("scenario_ranges.yaml")
    
    # Step 2: Flatten ranges to get sweepable parameters
    print(f"\n🗂️ STEP 2: EXTRACTING SWEEPABLE PARAMETERS")
    flat_ranges = flatten_ranges_dict(base_ranges)
    n_params = len(flat_ranges)
    
    print(f"\n📊 SWEEP SUMMARY:")
    print(f"   Total parameters to sweep: {n_params}")
    print(f"   Total parameter-value combinations: {n_params * k}")
    print(f"   Total scenarios to generate: {n_params * k * M}")
    
    # Step 3: Demonstrate OFAT sweep for first few parameters
    print(f"\n🔄 STEP 3: OFAT SWEEP DEMONSTRATION")
    
    # Create output directory
    output_dir = "debug_sweep_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   📁 Created output directory: {output_dir}")
    
    # Limit to first 3 parameters for demonstration
    demo_params = list(flat_ranges.items())[:3]
    print(f"\n   🎯 Demonstrating with first 3 parameters:")
    for i, (param, range_val) in enumerate(demo_params):
        print(f"      {i+1}. {param}: {range_val}")
    
    all_results = []
    
    for param_idx, (P, (mn, mx)) in enumerate(demo_params):
        print(f"\n" + "=" * 50)
        print(f"🔍 PARAMETER {param_idx + 1}/{len(demo_params)}: {P}")
        print(f"   Range: [{mn}, {mx}]")
        
        # Generate k values across the range
        v_list = np.linspace(mn, mx, k).tolist()
        print(f"   🎯 Sample values: {[f'{v:.3f}' for v in v_list]}")
        
        for v_idx, v in enumerate(v_list):
            print(f"\n   📍 VALUE {v_idx + 1}/{k}: {v:.6f}")
            
            # Step 3a: Create modified ranges
            print(f"      🔧 Creating modified ranges...")
            temp_ranges = demonstrate_parameter_override(base_ranges, P, v)
            
            # Step 3b: Generate scenarios
            print(f"      🎭 Generating {M} scenarios...")
            scenarios = []
            
            for i in range(M):
                scenario = generate_mock_scenario(P, v, i)
                
                # Package with metadata
                scenario_data = {
                    'parameter': P,
                    'value': v,
                    'scenario_id': i,
                    'config': scenario
                }
                scenarios.append(scenario_data)
            
            print(f"      ✅ Generated {len(scenarios)} scenarios")
            
            # Step 3c: Save scenarios to file
            output_file = f"{output_dir}/{P.replace('.', '_')}={v:.3f}.jsonl"
            with open(output_file, 'w') as f:
                for scenario in scenarios:
                    json.dump(scenario, f)
                    f.write('\n')
            
            print(f"      💾 Saved to: {output_file}")
            
            # Step 3d: Simulate test execution and collect results
            print(f"      🧪 Simulating test execution...")
            for scenario in scenarios:
                # Mock test results
                result = {
                    'parameter': P,
                    'value': v,
                    'scenario_id': scenario['scenario_id'],
                    'false_positive': np.random.beta(2, 8),
                    'false_negative': np.random.beta(3, 7),
                    'safety_margin': np.random.uniform(0.7, 0.95),
                    'extra_length': np.random.uniform(0.8, 1.2),
                    'interventions': np.random.poisson(2),
                    'entropy': np.random.exponential(0.1)
                }
                all_results.append(result)
            
            print(f"      ✅ Collected {len(scenarios)} test results")
    
    # Step 4: Analyze results
    print(f"\n" + "=" * 50)
    print(f"📊 STEP 4: RESULTS ANALYSIS")
    
    # Save all results
    results_file = f"{output_dir}/all_results.jsonl"
    with open(results_file, 'w') as f:
        for result in all_results:
            json.dump(result, f)
            f.write('\n')
    
    print(f"💾 Saved {len(all_results)} results to: {results_file}")
    
    # Group by parameter and analyze
    print(f"\n📈 PARAMETER SENSITIVITY ANALYSIS:")
    
    from collections import defaultdict
    param_results = defaultdict(list)
    
    for result in all_results:
        param_results[result['parameter']].append(result)
    
    for param, results in param_results.items():
        print(f"\n   🎯 Parameter: {param}")
        
        # Group by value
        value_stats = defaultdict(list)
        for r in results:
            value_stats[r['value']].append(r)
        
        print(f"      Values tested: {len(value_stats)}")
        
        for value, value_results in value_stats.items():
            fp_mean = np.mean([r['false_positive'] for r in value_results])
            fn_mean = np.mean([r['false_negative'] for r in value_results])
            safety_mean = np.mean([r['safety_margin'] for r in value_results])
            
            print(f"      • Value {value:.3f}: FP={fp_mean:.3f}, FN={fn_mean:.3f}, Safety={safety_mean:.3f}")
    
    # Step 5: Summary
    print(f"\n" + "=" * 50)
    print(f"🎯 STEP 5: SUMMARY")
    print(f"   Parameters processed: {len(param_results)}")
    print(f"   Total scenarios generated: {len(all_results)}")
    print(f"   Files created:")
    
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"      📄 {file} ({size} bytes)")
    
    print(f"\n✅ OFAT DEBUG TEST COMPLETED SUCCESSFULLY!")
    print(f"📁 Check the '{output_dir}' directory for all generated files.")
    
    # Step 6: Show what a real parameter sweep would look like
    print(f"\n" + "=" * 50)
    print(f"🔮 FULL SWEEP PROJECTION:")
    print(f"   If we processed ALL {n_params} parameters:")
    print(f"   • Total combinations: {n_params} × {k} = {n_params * k}")
    print(f"   • Total scenarios: {n_params * k} × {M} = {n_params * k * M}")
    print(f"   • Estimated files: {n_params * k + 1} (scenarios + results)")
    
    return output_dir

if __name__ == "__main__":
    try:
        output_dir = run_ofat_debug_test()
        print(f"\n🎉 Debug test completed! Check '{output_dir}' for results.")
    except Exception as e:
        print(f"\n❌ Error during debug test: {e}")
        import traceback
        traceback.print_exc()
