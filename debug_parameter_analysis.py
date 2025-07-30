#!/usr/bin/env python3
"""
Debug script for parameter-dependent false positive/negative analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
import os
from typing import Dict, List, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file and return as dictionary"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def flatten_ranges_dict(ranges_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, List]:
    """Flatten nested ranges dictionary to get parameter-value pairs"""
    items = []
    for k, v in ranges_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            if 'pool' in v and 'weights' in v:
                # Skip non-numeric parameters like aircraft types
                continue
            elif all(isinstance(val, list) and len(val) == 2 for val in v.values() if isinstance(val, list)):
                # This is a leaf node with range specifications
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, list) and len(sub_v) == 2 and all(isinstance(x, (int, float)) for x in sub_v):
                        items.append((f"{new_key}{sep}{sub_k}", sub_v))
            else:
                items.extend(flatten_ranges_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            # This is a numeric range
            items.append((new_key, v))
    return dict(items)

def calculate_parameter_dependent_rates(param: str, value: float, param_range: List[float]) -> tuple:
    """
    Calculate parameter-dependent false positive and false negative rates.
    """
    # Ensure we have numeric values
    try:
        min_val, max_val = float(param_range[0]), float(param_range[1])
        value = float(value)
    except (ValueError, TypeError) as e:
        print(f"Error converting to float: param={param}, value={value}, range={param_range}")
        raise e
    
    # Normalize value to [0, 1] within parameter range
    normalized_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    # Parameter-specific response patterns
    if 'altitude' in param.lower():
        # Higher altitudes -> more false negatives (harder to detect conflicts)
        # Lower altitudes -> more false positives (over-cautious detection)
        fp_base = 0.15 * (1 - normalized_value) + 0.05  # 5-20% range
        fn_base = 0.10 * normalized_value + 0.03        # 3-13% range
        
    elif 'speed' in param.lower() or 'mach' in param.lower():
        # Higher speeds -> more false negatives (detection lag)
        # Moderate speeds optimal, extremes cause issues
        speed_factor = abs(normalized_value - 0.5) * 2  # Distance from optimal (0.5)
        fp_base = 0.08 + 0.12 * speed_factor           # 8-20% range
        fn_base = 0.05 + 0.15 * normalized_value        # 5-20% range
        
    elif 'separation' in param.lower():
        # Closer separation -> more false positives (over-cautious)
        # Larger separation -> more false negatives (under-detection)
        fp_base = 0.20 * (1 - normalized_value) + 0.05  # 5-25% range
        fn_base = 0.15 * normalized_value + 0.02         # 2-17% range
        
    elif 'traffic' in param.lower() or 'density' in param.lower():
        # Higher traffic density -> both FP and FN increase (complexity)
        complexity_factor = normalized_value
        fp_base = 0.08 + 0.17 * complexity_factor       # 8-25% range
        fn_base = 0.04 + 0.16 * complexity_factor       # 4-20% range
        
    else:
        # Default pattern for unknown parameters
        fp_base = 0.10 + 0.10 * abs(normalized_value - 0.5) * 2  # U-shaped curve
        fn_base = 0.08 + 0.12 * normalized_value                  # Linear increase
    
    return fp_base, fn_base

def main():
    print("Debug: Parameter-Dependent False Positive/Negative Analysis")
    print("=" * 60)
    
    # Load parameter ranges
    try:
        base_ranges = load_yaml("scenario_ranges.yaml")
        flat_ranges = flatten_ranges_dict(base_ranges)
        print(f"Found {len(flat_ranges)} parameters:")
        for param, range_vals in flat_ranges.items():
            print(f"  {param}: {range_vals}")
    except Exception as e:
        print(f"Error loading ranges: {e}")
        return
    
    print("\n" + "=" * 60)
    print("GENERATING PARAMETER-DEPENDENT TEST DATA")
    print("=" * 60)
    
    # Generate test data for a subset of parameters
    test_params = list(flat_ranges.keys())[:3]  # Test first 3 parameters
    k = 5  # Grid resolution
    scenarios_per_point = 20  # Reduced for debugging
    
    all_results = []
    
    for param in test_params:
        param_range = flat_ranges[param]
        print(f"\nProcessing parameter: {param}")
        print(f"  Range: {param_range}")
        
        # Generate k values across the parameter range
        min_val, max_val = param_range
        values = np.linspace(min_val, max_val, k)
        
        for value in values:
            print(f"  Testing value: {value:.3f}")
            
            # Generate scenarios for this parameter-value combination
            for scenario_id in range(scenarios_per_point):
                try:
                    # Calculate parameter-dependent rates
                    fp_rate, fn_rate = calculate_parameter_dependent_rates(param, value, param_range)
                    
                    # Add realistic noise
                    fp_noise = np.random.normal(0, 0.02)
                    fn_noise = np.random.normal(0, 0.02)
                    
                    result = {
                        'parameter': param,
                        'value': value,
                        'scenario_id': scenario_id,
                        'false_positive': max(0, min(1, fp_rate + fp_noise)),
                        'false_negative': max(0, min(1, fn_rate + fn_noise)),
                        'safety_margin': np.random.uniform(0.7, 0.95),
                        'interventions': np.random.poisson(2),
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"    Error generating scenario {scenario_id}: {e}")
                    continue
    
    print(f"\nGenerated {len(all_results)} test results")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("FALSE POSITIVE & FALSE NEGATIVE ANALYSIS")
    print("=" * 60)
    
    summary = (
        df.groupby(['parameter', 'value'])
        .agg({
            'false_positive': ['mean', 'std', 'min', 'max'],
            'false_negative': ['mean', 'std', 'min', 'max'],
            'safety_margin': ['mean', 'std'],
            'interventions': ['mean', 'std']
        }).reset_index()
    )
    
    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
    
    # Analysis for each parameter
    for param in summary['parameter'].unique():
        param_data = summary[summary['parameter'] == param].copy()
        
        if len(param_data) > 1:
            # Calculate sensitivity (rate of change)
            fp_sensitivity = (param_data['false_positive_max'].max() - param_data['false_positive_min'].min())
            fn_sensitivity = (param_data['false_negative_max'].max() - param_data['false_negative_min'].min())
            
            print(f"\nParameter: {param}")
            print(f"  FP Rate Range: {param_data['false_positive_mean'].min():.3f} - {param_data['false_positive_mean'].max():.3f}")
            print(f"  FN Rate Range: {param_data['false_negative_mean'].min():.3f} - {param_data['false_negative_mean'].max():.3f}")
            print(f"  FP Sensitivity: {fp_sensitivity:.3f}")
            print(f"  FN Sensitivity: {fn_sensitivity:.3f}")
            
            # Find optimal parameter value (lowest combined error rate)
            param_data['combined_error'] = param_data['false_positive_mean'] + param_data['false_negative_mean']
            optimal_idx = param_data['combined_error'].idxmin()
            optimal_value = param_data.loc[optimal_idx, 'value']
            optimal_fp = param_data.loc[optimal_idx, 'false_positive_mean']
            optimal_fn = param_data.loc[optimal_idx, 'false_negative_mean']
            
            print(f"  Optimal Value: {optimal_value:.3f} (FP: {optimal_fp:.3f}, FN: {optimal_fn:.3f})")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    plots_dir = "debug_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    for param in summary['parameter'].unique():
        df_p = summary[summary['parameter'] == param].copy()
        
        # Create FP/FN plot
        plt.figure(figsize=(10, 6))
        
        # Plot with error bars
        plt.errorbar(
            df_p['value'], df_p['false_positive_mean'], 
            yerr=df_p['false_positive_std'],
            marker='o', linewidth=2, markersize=6, 
            label='False Positive', color='red', alpha=0.7
        )
        plt.errorbar(
            df_p['value'], df_p['false_negative_mean'],
            yerr=df_p['false_negative_std'],
            marker='s', linewidth=2, markersize=6,
            label='False Negative', color='blue', alpha=0.7
        )
        
        # Mark optimal point
        combined_error = df_p['false_positive_mean'] + df_p['false_negative_mean']
        optimal_idx = combined_error.idxmin()
        optimal_value = df_p.loc[optimal_idx, 'value']
        
        plt.axvline(x=optimal_value, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal Value: {optimal_value:.3f}')
        
        plt.title(f"False Positive & False Negative Rates\nvs {param.replace('_', ' ').title()}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("Parameter Value", fontsize=12)
        plt.ylabel("Error Rate", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        clean_param = param.replace('.', '_').replace(' ', '_')
        plot_file = f"{plots_dir}/{clean_param}_fp_fn_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated plot: {plot_file}")
    
    # Save summary data
    summary.to_csv(f"{plots_dir}/parameter_analysis_summary.csv", index=False)
    print(f"\nSummary saved to: {plots_dir}/parameter_analysis_summary.csv")
    
    print("\n" + "=" * 60)
    print("DEBUG ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
