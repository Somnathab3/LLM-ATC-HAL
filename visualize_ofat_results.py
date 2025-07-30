#!/usr/bin/env python3
"""
OFAT Results Visualizer
=======================
Visualize the results from the debug OFAT sweep to show parameter sensitivity.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def load_and_visualize_results():
    """Load and visualize the OFAT debug test results"""
    
    # Load results
    results_file = "debug_sweep_output/all_results.jsonl"
    
    if not os.path.exists(results_file):
        print("❌ Results file not found. Run ofat_debug_test.py first.")
        return
    
    print("📊 Loading and visualizing OFAT debug results...")
    
    # Read results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"✅ Loaded {len(results)} test results")
    
    # Group by parameter
    param_data = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        param = result['parameter']
        value = result['value']
        param_data[param][value].append(result)
    
    # Create visualizations
    metrics = ['false_positive', 'false_negative', 'safety_margin', 'extra_length', 'interventions']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        for j, (param, param_results) in enumerate(param_data.items()):
            values = sorted(param_results.keys())
            means = []
            stds = []
            
            for value in values:
                metric_values = [r[metric] for r in param_results[value]]
                means.append(np.mean(metric_values))
                stds.append(np.std(metric_values))
            
            # Plot with error bars
            ax.errorbar(values, means, yerr=stds, 
                       label=param.replace('_', ' ').replace('.', ' ').title(),
                       marker=markers[j % len(markers)], 
                       color=colors[j % len(colors)],
                       capsize=5, linewidth=2, markersize=8)
        
        ax.set_title(f'{metric.replace("_", " ").title()} vs Parameter Values', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save plot
    output_file = "debug_sweep_output/parameter_sensitivity_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved visualization: {output_file}")
    
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("📈 DETAILED PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    for param, param_results in param_data.items():
        print(f"\n🎯 Parameter: {param}")
        print(f"   Range tested: {min(param_results.keys()):.3f} to {max(param_results.keys()):.3f}")
        
        # Calculate sensitivity (range of means across values)
        for metric in metrics:
            value_means = []
            for value in param_results.keys():
                metric_values = [r[metric] for r in param_results[value]]
                value_means.append(np.mean(metric_values))
            
            sensitivity = max(value_means) - min(value_means)
            print(f"   {metric} sensitivity: {sensitivity:.4f} (range: {min(value_means):.3f} - {max(value_means):.3f})")
    
    print(f"\n✅ Analysis complete! Check {output_file} for visualization.")

if __name__ == "__main__":
    load_and_visualize_results()
