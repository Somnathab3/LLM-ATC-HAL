# analysis/visualisation.py
"""
Visualisation Helpers for LLM-ATC-HAL Analysis
==============================================
Provides plotting functions for conflict detection timelines and conflict resolution flowcharts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_cd_timeline(df: pd.DataFrame, sim_id: str, output_dir: str = "thesis_results") -> str:
    """
    Plot conflict detection timeline for a specific simulation.
    
    Args:
        df: Simulation results dataframe
        sim_id: Simulation ID to plot
        output_dir: Output directory for the plot
        
    Returns:
        Path to saved plot file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter data for specific simulation
    sim_data = df[df['scenario_id'] == sim_id]
    
    if len(sim_data) == 0:
        raise ValueError(f"No data found for simulation ID: {sim_id}")
    
    # Extract simulation details
    sim_row = sim_data.iloc[0]
    tier = sim_row['tier']
    aircraft_count = sim_row['aircraft_count']
    complexity = sim_row['complexity']
    
    # Generate mock timeline data (in real implementation, this would come from BlueSky logs)
    timeline_data = _generate_mock_timeline_data(sim_row)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Main timeline plot
    _plot_separation_timeline(ax1, timeline_data, sim_row)
    
    # Command timeline
    _plot_command_timeline(ax2, timeline_data, sim_row)
    
    # Overall title
    fig.suptitle(f'Conflict Detection Timeline - {sim_id}\n'
                f'Tier: {tier} | Aircraft: {aircraft_count} | Complexity: {complexity}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"cd_timeline_{sim_id}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def _plot_separation_timeline(ax, timeline_data: Dict, sim_row: pd.Series):
    """Plot the main separation timeline"""
    times = timeline_data['times']
    horizontal_sep = timeline_data['horizontal_separation']
    vertical_sep = timeline_data['vertical_separation']
    cd_triggers = timeline_data['cd_triggers']
    cr_commands = timeline_data['cr_commands']
    
    # Create scatter plot with color-coded vertical separation
    scatter = ax.scatter(times, horizontal_sep, c=vertical_sep, 
                        cmap='RdYlGn', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar for vertical separation
    cbar = plt.colorbar(scatter, ax=ax, label='Vertical Separation [ft]')
    cbar.ax.tick_params(labelsize=10)
    
    # Add horizontal line for minimum separation standard
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=2, alpha=0.8, 
               label='ICAO Min Horizontal (5 nm)')
    
    # Mark conflict detection triggers
    for trigger_time in cd_triggers:
        ax.axvline(x=trigger_time, color='orange', linestyle='-', linewidth=2, alpha=0.7)
        ax.annotate('CD Trigger', xy=(trigger_time, max(horizontal_sep) * 0.9),
                   xytext=(trigger_time + 10, max(horizontal_sep) * 0.9),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                   fontsize=9, color='orange', fontweight='bold')
    
    # Mark conflict resolution commands
    for cmd_time in cr_commands:
        ax.axvline(x=cmd_time, color='blue', linestyle='-', linewidth=2, alpha=0.7)
        ax.annotate('CR Command', xy=(cmd_time, max(horizontal_sep) * 0.8),
                   xytext=(cmd_time + 10, max(horizontal_sep) * 0.8),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                   fontsize=9, color='blue', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Closest Horizontal Separation [nm]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add safety zone shading
    ax.fill_between(times, 0, 5, alpha=0.2, color='red', label='Critical Zone (<5nm)')


def _plot_command_timeline(ax, timeline_data: Dict, sim_row: pd.Series):
    """Plot command execution timeline"""
    times = timeline_data['times']
    commands = timeline_data['commands']
    
    # Create command timeline
    command_types = ['Altitude Change', 'Heading Change', 'Speed Change', 'Hold Pattern']
    colors = ['red', 'blue', 'green', 'purple']
    
    y_positions = []
    colors_used = []
    labels = []
    
    for i, (cmd_time, cmd_type) in enumerate(commands):
        if cmd_type in command_types:
            type_idx = command_types.index(cmd_type)
            y_positions.append(cmd_time)
            colors_used.append(colors[type_idx])
            labels.append(cmd_type)
    
    if y_positions:
        ax.scatter(y_positions, [0.5] * len(y_positions), 
                  c=colors_used, s=100, alpha=0.8, edgecolors='black')
        
        # Add command labels
        for i, (x, label) in enumerate(zip(y_positions, labels)):
            ax.annotate(label, xy=(x, 0.5), xytext=(x, 0.7),
                       ha='center', fontsize=8, rotation=45)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Commands', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(min(times), max(times))
    ax.grid(True, alpha=0.3)
    
    # Create legend for command types
    for i, (cmd_type, color) in enumerate(zip(command_types, colors)):
        ax.scatter([], [], c=color, s=100, label=cmd_type, alpha=0.8, edgecolors='black')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))


def _generate_mock_timeline_data(sim_row: pd.Series) -> Dict:
    """Generate mock timeline data for visualization (replace with real BlueSky data)"""
    duration = 600  # 10 minutes
    times = np.linspace(0, duration, 120)
    
    # Generate realistic separation data
    base_separation = 8.0  # Start at safe separation
    
    # Create conflict scenario with gradual approach
    horizontal_sep = []
    for t in times:
        if t < 200:
            # Initial safe separation
            sep = base_separation + np.random.normal(0, 0.2)
        elif t < 400:
            # Gradual approach (conflict developing)
            progress = (t - 200) / 200
            sep = base_separation - progress * 4.5 + np.random.normal(0, 0.3)
        else:
            # Post-resolution (after CR commands)
            sep = 5.5 + np.random.normal(0, 0.2)
        
        horizontal_sep.append(max(0.5, sep))  # Minimum 0.5nm
    
    # Vertical separation (varying altitude differences)
    vertical_sep = []
    for t in times:
        if t < 250:
            # Decreasing vertical separation
            progress = t / 250
            vert = 2000 - progress * 1200 + np.random.normal(0, 100)
        else:
            # Post-maneuver increased separation
            vert = 1500 + np.random.normal(0, 100)
        
        vertical_sep.append(max(500, vert))  # Minimum 500ft
    
    # Conflict detection triggers
    cd_triggers = [220, 350]  # Times when conflicts are detected
    
    # Conflict resolution commands
    cr_commands = [240, 280, 370]  # Times when CR commands are issued
    
    # Command sequence
    commands = [
        (240, 'Altitude Change'),
        (280, 'Heading Change'),
        (370, 'Speed Change')
    ]
    
    return {
        'times': times,
        'horizontal_separation': horizontal_sep,
        'vertical_separation': vertical_sep,
        'cd_triggers': cd_triggers,
        'cr_commands': cr_commands,
        'commands': commands
    }


def plot_cr_flowchart(sim_id: str, tier: str, output_dir: str = "thesis_results") -> str:
    """
    Generate conflict resolution flowchart using networkx and matplotlib.
    
    Args:
        sim_id: Simulation ID for context
        tier: Distribution shift tier
        output_dir: Output directory for the plot
        
    Returns:
        Path to saved flowchart file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Define nodes with their properties
    nodes = {
        'BlueSky\nScenario': {'pos': (0, 4), 'color': 'lightblue', 'shape': 'box'},
        'Conflict\nDetection': {'pos': (2, 4), 'color': 'orange', 'shape': 'diamond'},
        'LLM\nEnsemble': {'pos': (4, 4), 'color': 'lightgreen', 'shape': 'box'},
        'Primary\nLLM': {'pos': (3, 6), 'color': 'palegreen', 'shape': 'ellipse'},
        'Validator\nLLM': {'pos': (4, 6), 'color': 'palegreen', 'shape': 'ellipse'},
        'Safety\nLLM': {'pos': (5, 6), 'color': 'palegreen', 'shape': 'ellipse'},
        'Hallucination\nDetector': {'pos': (6, 4), 'color': 'yellow', 'shape': 'diamond'},
        'Policy\nChecker': {'pos': (8, 4), 'color': 'pink', 'shape': 'diamond'},
        'Conflict\nSolver': {'pos': (10, 4), 'color': 'lightcoral', 'shape': 'box'},
        'Safety\nQuantifier': {'pos': (8, 2), 'color': 'lavender', 'shape': 'diamond'},
        'Command\nExecution': {'pos': (12, 4), 'color': 'lightsteelblue', 'shape': 'box'},
        'Experience\nReplay': {'pos': (6, 2), 'color': 'lightgray', 'shape': 'ellipse'},
    }
    
    # Add nodes to graph
    for node, props in nodes.items():
        G.add_node(node, **props)
    
    # Define edges (process flow)
    edges = [
        ('BlueSky\nScenario', 'Conflict\nDetection'),
        ('Conflict\nDetection', 'LLM\nEnsemble'),
        ('LLM\nEnsemble', 'Primary\nLLM'),
        ('LLM\nEnsemble', 'Validator\nLLM'),
        ('LLM\nEnsemble', 'Safety\nLLM'),
        ('Primary\nLLM', 'Hallucination\nDetector'),
        ('Validator\nLLM', 'Hallucination\nDetector'),
        ('Safety\nLLM', 'Hallucination\nDetector'),
        ('Hallucination\nDetector', 'Policy\nChecker'),
        ('Policy\nChecker', 'Conflict\nSolver'),
        ('Conflict\nSolver', 'Safety\nQuantifier'),
        ('Safety\nQuantifier', 'Command\nExecution'),
        ('Command\nExecution', 'Experience\nReplay'),
        ('Experience\nReplay', 'LLM\nEnsemble'),  # Feedback loop
    ]
    
    # Add edges to graph
    G.add_edges_from(edges)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Get positions
    pos = {node: props['pos'] for node, props in nodes.items()}
    
    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', width=2, alpha=0.7)
    
    # Draw nodes with different shapes and colors
    for node, props in nodes.items():
        x, y = props['pos']
        color = props['color']
        shape = props['shape']
        
        if shape == 'box':
            # Rectangle
            rect = patches.FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4, 
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        elif shape == 'diamond':
            # Diamond
            diamond = patches.FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4,
                                           boxstyle="round,pad=0.05",
                                           facecolor=color, edgecolor='black', linewidth=2,
                                           transform=ax.transData)
            ax.add_patch(diamond)
        else:  # ellipse
            # Ellipse
            ellipse = patches.Ellipse((x, y), 0.8, 0.4, 
                                    facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(ellipse)
        
        # Add text
        ax.text(x, y, node, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add special annotations for distribution shift
    if tier != 'in_distribution':
        ax.text(0, 5, f'Distribution Shift: {tier}', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
               fontsize=10, fontweight='bold', ha='center')
    
    # Add feedback loop annotation
    ax.annotate('Experience\nFeedback Loop', xy=(6, 3), xytext=(4, 1),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                             color='purple', lw=2, alpha=0.7),
               fontsize=9, color='purple', fontweight='bold', ha='center')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Data Processing'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', 
                  markersize=10, label='Decision Points'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='palegreen', 
                  markersize=10, label='LLM Components'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral', 
                  markersize=10, label='Action Generation'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Formatting
    ax.set_xlim(-1, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    plt.suptitle(f'LLM-ATC-HAL Conflict Resolution Flowchart\n'
                f'Simulation: {sim_id} | Tier: {tier}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"cr_flowchart_{tier}_{sim_id}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def plot_tier_comparison(df: pd.DataFrame, output_dir: str = "thesis_results") -> str:
    """
    Plot comparison across distribution shift tiers.
    
    Args:
        df: Complete experiment results dataframe
        output_dir: Output directory for plots
        
    Returns:
        Path to saved comparison plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution Shift Tier Comparison', fontsize=16, fontweight='bold')
    
    # 1. Safety Score by Tier
    df.boxplot(column='safety_score', by='tier', ax=axes[0,0])
    axes[0,0].set_title('Safety Score Distribution by Tier')
    axes[0,0].set_xlabel('Distribution Shift Tier')
    axes[0,0].set_ylabel('Safety Score')
    
    # 2. Hallucination Rate by Tier
    hall_rate = df.groupby('tier')['hallucination_detected'].mean()
    hall_rate.plot(kind='bar', ax=axes[0,1], color='orange', alpha=0.7)
    axes[0,1].set_title('Hallucination Detection Rate by Tier')
    axes[0,1].set_ylabel('Detection Rate')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Runtime by Tier
    df.boxplot(column='runtime_s', by='tier', ax=axes[1,0])
    axes[1,0].set_title('Runtime Distribution by Tier')
    axes[1,0].set_xlabel('Distribution Shift Tier')
    axes[1,0].set_ylabel('Runtime (seconds)')
    
    # 4. ICAO Compliance by Tier
    compliance_rate = df.groupby('tier')['icao_compliant'].mean()
    compliance_rate.plot(kind='bar', ax=axes[1,1], color='green', alpha=0.7)
    axes[1,1].set_title('ICAO Compliance Rate by Tier')
    axes[1,1].set_ylabel('Compliance Rate')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    filename = "tier_comparison.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def create_visualization_summary(output_dir: str = "thesis_results") -> str:
    """
    Create a summary plot showing all generated visualizations.
    
    Args:
        output_dir: Directory containing visualization files
        
    Returns:
        Path to summary image
    """
    output_path = Path(output_dir)
    
    # Find all generated plots
    cd_plots = list(output_path.glob("cd_timeline_*.png"))
    cr_plots = list(output_path.glob("cr_flowchart_*.png"))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create summary text
    summary_text = f"""
LLM-ATC-HAL Visualization Summary
================================

Generated Visualizations:
• {len(cd_plots)} Conflict Detection Timelines
• {len(cr_plots)} Conflict Resolution Flowcharts
• 1 Tier Comparison Analysis

Files created in: {output_path}

Timeline Plots:
{chr(10).join([f"  - {p.name}" for p in cd_plots[:5]])}
{f"  ... and {len(cd_plots)-5} more" if len(cd_plots) > 5 else ""}

Flowchart Plots:
{chr(10).join([f"  - {p.name}" for p in cr_plots[:5]])}
{f"  ... and {len(cr_plots)-5} more" if len(cr_plots) > 5 else ""}

For detailed interpretation, see README_visualisation.md
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save summary
    filepath = output_path / "visualization_summary.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(filepath)
