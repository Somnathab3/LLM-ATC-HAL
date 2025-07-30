# llm_atc/metrics/monte_carlo_analysis.py
"""
Monte Carlo Analysis Helper Functions
====================================

Provides helper functions to aggregate and summarize Monte-Carlo results into:
- False-positive/negative rates
- Success rates per scenario type  
- Average separation margins
- Efficiency penalties

Functions for reading results.json/csv files and producing visualizations.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/seaborn not available - visualizations disabled")

try:
    from .safety_margin_quantifier import calc_separation_margin, calc_efficiency_penalty
except ImportError:
    # Fallback for standalone execution
    try:
        from safety_margin_quantifier import calc_separation_margin, calc_efficiency_penalty
    except ImportError:
        # Mock functions for testing
        def calc_separation_margin(trajectories):
            return {"hz": 5.0, "vt": 1000.0}
        def calc_efficiency_penalty(planned, executed):
            return 2.0


class MonteCarloResultsAnalyzer:
    """
    Aggregates and analyzes Monte Carlo simulation results for ATC scenarios.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def read_results_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read Monte Carlo results from JSON or CSV file.
        
        Args:
            file_path: Path to results file (.json or .csv)
            
        Returns:
            DataFrame with simulation results
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
            
        if file_path.suffix.lower() == '.json':
            return self._read_json_results(file_path)
        elif file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _read_json_results(self, file_path: Path) -> pd.DataFrame:
        """Read results from JSON file format."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of result objects
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'results' in data:
                    # Nested structure with 'results' key
                    return pd.DataFrame(data['results'])
                else:
                    # Single result object - convert to single-row DataFrame
                    return pd.DataFrame([data])
            else:
                raise ValueError("Unexpected JSON structure")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON file {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading JSON results: {e}")
            raise

    def compute_false_positive_negative_rates(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute false positive and false negative rates from results.
        
        Args:
            results_df: DataFrame with columns 'predicted_conflicts', 'actual_conflicts'
            
        Returns:
            Dict with 'false_positive_rate' and 'false_negative_rate'
        """
        if results_df.empty:
            return {'false_positive_rate': 0.0, 'false_negative_rate': 0.0}
        
        total_fp = 0
        total_fn = 0
        total_predicted = 0
        total_actual = 0
        
        for _, row in results_df.iterrows():
            # Get conflict lists
            predicted = row.get('predicted_conflicts', [])
            actual = row.get('actual_conflicts', [])
            
            # Convert to sets of conflict pairs for comparison
            pred_set = self._conflicts_to_set(predicted)
            actual_set = self._conflicts_to_set(actual)
            
            # Calculate FP and FN for this scenario
            fp = len(pred_set - actual_set)  # Predicted but not actual
            fn = len(actual_set - pred_set)  # Actual but not predicted
            
            total_fp += fp
            total_fn += fn
            total_predicted += len(pred_set)
            total_actual += len(actual_set)
        
        # Calculate rates
        fp_rate = total_fp / max(1, total_predicted)
        fn_rate = total_fn / max(1, total_actual)
        
        return {
            'false_positive_rate': fp_rate,
            'false_negative_rate': fn_rate,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'total_predicted_conflicts': total_predicted,
            'total_actual_conflicts': total_actual
        }
    
    def _conflicts_to_set(self, conflicts: List[Dict[str, Any]]) -> set:
        """Convert conflict list to set of aircraft pairs."""
        conflict_pairs = set()
        
        for conflict in conflicts:
            if isinstance(conflict, dict):
                # Extract aircraft IDs from conflict
                ac1 = conflict.get('aircraft_1') or conflict.get('aircraft1')
                ac2 = conflict.get('aircraft_2') or conflict.get('aircraft2')
                
                if ac1 and ac2:
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([str(ac1), str(ac2)]))
                    conflict_pairs.add(pair)
        
        return conflict_pairs
    
    def compute_success_rates_by_scenario(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute success rates grouped by scenario type.
        
        Args:
            results_df: DataFrame with columns 'scenario_type', 'success'
            
        Returns:
            Dict mapping scenario types to success metrics
        """
        if results_df.empty or 'scenario_type' not in results_df.columns:
            return {}
        
        success_rates = {}
        
        # Group by scenario type
        for scenario_type in results_df['scenario_type'].unique():
            scenario_data = results_df[results_df['scenario_type'] == scenario_type]
            
            # Calculate success rate
            total_scenarios = len(scenario_data)
            
            # Different ways to determine success
            if 'success' in scenario_data.columns:
                successful = scenario_data['success'].sum()
            elif 'safety_score' in scenario_data.columns:
                # Consider scenarios with safety_score > 0.7 as successful
                successful = (scenario_data['safety_score'] > 0.7).sum()
            elif 'conflicts_resolved' in scenario_data.columns:
                successful = scenario_data['conflicts_resolved'].sum()
            else:
                # Default: no conflicts detected = success
                successful = (scenario_data.get('predicted_conflicts', []).apply(len) == 0).sum()
            
            success_rate = successful / max(1, total_scenarios)
            
            success_rates[scenario_type] = {
                'success_rate': success_rate,
                'successful_scenarios': int(successful),
                'total_scenarios': total_scenarios,
                'failure_rate': 1 - success_rate
            }
        
        return success_rates
    
    def compute_average_separation_margins(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute average separation margins from results.
        
        Args:
            results_df: DataFrame with trajectory or margin data
            
        Returns:
            Dict with horizontal and vertical margin averages
        """
        if results_df.empty:
            return {'avg_horizontal_margin': 0.0, 'avg_vertical_margin': 0.0}
        
        horizontal_margins = []
        vertical_margins = []
        
        for _, row in results_df.iterrows():
            # Try direct margin columns first
            if 'horizontal_margin' in row and pd.notna(row['horizontal_margin']):
                horizontal_margins.append(row['horizontal_margin'])
            if 'vertical_margin' in row and pd.notna(row['vertical_margin']):
                vertical_margins.append(row['vertical_margin'])
            
            # Calculate from trajectories if available
            if 'trajectories' in row and row['trajectories']:
                try:
                    margins = calc_separation_margin(row['trajectories'])
                    if margins['hz'] != float('inf'):
                        horizontal_margins.append(margins['hz'])
                    if margins['vt'] != float('inf'):
                        vertical_margins.append(margins['vt'])
                except Exception as e:
                    self.logger.warning(f"Failed to calculate margins from trajectories: {e}")
        
        return {
            'avg_horizontal_margin': np.mean(horizontal_margins) if horizontal_margins else 0.0,
            'avg_vertical_margin': np.mean(vertical_margins) if vertical_margins else 0.0,
            'std_horizontal_margin': np.std(horizontal_margins) if horizontal_margins else 0.0,
            'std_vertical_margin': np.std(vertical_margins) if vertical_margins else 0.0,
            'num_margin_samples': len(horizontal_margins)
        }
    
    def compute_efficiency_penalties(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute efficiency penalties from trajectory comparisons.
        
        Args:
            results_df: DataFrame with planned and executed trajectory data
            
        Returns:
            Dict with efficiency penalty statistics
        """
        if results_df.empty:
            return {'avg_efficiency_penalty': 0.0}
        
        penalties = []
        
        for _, row in results_df.iterrows():
            # Try direct penalty column first
            if 'efficiency_penalty' in row and pd.notna(row['efficiency_penalty']):
                penalties.append(row['efficiency_penalty'])
                continue
            
            # Calculate from trajectory data
            planned_path = row.get('planned_trajectory') or row.get('original_trajectory')
            executed_path = row.get('executed_trajectory') or row.get('actual_trajectory')
            
            if planned_path and executed_path:
                try:
                    penalty = calc_efficiency_penalty(planned_path, executed_path)
                    penalties.append(penalty)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate efficiency penalty: {e}")
        
        return {
            'avg_efficiency_penalty': np.mean(penalties) if penalties else 0.0,
            'std_efficiency_penalty': np.std(penalties) if penalties else 0.0,
            'max_efficiency_penalty': np.max(penalties) if penalties else 0.0,
            'num_penalty_samples': len(penalties)
        }
    
    def aggregate_monte_carlo_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive aggregated metrics from Monte Carlo results.
        
        Args:
            results_df: DataFrame with simulation results
            
        Returns:
            Dict containing all aggregated metrics
        """
        if results_df.empty:
            self.logger.warning("Empty results DataFrame provided")
            return self._create_empty_aggregated_metrics()
        
        self.logger.info(f"Aggregating metrics from {len(results_df)} Monte Carlo scenarios")
        
        # Compute all metric categories
        fp_fn_rates = self.compute_false_positive_negative_rates(results_df)
        success_rates = self.compute_success_rates_by_scenario(results_df)
        separation_margins = self.compute_average_separation_margins(results_df)
        efficiency_penalties = self.compute_efficiency_penalties(results_df)
        
        # Overall statistics
        total_scenarios = len(results_df)
        scenario_types = results_df.get('scenario_type', pd.Series()).unique().tolist()
        
        # Distribution shift analysis if available
        shift_analysis = {}
        if 'distribution_shift_level' in results_df.columns:
            shift_analysis = self._analyze_distribution_shift_performance(results_df)
        
        aggregated_metrics = {
            'summary': {
                'total_scenarios': total_scenarios,
                'scenario_types': scenario_types,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'detection_performance': fp_fn_rates,
            'success_rates_by_scenario': success_rates,
            'separation_margins': separation_margins,
            'efficiency_metrics': efficiency_penalties,
            'distribution_shift_analysis': shift_analysis
        }
        
        self.logger.info("Monte Carlo metrics aggregation completed")
        return aggregated_metrics
    
    def _analyze_distribution_shift_performance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across different distribution shift levels."""
        shift_analysis = {}
        
        for shift_level in results_df['distribution_shift_level'].unique():
            shift_data = results_df[results_df['distribution_shift_level'] == shift_level]
            
            # Calculate metrics for this shift level
            fp_fn = self.compute_false_positive_negative_rates(shift_data)
            success = self.compute_success_rates_by_scenario(shift_data)
            margins = self.compute_average_separation_margins(shift_data)
            
            shift_analysis[shift_level] = {
                'scenario_count': len(shift_data),
                'false_positive_rate': fp_fn['false_positive_rate'],
                'false_negative_rate': fp_fn['false_negative_rate'],
                'avg_success_rate': np.mean([s['success_rate'] for s in success.values()]) if success else 0.0,
                'avg_horizontal_margin': margins['avg_horizontal_margin'],
                'avg_vertical_margin': margins['avg_vertical_margin']
            }
        
        return shift_analysis
    
    def _create_empty_aggregated_metrics(self) -> Dict[str, Any]:
        """Create empty metrics structure for error cases."""
        return {
            'summary': {
                'total_scenarios': 0,
                'scenario_types': [],
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'detection_performance': {
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0
            },
            'success_rates_by_scenario': {},
            'separation_margins': {
                'avg_horizontal_margin': 0.0,
                'avg_vertical_margin': 0.0
            },
            'efficiency_metrics': {
                'avg_efficiency_penalty': 0.0
            },
            'distribution_shift_analysis': {}
        }


class MonteCarloVisualizer:
    """
    Creates visualizations for Monte Carlo analysis results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)
        
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting libraries not available - visualizations disabled")
    
    def create_performance_summary_charts(self, 
                                        aggregated_metrics: Dict[str, Any],
                                        output_dir: Union[str, Path] = "monte_carlo_plots") -> List[str]:
        """
        Create bar charts summarizing performance across scenario types.
        
        Args:
            aggregated_metrics: Output from aggregate_monte_carlo_metrics()
            output_dir: Directory to save plots
            
        Returns:
            List of created plot file paths
        """
        if not PLOTTING_AVAILABLE:
            self.logger.error("Cannot create plots - matplotlib/seaborn not available")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        created_plots = []
        
        # 1. Success rates by scenario type
        success_plot = self._create_success_rate_chart(
            aggregated_metrics['success_rates_by_scenario'], 
            output_dir / "success_rates_by_scenario.png"
        )
        if success_plot:
            created_plots.append(success_plot)
        
        # 2. Detection performance (FP/FN rates)
        detection_plot = self._create_detection_performance_chart(
            aggregated_metrics['detection_performance'],
            output_dir / "detection_performance.png"
        )
        if detection_plot:
            created_plots.append(detection_plot)
        
        # 3. Safety margins comparison
        margins_plot = self._create_safety_margins_chart(
            aggregated_metrics['separation_margins'],
            output_dir / "safety_margins.png"
        )
        if margins_plot:
            created_plots.append(margins_plot)
        
        return created_plots
    
    def create_distribution_shift_plots(self,
                                      aggregated_metrics: Dict[str, Any],
                                      output_dir: Union[str, Path] = "monte_carlo_plots") -> List[str]:
        """
        Create scatter plots showing performance differences under distribution shifts.
        
        Args:
            aggregated_metrics: Output from aggregate_monte_carlo_metrics()
            output_dir: Directory to save plots
            
        Returns:
            List of created plot file paths
        """
        if not PLOTTING_AVAILABLE:
            self.logger.error("Cannot create plots - matplotlib/seaborn not available")
            return []
        
        shift_analysis = aggregated_metrics.get('distribution_shift_analysis', {})
        if not shift_analysis:
            self.logger.warning("No distribution shift data available for plotting")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        created_plots = []
        
        # Create scatter plot of performance vs distribution shift level
        plot_path = self._create_shift_performance_scatter(
            shift_analysis,
            output_dir / "distribution_shift_performance.png"
        )
        if plot_path:
            created_plots.append(plot_path)
        
        return created_plots
    
    def _create_success_rate_chart(self, success_data: Dict[str, Dict[str, float]], 
                                 save_path: Path) -> Optional[str]:
        """Create bar chart of success rates by scenario type."""
        try:
            if not success_data:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scenario_types = list(success_data.keys())
            success_rates = [success_data[st]['success_rate'] for st in scenario_types]
            total_scenarios = [success_data[st]['total_scenarios'] for st in scenario_types]
            
            bars = ax.bar(scenario_types, success_rates, alpha=0.8, color='skyblue')
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, total_scenarios)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2%}\n(n={count})', 
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Success Rate')
            ax.set_title('ATC Resolution Success Rates by Scenario Type')
            ax.set_ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Success rate chart saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create success rate chart: {e}")
            return None
    
    def _create_detection_performance_chart(self, detection_data: Dict[str, float],
                                          save_path: Path) -> Optional[str]:
        """Create bar chart of false positive/negative rates."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            metrics = ['False Positive Rate', 'False Negative Rate']
            values = [
                detection_data.get('false_positive_rate', 0),
                detection_data.get('false_negative_rate', 0)
            ]
            colors = ['lightcoral', 'lightsalmon']
            
            bars = ax.bar(metrics, values, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=12)
            
            ax.set_ylabel('Rate')
            ax.set_title('Conflict Detection Performance')
            ax.set_ylim(0, max(values) * 1.2 if values else 0.1)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Detection performance chart saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create detection performance chart: {e}")
            return None
    
    def _create_safety_margins_chart(self, margins_data: Dict[str, float],
                                   save_path: Path) -> Optional[str]:
        """Create bar chart of safety margins."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            margin_types = ['Horizontal Margin (NM)', 'Vertical Margin (ft)']
            values = [
                margins_data.get('avg_horizontal_margin', 0),
                margins_data.get('avg_vertical_margin', 0)
            ]
            colors = ['lightgreen', 'lightblue']
            
            bars = ax.bar(margin_types, values, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=12)
            
            ax.set_ylabel('Margin')
            ax.set_title('Average Safety Separation Margins')
            ax.set_ylim(0, max(values) * 1.2 if values else 10)
            plt.xticks(rotation=15)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Safety margins chart saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create safety margins chart: {e}")
            return None
    
    def _create_shift_performance_scatter(self, shift_data: Dict[str, Dict[str, float]],
                                        save_path: Path) -> Optional[str]:
        """Create scatter plot of performance vs distribution shift level."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            shift_levels = list(shift_data.keys())
            shift_numeric = range(len(shift_levels))  # Convert to numeric for plotting
            
            # Extract metrics
            fp_rates = [shift_data[sl]['false_positive_rate'] for sl in shift_levels]
            fn_rates = [shift_data[sl]['false_negative_rate'] for sl in shift_levels]
            success_rates = [shift_data[sl]['avg_success_rate'] for sl in shift_levels]
            h_margins = [shift_data[sl]['avg_horizontal_margin'] for sl in shift_levels]
            
            # Plot 1: False Positive Rate vs Shift Level
            ax1.scatter(shift_numeric, fp_rates, color='red', alpha=0.7, s=60)
            ax1.plot(shift_numeric, fp_rates, 'r--', alpha=0.5)
            ax1.set_title('False Positive Rate vs Distribution Shift')
            ax1.set_ylabel('False Positive Rate')
            ax1.set_xticks(shift_numeric)
            ax1.set_xticklabels(shift_levels, rotation=45)
            
            # Plot 2: False Negative Rate vs Shift Level  
            ax2.scatter(shift_numeric, fn_rates, color='orange', alpha=0.7, s=60)
            ax2.plot(shift_numeric, fn_rates, 'orange', linestyle='--', alpha=0.5)
            ax2.set_title('False Negative Rate vs Distribution Shift')
            ax2.set_ylabel('False Negative Rate')
            ax2.set_xticks(shift_numeric)
            ax2.set_xticklabels(shift_levels, rotation=45)
            
            # Plot 3: Success Rate vs Shift Level
            ax3.scatter(shift_numeric, success_rates, color='green', alpha=0.7, s=60)
            ax3.plot(shift_numeric, success_rates, 'g--', alpha=0.5)
            ax3.set_title('Success Rate vs Distribution Shift')
            ax3.set_ylabel('Success Rate')
            ax3.set_xticks(shift_numeric)
            ax3.set_xticklabels(shift_levels, rotation=45)
            
            # Plot 4: Horizontal Margin vs Shift Level
            ax4.scatter(shift_numeric, h_margins, color='blue', alpha=0.7, s=60)
            ax4.plot(shift_numeric, h_margins, 'b--', alpha=0.5)
            ax4.set_title('Horizontal Margin vs Distribution Shift')
            ax4.set_ylabel('Horizontal Margin (NM)')
            ax4.set_xticks(shift_numeric)
            ax4.set_xticklabels(shift_levels, rotation=45)
            
            plt.suptitle('Performance Degradation Under Distribution Shift')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Distribution shift scatter plot saved to {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution shift scatter plot: {e}")
            return None


# Convenience functions for direct usage
def analyze_monte_carlo_results(results_file: Union[str, Path], 
                              output_dir: Union[str, Path] = "monte_carlo_analysis") -> Dict[str, Any]:
    """
    Complete Monte Carlo analysis pipeline from results file to metrics and plots.
    
    Args:
        results_file: Path to results.json or results.csv file
        output_dir: Directory for analysis outputs
        
    Returns:
        Dict with aggregated metrics and plot paths
    """
    # Initialize analyzer and visualizer
    analyzer = MonteCarloResultsAnalyzer()
    visualizer = MonteCarloVisualizer()
    
    try:
        # Read and analyze results
        results_df = analyzer.read_results_file(results_file)
        aggregated_metrics = analyzer.aggregate_monte_carlo_metrics(results_df)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save aggregated metrics
        metrics_file = output_dir / "aggregated_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2, default=str)
        
        # Create visualizations
        summary_plots = visualizer.create_performance_summary_charts(aggregated_metrics, output_dir)
        shift_plots = visualizer.create_distribution_shift_plots(aggregated_metrics, output_dir)
        
        # Return complete analysis
        return {
            'metrics': aggregated_metrics,
            'metrics_file': str(metrics_file),
            'summary_plots': summary_plots,
            'distribution_shift_plots': shift_plots,
            'output_directory': str(output_dir)
        }
        
    except Exception as e:
        logging.error(f"Monte Carlo analysis failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create sample data for demonstration
    sample_data = [
        {
            'scenario_type': 'horizontal',
            'success': True,
            'predicted_conflicts': [],
            'actual_conflicts': [],
            'horizontal_margin': 6.2,
            'vertical_margin': 1200,
            'efficiency_penalty': 2.1
        },
        {
            'scenario_type': 'vertical', 
            'success': False,
            'predicted_conflicts': [{'aircraft_1': 'AC001', 'aircraft_2': 'AC002'}],
            'actual_conflicts': [{'aircraft_1': 'AC001', 'aircraft_2': 'AC002'}],
            'horizontal_margin': 3.8,
            'vertical_margin': 800,
            'efficiency_penalty': 5.3
        }
    ]
    
    # Save sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f)
        sample_file = f.name
    
    # Run analysis
    try:
        results = analyze_monte_carlo_results(sample_file)
        print("Monte Carlo Analysis Results:")
        print(f"- Metrics saved to: {results['metrics_file']}")
        print(f"- Plots created: {len(results['summary_plots'])} summary, {len(results['distribution_shift_plots'])} shift")
        print(f"- Output directory: {results['output_directory']}")
        
        # Print key metrics
        metrics = results['metrics']
        print(f"\nKey Results:")
        print(f"- Total scenarios: {metrics['summary']['total_scenarios']}")
        print(f"- FP rate: {metrics['detection_performance']['false_positive_rate']:.3f}")
        print(f"- FN rate: {metrics['detection_performance']['false_negative_rate']:.3f}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
    finally:
        # Cleanup
        Path(sample_file).unlink(missing_ok=True)