# analysis/metrics.py
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import math
from pathlib import Path

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')  # Set a default style
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analysis.hallucination_taxonomy import analyze_hallucinations_in_log

def compute_metrics(log_file):
    """Compute hallucination and performance metrics from simulation logs."""
    try:
        # Get detailed hallucination analysis
        hallucination_analysis = analyze_hallucinations_in_log(log_file)
        
        # Read log file line by line and extract JSON entries
        data = []
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Try to extract JSON from log entries
                    if line.startswith('{') and line.endswith('}'):
                        # Direct JSON line
                        try:
                            entry = json.loads(line)
                            if 'best_by_llm' in entry and 'baseline_best' in entry:
                                data.append(entry)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse JSON line: {line}. Error: {e}")
        
        if not data:
            logging.warning("No valid JSON data found in log file")
            return create_empty_metrics()
        
        df = pd.DataFrame(data)
        logging.info(f"Found {len(data)} conflict resolution entries")
        
        # Calculate hallucination rate from detailed analysis
        total_hallucination_events = sum(hallucination_analysis['hallucination_breakdown'].values())
        
        metrics = {
            'scenarios_run': len(df['scenario'].unique()) if 'scenario' in df.columns else 0,
            'conflicts_detected': len(df),
            'resolutions_attempted': len(df),
            'hallucination_rate': total_hallucination_events / max(len(df), 1),
            'hallucination_breakdown': hallucination_analysis['hallucination_breakdown'],
            'hallucination_events': total_hallucination_events,
            'policy_violations': 0,
            'llm_errors': 0,
            'safety_margin_differences': [],
            'response_validity': {'valid': 0, 'invalid': 0},
            'maneuver_type_distribution': defaultdict(int),
            'avg_safety_score_llm': 0.0,
            'avg_safety_score_baseline': 0.0
        }
        safety_scores_llm = []
        safety_scores_baseline = []
        
        for _, row in df.iterrows():
            best_by_llm = row.get('best_by_llm')
            baseline_best = row.get('baseline_best')
            
            # Check for hallucination indicators
            if best_by_llm != baseline_best:
                metrics['hallucination_events'] += 1
            
            # Analyze safety scores
            if isinstance(best_by_llm, dict):
                llm_safety = best_by_llm.get('safety_score', 0.5)
                safety_scores_llm.append(llm_safety)
                
                # Track maneuver types
                maneuver_type = best_by_llm.get('type', 'unknown')
                metrics['maneuver_type_distribution'][maneuver_type] += 1
                
                # Check for policy violations
                if 'policy_violations' in best_by_llm:
                    metrics['policy_violations'] += len(best_by_llm['policy_violations'])
                
                metrics['response_validity']['valid'] += 1
            else:
                metrics['response_validity']['invalid'] += 1
                logging.warning(f"Invalid LLM response format: {best_by_llm}")
            
            if isinstance(baseline_best, dict):
                baseline_safety = baseline_best.get('safety_score', 0.5)
                safety_scores_baseline.append(baseline_safety)
                
                # Calculate safety margin difference
                if isinstance(best_by_llm, dict):
                    margin_diff = llm_safety - baseline_safety
                    metrics['safety_margin_differences'].append(margin_diff)
        
        # Calculate averages
        if safety_scores_llm:
            metrics['avg_safety_score_llm'] = np.mean(safety_scores_llm)
        if safety_scores_baseline:
            metrics['avg_safety_score_baseline'] = np.mean(safety_scores_baseline)
        
        # Calculate hallucination rate
        metrics['hallucination_rate'] = (
            total_hallucination_events / max(len(df), 1)
        )
        
        # Calculate average safety margin difference
        if metrics['safety_margin_differences']:
            metrics['avg_safety_margin_diff'] = np.mean(metrics['safety_margin_differences'])
        else:
            metrics['avg_safety_margin_diff'] = 0.0
        
        # Convert defaultdict to regular dict for JSON serialization
        metrics['maneuver_type_distribution'] = dict(metrics['maneuver_type_distribution'])
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        return create_empty_metrics()

def create_empty_metrics():
    """Create empty metrics structure for error cases."""
    return {
        'scenarios_run': 0,
        'conflicts_detected': 0,
        'resolutions_attempted': 0,
        'hallucination_rate': 0.0,
        'hallucination_breakdown': {
            'fabrication': 0,
            'omission': 0,
            'irrelevancy': 0,
            'contradiction': 0
        },
        'hallucination_events': 0,
        'policy_violations': 0,
        'llm_errors': 0,
        'safety_margin_differences': [],
        'avg_safety_margin_diff': 0.0,
        'response_validity': {'valid': 0, 'invalid': 0},
        'maneuver_type_distribution': {},
        'avg_safety_score_llm': 0.0,
        'avg_safety_score_baseline': 0.0
    }

def print_metrics_summary(metrics):
    """Print a formatted summary of the metrics."""
    print("\n" + "="*50)
    print("ATC HALLUCINATION TEST RESULTS")
    print("="*50)
    print(f"Scenarios Run: {metrics['scenarios_run']}")
    print(f"Total Conflicts Processed: {metrics['conflicts_detected']}")
    print(f"Resolutions Attempted: {metrics['resolutions_attempted']}")
    print(f"Hallucination Events: {metrics['hallucination_events']}")
    print(f"Hallucination Rate: {metrics['hallucination_rate']:.2%}")
    print(f"Policy Violations: {metrics['policy_violations']}")
    print(f"LLM Errors: {metrics['llm_errors']}")
    print(f"Average Safety Score (LLM): {metrics['avg_safety_score_llm']:.3f}")
    print(f"Average Safety Score (Baseline): {metrics['avg_safety_score_baseline']:.3f}")
    print(f"Average Safety Margin Difference: {metrics['avg_safety_margin_diff']:.3f}")
    print(f"Valid Responses: {metrics['response_validity']['valid']}")
    print(f"Invalid Responses: {metrics['response_validity']['invalid']}")
    
    if metrics.get('hallucination_breakdown'):
        print("\nHallucination Breakdown:")
        for h_type, count in metrics['hallucination_breakdown'].items():
            print(f"  {h_type.title()}: {count}")
    
    if metrics['maneuver_type_distribution']:
        print("\nManeuver Type Distribution:")
        for maneuver_type, count in metrics['maneuver_type_distribution'].items():
            print(f"  {maneuver_type}: {count}")
    
    print("="*50)

if __name__ == "__main__":
    # Test the metrics computation
    metrics = compute_metrics('simulation.log')
    print_metrics_summary(metrics)


def calc_fp_fn(pred_conflicts: List[Dict[str, Any]], 
               gt_conflicts: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Calculate false positives and false negatives in conflict detection.
    
    Args:
        pred_conflicts: List of predicted conflicts with fields:
                       [{'id1': str, 'id2': str, 'time': float, ...}]
        gt_conflicts: List of ground truth conflicts with same structure
        
    Returns:
        Tuple of (false_positives, false_negatives)
    """
    if not pred_conflicts and not gt_conflicts:
        return 0, 0
    
    # Create sets of conflict pairs for comparison
    def normalize_conflict_pair(conflict: Dict[str, Any]) -> Tuple[str, str]:
        """Normalize conflict pair to (id1, id2) with consistent ordering"""
        id1, id2 = conflict['id1'], conflict['id2']
        return (min(id1, id2), max(id1, id2))
    
    # Normalize time windows for matching (±30 seconds tolerance)
    time_tolerance = 30.0
    
    pred_pairs = set()
    gt_pairs = set()
    
    # Extract prediction pairs
    for pred in pred_conflicts:
        pair = normalize_conflict_pair(pred)
        time = pred.get('time_to_conflict', pred.get('time', 0))
        pred_pairs.add((pair[0], pair[1], round(time / time_tolerance)))
    
    # Extract ground truth pairs
    for gt in gt_conflicts:
        pair = normalize_conflict_pair(gt)
        time = gt.get('time_to_conflict', gt.get('time', 0))
        gt_pairs.add((pair[0], pair[1], round(time / time_tolerance)))
    
    # Calculate FP and FN
    false_positives = len(pred_pairs - gt_pairs)
    false_negatives = len(gt_pairs - pred_pairs)
    
    return false_positives, false_negatives


def calc_path_extra(actual_traj: List[Dict[str, Any]], 
                   original_traj: List[Dict[str, Any]]) -> float:
    """
    Calculate extra distance traveled due to conflict resolution.
    
    Args:
        actual_traj: List of actual trajectories after resolution
                    [{'aircraft_id': str, 'path': [{'lat': float, 'lon': float, 'time': float}]}]
        original_traj: List of original planned trajectories
                      Same structure as actual_traj
        
    Returns:
        Total extra distance in nautical miles
    """
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in nautical miles"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in nautical miles
        r_nm = 3440.065
        
        return c * r_nm
    
    def calculate_trajectory_distance(trajectory: Dict[str, Any]) -> float:
        """Calculate total distance for a single trajectory"""
        path = trajectory.get('path', [])
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            distance = haversine_distance(
                p1['lat'], p1['lon'], p2['lat'], p2['lon']
            )
            total_distance += distance
        
        return total_distance
    
    # Create mappings by aircraft ID
    actual_by_id = {traj['aircraft_id']: traj for traj in actual_traj}
    original_by_id = {traj['aircraft_id']: traj for traj in original_traj}
    
    total_extra_distance = 0.0
    
    # Calculate extra distance for each aircraft
    for aircraft_id in set(actual_by_id.keys()) & set(original_by_id.keys()):
        actual_dist = calculate_trajectory_distance(actual_by_id[aircraft_id])
        original_dist = calculate_trajectory_distance(original_by_id[aircraft_id])
        
        extra_distance = max(0, actual_dist - original_dist)
        total_extra_distance += extra_distance
    
    return total_extra_distance


def aggregate_thesis_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate metrics for thesis analysis across distribution shift experiments.
    
    Args:
        df: DataFrame with experiment results containing columns:
            ['tier', 'hallucination_detected', 'fp', 'fn', 'safety_score', 
             'icao_compliant', 'runtime_s', 'extra_nm', 'n_interventions', etc.]
    
    Returns:
        Dictionary with comprehensive aggregated metrics
    """
    if df.empty:
        return {
            'error': 'Empty dataframe provided',
            'total_experiments': 0
        }
    
    # Basic experiment info
    total_experiments = len(df)
    tiers = df['tier'].unique().tolist() if 'tier' in df.columns else []
    
    metrics = {
        'experiment_overview': {
            'total_experiments': total_experiments,
            'distribution_shift_tiers': tiers,
            'tier_counts': df['tier'].value_counts().to_dict() if 'tier' in df.columns else {},
            'complexity_distribution': df['complexity'].value_counts().to_dict() if 'complexity' in df.columns else {}
        },
        
        'hallucination_analysis': {
            'overall_detection_rate': df['hallucination_detected'].mean() if 'hallucination_detected' in df.columns else 0,
            'detection_rate_by_tier': df.groupby('tier')['hallucination_detected'].mean().to_dict() if 'tier' in df.columns else {},
            'detection_rate_by_complexity': df.groupby('complexity')['hallucination_detected'].mean().to_dict() if 'complexity' in df.columns else {}
        },
        
        'conflict_detection_performance': {
            'overall_fp_rate': df['fp'].mean() if 'fp' in df.columns else 0,
            'overall_fn_rate': df['fn'].mean() if 'fn' in df.columns else 0,
            'fp_rate_by_tier': df.groupby('tier')['fp'].mean().to_dict() if 'tier' in df.columns else {},
            'fn_rate_by_tier': df.groupby('tier')['fn'].mean().to_dict() if 'tier' in df.columns else {},
            'precision': _calculate_precision(df),
            'recall': _calculate_recall(df),
            'f1_score': _calculate_f1_score(df)
        },
        
        'safety_performance': {
            'overall_safety_score': df['safety_score'].mean() if 'safety_score' in df.columns else 0,
            'safety_score_by_tier': df.groupby('tier')['safety_score'].mean().to_dict() if 'tier' in df.columns else {},
            'icao_compliance_rate': df['icao_compliant'].mean() if 'icao_compliant' in df.columns else 0,
            'compliance_by_tier': df.groupby('tier')['icao_compliant'].mean().to_dict() if 'tier' in df.columns else {},
            'horizontal_margin_stats': _calculate_stats(df, 'horiz_margin_ft'),
            'vertical_margin_stats': _calculate_stats(df, 'vert_margin_nm')
        },
        
        'efficiency_metrics': {
            'avg_extra_distance_nm': df['extra_nm'].mean() if 'extra_nm' in df.columns else 0,
            'extra_distance_by_tier': df.groupby('tier')['extra_nm'].mean().to_dict() if 'tier' in df.columns else {},
            'avg_interventions': df['n_interventions'].mean() if 'n_interventions' in df.columns else 0,
            'interventions_by_tier': df.groupby('tier')['n_interventions'].mean().to_dict() if 'tier' in df.columns else {},
            'efficiency_correlation': _calculate_efficiency_correlation(df)
        },
        
        'performance_metrics': {
            'avg_runtime_s': df['runtime_s'].mean() if 'runtime_s' in df.columns else 0,
            'runtime_by_tier': df.groupby('tier')['runtime_s'].mean().to_dict() if 'tier' in df.columns else {},
            'avg_response_time_s': df['response_time_s'].mean() if 'response_time_s' in df.columns else 0,
            'avg_detection_time_s': df['detection_time_s'].mean() if 'detection_time_s' in df.columns else 0,
            'llm_confidence_stats': _calculate_stats(df, 'llm_confidence'),
            'ensemble_consensus_stats': _calculate_stats(df, 'ensemble_consensus')
        },
        
        'environmental_impact': {
            'wind_speed_correlation': _calculate_environmental_correlation(df, 'wind_speed_kts'),
            'turbulence_correlation': _calculate_environmental_correlation(df, 'turbulence_intensity'),
            'visibility_correlation': _calculate_environmental_correlation(df, 'visibility_nm'),
            'navigation_error_impact': _calculate_environmental_correlation(df, 'navigation_error_nm')
        },
        
        'statistical_tests': {
            'tier_performance_significance': _perform_tier_significance_tests(df),
            'correlation_matrix': _calculate_correlation_matrix(df),
            'effect_sizes': _calculate_effect_sizes(df)
        }
    }
    
    return metrics


def _calculate_precision(df: pd.DataFrame) -> float:
    """Calculate precision for conflict detection"""
    if 'fp' not in df.columns or 'fn' not in df.columns:
        return 0.0
    
    tp = df['fn'].sum()  # True positives (detected conflicts)
    fp = df['fp'].sum()  # False positives
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def _calculate_recall(df: pd.DataFrame) -> float:
    """Calculate recall for conflict detection"""
    if 'fp' not in df.columns or 'fn' not in df.columns:
        return 0.0
    
    tp = df['fn'].sum()  # True positives
    fn = df['fn'].sum()  # False negatives
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def _calculate_f1_score(df: pd.DataFrame) -> float:
    """Calculate F1 score for conflict detection"""
    precision = _calculate_precision(df)
    recall = _calculate_recall(df)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def _calculate_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate basic statistics for a column"""
    if column not in df.columns:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    series = df[column].dropna()
    if len(series) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    return {
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'median': float(series.median())
    }


def _calculate_efficiency_correlation(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate correlations between efficiency metrics"""
    efficiency_cols = ['extra_nm', 'n_interventions', 'runtime_s', 'safety_score']
    existing_cols = [col for col in efficiency_cols if col in df.columns]
    
    if len(existing_cols) < 2:
        return {}
    
    corr_matrix = df[existing_cols].corr()
    
    # Extract key correlations
    correlations = {}
    if 'extra_nm' in existing_cols and 'safety_score' in existing_cols:
        correlations['extra_distance_safety'] = float(corr_matrix.loc['extra_nm', 'safety_score'])
    
    if 'n_interventions' in existing_cols and 'safety_score' in existing_cols:
        correlations['interventions_safety'] = float(corr_matrix.loc['n_interventions', 'safety_score'])
    
    if 'runtime_s' in existing_cols and 'safety_score' in existing_cols:
        correlations['runtime_safety'] = float(corr_matrix.loc['runtime_s', 'safety_score'])
    
    return correlations


def _calculate_environmental_correlation(df: pd.DataFrame, env_column: str) -> Dict[str, float]:
    """Calculate correlation between environmental factor and performance metrics"""
    if env_column not in df.columns:
        return {}
    
    performance_cols = ['safety_score', 'hallucination_detected', 'icao_compliant', 'extra_nm']
    correlations = {}
    
    for perf_col in performance_cols:
        if perf_col in df.columns:
            corr = df[env_column].corr(df[perf_col])
            if not np.isnan(corr):
                correlations[f'{env_column}_{perf_col}'] = float(corr)
    
    return correlations


def _perform_tier_significance_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical significance tests between tiers"""
    if 'tier' not in df.columns:
        return {}
    
    try:
        from scipy import stats
        
        tiers = df['tier'].unique()
        if len(tiers) < 2:
            return {}
        
        # Compare safety scores between tiers
        significance_tests = {}
        
        if 'safety_score' in df.columns:
            tier_groups = [df[df['tier'] == tier]['safety_score'].dropna() for tier in tiers]
            
            if len(tier_groups) >= 2 and all(len(group) > 0 for group in tier_groups):
                # ANOVA for multiple groups
                f_stat, p_value = stats.f_oneway(*tier_groups)
                significance_tests['safety_score_anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        return significance_tests
        
    except ImportError:
        return {'error': 'scipy not available for significance tests'}


def _calculate_correlation_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate correlation matrix for key metrics"""
    key_metrics = ['safety_score', 'hallucination_detected', 'fp', 'fn', 
                   'extra_nm', 'n_interventions', 'runtime_s', 'icao_compliant']
    
    existing_metrics = [col for col in key_metrics if col in df.columns]
    
    if len(existing_metrics) < 2:
        return {}
    
    corr_matrix = df[existing_metrics].corr()
    
    # Convert to nested dictionary format
    correlation_dict = {}
    for i, col1 in enumerate(existing_metrics):
        correlation_dict[col1] = {}
        for j, col2 in enumerate(existing_metrics):
            corr_value = corr_matrix.iloc[i, j]
            if not np.isnan(corr_value):
                correlation_dict[col1][col2] = float(corr_value)
    
    return correlation_dict


def _calculate_effect_sizes(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate effect sizes for distribution shift impact"""
    if 'tier' not in df.columns:
        return {}
    
    effect_sizes = {}
    
    # Calculate Cohen's d for safety score between in_distribution and extreme_shift
    if 'safety_score' in df.columns:
        in_dist = df[df['tier'] == 'in_distribution']['safety_score'].dropna()
        extreme = df[df['tier'] == 'extreme_shift']['safety_score'].dropna()
        
        if len(in_dist) > 0 and len(extreme) > 0:
            # Cohen's d
            pooled_std = np.sqrt(((len(in_dist) - 1) * in_dist.var() + 
                                (len(extreme) - 1) * extreme.var()) / 
                               (len(in_dist) + len(extreme) - 2))
            
            if pooled_std > 0:
                cohens_d = (in_dist.mean() - extreme.mean()) / pooled_std
                effect_sizes['safety_score_cohens_d'] = float(cohens_d)
    
    return effect_sizes


def generate_thesis_report(results_file: str, output_dir: str = "thesis_results") -> str:
    """
    Generate comprehensive thesis report with tables and plots.
    
    Args:
        results_file: Path to parquet results file
        output_dir: Output directory for thesis results
        
    Returns:
        Path to generated report directory
    """
    # Check if plotting libraries are available
    plotting_enabled = PLOTTING_AVAILABLE
    
    if not plotting_enabled:
        print("⚠️  Matplotlib/seaborn not available. Installing visualization dependencies...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
            plotting_enabled = True
        except Exception as e:
            print(f"❌ Failed to install plotting dependencies: {e}")
            print("Generating report without plots...")
            plotting_enabled = False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(results_file)
    
    # Generate aggregated metrics
    metrics = aggregate_thesis_metrics(df)
    
    # 1. Performance by Tier Summary Table (always generate)
    tier_summary = _create_tier_summary_table(df, output_path)
    
    plot_count = 0
    if PLOTTING_AVAILABLE:
        try:
            # Set style for plots
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
            if sns is not None:
                sns.set_palette("husl")
            
            # 2. Hallucination Detection Analysis
            _create_hallucination_plots(df, output_path)
            
            # 3. Safety Performance Analysis  
            _create_safety_plots(df, output_path)
            
            # 4. Efficiency and Runtime Analysis
            _create_efficiency_plots(df, output_path)
            
            # 5. Environmental Impact Analysis
            _create_environmental_plots(df, output_path)
            
            # 6. Correlation Analysis
            _create_correlation_plots(df, output_path)
            
            plot_count = len(list(output_path.glob('*.png')))
        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")
            print("⚠️  Continuing without plots...")
            plot_count = 0
    else:
        print("⚠️  Skipping plots due to missing visualization dependencies")
    
    # 7. Generate comprehensive summary report (always generate)
    _create_summary_report(metrics, output_path)
    
    print(f"📊 Thesis report generated in: {output_path}")
    print(f"📈 Generated {plot_count} plots")
    print(f"📋 Generated {len(list(output_path.glob('*.csv')))} tables")
    
    return str(output_path)


def _create_tier_summary_table(df: pd.DataFrame, output_path: Path) -> str:
    """Create tier performance summary table"""
    
    # Group by tier and calculate summary statistics
    tier_stats = df.groupby('tier').agg({
        'hallucination_detected': ['count', 'mean', 'std'],
        'safety_score': ['mean', 'std', 'min', 'max'],
        'icao_compliant': 'mean',
        'fp': 'mean',
        'fn': 'mean',
        'extra_nm': 'mean',
        'n_interventions': 'mean',
        'runtime_s': 'mean'
    }).round(3)
    
    # Flatten column names
    tier_stats.columns = ['_'.join(col).strip() for col in tier_stats.columns]
    
    # Save to CSV
    table_file = output_path / "tier_performance_summary.csv"
    tier_stats.to_csv(table_file)
    
    # Print formatted table
    print("\n" + "="*80)
    print("TIER PERFORMANCE SUMMARY")
    print("="*80)
    print(tier_stats.to_string())
    print("="*80)
    
    return str(table_file)


def _create_hallucination_plots(df: pd.DataFrame, output_path: Path):
    """Create hallucination analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hallucination Detection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Hallucination rate by tier
    tier_hall_rate = df.groupby('tier')['hallucination_detected'].mean()
    axes[0,0].bar(tier_hall_rate.index, tier_hall_rate.values)
    axes[0,0].set_title('Hallucination Detection Rate by Tier')
    axes[0,0].set_ylabel('Detection Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Hallucination rate by complexity
    complexity_hall_rate = df.groupby('complexity')['hallucination_detected'].mean()
    axes[0,1].bar(complexity_hall_rate.index, complexity_hall_rate.values, color='orange')
    axes[0,1].set_title('Hallucination Detection Rate by Complexity')
    axes[0,1].set_ylabel('Detection Rate')
    
    # 3. False positive/negative rates by tier
    fp_fn_data = df.groupby('tier')[['fp', 'fn']].mean()
    fp_fn_data.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('False Positive/Negative Rates by Tier')
    axes[1,0].set_ylabel('Average Rate')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(['False Positives', 'False Negatives'])
    
    # 4. Hallucination vs Safety Score
    for tier in df['tier'].unique():
        tier_data = df[df['tier'] == tier]
        axes[1,1].scatter(tier_data['safety_score'], tier_data['hallucination_detected'], 
                         alpha=0.6, label=tier)
    axes[1,1].set_title('Hallucination Detection vs Safety Score')
    axes[1,1].set_xlabel('Safety Score')
    axes[1,1].set_ylabel('Hallucination Detected')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'hallucination_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_safety_plots(df: pd.DataFrame, output_path: Path):
    """Create safety performance plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Safety Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Safety score distribution by tier
    for tier in df['tier'].unique():
        tier_data = df[df['tier'] == tier]['safety_score']
        axes[0,0].hist(tier_data, alpha=0.7, label=tier, bins=20)
    axes[0,0].set_title('Safety Score Distribution by Tier')
    axes[0,0].set_xlabel('Safety Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    # 2. ICAO compliance rate by tier
    icao_compliance = df.groupby('tier')['icao_compliant'].mean()
    axes[0,1].bar(icao_compliance.index, icao_compliance.values, color='green', alpha=0.7)
    axes[0,1].set_title('ICAO Compliance Rate by Tier')
    axes[0,1].set_ylabel('Compliance Rate')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Safety margins by tier
    safety_margins = df.groupby('tier')[['horiz_margin_ft', 'vert_margin_nm']].mean()
    safety_margins.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Average Safety Margins by Tier')
    axes[1,0].set_ylabel('Margin Value')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(['Horizontal (ft)', 'Vertical (nm)'])
    
    # 4. Safety score vs interventions
    axes[1,1].scatter(df['safety_score'], df['n_interventions'], alpha=0.6)
    axes[1,1].set_title('Safety Score vs Controller Interventions')
    axes[1,1].set_xlabel('Safety Score')
    axes[1,1].set_ylabel('Number of Interventions')
    
    plt.tight_layout()
    plt.savefig(output_path / 'safety_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_efficiency_plots(df: pd.DataFrame, output_path: Path):
    """Create efficiency and runtime plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Efficiency and Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Runtime by tier
    runtime_by_tier = df.groupby('tier')['runtime_s'].mean()
    axes[0,0].bar(runtime_by_tier.index, runtime_by_tier.values, color='purple', alpha=0.7)
    axes[0,0].set_title('Average Runtime by Tier')
    axes[0,0].set_ylabel('Runtime (seconds)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Extra distance by tier
    extra_dist_by_tier = df.groupby('tier')['extra_nm'].mean()
    axes[0,1].bar(extra_dist_by_tier.index, extra_dist_by_tier.values, color='red', alpha=0.7)
    axes[0,1].set_title('Average Extra Distance by Tier')
    axes[0,1].set_ylabel('Extra Distance (nm)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Response time breakdown
    response_times = df.groupby('tier')[['response_time_s', 'detection_time_s']].mean()
    response_times.plot(kind='bar', stacked=True, ax=axes[1,0])
    axes[1,0].set_title('Response Time Breakdown by Tier')
    axes[1,0].set_ylabel('Time (seconds)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(['LLM Response', 'Detection Time'])
    
    # 4. Efficiency scatter: extra distance vs runtime
    for tier in df['tier'].unique():
        tier_data = df[df['tier'] == tier]
        axes[1,1].scatter(tier_data['runtime_s'], tier_data['extra_nm'], 
                         alpha=0.6, label=tier)
    axes[1,1].set_title('Runtime vs Extra Distance')
    axes[1,1].set_xlabel('Runtime (seconds)')
    axes[1,1].set_ylabel('Extra Distance (nm)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_environmental_plots(df: pd.DataFrame, output_path: Path):
    """Create environmental impact plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Environmental Impact Analysis', fontsize=16, fontweight='bold')
    
    # 1. Wind speed impact on safety
    axes[0,0].scatter(df['wind_speed_kts'], df['safety_score'], alpha=0.6)
    axes[0,0].set_title('Wind Speed vs Safety Score')
    axes[0,0].set_xlabel('Wind Speed (kts)')
    axes[0,0].set_ylabel('Safety Score')
    
    # 2. Turbulence impact on hallucinations
    axes[0,1].scatter(df['turbulence_intensity'], df['hallucination_detected'], alpha=0.6)
    axes[0,1].set_title('Turbulence vs Hallucination Detection')
    axes[0,1].set_xlabel('Turbulence Intensity')
    axes[0,1].set_ylabel('Hallucination Detected')
    
    # 3. Visibility impact on interventions
    axes[1,0].scatter(df['visibility_nm'], df['n_interventions'], alpha=0.6)
    axes[1,0].set_title('Visibility vs Controller Interventions')
    axes[1,0].set_xlabel('Visibility (nm)')
    axes[1,0].set_ylabel('Number of Interventions')
    
    # 4. Navigation error impact
    nav_error_data = df[df['navigation_error_nm'] > 0]
    if len(nav_error_data) > 0:
        axes[1,1].scatter(nav_error_data['navigation_error_nm'], 
                         nav_error_data['safety_score'], alpha=0.6, color='red')
        axes[1,1].set_title('Navigation Error vs Safety Score')
        axes[1,1].set_xlabel('Navigation Error (nm)')
        axes[1,1].set_ylabel('Safety Score')
    else:
        axes[1,1].text(0.5, 0.5, 'No Navigation Errors\nin Dataset', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Navigation Error Impact')
    
    plt.tight_layout()
    plt.savefig(output_path / 'environmental_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_correlation_plots(df: pd.DataFrame, output_path: Path):
    """Create correlation analysis plots"""
    
    # Select numeric columns for correlation
    numeric_cols = ['hallucination_detected', 'fp', 'fn', 'safety_score', 'icao_compliant',
                   'extra_nm', 'n_interventions', 'runtime_s', 'wind_speed_kts', 
                   'turbulence_intensity', 'visibility_nm']
    
    # Filter existing columns
    existing_cols = [col for col in numeric_cols if col in df.columns]
    corr_data = df[existing_cols]
    
    # Calculate correlation matrix
    corr_matrix = corr_data.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Correlation Matrix - Key Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_summary_report(metrics: Dict[str, Any], output_path: Path):
    """Create comprehensive summary report"""
    
    report_file = output_path / "thesis_summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LLM-ATC-HAL DISTRIBUTION SHIFT EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Experiment overview
        overview = metrics.get('experiment_overview', {})
        f.write("EXPERIMENT OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Simulations: {overview.get('total_experiments', 0)}\n")
        f.write(f"Distribution Shift Tiers: {overview.get('distribution_shift_tiers', [])}\n")
        f.write(f"Complexity Distribution: {overview.get('complexity_distribution', {})}\n\n")
        
        # Key findings
        hallucination = metrics.get('hallucination_analysis', {})
        safety = metrics.get('safety_performance', {})
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 12 + "\n")
        f.write(f"Overall Hallucination Detection Rate: {hallucination.get('overall_detection_rate', 0):.3f}\n")
        f.write(f"Overall Safety Score: {safety.get('overall_safety_score', 0):.3f}\n")
        f.write(f"ICAO Compliance Rate: {safety.get('icao_compliance_rate', 0):.3f}\n\n")
        
        # Per-tier performance
        f.write("PERFORMANCE BY TIER\n")
        f.write("-" * 20 + "\n")
        tier_performance = metrics.get('performance_by_tier', {})
        for tier, perf in tier_performance.items():
            f.write(f"\n{tier.upper()}:\n")
            f.write(f"  Simulations: {perf.get('n_simulations', 0)}\n")
            f.write(f"  Hallucination Rate: {perf.get('hallucination_rate', 0):.3f}\n")
            f.write(f"  Safety Score: {perf.get('safety_score', 0):.3f}\n")
            f.write(f"  ICAO Compliance: {perf.get('icao_compliance', 0):.3f}\n")
            f.write(f"  Avg Runtime: {perf.get('avg_runtime', 0):.2f}s\n")
        
        # Statistical significance
        statistical = metrics.get('statistical_tests', {})
        if statistical:
            f.write(f"\nSTATISTICAL TESTS\n")
            f.write("-" * 16 + "\n")
            for test_name, result in statistical.items():
                if isinstance(result, dict) and 'p_value' in result:
                    f.write(f"{test_name}: p={result['p_value']:.4f} ")
                    f.write(f"({'significant' if result.get('significant', False) else 'not significant'})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated on: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("="*80 + "\n")
    
    print(f"📄 Summary report saved to: {report_file}")


# CLI for aggregate analysis
def main_analysis():
    """Main CLI entry point for analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analysis and Reporting for Distribution Shift Experiments"
    )
    
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Generate comprehensive thesis report with tables and plots'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        help='Path to results parquet file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='thesis_results',
        help='Output directory for thesis results (default: thesis_results)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file to analyze (for legacy metric computation)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.aggregate:
            if not args.results:
                # Find most recent results file
                results_dir = Path("experiments/results")
                if results_dir.exists():
                    parquet_files = list(results_dir.glob("distribution_shift_experiment_*.parquet"))
                    if parquet_files:
                        args.results = str(max(parquet_files, key=lambda p: p.stat().st_mtime))
                        print(f"Using most recent results file: {args.results}")
                    else:
                        print("❌ No results files found. Run experiments first.")
                        return 1
                else:
                    print("❌ Results directory not found. Run experiments first.")
                    return 1
            
            # Generate thesis report
            report_dir = generate_thesis_report(args.results, args.output)
            print(f"✅ Thesis report generated successfully in: {report_dir}")
            
        elif args.log_file:
            # Legacy metrics computation
            metrics = compute_metrics(args.log_file)
            print_metrics_summary(metrics)
            
        else:
            parser.print_help()
            return 1
            
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main_analysis()
    exit(exit_code)
