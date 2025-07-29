# metrics/__init__.py
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use("default")  # Set a default style
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Hallucination analysis stub - removed legacy analysis module dependency
def analyze_hallucinations_in_log(log_file):
    """Analyze hallucinations in log file - simplified implementation"""
    return {"total_hallucinations": 0, "by_type": {}, "by_model": {}}

def compute_metrics(log_file):
    """Compute hallucination and performance metrics from simulation logs."""
    try:
        # Get detailed hallucination analysis
        hallucination_analysis = analyze_hallucinations_in_log(log_file)

        # Read log file line by line and extract JSON entries
        data = []
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    # Try to extract JSON from log entries
                    if line.startswith("{") and line.endswith("}"):
                        # Direct JSON line
                        try:
                            entry = json.loads(line)
                            if "best_by_llm" in entry and "baseline_best" in entry:
                                data.append(entry)
                        except json.JSONDecodeError as e:
                            logging.warning("Failed to parse JSON line: %s. Error: %s", line, e)
                    else:
                        # Try to extract JSON from within log line
                        try:
                            # Look for JSON patterns in the log line
                            json_start = line.find("{")
                            json_end = line.rfind("}")
                            if json_start >= 0 and json_end > json_start:
                                json_str = line[json_start:json_end + 1]
                                entry = json.loads(json_str)
                                if "best_by_llm" in entry and "baseline_best" in entry:
                                    data.append(entry)
                        except (json.JSONDecodeError, ValueError):
                            # Skip lines that don't contain valid JSON
                            continue

        if not data:
            logging.warning("No valid JSON entries found in %s", log_file)
            return create_empty_metrics()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)

        # Basic metrics
        total_tests = len(df)

        # Performance metrics
        llm_times = []
        baseline_times = []
        hallucination_counts = []

        fp_rates = []
        fn_rates = []
        safety_margins = []
        efficiency_penalties = []

        for _, row in df.iterrows():
            # Extract timing data
            if "llm_time" in row:
                llm_times.append(row["llm_time"])
            if "baseline_time" in row:
                baseline_times.append(row["baseline_time"])

            # Extract hallucination data
            if "hallucinations" in row:
                hallucination_counts.append(len(row["hallucinations"]))
            else:
                hallucination_counts.append(0)

            # Extract performance metrics
            if "fp_rate" in row:
                fp_rates.append(row["fp_rate"])
            if "fn_rate" in row:
                fn_rates.append(row["fn_rate"])
            if "safety_margin" in row:
                safety_margins.append(row["safety_margin"])
            if "efficiency_penalty" in row:
                efficiency_penalties.append(row["efficiency_penalty"])

        # Compute aggregate metrics
        return {
            "total_tests": total_tests,
            "avg_llm_time": np.mean(llm_times) if llm_times else 0,
            "avg_baseline_time": np.mean(baseline_times) if baseline_times else 0,
            "avg_hallucinations_per_test": np.mean(hallucination_counts),
            "total_hallucinations": sum(hallucination_counts),
            "avg_fp_rate": np.mean(fp_rates) if fp_rates else 0,
            "avg_fn_rate": np.mean(fn_rates) if fn_rates else 0,
            "avg_safety_margin": np.mean(safety_margins) if safety_margins else 0,
            "avg_efficiency_penalty": np.mean(efficiency_penalties) if efficiency_penalties else 0,
            "hallucination_analysis": hallucination_analysis,
        }


    except Exception:
        logging.exception("Error computing metrics from %s", log_file)
        return create_empty_metrics()

def create_empty_metrics():
    """Create empty metrics structure when no data is available."""
    return {
        "total_tests": 0,
        "avg_llm_time": 0,
        "avg_baseline_time": 0,
        "avg_hallucinations_per_test": 0,
        "total_hallucinations": 0,
        "avg_fp_rate": 0,
        "avg_fn_rate": 0,
        "avg_safety_margin": 0,
        "avg_efficiency_penalty": 0,
        "hallucination_analysis": {"total_hallucinations": 0, "by_type": {}, "by_model": {}},
    }

def print_metrics_summary(metrics):
    """Print a formatted summary of the metrics."""
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(f"Total Tests: {metrics['total_tests']}")
    print(f"Total Hallucinations: {metrics['total_hallucinations']}")
    print(f"Avg Hallucinations/Test: {metrics['avg_hallucinations_per_test']:.2f}")
    print(f"Avg LLM Time: {metrics['avg_llm_time']:.2f}s")
    print(f"Avg Baseline Time: {metrics['avg_baseline_time']:.2f}s")
    print(f"Avg FP Rate: {metrics['avg_fp_rate']:.3f}")
    print(f"Avg FN Rate: {metrics['avg_fn_rate']:.3f}")
    print(f"Avg Safety Margin: {metrics['avg_safety_margin']:.2f} NM")
    print(f"Avg Efficiency Penalty: {metrics['avg_efficiency_penalty']:.2f} NM")

    # Hallucination breakdown
    hal_analysis = metrics["hallucination_analysis"]
    if hal_analysis["by_type"]:
        print("\nHallucination Breakdown by Type:")
        for hal_type, count in hal_analysis["by_type"].items():
            print(f"  {hal_type}: {count}")

    if hal_analysis["by_model"]:
        print("\nHallucination Breakdown by Model:")
        for model, count in hal_analysis["by_model"].items():
            print(f"  {model}: {count}")

    print("="*60)

def calc_fp_fn(pred_conflicts: list[dict[str, Any]],
               gt_conflicts: list[dict[str, Any]]) -> tuple[float, float]:
    """Calculate false positive and false negative rates."""
    if not pred_conflicts and not gt_conflicts:
        return 0.0, 0.0

    # Convert to sets of conflict pairs for comparison
    pred_pairs = set()
    for conflict in pred_conflicts:
        if "aircraft_1" in conflict and "aircraft_2" in conflict:
            pair = tuple(sorted([conflict["aircraft_1"], conflict["aircraft_2"]]))
            pred_pairs.add(pair)

    gt_pairs = set()
    for conflict in gt_conflicts:
        if "aircraft_1" in conflict and "aircraft_2" in conflict:
            pair = tuple(sorted([conflict["aircraft_1"], conflict["aircraft_2"]]))
            gt_pairs.add(pair)

    # Calculate FP and FN
    false_positives = len(pred_pairs - gt_pairs)
    false_negatives = len(gt_pairs - pred_pairs)

    total_predicted = len(pred_pairs)
    total_actual = len(gt_pairs)

    fp_rate = false_positives / max(1, total_predicted)
    fn_rate = false_negatives / max(1, total_actual)

    return fp_rate, fn_rate

def calc_path_extra(actual_traj: list[dict[str, Any]],
                   original_traj: list[dict[str, Any]]) -> float:
    """Calculate extra distance traveled due to resolution maneuvers."""
    if not actual_traj or not original_traj:
        return 0.0

    def calc_trajectory_distance(traj):
        """Calculate total distance of a trajectory."""
        total_dist = 0.0
        for i in range(1, len(traj)):
            prev_point = traj[i-1]
            curr_point = traj[i]

            # Simple Euclidean distance (in practice, use great circle distance)
            if "lat" in prev_point and "lon" in prev_point and "lat" in curr_point and "lon" in curr_point:
                lat_diff = curr_point["lat"] - prev_point["lat"]
                lon_diff = curr_point["lon"] - prev_point["lon"]
                # Rough conversion to nautical miles (more accurate calculation needed)
                dist = math.sqrt(lat_diff**2 + lon_diff**2) * 60  # degrees to NM approximation
                total_dist += dist

        return total_dist

    actual_distance = calc_trajectory_distance(actual_traj)
    original_distance = calc_trajectory_distance(original_traj)

    return max(0.0, actual_distance - original_distance)

def aggregate_thesis_metrics(results_dir: str) -> dict[str, Any]:
    """Aggregate metrics from multiple test result files for thesis analysis."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logging.warning("Results directory %s does not exist", results_dir)
        return create_empty_metrics()

    all_metrics = []
    log_files = list(results_path.glob("*.log")) + list(results_path.glob("*.json"))

    for log_file in log_files:
        try:
            metrics = compute_metrics(str(log_file))
            if metrics["total_tests"] > 0:
                all_metrics.append(metrics)
        except Exception as e:
            logging.warning("Failed to process %s: %s", log_file, e)

    if not all_metrics:
        logging.warning("No valid metrics found in results directory")
        return create_empty_metrics()

    # Aggregate across all files
    aggregated = {
        "total_tests": sum(m["total_tests"] for m in all_metrics),
        "total_hallucinations": sum(m["total_hallucinations"] for m in all_metrics),
        "avg_llm_time": np.mean([m["avg_llm_time"] for m in all_metrics if m["avg_llm_time"] > 0]),
        "avg_baseline_time": np.mean([m["avg_baseline_time"] for m in all_metrics if m["avg_baseline_time"] > 0]),
        "avg_fp_rate": np.mean([m["avg_fp_rate"] for m in all_metrics if m["avg_fp_rate"] > 0]),
        "avg_fn_rate": np.mean([m["avg_fn_rate"] for m in all_metrics if m["avg_fn_rate"] > 0]),
        "avg_safety_margin": np.mean([m["avg_safety_margin"] for m in all_metrics if m["avg_safety_margin"] > 0]),
        "avg_efficiency_penalty": np.mean([m["avg_efficiency_penalty"] for m in all_metrics if m["avg_efficiency_penalty"] > 0]),
        "files_processed": len(all_metrics),
    }

    # Calculate overall hallucination rate
    if aggregated["total_tests"] > 0:
        aggregated["avg_hallucinations_per_test"] = aggregated["total_hallucinations"] / aggregated["total_tests"]
    else:
        aggregated["avg_hallucinations_per_test"] = 0

    return aggregated

# Visualization functions (if plotting is available)
def plot_metrics_comparison(llm_metrics: dict, baseline_metrics: dict, save_path: str = None):
    """Create comparison plots between LLM and baseline metrics."""
    if not PLOTTING_AVAILABLE:
        logging.warning("Matplotlib not available, skipping plots")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("LLM vs Baseline Performance Comparison", fontsize=16)

    # FP/FN Rate comparison
    categories = ["False Positive Rate", "False Negative Rate"]
    llm_values = [llm_metrics.get("avg_fp_rate", 0), llm_metrics.get("avg_fn_rate", 0)]
    baseline_values = [baseline_metrics.get("avg_fp_rate", 0), baseline_metrics.get("avg_fn_rate", 0)]

    x = np.arange(len(categories))
    width = 0.35

    axes[0, 0].bar(x - width/2, llm_values, width, label="LLM", alpha=0.8)
    axes[0, 0].bar(x + width/2, baseline_values, width, label="Baseline", alpha=0.8)
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Error Rates")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()

    # Response time comparison
    models = ["LLM", "Baseline"]
    times = [llm_metrics.get("avg_llm_time", 0), baseline_metrics.get("avg_baseline_time", 0)]

    axes[0, 1].bar(models, times, alpha=0.8, color=["blue", "orange"])
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].set_title("Average Response Time")

    # Safety margin comparison
    safety_metrics = ["Safety Margin", "Efficiency Penalty"]
    llm_safety = [llm_metrics.get("avg_safety_margin", 0), llm_metrics.get("avg_efficiency_penalty", 0)]
    baseline_safety = [baseline_metrics.get("avg_safety_margin", 0), baseline_metrics.get("avg_efficiency_penalty", 0)]

    x = np.arange(len(safety_metrics))
    axes[1, 0].bar(x - width/2, llm_safety, width, label="LLM", alpha=0.8)
    axes[1, 0].bar(x + width/2, baseline_safety, width, label="Baseline", alpha=0.8)
    axes[1, 0].set_ylabel("Distance (NM)")
    axes[1, 0].set_title("Safety and Efficiency Metrics")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(safety_metrics)
    axes[1, 0].legend()

    # Hallucination rate (LLM only)
    axes[1, 1].bar(["LLM"], [llm_metrics.get("avg_hallucinations_per_test", 0)], alpha=0.8, color="red")
    axes[1, 1].set_ylabel("Hallucinations per Test")
    axes[1, 1].set_title("Hallucination Rate")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info("Metrics comparison plot saved to %s", save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        if os.path.exists(log_file):
            metrics = compute_metrics(log_file)
            print_metrics_summary(metrics)
        else:
            print(f"Log file {log_file} not found")
    else:
        print("Usage: python -m metrics <log_file>")
