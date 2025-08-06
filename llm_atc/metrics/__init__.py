# metrics/__init__.py
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

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
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


# Hallucination analysis stub - removed legacy analysis module dependency
def analyze_hallucinations_in_log(_log_file: str) -> dict[str, Any]:
    """Analyze hallucinations in log file - simplified implementation"""
    return {"total_hallucinations": 0, "by_type": {}, "by_model": {}}


def compute_metrics(log_file: str) -> dict[str, Any]:
    """Compute hallucination and performance metrics from simulation logs."""
    try:
        # Get detailed hallucination analysis
        hallucination_analysis = analyze_hallucinations_in_log(log_file)

        # Read log file line by line and extract JSON entries
        data = []
        with Path(log_file).open() as f:
            for line in f:
                log_entry = line.strip()
                if log_entry:
                    # Try to extract JSON from log entries
                    if log_entry.startswith("{") and log_entry.endswith("}"):
                        # Direct JSON line
                        try:
                            parsed_entry = json.loads(log_entry)
                            if (
                                "best_by_llm" in parsed_entry
                                and "baseline_best" in parsed_entry
                            ):
                                data.append(parsed_entry)
                        except json.JSONDecodeError as e:
                            logging.warning(
                                "Failed to parse JSON line: %s. Error: %s",
                                log_entry,
                                e,
                            )
                    else:
                        # Try to extract JSON from within log line
                        try:
                            # Look for JSON patterns in the log line
                            json_start = log_entry.find("{")
                            json_end = log_entry.rfind("}")
                            if json_start >= 0 and json_end > json_start:
                                json_str = log_entry[json_start : json_end + 1]
                                parsed_entry = json.loads(json_str)
                                if (
                                    "best_by_llm" in parsed_entry
                                    and "baseline_best" in parsed_entry
                                ):
                                    data.append(parsed_entry)
                        except (json.JSONDecodeError, ValueError):
                            # Skip lines that don't contain valid JSON
                            continue

        if not data:
            logging.warning("No valid JSON entries found in %s", log_file)
            return create_empty_metrics()

        # Convert to DataFrame for easier analysis
        data_frame = pd.DataFrame(data)

        # Basic metrics
        total_tests = len(data_frame)

        # Performance metrics
        llm_times = []
        baseline_times = []
        hallucination_counts = []

        fp_rates = []
        fn_rates = []
        safety_margins = []
        efficiency_penalties = []

        for _, row in data_frame.iterrows():
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
            "avg_efficiency_penalty": (
                np.mean(efficiency_penalties) if efficiency_penalties else 0
            ),
            "hallucination_analysis": hallucination_analysis,
        }

    except Exception:
        logging.exception("Error computing metrics from %s", log_file)
        return create_empty_metrics()


def create_empty_metrics() -> dict[str, Any]:
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
        "hallucination_analysis": {
            "total_hallucinations": 0,
            "by_type": {},
            "by_model": {},
        },
    }


def print_metrics_summary(metrics: dict[str, Any]) -> None:
    """Print a formatted summary of the metrics using logging."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info("Total Tests: %s", metrics["total_tests"])
    logger.info("Total Hallucinations: %s", metrics["total_hallucinations"])
    logger.info("Avg Hallucinations/Test: %.2f", metrics["avg_hallucinations_per_test"])
    logger.info("Avg LLM Time: %.2fs", metrics["avg_llm_time"])
    logger.info("Avg Baseline Time: %.2fs", metrics["avg_baseline_time"])
    logger.info("Avg FP Rate: %.3f", metrics["avg_fp_rate"])
    logger.info("Avg FN Rate: %.3f", metrics["avg_fn_rate"])
    logger.info("Avg Safety Margin: %.2f NM", metrics["avg_safety_margin"])
    logger.info("Avg Efficiency Penalty: %.2f NM", metrics["avg_efficiency_penalty"])

    # Hallucination breakdown
    hal_analysis = metrics["hallucination_analysis"]
    if hal_analysis["by_type"]:
        logger.info("Hallucination Breakdown by Type:")
        for hal_type, count in hal_analysis["by_type"].items():
            logger.info("  %s: %s", hal_type, count)

    if hal_analysis["by_model"]:
        logger.info("Hallucination Breakdown by Model:")
        for model, count in hal_analysis["by_model"].items():
            logger.info("  %s: %s", model, count)

    logger.info("=" * 60)


def calc_fp_fn(
    pred_conflicts: list[dict[str, Any]],
    gt_conflicts: list[dict[str, Any]],
) -> tuple[float, float]:
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


def calc_path_extra(
    actual_traj: list[dict[str, Any]],
    original_traj: list[dict[str, Any]],
) -> float:
    """Calculate extra distance traveled due to resolution maneuvers."""
    if not actual_traj or not original_traj:
        return 0.0

    def calc_trajectory_distance(traj: list[dict[str, Any]]) -> float:
        """Calculate total distance of a trajectory."""
        total_dist = 0.0
        for i in range(1, len(traj)):
            prev_point = traj[i - 1]
            curr_point = traj[i]

            # Simple Euclidean distance (in practice, use great circle distance)
            if (
                "lat" in prev_point
                and "lon" in prev_point
                and "lat" in curr_point
                and "lon" in curr_point
            ):
                lat_diff = curr_point["lat"] - prev_point["lat"]
                lon_diff = curr_point["lon"] - prev_point["lon"]
                # Rough conversion to nautical miles (more accurate calculation needed)
                dist = (
                    math.sqrt(lat_diff**2 + lon_diff**2) * 60
                )  # degrees to NM approximation
                total_dist += dist

        return total_dist

    actual_distance = calc_trajectory_distance(actual_traj)
    original_distance = calc_trajectory_distance(original_traj)

    return max(0.0, actual_distance - original_distance)


def process_detailed_results(detailed_results_file: str) -> dict[str, Any]:
    """Process detailed_results.json files for comprehensive metrics."""
    try:
        with open(detailed_results_file, 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", [])
        if not scenarios:
            return create_empty_metrics()
        
        # Extract metrics from scenarios
        total_scenarios = len(scenarios)
        successful_scenarios = sum(1 for s in scenarios if s.get("success", False))
        
        # Conflict detection metrics
        all_fp_rates = []
        all_fn_rates = []
        detection_accuracies = []
        
        for scenario in scenarios:
            detection_perf = scenario.get("detection_performance", {})
            if detection_perf:
                all_fp_rates.append(detection_perf.get("false_positive_rate", 0))
                all_fn_rates.append(detection_perf.get("false_negative_rate", 0))
                detection_accuracies.append(detection_perf.get("detection_accuracy", 0))
        
        return {
            "total_tests": total_scenarios,
            "success_rate": successful_scenarios / total_scenarios if total_scenarios > 0 else 0,
            "average_fp_rate": np.mean(all_fp_rates) if all_fp_rates else 0,
            "average_fn_rate": np.mean(all_fn_rates) if all_fn_rates else 0,
            "average_detection_accuracy": np.mean(detection_accuracies) if detection_accuracies else 0,
            "total_hallucinations": sum(len(s.get("detected_conflicts", [])) for s in scenarios),
            "detailed_metrics": {
                "scenarios_analyzed": total_scenarios,
                "fp_rate_std": np.std(all_fp_rates) if all_fp_rates else 0,
                "fn_rate_std": np.std(all_fn_rates) if all_fn_rates else 0,
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to process detailed results {detailed_results_file}: {e}")
        return create_empty_metrics()


def process_benchmark_summary(benchmark_summary_file: str) -> dict[str, Any]:
    """Process benchmark_summary.json files for aggregate metrics."""
    try:
        with open(benchmark_summary_file, 'r') as f:
            data = json.load(f)
        
        # Extract aggregate metrics
        aggregate_metrics = data.get("aggregate_metrics", {})
        scenario_metrics = data.get("scenario_metrics", {})
        
        total_scenarios = aggregate_metrics.get("total_scenarios", 0)
        successful_scenarios = aggregate_metrics.get("successful_scenarios", 0)
        
        return {
            "total_tests": total_scenarios,
            "success_rate": successful_scenarios / total_scenarios if total_scenarios > 0 else 0,
            "average_fp_rate": aggregate_metrics.get("average_false_positive_rate", 0),
            "average_fn_rate": aggregate_metrics.get("average_false_negative_rate", 0),
            "average_detection_accuracy": aggregate_metrics.get("average_detection_accuracy", 0),
            "total_hallucinations": aggregate_metrics.get("total_false_positives", 0),
            "detailed_metrics": {
                "by_scenario_type": scenario_metrics,
                "execution_time": data.get("execution_time", "unknown"),
                "configuration": data.get("configuration", {})
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to process benchmark summary {benchmark_summary_file}: {e}")
        return create_empty_metrics()


def aggregate_thesis_metrics(results_dir: str) -> dict[str, Any]:
    """Aggregate metrics from multiple test result files for thesis analysis."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logging.warning("Results directory %s does not exist", results_dir)
        return create_empty_metrics()

    all_metrics = []
    
    # Search recursively for log and JSON files
    log_files = list(results_path.glob("**/*.log"))
    json_files = list(results_path.glob("**/*.json"))
    
    # Also check for specific result files
    detailed_results = list(results_path.glob("**/detailed_results.json"))
    summary_files = list(results_path.glob("**/benchmark_summary.json"))
    
    all_files = log_files + json_files
    
    # If we have detailed results, prioritize those
    if detailed_results:
        all_files.extend(detailed_results)
    if summary_files:
        all_files.extend(summary_files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    logging.info(f"Found {len(all_files)} files to analyze in {results_dir}")

    for result_file in all_files:
        try:
            if result_file.name == "detailed_results.json":
                # Handle detailed results file specifically
                metrics = process_detailed_results(str(result_file))
            elif result_file.name == "benchmark_summary.json":
                # Handle summary file specifically
                metrics = process_benchmark_summary(str(result_file))
            else:
                # Regular log/json processing
                metrics = compute_metrics(str(result_file))
            
            if metrics["total_tests"] > 0:
                all_metrics.append(metrics)
                logging.info(f"Processed {result_file.name}: {metrics['total_tests']} tests")
        except Exception as e:
            logging.warning("Failed to process %s: %s", result_file, e)

    if not all_metrics:
        logging.warning("No valid metrics found in results directory")
        return create_empty_metrics()

    # Aggregate across all files
    aggregated = {
        "total_tests": sum(m["total_tests"] for m in all_metrics),
        "total_hallucinations": sum(m["total_hallucinations"] for m in all_metrics),
        "avg_llm_time": np.nanmean(
            [m["avg_llm_time"] for m in all_metrics if m["avg_llm_time"] > 0]
        ) if any(m["avg_llm_time"] > 0 for m in all_metrics) else 0,
        "avg_baseline_time": np.nanmean(
            [m["avg_baseline_time"] for m in all_metrics if m["avg_baseline_time"] > 0]
        ) if any(m["avg_baseline_time"] > 0 for m in all_metrics) else 0,
        "avg_fp_rate": np.nanmean(
            [m["avg_fp_rate"] for m in all_metrics if m["avg_fp_rate"] > 0]
        ) if any(m["avg_fp_rate"] > 0 for m in all_metrics) else 0,
        "avg_fn_rate": np.nanmean(
            [m["avg_fn_rate"] for m in all_metrics if m["avg_fn_rate"] > 0]
        ) if any(m["avg_fn_rate"] > 0 for m in all_metrics) else 0,
        "avg_safety_margin": np.nanmean(
            [m["avg_safety_margin"] for m in all_metrics if m["avg_safety_margin"] > 0]
        ) if any(m["avg_safety_margin"] > 0 for m in all_metrics) else 0,
        "avg_efficiency_penalty": np.nanmean(
            [
                m["avg_efficiency_penalty"]
                for m in all_metrics
                if m["avg_efficiency_penalty"] > 0
            ]
        ) if any(m["avg_efficiency_penalty"] > 0 for m in all_metrics) else 0,
        "files_processed": len(all_metrics),
        "hallucination_analysis": {
            "total_hallucinations": sum(m["total_hallucinations"] for m in all_metrics),
            "by_type": {},
            "by_model": {},
        }
    }

    # Calculate overall hallucination rate
    if aggregated["total_tests"] > 0:
        aggregated["avg_hallucinations_per_test"] = (
            aggregated["total_hallucinations"] / aggregated["total_tests"]
        )
    else:
        aggregated["avg_hallucinations_per_test"] = 0

    return aggregated


def process_detailed_results(file_path: str) -> dict[str, Any]:
    """Process detailed_results.json file to extract metrics."""
    try:
        with open(file_path, 'r') as f:
            detailed_results = json.load(f)
        
        if not isinstance(detailed_results, list):
            return create_empty_metrics()
        
        total_tests = len(detailed_results)
        success_count = sum(1 for result in detailed_results if result.get("success", False))
        
        # Extract performance metrics
        conflicts_detected = []
        safety_margins = []
        efficiency_metrics = []
        
        for result in detailed_results:
            # Count conflicts
            predicted_conflicts = result.get("predicted_conflicts", [])
            true_conflicts = result.get("true_conflicts", [])
            conflicts_detected.append(len(predicted_conflicts))
            
            # Safety margins
            if "safety_assessment" in result:
                safety_data = result["safety_assessment"]
                if "horizontal_margin_nm" in safety_data:
                    safety_margins.append(safety_data["horizontal_margin_nm"])
            
            # Efficiency metrics
            if "efficiency_assessment" in result:
                eff_data = result["efficiency_assessment"]
                if "path_deviation_nm" in eff_data:
                    efficiency_metrics.append(eff_data["path_deviation_nm"])
        
        return {
            "total_tests": total_tests,
            "avg_llm_time": 0,  # Not available in this format
            "avg_baseline_time": 0,
            "avg_hallucinations_per_test": 0,  # Not tracked in this format
            "total_hallucinations": 0,
            "avg_fp_rate": 0,  # Would need more analysis
            "avg_fn_rate": 0,
            "avg_safety_margin": np.mean(safety_margins) if safety_margins else 0,
            "avg_efficiency_penalty": np.mean(efficiency_metrics) if efficiency_metrics else 0,
            "success_rate": success_count / total_tests if total_tests > 0 else 0,
            "hallucination_analysis": {"total_hallucinations": 0, "by_type": {}, "by_model": {}},
        }
    except Exception as e:
        logging.warning(f"Failed to process detailed results {file_path}: {e}")
        return create_empty_metrics()


def process_benchmark_summary(file_path: str) -> dict[str, Any]:
    """Process benchmark_summary.json file to extract metrics."""
    try:
        with open(file_path, 'r') as f:
            summary = json.load(f)
        
        # Extract metrics from summary
        scenario_counts = summary.get("scenario_counts", {})
        total_tests = scenario_counts.get("total_scenarios", 0)
        success_count = scenario_counts.get("successful_scenarios", 0)
        
        # Get performance metrics if available
        performance = summary.get("overall_performance", {})
        
        return {
            "total_tests": total_tests,
            "avg_llm_time": 0,  # Not typically available in summary
            "avg_baseline_time": 0,
            "avg_hallucinations_per_test": 0,
            "total_hallucinations": 0,
            "avg_fp_rate": 1 - performance.get("precision", 0) if performance.get("precision", 0) > 0 else 0,
            "avg_fn_rate": 1 - performance.get("recall", 0) if performance.get("recall", 0) > 0 else 0,
            "avg_safety_margin": performance.get("avg_min_separation_nm", 0),
            "avg_efficiency_penalty": 0,  # Not available in this format
            "success_rate": scenario_counts.get("success_rate", 0),
            "detection_accuracy": performance.get("detection_accuracy", 0),
            "precision": performance.get("precision", 0),
            "recall": performance.get("recall", 0),
            "total_violations": performance.get("total_violations", 0),
            "hallucination_analysis": {"total_hallucinations": 0, "by_type": {}, "by_model": {}},
        }
    except Exception as e:
        logging.warning(f"Failed to process benchmark summary {file_path}: {e}")
        return create_empty_metrics()


# Visualization functions (if plotting is available)
def plot_metrics_comparison(
    llm_metrics: dict,
    baseline_metrics: dict,
    save_path: Optional[str] = None,
) -> None:
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
    baseline_values = [
        baseline_metrics.get("avg_fp_rate", 0),
        baseline_metrics.get("avg_fn_rate", 0),
    ]

    x = np.arange(len(categories))
    width = 0.35

    axes[0, 0].bar(x - width / 2, llm_values, width, label="LLM", alpha=0.8)
    axes[0, 0].bar(x + width / 2, baseline_values, width, label="Baseline", alpha=0.8)
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Error Rates")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()

    # Response time comparison
    models = ["LLM", "Baseline"]
    times = [
        llm_metrics.get("avg_llm_time", 0),
        baseline_metrics.get("avg_baseline_time", 0),
    ]

    axes[0, 1].bar(models, times, alpha=0.8, color=["blue", "orange"])
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].set_title("Average Response Time")

    # Safety margin comparison
    safety_metrics = ["Safety Margin", "Efficiency Penalty"]
    llm_safety = [
        llm_metrics.get("avg_safety_margin", 0),
        llm_metrics.get("avg_efficiency_penalty", 0),
    ]
    baseline_safety = [
        baseline_metrics.get("avg_safety_margin", 0),
        baseline_metrics.get("avg_efficiency_penalty", 0),
    ]

    x = np.arange(len(safety_metrics))
    axes[1, 0].bar(x - width / 2, llm_safety, width, label="LLM", alpha=0.8)
    axes[1, 0].bar(x + width / 2, baseline_safety, width, label="Baseline", alpha=0.8)
    axes[1, 0].set_ylabel("Distance (NM)")
    axes[1, 0].set_title("Safety and Efficiency Metrics")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(safety_metrics)
    axes[1, 0].legend()

    # Hallucination rate (LLM only)
    axes[1, 1].bar(
        ["LLM"],
        [llm_metrics.get("avg_hallucinations_per_test", 0)],
        alpha=0.8,
        color="red",
    )
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
        if Path(log_file).exists():
            metrics = compute_metrics(log_file)
            print_metrics_summary(metrics)
        else:
            logging.error("Log file %s not found", log_file)
    else:
        logging.info("Usage: python -m metrics <log_file>")
