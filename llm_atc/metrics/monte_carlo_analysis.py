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
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# Plotting imports
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/seaborn not available - visualizations disabled")

try:
    from .safety_margin_quantifier import calc_efficiency_penalty, calc_separation_margin
except ImportError:
    # Fallback for standalone execution
    try:
        from safety_margin_quantifier import calc_efficiency_penalty, calc_separation_margin
    except ImportError:
        # Mock functions for testing
        def calc_separation_margin(trajectories):
            return {"hz": 5.0, "vt": 1000.0}

        def calc_efficiency_penalty(planned, executed) -> float:
            return 2.0


class MonteCarloResultsAnalyzer:
    """
    Aggregates and analyzes Monte Carlo simulation results for ATC scenarios.
    """

    def __init__(self) -> None:
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
            msg = f"Results file not found: {file_path}"
            raise FileNotFoundError(msg)

        if file_path.suffix.lower() == ".json":
            return self._read_json_results(file_path)
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        msg = f"Unsupported file format: {file_path.suffix}"
        raise ValueError(msg)

    def _read_json_results(self, file_path: Path) -> pd.DataFrame:
        """Read results from JSON file format."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                # List of result objects
                return pd.DataFrame(data)
            if isinstance(data, dict):
                if "results" in data:
                    # Nested structure with 'results' key
                    return pd.DataFrame(data["results"])
                # Single result object - convert to single-row DataFrame
                return pd.DataFrame([data])
            msg = "Unexpected JSON structure"
            raise ValueError(msg)

        except json.JSONDecodeError as e:
            self.logger.exception(f"Failed to parse JSON file {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error reading JSON results: {e}")
            raise

    def compute_false_positive_negative_rates(self, results_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute false positive and false negative rates from results.

        Args:
            results_df: DataFrame with columns 'predicted_conflicts', 'actual_conflicts'

        Returns:
            Dict with 'false_positive_rate' and 'false_negative_rate'
        """
        if results_df.empty:
            return {"false_positive_rate": 0.0, "false_negative_rate": 0.0}

        total_fp = 0
        total_fn = 0
        total_predicted = 0
        total_actual = 0

        for _, row in results_df.iterrows():
            # Get conflict lists
            predicted = row.get("predicted_conflicts", [])
            actual = row.get("actual_conflicts", [])

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
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "total_predicted_conflicts": total_predicted,
            "total_actual_conflicts": total_actual,
        }

    def _conflicts_to_set(self, conflicts: list[dict[str, Any]]) -> set:
        """Convert conflict list to set of aircraft pairs."""
        conflict_pairs = set()

        for conflict in conflicts:
            if isinstance(conflict, dict):
                # Extract aircraft IDs from conflict
                ac1 = conflict.get("aircraft_1") or conflict.get("aircraft1")
                ac2 = conflict.get("aircraft_2") or conflict.get("aircraft2")

                if ac1 and ac2:
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([str(ac1), str(ac2)]))
                    conflict_pairs.add(pair)

        return conflict_pairs

    def compute_success_rates_by_scenario(
        self, results_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """
        Compute success rates grouped by scenario type.

        Args:
            results_df: DataFrame with columns 'scenario_type', 'success'

        Returns:
            Dict mapping scenario types to success metrics
        """
        if results_df.empty or "scenario_type" not in results_df.columns:
            return {}

        success_rates = {}

        # Group by scenario type
        for scenario_type in results_df["scenario_type"].unique():
            scenario_data = results_df[results_df["scenario_type"] == scenario_type]

            # Calculate success rate
            total_scenarios = len(scenario_data)

            # Different ways to determine success
            if "success" in scenario_data.columns:
                successful = scenario_data["success"].sum()
            elif "safety_score" in scenario_data.columns:
                # Consider scenarios with safety_score > 0.7 as successful
                successful = (scenario_data["safety_score"] > 0.7).sum()
            elif "conflicts_resolved" in scenario_data.columns:
                successful = scenario_data["conflicts_resolved"].sum()
            else:
                # Default: no conflicts detected = success
                successful = (scenario_data.get("predicted_conflicts", []).apply(len) == 0).sum()

            success_rate = successful / max(1, total_scenarios)

            success_rates[scenario_type] = {
                "success_rate": success_rate,
                "successful_scenarios": int(successful),
                "total_scenarios": total_scenarios,
                "failure_rate": 1 - success_rate,
            }

        return success_rates

    def compute_success_rates_by_group(
        self, results_df: pd.DataFrame, group_cols: list[str],
    ) -> pd.DataFrame:
        """
        Compute success rates grouped by specified columns.

        Args:
            results_df: DataFrame with columns including 'success' and the grouping columns
            group_cols: List of column names to group by (e.g. ['scenario_type', 'complexity_tier', 'distribution_shift'])

        Returns:
            Multi-index DataFrame of success rates grouped by specified columns
        """
        if results_df.empty:
            return pd.DataFrame()

        # Ensure required columns exist
        missing_cols = [col for col in group_cols if col not in results_df.columns]
        if missing_cols:
            self.logger.warning(f"Missing grouping columns: {missing_cols}")
            return pd.DataFrame()

        if "success" not in results_df.columns:
            self.logger.warning("'success' column not found - using alternative success criteria")
            # Try to determine success from other columns
            if "errors" in results_df.columns:
                results_df = results_df.copy()
                results_df["success"] = results_df["errors"].apply(
                    lambda x: len(x) == 0 if isinstance(x, list) else True,
                )
            elif "resolution_success" in results_df.columns:
                results_df = results_df.copy()
                results_df["success"] = results_df["resolution_success"]
            else:
                self.logger.error("Cannot determine success criteria")
                return pd.DataFrame()

        # Group by specified columns and calculate success metrics
        agg_dict = {
            "success": ["count", "sum", "mean"],
        }

        # Add optional columns if they exist
        optional_columns = [
            "detection_accuracy",
            "precision",
            "recall",
            "min_separation_nm",
            "separation_violations",
        ]
        for col in optional_columns:
            if col in results_df.columns:
                if col == "separation_violations":
                    agg_dict[col] = "sum"
                else:
                    agg_dict[col] = "mean"

        grouped = results_df.groupby(group_cols).agg(agg_dict).round(4)

        # Flatten column names
        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

        # Rename for clarity
        grouped = grouped.rename(
            columns={
                "success_count": "total_scenarios",
                "success_sum": "successful_scenarios",
                "success_mean": "success_rate",
            },
        )

        # Add failure scenarios
        grouped["failed_scenarios"] = grouped["total_scenarios"] - grouped["successful_scenarios"]

        return grouped

    def compute_average_separation_margins(self, results_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute average separation margins from results.

        Args:
            results_df: DataFrame with trajectory or margin data

        Returns:
            Dict with horizontal and vertical margin averages
        """
        if results_df.empty:
            return {"avg_horizontal_margin": 0.0, "avg_vertical_margin": 0.0}

        horizontal_margins = []
        vertical_margins = []

        for _, row in results_df.iterrows():
            # Try direct margin columns first
            if "horizontal_margin" in row and pd.notna(row["horizontal_margin"]):
                horizontal_margins.append(row["horizontal_margin"])
            if "vertical_margin" in row and pd.notna(row["vertical_margin"]):
                vertical_margins.append(row["vertical_margin"])

            # Calculate from trajectories if available
            if row.get("trajectories"):
                try:
                    margins = calc_separation_margin(row["trajectories"])
                    if margins["hz"] != float("inf"):
                        horizontal_margins.append(margins["hz"])
                    if margins["vt"] != float("inf"):
                        vertical_margins.append(margins["vt"])
                except Exception as e:
                    self.logger.warning(f"Failed to calculate margins from trajectories: {e}")

        return {
            "avg_horizontal_margin": np.mean(horizontal_margins) if horizontal_margins else 0.0,
            "avg_vertical_margin": np.mean(vertical_margins) if vertical_margins else 0.0,
            "std_horizontal_margin": np.std(horizontal_margins) if horizontal_margins else 0.0,
            "std_vertical_margin": np.std(vertical_margins) if vertical_margins else 0.0,
            "num_margin_samples": len(horizontal_margins),
        }

    def compute_efficiency_penalties(self, results_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute efficiency penalties from trajectory comparisons.

        Args:
            results_df: DataFrame with planned and executed trajectory data

        Returns:
            Dict with efficiency penalty statistics
        """
        if results_df.empty:
            return {"avg_efficiency_penalty": 0.0}

        penalties = []

        for _, row in results_df.iterrows():
            # Try direct penalty column first
            if "efficiency_penalty" in row and pd.notna(row["efficiency_penalty"]):
                penalties.append(row["efficiency_penalty"])
                continue

            # Calculate from trajectory data
            planned_path = row.get("planned_trajectory") or row.get("original_trajectory")
            executed_path = row.get("executed_trajectory") or row.get("actual_trajectory")

            if planned_path and executed_path:
                try:
                    penalty = calc_efficiency_penalty(planned_path, executed_path)
                    penalties.append(penalty)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate efficiency penalty: {e}")

        return {
            "avg_efficiency_penalty": np.mean(penalties) if penalties else 0.0,
            "std_efficiency_penalty": np.std(penalties) if penalties else 0.0,
            "max_efficiency_penalty": np.max(penalties) if penalties else 0.0,
            "num_penalty_samples": len(penalties),
        }

    def generate_report(
        self,
        results_df: pd.DataFrame,
        aggregated_metrics: Optional[dict[str, Any]] = None,
        output_file: Union[str, Path] = "monte_carlo_report.md",
    ) -> str:
        """
        Generate a comprehensive markdown report with all metrics and analysis.

        Args:
            results_df: DataFrame with simulation results
            aggregated_metrics: Pre-computed metrics (if None, will compute from results_df)
            output_file: Path to save the markdown report

        Returns:
            Path to the generated report file
        """
        if aggregated_metrics is None:
            aggregated_metrics = self.aggregate_monte_carlo_metrics(results_df)

        output_file = Path(output_file)

        # Generate grouped success rates for detailed analysis
        group_cols = (
            ["scenario_type", "complexity_tier"]
            if "complexity_tier" in results_df.columns
            else ["scenario_type"]
        )
        if "distribution_shift_level" in results_df.columns:
            group_cols.append("distribution_shift_level")

        grouped_success_rates = self.compute_success_rates_by_group(results_df, group_cols)

        # Build the markdown report
        report_lines = []

        # Header
        report_lines.extend(
            [
                "# Monte Carlo Analysis Report",
                "",
                f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total Scenarios:** {aggregated_metrics['summary']['total_scenarios']}",
                f"**Scenario Types:** {', '.join(aggregated_metrics['summary']['scenario_types'])}",
                "",
                "---",
                "",
            ],
        )

        # Executive Summary
        report_lines.extend(
            [
                "## Executive Summary",
                "",
                self._generate_executive_summary(aggregated_metrics),
                "",
                "---",
                "",
            ],
        )

        # Detection Performance
        detection = aggregated_metrics["detection_performance"]
        report_lines.extend(
            [
                "## Detection Performance",
                "",
                f"- **False Positive Rate:** {detection['false_positive_rate']:.3f}",
                f"- **False Negative Rate:** {detection['false_negative_rate']:.3f}",
                f"- **Total False Positives:** {detection.get('total_false_positives', 'N/A')}",
                f"- **Total False Negatives:** {detection.get('total_false_negatives', 'N/A')}",
                f"- **Total Predicted Conflicts:** {detection.get('total_predicted_conflicts', 'N/A')}",
                f"- **Total Actual Conflicts:** {detection.get('total_actual_conflicts', 'N/A')}",
                "",
                "### Performance Assessment",
                self._assess_detection_performance(detection),
                "",
                "---",
                "",
            ],
        )

        # Success Rates by Scenario Type
        report_lines.extend(
            [
                "## Success Rates by Scenario Type",
                "",
            ],
        )

        success_rates = aggregated_metrics["success_rates_by_scenario"]
        if success_rates:
            for scenario_type, metrics in success_rates.items():
                report_lines.extend(
                    [
                        f"### {scenario_type.title()} Scenarios",
                        f"- **Success Rate:** {metrics['success_rate']:.3f} ({metrics['success_rate']*100:.1f}%)",
                        f"- **Successful Scenarios:** {metrics['successful_scenarios']}/{metrics['total_scenarios']}",
                        f"- **Failure Rate:** {metrics['failure_rate']:.3f} ({metrics['failure_rate']*100:.1f}%)",
                        "",
                    ],
                )
        else:
            report_lines.append("No scenario-specific success rate data available.\n")

        # Detailed Grouped Analysis
        if not grouped_success_rates.empty:
            report_lines.extend(
                [
                    "### Detailed Success Rate Analysis",
                    "",
                    self._format_grouped_success_table(grouped_success_rates),
                    "",
                ],
            )

        report_lines.extend(["---", ""])

        # Safety Margins
        margins = aggregated_metrics["separation_margins"]
        report_lines.extend(
            [
                "## Safety Margins",
                "",
                f"- **Average Horizontal Margin:** {margins['avg_horizontal_margin']:.2f} NM",
                f"- **Average Vertical Margin:** {margins['avg_vertical_margin']:.0f} ft",
                f"- **Std Horizontal Margin:** {margins.get('std_horizontal_margin', 0):.2f} NM",
                f"- **Std Vertical Margin:** {margins.get('std_vertical_margin', 0):.0f} ft",
                f"- **Margin Samples:** {margins.get('num_margin_samples', 0)}",
                "",
                "### Safety Assessment",
                self._assess_safety_margins(margins),
                "",
                "---",
                "",
            ],
        )

        # Efficiency Metrics
        efficiency = aggregated_metrics["efficiency_metrics"]
        report_lines.extend(
            [
                "## Efficiency Metrics",
                "",
                f"- **Average Efficiency Penalty:** {efficiency['avg_efficiency_penalty']:.2f}%",
                f"- **Std Efficiency Penalty:** {efficiency.get('std_efficiency_penalty', 0):.2f}%",
                f"- **Max Efficiency Penalty:** {efficiency.get('max_efficiency_penalty', 0):.2f}%",
                f"- **Penalty Samples:** {efficiency.get('num_penalty_samples', 0)}",
                "",
                "### Efficiency Assessment",
                self._assess_efficiency_performance(efficiency),
                "",
                "---",
                "",
            ],
        )

        # Distribution Shift Analysis
        shift_analysis = aggregated_metrics.get("distribution_shift_analysis", {})
        if shift_analysis:
            report_lines.extend(
                [
                    "## Distribution Shift Analysis",
                    "",
                    self._format_distribution_shift_analysis(shift_analysis),
                    "",
                    "---",
                    "",
                ],
            )

        # Recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                self._generate_recommendations(aggregated_metrics),
                "",
                "---",
                "",
            ],
        )

        # Technical Details
        report_lines.extend(
            [
                "## Technical Details",
                "",
                "- **Analysis Tool:** LLM-ATC Monte Carlo Analyzer",
                f"- **Results File:** {len(results_df)} scenarios",
                f"- **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                f"- **Data Columns:** {', '.join(results_df.columns.tolist())}",
                "",
            ],
        )

        # Write the report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        self.logger.info(f"Monte Carlo report generated: {output_file}")
        return str(output_file)

    def _generate_executive_summary(self, metrics: dict[str, Any]) -> str:
        """Generate executive summary section."""
        detection = metrics["detection_performance"]
        success_rates = metrics["success_rates_by_scenario"]
        margins = metrics["separation_margins"]

        # Calculate overall success rate
        if success_rates:
            overall_success = np.mean([s["success_rate"] for s in success_rates.values()])
        else:
            overall_success = 0.0

        summary = []
        summary.append(
            f"This Monte Carlo analysis evaluated {metrics['summary']['total_scenarios']} scenarios across {len(metrics['summary']['scenario_types'])} scenario types.",
        )

        # Performance assessment
        if overall_success >= 0.9:
            summary.append(
                f"**Overall Performance: EXCELLENT** - Success rate of {overall_success:.1%} indicates robust performance.",
            )
        elif overall_success >= 0.8:
            summary.append(
                f"**Overall Performance: GOOD** - Success rate of {overall_success:.1%} shows generally reliable operation.",
            )
        elif overall_success >= 0.7:
            summary.append(
                f"**Overall Performance: ACCEPTABLE** - Success rate of {overall_success:.1%} suggests room for improvement.",
            )
        else:
            summary.append(
                f"**Overall Performance: NEEDS IMPROVEMENT** - Success rate of {overall_success:.1%} indicates significant issues.",
            )

        # Detection assessment
        fp_rate = detection["false_positive_rate"]
        fn_rate = detection["false_negative_rate"]

        if fp_rate < 0.1 and fn_rate < 0.1:
            summary.append(
                "Detection accuracy is excellent with low false positive and false negative rates.",
            )
        elif fp_rate < 0.2 and fn_rate < 0.2:
            summary.append("Detection accuracy is good but could be improved.")
        else:
            summary.append("Detection accuracy shows significant issues requiring attention.")

        # Safety assessment
        h_margin = margins["avg_horizontal_margin"]
        if h_margin >= 5.0:
            summary.append("Safety margins are well maintained above regulatory minimums.")
        elif h_margin >= 3.0:
            summary.append("Safety margins meet regulatory requirements but are close to limits.")
        else:
            summary.append(
                "**SAFETY CONCERN**: Average horizontal margins below 3 NM indicate potential safety issues.",
            )

        return " ".join(summary)

    def _assess_detection_performance(self, detection: dict[str, float]) -> str:
        """Assess detection performance and provide interpretation."""
        fp_rate = detection["false_positive_rate"]
        fn_rate = detection["false_negative_rate"]

        assessment = []

        if fp_rate < 0.05:
            assessment.append(
                "✅ **False Positive Rate**: Excellent - very few unnecessary alerts.",
            )
        elif fp_rate < 0.15:
            assessment.append("⚠️ **False Positive Rate**: Good - acceptable level of false alerts.")
        else:
            assessment.append(
                "❌ **False Positive Rate**: Poor - too many false alerts may reduce trust.",
            )

        if fn_rate < 0.05:
            assessment.append("✅ **False Negative Rate**: Excellent - very few missed conflicts.")
        elif fn_rate < 0.15:
            assessment.append(
                "⚠️ **False Negative Rate**: Acceptable - some conflicts missed but manageable.",
            )
        else:
            assessment.append(
                "❌ **False Negative Rate**: Dangerous - too many conflicts missed, safety risk.",
            )

        return "\n".join(assessment)

    def _assess_safety_margins(self, margins: dict[str, float]) -> str:
        """Assess safety margin performance."""
        h_margin = margins["avg_horizontal_margin"]
        v_margin = margins["avg_vertical_margin"]

        assessment = []

        # Horizontal margin assessment (5 NM standard, 3 NM minimum)
        if h_margin >= 5.0:
            assessment.append(
                "✅ **Horizontal Margins**: Excellent - well above standard separation.",
            )
        elif h_margin >= 3.0:
            assessment.append(
                "⚠️ **Horizontal Margins**: Acceptable - meeting minimum separation requirements.",
            )
        else:
            assessment.append(
                "❌ **Horizontal Margins**: Critical - below minimum separation standards.",
            )

        # Vertical margin assessment (1000 ft standard)
        if v_margin >= 1000:
            assessment.append(
                "✅ **Vertical Margins**: Excellent - maintaining standard vertical separation.",
            )
        elif v_margin >= 500:
            assessment.append(
                "⚠️ **Vertical Margins**: Marginal - below standard but some separation maintained.",
            )
        else:
            assessment.append(
                "❌ **Vertical Margins**: Critical - insufficient vertical separation.",
            )

        return "\n".join(assessment)

    def _assess_efficiency_performance(self, efficiency: dict[str, float]) -> str:
        """Assess efficiency performance."""
        avg_penalty = efficiency["avg_efficiency_penalty"]

        if avg_penalty < 5.0:
            return "✅ **Efficiency**: Excellent - minimal impact on flight efficiency."
        if avg_penalty < 15.0:
            return "⚠️ **Efficiency**: Acceptable - moderate efficiency impact within acceptable bounds."
        return "❌ **Efficiency**: Poor - significant efficiency penalties affecting operational costs."

    def _format_grouped_success_table(self, grouped_df: pd.DataFrame) -> str:
        """Format grouped success rates as a markdown table."""
        if grouped_df.empty:
            return "No grouped success rate data available."

        lines = [
            "| Group | Success Rate | Successful | Total | Failed |",
            "|-------|--------------|------------|-------|--------|",
        ]

        for index, row in grouped_df.iterrows():
            # Handle multi-index
            if isinstance(index, tuple):
                group_name = " / ".join(str(x) for x in index)
            else:
                group_name = str(index)

            success_rate = row.get("success_rate", 0)
            successful = int(row.get("successful_scenarios", 0))
            total = int(row.get("total_scenarios", 0))
            failed = int(row.get("failed_scenarios", 0))

            lines.append(
                f"| {group_name} | {success_rate:.3f} ({success_rate*100:.1f}%) | {successful} | {total} | {failed} |",
            )

        return "\n".join(lines)

    def _format_distribution_shift_analysis(
        self, shift_analysis: dict[str, dict[str, float]],
    ) -> str:
        """Format distribution shift analysis as markdown."""
        lines = ["Performance degradation analysis across distribution shift levels:", ""]
        lines.extend(
            [
                "| Shift Level | Scenarios | FP Rate | FN Rate | Success Rate | H-Margin |",
                "|-------------|-----------|---------|---------|--------------|----------|",
            ],
        )

        for shift_level, metrics in shift_analysis.items():
            fp_rate = metrics["false_positive_rate"]
            fn_rate = metrics["false_negative_rate"]
            success_rate = metrics["avg_success_rate"]
            h_margin = metrics["avg_horizontal_margin"]
            count = metrics["scenario_count"]

            lines.append(
                f"| {shift_level} | {count} | {fp_rate:.3f} | {fn_rate:.3f} | {success_rate:.3f} | {h_margin:.2f} |",
            )

        return "\n".join(lines)

    def _generate_recommendations(self, metrics: dict[str, Any]) -> str:
        """Generate specific recommendations based on the analysis."""
        recommendations = []

        detection = metrics["detection_performance"]
        success_rates = metrics["success_rates_by_scenario"]
        margins = metrics["separation_margins"]
        efficiency = metrics["efficiency_metrics"]

        # Detection recommendations
        if detection["false_positive_rate"] > 0.15:
            recommendations.append(
                "🔧 **Reduce False Positives**: Consider tuning conflict detection thresholds to reduce unnecessary alerts.",
            )

        if detection["false_negative_rate"] > 0.10:
            recommendations.append(
                "🚨 **Critical - Improve Detection**: False negative rate is concerning. Review detection algorithms immediately.",
            )

        # Success rate recommendations
        if success_rates:
            worst_scenario = min(success_rates.items(), key=lambda x: x[1]["success_rate"])
            if worst_scenario[1]["success_rate"] < 0.7:
                recommendations.append(
                    f"📊 **Focus on {worst_scenario[0]} Scenarios**: Success rate of {worst_scenario[1]['success_rate']:.1%} needs attention.",
                )

        # Safety margin recommendations
        if margins["avg_horizontal_margin"] < 4.0:
            recommendations.append(
                "⚠️ **Improve Safety Margins**: Horizontal margins are close to minimum standards. Consider more conservative conflict resolution.",
            )

        # Efficiency recommendations
        if efficiency["avg_efficiency_penalty"] > 20.0:
            recommendations.append(
                "✈️ **Optimize Efficiency**: High efficiency penalties suggest room for route optimization improvements.",
            )

        # Distribution shift recommendations
        shift_analysis = metrics.get("distribution_shift_analysis", {})
        if shift_analysis:
            # Check for performance degradation
            shift_levels = list(shift_analysis.keys())
            if len(shift_levels) > 1:
                baseline = shift_analysis[shift_levels[0]]
                worst = shift_analysis[shift_levels[-1]]

                if worst["avg_success_rate"] < baseline["avg_success_rate"] * 0.8:
                    recommendations.append(
                        "🎯 **Address Distribution Shift**: Significant performance degradation under distribution shift. Consider domain adaptation techniques.",
                    )

        if not recommendations:
            recommendations.append(
                "✅ **Overall Good Performance**: No critical issues identified. Continue monitoring and gradual improvements.",
            )

        return "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))

    def aggregate_monte_carlo_metrics(self, results_df: pd.DataFrame) -> dict[str, Any]:
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
        scenario_types = results_df.get("scenario_type", pd.Series()).unique().tolist()

        # Distribution shift analysis if available
        shift_analysis = {}
        if "distribution_shift_level" in results_df.columns:
            shift_analysis = self._analyze_distribution_shift_performance(results_df)

        aggregated_metrics = {
            "summary": {
                "total_scenarios": total_scenarios,
                "scenario_types": scenario_types,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            },
            "detection_performance": fp_fn_rates,
            "success_rates_by_scenario": success_rates,
            "separation_margins": separation_margins,
            "efficiency_metrics": efficiency_penalties,
            "distribution_shift_analysis": shift_analysis,
        }

        self.logger.info("Monte Carlo metrics aggregation completed")
        return aggregated_metrics

    def _analyze_distribution_shift_performance(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze performance across different distribution shift levels."""
        shift_analysis = {}

        for shift_level in results_df["distribution_shift_level"].unique():
            shift_data = results_df[results_df["distribution_shift_level"] == shift_level]

            # Calculate metrics for this shift level
            fp_fn = self.compute_false_positive_negative_rates(shift_data)
            success = self.compute_success_rates_by_scenario(shift_data)
            margins = self.compute_average_separation_margins(shift_data)

            shift_analysis[shift_level] = {
                "scenario_count": len(shift_data),
                "false_positive_rate": fp_fn["false_positive_rate"],
                "false_negative_rate": fp_fn["false_negative_rate"],
                "avg_success_rate": (
                    np.mean([s["success_rate"] for s in success.values()]) if success else 0.0
                ),
                "avg_horizontal_margin": margins["avg_horizontal_margin"],
                "avg_vertical_margin": margins["avg_vertical_margin"],
            }

        return shift_analysis

    def _create_empty_aggregated_metrics(self) -> dict[str, Any]:
        """Create empty metrics structure for error cases."""
        return {
            "summary": {
                "total_scenarios": 0,
                "scenario_types": [],
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            },
            "detection_performance": {
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            },
            "success_rates_by_scenario": {},
            "separation_margins": {
                "avg_horizontal_margin": 0.0,
                "avg_vertical_margin": 0.0,
            },
            "efficiency_metrics": {
                "avg_efficiency_penalty": 0.0,
            },
            "distribution_shift_analysis": {},
        }


class MonteCarloVisualizer:
    """
    Creates visualizations for Monte Carlo analysis results.
    """

    def __init__(self) -> None:
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)

        if not PLOTTING_AVAILABLE:
            self.logger.warning("Plotting libraries not available - visualizations disabled")

    def create_performance_summary_charts(
        self, aggregated_metrics: dict[str, Any], output_dir: Union[str, Path] = "monte_carlo_plots",
    ) -> list[str]:
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
            aggregated_metrics["success_rates_by_scenario"],
            output_dir / "success_rates_by_scenario.png",
        )
        if success_plot:
            created_plots.append(success_plot)

        # 2. Detection performance (FP/FN rates)
        detection_plot = self._create_detection_performance_chart(
            aggregated_metrics["detection_performance"],
            output_dir / "detection_performance.png",
        )
        if detection_plot:
            created_plots.append(detection_plot)

        # 3. Safety margins comparison
        margins_plot = self._create_safety_margins_chart(
            aggregated_metrics["separation_margins"],
            output_dir / "safety_margins.png",
        )
        if margins_plot:
            created_plots.append(margins_plot)

        return created_plots

    def create_distribution_shift_plots(
        self, aggregated_metrics: dict[str, Any], output_dir: Union[str, Path] = "monte_carlo_plots",
    ) -> list[str]:
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

        shift_analysis = aggregated_metrics.get("distribution_shift_analysis", {})
        if not shift_analysis:
            self.logger.warning("No distribution shift data available for plotting")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        created_plots = []

        # Create scatter plot of performance vs distribution shift level
        plot_path = self._create_shift_performance_scatter(
            shift_analysis,
            output_dir / "distribution_shift_performance.png",
        )
        if plot_path:
            created_plots.append(plot_path)

        return created_plots

    def _create_success_rate_chart(
        self, success_data: dict[str, dict[str, float]], save_path: Path,
    ) -> Optional[str]:
        """Create bar chart of success rates by scenario type."""
        try:
            if not success_data:
                return None

            fig, ax = plt.subplots(figsize=(10, 6))

            scenario_types = list(success_data.keys())
            success_rates = [success_data[st]["success_rate"] for st in scenario_types]
            total_scenarios = [success_data[st]["total_scenarios"] for st in scenario_types]

            bars = ax.bar(scenario_types, success_rates, alpha=0.8, color="skyblue")

            # Add value labels on bars
            for _i, (bar, count) in enumerate(zip(bars, total_scenarios)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.2%}\n(n={count})",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            ax.set_ylabel("Success Rate")
            ax.set_title("ATC Resolution Success Rates by Scenario Type")
            ax.set_ylim(0, 1.1)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Success rate chart saved to {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.exception(f"Failed to create success rate chart: {e}")
            return None

    def _create_detection_performance_chart(
        self, detection_data: dict[str, float], save_path: Path,
    ) -> Optional[str]:
        """Create bar chart of false positive/negative rates."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            metrics = ["False Positive Rate", "False Negative Rate"]
            values = [
                detection_data.get("false_positive_rate", 0),
                detection_data.get("false_negative_rate", 0),
            ]
            colors = ["lightcoral", "lightsalmon"]

            bars = ax.bar(metrics, values, color=colors, alpha=0.8)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.005,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

            ax.set_ylabel("Rate")
            ax.set_title("Conflict Detection Performance")
            ax.set_ylim(0, max(values) * 1.2 if values else 0.1)
            plt.tight_layout()

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Detection performance chart saved to {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.exception(f"Failed to create detection performance chart: {e}")
            return None

    def _create_safety_margins_chart(
        self, margins_data: dict[str, float], save_path: Path,
    ) -> Optional[str]:
        """Create bar chart of safety margins."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            margin_types = ["Horizontal Margin (NM)", "Vertical Margin (ft)"]
            values = [
                margins_data.get("avg_horizontal_margin", 0),
                margins_data.get("avg_vertical_margin", 0),
            ]
            colors = ["lightgreen", "lightblue"]

            bars = ax.bar(margin_types, values, color=colors, alpha=0.8)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(values) * 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

            ax.set_ylabel("Margin")
            ax.set_title("Average Safety Separation Margins")
            ax.set_ylim(0, max(values) * 1.2 if values else 10)
            plt.xticks(rotation=15)
            plt.tight_layout()

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Safety margins chart saved to {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.exception(f"Failed to create safety margins chart: {e}")
            return None

    def _create_shift_performance_scatter(
        self, shift_data: dict[str, dict[str, float]], save_path: Path,
    ) -> Optional[str]:
        """Create scatter plot of performance vs distribution shift level."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            shift_levels = list(shift_data.keys())
            shift_numeric = range(len(shift_levels))  # Convert to numeric for plotting

            # Extract metrics
            fp_rates = [shift_data[sl]["false_positive_rate"] for sl in shift_levels]
            fn_rates = [shift_data[sl]["false_negative_rate"] for sl in shift_levels]
            success_rates = [shift_data[sl]["avg_success_rate"] for sl in shift_levels]
            h_margins = [shift_data[sl]["avg_horizontal_margin"] for sl in shift_levels]

            # Plot 1: False Positive Rate vs Shift Level
            ax1.scatter(shift_numeric, fp_rates, color="red", alpha=0.7, s=60)
            ax1.plot(shift_numeric, fp_rates, "r--", alpha=0.5)
            ax1.set_title("False Positive Rate vs Distribution Shift")
            ax1.set_ylabel("False Positive Rate")
            ax1.set_xticks(shift_numeric)
            ax1.set_xticklabels(shift_levels, rotation=45)

            # Plot 2: False Negative Rate vs Shift Level
            ax2.scatter(shift_numeric, fn_rates, color="orange", alpha=0.7, s=60)
            ax2.plot(shift_numeric, fn_rates, "orange", linestyle="--", alpha=0.5)
            ax2.set_title("False Negative Rate vs Distribution Shift")
            ax2.set_ylabel("False Negative Rate")
            ax2.set_xticks(shift_numeric)
            ax2.set_xticklabels(shift_levels, rotation=45)

            # Plot 3: Success Rate vs Shift Level
            ax3.scatter(shift_numeric, success_rates, color="green", alpha=0.7, s=60)
            ax3.plot(shift_numeric, success_rates, "g--", alpha=0.5)
            ax3.set_title("Success Rate vs Distribution Shift")
            ax3.set_ylabel("Success Rate")
            ax3.set_xticks(shift_numeric)
            ax3.set_xticklabels(shift_levels, rotation=45)

            # Plot 4: Horizontal Margin vs Shift Level
            ax4.scatter(shift_numeric, h_margins, color="blue", alpha=0.7, s=60)
            ax4.plot(shift_numeric, h_margins, "b--", alpha=0.5)
            ax4.set_title("Horizontal Margin vs Distribution Shift")
            ax4.set_ylabel("Horizontal Margin (NM)")
            ax4.set_xticks(shift_numeric)
            ax4.set_xticklabels(shift_levels, rotation=45)

            plt.suptitle("Performance Degradation Under Distribution Shift")
            plt.tight_layout()

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Distribution shift scatter plot saved to {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.exception(f"Failed to create distribution shift scatter plot: {e}")
            return None


# Convenience functions for direct usage
def analyze_monte_carlo_results(
    results_file: Union[str, Path], output_dir: Union[str, Path] = "monte_carlo_analysis",
) -> dict[str, Any]:
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
        with open(metrics_file, "w") as f:
            json.dump(aggregated_metrics, f, indent=2, default=str)

        # Create visualizations
        summary_plots = visualizer.create_performance_summary_charts(aggregated_metrics, output_dir)
        shift_plots = visualizer.create_distribution_shift_plots(aggregated_metrics, output_dir)

        # Return complete analysis
        return {
            "metrics": aggregated_metrics,
            "metrics_file": str(metrics_file),
            "summary_plots": summary_plots,
            "distribution_shift_plots": shift_plots,
            "output_directory": str(output_dir),
        }

    except Exception as e:
        logging.exception(f"Monte Carlo analysis failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import tempfile

    # Create sample data for demonstration
    sample_data = [
        {
            "scenario_type": "horizontal",
            "success": True,
            "predicted_conflicts": [],
            "actual_conflicts": [],
            "horizontal_margin": 6.2,
            "vertical_margin": 1200,
            "efficiency_penalty": 2.1,
        },
        {
            "scenario_type": "vertical",
            "success": False,
            "predicted_conflicts": [{"aircraft_1": "AC001", "aircraft_2": "AC002"}],
            "actual_conflicts": [{"aircraft_1": "AC001", "aircraft_2": "AC002"}],
            "horizontal_margin": 3.8,
            "vertical_margin": 800,
            "efficiency_penalty": 5.3,
        },
    ]

    # Save sample data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        sample_file = f.name

    # Run analysis
    try:
        results = analyze_monte_carlo_results(sample_file)

        # Print key metrics
        metrics = results["metrics"]

    except Exception:
        pass
    finally:
        # Cleanup
        Path(sample_file).unlink(missing_ok=True)
