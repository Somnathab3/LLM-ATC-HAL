"""
Compare LLM Model Performance
Analyzes and compares the performance of the fine-tuned BlueSky Gym model with the baseline model
"""

import pandas as pd
import json
import re
from pathlib import Path


def extract_llm_response_info(csv_path):
    """Extract key information from LLM responses in the CSV"""
    df = pd.read_csv(csv_path)

    results = []
    for idx, row in df.iterrows():
        scenario_id = row["scenario_id"]
        llm_response = row["llm_response"]
        resolution_response = row["resolution_response"]

        # Try to extract conflict detection from response
        conflict_detected = False
        confidence = 0.5

        if isinstance(llm_response, str):
            # Look for JSON-like structure
            if "conflict_detected" in llm_response:
                if "true" in llm_response.lower():
                    conflict_detected = True
                elif "false" in llm_response.lower():
                    conflict_detected = False

            # Extract confidence if available
            conf_match = re.search(r'"confidence":\s*([0-9.]+)', llm_response)
            if conf_match:
                confidence = float(conf_match.group(1))

        # Extract resolution information
        resolution_info = {
            "command": "No command",
            "rationale": "No rationale",
            "confidence": 0.5,
        }

        if isinstance(resolution_response, str):
            # Look for command
            cmd_match = re.search(r"COMMAND:\s*([^\n]+)", resolution_response)
            if cmd_match:
                resolution_info["command"] = cmd_match.group(1).strip()

            # Look for rationale
            rat_match = re.search(r"RATIONALE:\s*([^\n]+)", resolution_response)
            if rat_match:
                resolution_info["rationale"] = rat_match.group(1).strip()

            # Look for confidence
            conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", resolution_response)
            if conf_match:
                resolution_info["confidence"] = float(conf_match.group(1))

        results.append(
            {
                "scenario_id": scenario_id,
                "scenario_type": row["scenario_type"],
                "complexity": row["complexity_tier"],
                "ground_truth_conflicts": row["ground_truth_conflicts"],
                "llm_conflicts": row["llm_conflicts"],
                "llm_confidence": confidence,
                "detection_accuracy": row["detection_accuracy"],
                "resolution_command": resolution_info["command"],
                "resolution_rationale": resolution_info["rationale"],
                "resolution_confidence": resolution_info["confidence"],
                "execution_time_ms": row["execution_time_ms"],
            }
        )

    return results


def compare_models():
    """Compare the performance of both models"""

    bsky_gym_path = Path(
        "experiments/bsky_gym_model_test/benchmark_83ce5fd0_20250801_125603/detection_comparison.csv"
    )
    baseline_path = Path(
        "experiments/baseline_model_test/benchmark_73b5dfc4_20250801_125713/detection_comparison.csv"
    )

    print("=== LLM Model Performance Comparison ===\n")

    if bsky_gym_path.exists():
        print("📊 BlueSky Gym Fine-tuned Model Results:")
        bsky_results = extract_llm_response_info(bsky_gym_path)

        print(f"Total scenarios: {len(bsky_results)}")

        # Calculate averages (handle non-numeric values)
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        valid_detection = [safe_float(r["detection_accuracy"]) for r in bsky_results]
        valid_llm_conf = [safe_float(r["llm_confidence"]) for r in bsky_results]
        valid_res_conf = [safe_float(r["resolution_confidence"]) for r in bsky_results]
        valid_exec_time = [safe_float(r["execution_time_ms"]) for r in bsky_results]

        avg_detection_accuracy = (
            sum(valid_detection) / len(valid_detection) if valid_detection else 0
        )
        avg_llm_confidence = (
            sum(valid_llm_conf) / len(valid_llm_conf) if valid_llm_conf else 0
        )
        avg_resolution_confidence = (
            sum(valid_res_conf) / len(valid_res_conf) if valid_res_conf else 0
        )
        avg_execution_time = (
            sum(valid_exec_time) / len(valid_exec_time) if valid_exec_time else 0
        )

        print(f"Average Detection Accuracy: {avg_detection_accuracy:.3f}")
        print(f"Average LLM Confidence: {avg_llm_confidence:.3f}")
        print(f"Average Resolution Confidence: {avg_resolution_confidence:.3f}")
        print(f"Average Execution Time: {avg_execution_time:.2f}ms")

        # Show some examples
        print("\n🔍 Sample BlueSky Gym Model Responses:")
        for i, result in enumerate(bsky_results[:3]):
            print(f"\nScenario {i+1}: {result['scenario_id']}")
            print(f"  Type: {result['scenario_type']} - {result['complexity']}")
            print(f"  Ground Truth: {result['ground_truth_conflicts']} conflicts")
            print(
                f"  LLM Detected: {result['llm_conflicts']} conflicts (confidence: {result['llm_confidence']:.2f})"
            )
            print(f"  Detection Accuracy: {result['detection_accuracy']:.3f}")
            print(f"  Resolution: {result['resolution_command']}")
            print(f"  Rationale: {result['resolution_rationale']}")
    else:
        print("❌ BlueSky Gym model results not found")

    print("\n" + "=" * 60 + "\n")

    if baseline_path.exists():
        print("📊 Baseline Model (llama3.1:8b) Results:")
        baseline_results = extract_llm_response_info(baseline_path)

        print(f"Total scenarios: {len(baseline_results)}")

        # Calculate averages (handle non-numeric values)
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        valid_detection = [
            safe_float(r["detection_accuracy"]) for r in baseline_results
        ]
        valid_llm_conf = [safe_float(r["llm_confidence"]) for r in baseline_results]
        valid_res_conf = [
            safe_float(r["resolution_confidence"]) for r in baseline_results
        ]
        valid_exec_time = [safe_float(r["execution_time_ms"]) for r in baseline_results]

        avg_detection_accuracy = (
            sum(valid_detection) / len(valid_detection) if valid_detection else 0
        )
        avg_llm_confidence = (
            sum(valid_llm_conf) / len(valid_llm_conf) if valid_llm_conf else 0
        )
        avg_resolution_confidence = (
            sum(valid_res_conf) / len(valid_res_conf) if valid_res_conf else 0
        )
        avg_execution_time = (
            sum(valid_exec_time) / len(valid_exec_time) if valid_exec_time else 0
        )

        print(f"Average Detection Accuracy: {avg_detection_accuracy:.3f}")
        print(f"Average LLM Confidence: {avg_llm_confidence:.3f}")
        print(f"Average Resolution Confidence: {avg_resolution_confidence:.3f}")
        print(f"Average Execution Time: {avg_execution_time:.2f}ms")

        # Show some examples
        print("\n🔍 Sample Baseline Model Responses:")
        for i, result in enumerate(baseline_results[:3]):
            print(f"\nScenario {i+1}: {result['scenario_id']}")
            print(f"  Type: {result['scenario_type']} - {result['complexity']}")
            print(f"  Ground Truth: {result['ground_truth_conflicts']} conflicts")
            print(
                f"  LLM Detected: {result['llm_conflicts']} conflicts (confidence: {result['llm_confidence']:.2f})"
            )
            print(f"  Detection Accuracy: {result['detection_accuracy']:.3f}")
            print(f"  Resolution: {result['resolution_command']}")
            print(f"  Rationale: {result['resolution_rationale']}")
    else:
        print("❌ Baseline model results not found")


if __name__ == "__main__":
    compare_models()
