"""
Model Evaluation Script for Fine-tuned Ollama Models
Evaluates fine-tuned models against test datasets and benchmarks
"""

import json
import logging
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import ollama
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result"""

    input_text: str
    expected_output: str
    actual_output: str
    response_time: float
    safety_score: float
    relevance_score: float
    metadata: Dict[str, Any]


class ModelEvaluator:
    """Evaluate fine-tuned Ollama models"""

    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize evaluator"""
        self.config = self._load_config(config_path)
        self.client = ollama.Client()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation configuration"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def evaluate_model(
        self, model_name: str, test_data_path: str, output_dir: str = "logs"
    ) -> Dict[str, Any]:
        """Evaluate a model against test data"""

        logger.info(f"Evaluating model: {model_name}")

        # Load test data
        test_examples = self._load_test_data(test_data_path)
        logger.info(f"Loaded {len(test_examples)} test examples")

        # Run evaluation
        results = []
        for i, example in enumerate(test_examples):
            if i >= self.config["evaluation"]["benchmark_scenarios"]:
                break

            try:
                result = self._evaluate_single_example(model_name, example)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(test_examples)} examples")

            except Exception as e:
                logger.warning(f"Failed to evaluate example {i}: {e}")

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self._save_results(results, metrics, model_name, output_path)

        logger.info(f"Evaluation completed. Results saved to {output_path}")
        return metrics

    def _load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """Load test data from JSONL file"""

        examples = []
        with open(test_data_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    examples.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

        return examples

    def _evaluate_single_example(
        self, model_name: str, example: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single example"""

        input_text = example["input"]
        expected_output = example["output"]
        metadata = example.get("metadata", {})

        # Query the model
        start_time = time.time()
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{"role": "user", "content": input_text}],
                options={"temperature": 0.1, "top_p": 0.9, "num_predict": 500},
            )
            actual_output = response["message"]["content"]
        except Exception as e:
            logger.warning(f"Model query failed: {e}")
            actual_output = f"ERROR: {str(e)}"

        response_time = time.time() - start_time

        # Calculate scores
        safety_score = self._calculate_safety_score(actual_output)
        relevance_score = self._calculate_relevance_score(
            expected_output, actual_output
        )

        return EvaluationResult(
            input_text=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
            response_time=response_time,
            safety_score=safety_score,
            relevance_score=relevance_score,
            metadata=metadata,
        )

    def _calculate_safety_score(self, response: str) -> float:
        """Calculate safety score for response"""

        # Safety indicators
        safety_keywords = [
            "maintain",
            "altitude",
            "heading",
            "separation",
            "turn",
            "climb",
            "descend",
            "speed",
            "knots",
            "degrees",
            "feet",
            "nautical miles",
            "safe",
            "standard",
        ]

        # Unsafe indicators
        unsafe_keywords = [
            "crash",
            "collision",
            "emergency",
            "danger",
            "unsafe",
            "violation",
            "critical",
            "fail",
            "error",
        ]

        # Precision indicators
        precision_keywords = [
            "degrees",
            "feet",
            "knots",
            "nautical miles",
            "flight level",
            "FL",
            "nm",
            "ft",
            "kts",
        ]

        response_lower = response.lower()

        # Count indicators
        safety_count = sum(
            1 for keyword in safety_keywords if keyword in response_lower
        )
        unsafe_count = sum(
            1 for keyword in unsafe_keywords if keyword in response_lower
        )
        precision_count = sum(
            1 for keyword in precision_keywords if keyword in response_lower
        )

        # Calculate score
        base_score = min(1.0, safety_count * 0.1)
        precision_bonus = min(0.3, precision_count * 0.1)
        unsafe_penalty = unsafe_count * 0.5

        score = base_score + precision_bonus - unsafe_penalty
        return max(0.0, min(1.0, score))

    def _calculate_relevance_score(self, expected: str, actual: str) -> float:
        """Calculate relevance score between expected and actual outputs"""

        # Simple word overlap scoring
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words:
            return 0.0

        # Calculate overlap
        overlap = len(expected_words.intersection(actual_words))
        relevance = overlap / len(expected_words)

        # Bonus for action words
        action_words = {
            "turn",
            "climb",
            "descend",
            "maintain",
            "increase",
            "decrease",
            "altitude",
            "heading",
            "speed",
        }
        expected_actions = expected_words.intersection(action_words)
        actual_actions = actual_words.intersection(action_words)

        if expected_actions:
            action_overlap = len(expected_actions.intersection(actual_actions))
            action_score = action_overlap / len(expected_actions)
            relevance = 0.7 * relevance + 0.3 * action_score

        return min(1.0, relevance)

    def _calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""

        if not results:
            return {"error": "No results to calculate metrics"}

        # Extract scores
        safety_scores = [r.safety_score for r in results]
        relevance_scores = [r.relevance_score for r in results]
        response_times = [r.response_time for r in results]

        # Calculate statistics
        metrics = {
            "total_examples": len(results),
            "safety_score": {
                "mean": float(np.mean(safety_scores)),
                "std": float(np.std(safety_scores)),
                "min": float(np.min(safety_scores)),
                "max": float(np.max(safety_scores)),
                "median": float(np.median(safety_scores)),
            },
            "relevance_score": {
                "mean": float(np.mean(relevance_scores)),
                "std": float(np.std(relevance_scores)),
                "min": float(np.min(relevance_scores)),
                "max": float(np.max(relevance_scores)),
                "median": float(np.median(relevance_scores)),
            },
            "response_time": {
                "mean": float(np.mean(response_times)),
                "std": float(np.std(response_times)),
                "min": float(np.min(response_times)),
                "max": float(np.max(response_times)),
                "median": float(np.median(response_times)),
            },
        }

        # Calculate derived metrics
        high_safety_count = sum(1 for score in safety_scores if score >= 0.7)
        high_relevance_count = sum(1 for score in relevance_scores if score >= 0.5)

        metrics["performance"] = {
            "high_safety_rate": high_safety_count / len(results),
            "high_relevance_rate": high_relevance_count / len(results),
            "combined_score": (
                metrics["safety_score"]["mean"] + metrics["relevance_score"]["mean"]
            )
            / 2,
        }

        # Environment-specific metrics
        env_metrics = {}
        for result in results:
            env = result.metadata.get("environment", "unknown")
            if env not in env_metrics:
                env_metrics[env] = {"safety_scores": [], "relevance_scores": []}
            env_metrics[env]["safety_scores"].append(result.safety_score)
            env_metrics[env]["relevance_scores"].append(result.relevance_score)

        metrics["environment_performance"] = {}
        for env, scores in env_metrics.items():
            metrics["environment_performance"][env] = {
                "safety_mean": float(np.mean(scores["safety_scores"])),
                "relevance_mean": float(np.mean(scores["relevance_scores"])),
                "count": len(scores["safety_scores"]),
            }

        return metrics

    def _save_results(
        self,
        results: List[EvaluationResult],
        metrics: Dict[str, Any],
        model_name: str,
        output_path: Path,
    ) -> None:
        """Save evaluation results"""

        timestamp = int(time.time())
        safe_model_name = model_name.replace(":", "_").replace("/", "_")

        # Save detailed results
        results_file = (
            output_path / f"evaluation_results_{safe_model_name}_{timestamp}.json"
        )
        detailed_results = []

        for result in results:
            detailed_results.append(
                {
                    "input": result.input_text,
                    "expected_output": result.expected_output,
                    "actual_output": result.actual_output,
                    "response_time": result.response_time,
                    "safety_score": result.safety_score,
                    "relevance_score": result.relevance_score,
                    "metadata": result.metadata,
                }
            )

        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save metrics summary
        metrics_file = (
            output_path / f"evaluation_metrics_{safe_model_name}_{timestamp}.json"
        )
        evaluation_summary = {
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "config": self.config,
        }

        with open(metrics_file, "w") as f:
            json.dump(evaluation_summary, f, indent=2)

        # Save human-readable report
        report_file = (
            output_path / f"evaluation_report_{safe_model_name}_{timestamp}.txt"
        )
        self._generate_report(metrics, model_name, report_file)

        logger.info(f"Results saved:")
        logger.info(f"  Detailed: {results_file}")
        logger.info(f"  Metrics: {metrics_file}")
        logger.info(f"  Report: {report_file}")

    def _generate_report(
        self, metrics: Dict[str, Any], model_name: str, report_file: Path
    ) -> None:
        """Generate human-readable evaluation report"""

        with open(report_file, "w") as f:
            f.write(f"Evaluation Report: {model_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Examples: {metrics['total_examples']}\n\n")

            f.write("Safety Score:\n")
            f.write(f"  Mean: {metrics['safety_score']['mean']:.3f}\n")
            f.write(f"  Std:  {metrics['safety_score']['std']:.3f}\n")
            f.write(
                f"  Range: {metrics['safety_score']['min']:.3f} - {metrics['safety_score']['max']:.3f}\n\n"
            )

            f.write("Relevance Score:\n")
            f.write(f"  Mean: {metrics['relevance_score']['mean']:.3f}\n")
            f.write(f"  Std:  {metrics['relevance_score']['std']:.3f}\n")
            f.write(
                f"  Range: {metrics['relevance_score']['min']:.3f} - {metrics['relevance_score']['max']:.3f}\n\n"
            )

            f.write("Response Time:\n")
            f.write(f"  Mean: {metrics['response_time']['mean']:.3f}s\n")
            f.write(f"  Std:  {metrics['response_time']['std']:.3f}s\n")
            f.write(
                f"  Range: {metrics['response_time']['min']:.3f}s - {metrics['response_time']['max']:.3f}s\n\n"
            )

            f.write("Performance Metrics:\n")
            f.write(
                f"  High Safety Rate: {metrics['performance']['high_safety_rate']:.3f}\n"
            )
            f.write(
                f"  High Relevance Rate: {metrics['performance']['high_relevance_rate']:.3f}\n"
            )
            f.write(
                f"  Combined Score: {metrics['performance']['combined_score']:.3f}\n\n"
            )

            if "environment_performance" in metrics:
                f.write("Environment-Specific Performance:\n")
                for env, perf in metrics["environment_performance"].items():
                    f.write(f"  {env}:\n")
                    f.write(f"    Safety: {perf['safety_mean']:.3f}\n")
                    f.write(f"    Relevance: {perf['relevance_mean']:.3f}\n")
                    f.write(f"    Count: {perf['count']}\n")

    def compare_models(
        self, model_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple model evaluation results"""

        comparison = {"models": list(model_results.keys()), "comparison_metrics": {}}

        # Compare key metrics
        for metric_name in ["safety_score", "relevance_score", "response_time"]:
            comparison["comparison_metrics"][metric_name] = {}

            for model_name, results in model_results.items():
                if metric_name in results["metrics"]:
                    comparison["comparison_metrics"][metric_name][model_name] = results[
                        "metrics"
                    ][metric_name]["mean"]

        # Determine best model for each metric
        comparison["best_models"] = {}
        for metric_name, model_scores in comparison["comparison_metrics"].items():
            if model_scores:
                if metric_name == "response_time":
                    # Lower is better for response time
                    best_model = min(model_scores, key=model_scores.get)
                else:
                    # Higher is better for safety and relevance
                    best_model = max(model_scores, key=model_scores.get)
                comparison["best_models"][metric_name] = best_model

        return comparison


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Ollama models")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument(
        "--test-data", required=True, help="Path to test data JSONL file"
    )
    parser.add_argument(
        "--output-dir", default="logs", help="Output directory for results"
    )
    parser.add_argument(
        "--config", default="config/training_config.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--compare", nargs="+", help="Compare multiple model result files"
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.config)

    if args.compare:
        # Compare multiple models
        model_results = {}
        for result_file in args.compare:
            with open(result_file, "r") as f:
                data = json.load(f)
                model_name = data["model_name"]
                model_results[model_name] = data

        comparison = evaluator.compare_models(model_results)

        comparison_file = (
            Path(args.output_dir) / f"model_comparison_{int(time.time())}.json"
        )
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"Model comparison saved to: {comparison_file}")

        # Print summary
        print("\nModel Comparison Summary:")
        print("-" * 30)
        for metric, best_model in comparison["best_models"].items():
            print(f"{metric}: {best_model}")

    else:
        # Evaluate single model
        metrics = evaluator.evaluate_model(args.model, args.test_data, args.output_dir)

        print(f"\nEvaluation Results for {args.model}:")
        print("-" * 40)
        print(
            f"Safety Score: {metrics['safety_score']['mean']:.3f} ± {metrics['safety_score']['std']:.3f}"
        )
        print(
            f"Relevance Score: {metrics['relevance_score']['mean']:.3f} ± {metrics['relevance_score']['std']:.3f}"
        )
        print(
            f"Response Time: {metrics['response_time']['mean']:.3f}s ± {metrics['response_time']['std']:.3f}s"
        )
        print(f"Combined Score: {metrics['performance']['combined_score']:.3f}")


if __name__ == "__main__":
    main()
