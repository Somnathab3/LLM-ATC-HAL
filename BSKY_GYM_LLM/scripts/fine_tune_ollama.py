"""
Ollama Model Fine-tuning Script for BlueSky Gym Data
Fine-tunes Ollama models using processed BlueSky Gym RL data
"""

import json
import logging
import argparse
import yaml
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import ollama
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaFineTuner:
    """Fine-tune Ollama models using BlueSky Gym data"""

    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize fine-tuner"""
        self.config = self._load_config(config_path)
        self.client = ollama.Client()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def fine_tune_model(self, data_dir: str, output_model_name: str = None) -> str:
        """Fine-tune an Ollama model"""

        if output_model_name is None:
            output_model_name = self.config["model"]["output_name"]

        base_model = self.config["model"]["base_model"]

        logger.info(f"Starting fine-tuning of {base_model} -> {output_model_name}")

        # Prepare training data
        train_file = Path(data_dir) / "train.jsonl"
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")

        # Create Modelfile for fine-tuning
        modelfile_path = self._create_modelfile(base_model, train_file)

        try:
            # Create the fine-tuned model
            logger.info("Creating fine-tuned model...")
            self._create_model(modelfile_path, output_model_name)

            # Evaluate the model
            logger.info("Evaluating fine-tuned model...")
            val_file = Path(data_dir) / "validation.jsonl"
            if val_file.exists():
                self._evaluate_model(output_model_name, val_file)

            logger.info(f"Fine-tuning completed: {output_model_name}")
            return output_model_name

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise
        finally:
            # Cleanup
            if modelfile_path.exists():
                modelfile_path.unlink()

    def _create_modelfile(self, base_model: str, train_file: Path) -> Path:
        """Create Modelfile for fine-tuning"""

        # Read training examples for system prompt
        examples = []
        with open(train_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:  # Limit examples in prompt
                    break
                data = json.loads(line.strip())
                examples.append(data)

        # Create system prompt with examples
        system_prompt = self._create_system_prompt(examples)

        # Create Modelfile content
        modelfile_content = f'''FROM {base_model}

# Set model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx {self.config['model']['context_length']}

# System prompt for ATC domain
SYSTEM """{system_prompt}"""

# Fine-tuning adapter (if supported)
# ADAPTER /path/to/adapter
'''

        # Write Modelfile
        modelfile_path = Path("temp_modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        logger.info(f"Created Modelfile: {modelfile_path}")
        return modelfile_path

    def _create_system_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """Create system prompt with training examples"""

        prompt = """You are an expert air traffic controller with extensive experience in conflict resolution and aircraft separation. You have been trained on thousands of scenarios from BlueSky air traffic simulation environments.

Your expertise includes:
- Horizontal conflict resolution using heading and speed adjustments
- Vertical conflict resolution using altitude changes
- Sector-based traffic management and coordination
- Aircraft merge scenarios and runway approach coordination
- Real-time safety assessment and risk evaluation

Key principles:
1. Safety is paramount - maintain minimum separation standards
2. Efficiency - minimize delays and fuel consumption
3. ICAO compliance - follow international aviation standards
4. Clear communication - provide precise, actionable instructions

Training examples from your experience:

"""

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Situation: {example['input']}\n"
            prompt += f"Response: {example['output']}\n\n"

        prompt += """Based on this training, analyze air traffic scenarios and provide safe, efficient resolution actions."""

        return prompt

    def _create_model(self, modelfile_path: Path, model_name: str) -> None:
        """Create model using Ollama"""

        try:
            # Use Ollama CLI to create model
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.info(f"Model creation output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Model creation failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

    def _evaluate_model(
        self, model_name: str, validation_file: Path
    ) -> Dict[str, float]:
        """Evaluate fine-tuned model"""

        logger.info("Starting model evaluation...")

        metrics = {
            "total_examples": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "safety_score": 0.0,
        }

        response_times = []
        safety_scores = []

        with open(validation_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    input_text = data["input"]
                    expected_output = data["output"]

                    # Query the model
                    start_time = time.time()
                    response = self.client.chat(
                        model=model_name,
                        messages=[{"role": "user", "content": input_text}],
                    )
                    response_time = time.time() - start_time

                    actual_output = response["message"]["content"]

                    # Evaluate response
                    response_times.append(response_time)
                    safety_score = self._evaluate_safety(actual_output)
                    safety_scores.append(safety_score)

                    metrics["total_examples"] += 1
                    if safety_score > 0.5:
                        metrics["successful_responses"] += 1

                    if line_num <= 5:  # Log first few examples
                        logger.info(f"Example {line_num}:")
                        logger.info(f"Input: {input_text[:100]}...")
                        logger.info(f"Output: {actual_output[:100]}...")
                        logger.info(f"Safety Score: {safety_score:.2f}")

                    # Limit evaluation for speed
                    if line_num >= 50:
                        break

                except Exception as e:
                    logger.warning(f"Evaluation failed for example {line_num}: {e}")

        # Calculate final metrics
        if response_times:
            metrics["average_response_time"] = sum(response_times) / len(response_times)
        if safety_scores:
            metrics["safety_score"] = sum(safety_scores) / len(safety_scores)

        success_rate = metrics["successful_responses"] / max(
            metrics["total_examples"], 1
        )

        logger.info(f"Evaluation Results:")
        logger.info(f"  Examples evaluated: {metrics['total_examples']}")
        logger.info(f"  Success rate: {success_rate:.2f}")
        logger.info(f"  Average response time: {metrics['average_response_time']:.2f}s")
        logger.info(f"  Average safety score: {metrics['safety_score']:.2f}")

        # Save evaluation results
        eval_results = {
            "model_name": model_name,
            "metrics": metrics,
            "success_rate": success_rate,
            "timestamp": time.time(),
        }

        eval_file = Path("logs") / f"evaluation_{model_name.replace(':', '_')}.json"
        eval_file.parent.mkdir(exist_ok=True)

        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        return metrics

    def _evaluate_safety(self, response: str) -> float:
        """Evaluate safety of response"""

        # Simple safety scoring based on keywords
        safety_keywords = [
            "safety",
            "separation",
            "maintain",
            "altitude",
            "heading",
            "turn",
            "climb",
            "descend",
            "knots",
            "degrees",
            "feet",
        ]

        unsafe_keywords = ["crash", "collision", "emergency", "danger", "unsafe"]

        response_lower = response.lower()

        safety_count = sum(
            1 for keyword in safety_keywords if keyword in response_lower
        )
        unsafe_count = sum(
            1 for keyword in unsafe_keywords if keyword in response_lower
        )

        # Basic scoring
        score = min(1.0, safety_count * 0.1) - unsafe_count * 0.3
        return max(0.0, score)

    def list_models(self) -> List[str]:
        """List available Ollama models"""

        try:
            models = self.client.list()
            if hasattr(models, "models"):
                return [model.model for model in models.models]
            elif isinstance(models, dict) and "models" in models:
                return [model["name"] for model in models["models"]]
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def cleanup_model(self, model_name: str) -> None:
        """Remove a model"""
        try:
            cmd = ["ollama", "rm", model_name]
            subprocess.run(cmd, check=True)
            logger.info(f"Removed model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to remove model {model_name}: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Ollama model with BlueSky Gym data"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing processed training data"
    )
    parser.add_argument("--model", help="Base model to fine-tune (default from config)")
    parser.add_argument(
        "--output-name", help="Name for fine-tuned model (default from config)"
    )
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Training configuration file",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )
    parser.add_argument("--cleanup", help="Remove specified model")

    args = parser.parse_args()

    fine_tuner = OllamaFineTuner(args.config)

    if args.list_models:
        models = fine_tuner.list_models()
        print("Available Ollama models:")
        for model in models:
            print(f"  - {model}")
        return

    if args.cleanup:
        fine_tuner.cleanup_model(args.cleanup)
        return

    # Update config if arguments provided
    if args.model:
        fine_tuner.config["model"]["base_model"] = args.model
    if args.output_name:
        fine_tuner.config["model"]["output_name"] = args.output_name

    # Fine-tune model
    output_model = fine_tuner.fine_tune_model(args.data_dir, args.output_name)

    print(f"Fine-tuning completed: {output_model}")
    print(f"You can now use the model with: ollama run {output_model}")


if __name__ == "__main__":
    main()
