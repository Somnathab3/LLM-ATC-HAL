"""
Automated Setup and Fine-tuning Script for BlueSky Gym LLM
Handles the complete pipeline from data processing to model evaluation
"""

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
import yaml
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlueSkyLLMSetup:
    """Automated setup for BlueSky Gym LLM fine-tuning"""

    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.config_path = self.root_dir / "config" / "training_config.yaml"

    def setup_environment(self) -> None:
        """Setup Python environment and dependencies"""
        logger.info("Setting up environment...")

        # Install requirements
        requirements_file = self.root_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ],
                    check=True,
                )
                logger.info("Dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                raise
        else:
            logger.warning("requirements.txt not found")

    def copy_gym_data(self, source_path: str) -> str:
        """Copy gym distill data to the data directory"""
        logger.info(f"Copying gym data from {source_path}")

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source data file not found: {source_path}")

        data_dir = self.root_dir / "data"
        data_dir.mkdir(exist_ok=True)

        target = data_dir / "gym_distill.jsonl"
        shutil.copy2(source, target)

        logger.info(f"Data copied to {target}")
        return str(target)

    def process_data(self, gym_data_path: str = None) -> str:
        """Process gym data for training"""
        logger.info("Processing gym data...")

        if gym_data_path is None:
            gym_data_path = self.root_dir / "data" / "gym_distill.jsonl"

        output_dir = self.root_dir / "data" / "processed"

        # Run data processor
        processor_script = self.root_dir / "scripts" / "data_processor.py"

        cmd = [
            sys.executable,
            str(processor_script),
            "--input",
            str(gym_data_path),
            "--output",
            str(output_dir),
            "--config",
            str(self.config_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Data processing completed successfully")
            logger.info(f"Output: {result.stdout}")
            return str(output_dir)
        except subprocess.CalledProcessError as e:
            logger.error(f"Data processing failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

    def fine_tune_model(self, data_dir: str, model_name: str = None) -> str:
        """Fine-tune the model"""
        logger.info("Starting model fine-tuning...")

        fine_tune_script = self.root_dir / "scripts" / "fine_tune_ollama.py"

        cmd = [
            sys.executable,
            str(fine_tune_script),
            "--data-dir",
            data_dir,
            "--config",
            str(self.config_path),
        ]

        if model_name:
            cmd.extend(["--output-name", model_name])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Model fine-tuning completed successfully")
            logger.info(f"Output: {result.stdout}")

            # Extract model name from output
            lines = result.stdout.split("\n")
            for line in lines:
                if "Fine-tuning completed:" in line:
                    model_name = line.split(": ")[-1].strip()
                    return model_name

            # Fallback to config
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config["model"]["output_name"]

        except subprocess.CalledProcessError as e:
            logger.error(f"Model fine-tuning failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

    def evaluate_model(self, model_name: str, data_dir: str) -> None:
        """Evaluate the fine-tuned model"""
        logger.info(f"Evaluating model: {model_name}")

        evaluation_script = self.root_dir / "scripts" / "evaluation.py"
        test_data = Path(data_dir) / "test.jsonl"

        if not test_data.exists():
            logger.warning(f"Test data not found: {test_data}")
            return

        cmd = [
            sys.executable,
            str(evaluation_script),
            "--model",
            model_name,
            "--test-data",
            str(test_data),
            "--output-dir",
            str(self.root_dir / "logs"),
            "--config",
            str(self.config_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Model evaluation completed successfully")
            logger.info(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Model evaluation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            # Don't raise - evaluation failure shouldn't stop the pipeline

    def check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def list_available_models(self) -> None:
        """List available Ollama models"""
        if not self.check_ollama_available():
            logger.error("Ollama is not available. Please install Ollama first.")
            return

        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            logger.info("Available Ollama models:")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list models: {e}")

    def run_full_pipeline(self, gym_data_path: str, model_name: str = None) -> str:
        """Run the complete fine-tuning pipeline"""
        logger.info("Starting full fine-tuning pipeline...")

        # Check prerequisites
        if not self.check_ollama_available():
            raise RuntimeError("Ollama is not available. Please install Ollama first.")

        # Setup environment
        self.setup_environment()

        # Copy and process data
        if not Path(gym_data_path).exists():
            raise FileNotFoundError(f"Gym data file not found: {gym_data_path}")

        data_file = self.copy_gym_data(gym_data_path)
        processed_dir = self.process_data(data_file)

        # Fine-tune model
        fine_tuned_model = self.fine_tune_model(processed_dir, model_name)

        # Evaluate model
        self.evaluate_model(fine_tuned_model, processed_dir)

        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Fine-tuned model: {fine_tuned_model}")
        logger.info(f"You can now use the model with: ollama run {fine_tuned_model}")

        return fine_tuned_model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BlueSky Gym LLM Fine-tuning Setup")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run full fine-tuning pipeline"
    )
    pipeline_parser.add_argument(
        "--gym-data", required=True, help="Path to gym_distill.jsonl file"
    )
    pipeline_parser.add_argument("--model-name", help="Name for fine-tuned model")

    # Individual commands
    setup_parser = subparsers.add_parser("setup", help="Setup environment only")

    process_parser = subparsers.add_parser("process", help="Process data only")
    process_parser.add_argument(
        "--gym-data", required=True, help="Path to gym_distill.jsonl file"
    )

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune model only")
    finetune_parser.add_argument(
        "--data-dir", required=True, help="Processed data directory"
    )
    finetune_parser.add_argument("--model-name", help="Name for fine-tuned model")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model only")
    eval_parser.add_argument("--model", required=True, help="Model name to evaluate")
    eval_parser.add_argument(
        "--data-dir", required=True, help="Data directory with test data"
    )

    list_parser = subparsers.add_parser(
        "list-models", help="List available Ollama models"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup = BlueSkyLLMSetup()

    try:
        if args.command == "pipeline":
            setup.run_full_pipeline(args.gym_data, args.model_name)

        elif args.command == "setup":
            setup.setup_environment()

        elif args.command == "process":
            processed_dir = setup.process_data(args.gym_data)
            print(f"Data processed and saved to: {processed_dir}")

        elif args.command == "finetune":
            model_name = setup.fine_tune_model(args.data_dir, args.model_name)
            print(f"Model fine-tuned: {model_name}")

        elif args.command == "evaluate":
            setup.evaluate_model(args.model, args.data_dir)

        elif args.command == "list-models":
            setup.list_available_models()

    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
