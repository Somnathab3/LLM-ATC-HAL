"""
Data Processor for BlueSky Gym LLM Fine-tuning
Converts RL training data to format suitable for LLM fine-tuning
"""

import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example for LLM fine-tuning"""

    input_text: str
    output_text: str
    metadata: Dict[str, Any]


class BlueSkyGymDataProcessor:
    """Process BlueSky Gym RL data for LLM fine-tuning"""

    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize data processor"""
        self.config = self._load_config(config_path)
        self.environments = self.config["data"]["include_environments"]
        self.algorithms = self.config["data"]["include_algorithms"]

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def process_jsonl_data(self, input_file: str, output_dir: str) -> None:
        """Process JSONL data file"""
        logger.info(f"Processing JSONL data from {input_file}")

        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Read and parse JSONL data
        examples = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    processed_examples = self._convert_gym_distill_format(data)
                    examples.extend(processed_examples)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")

        logger.info(f"Processed {len(examples)} training examples")

        # Split data
        train_examples, val_examples, test_examples = self._split_data(examples)

        # Save processed data
        self._save_examples(train_examples, output_path / "train.jsonl")
        self._save_examples(val_examples, output_path / "validation.jsonl")
        self._save_examples(test_examples, output_path / "test.jsonl")

        # Generate statistics
        self._generate_statistics(examples, output_path / "statistics.json")

    def _convert_gym_distill_format(
        self, rl_data: Dict[str, Any]
    ) -> List[TrainingExample]:
        """Convert gym distill format to LLM training format"""
        examples = []

        # Extract data from gym distill format
        instruction = rl_data.get("instruction", "")
        input_text = rl_data.get("input", "")
        output_text = rl_data.get("output", "")
        metadata = rl_data.get("metadata", {})

        # Extract environment and algorithm info
        environment = metadata.get("environment", "unknown")
        algorithm = metadata.get("algorithm", "unknown")

        # Filter by configured environments and algorithms
        if environment not in self.environments or algorithm not in self.algorithms:
            return examples

        # Create enhanced input prompt
        enhanced_input = f"{instruction}\n\n{input_text}"

        # Create metadata for tracking
        training_metadata = {
            "environment": environment,
            "algorithm": algorithm,
            "reward": metadata.get("reward", 0.0),
            "episode_id": metadata.get("episode_info", {}).get("episode_id", 0),
            "conflict_detected": metadata.get("episode_info", {}).get(
                "conflict_detected", "False"
            ),
            "safety_violation": metadata.get("episode_info", {}).get(
                "safety_violation", False
            ),
            "model_confidence": metadata.get("episode_info", {}).get(
                "model_confidence", 0.5
            ),
        }

        # Create training example
        example = TrainingExample(
            input_text=enhanced_input,
            output_text=output_text,
            metadata=training_metadata,
        )

        examples.append(example)
        return examples

    def _convert_rl_to_llm_format(
        self, rl_data: Dict[str, Any]
    ) -> List[TrainingExample]:
        """Convert RL training data to LLM format"""
        examples = []

        # Extract relevant information
        environment = rl_data.get("environment", "unknown")
        algorithm = rl_data.get("algorithm", "unknown")

        # Filter by configured environments and algorithms
        if environment not in self.environments or algorithm not in self.algorithms:
            return examples

        # Convert state-action pairs to natural language
        if "episodes" in rl_data:
            for episode in rl_data["episodes"]:
                examples.extend(self._process_episode(episode, environment, algorithm))
        elif "states" in rl_data and "actions" in rl_data:
            examples.extend(
                self._process_state_action_pairs(rl_data, environment, algorithm)
            )

        return examples

    def _process_episode(
        self, episode: Dict[str, Any], env: str, algo: str
    ) -> List[TrainingExample]:
        """Process a single episode"""
        examples = []

        states = episode.get("states", [])
        actions = episode.get("actions", [])
        rewards = episode.get("rewards", [])

        for i, (state, action) in enumerate(zip(states, actions)):
            try:
                # Create natural language description of state
                state_description = self._state_to_text(state, env)

                # Create natural language description of action
                action_description = self._action_to_text(action, env)

                # Create training example
                input_text = self._create_input_prompt(state_description, env)
                output_text = self._create_output_response(
                    action_description, rewards[i] if i < len(rewards) else 0.0
                )

                metadata = {
                    "environment": env,
                    "algorithm": algo,
                    "episode_step": i,
                    "reward": rewards[i] if i < len(rewards) else 0.0,
                }

                examples.append(TrainingExample(input_text, output_text, metadata))

            except Exception as e:
                logger.warning(f"Failed to process episode step {i}: {e}")

        return examples

    def _process_state_action_pairs(
        self, data: Dict[str, Any], env: str, algo: str
    ) -> List[TrainingExample]:
        """Process state-action pairs directly"""
        examples = []

        states = data.get("states", [])
        actions = data.get("actions", [])
        rewards = data.get("rewards", [])

        for i, (state, action) in enumerate(zip(states, actions)):
            try:
                state_description = self._state_to_text(state, env)
                action_description = self._action_to_text(action, env)

                input_text = self._create_input_prompt(state_description, env)
                output_text = self._create_output_response(
                    action_description, rewards[i] if i < len(rewards) else 0.0
                )

                metadata = {
                    "environment": env,
                    "algorithm": algo,
                    "step": i,
                    "reward": rewards[i] if i < len(rewards) else 0.0,
                }

                examples.append(TrainingExample(input_text, output_text, metadata))

            except Exception as e:
                logger.warning(f"Failed to process step {i}: {e}")

        return examples

    def _state_to_text(self, state: Any, environment: str) -> str:
        """Convert state representation to natural language"""

        if environment == "HorizontalCREnv-v0":
            return self._horizontal_state_to_text(state)
        elif environment == "VerticalCREnv-v0":
            return self._vertical_state_to_text(state)
        elif environment == "SectorCREnv-v0":
            return self._sector_state_to_text(state)
        elif environment == "MergeEnv-v0":
            return self._merge_state_to_text(state)
        else:
            return self._generic_state_to_text(state)

    def _horizontal_state_to_text(self, state: Any) -> str:
        """Convert horizontal conflict state to text"""
        if isinstance(state, (list, np.ndarray)):
            # Assume state format: [own_x, own_y, own_heading, own_speed, other_x, other_y, other_heading, other_speed, ...]
            if len(state) >= 8:
                return (
                    f"Aircraft conflict scenario: Own aircraft at position ({state[0]:.1f}, {state[1]:.1f}) "
                    f"heading {state[2]:.1f}° at {state[3]:.1f} knots. "
                    f"Conflicting aircraft at position ({state[4]:.1f}, {state[5]:.1f}) "
                    f"heading {state[6]:.1f}° at {state[7]:.1f} knots."
                )

        return f"Horizontal conflict scenario with aircraft state: {state}"

    def _vertical_state_to_text(self, state: Any) -> str:
        """Convert vertical conflict state to text"""
        if isinstance(state, (list, np.ndarray)):
            if len(state) >= 6:
                return (
                    f"Vertical conflict scenario: Aircraft 1 at altitude {state[0]:.0f} ft "
                    f"climbing/descending at {state[1]:.1f} ft/min. "
                    f"Aircraft 2 at altitude {state[2]:.0f} ft "
                    f"climbing/descending at {state[3]:.1f} ft/min. "
                    f"Horizontal separation: {state[4]:.1f} nm."
                )

        return f"Vertical conflict scenario with aircraft state: {state}"

    def _sector_state_to_text(self, state: Any) -> str:
        """Convert sector conflict state to text"""
        return f"Sector conflict scenario: Multiple aircraft coordination required. State: {state}"

    def _merge_state_to_text(self, state: Any) -> str:
        """Convert merge scenario state to text"""
        return f"Aircraft merge scenario: Coordination for runway approach or airway merge. State: {state}"

    def _generic_state_to_text(self, state: Any) -> str:
        """Generic state to text conversion"""
        return f"Air traffic control scenario with state: {state}"

    def _action_to_text(self, action: Any, environment: str) -> str:
        """Convert action to natural language"""

        if environment == "HorizontalCREnv-v0":
            return self._horizontal_action_to_text(action)
        elif environment == "VerticalCREnv-v0":
            return self._vertical_action_to_text(action)
        elif environment == "SectorCREnv-v0":
            return self._sector_action_to_text(action)
        elif environment == "MergeEnv-v0":
            return self._merge_action_to_text(action)
        else:
            return self._generic_action_to_text(action)

    def _horizontal_action_to_text(self, action: Any) -> str:
        """Convert horizontal conflict action to text"""
        if isinstance(action, (list, np.ndarray)):
            if len(action) >= 2:
                heading_change = action[0]
                speed_change = action[1] if len(action) > 1 else 0

                result = []
                if abs(heading_change) > 0.1:
                    direction = "right" if heading_change > 0 else "left"
                    result.append(f"turn {direction} {abs(heading_change):.1f} degrees")

                if abs(speed_change) > 0.1:
                    change_type = "increase" if speed_change > 0 else "decrease"
                    result.append(
                        f"{change_type} speed by {abs(speed_change):.1f} knots"
                    )

                if not result:
                    return "maintain current heading and speed"

                return "Recommended action: " + " and ".join(result)

        return f"Execute maneuver: {action}"

    def _vertical_action_to_text(self, action: Any) -> str:
        """Convert vertical conflict action to text"""
        if isinstance(action, (list, np.ndarray)) and len(action) >= 1:
            altitude_change = action[0]

            if abs(altitude_change) < 10:
                return "Recommended action: maintain current altitude"
            elif altitude_change > 0:
                return f"Recommended action: climb {altitude_change:.0f} feet"
            else:
                return f"Recommended action: descend {abs(altitude_change):.0f} feet"

        return f"Execute altitude maneuver: {action}"

    def _sector_action_to_text(self, action: Any) -> str:
        """Convert sector action to text"""
        return f"Sector coordination action: {action}"

    def _merge_action_to_text(self, action: Any) -> str:
        """Convert merge action to text"""
        return f"Merge coordination action: {action}"

    def _generic_action_to_text(self, action: Any) -> str:
        """Generic action to text conversion"""
        return f"Recommended action: {action}"

    def _create_input_prompt(self, state_description: str, environment: str) -> str:
        """Create input prompt for LLM training"""
        env_context = {
            "HorizontalCREnv-v0": "horizontal conflict resolution",
            "VerticalCREnv-v0": "vertical conflict resolution",
            "SectorCREnv-v0": "sector-based traffic management",
            "MergeEnv-v0": "aircraft merge coordination",
        }.get(environment, "air traffic control")

        return (
            f"You are an expert air traffic controller specializing in {env_context}. "
            f"Analyze the following situation and recommend the appropriate action:\n\n"
            f"{state_description}\n\n"
            f"What action should be taken to ensure safe and efficient traffic flow?"
        )

    def _create_output_response(self, action_description: str, reward: float) -> str:
        """Create output response for LLM training"""
        safety_assessment = (
            "This action maintains safety standards"
            if reward >= 0
            else "This action requires careful monitoring"
        )

        return (
            f"{action_description}\n\n"
            f"Safety assessment: {safety_assessment}. "
            f"This maneuver is based on optimal air traffic control practices and "
            f"ensures compliance with separation requirements."
        )

    def _split_data(
        self, examples: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """Split data into train/validation/test sets"""
        np.random.shuffle(examples)

        train_split = self.config["data"]["train_split"]
        val_split = self.config["data"]["validation_split"]

        train_size = int(len(examples) * train_split)
        val_size = int(len(examples) * val_split)

        train_examples = examples[:train_size]
        val_examples = examples[train_size : train_size + val_size]
        test_examples = examples[train_size + val_size :]

        logger.info(
            f"Data split: {len(train_examples)} train, {len(val_examples)} validation, {len(test_examples)} test"
        )

        return train_examples, val_examples, test_examples

    def _save_examples(
        self, examples: List[TrainingExample], output_file: Path
    ) -> None:
        """Save examples to JSONL format"""
        with open(output_file, "w") as f:
            for example in examples:
                data = {
                    "input": example.input_text,
                    "output": example.output_text,
                    "metadata": example.metadata,
                }
                f.write(json.dumps(data) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_file}")

    def _generate_statistics(
        self, examples: List[TrainingExample], output_file: Path
    ) -> None:
        """Generate and save dataset statistics"""
        stats = {
            "total_examples": len(examples),
            "environments": {},
            "algorithms": {},
            "average_input_length": 0,
            "average_output_length": 0,
        }

        input_lengths = []
        output_lengths = []

        for example in examples:
            env = example.metadata.get("environment", "unknown")
            algo = example.metadata.get("algorithm", "unknown")

            stats["environments"][env] = stats["environments"].get(env, 0) + 1
            stats["algorithms"][algo] = stats["algorithms"].get(algo, 0) + 1

            input_lengths.append(len(example.input_text))
            output_lengths.append(len(example.output_text))

        stats["average_input_length"] = np.mean(input_lengths)
        stats["average_output_length"] = np.mean(output_lengths)

        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Generated statistics: {stats}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process BlueSky Gym data for LLM fine-tuning"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument(
        "--output", required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--config",
        default="config/training_config.yaml",
        help="Training configuration file",
    )

    args = parser.parse_args()

    processor = BlueSkyGymDataProcessor(args.config)
    processor.process_jsonl_data(args.input, args.output)


if __name__ == "__main__":
    main()
