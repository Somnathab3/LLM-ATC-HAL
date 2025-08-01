"""
Integration Script for Fine-tuned BlueSky Gym Models
Integrates fine-tuned models back into the main LLM-ATC system
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add the main project to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_interface.ensemble import OllamaEnsembleClient, ModelConfig, ModelRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlueSkyGymModelIntegrator:
    """Integrate fine-tuned BlueSky Gym models into the main system"""

    def __init__(self):
        self.main_project_root = Path(__file__).parent.parent
        self.bsky_gym_root = Path(__file__).parent

    def register_fine_tuned_model(
        self, model_name: str, model_role: str = "primary"
    ) -> None:
        """Register a fine-tuned model in the ensemble system"""

        logger.info(f"Registering fine-tuned model: {model_name} as {model_role}")

        # Create enhanced ensemble client with fine-tuned model
        ensemble_client = self._create_enhanced_ensemble(model_name, model_role)

        # Test the integrated model
        self._test_integrated_model(ensemble_client, model_name)

        # Update the main ensemble configuration
        self._update_ensemble_config(model_name, model_role)

        logger.info(f"Successfully integrated model: {model_name}")

    def _create_enhanced_ensemble(
        self, model_name: str, model_role: str
    ) -> OllamaEnsembleClient:
        """Create enhanced ensemble with fine-tuned model"""

        ensemble = OllamaEnsembleClient()

        # Add the fine-tuned model to the ensemble
        role = ModelRole(model_role)

        fine_tuned_config = ModelConfig(
            name="fine_tuned_bsky",
            model_id=model_name,
            role=role,
            weight=0.4,  # High weight for fine-tuned model
            temperature=0.1,  # Low temperature for consistency
            max_tokens=500,
            timeout=15.0,
        )

        # Add to models dictionary
        ensemble.models["fine_tuned_bsky"] = fine_tuned_config

        # Adjust weights of other models
        total_other_weight = 0.6
        other_models = [
            m for m in ensemble.models.values() if m.name != "fine_tuned_bsky"
        ]

        for model in other_models:
            model.weight = (
                total_other_weight / len(other_models) if other_models else 0.0
            )

        logger.info(f"Enhanced ensemble created with {len(ensemble.models)} models")
        return ensemble

    def _test_integrated_model(
        self, ensemble: OllamaEnsembleClient, model_name: str
    ) -> None:
        """Test the integrated model with sample scenarios"""

        logger.info("Testing integrated model...")

        test_scenarios = [
            {
                "prompt": "Analyze this conflict resolution scenario: Aircraft AC001 and AC002 are converging at FL350. Current separation: 6 nautical miles. Time to conflict: 120 seconds. Recommend the best resolution maneuver.",
                "context": {
                    "aircraft_1": {
                        "id": "AC001",
                        "altitude": 35000,
                        "heading": 90,
                        "speed": 450,
                    },
                    "aircraft_2": {
                        "id": "AC002",
                        "altitude": 35000,
                        "heading": 270,
                        "speed": 420,
                    },
                    "separation": 6.0,
                    "conflict_time": 120,
                },
            },
            {
                "prompt": "Vertical conflict scenario: Aircraft A is at FL330 climbing to FL350. Aircraft B is at FL370 descending to FL330. Current vertical separation: 4000 feet. What action should be taken?",
                "context": {
                    "aircraft_a": {
                        "id": "A",
                        "altitude": 33000,
                        "target_altitude": 35000,
                        "climb_rate": 1500,
                    },
                    "aircraft_b": {
                        "id": "B",
                        "altitude": 37000,
                        "target_altitude": 33000,
                        "descent_rate": -1500,
                    },
                },
            },
        ]

        for i, scenario in enumerate(test_scenarios, 1):
            try:
                logger.info(f"Testing scenario {i}...")

                response = ensemble.query_ensemble(
                    prompt=scenario["prompt"],
                    context=scenario["context"],
                    require_json=True,
                    timeout=30.0,
                )

                logger.info(f"Scenario {i} - Consensus: {response.consensus_response}")
                logger.info(f"Scenario {i} - Confidence: {response.confidence:.3f}")
                logger.info(f"Scenario {i} - Safety flags: {response.safety_flags}")

                # Check if fine-tuned model participated
                if "fine_tuned_bsky" in response.individual_responses:
                    ft_response = response.individual_responses["fine_tuned_bsky"]
                    logger.info(
                        f"Scenario {i} - Fine-tuned model response: {ft_response}"
                    )

            except Exception as e:
                logger.error(f"Test scenario {i} failed: {e}")

    def _update_ensemble_config(self, model_name: str, model_role: str) -> None:
        """Update ensemble configuration files"""

        config_file = self.main_project_root / "llm_interface" / "ensemble_config.yaml"

        # Create configuration if it doesn't exist
        if not config_file.exists():
            config = {
                "fine_tuned_models": {},
                "ensemble_settings": {
                    "default_timeout": 30.0,
                    "max_models": 5,
                    "consensus_threshold": 0.6,
                },
            }
        else:
            with open(config_file, "r") as f:
                import yaml

                config = yaml.safe_load(f)

        # Add fine-tuned model configuration
        config["fine_tuned_models"][model_name] = {
            "role": model_role,
            "weight": 0.4,
            "temperature": 0.1,
            "max_tokens": 500,
            "timeout": 15.0,
            "training_data": "BlueSky Gym RL data",
            "environments": [
                "HorizontalCREnv-v0",
                "VerticalCREnv-v0",
                "SectorCREnv-v0",
            ],
            "algorithms": ["DDPG", "PPO", "SAC", "TD3"],
        }

        # Save updated configuration
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Updated ensemble configuration: {config_file}")

    def create_updated_ensemble_class(self, model_name: str) -> str:
        """Create updated ensemble class file with fine-tuned model"""

        updated_ensemble_file = (
            self.main_project_root / "llm_interface" / "enhanced_ensemble.py"
        )

        # Read the original ensemble file
        original_file = self.main_project_root / "llm_interface" / "ensemble.py"
        with open(original_file, "r") as f:
            original_content = f.read()

        # Create enhanced version with fine-tuned model integration
        enhanced_content = f'''"""
Enhanced LLM Ensemble System with BlueSky Gym Fine-tuned Models
Extends the original ensemble with specialized fine-tuned models
"""

{original_content}

class EnhancedOllamaEnsembleClient(OllamaEnsembleClient):
    """Enhanced ensemble client with BlueSky Gym fine-tuned models"""
    
    def __init__(self, fine_tuned_model: str = "{model_name}") -> None:
        super().__init__()
        self.fine_tuned_model = fine_tuned_model
        self._add_fine_tuned_model()
    
    def _add_fine_tuned_model(self) -> None:
        """Add fine-tuned BlueSky Gym model to ensemble"""
        
        if self.fine_tuned_model:
            # Check if model is available
            available_models = self._get_available_models()
            
            if self.fine_tuned_model in available_models:
                self.models["fine_tuned_bsky"] = ModelConfig(
                    name="fine_tuned_bsky",
                    model_id=self.fine_tuned_model,
                    role=ModelRole.PRIMARY,
                    weight=0.5,  # High weight for specialized model
                    temperature=0.1,
                    max_tokens=500,
                    timeout=15.0
                )
                
                # Adjust other model weights
                other_weight = 0.5 / (len(self.models) - 1) if len(self.models) > 1 else 0.0
                for model_name, model in self.models.items():
                    if model_name != "fine_tuned_bsky":
                        model.weight = other_weight
                
                logging.info(f"Added fine-tuned model to ensemble: {{self.fine_tuned_model}}")
            else:
                logging.warning(f"Fine-tuned model not available: {{self.fine_tuned_model}}")
    
    def query_ensemble_with_specialization(
        self, prompt: str, context: dict, scenario_type: str = "general"
    ) -> EnsembleResponse:
        """Query ensemble with scenario-type specific weighting"""
        
        # Adjust model weights based on scenario type
        if scenario_type in ["horizontal_conflict", "vertical_conflict", "sector_management"]:
            # Increase weight of fine-tuned model for scenarios it was trained on
            if "fine_tuned_bsky" in self.models:
                self.models["fine_tuned_bsky"].weight = 0.6
                other_weight = 0.4 / (len(self.models) - 1) if len(self.models) > 1 else 0.0
                for model_name, model in self.models.items():
                    if model_name != "fine_tuned_bsky":
                        model.weight = other_weight
        
        # Use standard ensemble query
        response = self.query_ensemble(prompt, context, require_json=True)
        
        # Add specialization metadata
        response.specialized_for = scenario_type
        
        return response


# Convenience function for easy integration
def create_enhanced_ensemble(fine_tuned_model: str = "{model_name}") -> EnhancedOllamaEnsembleClient:
    """Create enhanced ensemble with BlueSky Gym fine-tuned model"""
    return EnhancedOllamaEnsembleClient(fine_tuned_model)
'''

        # Write enhanced ensemble file
        with open(updated_ensemble_file, "w") as f:
            f.write(enhanced_content)

        logger.info(f"Created enhanced ensemble class: {updated_ensemble_file}")
        return str(updated_ensemble_file)

    def generate_integration_report(self, model_name: str) -> str:
        """Generate integration report"""

        report_file = (
            self.bsky_gym_root
            / "logs"
            / f"integration_report_{model_name.replace(':', '_')}.md"
        )
        report_file.parent.mkdir(exist_ok=True)

        # Load evaluation results if available
        eval_files = list(
            self.bsky_gym_root.glob(
                f"logs/evaluation_*{model_name.replace(':', '_')}*.json"
            )
        )

        evaluation_data = None
        if eval_files:
            with open(eval_files[0], "r") as f:
                evaluation_data = json.load(f)

        # Generate report
        with open(report_file, "w") as f:
            f.write(f"# BlueSky Gym Model Integration Report\\n")
            f.write(f"**Model:** {model_name}\\n\\n")
            f.write(f"## Integration Summary\\n\\n")
            f.write(f"- **Fine-tuned Model:** {model_name}\\n")
            f.write(f"- **Training Data:** BlueSky Gym RL scenarios\\n")
            f.write(
                f"- **Environments:** Horizontal/Vertical/Sector conflict resolution\\n"
            )
            f.write(f"- **Algorithms:** DDPG, PPO, SAC, TD3\\n")
            f.write(f"- **Integration Date:** {Path(__file__).stat().st_mtime}\\n\\n")

            if evaluation_data:
                metrics = evaluation_data.get("metrics", {})
                f.write(f"## Evaluation Results\\n\\n")

                if "safety_score" in metrics:
                    safety = metrics["safety_score"]
                    f.write(f"### Safety Score\\n")
                    f.write(f"- **Mean:** {safety.get('mean', 0.0):.3f}\\n")
                    f.write(
                        f"- **Standard Deviation:** {safety.get('std', 0.0):.3f}\\n"
                    )
                    f.write(
                        f"- **Range:** {safety.get('min', 0.0):.3f} - {safety.get('max', 0.0):.3f}\\n\\n"
                    )

                if "relevance_score" in metrics:
                    relevance = metrics["relevance_score"]
                    f.write(f"### Relevance Score\\n")
                    f.write(f"- **Mean:** {relevance.get('mean', 0.0):.3f}\\n")
                    f.write(
                        f"- **Standard Deviation:** {relevance.get('std', 0.0):.3f}\\n"
                    )
                    f.write(
                        f"- **Range:** {relevance.get('min', 0.0):.3f} - {relevance.get('max', 0.0):.3f}\\n\\n"
                    )

            f.write(f"## Usage Instructions\\n\\n")
            f.write(f"### Using in Main System\\n\\n")
            f.write(f"```python\\n")
            f.write(
                f"from llm_interface.enhanced_ensemble import create_enhanced_ensemble\\n"
            )
            f.write(f"\\n")
            f.write(f"# Create ensemble with fine-tuned model\\n")
            f.write(f"ensemble = create_enhanced_ensemble('{model_name}')\\n")
            f.write(f"\\n")
            f.write(f"# Query for conflict resolution\\n")
            f.write(f"response = ensemble.query_ensemble_with_specialization(\\n")
            f.write(f"    prompt='Analyze conflict scenario...',\\n")
            f.write(f"    context={{...}},\\n")
            f.write(f"    scenario_type='horizontal_conflict'\\n")
            f.write(f")\\n")
            f.write(f"```\\n\\n")

            f.write(f"### Testing the Integration\\n\\n")
            f.write(f"```bash\\n")
            f.write(f"cd {self.main_project_root}\\n")
            f.write(
                f"python -c \\\"from llm_interface.enhanced_ensemble import create_enhanced_ensemble; e = create_enhanced_ensemble('{model_name}'); print('Integration successful!')\\\"\\n"
            )
            f.write(f"```\\n\\n")

        logger.info(f"Generated integration report: {report_file}")
        return str(report_file)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate fine-tuned BlueSky Gym models"
    )
    parser.add_argument("--model", required=True, help="Fine-tuned model name")
    parser.add_argument(
        "--role",
        default="primary",
        choices=["primary", "validator", "technical", "safety"],
        help="Model role in ensemble",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test integration, don't modify files",
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate integration report"
    )

    args = parser.parse_args()

    integrator = BlueSkyGymModelIntegrator()

    try:
        if args.test_only:
            # Only test the model
            ensemble = integrator._create_enhanced_ensemble(args.model, args.role)
            integrator._test_integrated_model(ensemble, args.model)
        else:
            # Full integration
            integrator.register_fine_tuned_model(args.model, args.role)
            integrator.create_updated_ensemble_class(args.model)

        if args.generate_report:
            report_file = integrator.generate_integration_report(args.model)
            print(f"Integration report generated: {report_file}")

        print(f"Integration completed for model: {args.model}")

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
