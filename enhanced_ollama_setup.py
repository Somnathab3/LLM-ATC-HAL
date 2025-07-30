#!/usr/bin/env python3
"""
Enhanced Ollama Configuration for LLM-ATC-HAL
==============================================
Additional utilities and configurations for optimizing Ollama integration.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import ollama


@dataclass
class OllamaModelConfig:
    """Configuration for Ollama model optimization"""

    name: str
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    context_length: int
    gpu_layers: int = -1  # -1 = use all available GPU layers


class EnhancedOllamaManager:
    """Enhanced Ollama integration manager for ATC operations"""

    def __init__(self) -> None:
        self.client = ollama.Client()
        self.logger = logging.getLogger(__name__)

        # Optimized model configurations for different ATC tasks
        self.model_configs = {
            "conflict_resolution": OllamaModelConfig(
                name="llama3.1:8b",
                temperature=0.1,  # Low for consistent, safe decisions
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.1,
                context_length=4096,
                gpu_layers=-1,
            ),
            "conflict_detection": OllamaModelConfig(
                name="llama3.1:8b",
                temperature=0.05,  # Very low for reliable detection
                top_p=0.7,
                top_k=15,
                repeat_penalty=1.0,
                context_length=2048,
                gpu_layers=-1,
            ),
            "safety_assessment": OllamaModelConfig(
                name="mistral:7b",  # Alternative model for cross-validation
                temperature=0.2,
                top_p=0.9,
                top_k=25,
                repeat_penalty=1.1,
                context_length=2048,
                gpu_layers=-1,
            ),
        }

    def check_ollama_status(self) -> dict[str, any]:
        """Check Ollama service status and available models"""
        try:
            models = self.client.list()
            available_models = [model["name"] for model in models["models"]]

            return {
                "status": "running",
                "available_models": available_models,
                "model_count": len(available_models),
                "recommended_models": {
                    "primary": "llama3.1:8b" if "llama3.1:8b" in available_models else None,
                    "secondary": "mistral:7b" if "mistral:7b" in available_models else None,
                    "technical": "codellama:7b" if "codellama:7b" in available_models else None,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "available_models": [],
                "model_count": 0,
            }

    def optimize_model_for_task(self, task_type: str) -> Optional[OllamaModelConfig]:
        """Get optimized model configuration for specific ATC task"""
        return self.model_configs.get(task_type)

    def pull_recommended_models(self) -> dict[str, bool]:
        """Pull recommended models for ATC operations"""
        recommended_models = [
            "llama3.1:8b",  # Primary model for general ATC tasks
            "mistral:7b",  # Secondary model for validation
            "codellama:7b",  # Technical analysis model
        ]

        results = {}
        for model in recommended_models:
            try:
                self.logger.info(f"Pulling model: {model}")
                self.client.pull(model)
                results[model] = True
                self.logger.info(f"Successfully pulled: {model}")
            except Exception as e:
                self.logger.exception(f"Failed to pull {model}: {e}")
                results[model] = False

        return results

    def create_atc_modelfile(self, base_model: str = "llama3.1:8b") -> str:
        """Create optimized Modelfile for ATC operations"""
        return f"""
FROM {base_model}

# ATC-specific system prompt
SYSTEM \"\"\"
You are an expert Air Traffic Controller with extensive experience in conflict detection and resolution.

Core Responsibilities:
- Maintain aircraft separation (minimum 5 NM horizontal OR 1000 ft vertical)
- Ensure flight safety through precise, timely decisions
- Follow ICAO standards and procedures
- Minimize disruption to flight paths
- Provide clear, actionable commands

Response Format:
- Always provide specific BlueSky commands (e.g., "HDG AC001 270")
- Include rationale for each decision
- Assess confidence levels (0.0-1.0)
- Consider environmental factors

Safety is paramount. When in doubt, prioritize separation over efficiency.
\"\"\"

# Optimized parameters for ATC tasks
PARAMETER temperature 0.1
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Stop sequences
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
"""

    def create_custom_atc_model(self, model_name: str = "atc-controller:latest") -> bool:
        """Create custom ATC-optimized model"""
        try:
            modelfile = self.create_atc_modelfile()

            # Create the custom model
            self.client.create(model_name, modelfile)
            self.logger.info(f"Created custom ATC model: {model_name}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to create custom model: {e}")
            return False

    def benchmark_models(self) -> dict[str, dict[str, float]]:
        """Benchmark available models for ATC tasks"""
        test_prompt = """
        Aircraft conflict detected:
        - AC001: Position 52.3676°N, 4.9041°E, Alt 35000 ft, Heading 090°, Speed 450 kts
        - AC002: Position 52.3700°N, 4.9100°E, Alt 35000 ft, Heading 270°, Speed 460 kts
        - Time to conflict: 90 seconds

        Provide a single BlueSky command to resolve this conflict safely.
        """

        status = self.check_ollama_status()
        available_models = status.get("available_models", [])

        results = {}

        for model in available_models:
            try:
                import time

                start_time = time.time()

                response = self.client.chat(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}],
                )

                end_time = time.time()
                response_time = end_time - start_time

                content = response["message"]["content"]

                results[model] = {
                    "response_time": response_time,
                    "response_length": len(content),
                    "contains_command": any(
                        cmd in content.upper() for cmd in ["HDG", "ALT", "SPD"]
                    ),
                    "status": "success",
                }

            except Exception as e:
                results[model] = {
                    "response_time": float("inf"),
                    "error": str(e),
                    "status": "failed",
                }

        return results

    def get_model_info(self, model_name: str) -> Optional[dict]:
        """Get detailed information about a specific model"""
        try:
            return self.client.show(model_name)
        except Exception as e:
            self.logger.exception(f"Failed to get info for {model_name}: {e}")
            return None

    def health_check(self) -> dict[str, any]:
        """Comprehensive health check for Ollama integration"""
        health_status = {
            "ollama_service": False,
            "models_available": False,
            "primary_model_ready": False,
            "performance_acceptable": False,
            "recommendations": [],
        }

        # Check Ollama service
        status = self.check_ollama_status()
        if status["status"] == "running":
            health_status["ollama_service"] = True

            # Check models
            if status["model_count"] > 0:
                health_status["models_available"] = True

                # Check primary model
                if "llama3.1:8b" in status["available_models"]:
                    health_status["primary_model_ready"] = True
                else:
                    health_status["recommendations"].append(
                        "Install primary model: ollama pull llama3.1:8b",
                    )

                # Quick performance test
                try:
                    benchmark = self.benchmark_models()
                    avg_response_time = sum(
                        r["response_time"]
                        for r in benchmark.values()
                        if r.get("status") == "success"
                    ) / len([r for r in benchmark.values() if r.get("status") == "success"])

                    if avg_response_time < 10.0:  # Less than 10 seconds
                        health_status["performance_acceptable"] = True
                    else:
                        health_status["recommendations"].append(
                            "Consider GPU acceleration for better performance",
                        )

                except Exception:
                    health_status["recommendations"].append(
                        "Unable to benchmark performance - check model status",
                    )
            else:
                health_status["recommendations"].append(
                    "No models installed. Run: ollama pull llama3.1:8b",
                )
        else:
            health_status["recommendations"].append(
                "Ollama service not running. Start with: ollama serve",
            )

        return health_status


def setup_atc_environment():
    """Setup and optimize Ollama environment for ATC operations"""
    manager = EnhancedOllamaManager()

    # Health check
    health = manager.health_check()
    for check, _status in health.items():
        if check != "recommendations":
            pass

    # Recommendations
    if health["recommendations"]:
        for _rec in health["recommendations"]:
            pass

    # Model benchmark
    if health["models_available"]:
        benchmarks = manager.benchmark_models()

        for _model, metrics in benchmarks.items():
            if metrics["status"] == "success":
                pass

    # Create custom model
    success = manager.create_custom_atc_model()
    if success:
        pass
    else:
        pass

    return manager


if __name__ == "__main__":
    # Setup ATC environment
    manager = setup_atc_environment()
