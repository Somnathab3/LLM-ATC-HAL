# memory/experience_integrator.py
"""
Experience Replay Integration for LLM-ATC-HAL System
Connects vector memory store with real-time conflict resolution
"""

import logging
import time
from typing import Any, Optional

from analysis.enhanced_hallucination_detection import EnhancedHallucinationDetector
from llm_atc.memory.replay_store import ConflictExperience, SimilarityResult, VectorReplayStore
from llm_atc.metrics.safety_margin_quantifier import SafetyMarginQuantifier


class ExperienceIntegrator:
    """Integrates experience replay with real-time conflict resolution"""

    def __init__(self, replay_store: VectorReplayStore) -> None:
        self.replay_store = replay_store
        self.hallucination_detector = EnhancedHallucinationDetector()
        self.safety_quantifier = SafetyMarginQuantifier()

        # Experience collection settings
        self.auto_store_enabled = True
        self.similarity_threshold = 0.7
        self.max_similar_experiences = 5

        logging.info("Experience integrator initialized")

    def process_conflict_resolution(
        self,
        scenario_context: dict[str, Any],
        conflict_geometry: dict[str, float],
        environmental_conditions: dict[str, Any],
        llm_decision: dict[str, Any],
        baseline_decision: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Process a conflict resolution with experience replay integration

        Returns:
            Tuple of (enhanced_decision, lessons_learned)
        """

        try:
            # Step 1: Find similar past experiences
            similar_experiences = self._find_relevant_experiences(
                scenario_context,
                conflict_geometry,
                environmental_conditions,
            )

            # Step 2: Extract lessons and patterns
            lessons_learned = self._extract_lessons(similar_experiences)
            pattern_warnings = self._check_hallucination_patterns(
                scenario_context,
                environmental_conditions,
                similar_experiences,
            )

            # Step 3: Enhance current decision with historical insights
            enhanced_decision = self._enhance_decision_with_experience(
                llm_decision,
                baseline_decision,
                similar_experiences,
            )

            # Step 4: Prepare comprehensive guidance
            guidance = lessons_learned + pattern_warnings

            logging.info("Processed conflict with %d similar experiences", len(similar_experiences))
            return enhanced_decision, guidance

        except Exception:
            logging.exception("Failed to process conflict resolution")
            return llm_decision, []

    def _find_relevant_experiences(
        self,
        scenario_context: dict[str, Any],
        conflict_geometry: dict[str, float],
        environmental_conditions: dict[str, Any],
    ) -> list[SimilarityResult]:
        """Find experiences relevant to current scenario"""

        try:
            # Create query experience
            query_experience = ConflictExperience(
                experience_id="",
                timestamp=time.time(),
                scenario_context=scenario_context,
                conflict_geometry=conflict_geometry,
                environmental_conditions=environmental_conditions,
                llm_decision={},
                baseline_decision={},
                actual_outcome={},
                safety_metrics={},
                hallucination_detected=False,
                hallucination_types=[],
                controller_override=None,
                lessons_learned="",
            )

            # Search for similar experiences
            return self.replay_store.find_similar_experiences(
                query_experience,
                top_k=self.max_similar_experiences,
                similarity_threshold=self.similarity_threshold,
            )

        except Exception:
            logging.exception("Failed to find relevant experiences")
            return []

    def _extract_lessons(self, similar_experiences: list[SimilarityResult]) -> list[str]:
        """Extract actionable lessons from similar experiences"""

        lessons = []

        try:
            for result in similar_experiences:
                experience = result.experience

                # Add stored lessons
                if experience.lessons_learned:
                    lessons.append(f"Similar scenario lesson: {experience.lessons_learned}")

                # Analyze decision effectiveness
                if experience.actual_outcome:
                    outcome_success = experience.actual_outcome.get("resolution_success", False)
                    llm_action = experience.llm_decision.get("action", "unknown")

                    if outcome_success:
                        lessons.append(f"Effective action in similar case: {llm_action}")
                    else:
                        lessons.append(f"Avoid action from similar failed case: {llm_action}")

                # Safety margin insights
                if experience.safety_metrics:
                    safety_margin = experience.safety_metrics.get("effective_margin", 0)
                    if safety_margin < 0.5:
                        lessons.append("Warning: Similar scenarios had low safety margins")

                # Hallucination warnings
                if experience.hallucination_detected:
                    h_types = ", ".join(experience.hallucination_types)
                    lessons.append(
                        f"Hallucination risk: Previous {h_types} detected in similar scenarios",
                    )

            # Remove duplicates while preserving order
            unique_lessons = []
            seen = set()
            for lesson in lessons:
                if lesson not in seen:
                    unique_lessons.append(lesson)
                    seen.add(lesson)

            return unique_lessons[:10]  # Limit to top 10 lessons

        except Exception:
            logging.exception("Failed to extract lessons")
            return []

    def _check_hallucination_patterns(
        self,
        scenario_context: dict[str, Any],
        environmental_conditions: dict[str, Any],
        similar_experiences: list[SimilarityResult],
    ) -> list[str]:
        """Check for hallucination risk patterns"""

        warnings = []

        try:
            # Get overall hallucination patterns
            patterns = self.replay_store.get_hallucination_patterns()

            if patterns.get("no_data"):
                return warnings

            # Check environmental risk factors
            current_weather = environmental_conditions.get("weather", "unknown")
            if current_weather in patterns.get("environmental_correlations", {}):
                count = patterns["environmental_correlations"][current_weather]
                total = patterns.get("total_hallucinations", 1)
                if count / total > 0.3:  # More than 30% of hallucinations
                    warnings.append(
                        f"High hallucination risk: {current_weather} weather conditions",
                    )

            # Check aircraft type risk factors
            for aircraft in scenario_context.get("aircraft_list", []):
                ac_type = aircraft.get("aircraft_type", "unknown")
                if ac_type in patterns.get("aircraft_type_correlations", {}):
                    count = patterns["aircraft_type_correlations"][ac_type]
                    total = patterns.get("total_hallucinations", 1)
                    if count / total > 0.4:  # More than 40% of hallucinations
                        warnings.append(f"Hallucination risk with {ac_type} aircraft")

            # Check similar experience hallucination rate
            similar_hallucinations = sum(
                1 for result in similar_experiences if result.experience.hallucination_detected
            )

            if similar_experiences and similar_hallucinations / len(similar_experiences) > 0.5:
                warnings.append(
                    "High hallucination risk: Similar scenarios had frequent hallucinations",
                )

            # Check geometric risk factors
            geom_factors = patterns.get("geometric_factors", {})
            if geom_factors:
                avg_sep = geom_factors.get("avg_separation", 10)
                geom_factors.get("min_separation", 10)

                current_sep = 10  # Default if not available
                for result in similar_experiences:
                    current_sep = result.experience.conflict_geometry.get(
                        "closest_approach_distance", 10,
                    )
                    break

                if current_sep < avg_sep * 0.8:  # 20% below average
                    warnings.append(
                        "Hallucination risk: Close separation scenarios are problematic",
                    )

            return warnings

        except Exception:
            logging.exception("Failed to check hallucination patterns")
            return []

    def _enhance_decision_with_experience(
        self,
        llm_decision: dict[str, Any],
        baseline_decision: dict[str, Any],
        similar_experiences: list[SimilarityResult],
    ) -> dict[str, Any]:
        """Enhance current decision using historical experience"""

        try:
            enhanced_decision = llm_decision.copy()

            if not similar_experiences:
                return enhanced_decision

            # Analyze successful actions from similar experiences
            successful_actions = []
            failed_actions = []

            for result in similar_experiences:
                experience = result.experience
                outcome = experience.actual_outcome

                if outcome.get("resolution_success", False):
                    successful_actions.append(experience.llm_decision)
                else:
                    failed_actions.append(experience.llm_decision)

            # Calculate confidence adjustment based on experience
            original_confidence = llm_decision.get("confidence", 0.5)

            # Boost confidence if similar successful experiences
            if successful_actions:
                similar_successful_actions = [
                    action
                    for action in successful_actions
                    if action.get("type") == llm_decision.get("type")
                ]

                if similar_successful_actions:
                    confidence_boost = min(0.2, len(similar_successful_actions) * 0.05)
                    enhanced_decision["confidence"] = min(
                        1.0, original_confidence + confidence_boost,
                    )
                    enhanced_decision["experience_support"] = "positive"

            # Reduce confidence if similar failed experiences
            if failed_actions:
                similar_failed_actions = [
                    action
                    for action in failed_actions
                    if action.get("type") == llm_decision.get("type")
                ]

                if similar_failed_actions:
                    confidence_penalty = min(0.3, len(similar_failed_actions) * 0.1)
                    enhanced_decision["confidence"] = max(
                        0.1, original_confidence - confidence_penalty,
                    )
                    enhanced_decision["experience_support"] = "negative"

            # Add experience-based safety score
            if similar_experiences:
                safety_scores = [
                    exp.experience.safety_metrics.get("effective_margin", 0.5)
                    for exp in similar_experiences
                    if exp.experience.safety_metrics
                ]

                if safety_scores:
                    avg_safety = sum(safety_scores) / len(safety_scores)
                    enhanced_decision["historical_safety_score"] = avg_safety

            # Add alternative recommendations based on successful experiences
            alternative_actions = []
            for result in similar_experiences:
                if result.experience.actual_outcome.get(
                    "resolution_success", False,
                ) and result.experience.llm_decision.get("type") != llm_decision.get("type"):

                    alternative = {
                        "action": result.experience.llm_decision.get("action", ""),
                        "type": result.experience.llm_decision.get("type", ""),
                        "similarity_score": result.similarity_score,
                        "success_rate": 1.0,  # Could be calculated from multiple examples
                    }
                    alternative_actions.append(alternative)

            if alternative_actions:
                # Sort by similarity and limit to top 3
                alternative_actions.sort(key=lambda x: x["similarity_score"], reverse=True)
                enhanced_decision["alternative_actions"] = alternative_actions[:3]

            return enhanced_decision

        except Exception:
            logging.exception("Failed to enhance decision")
            return llm_decision

    def record_resolution_outcome(
        self,
        scenario_context: dict[str, Any],
        conflict_geometry: dict[str, float],
        environmental_conditions: dict[str, Any],
        llm_decision: dict[str, Any],
        baseline_decision: dict[str, Any],
        actual_outcome: dict[str, Any],
        safety_metrics: dict[str, float],
        hallucination_result: dict[str, Any],
        controller_override: Optional[dict[str, Any]] = None,
        lessons_learned: str = "",
    ) -> str:
        """Record the outcome of a conflict resolution for future learning"""

        try:
            if not self.auto_store_enabled:
                return ""

            # Create experience record
            experience = ConflictExperience(
                experience_id="",
                timestamp=time.time(),
                scenario_context=scenario_context,
                conflict_geometry=conflict_geometry,
                environmental_conditions=environmental_conditions,
                llm_decision=llm_decision,
                baseline_decision=baseline_decision,
                actual_outcome=actual_outcome,
                safety_metrics=safety_metrics,
                hallucination_detected=hallucination_result.get("detected", False),
                hallucination_types=hallucination_result.get("types", []),
                controller_override=controller_override,
                lessons_learned=lessons_learned,
            )

            # Store in replay store
            experience_id = self.replay_store.store_experience(experience)

            logging.info("Recorded resolution outcome: %s", experience_id)
            return experience_id

        except Exception:
            logging.exception("Failed to record resolution outcome")
            return ""

    def get_experience_summary(self) -> dict[str, Any]:
        """Get summary of stored experiences"""

        try:
            stats = self.replay_store.get_statistics()
            patterns = self.replay_store.get_hallucination_patterns()

            return {
                "storage_stats": stats,
                "hallucination_patterns": patterns,
                "learning_insights": self._generate_learning_insights(stats, patterns),
            }

        except Exception as e:
            logging.exception("Failed to get experience summary")
            return {"error": str(e)}

    def _generate_learning_insights(
        self, stats: dict[str, Any], patterns: dict[str, Any],
    ) -> list[str]:
        """Generate insights from experience data"""

        insights = []

        try:
            # Statistical insights
            total_exp = stats.get("total_experiences", 0)
            if total_exp > 0:
                hal_rate = stats.get("hallucination_rate", 0)
                override_rate = stats.get("override_rate", 0)

                insights.append(f"Analyzed {total_exp} conflict resolution experiences")
                insights.append(f"Hallucination detection rate: {hal_rate:.1%}")
                insights.append(f"Controller override rate: {override_rate:.1%}")

            # Pattern insights
            if not patterns.get("no_data"):
                hal_types = patterns.get("hallucination_types", {})
                if hal_types:
                    most_common = max(hal_types, key=hal_types.get)
                    insights.append(f"Most common hallucination type: {most_common}")

                env_corr = patterns.get("environmental_correlations", {})
                if env_corr:
                    risky_weather = max(env_corr, key=env_corr.get)
                    insights.append(f"Highest hallucination risk weather: {risky_weather}")

            # Recommendations
            if total_exp < 100:
                insights.append(
                    "Recommendation: Collect more experience data for better pattern analysis",
                )

            if stats.get("hallucination_rate", 0) > 0.2:
                insights.append(
                    "Alert: High hallucination rate detected - review detection thresholds",
                )

            if stats.get("override_rate", 0) > 0.3:
                insights.append("Alert: High override rate - LLM decisions may need improvement")

            return insights

        except Exception:
            logging.exception("Failed to generate learning insights")
            return ["Error generating insights"]

    def store_experience(self, experience_data: dict[str, Any]) -> str:
        """Simple interface to store experience data directly"""
        try:
            # Create a basic ConflictExperience from the provided data
            from llm_atc.memory.replay_store import ConflictExperience

            experience = ConflictExperience(
                experience_id="",
                timestamp=time.time(),
                scenario_context=experience_data.get("scenario", {}),
                conflict_geometry=experience_data.get("conflict_geometry", {}),
                environmental_conditions=experience_data.get("environmental_conditions", {}),
                llm_decision=experience_data.get("action", {}),
                baseline_decision=experience_data.get("baseline_decision", {}),
                actual_outcome=experience_data.get("outcome", {}),
                safety_metrics=experience_data.get("outcome", {}),
                hallucination_detected=experience_data.get("outcome", {}).get(
                    "hallucination_detected", False,
                ),
                hallucination_types=experience_data.get("outcome", {}).get(
                    "hallucination_types", [],
                ),
                controller_override=experience_data.get("controller_override"),
                lessons_learned=experience_data.get("lessons_learned", ""),
            )

            return self.replay_store.store_experience(experience)

        except Exception:
            logging.exception("Failed to store experience")
            return ""


# Testing and usage example
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create replay store and integrator
    replay_store = VectorReplayStore()
    integrator = ExperienceIntegrator(replay_store)

    # Example scenario
    scenario_context = {
        "aircraft_list": [
            {"aircraft_type": "B737", "altitude": 35000, "callsign": "UAL123"},
            {"aircraft_type": "A320", "altitude": 35000, "callsign": "DAL456"},
        ],
    }

    conflict_geometry = {
        "time_to_closest_approach": 120,
        "closest_approach_distance": 4.8,
        "closest_approach_altitude_diff": 0,
    }

    environmental_conditions = {
        "weather": "clear",
        "wind_speed": 12,
        "visibility": 10,
        "turbulence_intensity": 0.1,
    }

    llm_decision = {
        "action": "turn left 15 degrees",
        "type": "heading",
        "confidence": 0.8,
        "safety_score": 0.85,
    }

    baseline_decision = {
        "action": "climb 1000 ft",
        "type": "altitude",
        "confidence": 0.9,
        "safety_score": 0.9,
    }

    # Process with experience integration
    enhanced_decision, lessons = integrator.process_conflict_resolution(
        scenario_context,
        conflict_geometry,
        environmental_conditions,
        llm_decision,
        baseline_decision,
    )

    for _lesson in lessons:
        pass

    # Record outcome
    actual_outcome = {
        "resolution_success": True,
        "separation_achieved": 6.5,
        "time_to_resolution": 95,
    }

    safety_metrics = {
        "effective_margin": 0.82,
        "safety_level": "adequate",
    }

    hallucination_result = {
        "detected": False,
        "types": [],
        "confidence": 0.95,
    }

    exp_id = integrator.record_resolution_outcome(
        scenario_context,
        conflict_geometry,
        environmental_conditions,
        llm_decision,
        baseline_decision,
        actual_outcome,
        safety_metrics,
        hallucination_result,
        lessons_learned="Heading changes effective for parallel conflicts",
    )

    # Get summary
    summary = integrator.get_experience_summary()
