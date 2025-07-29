# agents/scratchpad.py
"""
Scratchpad Agent - Logs and manages step-by-step reasoning and history
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .executor import ExecutionResult
from .planner import ActionPlan, ConflictAssessment
from .verifier import VerificationResult


class StepType(Enum):
    ASSESSMENT = "assessment"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    MONITORING = "monitoring"
    ERROR = "error"
    COMPLETION = "completion"


@dataclass
class ReasoningStep:
    """Individual step in the reasoning process"""
    step_id: str
    step_type: StepType
    timestamp: float
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class SessionSummary:
    """Summary of a complete reasoning session"""
    session_id: str
    start_time: float
    end_time: float
    total_steps: int
    success: bool
    final_status: str
    conflicts_resolved: int
    commands_executed: int
    average_confidence: float
    key_decisions: List[str]
    lessons_learned: List[str]


class Scratchpad:
    """
    Scratchpad agent for logging step-by-step reasoning and maintaining session history
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"
        self.logger = logging.getLogger(__name__)

        # Current session data
        self.current_steps: List[ReasoningStep] = []
        self.session_start_time = time.time()
        self.session_metadata: Dict[str, Any] = {}

        # Historical data
        self.session_history: List[SessionSummary] = []
        self.all_steps_history: List[ReasoningStep] = []

        self.logger.info(f"Scratchpad initialized for session: {self.session_id}")

    def log_step(self, step_data: Dict[str, Any]) -> str:
        """
        Log a reasoning step in the current session
        
        Args:
            step_data: Dictionary containing step information
            
        Returns:
            step_id of the logged step
        """
        try:
            # Generate step ID
            step_id = f"step_{len(self.current_steps) + 1}_{int(time.time() * 1000)}"

            # Extract and validate required fields
            step_type = StepType(step_data.get("type", "monitoring"))
            description = step_data.get("description", "No description provided")
            confidence = float(step_data.get("confidence", 0.5))
            reasoning = step_data.get("reasoning", "No reasoning provided")

            # Create reasoning step
            step = ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                timestamp=time.time(),
                description=description,
                input_data=step_data.get("input_data", {}),
                output_data=step_data.get("output_data", {}),
                confidence=confidence,
                reasoning=reasoning,
                metadata=step_data.get("metadata", {}),
            )

            # Add to current session
            self.current_steps.append(step)

            # Log the step
            self.logger.info(f"Step logged: {step_id} - {step_type.value} - {description}")

            return step_id

        except Exception as e:
            self.logger.error(f"Error logging step: {e}")
            return ""

    def log_assessment_step(self, assessment: ConflictAssessment) -> str:
        """Log a conflict assessment step"""
        step_data = {
            "type": "assessment",
            "description": f"Assessed conflict {assessment.conflict_id}",
            "confidence": assessment.confidence,
            "reasoning": assessment.reasoning,
            "input_data": {
                "aircraft_involved": assessment.aircraft_involved,
                "severity": assessment.severity,
                "time_to_conflict": assessment.time_to_conflict,
            },
            "output_data": {
                "conflict_id": assessment.conflict_id,
                "recommended_action": assessment.recommended_action.value,
                "confidence": assessment.confidence,
            },
            "metadata": assessment.metadata,
        }
        return self.log_step(step_data)

    def log_planning_step(self, plan: ActionPlan) -> str:
        """Log an action planning step"""
        step_data = {
            "type": "planning",
            "description": f"Generated plan {plan.plan_id}",
            "confidence": plan.confidence,
            "reasoning": plan.reasoning,
            "input_data": {
                "conflict_id": plan.conflict_id,
                "plan_type": plan.plan_type.value,
                "target_aircraft": plan.target_aircraft,
            },
            "output_data": {
                "plan_id": plan.plan_id,
                "commands": plan.commands,
                "expected_outcome": plan.expected_outcome,
                "priority": plan.priority,
            },
            "metadata": {
                "created_at": plan.created_at,
            },
        }
        return self.log_step(step_data)

    def log_execution_step(self, execution: ExecutionResult) -> str:
        """Log a plan execution step"""
        step_data = {
            "type": "execution",
            "description": f"Executed plan {execution.plan_id}",
            "confidence": execution.success_rate,
            "reasoning": f"Sent {len(execution.commands_sent)} commands with {execution.success_rate:.1%} success rate",
            "input_data": {
                "plan_id": execution.plan_id,
                "commands": execution.commands_sent,
            },
            "output_data": {
                "execution_id": execution.execution_id,
                "status": execution.status.value,
                "success_rate": execution.success_rate,
                "execution_time": execution.execution_time,
                "responses": execution.responses,
            },
            "metadata": {
                "error_messages": execution.error_messages,
                "created_at": execution.created_at,
            },
        }
        return self.log_step(step_data)

    def log_verification_step(self, verification: VerificationResult) -> str:
        """Log a verification step"""
        step_data = {
            "type": "verification",
            "description": f"Verified execution {verification.execution_id}",
            "confidence": verification.confidence,
            "reasoning": f"Performed {len(verification.checks_performed)} checks, status: {verification.status.value}",
            "input_data": {
                "execution_id": verification.execution_id,
                "checks_performed": verification.checks_performed,
            },
            "output_data": {
                "verification_id": verification.verification_id,
                "status": verification.status.value,
                "safety_score": verification.safety_score,
                "passed_checks": verification.passed_checks,
                "failed_checks": verification.failed_checks,
                "warnings": verification.warnings,
            },
            "metadata": {
                "verification_time": verification.verification_time,
                "created_at": verification.created_at,
            },
        }
        return self.log_step(step_data)

    def log_error_step(self, error_msg: str, error_data: Optional[Dict[str, Any]] = None) -> str:
        """Log an error step"""
        step_data = {
            "type": "error",
            "description": f"Error occurred: {error_msg}",
            "confidence": 0.0,
            "reasoning": error_msg,
            "input_data": error_data or {},
            "output_data": {"error": error_msg},
            "metadata": {"timestamp": time.time()},
        }
        return self.log_step(step_data)

    def log_monitoring_step(self, monitoring_data: Dict[str, Any]) -> str:
        """Log a monitoring step"""
        step_data = {
            "type": "monitoring",
            "description": "Monitoring aircraft status",
            "confidence": 1.0,
            "reasoning": "Continuous monitoring of aircraft positions and trajectories",
            "input_data": monitoring_data,
            "output_data": {"status": "monitoring"},
            "metadata": {},
        }
        return self.log_step(step_data)

    def get_history(self) -> Dict[str, Any]:
        """
        Get complete history of the current session
        
        Returns:
            Dictionary containing session history and steps
        """
        return {
            "session_id": self.session_id,
            "session_start_time": self.session_start_time,
            "current_time": time.time(),
            "session_duration": time.time() - self.session_start_time,
            "total_steps": len(self.current_steps),
            "steps": [asdict(step) for step in self.current_steps],
            "session_metadata": self.session_metadata,
            "summary": self._generate_session_summary(),
        }

    def get_step_by_id(self, step_id: str) -> Optional[ReasoningStep]:
        """Get a specific step by its ID"""
        for step in self.current_steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_by_type(self, step_type: StepType) -> List[ReasoningStep]:
        """Get all steps of a specific type"""
        return [step for step in self.current_steps if step.step_type == step_type]

    def get_recent_steps(self, count: int = 5) -> List[ReasoningStep]:
        """Get the most recent steps"""
        return self.current_steps[-count:] if count <= len(self.current_steps) else self.current_steps

    def complete_session(self, success: bool = True, final_status: str = "completed") -> SessionSummary:
        """
        Complete the current session and generate summary
        
        Args:
            success: Whether the session completed successfully
            final_status: Final status description
            
        Returns:
            SessionSummary of the completed session
        """
        try:
            # Generate session summary
            summary = SessionSummary(
                session_id=self.session_id,
                start_time=self.session_start_time,
                end_time=time.time(),
                total_steps=len(self.current_steps),
                success=success,
                final_status=final_status,
                conflicts_resolved=len(self.get_steps_by_type(StepType.ASSESSMENT)),
                commands_executed=len(self.get_steps_by_type(StepType.EXECUTION)),
                average_confidence=self._calculate_average_confidence(),
                key_decisions=self._extract_key_decisions(),
                lessons_learned=self._extract_lessons_learned(),
            )

            # Store in history
            self.session_history.append(summary)
            self.all_steps_history.extend(self.current_steps)

            # Log completion
            self.logger.info(f"Session completed: {self.session_id} - {final_status}")

            return summary

        except Exception as e:
            self.logger.error(f"Error completing session: {e}")
            return SessionSummary(
                session_id=self.session_id,
                start_time=self.session_start_time,
                end_time=time.time(),
                total_steps=len(self.current_steps),
                success=False,
                final_status=f"error: {e!s}",
                conflicts_resolved=0,
                commands_executed=0,
                average_confidence=0.0,
                key_decisions=[],
                lessons_learned=[],
            )

    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new reasoning session
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            New session ID
        """
        # Complete current session if not already completed
        if self.current_steps:
            self.complete_session(success=True, final_status="auto_completed")

        # Reset for new session
        self.session_id = session_id or f"session_{int(time.time() * 1000)}"
        self.current_steps = []
        self.session_start_time = time.time()
        self.session_metadata = {}

        self.logger.info(f"New session started: {self.session_id}")
        return self.session_id

    def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current session"""
        if not self.current_steps:
            return {"summary": "No steps recorded"}

        step_types = {}
        for step in self.current_steps:
            step_type = step.step_type.value
            step_types[step_type] = step_types.get(step_type, 0) + 1

        return {
            "total_steps": len(self.current_steps),
            "step_types": step_types,
            "average_confidence": self._calculate_average_confidence(),
            "session_duration": time.time() - self.session_start_time,
            "first_step": self.current_steps[0].description if self.current_steps else None,
            "last_step": self.current_steps[-1].description if self.current_steps else None,
        }

    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all steps"""
        if not self.current_steps:
            return 0.0

        total_confidence = sum(step.confidence for step in self.current_steps)
        return total_confidence / len(self.current_steps)

    def _extract_key_decisions(self) -> List[str]:
        """Extract key decisions from the session"""
        key_decisions = []

        # Look for planning and execution steps with high confidence
        for step in self.current_steps:
            if step.step_type in [StepType.PLANNING, StepType.EXECUTION] and step.confidence > 0.7:
                key_decisions.append(f"{step.step_type.value}: {step.description}")

        return key_decisions

    def _extract_lessons_learned(self) -> List[str]:
        """Extract lessons learned from errors and low-confidence steps"""
        lessons = []

        # Look for error steps
        error_steps = self.get_steps_by_type(StepType.ERROR)
        for step in error_steps:
            lessons.append(f"Error: {step.reasoning}")

        # Look for low-confidence steps
        low_confidence_steps = [step for step in self.current_steps if step.confidence < 0.5]
        for step in low_confidence_steps:
            lessons.append(f"Low confidence in {step.step_type.value}: {step.reasoning}")

        return lessons

    def export_session_data(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export session data in specified format
        
        Args:
            format: Export format ('json', 'dict')
            
        Returns:
            Session data in requested format
        """
        data = self.get_history()

        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        return data

    def set_session_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata for the current session"""
        self.session_metadata.update(metadata)

    def get_session_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the current session"""
        if not self.current_steps:
            return {"error": "No steps recorded"}

        return {
            "session_id": self.session_id,
            "duration": time.time() - self.session_start_time,
            "total_steps": len(self.current_steps),
            "average_confidence": self._calculate_average_confidence(),
            "step_types": {step_type.value: len(self.get_steps_by_type(step_type))
                          for step_type in StepType},
            "error_count": len(self.get_steps_by_type(StepType.ERROR)),
            "completion_rate": len([s for s in self.current_steps if s.confidence > 0.7]) / len(self.current_steps),
        }
