# agents/executor.py
"""
Executor Agent - Executes action plans and sends commands to BlueSky
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from .planner import ActionPlan

# Constants
PARTIAL_SUCCESS_THRESHOLD = 0.5


class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of plan execution"""

    execution_id: str
    plan_id: str
    status: ExecutionStatus
    commands_sent: list[str]
    responses: list[dict[str, Any]]
    success_rate: float
    execution_time: float
    error_messages: list[str]
    created_at: float


class Executor:
    """
    Executor agent responsible for sending action plans to BlueSky simulator
    """

    def __init__(self, command_sender: Optional[Callable] = None) -> None:
        self.command_sender = command_sender
        self.logger = logging.getLogger(__name__)
        self.execution_history: list[ExecutionResult] = []
        self.active_executions: dict[str, ExecutionResult] = {}

    def send_plan(self, plan: ActionPlan) -> ExecutionResult:
        """
        Execute an action plan by sending commands to BlueSky

        Args:
            plan: ActionPlan to execute

        Returns:
            ExecutionResult with execution status and details
        """
        try:
            self.logger.info(
                "Executing plan %s for conflict %s", plan.plan_id, plan.conflict_id
            )

            # Generate execution ID
            execution_id = f"exec_{int(time.time() * 1000)}"
            start_time = time.time()

            # Initialize execution result
            result = ExecutionResult(
                execution_id=execution_id,
                plan_id=plan.plan_id,
                status=ExecutionStatus.EXECUTING,
                commands_sent=[],
                responses=[],
                success_rate=0.0,
                execution_time=0.0,
                error_messages=[],
                created_at=start_time,
            )

            # Add to active executions
            self.active_executions[execution_id] = result

            # Execute commands sequentially
            successful_commands = 0

            for command in plan.commands:
                try:
                    self.logger.info("Sending command: %s", command)

                    # Send command through the command sender
                    response = self._send_command(command)

                    result.commands_sent.append(command)
                    result.responses.append(response)

                    if response.get("success", False):
                        successful_commands += 1
                        self.logger.info("Command executed successfully: %s", command)
                    else:
                        error_msg = f"Command failed: {command} - {response.get('error', 'Unknown error')}"
                        result.error_messages.append(error_msg)
                        self.logger.error(error_msg)

                    # Brief delay between commands to avoid overwhelming the simulator
                    time.sleep(0.1)

                except Exception as e:
                    error_msg = f"Exception executing command {command}: {e!s}"
                    result.error_messages.append(error_msg)
                    self.logger.exception("Exception executing command")

            # Calculate execution metrics
            result.success_rate = (
                successful_commands / len(plan.commands) if plan.commands else 0.0
            )
            result.execution_time = time.time() - start_time

            # Determine final status
            if result.success_rate == 1.0:
                result.status = ExecutionStatus.COMPLETED
                self.logger.info(
                    "Plan execution completed successfully: %s", execution_id
                )
            elif result.success_rate > PARTIAL_SUCCESS_THRESHOLD:
                # Partial success still considered completed
                result.status = ExecutionStatus.COMPLETED
                self.logger.warning(
                    "Plan execution completed with partial success: %s",
                    execution_id,
                )
            else:
                result.status = ExecutionStatus.FAILED
                self.logger.error("Plan execution failed: %s", execution_id)

            # Remove from active executions and add to history
            del self.active_executions[execution_id]
            self.execution_history.append(result)

        except Exception as e:
            self.logger.exception("Critical error in plan execution")

            # Create failure result
            failure_result = ExecutionResult(
                execution_id=f"exec_failed_{int(time.time() * 1000)}",
                plan_id=plan.plan_id,
                status=ExecutionStatus.FAILED,
                commands_sent=[],
                responses=[],
                success_rate=0.0,
                execution_time=(
                    time.time() - start_time if "start_time" in locals() else 0.0
                ),
                error_messages=[str(e)],
                created_at=time.time(),
            )

            self.execution_history.append(failure_result)
            return failure_result

        return result

    def _send_command(self, command: str) -> dict[str, Any]:
        """
        Send a single command to BlueSky simulator

        Args:
            command: BlueSky command string

        Returns:
            Response dictionary with success status and details
        """
        try:
            # Handle special monitoring command
            if command == "CONTINUE_MONITORING":
                return {
                    "success": True,
                    "command": command,
                    "response": "Monitoring continued",
                    "timestamp": time.time(),
                }

            # Use command sender if available, otherwise simulate
            if self.command_sender:
                response = self.command_sender(command)
                return {
                    "success": True,
                    "command": command,
                    "response": response,
                    "timestamp": time.time(),
                }
            # Simulate command execution for testing
            return self._simulate_command_execution(command)

        except Exception as e:
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "timestamp": time.time(),
            }

    def _simulate_command_execution(self, command: str) -> dict[str, Any]:
        """
        Simulate command execution for testing purposes

        Args:
            command: BlueSky command string

        Returns:
            Simulated response dictionary
        """
        # Parse command type
        command_parts = command.split()
        if not command_parts:
            return {
                "success": False,
                "command": command,
                "error": "Empty command",
                "timestamp": time.time(),
            }

        command_type = command_parts[0].upper()

        # Simulate different command types
        if command_type in ["ALT", "HDG", "SPD"]:
            return {
                "success": True,
                "command": command,
                "response": f"{command_type} command acknowledged",
                "simulation": True,
                "timestamp": time.time(),
            }
        return {
            "success": False,
            "command": command,
            "error": f"Unknown command type: {command_type}",
            "timestamp": time.time(),
        }

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        if execution_id in self.active_executions:
            self.active_executions[execution_id].status = ExecutionStatus.CANCELLED
            self.logger.info("Execution cancelled: %s", execution_id)
            return True
        self.logger.warning("Cannot cancel execution - not found: %s", execution_id)
        return False

    def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """
        Get current status of an execution

        Args:
            execution_id: ID of execution to check

        Returns:
            ExecutionStatus or None if not found
        """
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].status

        # Check history
        for result in self.execution_history:
            if result.execution_id == execution_id:
                return result.status

        return None

    def get_active_executions(self) -> dict[str, ExecutionResult]:
        """Get all currently active executions"""
        return self.active_executions.copy()

    def get_execution_history(self) -> list[ExecutionResult]:
        """Get history of all executions"""
        return self.execution_history.copy()

    def get_execution_metrics(self) -> dict[str, Any]:
        """Get overall execution performance metrics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "total_commands_sent": 0,
            }

        total_executions = len(self.execution_history)
        successful_executions = len(
            [
                r
                for r in self.execution_history
                if r.status == ExecutionStatus.COMPLETED
            ],
        )
        total_execution_time = sum(r.execution_time for r in self.execution_history)
        total_commands = sum(len(r.commands_sent) for r in self.execution_history)

        return {
            "total_executions": total_executions,
            "success_rate": successful_executions / total_executions,
            "average_execution_time": total_execution_time / total_executions,
            "total_commands_sent": total_commands,
        }

    def set_command_sender(self, command_sender: Callable) -> None:
        """
        Set the command sender function for actual BlueSky integration

        Args:
            command_sender: Function that takes a command string and returns response
        """
        self.command_sender = command_sender
        self.logger.info("Command sender configured")
