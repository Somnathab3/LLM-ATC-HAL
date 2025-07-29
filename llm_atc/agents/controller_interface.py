# agents/controller_interface.py
"""
Real-Time Human-AI Oversight System for ATC Operations
Provides controller interface with confidence displays and override capabilities
Enhanced with embodied agent planning loop
"""

import json
import logging
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from llm_atc.tools import bluesky_tools

from .executor import ExecutionResult, Executor

# Import embodied agent components
from .planner import ActionPlan, ConflictAssessment, Planner
from .scratchpad import Scratchpad
from .verifier import Verifier


class ConfidenceLevel(Enum):
    CRITICAL = "critical"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXCELLENT = "excellent"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ConflictDisplay:
    """Conflict information for display"""
    conflict_id: str
    aircraft_ids: list[str]
    time_to_conflict: float
    current_separation: float
    severity: str
    llm_recommendation: dict
    baseline_recommendation: dict
    confidence_level: ConfidenceLevel
    safety_flags: list[str]

@dataclass
class OverrideDecision:
    """Controller override decision"""
    conflict_id: str
    override_action: str
    reason: str
    timestamp: float
    controller_id: str

class SafetyMonitor:
    """Real-time safety monitoring system"""

    def __init__(self):
        self.thresholds = {
            "response_time_limit": 5.0,  # seconds
            "confidence_threshold": 0.6,
            "safety_score_minimum": 0.4,
            "uncertainty_maximum": 0.7,
        }

        self.alerts = []
        self.monitoring_active = False

    def monitor_llm_output(self,
                          response: dict,
                          response_time: float,
                          confidence: float,
                          uncertainty: float) -> list[str]:
        """Monitor LLM output for safety concerns"""

        alerts = []

        # Check response time
        if response_time > self.thresholds["response_time_limit"]:
            alerts.append(f"LLM response time exceeded limit: {response_time:.2f}s")

        # Check confidence level
        if confidence < self.thresholds["confidence_threshold"]:
            alerts.append(f"Low LLM confidence: {confidence:.3f}")

        # Check safety score
        safety_score = response.get("safety_score", 0.0)
        if safety_score < self.thresholds["safety_score_minimum"]:
            alerts.append(f"Low safety score: {safety_score:.3f}")

        # Check uncertainty
        if uncertainty > self.thresholds["uncertainty_maximum"]:
            alerts.append(f"High uncertainty: {uncertainty:.3f}")

        # Check for error indicators
        if "error" in response or response.get("action", "").lower().startswith("error"):
            alerts.append("LLM returned error response")

        # Store alerts for history
        for alert in alerts:
            self.alerts.append({
                "timestamp": time.time(),
                "alert": alert,
                "response": response,
            })

        return alerts

    def escalate_to_human(self, conflict_id: str, reason: str) -> bool:
        """Escalate decision to human controller"""

        escalation_log = {
            "timestamp": time.time(),
            "conflict_id": conflict_id,
            "reason": reason,
            "action": "escalated_to_human",
        }

        logging.warning("Escalating to human: %s", escalation_log)
        return True

    def get_recent_alerts(self, time_window: float = 300.0) -> list[dict]:
        """Get alerts from recent time window"""

        current_time = time.time()
        [
            alert for alert in self.alerts
            if current_time - alert["timestamp"] <= time_window
        ]

        return [
            alert for alert in self.alert_history
            if current_time - alert["timestamp"] <= time_window
        ]

class ControllerInterface:
    """Main controller interface for human-AI oversight with embodied agent planning loop"""

    def __init__(self, llm_client=None):
        self.root = tk.Tk()
        self.root.title("ATC Hallucination Detection & Safety Assurance System")
        self.root.geometry("1400x900")

        # Initialize safety monitor
        self.safety_monitor = SafetyMonitor()
        self.active_conflicts = {}
        self.override_history = []
        self.update_callbacks = []

        # Initialize embodied agent components
        self.planner = Planner(llm_client=llm_client)
        self.executor = Executor(command_sender=self._send_bluesky_command)
        self.verifier = Verifier()
        self.scratchpad = Scratchpad()

        # Planning loop control
        self.planning_active = False
        self.planning_thread = None
        self.max_planning_iterations = 10

        # Setup UI
        self._setup_ui()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_active = True
        self.monitoring_thread.start()

    def start_planning_loop(self) -> dict[str, Any]:
        """
        Start the embodied agent planning loop

        Returns:
            Dictionary with status and history
        """
        if self.planning_active:
            return {"status": "already_running", "message": "Planning loop already active"}

        self.planning_active = True

        try:
            self.scratchpad.start_new_session()

            # Log initial monitoring step
            initial_data = bluesky_tools.GetAllAircraftInfo()
            self.scratchpad.log_monitoring_step(initial_data)

            iteration_count = 0

            while self.planning_active and iteration_count < self.max_planning_iterations:
                try:
                    # Step 1: Get aircraft information
                    info = bluesky_tools.GetAllAircraftInfo()

                    # Step 2: Assess conflicts
                    assessment = self.planner.assess_conflict(info)

                    if assessment is None:
                        # No conflicts detected, continue monitoring
                        self.scratchpad.log_monitoring_step(info)
                        time.sleep(5)  # Wait before next check
                        continue

                    # Log assessment step
                    self.scratchpad.log_assessment_step(assessment)

                    # Step 3: Generate action plan
                    plan = self.planner.generate_action_plan(assessment)

                    if plan is None:
                        self.scratchpad.log_error_step("Failed to generate action plan")
                        break

                    # Log planning step
                    self.scratchpad.log_planning_step(plan)

                    # Step 4: Execute plan
                    exec_result = self.executor.send_plan(plan)

                    # Log execution step
                    self.scratchpad.log_execution_step(exec_result)

                    # Step 5: Verify execution
                    verification_passed = self.verifier.check(exec_result, timeout_seconds=5)

                    # Get verification result from history
                    verification_results = self.verifier.get_verification_history()
                    if verification_results:
                        latest_verification = verification_results[-1]
                        self.scratchpad.log_verification_step(latest_verification)

                    # Step 6: Check if we should continue
                    if not verification_passed:
                        self.scratchpad.log_error_step("Verification failed, stopping planning loop")
                        break

                    # Update UI with current conflict
                    self._update_conflict_display(assessment, plan, exec_result)

                    iteration_count += 1

                    # Brief pause between iterations
                    time.sleep(1)

                except Exception as e:
                    self.scratchpad.log_error_step(f"Planning loop iteration error: {e!s}")
                    logging.exception("Planning loop error")
                    break

            # Complete session
            session_summary = self.scratchpad.complete_session(
                success=True,
                final_status="completed" if iteration_count < self.max_planning_iterations else "max_iterations_reached",
            )

            return {
                "status": "resolved",
                "history": self.scratchpad.get_history(),
                "session_summary": session_summary,
                "iterations": iteration_count,
            }

        except Exception as e:
            self.scratchpad.log_error_step(f"Critical planning loop error: {e!s}")
            logging.exception("Critical planning loop error")

            return {
                "status": "error",
                "error": str(e),
                "history": self.scratchpad.get_history(),
            }
        finally:
            self.planning_active = False

    def stop_planning_loop(self):
        """Stop the planning loop"""
        self.planning_active = False

    def _send_bluesky_command(self, command: str) -> dict[str, Any]:
        """
        Send command to BlueSky through the tools interface

        Args:
            command: BlueSky command string

        Returns:
            Command response dictionary
        """
        try:
            return bluesky_tools.SendCommand(command)
        except Exception as e:
            logging.exception("Error sending BlueSky command")
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "timestamp": time.time(),
            }

    def _update_conflict_display(self, assessment: ConflictAssessment, plan: ActionPlan, execution: ExecutionResult):
        """Update the UI with current conflict information"""
        try:
            # Create conflict display object
            conflict_display = ConflictDisplay(
                conflict_id=assessment.conflict_id,
                aircraft_ids=assessment.aircraft_involved,
                time_to_conflict=assessment.time_to_conflict,
                current_separation=assessment.metadata.get("separation", {}).get("horizontal", 0.0),
                severity=assessment.severity,
                llm_recommendation={
                    "action_type": assessment.recommended_action.value,
                    "reasoning": assessment.reasoning,
                    "commands": plan.commands if plan else [],
                },
                baseline_recommendation={
                    "action_type": assessment.recommended_action.value,
                    "commands": plan.commands if plan else [],
                },
                confidence_level=self._convert_confidence_to_level(assessment.confidence),
                safety_flags=[],
            )

            # Update conflicts tree
            self._add_conflict_to_tree(conflict_display)

            # Update details if this conflict is selected
            self._update_conflict_details(conflict_display)

        except Exception:
            logging.exception("Error updating conflict display")

    def _convert_confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numerical confidence to ConfidenceLevel enum"""
        if confidence >= 0.9:
            return ConfidenceLevel.EXCELLENT
        if confidence >= 0.75:
            return ConfidenceLevel.HIGH
        if confidence >= 0.6:
            return ConfidenceLevel.MODERATE
        if confidence >= 0.4:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.CRITICAL

    def _add_conflict_to_tree(self, conflict: ConflictDisplay):
        """Add conflict to the conflicts tree view"""
        try:
            aircraft_str = ", ".join(conflict.aircraft_ids)
            time_str = f"{conflict.time_to_conflict:.0f}s"
            sep_str = f"{conflict.current_separation:.1f}nm"
            conf_str = conflict.confidence_level.value
            action_str = conflict.llm_recommendation.get("action_type", "unknown")

            # Insert into tree
            item = self.conflicts_tree.insert("", "end", values=(
                conflict.conflict_id,
                aircraft_str,
                time_str,
                sep_str,
                conf_str,
                action_str,
            ))

            # Store conflict data with tree item
            self.active_conflicts[item] = conflict

        except Exception:
            logging.exception("Error adding conflict to tree")

    def _update_conflict_details(self, conflict: ConflictDisplay):
        """Update the conflict details text area"""
        try:
            details = f"""Conflict ID: {conflict.conflict_id}
Aircraft: {', '.join(conflict.aircraft_ids)}
Severity: {conflict.severity}
Time to Conflict: {conflict.time_to_conflict:.0f} seconds
Current Separation: {conflict.current_separation:.1f} nautical miles
Confidence Level: {conflict.confidence_level.value}

AI Recommendation:
Action Type: {conflict.llm_recommendation.get('action_type', 'N/A')}
Commands: {', '.join(conflict.llm_recommendation.get('commands', []))}
Reasoning: {conflict.llm_recommendation.get('reasoning', 'N/A')}

Safety Flags: {', '.join(conflict.safety_flags) if conflict.safety_flags else 'None'}
"""

            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, details)

        except Exception:
            logging.exception("Error updating conflict details")

    def get_agent_status(self) -> dict[str, Any]:
        """Get status of all embodied agent components"""
        return {
            "planner": {
                "assessments_made": len(self.planner.get_assessment_history()),
                "plans_generated": len(self.planner.get_plan_history()),
            },
            "executor": self.executor.get_execution_metrics(),
            "verifier": self.verifier.get_verification_metrics(),
            "scratchpad": self.scratchpad.get_session_metrics(),
            "planning_active": self.planning_active,
        }

    def _setup_ui(self):
        """Setup the user interface"""

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="ATC AI Oversight System",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Active Conflicts Tab
        self.conflicts_frame = ttk.Frame(notebook)
        notebook.add(self.conflicts_frame, text="Active Conflicts")
        self._setup_conflicts_tab()

        # Safety Monitor Tab
        self.safety_frame = ttk.Frame(notebook)
        notebook.add(self.safety_frame, text="Safety Monitor")
        self._setup_safety_tab()

        # System Status Tab
        self.status_frame = ttk.Frame(notebook)
        notebook.add(self.status_frame, text="System Status")
        self._setup_status_tab()

        # Override History Tab
        self.history_frame = ttk.Frame(notebook)
        notebook.add(self.history_frame, text="Override History")
        self._setup_history_tab()

    def _setup_conflicts_tab(self):
        """Setup active conflicts display"""

        # Conflicts list
        conflicts_label = ttk.Label(self.conflicts_frame, text="Active Conflicts",
                                   font=("Arial", 12, "bold"))
        conflicts_label.pack(anchor=tk.W, pady=(0, 5))

        # Treeview for conflicts
        columns = ("ID", "Aircraft", "Time", "Separation", "Confidence", "Action")
        self.conflicts_tree = ttk.Treeview(self.conflicts_frame, columns=columns, show="headings", height=8)

        for col in columns:
            self.conflicts_tree.heading(col, text=col)
            self.conflicts_tree.column(col, width=120)

        self.conflicts_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Bind selection event
        self.conflicts_tree.bind("<<TreeviewSelect>>", self._on_conflict_select)

        # Conflict details frame
        details_frame = ttk.LabelFrame(self.conflicts_frame, text="Conflict Details")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Details display
        self.details_text = tk.Text(details_frame, height=8, wrap=tk.WORD)
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)

        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Override controls
        override_frame = ttk.Frame(details_frame)
        override_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(override_frame, text="Override Action:").pack(side=tk.LEFT)
        self.override_entry = ttk.Entry(override_frame, width=40)
        self.override_entry.pack(side=tk.LEFT, padx=(5, 0))

        self.override_button = ttk.Button(override_frame, text="Override",
                                         command=self._override_decision)
        self.override_button.pack(side=tk.LEFT, padx=(10, 0))

        self.accept_button = ttk.Button(override_frame, text="Accept AI",
                                       command=self._accept_ai_decision)
        self.accept_button.pack(side=tk.LEFT, padx=(5, 0))

    def _setup_safety_tab(self):
        """Setup safety monitoring display"""

        # Safety alerts
        alerts_label = ttk.Label(self.safety_frame, text="Safety Alerts",
                                font=("Arial", 12, "bold"))
        alerts_label.pack(anchor=tk.W, pady=(0, 5))

        # Alerts listbox
        alerts_frame = ttk.Frame(self.safety_frame)
        alerts_frame.pack(fill=tk.BOTH, expand=True)

        self.alerts_listbox = tk.Listbox(alerts_frame, height=10)
        alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL,
                                        command=self.alerts_listbox.yview)
        self.alerts_listbox.configure(yscrollcommand=alerts_scrollbar.set)

        self.alerts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Safety metrics
        metrics_frame = ttk.LabelFrame(self.safety_frame, text="Safety Metrics")
        metrics_frame.pack(fill=tk.X, pady=(10, 0))

        # Create figure for metrics visualization
        self.metrics_figure = Figure(figsize=(12, 4), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_figure, metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._update_metrics_plot()

    def _setup_status_tab(self):
        """Setup system status display"""

        status_label = ttk.Label(self.status_frame, text="System Status",
                                font=("Arial", 12, "bold"))
        status_label.pack(anchor=tk.W, pady=(0, 10))

        # System info
        info_frame = ttk.LabelFrame(self.status_frame, text="System Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Performance metrics
        perf_frame = ttk.LabelFrame(self.status_frame, text="Performance Metrics")
        perf_frame.pack(fill=tk.X, pady=(0, 10))

        self.perf_text = tk.Text(perf_frame, height=8, wrap=tk.WORD)
        self.perf_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control buttons
        control_frame = ttk.Frame(self.status_frame)
        control_frame.pack(fill=tk.X)

        ttk.Button(control_frame, text="Export Report",
                  command=self._export_report).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Reset Alerts",
                  command=self._reset_alerts).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Emergency Stop",
                  command=self._emergency_stop).pack(side=tk.RIGHT)

    def _setup_history_tab(self):
        """Setup override history display"""

        history_label = ttk.Label(self.history_frame, text="Override History",
                                 font=("Arial", 12, "bold"))
        history_label.pack(anchor=tk.W, pady=(0, 5))

        # History treeview
        hist_columns = ("Timestamp", "Conflict ID", "Action", "Reason")
        self.history_tree = ttk.Treeview(self.history_frame, columns=hist_columns,
                                        show="headings", height=15)

        for col in hist_columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=150)

        self.history_tree.pack(fill=tk.BOTH, expand=True)

    def add_conflict(self, conflict: ConflictDisplay):
        """Add a new conflict to the display"""

        self.active_conflicts[conflict.conflict_id] = conflict

        # Add to treeview
        self.conflicts_tree.insert("", tk.END, iid=conflict.conflict_id, values=(
            conflict.conflict_id,
            ", ".join(conflict.aircraft_ids),
            f"{conflict.time_to_conflict:.0f}s",
            f"{conflict.current_separation:.1f}NM",
            conflict.confidence_level.value,
            conflict.llm_recommendation.get("action", "Unknown"),
        ))

        # Color code by confidence level
        self._color_code_conflict(conflict.conflict_id, conflict.confidence_level)

        # Check for escalation
        if (conflict.confidence_level in [ConfidenceLevel.CRITICAL, ConfidenceLevel.LOW] or
            conflict.safety_flags):
            self.safety_monitor.escalate_to_human(conflict.conflict_id,
                                                 "Low confidence or safety flags")

    def _color_code_conflict(self, conflict_id: str, confidence: ConfidenceLevel):
        """Color code conflict based on confidence level"""

        color_map = {
            ConfidenceLevel.CRITICAL: "red",
            ConfidenceLevel.LOW: "orange",
            ConfidenceLevel.MODERATE: "yellow",
            ConfidenceLevel.HIGH: "lightgreen",
            ConfidenceLevel.EXCELLENT: "green",
        }

        color = color_map.get(confidence, "white")
        self.conflicts_tree.set(conflict_id, "Confidence", confidence.value)

        # Configure tag for color
        tag_name = f"confidence_{confidence.value}"
        self.conflicts_tree.tag_configure(tag_name, background=color)
        self.conflicts_tree.item(conflict_id, tags=(tag_name,))

    def _on_conflict_select(self, event):
        """Handle conflict selection"""

        selection = self.conflicts_tree.selection()
        if not selection:
            return

        conflict_id = selection[0]
        conflict = self.active_conflicts.get(conflict_id)

        if conflict:
            self._display_conflict_details(conflict)

    def _display_conflict_details(self, conflict: ConflictDisplay):
        """Display detailed conflict information"""

        self.details_text.delete(1.0, tk.END)

        details = f"""Conflict ID: {conflict.conflict_id}
Aircraft: {', '.join(conflict.aircraft_ids)}
Time to Conflict: {conflict.time_to_conflict:.1f} seconds
Current Separation: {conflict.current_separation:.1f} nautical miles
Severity: {conflict.severity}
Confidence Level: {conflict.confidence_level.value}

LLM Recommendation:
  Action: {conflict.llm_recommendation.get('action', 'N/A')}
  Type: {conflict.llm_recommendation.get('type', 'N/A')}
  Safety Score: {conflict.llm_recommendation.get('safety_score', 'N/A')}
  Reasoning: {conflict.llm_recommendation.get('reasoning', 'N/A')}

Baseline Recommendation:
  Action: {conflict.baseline_recommendation.get('action', 'N/A')}
  Type: {conflict.baseline_recommendation.get('type', 'N/A')}
  Safety Score: {conflict.baseline_recommendation.get('safety_score', 'N/A')}

Safety Flags:
{chr(10).join(f"  - {flag}" for flag in conflict.safety_flags) if conflict.safety_flags else "  None"}
"""

        self.details_text.insert(1.0, details)

    def _override_decision(self):
        """Handle controller override"""

        selection = self.conflicts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conflict to override.")
            return

        conflict_id = selection[0]
        override_action = self.override_entry.get().strip()

        if not override_action:
            messagebox.showwarning("Invalid Action", "Please enter an override action.")
            return

        # Create override decision
        override = OverrideDecision(
            conflict_id=conflict_id,
            override_action=override_action,
            reason=messagebox.askstring("Override Reason", "Reason for override:") or "No reason provided",
            timestamp=time.time(),
            controller_id="CTRL001",  # In practice, get from user authentication
        )

        self.override_history.append(override)

        # Update history display
        self._update_history_display(override)

        # Log override
        logging.info("Controller override: %s", override)

        # Clear entry
        self.override_entry.delete(0, tk.END)

        messagebox.showinfo("Override Recorded", f"Override recorded for conflict {conflict_id}")

    def _accept_ai_decision(self):
        """Handle controller acceptance of AI decision"""

        selection = self.conflicts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conflict.")
            return

        conflict_id = selection[0]

        # Log acceptance
        logging.info("Controller accepted AI decision for conflict %s", conflict_id)

        messagebox.showinfo("Decision Accepted", f"AI decision accepted for conflict {conflict_id}")

    def _update_history_display(self, override: OverrideDecision):
        """Update override history display"""

        timestamp_str = time.strftime("%H:%M:%S", time.localtime(override.timestamp))

        self.history_tree.insert("", 0, values=(
            timestamp_str,
            override.conflict_id,
            override.override_action,
            override.reason,
        ))

    def _update_metrics_plot(self):
        """Update safety metrics visualization"""

        self.metrics_figure.clear()

        # Create subplots
        ax1 = self.metrics_figure.add_subplot(131)
        ax2 = self.metrics_figure.add_subplot(132)
        ax3 = self.metrics_figure.add_subplot(133)

        # Sample data (in practice, get from actual metrics)
        times = np.linspace(0, 60, 100)  # Last 60 seconds
        confidence_data = 0.7 + 0.2 * np.sin(times / 10) + np.random.normal(0, 0.05, 100)
        response_times = 2.0 + 0.5 * np.sin(times / 15) + np.random.normal(0, 0.1, 100)
        safety_scores = 0.8 + 0.15 * np.cos(times / 8) + np.random.normal(0, 0.03, 100)

        # Plot confidence over time
        ax1.plot(times, confidence_data, "b-", linewidth=2)
        ax1.axhline(y=0.6, color="r", linestyle="--", alpha=0.7, label="Threshold")
        ax1.set_title("LLM Confidence")
        ax1.set_ylabel("Confidence")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot response times
        ax2.plot(times, response_times, "g-", linewidth=2)
        ax2.axhline(y=5.0, color="r", linestyle="--", alpha=0.7, label="Limit")
        ax2.set_title("Response Times")
        ax2.set_ylabel("Time (s)")
        ax2.set_ylim(0, 6)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot safety scores
        ax3.plot(times, safety_scores, "m-", linewidth=2)
        ax3.axhline(y=0.4, color="r", linestyle="--", alpha=0.7, label="Minimum")
        ax3.set_title("Safety Scores")
        ax3.set_ylabel("Score")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        self.metrics_figure.tight_layout()
        self.metrics_canvas.draw()

    def _monitoring_loop(self):
        """Main monitoring loop"""

        while self.monitoring_active:
            try:
                # Update alerts display
                recent_alerts = self.safety_monitor.get_recent_alerts()

                # Update UI in main thread
                self.root.after(0, self._update_alerts_display, recent_alerts)

                # Update metrics plot
                self.root.after(0, self._update_metrics_plot)

                # Update system status
                self.root.after(0, self._update_status_display)

                time.sleep(1)  # Update every second

            except Exception:
                logging.exception("Monitoring loop error")
                time.sleep(5)

    def _update_alerts_display(self, alerts: list[dict]):
        """Update alerts listbox"""

        self.alerts_listbox.delete(0, tk.END)

        for alert in alerts[-20:]:  # Show last 20 alerts
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(alert["timestamp"]))
            alert_text = f"{timestamp_str} - {alert['alert']}"
            self.alerts_listbox.insert(tk.END, alert_text)

        # Scroll to bottom
        self.alerts_listbox.see(tk.END)

    def _update_status_display(self):
        """Update system status display"""

        # System information
        status_info = f"""System Status: OPERATIONAL
LLM Models: Active
Safety Monitor: Running
Last Update: {time.strftime('%H:%M:%S')}
Active Conflicts: {len(self.active_conflicts)}
Total Overrides: {len(self.override_history)}
"""

        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, status_info)

        # Performance metrics (mock data)
        perf_info = f"""Performance Metrics:
Average Response Time: 2.3s
LLM Availability: 99.5%
Confidence Average: 0.75
Safety Score Average: 0.82
Hallucination Rate: 15.2%
Override Rate: 8.7%
Critical Alerts: {len([a for a in self.safety_monitor.get_recent_alerts() if 'critical' in a['alert'].lower()])}
"""

        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, perf_info)

    def _export_report(self):
        """Export system report"""

        report_data = {
            "timestamp": time.time(),
            "active_conflicts": len(self.active_conflicts),
            "override_history": [
                {
                    "conflict_id": o.conflict_id,
                    "action": o.override_action,
                    "reason": o.reason,
                    "timestamp": o.timestamp,
                } for o in self.override_history
            ],
            "recent_alerts": self.safety_monitor.get_recent_alerts(),
            "system_status": "operational",
        }

        filename = f"atc_oversight_report_{int(time.time())}.json"
        try:
            with open(filename, "w") as f:
                json.dump(report_data, f, indent=2)
            messagebox.showinfo("Export Complete", f"Report exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report: {e}")

    def _reset_alerts(self):
        """Reset safety alerts"""

        if messagebox.askyesno("Reset Alerts", "Are you sure you want to reset all alerts?"):
            self.safety_monitor.alerts.clear()
            self.alerts_listbox.delete(0, tk.END)
            messagebox.showinfo("Alerts Reset", "All alerts have been cleared.")

    def _emergency_stop(self):
        """Emergency stop procedure"""

        if messagebox.askyesno("Emergency Stop",
                              "This will halt all AI operations. Continue?",
                              icon="warning"):
            self.monitoring_active = False
            logging.critical("EMERGENCY STOP activated by controller")
            messagebox.showwarning("Emergency Stop", "AI operations halted. Manual control only.")

    def run(self):
        """Start the interface"""

        try:
            self.root.mainloop()
        finally:
            self.monitoring_active = False

# Example integration function
def create_test_interface():
    """Create test interface with sample data"""

    interface = ControllerInterface()

    # Add sample conflict
    sample_conflict = ConflictDisplay(
        conflict_id="CONF001",
        aircraft_ids=["AC001", "AC002"],
        time_to_conflict=120.0,
        current_separation=6.2,
        severity="moderate",
        llm_recommendation={
            "action": "turn left 10 degrees",
            "type": "heading",
            "safety_score": 0.75,
            "reasoning": "Minimal deviation maintains separation",
        },
        baseline_recommendation={
            "action": "climb 1000 ft",
            "type": "altitude",
            "safety_score": 0.85,
        },
        confidence_level=ConfidenceLevel.MODERATE,
        safety_flags=[],
    )

    interface.add_conflict(sample_conflict)

    return interface

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create and run test interface
    interface = create_test_interface()
    interface.run()
