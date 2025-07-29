# agents/verifier.py
"""
Verifier Agent - Checks execution results and validates safety
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .executor import ExecutionResult, ExecutionStatus


class VerificationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class VerificationResult:
    """Result of verification check"""
    verification_id: str
    execution_id: str
    status: VerificationStatus
    checks_performed: List[str]
    passed_checks: List[str]
    failed_checks: List[str]
    warnings: List[str]
    safety_score: float
    confidence: float
    verification_time: float
    created_at: float


class Verifier:
    """
    Verifier agent responsible for checking execution results and validating safety
    """
    
    def __init__(self, safety_thresholds: Optional[Dict[str, float]] = None):
        self.logger = logging.getLogger(__name__)
        self.verification_history: List[VerificationResult] = []
        
        # Default safety thresholds
        self.safety_thresholds = safety_thresholds or {
            'minimum_separation_horizontal': 5.0,  # nautical miles
            'minimum_separation_vertical': 1000.0,  # feet
            'maximum_response_time': 5.0,  # seconds
            'minimum_success_rate': 0.8,  # 80%
            'minimum_safety_score': 0.6,  # 60%
            'maximum_execution_time': 30.0  # seconds
        }
    
    def check(self, execution_result: ExecutionResult, timeout_seconds: float = 5.0) -> bool:
        """
        Perform verification check on execution result
        
        Args:
            execution_result: ExecutionResult to verify
            timeout_seconds: Maximum time to wait for verification
            
        Returns:
            True if verification passes, False otherwise
        """
        try:
            self.logger.info(f"Starting verification for execution {execution_result.execution_id}")
            
            start_time = time.time()
            verification_id = f"verify_{int(time.time() * 1000)}"
            
            # Initialize verification result
            verification = VerificationResult(
                verification_id=verification_id,
                execution_id=execution_result.execution_id,
                status=VerificationStatus.PENDING,
                checks_performed=[],
                passed_checks=[],
                failed_checks=[],
                warnings=[],
                safety_score=0.0,
                confidence=0.0,
                verification_time=0.0,
                created_at=start_time
            )
            
            # Perform verification checks
            self._check_execution_status(execution_result, verification)
            self._check_execution_timing(execution_result, verification)
            self._check_command_success_rate(execution_result, verification)
            self._check_safety_compliance(execution_result, verification)
            self._check_response_validity(execution_result, verification)
            
            # Calculate overall safety score
            verification.safety_score = self._calculate_safety_score(verification)
            verification.confidence = self._calculate_confidence(verification)
            verification.verification_time = time.time() - start_time
            
            # Determine final verification status
            if verification.failed_checks:
                verification.status = VerificationStatus.FAILED
                self.logger.warning(f"Verification failed: {verification.failed_checks}")
            elif verification.warnings:
                verification.status = VerificationStatus.WARNING
                self.logger.info(f"Verification passed with warnings: {verification.warnings}")
            else:
                verification.status = VerificationStatus.PASSED
                self.logger.info(f"Verification passed successfully")
            
            # Store verification result
            self.verification_history.append(verification)
            
            # Return True if passed or warning (warnings don't block execution)
            return verification.status in [VerificationStatus.PASSED, VerificationStatus.WARNING]
            
        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return False
    
    def _check_execution_status(self, execution: ExecutionResult, verification: VerificationResult):
        """Check if execution completed successfully"""
        check_name = "execution_status"
        verification.checks_performed.append(check_name)
        
        if execution.status == ExecutionStatus.COMPLETED:
            verification.passed_checks.append(check_name)
        elif execution.status == ExecutionStatus.FAILED:
            verification.failed_checks.append(f"{check_name}: Execution failed")
        elif execution.status == ExecutionStatus.CANCELLED:
            verification.failed_checks.append(f"{check_name}: Execution was cancelled")
        else:
            verification.warnings.append(f"{check_name}: Execution status is {execution.status.value}")
    
    def _check_execution_timing(self, execution: ExecutionResult, verification: VerificationResult):
        """Check execution timing constraints"""
        check_name = "execution_timing"
        verification.checks_performed.append(check_name)
        
        max_time = self.safety_thresholds['maximum_execution_time']
        
        if execution.execution_time <= max_time:
            verification.passed_checks.append(check_name)
        elif execution.execution_time <= max_time * 1.5:  # 50% tolerance for warnings
            verification.warnings.append(f"{check_name}: Execution took {execution.execution_time:.2f}s (limit: {max_time}s)")
        else:
            verification.failed_checks.append(f"{check_name}: Execution too slow ({execution.execution_time:.2f}s > {max_time}s)")
    
    def _check_command_success_rate(self, execution: ExecutionResult, verification: VerificationResult):
        """Check command success rate"""
        check_name = "command_success_rate"
        verification.checks_performed.append(check_name)
        
        min_success_rate = self.safety_thresholds['minimum_success_rate']
        
        if execution.success_rate >= min_success_rate:
            verification.passed_checks.append(check_name)
        elif execution.success_rate >= min_success_rate * 0.8:  # 80% of minimum for warnings
            verification.warnings.append(f"{check_name}: Low success rate ({execution.success_rate:.2f} < {min_success_rate})")
        else:
            verification.failed_checks.append(f"{check_name}: Success rate too low ({execution.success_rate:.2f} < {min_success_rate})")
    
    def _check_safety_compliance(self, execution: ExecutionResult, verification: VerificationResult):
        """Check safety compliance of executed commands"""
        check_name = "safety_compliance"
        verification.checks_performed.append(check_name)
        
        # Check for safety-critical command patterns
        safety_violations = []
        
        for command in execution.commands_sent:
            if self._is_unsafe_command(command):
                safety_violations.append(command)
        
        if not safety_violations:
            verification.passed_checks.append(check_name)
        else:
            verification.failed_checks.append(f"{check_name}: Unsafe commands detected: {safety_violations}")
    
    def _check_response_validity(self, execution: ExecutionResult, verification: VerificationResult):
        """Check validity of command responses"""
        check_name = "response_validity"
        verification.checks_performed.append(check_name)
        
        invalid_responses = []
        
        for response in execution.responses:
            if not self._is_valid_response(response):
                invalid_responses.append(response.get('command', 'unknown'))
        
        if not invalid_responses:
            verification.passed_checks.append(check_name)
        elif len(invalid_responses) <= len(execution.responses) * 0.2:  # 20% tolerance
            verification.warnings.append(f"{check_name}: Some invalid responses: {invalid_responses}")
        else:
            verification.failed_checks.append(f"{check_name}: Too many invalid responses: {invalid_responses}")
    
    def _is_unsafe_command(self, command: str) -> bool:
        """Check if a command is potentially unsafe"""
        command_upper = command.upper()
        
        # Check for extreme altitude changes
        if 'ALT' in command_upper:
            parts = command.split()
            if len(parts) >= 3:
                try:
                    # Extract altitude (simplified)
                    alt_str = parts[2].replace('FL', '').replace('ft', '')
                    altitude = float(alt_str)
                    
                    # Flag very low or very high altitudes as potentially unsafe
                    if altitude < 50 or altitude > 600:  # FL050 to FL600
                        return True
                except (ValueError, IndexError):
                    pass
        
        # Check for extreme heading changes
        if 'HDG' in command_upper:
            # Could add logic to check for extreme heading changes
            pass
        
        # Check for extreme speed changes
        if 'SPD' in command_upper:
            parts = command.split()
            if len(parts) >= 3:
                try:
                    speed = float(parts[2])
                    # Flag very low or very high speeds as potentially unsafe
                    if speed < 100 or speed > 500:  # knots
                        return True
                except (ValueError, IndexError):
                    pass
        
        return False
    
    def _is_valid_response(self, response: Dict[str, Any]) -> bool:
        """Check if a command response is valid"""
        # Check for required fields
        if 'command' not in response or 'timestamp' not in response:
            return False
        
        # Check for success indication or error details
        if not response.get('success', False) and not response.get('error'):
            return False
        
        return True
    
    def _calculate_safety_score(self, verification: VerificationResult) -> float:
        """Calculate overall safety score based on verification results"""
        if not verification.checks_performed:
            return 0.0
        
        total_checks = len(verification.checks_performed)
        passed_checks = len(verification.passed_checks)
        warning_checks = len(verification.warnings)
        failed_checks = len(verification.failed_checks)
        
        # Calculate weighted score
        score = (passed_checks + warning_checks * 0.7) / total_checks
        
        # Apply penalty for failed checks
        if failed_checks > 0:
            penalty = min(failed_checks * 0.2, 0.8)  # Max 80% penalty
            score = max(score - penalty, 0.0)
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, verification: VerificationResult) -> float:
        """Calculate confidence in verification results"""
        if not verification.checks_performed:
            return 0.0
        
        # Higher confidence with more checks and fewer failures
        base_confidence = 0.8
        
        # Adjust based on check results
        total_checks = len(verification.checks_performed)
        failed_checks = len(verification.failed_checks)
        
        if failed_checks == 0:
            confidence = base_confidence + 0.15
        else:
            confidence = base_confidence - (failed_checks / total_checks) * 0.3
        
        return max(min(confidence, 1.0), 0.1)
    
    def get_verification_history(self) -> List[VerificationResult]:
        """Get history of all verification results"""
        return self.verification_history.copy()
    
    def get_verification_metrics(self) -> Dict[str, Any]:
        """Get overall verification performance metrics"""
        if not self.verification_history:
            return {
                'total_verifications': 0,
                'pass_rate': 0.0,
                'average_safety_score': 0.0,
                'average_confidence': 0.0
            }
        
        total_verifications = len(self.verification_history)
        passed_verifications = len([v for v in self.verification_history 
                                  if v.status in [VerificationStatus.PASSED, VerificationStatus.WARNING]])
        avg_safety_score = sum(v.safety_score for v in self.verification_history) / total_verifications
        avg_confidence = sum(v.confidence for v in self.verification_history) / total_verifications
        
        return {
            'total_verifications': total_verifications,
            'pass_rate': passed_verifications / total_verifications,
            'average_safety_score': avg_safety_score,
            'average_confidence': avg_confidence
        }
    
    def update_safety_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update safety thresholds for verification"""
        self.safety_thresholds.update(new_thresholds)
        self.logger.info(f"Safety thresholds updated: {new_thresholds}")
