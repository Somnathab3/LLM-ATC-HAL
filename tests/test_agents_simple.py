# tests/test_agents_simple.py
"""
Simplified test suite for embodied agent system without GUI dependencies
"""

import unittest
import time
from unittest.mock import patch, MagicMock

# Import agent components
from llm_atc.agents.planner import Planner, ConflictAssessment, ActionPlan, PlanType
from llm_atc.agents.executor import Executor, ExecutionResult, ExecutionStatus
from llm_atc.agents.verifier import Verifier, VerificationResult, VerificationStatus
from llm_atc.agents.scratchpad import Scratchpad, ReasoningStep, StepType
from llm_atc.tools import bluesky_tools


class TestEmbodiedAgentsCore(unittest.TestCase):
    """Test cases for core embodied agent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.planner = Planner()
        self.executor = Executor()
        self.verifier = Verifier()
        self.scratchpad = Scratchpad()
    
    def test_planner_conflict_assessment(self):
        """Test planner conflict assessment functionality"""
        # Mock aircraft data with conflict
        aircraft_info = {
            'aircraft': {
                'AAL123': {
                    'lat': 52.3676,
                    'lon': 4.9041,
                    'alt': 35000,
                    'hdg': 90,
                    'spd': 450
                },
                'DLH456': {
                    'lat': 52.3676,
                    'lon': 4.9041,
                    'alt': 35000,
                    'hdg': 270,
                    'spd': 460
                }
            }
        }
        
        # Test conflict assessment
        assessment = self.planner.assess_conflict(aircraft_info)
        
        # Should detect conflict
        self.assertIsNotNone(assessment)
        self.assertIsInstance(assessment, ConflictAssessment)
        self.assertEqual(len(assessment.aircraft_involved), 2)
        self.assertIn('AAL123', assessment.aircraft_involved)
        self.assertIn('DLH456', assessment.aircraft_involved)
        print(f"✓ Conflict detected: {assessment.conflict_id} between {assessment.aircraft_involved}")
    
    def test_executor_command_execution(self):
        """Test executor command execution"""
        # Create mock action plan
        plan = ActionPlan(
            plan_id="test_plan_001",
            conflict_id="test_conflict_001",
            plan_type=PlanType.ALTITUDE_CHANGE,
            target_aircraft=["AAL123"],
            commands=["ALT AAL123 FL350"],
            priority=7,
            expected_outcome={'resolution_time': 180},
            confidence=0.85,
            reasoning="Test altitude change",
            created_at=time.time()
        )
        
        # Execute plan
        result = self.executor.send_plan(plan)
        
        # Verify execution result
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.plan_id, plan.plan_id)
        self.assertEqual(len(result.commands_sent), 1)
        self.assertEqual(result.commands_sent[0], "ALT AAL123 FL350")
        self.assertTrue(result.success_rate > 0)
        print(f"✓ Plan executed: {result.execution_id} with {result.success_rate:.1%} success rate")
    
    def test_verifier_success_verification(self):
        """Test verifier with successful execution"""
        # Create successful execution result
        execution_result = ExecutionResult(
            execution_id="test_exec_001",
            plan_id="test_plan_001",
            status=ExecutionStatus.COMPLETED,
            commands_sent=["ALT AAL123 FL350"],
            responses=[{'success': True, 'command': 'ALT AAL123 FL350', 'timestamp': time.time()}],
            success_rate=1.0,
            execution_time=2.5,
            error_messages=[],
            created_at=time.time()
        )
        
        # Verify execution
        passed = self.verifier.check(execution_result)
        
        # Should pass verification
        self.assertTrue(passed)
        print(f"✓ Verification passed for execution: {execution_result.execution_id}")
    
    def test_scratchpad_logging(self):
        """Test scratchpad step logging"""
        # Log a test step
        step_data = {
            'type': 'assessment',
            'description': 'Test conflict assessment',
            'confidence': 0.8,
            'reasoning': 'Test reasoning',
            'input_data': {'test': 'data'},
            'output_data': {'result': 'success'}
        }
        
        step_id = self.scratchpad.log_step(step_data)
        
        # Verify step was logged
        self.assertIsNotNone(step_id)
        self.assertTrue(len(step_id) > 0)
        
        # Get history and verify
        history = self.scratchpad.get_history()
        self.assertEqual(history['total_steps'], 1)
        self.assertEqual(len(history['steps']), 1)
        self.assertEqual(history['steps'][0]['description'], 'Test conflict assessment')
        print(f"✓ Step logged: {step_id}")


class TestToolStubs(unittest.TestCase):
    """Test BlueSky tool stubs"""
    
    def test_get_all_aircraft_info(self):
        """Test GetAllAircraftInfo tool stub"""
        result = bluesky_tools.GetAllAircraftInfo()
        
        self.assertIsInstance(result, dict)
        self.assertIn('aircraft', result)
        self.assertIn('total_aircraft', result)
        self.assertIn('timestamp', result)
        print(f"✓ Aircraft info: {result['total_aircraft']} aircraft")
    
    def test_send_command(self):
        """Test SendCommand tool stub"""
        command = "ALT TEST123 FL350"
        result = bluesky_tools.SendCommand(command)
        
        self.assertIsInstance(result, dict)
        self.assertIn('command', result)
        self.assertIn('success', result)
        self.assertIn('timestamp', result)
        self.assertEqual(result['command'], command)
        self.assertTrue(result['success'])
        print(f"✓ Command sent: {command} -> {result['status']}")
    
    def test_continue_monitoring(self):
        """Test ContinueMonitoring tool stub"""
        result = bluesky_tools.ContinueMonitoring()
        
        self.assertIsInstance(result, dict)
        self.assertIn('action', result)
        self.assertIn('status', result)
        self.assertEqual(result['action'], 'continue_monitoring')
        print(f"✓ Monitoring continued: {result['status']}")


class TestPlanningLoopCore(unittest.TestCase):
    """Test core planning loop without GUI"""
    
    def setUp(self):
        """Set up planning loop test"""
        self.planner = Planner()
        self.executor = Executor()
        self.verifier = Verifier()
        self.scratchpad = Scratchpad()
        
        # Setup mock data
        self.iteration_count = 0
        self.original_get_aircraft = bluesky_tools.GetAllAircraftInfo
        
        # Mock function that simulates conflict resolution
        def mock_get_aircraft_info():
            self.iteration_count += 1
            
            if self.iteration_count <= 2:
                # Return conflicted aircraft for first few iterations
                return {
                    'aircraft': {
                        'TEST001': {'lat': 52.3676, 'lon': 4.9041, 'alt': 35000, 'hdg': 90, 'spd': 450},
                        'TEST002': {'lat': 52.3676, 'lon': 4.9041, 'alt': 35000, 'hdg': 270, 'spd': 460}
                    },
                    'total_aircraft': 2,
                    'timestamp': time.time()
                }
            else:
                # Return resolved aircraft (no conflicts)
                return {
                    'aircraft': {
                        'TEST001': {'lat': 52.3676, 'lon': 4.9041, 'alt': 37000, 'hdg': 90, 'spd': 450},
                        'TEST002': {'lat': 53.3676, 'lon': 5.9041, 'alt': 35000, 'hdg': 270, 'spd': 460}
                    },
                    'total_aircraft': 2,
                    'timestamp': time.time()
                }
        
        # Apply mock
        bluesky_tools.GetAllAircraftInfo = mock_get_aircraft_info
        bluesky_tools.TOOL_REGISTRY['GetAllAircraftInfo'] = mock_get_aircraft_info
    
    def tearDown(self):
        """Restore original functions"""
        bluesky_tools.GetAllAircraftInfo = self.original_get_aircraft
        bluesky_tools.TOOL_REGISTRY['GetAllAircraftInfo'] = self.original_get_aircraft
    
    def test_planning_loop_logic(self):
        """Test core planning loop logic without GUI"""
        max_iterations = 5
        planning_active = True
        iteration_count = 0
        
        print("\n--- Starting Planning Loop Test ---")
        
        # Start session
        session_id = self.scratchpad.start_new_session()
        print(f"✓ Session started: {session_id}")
        
        # Initial monitoring
        initial_data = bluesky_tools.GetAllAircraftInfo()
        self.scratchpad.log_monitoring_step(initial_data)
        print(f"✓ Initial monitoring logged")
        
        while planning_active and iteration_count < max_iterations:
            iteration_count += 1
            print(f"\n--- Iteration {iteration_count} ---")
            
            # Step 1: Get aircraft information
            info = bluesky_tools.GetAllAircraftInfo()
            print(f"✓ Aircraft info retrieved: {info['total_aircraft']} aircraft")
            
            # Step 2: Assess conflicts
            assessment = self.planner.assess_conflict(info)
            
            if assessment is None:
                print("✓ No conflicts detected, continuing monitoring")
                self.scratchpad.log_monitoring_step(info)
                break  # Exit loop when no conflicts
            
            print(f"✓ Conflict assessed: {assessment.conflict_id}")
            self.scratchpad.log_assessment_step(assessment)
            
            # Step 3: Generate action plan
            plan = self.planner.generate_action_plan(assessment)
            
            if plan is None:
                print("✗ Failed to generate action plan")
                self.scratchpad.log_error_step("Failed to generate action plan")
                break
            
            print(f"✓ Plan generated: {plan.plan_id} with {len(plan.commands)} commands")
            self.scratchpad.log_planning_step(plan)
            
            # Step 4: Execute plan
            exec_result = self.executor.send_plan(plan)
            print(f"✓ Plan executed: {exec_result.execution_id} ({exec_result.success_rate:.1%} success)")
            self.scratchpad.log_execution_step(exec_result)
            
            # Step 5: Verify execution
            verification_passed = self.verifier.check(exec_result, timeout_seconds=5)
            
            verification_results = self.verifier.get_verification_history()
            if verification_results:
                latest_verification = verification_results[-1]
                self.scratchpad.log_verification_step(latest_verification)
                print(f"✓ Verification: {'PASSED' if verification_passed else 'FAILED'}")
            
            # Step 6: Check if we should continue
            if not verification_passed:
                print("✗ Verification failed, stopping planning loop")
                self.scratchpad.log_error_step("Verification failed, stopping planning loop")
                break
        
        # Complete session
        session_summary = self.scratchpad.complete_session(
            success=True,
            final_status="completed" if iteration_count < max_iterations else "max_iterations_reached"
        )
        
        print(f"\n--- Planning Loop Completed ---")
        print(f"✓ Session completed: {session_summary.session_id}")
        print(f"✓ Total iterations: {iteration_count}")
        print(f"✓ Total steps logged: {session_summary.total_steps}")
        print(f"✓ Final status: {session_summary.final_status}")
        
        # Verify results
        history = self.scratchpad.get_history()
        self.assertGreater(history['total_steps'], 0)
        self.assertTrue(session_summary.success)
        
        # Print step summary
        steps = history['steps']
        step_types = {}
        for step in steps:
            step_type = step['step_type']
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        print(f"✓ Step types logged: {step_types}")
        
        return {
            'status': 'resolved',
            'history': history,
            'session_summary': session_summary,
            'iterations': iteration_count
        }


if __name__ == '__main__':
    # Run tests with verbose output
    print("="*60)
    print("EMBODIED AGENT SYSTEM TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbodiedAgentsCore))
    suite.addTests(loader.loadTestsFromTestCase(TestToolStubs))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanningLoopCore))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED - EMBODIED AGENT SYSTEM WORKING")
    else:
        print("✗ SOME TESTS FAILED")
        print(f"Errors: {len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
    print("="*60)
