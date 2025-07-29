# tests/test_agents.py
"""
Test suite for embodied agent system
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import agent components
from agents.planner import Planner, ConflictAssessment, ActionPlan, PlanType
from agents.executor import Executor, ExecutionResult, ExecutionStatus
from agents.verifier import Verifier, VerificationResult, VerificationStatus
from agents.scratchpad import Scratchpad, ReasoningStep, StepType
from agents.controller_interface import ControllerInterface
from tools import bluesky_tools


class TestEmbodiedAgents(unittest.TestCase):
    """Test cases for embodied agent system"""
    
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
    
    def test_planner_no_conflict(self):
        """Test planner with no conflicts"""
        # Mock aircraft data with no conflicts
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
                    'lat': 53.3676,  # Different position, no conflict
                    'lon': 5.9041,
                    'alt': 37000,
                    'hdg': 270,
                    'spd': 460
                }
            }
        }
        
        # Test no conflict assessment
        assessment = self.planner.assess_conflict(aircraft_info)
        
        # Should not detect conflict
        self.assertIsNone(assessment)
    
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
    
    def test_verifier_failure_verification(self):
        """Test verifier with failed execution"""
        # Create failed execution result
        execution_result = ExecutionResult(
            execution_id="test_exec_002",
            plan_id="test_plan_002",
            status=ExecutionStatus.FAILED,
            commands_sent=["ALT AAL123 FL350"],
            responses=[{'success': False, 'command': 'ALT AAL123 FL350', 'error': 'Unknown aircraft'}],
            success_rate=0.0,
            execution_time=1.0,
            error_messages=["Unknown aircraft"],
            created_at=time.time()
        )
        
        # Verify execution
        passed = self.verifier.check(execution_result)
        
        # Should fail verification
        self.assertFalse(passed)
    
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
    
    def test_scratchpad_session_completion(self):
        """Test scratchpad session completion"""
        # Log some steps
        self.scratchpad.log_step({
            'type': 'assessment',
            'description': 'Step 1',
            'confidence': 0.8,
            'reasoning': 'Test'
        })
        
        self.scratchpad.log_step({
            'type': 'execution',
            'description': 'Step 2',
            'confidence': 0.9,
            'reasoning': 'Test'
        })
        
        # Complete session
        summary = self.scratchpad.complete_session(success=True, final_status="test_completed")
        
        # Verify summary
        self.assertEqual(summary.total_steps, 2)
        self.assertTrue(summary.success)
        self.assertEqual(summary.final_status, "test_completed")


class TestControllerInterfaceWithMocks(unittest.TestCase):
    """Test controller interface with mocked tools"""
    
    def setUp(self):
        """Set up test fixtures with mocked tools"""
        self.original_tools = {}
        self._setup_tool_mocks()
    
    def _setup_tool_mocks(self):
        """Set up mocked tool functions"""
        # Mock GetAllAircraftInfo
        def mock_get_all_aircraft_info():
            return {
                'aircraft': {
                    'TEST001': {
                        'lat': 52.3676,
                        'lon': 4.9041,
                        'alt': 35000,
                        'hdg': 90,
                        'spd': 450
                    },
                    'TEST002': {
                        'lat': 52.3676,
                        'lon': 4.9041,
                        'alt': 35000,
                        'hdg': 270,
                        'spd': 460
                    }
                },
                'total_aircraft': 2,
                'timestamp': time.time()
            }
        
        # Mock SendCommand
        def mock_send_command(command: str):
            return {
                'command': command,
                'success': True,
                'response': f'{command.split()[0]} command acknowledged',
                'timestamp': time.time()
            }
        
        # Patch the tools
        self.original_tools['GetAllAircraftInfo'] = bluesky_tools.GetAllAircraftInfo
        self.original_tools['SendCommand'] = bluesky_tools.SendCommand
        
        bluesky_tools.GetAllAircraftInfo = mock_get_all_aircraft_info
        bluesky_tools.SendCommand = mock_send_command
        bluesky_tools.TOOL_REGISTRY['GetAllAircraftInfo'] = mock_get_all_aircraft_info
        bluesky_tools.TOOL_REGISTRY['SendCommand'] = mock_send_command
    
    def tearDown(self):
        """Restore original tools"""
        for tool_name, original_func in self.original_tools.items():
            setattr(bluesky_tools, tool_name, original_func)
            bluesky_tools.TOOL_REGISTRY[tool_name] = original_func
    
    @patch('tkinter.Tk')  # Mock tkinter to avoid GUI in tests
    def test_controller_interface_initialization(self, mock_tk):
        """Test controller interface initialization"""
        # Mock tkinter components
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        
        # Initialize controller interface
        controller = ControllerInterface()
        
        # Verify components are initialized
        self.assertIsNotNone(controller.planner)
        self.assertIsNotNone(controller.executor)
        self.assertIsNotNone(controller.verifier)
        self.assertIsNotNone(controller.scratchpad)
    
    @patch('tkinter.Tk')
    def test_planning_loop_execution(self, mock_tk):
        """Test planning loop execution with mocked tools"""
        # Mock tkinter components
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        
        # Initialize controller interface
        controller = ControllerInterface()
        
        # Override max iterations for testing
        controller.max_planning_iterations = 2
        
        # Run planning loop
        result = controller.start_planning_loop()
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('history', result)
        
        # Verify history contains steps
        history = result['history']
        self.assertGreater(history['total_steps'], 0)
        
        # Verify session completed
        self.assertIsNotNone(result.get('session_summary'))
    
    @patch('tkinter.Tk')
    def test_agent_status_reporting(self, mock_tk):
        """Test agent status reporting"""
        # Mock tkinter components
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        
        # Initialize controller interface
        controller = ControllerInterface()
        
        # Get agent status
        status = controller.get_agent_status()
        
        # Verify status structure
        self.assertIn('planner', status)
        self.assertIn('executor', status)
        self.assertIn('verifier', status)
        self.assertIn('scratchpad', status)
        self.assertIn('planning_active', status)
        
        # Verify boolean flag
        self.assertIsInstance(status['planning_active'], bool)


class TestToolStubs(unittest.TestCase):
    """Test BlueSky tool stubs"""
    
    def test_get_all_aircraft_info(self):
        """Test GetAllAircraftInfo tool stub"""
        result = bluesky_tools.GetAllAircraftInfo()
        
        self.assertIsInstance(result, dict)
        self.assertIn('aircraft', result)
        self.assertIn('total_aircraft', result)
        self.assertIn('timestamp', result)
    
    def test_get_conflict_info(self):
        """Test GetConflictInfo tool stub"""
        result = bluesky_tools.GetConflictInfo()
        
        self.assertIsInstance(result, dict)
        self.assertIn('conflicts', result)
        self.assertIn('total_conflicts', result)
        self.assertIn('timestamp', result)
    
    def test_send_command(self):
        """Test SendCommand tool stub"""
        command = "ALT TEST123 FL350"
        result = bluesky_tools.SendCommand(command)
        
        self.assertIsInstance(result, dict)
        self.assertIn('command', result)
        self.assertIn('success', result)
        self.assertIn('timestamp', result)
        self.assertEqual(result['command'], command)
    
    def test_continue_monitoring(self):
        """Test ContinueMonitoring tool stub"""
        result = bluesky_tools.ContinueMonitoring()
        
        self.assertIsInstance(result, dict)
        self.assertIn('action', result)
        self.assertIn('status', result)
        self.assertEqual(result['action'], 'continue_monitoring')
    
    def test_search_experience_library(self):
        """Test SearchExperienceLibrary tool stub"""
        result = bluesky_tools.SearchExperienceLibrary("altitude_conflict", 0.8)
        
        self.assertIsInstance(result, dict)
        self.assertIn('matches', result)
        self.assertIn('total_matches', result)
        self.assertIn('recommendations', result)
    
    def test_tool_registry(self):
        """Test tool registry functionality"""
        # Test execute_tool function
        result = bluesky_tools.execute_tool('GetAllAircraftInfo')
        
        self.assertIsInstance(result, dict)
        self.assertIn('tool_name', result)
        self.assertIn('success', result)
        self.assertIn('result', result)
        self.assertTrue(result['success'])
        
        # Test get_available_tools
        available_tools = bluesky_tools.get_available_tools()
        self.assertIsInstance(available_tools, list)
        self.assertIn('GetAllAircraftInfo', available_tools)
        self.assertIn('SendCommand', available_tools)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete embodied agent system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self._setup_tool_mocks()
    
    def _setup_tool_mocks(self):
        """Set up mocked tools for integration testing"""
        # Store original functions
        self.original_get_aircraft = bluesky_tools.GetAllAircraftInfo
        self.original_send_command = bluesky_tools.SendCommand
        
        # Counter for monitoring iterations
        self.iteration_count = 0
        
        # Mock GetAllAircraftInfo to simulate conflict resolution
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
        
        # Mock SendCommand
        def mock_send_command(command: str):
            return {
                'command': command,
                'success': True,
                'response': 'Command executed successfully',
                'timestamp': time.time()
            }
        
        # Apply mocks
        bluesky_tools.GetAllAircraftInfo = mock_get_aircraft_info
        bluesky_tools.SendCommand = mock_send_command
        bluesky_tools.TOOL_REGISTRY['GetAllAircraftInfo'] = mock_get_aircraft_info
        bluesky_tools.TOOL_REGISTRY['SendCommand'] = mock_send_command
    
    def tearDown(self):
        """Restore original functions"""
        bluesky_tools.GetAllAircraftInfo = self.original_get_aircraft
        bluesky_tools.SendCommand = self.original_send_command
        bluesky_tools.TOOL_REGISTRY['GetAllAircraftInfo'] = self.original_get_aircraft
        bluesky_tools.TOOL_REGISTRY['SendCommand'] = self.original_send_command
    
    @patch('tkinter.Tk')
    def test_complete_planning_loop_integration(self, mock_tk):
        """Test complete planning loop integration"""
        # Mock tkinter
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        
        # Initialize controller with limited iterations for testing
        controller = ControllerInterface()
        controller.max_planning_iterations = 5
        
        # Run planning loop
        result = controller.start_planning_loop()
        
        # Verify successful completion
        self.assertEqual(result['status'], 'resolved')
        self.assertIn('history', result)
        
        # Verify history has multiple steps
        history = result['history']
        self.assertGreater(history['total_steps'], 0)
        
        # Verify different step types were logged
        steps = history['steps']
        step_types = [step['step_type'] for step in steps]
        
        # Should have monitoring steps at minimum
        self.assertIn('monitoring', step_types)
        
        # If conflicts were detected, should have assessment and planning steps
        if 'assessment' in step_types:
            self.assertIn('planning', step_types)
            self.assertIn('execution', step_types)
            self.assertIn('verification', step_types)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
