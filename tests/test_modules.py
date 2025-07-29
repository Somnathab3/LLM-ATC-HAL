# tests/test_modules.py
import sys
import os
import unittest
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestModuleImports(unittest.TestCase):
    """Test that all modules can be imported without exceptions."""
    
    def test_conflict_solver_import(self):
        """Test ConflictSolver can be imported and instantiated."""
        try:
            from solver.conflict_solver import ConflictSolver
            solver = ConflictSolver()
            self.assertIsNotNone(solver)
        except ImportError as e:
            self.fail(f"Failed to import ConflictSolver: {e}")
    
    def test_conflict_solver_functionality(self):
        """Test ConflictSolver basic functionality."""
        from solver.conflict_solver import ConflictSolver
        solver = ConflictSolver()
        
        # Test solve method
        test_conflict = {'id1': 'AC001', 'id2': 'AC002', 'distance': 4.5, 'time': 120}
        candidates = solver.solve(test_conflict)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        # Test score_best method
        best = solver.score_best(candidates)
        self.assertIsNotNone(best)
        self.assertIn('action', best)
    
    def test_llm_client_import(self):
        """Test LLMClient can be imported (may fail if ollama not available)."""
        try:
            from llm_interface.llm_client import LLMClient
            # Don't instantiate since ollama might not be available
            self.assertTrue(True)  # Just test import
        except ImportError as e:
            self.skipTest(f"LLMClient import failed (expected if ollama not available): {e}")
    
    def test_filter_sort_import(self):
        """Test filter_sort module can be imported."""
        try:
            from llm_interface.filter_sort import select_best_solution
            self.assertTrue(callable(select_best_solution))
        except ImportError as e:
            self.fail(f"Failed to import select_best_solution: {e}")
    
    def test_scenarios_import(self):
        """Test scenarios module can be imported."""
        try:
            from bluesky_sim.scenarios import generate_all_scenarios
            self.assertTrue(callable(generate_all_scenarios))
        except ImportError as e:
            self.fail(f"Failed to import scenarios: {e}")
    
    def test_analysis_import(self):
        """Test analysis module can be imported."""
        try:
            from analysis.metrics import compute_metrics, create_empty_metrics
            self.assertTrue(callable(compute_metrics))
            self.assertTrue(callable(create_empty_metrics))
        except ImportError as e:
            self.fail(f"Failed to import analysis metrics: {e}")

class TestConflictSolver(unittest.TestCase):
    """Test ConflictSolver functionality."""
    
    def setUp(self):
        from solver.conflict_solver import ConflictSolver
        self.solver = ConflictSolver()
    
    def test_solve_returns_valid_candidates(self):
        """Test that solve returns valid candidate solutions."""
        conflict = {'id1': 'AC001', 'id2': 'AC002', 'distance': 4.0, 'time': 90}
        candidates = self.solver.solve(conflict)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        for candidate in candidates:
            self.assertIsInstance(candidate, dict)
            self.assertIn('action', candidate)
            self.assertIn('type', candidate)
            self.assertIn('safety_score', candidate)
    
    def test_score_best_selects_candidate(self):
        """Test that score_best selects a valid candidate."""
        candidates = [
            {'action': 'turn left 10 degrees', 'type': 'heading', 'safety_score': 0.8},
            {'action': 'climb 1000 ft', 'type': 'altitude', 'safety_score': 0.6}
        ]
        
        best = self.solver.score_best(candidates)
        self.assertIsNotNone(best)
        self.assertIn(best, candidates)

class TestAnalysisMetrics(unittest.TestCase):
    """Test analysis metrics functionality."""
    
    def setUp(self):
        from analysis.metrics import compute_metrics, create_empty_metrics
        self.compute_metrics = compute_metrics
        self.create_empty_metrics = create_empty_metrics
    
    def test_create_empty_metrics(self):
        """Test empty metrics creation."""
        metrics = self.create_empty_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_conflicts', metrics)
        self.assertIn('hallucination_rate', metrics)
        self.assertEqual(metrics['total_conflicts'], 0)
    
    def test_compute_metrics_with_empty_file(self):
        """Test metrics computation with non-existent file."""
        metrics = self.compute_metrics('nonexistent.log')
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['total_conflicts'], 0)

if __name__ == '__main__':
    unittest.main()
