# quick_test.py
"""
Quick test of the LLM-ATC-HAL system with a simple scenario
"""
import asyncio
import time
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from llm_interface.llm_client import LLMClient
from llm_interface.ensemble import OllamaEnsembleClient
from solver.conflict_solver import ConflictSolver
from analysis.metrics import calc_fp_fn, calc_path_extra
from metrics.safety_margin_quantifier import SafetyMarginQuantifier
from scenarios.monte_carlo_framework import BlueSkyScenarioGenerator, ComplexityTier

def test_llm_basic_functionality():
    """Test basic LLM functionality"""
    print("="*60)
    print("TESTING LLM BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        client = LLMClient(model='llama3.1:8b')
        
        # Test basic ATC scenario
        prompt = """You are an Air Traffic Controller. Given this scenario:
        Aircraft AC001 at position (52.3, 4.8) altitude 35000ft, heading 090, speed 450kts
        Aircraft AC002 at position (52.35, 4.85) altitude 35000ft, heading 270, speed 420kts
        
        Analyze if there is a potential conflict and suggest resolution if needed.
        Respond in JSON format with: conflict_detected (boolean), severity (low/medium/high), suggested_action (string)"""
        
        start_time = time.time()
        response = client.ask(prompt, expect_json=True)
        response_time = time.time() - start_time
        
        print(f"✓ LLM Response Time: {response_time:.3f}s")
        print(f"✓ Response: {response}")
        
        return True, response_time, response
        
    except Exception as e:
        print(f"✗ LLM Test Failed: {str(e)}")
        return False, 0, None

def test_ensemble_functionality():
    """Test ensemble LLM functionality"""
    print("\n" + "="*60)
    print("TESTING ENSEMBLE LLM FUNCTIONALITY")
    print("="*60)
    
    try:
        ensemble = OllamaEnsembleClient()
        
        # Test ensemble decision making
        scenario = {
            "aircraft": [
                {"id": "AC001", "lat": 52.3, "lon": 4.8, "alt": 35000, "heading": 90, "speed": 450},
                {"id": "AC002", "lat": 52.35, "lon": 4.85, "alt": 35000, "heading": 270, "speed": 420}
            ],
            "conflict_predicted": True,
            "time_to_conflict": 120
        }
        
        start_time = time.time()
        ensemble_response = ensemble.generate_consensus_decision(scenario)
        response_time = time.time() - start_time
        
        print(f"✓ Ensemble Response Time: {response_time:.3f}s")
        print(f"✓ Ensemble Response: {ensemble_response}")
        
        return True, response_time, ensemble_response
        
    except Exception as e:
        print(f"✗ Ensemble Test Failed: {str(e)}")
        return False, 0, None

def test_conflict_solver():
    """Test conflict solver functionality"""
    print("\n" + "="*60)
    print("TESTING CONFLICT SOLVER")
    print("="*60)
    
    try:
        solver = ConflictSolver()
        
        # Test conflict scenario
        conflict = {
            'aircraft1': {'id': 'AC001', 'lat': 52.3, 'lon': 4.8, 'alt': 35000, 'heading': 90, 'speed': 450},
            'aircraft2': {'id': 'AC002', 'lat': 52.35, 'lon': 4.85, 'alt': 35000, 'heading': 270, 'speed': 420},
            'time_to_conflict': 120,
            'severity': 'medium'
        }
        
        start_time = time.time()
        solutions = solver.solve(conflict)
        solve_time = time.time() - start_time
        
        print(f"✓ Solver Response Time: {solve_time:.3f}s")
        print(f"✓ Solutions Generated: {len(solutions)}")
        for i, solution in enumerate(solutions[:3]):  # Show first 3
            print(f"  Solution {i+1}: {solution}")
            
        return True, solve_time, solutions
        
    except Exception as e:
        print(f"✗ Conflict Solver Test Failed: {str(e)}")
        return False, 0, None

def test_safety_metrics():
    """Test safety metrics calculation"""
    print("\n" + "="*60)
    print("TESTING SAFETY METRICS")
    print("="*60)
    
    try:
        quantifier = SafetyMarginQuantifier()
        
        # Test trajectory data
        trajectories = [
            {
                'aircraft_id': 'AC001',
                'points': [
                    {'lat': 52.3, 'lon': 4.8, 'alt': 35000, 'time': 0},
                    {'lat': 52.31, 'lon': 4.81, 'alt': 35000, 'time': 60},
                    {'lat': 52.32, 'lon': 4.82, 'alt': 35000, 'time': 120}
                ]
            },
            {
                'aircraft_id': 'AC002',
                'points': [
                    {'lat': 52.35, 'lon': 4.85, 'alt': 35000, 'time': 0},
                    {'lat': 52.34, 'lon': 4.84, 'alt': 35000, 'time': 60},
                    {'lat': 52.33, 'lon': 4.83, 'alt': 35000, 'time': 120}
                ]
            }
        ]
        
        start_time = time.time()
        safety_metrics = quantifier.calculate_comprehensive_safety_metrics(trajectories)
        calc_time = time.time() - start_time
        
        print(f"✓ Safety Metrics Calculation Time: {calc_time:.3f}s")
        print(f"✓ Safety Metrics: {safety_metrics}")
        
        return True, calc_time, safety_metrics
        
    except Exception as e:
        print(f"✗ Safety Metrics Test Failed: {str(e)}")
        return False, 0, None

def test_scenario_generation():
    """Test scenario generation"""
    print("\n" + "="*60)
    print("TESTING SCENARIO GENERATION")
    print("="*60)
    
    try:
        generator = BlueSkyScenarioGenerator()
        
        start_time = time.time()
        scenario = generator.generate_scenario(
            complexity=ComplexityTier.MODERATE,
            num_aircraft=4,
            duration=300
        )
        gen_time = time.time() - start_time
        
        print(f"✓ Scenario Generation Time: {gen_time:.3f}s")
        print(f"✓ Generated Scenario: {scenario}")
        
        return True, gen_time, scenario
        
    except Exception as e:
        print(f"✗ Scenario Generation Test Failed: {str(e)}")
        return False, 0, None

async def run_integrated_test():
    """Run an integrated test of the full system"""
    print("\n" + "="*60)
    print("INTEGRATED SYSTEM TEST")
    print("="*60)
    
    try:
        # Generate a test scenario
        print("1. Generating test scenario...")
        generator = BlueSkyScenarioGenerator()
        scenario = generator.generate_scenario(
            complexity=ComplexityTier.SIMPLE,
            num_aircraft=2,
            duration=300
        )
        
        # Run LLM analysis
        print("2. Running LLM analysis...")
        client = LLMClient(model='llama3.1:8b')
        
        prompt = f"""Analyze this ATC scenario and detect any potential conflicts:
        Scenario: {scenario}
        
        Provide analysis in JSON format with:
        - conflicts_detected: boolean
        - num_conflicts: integer
        - severity_assessment: string
        - recommended_actions: list of strings
        """
        
        start_time = time.time()
        llm_response = client.ask(prompt, expect_json=True)
        llm_time = time.time() - start_time
        
        print(f"   LLM Analysis Time: {llm_time:.3f}s")
        print(f"   LLM Response: {llm_response}")
        
        # Calculate safety metrics
        print("3. Calculating safety metrics...")
        if scenario.get('aircraft_trajectories'):
            quantifier = SafetyMarginQuantifier()
            safety_metrics = quantifier.calculate_comprehensive_safety_metrics(
                scenario['aircraft_trajectories']
            )
            print(f"   Safety Metrics: {safety_metrics}")
        
        print("✓ Integrated test completed successfully")
        return True, {
            'scenario': scenario,
            'llm_response': llm_response,
            'llm_time': llm_time,
            'safety_metrics': safety_metrics if 'safety_metrics' in locals() else None
        }
        
    except Exception as e:
        print(f"✗ Integrated Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main test function"""
    print("LLM-ATC-HAL SYSTEM QUICK TEST")
    print("=" * 60)
    print(f"Test Started: {datetime.now()}")
    print()
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Basic LLM functionality
    total_tests += 1
    success, response_time, response = test_llm_basic_functionality()
    if success:
        passed_tests += 1
    results['llm_basic'] = {'success': success, 'response_time': response_time, 'response': response}
    
    # Test 2: Ensemble functionality
    total_tests += 1
    success, response_time, response = test_ensemble_functionality()
    if success:
        passed_tests += 1
    results['ensemble'] = {'success': success, 'response_time': response_time, 'response': response}
    
    # Test 3: Conflict solver
    total_tests += 1
    success, response_time, response = test_conflict_solver()
    if success:
        passed_tests += 1
    results['conflict_solver'] = {'success': success, 'response_time': response_time, 'response': response}
    
    # Test 4: Safety metrics
    total_tests += 1
    success, response_time, response = test_safety_metrics()
    if success:
        passed_tests += 1
    results['safety_metrics'] = {'success': success, 'response_time': response_time, 'response': response}
    
    # Test 5: Scenario generation
    total_tests += 1
    success, response_time, response = test_scenario_generation()
    if success:
        passed_tests += 1
    results['scenario_generation'] = {'success': success, 'response_time': response_time, 'response': response}
    
    # Test 6: Integrated test
    total_tests += 1
    success, response = asyncio.run(run_integrated_test())
    if success:
        passed_tests += 1
    results['integrated'] = {'success': success, 'response': response}
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    # Performance summary
    print("\nPerformance Summary:")
    for test_name, result in results.items():
        if result['success'] and 'response_time' in result:
            print(f"  {test_name}: {result['response_time']:.3f}s")
    
    # Identify deficiencies
    print("\nIdentified Deficiencies:")
    for test_name, result in results.items():
        if not result['success']:
            print(f"  ✗ {test_name}: Failed")
        elif 'response_time' in result and result['response_time'] > 5.0:
            print(f"  ⚠ {test_name}: Slow response ({result['response_time']:.3f}s)")
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed! System is ready for comprehensive testing.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Address issues before comprehensive testing.")
    
    return results

if __name__ == "__main__":
    results = main()
