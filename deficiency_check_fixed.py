#!/usr/bin/env python3
"""
Deficiency Check Script for LLM-ATC-HAL
Runs basic tests to identify system deficiencies before full testing
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup logging for deficiency check"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('deficiency_check.log')
        ]
    )
    return logging.getLogger(__name__)

async def check_import_dependencies():
    """Check if all critical dependencies can be imported"""
    logger = logging.getLogger(__name__)
    logger.info("=== Checking Import Dependencies ===")
    
    issues = []
    
    # Core testing modules
    try:
        from testing import TestExecutor, ScenarioManager, ResultAnalyzer, ResultStreamer
        logger.info("[OK] Testing modules imported successfully")
    except ImportError as e:
        issues.append(f"Testing modules: {e}")
        logger.error(f"[ERROR] Testing modules failed: {e}")
    
    # LLM interface
    try:
        from llm_interface.ensemble import OllamaEnsembleClient
        logger.info("[OK] LLM interface imported successfully")
    except ImportError as e:
        issues.append(f"LLM interface: {e}")
        logger.error(f"[ERROR] LLM interface failed: {e}")
    
    # Analysis modules
    try:
        from analysis.enhanced_hallucination_detection import EnhancedHallucinationDetector
        logger.info("[OK] Hallucination detection imported successfully")
    except ImportError as e:
        issues.append(f"Hallucination detection: {e}")
        logger.error(f"[ERROR] Hallucination detection failed: {e}")
    
    # Metrics
    try:
        from llm_atc.metrics.safety_margin_quantifier import SafetyMarginQuantifier
        logger.info("[OK] Safety margin quantifier imported successfully")
    except ImportError as e:
        issues.append(f"Safety margin quantifier: {e}")
        logger.error(f"[ERROR] Safety margin quantifier failed: {e}")
    
    # Memory modules (optional, may have warnings)
    try:
        from llm_atc.memory.experience_integrator import ExperienceIntegrator
        from llm_atc.memory.replay_store import VectorReplayStore
        logger.info("[OK] Memory modules imported successfully (with warnings expected)")
    except ImportError as e:
        issues.append(f"Memory modules: {e}")
        logger.error(f"[ERROR] Memory modules failed: {e}")
    
    return issues

async def check_ollama_connectivity():
    """Check if Ollama is running and accessible"""
    logger = logging.getLogger(__name__)
    logger.info("=== Checking Ollama Connectivity ===")
    
    issues = []
    
    try:
        from llm_interface.ensemble import OllamaEnsembleClient
        client = OllamaEnsembleClient()
        
        # Test a simple prompt
        test_prompt = "Respond with 'OK' if you can understand this message."
        logger.info("Testing Ollama connectivity with simple prompt...")
        
        response = client.query_ensemble(
            test_prompt, 
            {"test": True}, 
            require_json=False
        )
        
        if response and hasattr(response, 'consensus_response'):
            logger.info(f"[OK] Ollama responded: {str(response.consensus_response)[:50]}...")
        else:
            issues.append("Ollama returned invalid response")
            logger.error("[ERROR] Ollama returned invalid response")
            
    except Exception as e:
        issues.append(f"Ollama connectivity: {e}")
        logger.error(f"[ERROR] Ollama connectivity failed: {e}")
    
    return issues

async def check_scenario_generation():
    """Check if scenario generation works"""
    logger = logging.getLogger(__name__)
    logger.info("=== Checking Scenario Generation ===")
    
    issues = []
    
    try:
        from testing import ScenarioManager
        
        manager = ScenarioManager()
        
        # Generate a few test scenarios
        logger.info("Generating test scenarios...")
        scenarios = manager.generate_comprehensive_scenarios(
            num_scenarios=5,
            complexity_distribution={'simple': 1.0}
        )
        
        if len(scenarios) > 0:
            logger.info(f"[OK] Generated {len(scenarios)} test scenarios")
            
            # Validate one scenario
            if manager.validate_scenario_integrity(scenarios[0]):
                logger.info("[OK] Scenario validation working")
            else:
                issues.append("Scenario validation failed")
                logger.error("[ERROR] Scenario validation failed")
        else:
            issues.append("No scenarios generated")
            logger.error("[ERROR] No scenarios generated")
            
    except Exception as e:
        issues.append(f"Scenario generation: {e}")
        logger.error(f"[ERROR] Scenario generation failed: {e}")
    
    return issues

async def check_hallucination_detection():
    """Check if hallucination detection works"""
    logger = logging.getLogger(__name__)
    logger.info("=== Checking Hallucination Detection ===")
    
    issues = []
    
    try:
        from analysis.enhanced_hallucination_detection import EnhancedHallucinationDetector
        
        detector = EnhancedHallucinationDetector()
        
        # Test with a sample response
        test_response = "Aircraft AC999 should turn to heading 270 degrees immediately"
        test_context = {
            'aircraft_data': [
                {'callsign': 'AC001', 'altitude': 35000, 'heading': 180},
                {'callsign': 'AC002', 'altitude': 36000, 'heading': 90}
            ]
        }
        
        logger.info("Testing hallucination detection...")
        result = detector.detect_hallucinations(test_response, {}, test_context)
        
        if result:
            logger.info(f"[OK] Hallucination detection working: {result.detected}")
        else:
            issues.append("Hallucination detection returned None")
            logger.error("[ERROR] Hallucination detection returned None")
            
    except Exception as e:
        issues.append(f"Hallucination detection: {e}")
        logger.error(f"[ERROR] Hallucination detection failed: {e}")
    
    return issues

async def check_safety_quantification():
    """Check if safety margin quantification works"""
    logger = logging.getLogger(__name__)
    logger.info("=== Checking Safety Quantification ===")
    
    issues = []
    
    try:
        from llm_atc.metrics.safety_margin_quantifier import SafetyMarginQuantifier
        
        quantifier = SafetyMarginQuantifier()
        
        logger.info("Testing safety margin quantification...")
        
        # Create test conflict geometry and resolution
        from llm_atc.metrics.safety_margin_quantifier import ConflictGeometry
        
        conflict_geometry = ConflictGeometry(
            aircraft1_pos=(40.0, -74.0, 35000),
            aircraft2_pos=(40.1, -74.1, 36000),
            aircraft1_velocity=(450, 0, 180),
            aircraft2_velocity=(500, 0, 90),
            time_to_closest_approach=120,
            closest_approach_distance=3.0,
            closest_approach_altitude_diff=1000
        )
        
        test_resolution = {
            'type': 'heading',
            'aircraft_id': 'AC001',
            'heading_change': 20
        }
        
        # Test safety margin calculation
        margins = quantifier.calculate_safety_margins(conflict_geometry, test_resolution)
        
        if margins:
            logger.info(f"[OK] Safety quantification working: {margins.safety_level}")
        else:
            issues.append("Safety quantification returned None")
            logger.error("[ERROR] Safety quantification returned None")
            
    except Exception as e:
        issues.append(f"Safety quantification: {e}")
        logger.error(f"[ERROR] Safety quantification failed: {e}")
    
    return issues

async def run_comprehensive_deficiency_check():
    """Run all deficiency checks"""
    logger = setup_logging()
    logger.info("Starting Comprehensive Deficiency Check for LLM-ATC-HAL")
    logger.info("=" * 60)
    
    start_time = time.time()
    all_issues = []
    
    # Run all checks
    checks = [
        ("Import Dependencies", check_import_dependencies()),
        ("Ollama Connectivity", check_ollama_connectivity()),
        ("Scenario Generation", check_scenario_generation()),
        ("Hallucination Detection", check_hallucination_detection()),
        ("Safety Quantification", check_safety_quantification())
    ]
    
    for check_name, check_coro in checks:
        logger.info(f"\nRunning: {check_name}")
        try:
            issues = await check_coro
            if issues:
                all_issues.extend([f"{check_name}: {issue}" for issue in issues])
        except Exception as e:
            all_issues.append(f"{check_name}: Unexpected error - {e}")
            logger.error(f"Unexpected error in {check_name}: {e}")
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("DEFICIENCY CHECK SUMMARY")
    logger.info("=" * 60)
    
    if all_issues:
        logger.error(f"Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            logger.error(f"  {i}. {issue}")
        logger.error("\nRecommendation: Fix these issues before running full test suite")
    else:
        logger.info("[OK] All checks passed! System appears ready for testing.")
    
    logger.info(f"\nDeficiency check completed in {elapsed_time:.2f} seconds")
    
    return all_issues

if __name__ == "__main__":
    asyncio.run(run_comprehensive_deficiency_check())
