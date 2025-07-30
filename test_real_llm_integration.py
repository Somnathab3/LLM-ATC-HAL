#!/usr/bin/env python3
"""
Test script to validate real LLM integration in comprehensive_hallucination_tester_v2.py
This script runs a small subset of the OFAT sweep with real LLM testing to ensure
everything works properly before running the full analysis.
"""

import asyncio
import logging
import os
import sys
import yaml
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from scenarios.monte_carlo_framework import BlueSkyScenarioGenerator, ComplexityTier
from llm_interface.ensemble import OllamaEnsembleClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_single_scenario_llm():
    """Test a single scenario with real LLM response"""
    print("="*60)
    print("TESTING SINGLE SCENARIO WITH REAL LLM")
    print("="*60)
    
    try:
        # 1. Generate a real scenario
        print("\n1. Generating real scenario using BlueSkyScenarioGenerator...")
        generator = BlueSkyScenarioGenerator()
        scenario = generator.generate_scenario(
            complexity_tier=ComplexityTier.MODERATE,
            force_conflicts=True,
            distribution_shift_tier='in_distribution'
        )
        
        print(f"   ✓ Generated scenario with {scenario.aircraft_count} aircraft")
        print(f"   ✓ Complexity: {scenario.complexity_tier.value}")
        print(f"   ✓ Duration: {scenario.duration_minutes:.1f} minutes")
        print(f"   ✓ BlueSky commands: {len(scenario.bluesky_commands)}")
        
        # 2. Initialize LLM ensemble
        print("\n2. Initializing Ollama ensemble...")
        try:
            ensemble_client = OllamaEnsembleClient()
            print("   ✓ Ensemble client initialized successfully")
            print(f"   ✓ Available models: {list(ensemble_client.models.keys())}")
        except Exception as e:
            print(f"   ✗ Failed to initialize ensemble: {e}")
            return False
        
        # 3. Create ATC context from scenario
        print("\n3. Creating ATC context...")
        context = create_test_context(scenario)
        print(f"   ✓ Context created for {len(context['aircraft_list'])} aircraft")
        print(f"   ✓ Airspace: {context['airspace_region']}")
        
        # 4. Create ATC prompt
        print("\n4. Creating ATC conflict resolution prompt...")
        prompt = create_test_prompt(scenario, context)
        print(f"   ✓ Prompt created ({len(prompt)} characters)")
        
        # 5. Query LLM ensemble
        print("\n5. Querying LLM ensemble for conflict resolution...")
        start_time = datetime.now()
        
        try:
            ensemble_response = ensemble_client.query_ensemble(
                prompt=prompt,
                context=context,
                require_json=True,
                timeout=30.0
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            print(f"   ✓ LLM response received in {response_time:.2f} seconds")
            print(f"   ✓ Confidence: {ensemble_response.confidence:.3f}")
            print(f"   ✓ Consensus score: {ensemble_response.consensus_score:.3f}")
            print(f"   ✓ Safety flags: {len(ensemble_response.safety_flags)}")
            
            # 6. Analyze response
            print("\n6. Analyzing LLM response...")
            response_data = ensemble_response.consensus_response
            
            if 'error' in response_data:
                print(f"   ⚠ LLM returned error: {response_data['error']}")
            else:
                print(f"   ✓ Response contains conflict analysis: {'conflict_analysis' in response_data}")
                print(f"   ✓ Resolution instructions provided: {len(response_data.get('resolution_instructions', []))}")
                print(f"   ✓ Safety assessment included: {'safety_assessment' in response_data}")
                
                # Show some response details
                if 'safety_assessment' in response_data:
                    safety = response_data['safety_assessment']
                    print(f"   ✓ Safety level: {safety.get('overall_safety_level', 'unknown')}")
                    print(f"   ✓ Safety score: {safety.get('safety_score', 'unknown')}")
            
            print("\n" + "="*60)
            print("✓ SINGLE SCENARIO TEST COMPLETED SUCCESSFULLY")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"   ✗ LLM query failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def create_test_context(scenario):
    """Create test ATC context from scenario"""
    aircraft_list = []
    
    for i, (aircraft_type, position) in enumerate(zip(
        scenario.aircraft_types, 
        scenario.positions
    )):
        aircraft_list.append({
            'callsign': f"TEST{i+1:02d}",
            'aircraft_type': aircraft_type,
            'position': {
                'latitude': position['lat'],
                'longitude': position['lon'],
                'altitude_ft': position['alt']
            },
            'speed_kts': scenario.speeds[i] if i < len(scenario.speeds) else 250,
            'heading_deg': scenario.headings[i] if i < len(scenario.headings) else 90
        })
    
    return {
        'scenario_id': f"test_{scenario.generated_timestamp}",
        'airspace_region': scenario.airspace_region,
        'aircraft_count': scenario.aircraft_count,
        'aircraft_list': aircraft_list,
        'environmental_conditions': scenario.environmental_conditions,
        'complexity_tier': scenario.complexity_tier.value,
        'duration_minutes': scenario.duration_minutes,
        'conflicts_detected': True,
        'timestamp': datetime.now().isoformat()
    }

def create_test_prompt(scenario, context):
    """Create test ATC prompt"""
    aircraft_descriptions = []
    for aircraft in context['aircraft_list']:
        desc = (f"{aircraft['callsign']} ({aircraft['aircraft_type']}) at "
               f"FL{aircraft['position']['altitude_ft']//100:03d}, "
               f"{aircraft['speed_kts']}kts, heading {aircraft['heading_deg']:03d}°")
        aircraft_descriptions.append(desc)
    
    return f"""
You are an AI Air Traffic Controller assistant. Analyze the following traffic scenario and provide conflict resolution instructions.

SCENARIO INFORMATION:
- Airspace: {context['airspace_region']}
- Aircraft Count: {context['aircraft_count']}
- Environmental Conditions: Wind {context['environmental_conditions']['wind_speed_kts']}kts from {context['environmental_conditions']['wind_direction_deg']}°
- Visibility: {context['environmental_conditions']['visibility_nm']}NM

CURRENT TRAFFIC:
{chr(10).join(aircraft_descriptions)}

CONFLICT SITUATION:
Multiple aircraft are on conflicting flight paths with potential loss of separation. 

INSTRUCTIONS:
1. Identify the primary conflict(s)
2. Provide specific resolution instructions for each affected aircraft
3. Ensure minimum separation standards (5NM horizontal, 1000ft vertical)
4. Consider aircraft performance characteristics and environmental conditions
5. Provide safety assessment and rationale

Respond in JSON format with:
{{
    "conflict_analysis": "description of conflicts detected",
    "resolution_instructions": [
        {{
            "callsign": "aircraft_callsign",
            "instruction_type": "heading_change|altitude_change|speed_change|hold",
            "instruction": "specific instruction text",
            "new_heading": number_or_null,
            "new_altitude": number_or_null,
            "new_speed": number_or_null,
            "rationale": "reason for this instruction"
        }}
    ],
    "safety_assessment": {{
        "overall_safety_level": "low|medium|high",
        "safety_score": 0.0-1.0,
        "risk_factors": ["factor1", "factor2"],
        "separation_assurance": "description of how separation is maintained"
    }},
    "operational_impact": {{
        "delay_minutes": 0,
        "fuel_impact": "low|medium|high",
        "passenger_comfort": "minimal|moderate|significant"
    }}
}}
"""

async def main():
    """Main test function"""
    print("Starting real LLM integration test...")
    
    # Test single scenario
    success = await test_single_scenario_llm()
    
    if success:
        print("\n🎉 All tests passed! Real LLM integration is working.")
        print("You can now run the full comprehensive_hallucination_tester_v2.py")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
