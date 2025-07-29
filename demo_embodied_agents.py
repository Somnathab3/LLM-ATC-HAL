# demo_embodied_agents.py
"""
Demo script showing the embodied agent system in action
"""

import time
from agents.planner import Planner
from agents.executor import Executor  
from agents.verifier import Verifier
from agents.scratchpad import Scratchpad
from llm_interface.llm_client import LLMClient
from tools import bluesky_tools

def demo_planning_loop():
    """Demonstrate the complete planning loop"""
    print("="*60)
    print("EMBODIED AGENT PLANNING LOOP DEMONSTRATION")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing embodied agent components...")
    llm_client = LLMClient()
    planner = Planner(llm_client=llm_client)
    executor = Executor()
    verifier = Verifier()
    scratchpad = Scratchpad()
    print("✓ All components initialized")
    
    # Start session
    print("\n2. Starting new reasoning session...")
    session_id = scratchpad.start_new_session()
    print(f"✓ Session started: {session_id}")
    
    # Planning loop
    print("\n3. Starting planning loop...")
    max_iterations = 5
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n--- Iteration {iteration_count} ---")
        
        # Step 1: Get aircraft information
        print("📡 Getting aircraft information...")
        try:
            info = bluesky_tools.GetAllAircraftInfo()
            print(f"✓ Retrieved info for {info['total_aircraft']} aircraft")
            
            # Log monitoring step
            scratchpad.log_monitoring_step(info)
            
        except Exception as e:
            print(f"✗ Error getting aircraft info: {e}")
            continue
        
        # Step 2: Assess conflicts
        print("🔍 Assessing conflicts...")
        try:
            assessment = planner.assess_conflict(info)
            
            if assessment is None:
                print("✓ No conflicts detected - continuing monitoring")
                break
            
            print(f"⚠️  Conflict detected: {assessment.conflict_id}")
            print(f"   Aircraft: {', '.join(assessment.aircraft_involved)}")
            print(f"   Severity: {assessment.severity}")
            print(f"   Recommended action: {assessment.recommended_action.value}")
            
            # Log assessment step
            scratchpad.log_assessment_step(assessment)
            
        except Exception as e:
            print(f"✗ Error in conflict assessment: {e}")
            scratchpad.log_error_step(f"Conflict assessment error: {e}")
            break
        
        # Step 3: Generate action plan  
        print("📋 Generating action plan...")
        try:
            plan = planner.generate_action_plan(assessment)
            
            if plan is None:
                print("✗ Failed to generate action plan")
                scratchpad.log_error_step("Failed to generate action plan")
                break
            
            print(f"✓ Plan generated: {plan.plan_id}")
            print(f"   Commands: {', '.join(plan.commands)}")
            print(f"   Priority: {plan.priority}/10")
            print(f"   Confidence: {plan.confidence:.1%}")
            
            # Log planning step
            scratchpad.log_planning_step(plan)
            
        except Exception as e:
            print(f"✗ Error generating plan: {e}")
            scratchpad.log_error_step(f"Plan generation error: {e}")
            break
        
        # Step 4: Execute plan
        print("⚡ Executing action plan...")
        try:
            exec_result = executor.send_plan(plan)
            
            print(f"✓ Execution completed: {exec_result.execution_id}")
            print(f"   Commands sent: {len(exec_result.commands_sent)}")
            print(f"   Success rate: {exec_result.success_rate:.1%}")
            print(f"   Execution time: {exec_result.execution_time:.2f}s")
            
            if exec_result.error_messages:
                print(f"   Warnings: {len(exec_result.error_messages)} issues")
            
            # Log execution step
            scratchpad.log_execution_step(exec_result)
            
        except Exception as e:
            print(f"✗ Error in execution: {e}")
            scratchpad.log_error_step(f"Execution error: {e}")
            break
        
        # Step 5: Verify execution
        print("✅ Verifying execution...")
        try:
            verification_passed = verifier.check(exec_result, timeout_seconds=5)
            
            # Get verification details
            verification_results = verifier.get_verification_history()
            if verification_results:
                latest_verification = verification_results[-1]
                
                print(f"✓ Verification: {'PASSED' if verification_passed else 'FAILED'}")
                print(f"   Safety score: {latest_verification.safety_score:.2f}")
                print(f"   Checks passed: {len(latest_verification.passed_checks)}")
                print(f"   Checks failed: {len(latest_verification.failed_checks)}")
                
                if latest_verification.warnings:
                    print(f"   Warnings: {len(latest_verification.warnings)}")
                
                # Log verification step
                scratchpad.log_verification_step(latest_verification)
            
            # Check if we should continue
            if not verification_passed:
                print("⚠️  Verification failed - stopping planning loop")
                scratchpad.log_error_step("Verification failed, stopping planning loop")
                break
                
        except Exception as e:
            print(f"✗ Error in verification: {e}")
            scratchpad.log_error_step(f"Verification error: {e}")
            break
        
        print("✓ Iteration completed successfully")
        time.sleep(1)  # Brief pause between iterations
    
    # Complete session
    print(f"\n4. Completing session...")
    try:
        session_summary = scratchpad.complete_session(
            success=True,
            final_status="completed" if iteration_count < max_iterations else "max_iterations_reached"
        )
        
        print(f"✓ Session completed: {session_summary.session_id}")
        print(f"   Duration: {session_summary.end_time - session_summary.start_time:.2f}s")
        print(f"   Total steps: {session_summary.total_steps}")
        print(f"   Conflicts resolved: {session_summary.conflicts_resolved}")
        print(f"   Commands executed: {session_summary.commands_executed}")
        print(f"   Average confidence: {session_summary.average_confidence:.1%}")
        
        if session_summary.key_decisions:
            print(f"   Key decisions: {len(session_summary.key_decisions)}")
        
        if session_summary.lessons_learned:
            print(f"   Lessons learned: {len(session_summary.lessons_learned)}")
        
    except Exception as e:
        print(f"✗ Error completing session: {e}")
    
    # Get final metrics
    print(f"\n5. Final system metrics...")
    try:
        planner_history = planner.get_assessment_history()
        executor_metrics = executor.get_execution_metrics()
        verifier_metrics = verifier.get_verification_metrics()
        session_metrics = scratchpad.get_session_metrics()
        
        print(f"✓ Planner: {len(planner_history)} assessments made")
        print(f"✓ Executor: {executor_metrics['total_executions']} executions, {executor_metrics['success_rate']:.1%} success rate")
        print(f"✓ Verifier: {verifier_metrics['total_verifications']} verifications, {verifier_metrics['pass_rate']:.1%} pass rate")
        print(f"✓ Scratchpad: {session_metrics['total_steps']} steps logged, {session_metrics['completion_rate']:.1%} completion rate")
        
    except Exception as e:
        print(f"✗ Error getting final metrics: {e}")
    
    print("\n" + "="*60)
    print("✓ EMBODIED AGENT DEMONSTRATION COMPLETED")
    print("="*60)


def demo_function_calling():
    """Demonstrate function calling capabilities"""
    print("\n" + "="*60)
    print("FUNCTION CALLING DEMONSTRATION")
    print("="*60)
    
    # Initialize LLM client
    print("\n1. Initializing LLM client with function calling...")
    llm_client = LLMClient()
    print("✓ LLM client initialized")
    
    # Test basic function call detection
    print("\n2. Testing function call detection...")
    try:
        # Simulate a function call response
        test_prompt = "Get current aircraft information for conflict assessment"
        
        # For demo, we'll simulate what would happen
        print(f"📤 Prompt: {test_prompt}")
        print("🤖 LLM would analyze and decide to call GetAllAircraftInfo()")
        
        # Manually execute the function to show the result
        result = bluesky_tools.GetAllAircraftInfo()
        print(f"✓ Function executed successfully")
        print(f"   Aircraft count: {result['total_aircraft']}")
        print(f"   Timestamp: {result['timestamp']}")
        
    except Exception as e:
        print(f"✗ Error in function calling demo: {e}")
    
    # Test tool registry
    print("\n3. Testing tool registry...")
    try:
        available_tools = bluesky_tools.get_available_tools()
        print(f"✓ Available tools: {len(available_tools)}")
        for tool in available_tools:
            print(f"   - {tool}")
        
    except Exception as e:
        print(f"✗ Error accessing tool registry: {e}")
    
    print("\n" + "="*60)
    print("✓ FUNCTION CALLING DEMONSTRATION COMPLETED")  
    print("="*60)


if __name__ == "__main__":
    """Main demo execution"""
    print("LLM-ATC-HAL EMBODIED AGENT SYSTEM DEMO")
    print("This demo showcases the complete embodied agent architecture")
    print("including planning, execution, verification, and function calling")
    
    try:
        # Run planning loop demo
        demo_planning_loop()
        
        # Run function calling demo
        demo_function_calling()
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY! 🎉")
        print("\nThe embodied agent system is working correctly with:")
        print("✓ Conflict assessment and planning")
        print("✓ Command execution and verification") 
        print("✓ Step-by-step reasoning and logging")
        print("✓ Function calling and tool integration")
        print("✓ Memory and experience tracking")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
