#!/usr/bin/env python3
"""
Test script for extended BlueSky tools functionality
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools import (
    get_distance,
    step_simulation,
    reset_simulation,
    get_minimum_separation,
    check_separation_violation,
    get_all_aircraft_info,
    send_command
)


def test_distance_calculation():
    """Test distance calculation between aircraft"""
    print("🧮 Testing Distance Calculation")
    print("=" * 50)
    
    try:
        # Test distance calculation using default aircraft data
        distance_result = get_distance("AAL123", "DLH456")
        
        print(f"✅ Distance calculation successful!")
        print(f"   📏 Horizontal separation: {distance_result['horizontal_nm']:.2f} nm")
        print(f"   📐 Vertical separation: {distance_result['vertical_ft']:.0f} ft") 
        print(f"   📊 3D separation: {distance_result['total_3d_nm']:.2f} nm")
        
        # Test with non-existent aircraft
        try:
            get_distance("NONEXISTENT1", "NONEXISTENT2")
            print("❌ Should have failed with non-existent aircraft")
            return False
        except Exception as e:
            print(f"✅ Correctly handled non-existent aircraft: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Distance calculation failed: {e}")
        return False


def test_simulation_stepping():
    """Test simulation time stepping"""
    print("\n⏰ Testing Simulation Stepping")
    print("=" * 50)
    
    try:
        # Test stepping simulation forward
        step_result = step_simulation(minutes=2.0, dtmult=2.0)
        
        if step_result['success']:
            print(f"✅ Simulation stepping successful!")
            print(f"   ⏱️ Minutes advanced: {step_result['minutes_advanced']}")
            print(f"   🔢 Seconds advanced: {step_result['seconds_advanced']}")
            print(f"   ⚡ Time multiplier: {step_result['dtmult']}")
            print(f"   📝 Command sent: {step_result['command_sent']}")
        else:
            print(f"❌ Simulation stepping failed: {step_result.get('error', 'Unknown error')}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Simulation stepping test failed: {e}")
        return False


def test_simulation_reset():
    """Test simulation reset functionality"""
    print("\n🔄 Testing Simulation Reset")
    print("=" * 50)
    
    try:
        # Test resetting simulation
        reset_result = reset_simulation()
        
        if reset_result['success']:
            print(f"✅ Simulation reset successful!")
            print(f"   🎯 Status: {reset_result['status']}")
            print(f"   🛩️ Aircraft count: {reset_result['aircraft_count']}")
            print(f"   📋 Setup commands executed: {len(reset_result['setup_commands'])}")
        else:
            print(f"❌ Simulation reset failed: {reset_result.get('error', 'Unknown error')}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Simulation reset test failed: {e}")
        return False


def test_separation_standards():
    """Test separation standards and violation checking"""
    print("\n📐 Testing Separation Standards")
    print("=" * 50)
    
    try:
        # Test getting minimum separation standards
        min_sep = get_minimum_separation()
        
        print(f"✅ Separation standards retrieved!")
        print(f"   📏 Standard horizontal: {min_sep['horizontal_nm']} nm")
        print(f"   📐 Standard vertical: {min_sep['vertical_ft']} ft")
        print(f"   🛬 Approach horizontal: {min_sep['approach_horizontal_nm']} nm")
        print(f"   🌊 Oceanic horizontal: {min_sep['oceanic_horizontal_nm']} nm")
        
        # Test separation violation checking
        violation_check = check_separation_violation("AAL123", "DLH456")
        
        print(f"\n✅ Separation violation check completed!")
        print(f"   ✈️ Aircraft pair: {violation_check['aircraft_pair']}")
        
        violations = violation_check.get('violations', {})
        if violations:
            print(f"   🚨 Horizontal violation: {'YES' if violations.get('horizontal', False) else 'NO'}")
            print(f"   📐 Vertical violation: {'YES' if violations.get('vertical', False) else 'NO'}")
            print(f"   ⚠️ Separation loss: {'YES' if violations.get('separation_loss', False) else 'NO'}")
        
        margins = violation_check.get('safety_margins', {})
        if margins:
            print(f"   💚 Horizontal margin: {margins.get('horizontal_nm', 0):.2f} nm")
            print(f"   💚 Vertical margin: {margins.get('vertical_ft', 0):.0f} ft")
        
        return True
        
    except Exception as e:
        print(f"❌ Separation standards test failed: {e}")
        return False


def test_expanded_commands():
    """Test expanded BlueSky command validation"""
    print("\n🎮 Testing Expanded Commands")
    print("=" * 50)
    
    # Test new commands that should be recognized
    new_commands = [
        "DT 60",
        "DTMULT 2.0", 
        "RESET",
        "AREA 52.0,4.0",
        "CDMETHOD SWARM",
        "CDSEP 5.0 1000",
        "WIND 52.0,4.0,ALL,270,15",
        "TURB 0.5",
        "VS AAL123 500",
        "GO",
        "PAUSE",
        "FF 2"
    ]
    
    success_count = 0
    total_count = len(new_commands)
    
    for cmd in new_commands:
        try:
            result = send_command(cmd)
            if result['success']:
                print(f"   ✅ {cmd} - {result['status']}")
                success_count += 1
            else:
                print(f"   ❌ {cmd} - {result.get('error', 'Failed')}")
        except Exception as e:
            print(f"   💥 {cmd} - Exception: {e}")
    
    print(f"\n📊 Command validation: {success_count}/{total_count} successful")
    return success_count == total_count


def test_aircraft_info_integration():
    """Test integration with aircraft information"""
    print("\n✈️ Testing Aircraft Info Integration")
    print("=" * 50)
    
    try:
        # Get aircraft info
        aircraft_data = get_all_aircraft_info()
        aircraft_dict = aircraft_data.get("aircraft", {})
        
        print(f"✅ Retrieved info for {len(aircraft_dict)} aircraft")
        
        # Test with actual aircraft IDs
        if len(aircraft_dict) >= 2:
            aircraft_ids = list(aircraft_dict.keys())
            ac1, ac2 = aircraft_ids[0], aircraft_ids[1]
            
            print(f"   🎯 Testing with {ac1} and {ac2}")
            
            # Test distance calculation
            distance = get_distance(ac1, ac2)
            print(f"   📏 Distance: {distance['horizontal_nm']:.2f} nm horizontal")
            
            # Test separation check
            violation = check_separation_violation(ac1, ac2)
            sep_loss = violation.get('violations', {}).get('separation_loss', False)
            print(f"   ⚠️ Separation violation: {'YES' if sep_loss else 'NO'}")
            
        return True
        
    except Exception as e:
        print(f"❌ Aircraft info integration test failed: {e}")
        return False


def main():
    """Run all extended BlueSky tools tests"""
    print("🚀 EXTENDED BLUESKY TOOLS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Distance Calculation", test_distance_calculation),
        ("Simulation Stepping", test_simulation_stepping),
        ("Simulation Reset", test_simulation_reset),
        ("Separation Standards", test_separation_standards),
        ("Expanded Commands", test_expanded_commands),
        ("Aircraft Info Integration", test_aircraft_info_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 60)
    print(f"🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Extended BlueSky tools are working correctly!")
        print("\n🚀 Ready for Monte Carlo runner integration!")
        print("\n📋 New capabilities added:")
        print("• Distance calculation with haversine formula")
        print("• Simulation time stepping with DT commands")
        print("• Simulation reset with setup commands")
        print("• Separation violation checking")
        print("• Expanded command validation (DT, DTMULT, VS, etc.)")
        print("• Function calling registry updated")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
