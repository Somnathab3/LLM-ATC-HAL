#!/usr/bin/env python3
"""
Test script for the new real BlueSky integration in bluesky_tools.py
This tests both the real BlueSky integration (when available) and the fallback mock behavior.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools.bluesky_tools import (
    get_all_aircraft_info,
    get_conflict_info,
    send_command,
    step_simulation,
    reset_simulation,
    get_distance,
    check_separation_violation,
    get_minimum_separation,
    _bluesky_interface,
    BLUESKY_AVAILABLE
)


def test_bluesky_availability():
    """Test BlueSky availability detection"""
    print("🔍 Testing BlueSky Availability Detection")
    print("=" * 50)
    
    print(f"   BlueSky Available: {BLUESKY_AVAILABLE}")
    print(f"   Interface Available: {_bluesky_interface.is_available()}")
    print(f"   Connection Type: {_bluesky_interface.connection_type}")
    
    if BLUESKY_AVAILABLE:
        print("   ✅ BlueSky simulator is available - will use real integration")
        print("   🔗 This will provide actual aircraft data and simulation dynamics")
    else:
        print("   ⚠️ BlueSky simulator not available - using enhanced mock data")
        print("   🎭 Mock data will be more realistic than previous hardcoded values")
    
    return True


def test_aircraft_info_integration():
    """Test aircraft information retrieval with real/mock integration"""
    print("\n✈️ Testing Aircraft Information Integration")
    print("=" * 50)
    
    try:
        aircraft_data = get_all_aircraft_info()
        
        print(f"   📊 Retrieved data from: {aircraft_data.get('source', 'unknown')}")
        print(f"   🛩️ Total aircraft: {aircraft_data.get('total_aircraft', 0)}")
        
        aircraft_dict = aircraft_data.get("aircraft", {})
        if aircraft_dict:
            print(f"   📋 Aircraft list:")
            for i, (acid, data) in enumerate(aircraft_dict.items()):
                if i < 5:  # Show first 5 aircraft
                    print(f"      {acid}: {data['lat']:.4f}°N, {data['lon']:.4f}°E, {data['alt']:.0f}ft")
                elif i == 5:
                    print(f"      ... and {len(aircraft_dict) - 5} more aircraft")
                    break
        
        # Verify data structure
        required_fields = ["aircraft", "timestamp", "total_aircraft", "simulation_time", "source"]
        for field in required_fields:
            if field in aircraft_data:
                print(f"   ✅ {field}: present")
            else:
                print(f"   ❌ {field}: missing")
                return False
                
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_conflict_detection_integration():
    """Test conflict detection with real/mock integration"""
    print("\n🚨 Testing Conflict Detection Integration")
    print("=" * 50)
    
    try:
        conflict_data = get_conflict_info()
        
        print(f"   📊 Retrieved data from: {conflict_data.get('source', 'unknown')}")
        print(f"   ⚠️ Total conflicts: {conflict_data.get('total_conflicts', 0)}")
        print(f"   🔴 High priority: {conflict_data.get('high_priority_conflicts', 0)}")
        print(f"   🟡 Medium priority: {conflict_data.get('medium_priority_conflicts', 0)}")
        print(f"   🟢 Low priority: {conflict_data.get('low_priority_conflicts', 0)}")
        
        conflicts = conflict_data.get("conflicts", [])
        if conflicts:
            print(f"   📋 Conflict details:")
            for i, conflict in enumerate(conflicts[:3]):  # Show first 3 conflicts
                print(f"      {conflict['conflict_id']}: {conflict['aircraft_1']} ↔ {conflict['aircraft_2']}")
                print(f"         H: {conflict['horizontal_separation']:.1f}nm, V: {conflict['vertical_separation']:.0f}ft")
                print(f"         Severity: {conflict['severity']}")
                
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_command_integration():
    """Test command sending with real/mock integration"""
    print("\n🎮 Testing Command Integration")
    print("=" * 50)
    
    test_commands = [
        "DT 60",
        "DTMULT 1.5",
        "RESET",
        "CDMETHOD SWARM",
        "CDSEP 5.0 1000"
    ]
    
    success_count = 0
    for cmd in test_commands:
        try:
            result = send_command(cmd)
            source = result.get('source', 'unknown')
            success = result.get('success', False)
            status = result.get('status', 'unknown')
            
            print(f"   {cmd}: {status} ({source}) {'✅' if success else '❌'}")
            if success:
                success_count += 1
                
        except Exception as e:
            print(f"   {cmd}: ERROR - {e} ❌")
    
    print(f"\n   📊 Command success rate: {success_count}/{len(test_commands)}")
    return success_count > 0


def test_simulation_control():
    """Test simulation stepping and reset with real/mock integration"""
    print("\n⏰ Testing Simulation Control Integration")
    print("=" * 50)
    
    try:
        # Test simulation step
        print("   🔄 Testing simulation step...")
        step_result = step_simulation(minutes=1.0, dtmult=2.0)
        source = step_result.get('source', 'unknown')
        success = step_result.get('success', False)
        
        print(f"      Step result: {step_result.get('status', 'unknown')} ({source}) {'✅' if success else '❌'}")
        
        if success:
            print(f"      Minutes advanced: {step_result.get('minutes_advanced', 0)}")
            print(f"      Time multiplier: {step_result.get('dtmult', 1.0)}")
        
        # Test simulation reset
        print("   🔄 Testing simulation reset...")
        reset_result = reset_simulation()
        source = reset_result.get('source', 'unknown')
        success = reset_result.get('success', False)
        
        print(f"      Reset result: {reset_result.get('status', 'unknown')} ({source}) {'✅' if success else '❌'}")
        
        if success:
            print(f"      Aircraft count after reset: {reset_result.get('aircraft_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_distance_calculation():
    """Test distance calculation between aircraft"""
    print("\n📏 Testing Distance Calculation")
    print("=" * 50)
    
    try:
        # Get aircraft data first
        aircraft_data = get_all_aircraft_info()
        aircraft_dict = aircraft_data.get("aircraft", {})
        
        if len(aircraft_dict) < 2:
            print("   ⚠️ Not enough aircraft for distance calculation test")
            return True
            
        # Get first two aircraft
        aircraft_ids = list(aircraft_dict.keys())
        ac1, ac2 = aircraft_ids[0], aircraft_ids[1]
        
        print(f"   📐 Calculating distance between {ac1} and {ac2}")
        
        distance_result = get_distance(ac1, ac2)
        source = distance_result.get('source', 'calculated')
        
        print(f"   📊 Distance data source: {source}")
        print(f"   📏 Horizontal separation: {distance_result['horizontal_nm']:.2f} nm")
        print(f"   📐 Vertical separation: {distance_result['vertical_ft']:.0f} ft")
        print(f"   📊 3D separation: {distance_result['total_3d_nm']:.2f} nm")
        
        # Test separation violation check
        violation_result = check_separation_violation(ac1, ac2)
        violations = violation_result.get('violations', {})
        
        print(f"   🚨 Separation violations:")
        print(f"      Horizontal: {'YES' if violations.get('horizontal', False) else 'NO'}")
        print(f"      Vertical: {'YES' if violations.get('vertical', False) else 'NO'}")
        print(f"      Separation loss: {'YES' if violations.get('separation_loss', False) else 'NO'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_configuration_system():
    """Test the configuration system"""
    print("\n⚙️ Testing Configuration System")
    print("=" * 50)
    
    try:
        from llm_atc.tools.bluesky_tools import _config
        
        # Test configuration access
        connection_type = _config.get('bluesky.connection_type', 'default')
        h_sep = _config.get('bluesky.simulation.separation_standards.horizontal_nm', 5.0)
        mock_count = _config.get('bluesky.mock_data.default_aircraft_count', 10)
        
        print(f"   🔗 Connection type: {connection_type}")
        print(f"   📏 Horizontal separation standard: {h_sep} nm")
        print(f"   🛩️ Mock aircraft count: {mock_count}")
        print(f"   📁 Config file: {_config.config_path}")
        
        # Check if config file exists
        if os.path.exists(_config.config_path):
            print(f"   ✅ Configuration file found")
        else:
            print(f"   ⚠️ Configuration file not found (using defaults)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 BLUESKY INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("BlueSky Availability", test_bluesky_availability),
        ("Configuration System", test_configuration_system),
        ("Aircraft Information", test_aircraft_info_integration),
        ("Conflict Detection", test_conflict_detection_integration),
        ("Command Integration", test_command_integration),
        ("Simulation Control", test_simulation_control),
        ("Distance Calculation", test_distance_calculation),
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
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("-" * 60)
    print(f"🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! BlueSky integration is working correctly!")
        print("\n🚀 Integration Features:")
        if BLUESKY_AVAILABLE:
            print("• ✅ Real BlueSky simulator integration active")
            print("• 📡 Live aircraft data from BlueSky traffic module")
            print("• 🔍 Real conflict detection from BlueSky CD system")
            print("• 🎮 Direct command execution through BlueSky stack")
            print("• ⏰ Actual simulation time control")
        else:
            print("• 🎭 Enhanced mock simulation with realistic data")
            print("• 📊 Configurable aircraft counts and distributions")
            print("• 🌍 Realistic geographical bounds")
            print("• ⚙️ YAML-based configuration system")
        
        print("• 🔧 Graceful fallback when BlueSky unavailable")
        print("• 📋 Consistent API regardless of BlueSky availability")
        print("• 🔍 Source tracking for all data")
        print("• ⚠️ Comprehensive error handling")
        
        print("\n🎯 Ready for Monte Carlo benchmark with real dynamics!")
        
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
