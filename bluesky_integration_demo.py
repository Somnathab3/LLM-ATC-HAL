#!/usr/bin/env python3
"""
Example script demonstrating the new BlueSky integration
This shows how to use the enhanced bluesky_tools.py with real BlueSky simulator integration.
"""

import sys
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
    BLUESKY_AVAILABLE
)


def main():
    """Demonstrate BlueSky integration capabilities"""
    print("🛩️ BlueSky Integration Demo")
    print("=" * 40)
    
    # Show integration status
    if BLUESKY_AVAILABLE:
        print("✅ BlueSky simulator detected - using real integration")
        print("🔗 Data will come from actual BlueSky simulation")
    else:
        print("⚠️ BlueSky not available - using enhanced mock simulation")
        print("🎭 Data will be realistic but simulated")
    
    print("\n1️⃣ Resetting simulation...")
    reset_result = reset_simulation()
    print(f"   Status: {reset_result.get('status')} (source: {reset_result.get('source')})")
    
    print("\n2️⃣ Creating test aircraft...")
    # Create some test aircraft
    test_aircraft = [
        "CRE KLM123,B738,52.3,4.8,90,35000,250",
        "CRE AFR456,A320,52.35,4.85,270,35000,260", 
        "CRE DLH789,B777,52.32,4.82,45,36000,280"
    ]
    
    for cmd in test_aircraft:
        result = send_command(cmd)
        callsign = cmd.split(',')[1]
        print(f"   Created {callsign}: {result.get('status')} (source: {result.get('source')})")
    
    print("\n3️⃣ Stepping simulation forward...")
    step_result = step_simulation(minutes=2.0, dtmult=1.0)
    print(f"   Advanced {step_result.get('minutes_advanced', 0)} minutes")
    print(f"   Status: {step_result.get('status')} (source: {step_result.get('source')})")
    
    print("\n4️⃣ Getting aircraft information...")
    aircraft_data = get_all_aircraft_info()
    aircraft_count = aircraft_data.get('total_aircraft', 0)
    data_source = aircraft_data.get('source', 'unknown')
    
    print(f"   Found {aircraft_count} aircraft (source: {data_source})")
    
    if aircraft_count > 0:
        aircraft_dict = aircraft_data.get('aircraft', {})
        print("   Aircraft positions:")
        for acid, data in list(aircraft_dict.items())[:5]:  # Show first 5
            print(f"      {acid}: {data['lat']:.3f}°N, {data['lon']:.3f}°E, {data['alt']:.0f}ft")
    
    print("\n5️⃣ Checking for conflicts...")
    conflict_data = get_conflict_info()
    conflict_count = conflict_data.get('total_conflicts', 0)
    data_source = conflict_data.get('source', 'unknown')
    
    print(f"   Found {conflict_count} conflicts (source: {data_source})")
    
    if conflict_count > 0:
        conflicts = conflict_data.get('conflicts', [])
        for conflict in conflicts[:3]:  # Show first 3
            print(f"      {conflict['conflict_id']}: {conflict['aircraft_1']} ↔ {conflict['aircraft_2']}")
            print(f"         Separation: {conflict['horizontal_separation']:.1f}nm H, {conflict['vertical_separation']:.0f}ft V")
    
    print("\n6️⃣ Distance calculation example...")
    if aircraft_count >= 2:
        aircraft_ids = list(aircraft_data.get('aircraft', {}).keys())
        ac1, ac2 = aircraft_ids[0], aircraft_ids[1]
        
        try:
            distance_result = get_distance(ac1, ac2)
            print(f"   Distance between {ac1} and {ac2}:")
            print(f"      Horizontal: {distance_result['horizontal_nm']:.2f} nm")
            print(f"      Vertical: {distance_result['vertical_ft']:.0f} ft")
            print(f"      3D: {distance_result['total_3d_nm']:.2f} nm")
            print(f"      Source: {distance_result.get('source', 'calculated')}")
        except Exception as e:
            print(f"   Error calculating distance: {e}")
    else:
        print("   Not enough aircraft for distance calculation")
    
    print("\n✅ Demo completed!")
    
    if BLUESKY_AVAILABLE:
        print("\n🚀 Key Benefits of Real BlueSky Integration:")
        print("   • Live aircraft dynamics and physics")
        print("   • Real conflict detection algorithms")
        print("   • Accurate flight models and performance")
        print("   • Realistic weather and environmental effects")
        print("   • True simulation time progression")
        print("\n📈 This enables realistic Monte Carlo benchmarking!")
    else:
        print("\n🎭 Enhanced Mock Simulation Features:")
        print("   • Configurable aircraft distributions")
        print("   • Realistic geographical bounds")
        print("   • Consistent API for testing")
        print("   • Graceful fallback behavior")
        print("\n💡 Install BlueSky for full real-time simulation!")


if __name__ == "__main__":
    main()
