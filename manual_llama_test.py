#!/usr/bin/env python3
"""
Manual Llama Testing Interface
=============================
Interactive script to manually test Llama with your own inputs for:
1. Conflict Detection
2. Resolution Generation  
3. Safety Assessment
4. Raw LLM Queries

This allows you to provide specific scenarios and see exactly what Llama responds with.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools.llm_prompt_engine import LLMPromptEngine
from llm_interface.llm_client import LLMClient


class ManualLlamaTest:
    """Interactive testing interface for Llama"""
    
    def __init__(self):
        self.llm_engine = LLMPromptEngine(model='llama3.1:8b', enable_function_calls=False)
        self.llm_client = LLMClient(model='llama3.1:8b')
        
    def main_menu(self):
        """Display main menu and handle user choices"""
        while True:
            print("\n" + "🤖" * 20)
            print("MANUAL LLAMA TESTING INTERFACE")
            print("🤖" * 20)
            print("\nChoose a test type:")
            print("1. 🚨 Conflict Detection Test")
            print("2. 💡 Resolution Generation Test")
            print("3. 🛡️ Safety Assessment Test")
            print("4. 💬 Raw LLM Query Test")
            print("5. 📊 Predefined Scenario Tests")
            print("6. 📈 LLM Statistics")
            print("0. 🚪 Exit")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                self.test_conflict_detection()
            elif choice == '2':
                self.test_resolution_generation()
            elif choice == '3':
                self.test_safety_assessment()
            elif choice == '4':
                self.test_raw_llm_query()
            elif choice == '5':
                self.test_predefined_scenarios()
            elif choice == '6':
                self.show_llm_statistics()
            else:
                print("❌ Invalid choice. Please try again.")
    
    def test_conflict_detection(self):
        """Manual conflict detection test"""
        print("\n🚨 CONFLICT DETECTION TEST")
        print("=" * 40)
        
        # Get user input for aircraft states
        aircraft_states = []
        
        print("Enter aircraft information (minimum 2 aircraft needed):")
        
        for i in range(2, 10):  # Allow up to 8 aircraft
            print(f"\n✈️ Aircraft {i-1} (or press Enter to stop):")
            
            aircraft_id = input(f"  Aircraft ID (e.g., AAL123): ").strip()
            if not aircraft_id:
                break
                
            try:
                lat = float(input(f"  Latitude (e.g., 52.3676): "))
                lon = float(input(f"  Longitude (e.g., 4.9041): "))
                alt = int(input(f"  Altitude (ft, e.g., 35000): "))
                hdg = int(input(f"  Heading (0-359, e.g., 270): "))
                spd = int(input(f"  Speed (kts, e.g., 450): "))
                vs = int(input(f"  Vertical Speed (fpm, e.g., 0): "))
                
                aircraft_states.append({
                    'id': aircraft_id,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt,
                    'hdg': hdg,
                    'spd': spd,
                    'vs': vs
                })
                
            except ValueError:
                print("❌ Invalid input. Aircraft skipped.")
                continue
        
        if len(aircraft_states) < 2:
            print("❌ Need at least 2 aircraft for conflict detection.")
            return
        
        # Get time horizon
        try:
            time_horizon = float(input("\nTime horizon (minutes, e.g., 5.0): ") or "5.0")
        except ValueError:
            time_horizon = 5.0
        
        # Run conflict detection
        print(f"\n🤖 Calling Llama with {len(aircraft_states)} aircraft...")
        print("📡 Please wait for LLM response...")
        
        start_time = time.time()
        try:
            result = self.llm_engine.detect_conflict_via_llm(aircraft_states, time_horizon)
            end_time = time.time()
            
            print(f"\n✅ Response received in {end_time - start_time:.2f} seconds")
            print("\n📋 LLAMA CONFLICT DETECTION RESULT:")
            print("-" * 40)
            
            for key, value in result.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def test_resolution_generation(self):
        """Manual resolution generation test"""
        print("\n💡 RESOLUTION GENERATION TEST")
        print("=" * 40)
        
        # Get conflict information
        print("Enter conflict scenario details:")
        
        aircraft_1_id = input("Aircraft 1 ID (e.g., AAL123): ").strip()
        aircraft_2_id = input("Aircraft 2 ID (e.g., UAL456): ").strip()
        
        try:
            time_to_conflict = float(input("Time to conflict (seconds, e.g., 120): "))
            closest_approach = float(input("Closest approach distance (NM, e.g., 2.5): "))
        except ValueError:
            print("❌ Invalid numeric input.")
            return
        
        conflict_type = input("Conflict type (convergent/head-on/overtaking): ").strip() or "convergent"
        urgency = input("Urgency level (low/medium/high/critical): ").strip() or "medium"
        
        # Aircraft 1 details
        print(f"\n✈️ {aircraft_1_id} Details:")
        try:
            ac1_lat = float(input("  Latitude: "))
            ac1_lon = float(input("  Longitude: "))
            ac1_alt = int(input("  Altitude (ft): "))
            ac1_hdg = int(input("  Heading (0-359): "))
            ac1_spd = int(input("  Speed (kts): "))
            ac1_type = input("  Aircraft type (e.g., B737): ").strip() or "B737"
        except ValueError:
            print("❌ Invalid input for Aircraft 1.")
            return
        
        # Aircraft 2 details
        print(f"\n✈️ {aircraft_2_id} Details:")
        try:
            ac2_lat = float(input("  Latitude: "))
            ac2_lon = float(input("  Longitude: "))
            ac2_alt = int(input("  Altitude (ft): "))
            ac2_hdg = int(input("  Heading (0-359): "))
            ac2_spd = int(input("  Speed (kts): "))
            ac2_type = input("  Aircraft type (e.g., A320): ").strip() or "A320"
        except ValueError:
            print("❌ Invalid input for Aircraft 2.")
            return
        
        # Environmental conditions
        print("\n🌤️ Environmental Conditions:")
        try:
            wind_dir = int(input("  Wind direction (0-359): ") or "270")
            wind_spd = int(input("  Wind speed (kts): ") or "15")
            visibility = input("  Visibility (e.g., 10+ km): ").strip() or "10+ km"
            weather = input("  Weather conditions (e.g., clear): ").strip() or "clear"
        except ValueError:
            wind_dir, wind_spd = 270, 15
            visibility, weather = "10+ km", "clear"
        
        # Build conflict info
        conflict_info = {
            'aircraft_1_id': aircraft_1_id,
            'aircraft_2_id': aircraft_2_id,
            'time_to_conflict': time_to_conflict,
            'closest_approach_distance': closest_approach,
            'conflict_type': conflict_type,
            'urgency_level': urgency,
            'aircraft_1': {
                'lat': ac1_lat, 'lon': ac1_lon, 'alt': ac1_alt,
                'hdg': ac1_hdg, 'spd': ac1_spd, 'type': ac1_type
            },
            'aircraft_2': {
                'lat': ac2_lat, 'lon': ac2_lon, 'alt': ac2_alt,
                'hdg': ac2_hdg, 'spd': ac2_spd, 'type': ac2_type
            },
            'environmental_conditions': {
                'wind_direction': wind_dir,
                'wind_speed': wind_spd,
                'visibility': visibility,
                'weather_conditions': weather
            }
        }
        
        # Generate resolution
        print(f"\n🤖 Calling Llama for conflict resolution...")
        print("📡 Please wait for LLM response...")
        
        start_time = time.time()
        try:
            resolution = self.llm_engine.get_conflict_resolution(conflict_info)
            end_time = time.time()
            
            print(f"\n✅ Response received in {end_time - start_time:.2f} seconds")
            print("\n📋 LLAMA RESOLUTION RESULT:")
            print("-" * 40)
            
            if resolution:
                print(f"  Command: {resolution}")
                print(f"  Source: Real Llama LLM")
            else:
                print("  No resolution generated")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def test_safety_assessment(self):
        """Manual safety assessment test"""
        print("\n🛡️ SAFETY ASSESSMENT TEST")
        print("=" * 40)
        
        # Get command to assess
        command = input("Enter command to assess (e.g., HDG AAL123 045): ").strip()
        if not command:
            print("❌ No command provided.")
            return
        
        # Get basic conflict context
        aircraft_1_id = input("Aircraft 1 ID: ").strip()
        aircraft_2_id = input("Aircraft 2 ID: ").strip()
        
        try:
            time_to_conflict = float(input("Time to conflict (seconds): "))
            closest_approach = float(input("Closest approach distance (NM): "))
        except ValueError:
            time_to_conflict, closest_approach = 120.0, 2.5
        
        conflict_info = {
            'aircraft_1_id': aircraft_1_id,
            'aircraft_2_id': aircraft_2_id,
            'time_to_conflict': time_to_conflict,
            'closest_approach_distance': closest_approach
        }
        
        # Assess safety
        print(f"\n🤖 Calling Llama for safety assessment...")
        print("📡 Please wait for LLM response...")
        
        start_time = time.time()
        try:
            safety_result = self.llm_engine.assess_resolution_safety(command, conflict_info)
            end_time = time.time()
            
            print(f"\n✅ Response received in {end_time - start_time:.2f} seconds")
            print("\n📋 LLAMA SAFETY ASSESSMENT:")
            print("-" * 40)
            
            for key, value in safety_result.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def test_raw_llm_query(self):
        """Test raw LLM queries"""
        print("\n💬 RAW LLM QUERY TEST")
        print("=" * 40)
        print("Enter your custom prompt for Llama (press Enter twice to finish):")
        
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        
        prompt = "\n".join(lines[:-1])  # Remove last empty line
        
        if not prompt.strip():
            print("❌ No prompt provided.")
            return
        
        print(f"\n🤖 Sending to Llama...")
        print("📡 Please wait for LLM response...")
        
        start_time = time.time()
        try:
            response = self.llm_client.ask(prompt, enable_function_calls=False)
            end_time = time.time()
            
            print(f"\n✅ Response received in {end_time - start_time:.2f} seconds")
            print("\n📋 LLAMA RAW RESPONSE:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def test_predefined_scenarios(self):
        """Test with predefined realistic scenarios"""
        print("\n📊 PREDEFINED SCENARIO TESTS")
        print("=" * 40)
        
        scenarios = {
            "1": {
                "name": "London Heathrow Approach Conflict",
                "description": "Two aircraft converging on LHR runway 27L",
                "aircraft_states": [
                    {'id': 'BAW123', 'lat': 51.4700, 'lon': -0.4543, 'alt': 3000, 'hdg': 270, 'spd': 180, 'vs': -500},
                    {'id': 'VIR456', 'lat': 51.4750, 'lon': -0.4200, 'alt': 3500, 'hdg': 225, 'spd': 190, 'vs': -600}
                ]
            },
            "2": {
                "name": "Amsterdam Terminal Area",
                "description": "Multiple aircraft in AMS terminal area",
                "aircraft_states": [
                    {'id': 'KLM789', 'lat': 52.3676, 'lon': 4.9041, 'alt': 35000, 'hdg': 95, 'spd': 465, 'vs': 0},
                    {'id': 'AFR234', 'lat': 52.3720, 'lon': 4.9180, 'alt': 35000, 'hdg': 275, 'spd': 455, 'vs': 0},
                    {'id': 'DLH567', 'lat': 52.3800, 'lon': 4.8900, 'alt': 36000, 'hdg': 180, 'spd': 470, 'vs': 0}
                ]
            },
            "3": {
                "name": "Oceanic Crossing",
                "description": "Trans-Atlantic crossing conflict",
                "aircraft_states": [
                    {'id': 'UAL890', 'lat': 45.0000, 'lon': -30.0000, 'alt': 39000, 'hdg': 45, 'spd': 485, 'vs': 0},
                    {'id': 'AAL345', 'lat': 45.2000, 'lon': -29.8000, 'alt': 39000, 'hdg': 225, 'spd': 480, 'vs': 0}
                ]
            }
        }
        
        print("Choose a predefined scenario:")
        for key, scenario in scenarios.items():
            print(f"{key}. {scenario['name']} - {scenario['description']}")
        
        choice = input("\nEnter scenario number (1-3): ").strip()
        
        if choice not in scenarios:
            print("❌ Invalid choice.")
            return
        
        scenario = scenarios[choice]
        aircraft_states = scenario['aircraft_states']
        
        print(f"\n🎯 Testing: {scenario['name']}")
        print(f"📄 Description: {scenario['description']}")
        print(f"✈️ Aircraft count: {len(aircraft_states)}")
        
        # Test conflict detection
        print(f"\n🤖 Running conflict detection...")
        start_time = time.time()
        try:
            result = self.llm_engine.detect_conflict_via_llm(aircraft_states, 5.0)
            end_time = time.time()
            
            print(f"✅ Detection completed in {end_time - start_time:.2f} seconds")
            print("\n📋 DETECTION RESULT:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            # If conflict detected, test resolution
            if result.get('conflict_detected', False) and result.get('aircraft_pairs'):
                print(f"\n💡 Generating resolution for detected conflict...")
                
                # Create conflict info for first pair
                pair = result['aircraft_pairs'][0]
                ac1_id, ac2_id = pair[0], pair[1]
                
                # Find aircraft data
                ac1_data = next((ac for ac in aircraft_states if ac['id'] == ac1_id), aircraft_states[0])
                ac2_data = next((ac for ac in aircraft_states if ac['id'] == ac2_id), aircraft_states[1])
                
                conflict_info = {
                    'aircraft_1_id': ac1_id,
                    'aircraft_2_id': ac2_id,
                    'time_to_conflict': result.get('time_to_conflict', [120])[0],
                    'closest_approach_distance': 2.5,
                    'conflict_type': 'convergent',
                    'urgency_level': result.get('priority', 'medium'),
                    'aircraft_1': ac1_data,
                    'aircraft_2': ac2_data,
                    'environmental_conditions': {
                        'wind_direction': 270,
                        'wind_speed': 15,
                        'visibility': '10+ km',
                        'weather_conditions': 'clear'
                    }
                }
                
                resolution = self.llm_engine.get_conflict_resolution(conflict_info)
                print(f"\n📋 RESOLUTION: {resolution}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def show_llm_statistics(self):
        """Show LLM usage statistics"""
        print("\n📈 LLM STATISTICS")
        print("=" * 40)
        
        try:
            count = self.llm_client.get_inference_count()
            total_time = self.llm_client.get_total_inference_time()
            avg_time = self.llm_client.get_average_inference_time()
            
            print(f"Total LLM calls: {count}")
            print(f"Total inference time: {total_time:.2f} seconds")
            print(f"Average response time: {avg_time:.2f} seconds")
            print(f"Model: {self.llm_client.model}")
            
            if count > 0:
                print("\n📊 Performance Analysis:")
                if avg_time < 1.0:
                    print("  🚀 Very Fast (< 1s average)")
                elif avg_time < 3.0:
                    print("  ⚡ Fast (1-3s average)")
                elif avg_time < 10.0:
                    print("  ✅ Normal (3-10s average)")
                else:
                    print("  🐌 Slow (> 10s average)")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")


def main():
    """Run the manual Llama testing interface"""
    print("🤖 Welcome to Manual Llama Testing!")
    print("This tool lets you test Llama with your own custom inputs.")
    print("All responses come directly from the Ollama LLM - no fake data!")
    
    tester = ManualLlamaTest()
    tester.main_menu()


if __name__ == "__main__":
    main()
