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
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_atc.tools.llm_prompt_engine import LLMPromptEngine
from llm_interface.llm_client import LLMClient


class ManualLlamaTest:
    """Interactive testing interface for Llama"""

    def __init__(self) -> None:
        self.llm_engine = LLMPromptEngine(model="llama3.1:8b", enable_function_calls=False)
        self.llm_client = LLMClient(model="llama3.1:8b")

    def main_menu(self) -> None:
        """Display main menu and handle user choices"""
        while True:

            choice = input("\nEnter your choice (0-6): ").strip()

            if choice == "0":
                break
            if choice == "1":
                self.test_conflict_detection()
            elif choice == "2":
                self.test_resolution_generation()
            elif choice == "3":
                self.test_safety_assessment()
            elif choice == "4":
                self.test_raw_llm_query()
            elif choice == "5":
                self.test_predefined_scenarios()
            elif choice == "6":
                self.show_llm_statistics()
            else:
                pass

    def test_conflict_detection(self) -> None:
        """Manual conflict detection test"""

        # Get user input for aircraft states
        aircraft_states = []

        for _i in range(2, 10):  # Allow up to 8 aircraft

            aircraft_id = input("  Aircraft ID (e.g., AAL123): ").strip()
            if not aircraft_id:
                break

            try:
                lat = float(input("  Latitude (e.g., 52.3676): "))
                lon = float(input("  Longitude (e.g., 4.9041): "))
                alt = int(input("  Altitude (ft, e.g., 35000): "))
                hdg = int(input("  Heading (0-359, e.g., 270): "))
                spd = int(input("  Speed (kts, e.g., 450): "))
                vs = int(input("  Vertical Speed (fpm, e.g., 0): "))

                aircraft_states.append(
                    {
                        "id": aircraft_id,
                        "lat": lat,
                        "lon": lon,
                        "alt": alt,
                        "hdg": hdg,
                        "spd": spd,
                        "vs": vs,
                    },
                )

            except ValueError:
                continue

        if len(aircraft_states) < 2:
            return

        # Get time horizon
        try:
            time_horizon = float(input("\nTime horizon (minutes, e.g., 5.0): ") or "5.0")
        except ValueError:
            time_horizon = 5.0

        # Run conflict detection

        time.time()
        try:
            result = self.llm_engine.detect_conflict_via_llm(aircraft_states, time_horizon)
            time.time()

            for _key, _value in result.items():
                pass

        except Exception:
            pass

    def test_resolution_generation(self) -> None:
        """Manual resolution generation test"""

        # Get conflict information

        aircraft_1_id = input("Aircraft 1 ID (e.g., AAL123): ").strip()
        aircraft_2_id = input("Aircraft 2 ID (e.g., UAL456): ").strip()

        try:
            time_to_conflict = float(input("Time to conflict (seconds, e.g., 120): "))
            closest_approach = float(input("Closest approach distance (NM, e.g., 2.5): "))
        except ValueError:
            return

        conflict_type = (
            input("Conflict type (convergent/head-on/overtaking): ").strip() or "convergent"
        )
        urgency = input("Urgency level (low/medium/high/critical): ").strip() or "medium"

        # Aircraft 1 details
        try:
            ac1_lat = float(input("  Latitude: "))
            ac1_lon = float(input("  Longitude: "))
            ac1_alt = int(input("  Altitude (ft): "))
            ac1_hdg = int(input("  Heading (0-359): "))
            ac1_spd = int(input("  Speed (kts): "))
            ac1_type = input("  Aircraft type (e.g., B737): ").strip() or "B737"
        except ValueError:
            return

        # Aircraft 2 details
        try:
            ac2_lat = float(input("  Latitude: "))
            ac2_lon = float(input("  Longitude: "))
            ac2_alt = int(input("  Altitude (ft): "))
            ac2_hdg = int(input("  Heading (0-359): "))
            ac2_spd = int(input("  Speed (kts): "))
            ac2_type = input("  Aircraft type (e.g., A320): ").strip() or "A320"
        except ValueError:
            return

        # Environmental conditions
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
            "aircraft_1_id": aircraft_1_id,
            "aircraft_2_id": aircraft_2_id,
            "time_to_conflict": time_to_conflict,
            "closest_approach_distance": closest_approach,
            "conflict_type": conflict_type,
            "urgency_level": urgency,
            "aircraft_1": {
                "lat": ac1_lat,
                "lon": ac1_lon,
                "alt": ac1_alt,
                "hdg": ac1_hdg,
                "spd": ac1_spd,
                "type": ac1_type,
            },
            "aircraft_2": {
                "lat": ac2_lat,
                "lon": ac2_lon,
                "alt": ac2_alt,
                "hdg": ac2_hdg,
                "spd": ac2_spd,
                "type": ac2_type,
            },
            "environmental_conditions": {
                "wind_direction": wind_dir,
                "wind_speed": wind_spd,
                "visibility": visibility,
                "weather_conditions": weather,
            },
        }

        # Generate resolution

        time.time()
        try:
            resolution = self.llm_engine.get_conflict_resolution(conflict_info)
            time.time()

            if resolution:
                pass
            else:
                pass

        except Exception:
            pass

    def test_safety_assessment(self) -> None:
        """Manual safety assessment test"""

        # Get command to assess
        command = input("Enter command to assess (e.g., HDG AAL123 045): ").strip()
        if not command:
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
            "aircraft_1_id": aircraft_1_id,
            "aircraft_2_id": aircraft_2_id,
            "time_to_conflict": time_to_conflict,
            "closest_approach_distance": closest_approach,
        }

        # Assess safety

        time.time()
        try:
            safety_result = self.llm_engine.assess_resolution_safety(command, conflict_info)
            time.time()

            for _key, _value in safety_result.items():
                pass

        except Exception:
            pass

    def test_raw_llm_query(self) -> None:
        """Test raw LLM queries"""

        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        prompt = "\n".join(lines[:-1])  # Remove last empty line

        if not prompt.strip():
            return

        time.time()
        try:
            self.llm_client.ask(prompt, enable_function_calls=False)
            time.time()

        except Exception:
            pass

    def test_predefined_scenarios(self) -> None:
        """Test with predefined realistic scenarios"""

        scenarios = {
            "1": {
                "name": "London Heathrow Approach Conflict",
                "description": "Two aircraft converging on LHR runway 27L",
                "aircraft_states": [
                    {
                        "id": "BAW123",
                        "lat": 51.4700,
                        "lon": -0.4543,
                        "alt": 3000,
                        "hdg": 270,
                        "spd": 180,
                        "vs": -500,
                    },
                    {
                        "id": "VIR456",
                        "lat": 51.4750,
                        "lon": -0.4200,
                        "alt": 3500,
                        "hdg": 225,
                        "spd": 190,
                        "vs": -600,
                    },
                ],
            },
            "2": {
                "name": "Amsterdam Terminal Area",
                "description": "Multiple aircraft in AMS terminal area",
                "aircraft_states": [
                    {
                        "id": "KLM789",
                        "lat": 52.3676,
                        "lon": 4.9041,
                        "alt": 35000,
                        "hdg": 95,
                        "spd": 465,
                        "vs": 0,
                    },
                    {
                        "id": "AFR234",
                        "lat": 52.3720,
                        "lon": 4.9180,
                        "alt": 35000,
                        "hdg": 275,
                        "spd": 455,
                        "vs": 0,
                    },
                    {
                        "id": "DLH567",
                        "lat": 52.3800,
                        "lon": 4.8900,
                        "alt": 36000,
                        "hdg": 180,
                        "spd": 470,
                        "vs": 0,
                    },
                ],
            },
            "3": {
                "name": "Oceanic Crossing",
                "description": "Trans-Atlantic crossing conflict",
                "aircraft_states": [
                    {
                        "id": "UAL890",
                        "lat": 45.0000,
                        "lon": -30.0000,
                        "alt": 39000,
                        "hdg": 45,
                        "spd": 485,
                        "vs": 0,
                    },
                    {
                        "id": "AAL345",
                        "lat": 45.2000,
                        "lon": -29.8000,
                        "alt": 39000,
                        "hdg": 225,
                        "spd": 480,
                        "vs": 0,
                    },
                ],
            },
        }

        for _key, scenario in scenarios.items():
            pass

        choice = input("\nEnter scenario number (1-3): ").strip()

        if choice not in scenarios:
            return

        scenario = scenarios[choice]
        aircraft_states = scenario["aircraft_states"]

        # Test conflict detection
        time.time()
        try:
            result = self.llm_engine.detect_conflict_via_llm(aircraft_states, 5.0)
            time.time()

            for _key, _value in result.items():
                pass

            # If conflict detected, test resolution
            if result.get("conflict_detected", False) and result.get("aircraft_pairs"):

                # Create conflict info for first pair
                pair = result["aircraft_pairs"][0]
                ac1_id, ac2_id = pair[0], pair[1]

                # Find aircraft data
                ac1_data = next(
                    (ac for ac in aircraft_states if ac["id"] == ac1_id), aircraft_states[0],
                )
                ac2_data = next(
                    (ac for ac in aircraft_states if ac["id"] == ac2_id), aircraft_states[1],
                )

                conflict_info = {
                    "aircraft_1_id": ac1_id,
                    "aircraft_2_id": ac2_id,
                    "time_to_conflict": result.get("time_to_conflict", [120])[0],
                    "closest_approach_distance": 2.5,
                    "conflict_type": "convergent",
                    "urgency_level": result.get("priority", "medium"),
                    "aircraft_1": ac1_data,
                    "aircraft_2": ac2_data,
                    "environmental_conditions": {
                        "wind_direction": 270,
                        "wind_speed": 15,
                        "visibility": "10+ km",
                        "weather_conditions": "clear",
                    },
                }

                self.llm_engine.get_conflict_resolution(conflict_info)

        except Exception:
            pass

    def show_llm_statistics(self) -> None:
        """Show LLM usage statistics"""

        try:
            count = self.llm_client.get_inference_count()
            self.llm_client.get_total_inference_time()
            avg_time = self.llm_client.get_average_inference_time()

            if count > 0:
                if avg_time < 1.0 or avg_time < 3.0 or avg_time < 10.0:
                    pass
                else:
                    pass

        except Exception:
            pass


def main() -> None:
    """Run the manual Llama testing interface"""

    tester = ManualLlamaTest()
    tester.main_menu()


if __name__ == "__main__":
    main()
