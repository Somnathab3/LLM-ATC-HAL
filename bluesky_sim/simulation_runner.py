# bluesky_sim/simulation_runner.py
import json
import logging
import os
import sys
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from typing import Optional

from bluesky import stack, traf

from llm_interface.filter_sort import get_llm_stats, select_best_solution
from solver.conflict_solver import ConflictSolver

# Set up file and console logging
log_filepath = os.path.join(project_root, "simulation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filepath}")


def initialize_bluesky() -> Optional[bool]:
    """Initialize BlueSky simulation environment."""
    try:
        stack.stack("IC")
        stack.stack("AREA EHAM")  # Set area around Amsterdam
        stack.stack("CDMETHOD SWARM")  # Use SWARM conflict detection
        stack.stack("CDSEP 5.0 1000")  # Set separation criteria (5 NM, 1000 ft)
        logger.info("BlueSky initialized successfully")
        return True
    except Exception as e:
        logger.exception(f"Failed to initialize BlueSky: {e}")
        return False


def detect_conflicts():
    """Detect conflicts using BlueSky's conflict detection system."""
    conflicts = []

    try:
        # Check if traffic and conflict detection are available
        if traf is not None and hasattr(traf, "cd") and traf.cd is not None:
            # Get conflict pairs from BlueSky
            if hasattr(traf.cd, "confpairs_all") and traf.cd.confpairs_all is not None:
                confpairs = traf.cd.confpairs_all
                logger.info(
                    f"Found {len(confpairs) if confpairs else 0} conflict pairs"
                )

                for i, pair in enumerate(confpairs or []):
                    ac1_idx, ac2_idx = pair
                    if ac1_idx < len(traf.id) and ac2_idx < len(traf.id):
                        conflict = {
                            "id1": traf.id[ac1_idx],
                            "id2": traf.id[ac2_idx],
                            "time": (
                                getattr(traf.cd, "tcpa", [120])[i]
                                if hasattr(traf.cd, "tcpa")
                                else 120
                            ),
                            "distance": (
                                getattr(traf.cd, "dcpa", [3.0])[i]
                                if hasattr(traf.cd, "dcpa")
                                else 3.0
                            ),
                            "lat1": traf.lat[ac1_idx] if hasattr(traf, "lat") else 52.3,
                            "lon1": traf.lon[ac1_idx] if hasattr(traf, "lon") else 4.8,
                            "lat2": traf.lat[ac2_idx] if hasattr(traf, "lat") else 52.4,
                            "lon2": traf.lon[ac2_idx] if hasattr(traf, "lon") else 4.6,
                            "alt1": (
                                traf.alt[ac1_idx] if hasattr(traf, "alt") else 35000
                            ),
                            "alt2": (
                                traf.alt[ac2_idx] if hasattr(traf, "alt") else 35000
                            ),
                        }
                        conflicts.append(conflict)

        # If no real conflicts detected, create mock conflicts for testing
        if not conflicts:
            logger.info(
                "No real conflicts detected, creating mock conflicts for testing"
            )
            conflicts = [
                {
                    "id1": "AC001",
                    "id2": "AC002",
                    "time": 120,
                    "distance": 4.5,
                    "lat1": 52.3,
                    "lon1": 4.8,
                    "alt1": 35000,
                    "lat2": 52.4,
                    "lon2": 4.6,
                    "alt2": 35000,
                    "mock": True,
                },
                {
                    "id1": "AC003",
                    "id2": "AC004",
                    "time": 180,
                    "distance": 3.2,
                    "lat1": 52.2,
                    "lon1": 4.9,
                    "alt1": 33000,
                    "lat2": 52.6,
                    "lon2": 4.9,
                    "alt2": 33000,
                    "mock": True,
                },
            ]

    except Exception as e:
        logger.exception(f"Error in conflict detection: {e}")
        # Return mock conflicts on error
        conflicts = [
            {
                "id1": "AC001",
                "id2": "AC002",
                "time": 120,
                "distance": 4.5,
                "mock": True,
                "error_fallback": True,
            },
        ]

    return conflicts


def run_simulation():
    """Main simulation execution function."""
    logger.info("Starting ATC hallucination test simulation")

    # Initialize BlueSky
    if not initialize_bluesky():
        logger.error("Failed to initialize BlueSky, continuing with mock data")

    scenarios = ["data/scenarios/standard.scn", "data/scenarios/edge_case.scn"]
    solver = ConflictSolver()

    total_scenarios = 0
    total_conflicts = 0
    total_resolutions = 0
    hallucination_events = []
    safety_margin_diffs = []

    for scenario in scenarios:
        logger.info(f"Processing scenario: {scenario}")
        total_scenarios += 1

        # Check if scenario file exists
        if not os.path.exists(scenario):
            logger.warning(f"Scenario file {scenario} not found. Skipping...")
            continue

        try:
            # Load scenario
            stack.stack("RESET")
            stack.stack(f"IC {scenario}")
            stack.stack("OP")
            stack.stack("TMAX 3600")  # 1 hour simulation

            # Run simulation for a short time to develop conflicts
            for _step in range(10):  # 10 simulation steps
                stack.stack("FF")
                time.sleep(0.1)  # Brief pause between steps

            # Detect conflicts
            conflicts = detect_conflicts()
            total_conflicts += len(conflicts)

            logger.info(f"Found {len(conflicts)} conflicts to resolve")

            for conflict in conflicts:
                try:
                    # Generate candidate solutions
                    candidates = solver.solve(conflict)

                    if not candidates:
                        logger.warning(
                            f"No candidates generated for conflict {conflict}"
                        )
                        continue

                    # Use LLM to select best solution
                    policies = [
                        "prefer minimal path deviation",
                        "avoid altitude changes",
                        "maintain safety margins",
                    ]
                    best_by_llm = select_best_solution(candidates, policies)
                    baseline_best = solver.score_best(candidates)

                    # Calculate safety margin difference
                    if best_by_llm and baseline_best:
                        llm_safety = best_by_llm.get("safety_score", 0.5)
                        baseline_safety = baseline_best.get("safety_score", 0.5)
                        safety_margin_diffs.append(llm_safety - baseline_safety)

                    # Check for hallucinations
                    if best_by_llm != baseline_best:
                        hallucination_events.append(
                            {
                                "conflict": conflict,
                                "llm_choice": best_by_llm,
                                "baseline_choice": baseline_best,
                                "scenario": scenario,
                            },
                        )

                    total_resolutions += 1

                    # Log results
                    log_data = {
                        "timestamp": time.time(),
                        "scenario": scenario,
                        "conflict": conflict,
                        "candidates": candidates,
                        "best_by_llm": best_by_llm,
                        "baseline_best": baseline_best,
                        "policies": policies,
                    }

                    # Log as JSON line to our file handler
                    json_log_entry = json.dumps(log_data, default=str)
                    with open(log_filepath, "a") as f:
                        f.write(json_log_entry + "\n")
                    logger.info(f"Logged conflict resolution data to {log_filepath}")
                    logger.info(
                        f"Processed conflict between {conflict.get('id1', 'Unknown')} and {conflict.get('id2', 'Unknown')}",
                    )

                except Exception as e:
                    logger.exception(f"Error processing conflict {conflict}: {e}")

        except Exception as e:
            logger.exception(f"Error processing scenario {scenario}: {e}")

    # Get LLM statistics
    llm_stats = get_llm_stats()

    # Calculate final metrics
    hallucination_rate = len(hallucination_events) / max(total_resolutions, 1)
    avg_safety_margin_diff = (
        sum(safety_margin_diffs) / max(len(safety_margin_diffs), 1)
        if safety_margin_diffs
        else 0
    )

    # Generate summary report
    summary = {
        "scenarios_run": total_scenarios,
        "conflicts_detected": total_conflicts,
        "resolutions_attempted": total_resolutions,
        "hallucination_rate": hallucination_rate,
        "avg_safety_margin_diff": avg_safety_margin_diff,
        "total_hallucination_events": len(hallucination_events),
        "llm_stats": llm_stats,
    }

    logger.info(f"Simulation completed. Summary: {summary}")

    return summary


if __name__ == "__main__":
    summary = run_simulation()
