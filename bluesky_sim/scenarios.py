from pathlib import Path


def generate_standard_scenario() -> None:
    """Generate a standard traffic scenario with moderate traffic density and convergent paths."""
    Path("data/scenarios").mkdir(parents=True, exist_ok=True)
    scenario_file = Path("data/scenarios/standard.scn")
    scenario_file.write_text(
        "00:00:00.00>CRE AC001 B737 52.3 4.8 090 35000 350\n"
        "00:00:00.00>CRE AC002 B737 52.4 4.6 270 35000 350\n"
        "00:01:00.00>AC001 HDG 270\n"  # Turn towards AC002
        "00:01:00.00>AC002 HDG 090\n"  # Turn towards AC001
        "00:05:00.00>HOLD\n",  # Hold simulation for analysis
    )


def generate_edge_case_scenario() -> None:
    """Generate an edge-case scenario with high-density traffic and multiple potential conflicts."""
    Path("data/scenarios").mkdir(parents=True, exist_ok=True)
    scenario_file = Path("data/scenarios/edge_case.scn")
    scenario_content = (
        "00:00:00.00>CRE AC003 A320 52.2 4.9 180 33000 320\n"
        "00:00:00.00>CRE AC004 A320 52.6 4.9 360 33000 320\n"
        "00:00:00.00>CRE AC005 B777 52.4 4.7 090 35000 480\n"
        "00:00:30.00>AC003 HDG 360\n"
        "00:00:30.00>AC004 HDG 180\n"
        "00:01:00.00>AC005 HDG 270\n"
        "00:10:00.00>HOLD\n"
    )
    scenario_file.write_text(scenario_content)


def generate_all_scenarios() -> list[str]:
    """Generate all test scenarios."""
    generate_standard_scenario()
    generate_edge_case_scenario()
    return ["data/scenarios/standard.scn", "data/scenarios/edge_case.scn"]


if __name__ == "__main__":
    scenarios = generate_all_scenarios()
