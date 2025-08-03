# Repository Analysis Summary

Generated: 2025-08-03 19:51:08

## Current Repository State

- **Total Functions**: 437
- **Total Imports**: 217
- **Total Classes**: 67
- **Modified Files**: 2

## Changes from Previous Commit

- **Removed Functions**: 0
- **Added Functions**: 437
- **Removed Imports**: 0
- **Added Imports**: 217
- **Removed Classes**: 0
- **Added Classes**: 67

## Function Dependencies Table

| File | Function | Arguments | Intra-repo Dependencies | Import Dependencies |
|------|----------|-----------|------------------------|---------------------|
| `llm_atc\agents\executor.py` | `__init__` | `self, command_sender` | ExecutionResult | None |
| `llm_atc\agents\executor.py` | `send_plan` | `self, plan` | ActionPlan, ExecutionResult, ExecutionStatus | None |
| `llm_atc\agents\executor.py` | `_send_command` | `self, command` | None | None |
| `llm_atc\agents\executor.py` | `_simulate_command_execution` | `self, command` | None | None |
| `llm_atc\agents\executor.py` | `cancel_execution` | `self, execution_id` | ExecutionStatus | None |
| `llm_atc\agents\executor.py` | `get_execution_status` | `self, execution_id` | ExecutionStatus | None |
| `llm_atc\agents\executor.py` | `get_active_executions` | `self` | ExecutionResult | None |
| `llm_atc\agents\executor.py` | `get_execution_history` | `self` | ExecutionResult | None |
| `llm_atc\agents\executor.py` | `get_execution_metrics` | `self` | ExecutionStatus | None |
| `llm_atc\agents\executor.py` | `set_command_sender` | `self, command_sender` | None | None |
| `llm_atc\agents\planner.py` | `__init__` | `self, llm_client` | ActionPlan, ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `assess_conflict` | `self, aircraft_info` | ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `generate_action_plan` | `self, assessment` | ActionPlan, ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `_detect_proximity_conflicts` | `self, aircraft_data` | aircraft_list | None |
| `llm_atc\agents\planner.py` | `_calculate_separation` | `self, ac1_data, ac2_data` | None | None |
| `llm_atc\agents\planner.py` | `_assess_severity` | `self, separation` | None | None |
| `llm_atc\agents\planner.py` | `_estimate_time_to_conflict` | `self, _ac1_data, _ac2_data` | None | None |
| `llm_atc\agents\planner.py` | `_prioritize_conflicts` | `self, conflicts` | None | None |
| `llm_atc\agents\planner.py` | `_generate_assessment` | `self, conflict, aircraft_data` | ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `_determine_recommended_action` | `self, conflict, _aircraft_data` | PlanType | None |
| `llm_atc\agents\planner.py` | `_generate_reasoning` | `self, conflict, action` | PlanType | None |
| `llm_atc\agents\planner.py` | `_generate_commands` | `self, assessment` | ConflictAssessment, PlanType | None |
| `llm_atc\agents\planner.py` | `_calculate_expected_outcome` | `self, _assessment, _commands` | ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `_calculate_priority` | `self, assessment` | ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `get_assessment_history` | `self` | ConflictAssessment | None |
| `llm_atc\agents\planner.py` | `get_plan_history` | `self` | ActionPlan | None |
| `llm_atc\agents\scratchpad.py` | `__init__` | `self, session_id` | ReasoningStep, SessionSummary | None |
| `llm_atc\agents\scratchpad.py` | `log_step` | `self, step_data` | ReasoningStep, StepType | None |
| `llm_atc\agents\scratchpad.py` | `log_assessment_step` | `self, assessment` | ConflictAssessment | None |
| `llm_atc\agents\scratchpad.py` | `log_planning_step` | `self, plan` | ActionPlan | None |
| `llm_atc\agents\scratchpad.py` | `log_execution_step` | `self, execution` | ExecutionResult | None |
| `llm_atc\agents\scratchpad.py` | `log_verification_step` | `self, verification` | VerificationResult | None |
| `llm_atc\agents\scratchpad.py` | `log_error_step` | `self, error_msg, error_data` | None | None |
| `llm_atc\agents\scratchpad.py` | `log_monitoring_step` | `self, monitoring_data` | None | None |
| `llm_atc\agents\scratchpad.py` | `get_history` | `self` | None | None |
| `llm_atc\agents\scratchpad.py` | `get_step_by_id` | `self, step_id` | ReasoningStep | None |
| `llm_atc\agents\scratchpad.py` | `get_steps_by_type` | `self, step_type` | ReasoningStep, StepType | None |
| `llm_atc\agents\scratchpad.py` | `get_recent_steps` | `self, count` | ReasoningStep | None |
| `llm_atc\agents\scratchpad.py` | `complete_session` | `self, success, final_status` | SessionSummary, StepType | None |
| `llm_atc\agents\scratchpad.py` | `start_new_session` | `self, session_id` | None | None |
| `llm_atc\agents\scratchpad.py` | `_generate_session_summary` | `self` | None | None |
| `llm_atc\agents\scratchpad.py` | `_calculate_average_confidence` | `self` | None | None |
| `llm_atc\agents\scratchpad.py` | `_extract_key_decisions` | `self` | StepType | None |
| `llm_atc\agents\scratchpad.py` | `_extract_lessons_learned` | `self` | StepType | None |
| `llm_atc\agents\scratchpad.py` | `export_session_data` | `self, format` | None | None |
| `llm_atc\agents\scratchpad.py` | `set_session_metadata` | `self, metadata` | None | None |
| `llm_atc\agents\scratchpad.py` | `get_session_metrics` | `self` | StepType | None |
| `llm_atc\agents\verifier.py` | `__init__` | `self, safety_thresholds` | VerificationResult | None |
| `llm_atc\agents\verifier.py` | `check` | `self, execution_result, timeout_seconds` | ExecutionResult, VerificationResult, VerificationStatus | None |
| `llm_atc\agents\verifier.py` | `_check_execution_status` | `self, execution, verification` | ExecutionResult, ExecutionStatus, VerificationResult | None |
| `llm_atc\agents\verifier.py` | `_check_execution_timing` | `self, execution, verification` | ExecutionResult, VerificationResult | None |
| `llm_atc\agents\verifier.py` | `_check_command_success_rate` | `self, execution, verification` | ExecutionResult, VerificationResult | None |
| `llm_atc\agents\verifier.py` | `_check_safety_compliance` | `self, execution, verification` | ExecutionResult, VerificationResult | None |
| `llm_atc\agents\verifier.py` | `_check_response_validity` | `self, execution, verification` | ExecutionResult, VerificationResult | None |
| `llm_atc\agents\verifier.py` | `_is_unsafe_command` | `self, command` | None | None |
| `llm_atc\agents\verifier.py` | `_is_valid_response` | `self, response` | None | None |
| `llm_atc\agents\verifier.py` | `_calculate_safety_score` | `self, verification` | VerificationResult | None |
| `llm_atc\agents\verifier.py` | `_calculate_confidence` | `self, verification` | VerificationResult | None |
| `llm_atc\agents\verifier.py` | `get_verification_history` | `self` | VerificationResult | None |
| `llm_atc\agents\verifier.py` | `get_verification_metrics` | `self` | VerificationStatus | None |
| `llm_atc\agents\verifier.py` | `update_safety_thresholds` | `self, new_thresholds` | None | None |
| `llm_atc\memory\experience_integrator.py` | `__init__` | `self, replay_store` | EnhancedHallucinationDetector, SafetyMarginQuantifier, VectorReplayStore | None |
| `llm_atc\memory\experience_integrator.py` | `process_conflict_resolution` | `self, scenario_context, conflict_geometry, environmental_conditions, llm_decision, baseline_decision` | None | None |
| `llm_atc\memory\experience_integrator.py` | `_find_relevant_experiences` | `self, scenario_context, conflict_geometry, environmental_conditions` | ConflictExperience, SimilarityResult | None |
| `llm_atc\memory\experience_integrator.py` | `_extract_lessons` | `self, similar_experiences` | SimilarityResult | None |
| `llm_atc\memory\experience_integrator.py` | `_check_hallucination_patterns` | `self, scenario_context, environmental_conditions, similar_experiences` | SimilarityResult | None |
| `llm_atc\memory\experience_integrator.py` | `_enhance_decision_with_experience` | `self, llm_decision, baseline_decision, similar_experiences` | SimilarityResult | None |
| `llm_atc\memory\experience_integrator.py` | `record_resolution_outcome` | `self, scenario_context, conflict_geometry, environmental_conditions, llm_decision, baseline_decision, actual_outcome, safety_metrics, hallucination_result, controller_override, lessons_learned` | ConflictExperience | None |
| `llm_atc\memory\experience_integrator.py` | `get_experience_summary` | `self` | None | None |
| `llm_atc\memory\experience_integrator.py` | `_generate_learning_insights` | `self, stats, patterns` | None | None |
| `llm_atc\memory\experience_integrator.py` | `store_experience` | `self, experience_data` | ConflictExperience | None |
| `llm_atc\memory\replay_store.py` | `__post_init__` | `self` | None | None |
| `llm_atc\memory\replay_store.py` | `__init__` | `self, storage_dir` | None | None |
| `llm_atc\memory\replay_store.py` | `store_experience` | `self, experience` | ConflictExperience | None |
| `llm_atc\memory\replay_store.py` | `retrieve_experience` | `self, conflict_desc, conflict_type, num_ac, k` | None | None |
| `llm_atc\memory\replay_store.py` | `get_all_experiences` | `self, conflict_type, num_ac, limit` | None | None |
| `llm_atc\memory\replay_store.py` | `get_stats` | `self` | None | None |
| `llm_atc\memory\replay_store.py` | `delete_experience` | `self, experience_id` | None | None |
| `llm_atc\memory\replay_store.py` | `clear_all` | `self` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `calc_separation_margin` | `trajectories` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `calc_efficiency_penalty` | `planned, executed` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `__init__` | `self` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `read_results_file` | `self, file_path` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_read_json_results` | `self, file_path` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `compute_false_positive_negative_rates` | `self, results_df` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_conflicts_to_set` | `self, conflicts` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `compute_success_rates_by_scenario` | `self, results_df` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `compute_success_rates_by_group` | `self, results_df, group_cols` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `compute_average_separation_margins` | `self, results_df` | calc_separation_margin | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `compute_efficiency_penalties` | `self, results_df` | calc_efficiency_penalty | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `generate_report` | `self, results_df, aggregated_metrics, output_file` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_generate_executive_summary` | `self, metrics` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_assess_detection_performance` | `self, detection` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_assess_safety_margins` | `self, margins` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_assess_efficiency_performance` | `self, efficiency` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_format_grouped_success_table` | `self, grouped_df` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_format_distribution_shift_analysis` | `self, shift_analysis` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_generate_recommendations` | `self, metrics` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `aggregate_monte_carlo_metrics` | `self, results_df` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_analyze_distribution_shift_performance` | `self, results_df` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_create_empty_aggregated_metrics` | `self` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `__init__` | `self` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `create_performance_summary_charts` | `self, aggregated_metrics, output_dir` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `create_distribution_shift_plots` | `self, aggregated_metrics, output_dir` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_create_success_rate_chart` | `self, success_data, save_path` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_create_detection_performance_chart` | `self, detection_data, save_path` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_create_safety_margins_chart` | `self, margins_data, save_path` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `_create_shift_performance_scatter` | `self, shift_data, save_path` | None | None |
| `llm_atc\metrics\monte_carlo_analysis.py` | `analyze_monte_carlo_results` | `results_file, output_dir` | MonteCarloResultsAnalyzer, MonteCarloVisualizer | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `__init__` | `self` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `calculate_safety_margins` | `self, conflict_geometry, resolution_maneuver, environmental_conditions` | ConflictGeometry, SafetyMargin | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_apply_resolution_maneuver` | `self, geometry, maneuver` | ConflictGeometry | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_predict_position` | `self, position, velocity, time` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_closest_approach` | `self, pos1, pos2, vel1, vel2` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_horizontal_margin` | `self, geometry` | ConflictGeometry | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_vertical_margin` | `self, geometry` | ConflictGeometry | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_temporal_margin` | `self, geometry` | ConflictGeometry | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_effective_margin` | `self, h_margin, v_margin, t_margin` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_total_uncertainty` | `self, environmental_conditions` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_calculate_baseline_margin` | `self, geometry` | ConflictGeometry | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_determine_safety_level` | `self, effective_margin` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `_create_default_safety_margin` | `self` | SafetyMargin | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `__init__` | `self` | SafetyMarginQuantifier | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `add_conflict_resolution` | `self, conflict_id, geometry, llm_resolution, baseline_resolution, environmental_conditions` | ConflictGeometry | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `generate_safety_summary` | `self` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `export_detailed_metrics` | `self, filepath` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `calc_separation_margin` | `trajectories` | SeparationStandard | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `calc_efficiency_penalty` | `planned_path, executed_path` | calculate_path_distance | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `calculate_path_distance` | `path` | None | None |
| `llm_atc\metrics\safety_margin_quantifier.py` | `count_interventions` | `commands` | None | None |
| `llm_atc\metrics\__init__.py` | `analyze_hallucinations_in_log` | `_log_file` | None | None |
| `llm_atc\metrics\__init__.py` | `compute_metrics` | `log_file` | analyze_hallucinations_in_log, create_empty_metrics | None |
| `llm_atc\metrics\__init__.py` | `create_empty_metrics` | `` | None | None |
| `llm_atc\metrics\__init__.py` | `print_metrics_summary` | `metrics` | None | None |
| `llm_atc\metrics\__init__.py` | `calc_fp_fn` | `pred_conflicts, gt_conflicts` | None | None |
| `llm_atc\metrics\__init__.py` | `calc_path_extra` | `actual_traj, original_traj` | calc_trajectory_distance | None |
| `llm_atc\metrics\__init__.py` | `calc_trajectory_distance` | `traj` | None | None |
| `llm_atc\metrics\__init__.py` | `aggregate_thesis_metrics` | `results_dir` | compute_metrics, create_empty_metrics | None |
| `llm_atc\metrics\__init__.py` | `plot_metrics_comparison` | `llm_metrics, baseline_metrics, save_path` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `__init__` | `self, config_path` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_find_config_file` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_create_default_config` | `self, path` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_load_config` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_get_default_config` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `get` | `self, key_path, default` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `__init__` | `self, strict_mode` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_initialize_bluesky` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_setup_simulation` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_test_network_connection` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `is_available` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `get_aircraft_data` | `self` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `get_conflict_data` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_calculate_horizontal_separation` | `self, ac1_idx, ac2_idx` | haversine_distance | None |
| `llm_atc\tools\bluesky_tools.py` | `_assess_conflict_severity` | `self, h_sep, v_sep` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `send_bluesky_command` | `self, command` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `step_simulation_real` | `self, minutes, dtmult` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `reset_simulation_real` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_get_mock_aircraft_data` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_get_mock_conflict_data` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_simulate_command_execution` | `self, command` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_simulate_step` | `self, minutes, dtmult` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `_simulate_reset` | `self` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `set_strict_mode` | `enabled` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `haversine_distance` | `lat1, lon1, lat2, lon2` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `get_all_aircraft_info` | `` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `get_conflict_info` | `` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `continue_monitoring` | `` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `send_command` | `command` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `search_experience_library` | `scenario_type, similarity_threshold` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `get_weather_info` | `lat, lon` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `get_airspace_info` | `` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `get_distance` | `aircraft_id1, aircraft_id2` | BlueSkyToolsError, get_all_aircraft_info, haversine_distance | None |
| `llm_atc\tools\bluesky_tools.py` | `_haversine_distance` | `lat1, lon1, lat2, lon2` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `step_simulation` | `minutes, dtmult` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `reset_simulation` | `` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `get_minimum_separation` | `` | None | None |
| `llm_atc\tools\bluesky_tools.py` | `check_separation_violation` | `aircraft_id1, aircraft_id2` | get_distance, get_minimum_separation | None |
| `llm_atc\tools\bluesky_tools.py` | `execute_tool` | `tool_name` | BlueSkyToolsError | None |
| `llm_atc\tools\bluesky_tools.py` | `get_available_tools` | `` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `__init__` | `self` | ConflictDetectionMethod | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `detect_conflicts_comprehensive` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_detect_with_swarm` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_detect_with_statebased` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_detect_with_enhanced_analysis` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_analyze_aircraft_pair` | `self, ac1_idx, ac2_idx, method` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_calculate_cpa` | `self, lat1, lon1, alt1, hdg1, spd1, vs1, lat2, lon2, alt2, hdg2, spd2, vs2` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_calculate_horizontal_distance` | `self, lat1, lon1, lat2, lon2` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_assess_conflict_severity` | `self, h_sep, v_sep, time_to_cpa, violates_icao` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_calculate_confidence` | `self, method, h_sep, v_sep, time_to_cpa` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_cross_validate_conflicts` | `self, all_conflicts` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_merge_conflict_detections` | `self, detections` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_get_aircraft_pair_key` | `self, ac1, ac2` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `validate_llm_conflicts` | `self, llm_conflicts` | None | None |
| `llm_atc\tools\enhanced_conflict_detector.py` | `_mock_conflict_detection` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `__init__` | `self` | ConflictDetectionMethod | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `detect_conflicts_comprehensive` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_detect_with_swarm` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_detect_with_statebased` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_detect_with_enhanced_analysis` | `self` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_analyze_aircraft_pair` | `self, ac1_idx, ac2_idx, method` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_calculate_cpa` | `self, lat1, lon1, alt1, hdg1, spd1, vs1, lat2, lon2, alt2, hdg2, spd2, vs2` | None | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_calculate_horizontal_distance` | `self, lat1, lon1, lat2, lon2` | None | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_assess_conflict_severity` | `self, h_sep, v_sep, time_to_cpa, violates_icao` | None | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_calculate_confidence` | `self, method, h_sep, v_sep, time_to_cpa` | None | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_cross_validate_conflicts` | `self, all_conflicts` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_merge_conflict_detections` | `self, detections` | ConflictData | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_get_aircraft_pair_key` | `self, ac1, ac2` | None | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `validate_llm_conflicts` | `self, llm_conflicts` | None | None |
| `llm_atc\tools\enhanced_conflict_detector_clean.py` | `_mock_conflict_detection` | `self` | ConflictData | None |
| `llm_atc\tools\llm_prompt_engine.py` | `__init__` | `self, model, enable_function_calls, aircraft_id_regex, enable_streaming, enable_caching, enable_optimized_prompts` | LLMClient | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_init_prompt_templates` | `self` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `format_conflict_prompt` | `self, conflict_info` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `format_detector_prompt` | `self, aircraft_states, time_horizon, cpa_data` | aircraft_list | None |
| `llm_atc\tools\llm_prompt_engine.py` | `parse_resolution_response` | `self, response_text` | ResolutionResponse | None |
| `llm_atc\tools\llm_prompt_engine.py` | `parse_detector_response` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_is_distilled_model_response` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_distilled_model_response` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_detector_response_legacy` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_validate_detector_response` | `self, json_data` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_validate_aircraft_pairs` | `self, pairs` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_validate_confidence` | `self, confidence` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_validate_priority` | `self, priority` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_validate_sector_response` | `self, json_data` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_validate_calculation_details` | `self, calc_details` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_extract_json_from_response` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `get_conflict_resolution` | `self, conflict_info, use_function_calls` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `get_conflict_resolution_with_prompts` | `self, conflict_info, use_function_calls` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `detect_conflict_via_llm` | `self, aircraft_states, time_horizon, cpa_data` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `detect_conflict_via_llm_with_prompts` | `self, aircraft_states, time_horizon, cpa_data` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `assess_resolution_safety` | `self, command, conflict_info` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_get_fallback_conflict_prompt` | `self, conflict_info` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_function_call_response` | `self, response_dict` | ResolutionResponse | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_extract_bluesky_command` | `self, text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_normalize_bluesky_command` | `self, command` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_extract_aircraft_id` | `self, command` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_determine_maneuver_type` | `self, command` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_aircraft_pairs` | `self, pairs_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_time_values` | `self, time_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_safety_response` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `format_conflict_resolution_prompt_optimized` | `self, conflict_info` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `format_conflict_detection_prompt_optimized` | `self, aircraft_states, time_horizon` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `get_conflict_resolution_optimized` | `self, conflict_info, priority` | ResolutionResponse | None |
| `llm_atc\tools\llm_prompt_engine.py` | `get_conflict_detection_optimized` | `self, aircraft_states, time_horizon, priority` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_resolution_response_fast` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_parse_detection_response_fast` | `self, response_text` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_extract_aircraft_id_fast` | `self, command` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `_determine_maneuver_type_fast` | `self, command` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `get_performance_stats` | `self` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `reset_performance_stats` | `self` | None | None |
| `llm_atc\tools\llm_prompt_engine.py` | `quick_resolve_conflict` | `aircraft_1, aircraft_2, time_to_conflict, engine` | LLMPromptEngine, ResolutionResponse | None |
| `llm_atc\tools\llm_prompt_engine.py` | `quick_detect_conflicts` | `aircraft_states, engine` | LLMPromptEngine | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `__init__` | `self` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `check_prerequisites` | `self` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `merge_lora_adapter` | `self` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `_save_model_metadata` | `self` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `convert_to_gguf` | `self` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `create_ollama_model` | `self, use_gguf` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `_create_enhanced_modelfile` | `self, model_source, final_loss, eval_loss` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `_verify_ollama_model` | `self, model_name` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `run_complete_pipeline` | `self` | None | None |
| `BSKY_GYM_LLM\merge_lora_and_convert.py` | `main` | `` | LoRAMerger | None |
| `scenarios\monte_carlo_framework.py` | `aircraft_list` | `self` | aircraft_list | None |
| `scenarios\monte_carlo_framework.py` | `environmental` | `self` | None | None |
| `scenarios\monte_carlo_framework.py` | `__init__` | `self, ranges_file, distribution_shift_file, ranges_dict` | None | None |
| `scenarios\monte_carlo_framework.py` | `_load_ranges` | `self` | None | None |
| `scenarios\monte_carlo_framework.py` | `_load_distribution_shift_config` | `self` | None | None |
| `scenarios\monte_carlo_framework.py` | `_get_default_ranges` | `self` | None | None |
| `scenarios\monte_carlo_framework.py` | `sample_from_range` | `self, range_spec` | None | None |
| `scenarios\monte_carlo_framework.py` | `weighted_choice` | `self, choices, weights` | None | None |
| `scenarios\monte_carlo_framework.py` | `apply_distribution_shift` | `self, base_ranges, shift_tier` | None | None |
| `scenarios\monte_carlo_framework.py` | `generate_scenario` | `self, complexity_tier, force_conflicts, airspace_region, distribution_shift_tier` | ComplexityTier, ScenarioConfiguration | None |
| `scenarios\monte_carlo_framework.py` | `_generate_environmental_conditions` | `self, ranges` | None | None |
| `scenarios\monte_carlo_framework.py` | `_generate_bluesky_commands` | `self, aircraft_count, aircraft_types, positions, speeds, headings, environmental_conditions, force_conflicts, ranges, distribution_shift_tier` | None | None |
| `scenarios\monte_carlo_framework.py` | `_generate_conflict_commands` | `self, aircraft_count` | None | None |
| `scenarios\monte_carlo_framework.py` | `_calculate_bearing` | `self, lat1, lon1, lat2, lon2` | None | None |
| `scenarios\monte_carlo_framework.py` | `execute_scenario` | `self, scenario` | ScenarioConfiguration | None |
| `scenarios\monte_carlo_framework.py` | `_mock_execution` | `self, scenario` | ComplexityTier, ScenarioConfiguration | None |
| `scenarios\monte_carlo_framework.py` | `generate_scenario_batch` | `self, count, complexity_distribution, distribution_shift_distribution` | ComplexityTier, ScenarioConfiguration | None |
| `scenarios\monte_carlo_framework.py` | `get_command_log` | `self` | None | None |
| `scenarios\monte_carlo_framework.py` | `validate_ranges` | `self, scenario` | ScenarioConfiguration | None |
| `scenarios\monte_carlo_framework.py` | `generate_scenario` | `complexity_tier, force_conflicts, distribution_shift_tier` | BlueSkyScenarioGenerator, ComplexityTier, ScenarioConfiguration | None |
| `scenarios\monte_carlo_framework.py` | `generate_monte_carlo_scenarios` | `count, complexity_distribution, distribution_shift_distribution` | BlueSkyScenarioGenerator, ScenarioConfiguration | None |
| `scenarios\monte_carlo_runner.py` | `__post_init__` | `self` | ComplexityTier, ScenarioType | None |
| `scenarios\monte_carlo_runner.py` | `__post_init__` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `__init__` | `self, config` | BenchmarkConfiguration, DetectionComparison, LLMPromptEngine, ScenarioGenerator, ScenarioResult, set_strict_mode | None |
| `scenarios\monte_carlo_runner.py` | `_setup_output_directory` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_setup_logging` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_setup_enhanced_logging` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_init_csv_file` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `run` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_calculate_total_scenarios` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_run_scenario_batch` | `self, scenario_type, complexity_tier, shift_level` | ComplexityTier, ScenarioType | None |
| `scenarios\monte_carlo_runner.py` | `_generate_scenario` | `self, scenario_type, complexity_tier, shift_level, scenario_id` | ComplexityTier, ScenarioType, generate_horizontal_scenario, generate_sector_scenario, generate_vertical_scenario | None |
| `scenarios\monte_carlo_runner.py` | `_get_aircraft_count_for_complexity` | `self, complexity_tier` | ComplexityTier | None |
| `scenarios\monte_carlo_runner.py` | `_run_single_scenario` | `self, scenario, scenario_id` | ComplexityTier, ScenarioResult, ScenarioType | None |
| `scenarios\monte_carlo_runner.py` | `_execute_scenario_pipeline` | `self, scenario, scenario_id` | ComplexityTier, ScenarioResult, ScenarioType | None |
| `scenarios\monte_carlo_runner.py` | `_reset_bluesky_simulation` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_load_scenario_commands` | `self, scenario` | None | None |
| `scenarios\monte_carlo_runner.py` | `_extract_ground_truth_conflicts` | `self, scenario` | None | None |
| `scenarios\monte_carlo_runner.py` | `_detect_conflicts` | `self, scenario` | EnhancedConflictDetector | None |
| `scenarios\monte_carlo_runner.py` | `_basic_conflict_detection_fallback` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_get_aircraft_states_for_llm` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_validate_llm_conflicts_with_bluesky` | `self, llm_pairs, bluesky_conflicts` | None | None |
| `scenarios\monte_carlo_runner.py` | `_resolve_conflicts` | `self, conflicts, scenario` | None | None |
| `scenarios\monte_carlo_runner.py` | `_is_valid_bluesky_command` | `self, command` | None | None |
| `scenarios\monte_carlo_runner.py` | `_format_conflict_for_llm` | `self, conflict, scenario` | None | None |
| `scenarios\monte_carlo_runner.py` | `_verify_resolutions` | `self, scenario, resolutions` | None | None |
| `scenarios\monte_carlo_runner.py` | `_calculate_all_separations` | `self, aircraft_info` | aircraft_list | None |
| `scenarios\monte_carlo_runner.py` | `_calculate_scenario_metrics` | `self, ground_truth, detected, resolutions, verification` | None | None |
| `scenarios\monte_carlo_runner.py` | `_create_error_result` | `self, scenario_id, scenario_type, complexity_tier, shift_level, error` | ComplexityTier, ScenarioResult, ScenarioType | None |
| `scenarios\monte_carlo_runner.py` | `_generate_summary` | `self` | MonteCarloResultsAnalyzer | None |
| `scenarios\monte_carlo_runner.py` | `_get_serializable_config` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_generate_summary_by_group` | `self, df, group_column` | None | None |
| `scenarios\monte_carlo_runner.py` | `_generate_combined_summary` | `self, df` | None | None |
| `scenarios\monte_carlo_runner.py` | `_print_detailed_analysis` | `self, analysis` | None | None |
| `scenarios\monte_carlo_runner.py` | `_generate_visualizations` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_plot_detection_performance` | `self, df, fig_size` | None | None |
| `scenarios\monte_carlo_runner.py` | `_plot_safety_margins` | `self, df, fig_size` | None | None |
| `scenarios\monte_carlo_runner.py` | `_plot_efficiency_metrics` | `self, df, fig_size` | None | None |
| `scenarios\monte_carlo_runner.py` | `_plot_performance_by_type` | `self, df, fig_size` | None | None |
| `scenarios\monte_carlo_runner.py` | `_plot_distribution_shift_impact` | `self, df, fig_size` | None | None |
| `scenarios\monte_carlo_runner.py` | `_save_results` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `_run_enhanced_scenario` | `self, scenario, scenario_id` | ScenarioResult | None |
| `scenarios\monte_carlo_runner.py` | `_create_detection_comparison` | `self, scenario, scenario_id, result, execution_time` | DetectionComparison, ScenarioResult | None |
| `scenarios\monte_carlo_runner.py` | `_write_csv_row` | `self, comparison` | DetectionComparison | None |
| `scenarios\monte_carlo_runner.py` | `_save_detection_analysis` | `self` | None | None |
| `scenarios\monte_carlo_runner.py` | `run_benchmark_with_config` | `config_path` | BenchmarkConfiguration, ComplexityTier, MonteCarloBenchmark, ScenarioType | None |
| `scenarios\monte_carlo_runner.py` | `main` | `` | BenchmarkConfiguration, MonteCarloBenchmark, run_benchmark_with_config | None |
| `scenarios\scenario_generator.py` | `__post_init__` | `self` | None | None |
| `scenarios\scenario_generator.py` | `to_dict` | `self` | None | None |
| `scenarios\scenario_generator.py` | `__init__` | `self, ranges_file, distribution_shift_file` | BlueSkyScenarioGenerator | None |
| `scenarios\scenario_generator.py` | `generate_scenario` | `self, scenario_type` | Scenario, ScenarioType | None |
| `scenarios\scenario_generator.py` | `generate_horizontal_scenario` | `self, n_aircraft, conflict, complexity_tier, distribution_shift_tier` | BlueSkyScenarioGenerator, ComplexityTier, Scenario, ScenarioType | None |
| `scenarios\scenario_generator.py` | `generate_vertical_scenario` | `self, n_aircraft, conflict, climb_rates, crossing_altitudes, complexity_tier, distribution_shift_tier` | ComplexityTier, Scenario, ScenarioType | None |
| `scenarios\scenario_generator.py` | `generate_sector_scenario` | `self, complexity, shift_level, force_conflicts` | ComplexityTier, Scenario, ScenarioType | None |
| `scenarios\scenario_generator.py` | `_create_horizontal_conflicts` | `self, aircraft_states` | None | None |
| `scenarios\scenario_generator.py` | `_avoid_horizontal_conflicts` | `self, aircraft_states` | None | None |
| `scenarios\scenario_generator.py` | `_create_vertical_conflicts` | `self, aircraft_states` | None | None |
| `scenarios\scenario_generator.py` | `_avoid_vertical_conflicts` | `self, aircraft_states` | None | None |
| `scenarios\scenario_generator.py` | `_create_vertical_conflicts_enhanced` | `self, aircraft_states, climb_rates` | None | None |
| `scenarios\scenario_generator.py` | `_avoid_vertical_conflicts_enhanced` | `self, aircraft_states, climb_rates` | None | None |
| `scenarios\scenario_generator.py` | `_optimize_conflict_timing` | `self, aircraft_states, commands` | None | None |
| `scenarios\scenario_generator.py` | `_add_environmental_commands` | `self, env_conditions` | None | None |
| `scenarios\scenario_generator.py` | `_calculate_horizontal_ground_truth` | `self, aircraft_states, expect_conflicts` | GroundTruthConflict | None |
| `scenarios\scenario_generator.py` | `_analyze_aircraft_pair_trajectory` | `self, ac1, ac2` | None | None |
| `scenarios\scenario_generator.py` | `_calculate_vertical_ground_truth` | `self, aircraft_states, expect_conflicts` | GroundTruthConflict | None |
| `scenarios\scenario_generator.py` | `_calculate_sector_ground_truth` | `self, aircraft_states, base_scenario` | GroundTruthConflict, ScenarioConfiguration | None |
| `scenarios\scenario_generator.py` | `_analyze_trajectory_conflict` | `self, ac1, ac2` | None | None |
| `scenarios\scenario_generator.py` | `_determine_conflict_severity` | `self, horizontal_sep, vertical_sep` | None | None |
| `scenarios\scenario_generator.py` | `_calculate_bearing` | `self, lat1, lon1, lat2, lon2` | None | None |
| `scenarios\scenario_generator.py` | `_calculate_distance_nm` | `self, lat1, lon1, lat2, lon2` | None | None |
| `scenarios\scenario_generator.py` | `_are_headings_convergent` | `self, lat1, lon1, hdg1, lat2, lon2, hdg2` | None | None |
| `scenarios\scenario_generator.py` | `_project_position` | `self, lat, lon, heading, speed_kts, time_min` | None | None |
| `scenarios\scenario_generator.py` | `__init__` | `self, generator` | ScenarioGenerator | None |
| `scenarios\scenario_generator.py` | `generate_scenario` | `self, n_aircraft, conflict` | Scenario | None |
| `scenarios\scenario_generator.py` | `__init__` | `self, generator` | ScenarioGenerator | None |
| `scenarios\scenario_generator.py` | `generate_scenario` | `self, n_aircraft, conflict` | Scenario | None |
| `scenarios\scenario_generator.py` | `__init__` | `self, generator` | ScenarioGenerator | None |
| `scenarios\scenario_generator.py` | `generate_scenario` | `self, complexity, shift_level, force_conflicts` | ComplexityTier, Scenario | None |
| `scenarios\scenario_generator.py` | `generate_horizontal_scenario` | `n_aircraft, conflict` | Scenario, ScenarioGenerator | None |
| `scenarios\scenario_generator.py` | `generate_vertical_scenario` | `n_aircraft, conflict` | Scenario, ScenarioGenerator | None |
| `scenarios\scenario_generator.py` | `generate_sector_scenario` | `complexity, shift_level, force_conflicts` | ComplexityTier, Scenario, ScenarioGenerator | None |
| `llm_interface\ensemble.py` | `__init__` | `self` | None | None |
| `llm_interface\ensemble.py` | `_initialize_models` | `self` | ModelConfig, ModelRole | None |
| `llm_interface\ensemble.py` | `_get_available_models` | `self` | None | None |
| `llm_interface\ensemble.py` | `query_ensemble` | `self, prompt, context, require_json, timeout` | EnsembleResponse | None |
| `llm_interface\ensemble.py` | `_create_role_specific_prompts` | `self, base_prompt, context` | ModelRole | None |
| `llm_interface\ensemble.py` | `_query_single_model` | `self, model_config, prompt, require_json` | ModelConfig | None |
| `llm_interface\ensemble.py` | `_analyze_safety_flags` | `self, responses` | None | None |
| `llm_interface\ensemble.py` | `_calculate_consensus` | `self, responses` | None | None |
| `llm_interface\ensemble.py` | `_calculate_uncertainty_metrics` | `self, responses` | None | None |
| `llm_interface\ensemble.py` | `_create_error_response` | `self, error_msg, response_time` | EnsembleResponse | None |
| `llm_interface\ensemble.py` | `get_ensemble_statistics` | `self` | None | None |
| `llm_interface\ensemble.py` | `_clean_json_response` | `self, json_str` | None | None |
| `llm_interface\ensemble.py` | `_create_valid_response_structure` | `self, raw_content` | None | None |
| `llm_interface\ensemble.py` | `_extract_partial_response_data` | `self, raw_content` | None | None |
| `llm_interface\ensemble.py` | `__init__` | `self` | None | None |
| `llm_interface\ensemble.py` | `_initialize_knowledge_base` | `self` | None | None |
| `llm_interface\ensemble.py` | `validate_response` | `self, response, context` | None | None |
| `llm_interface\filter_sort.py` | `get_llm_client` | `` | LLMClient | None |
| `llm_interface\filter_sort.py` | `get_llm_stats` | `` | get_llm_client | None |
| `llm_interface\filter_sort.py` | `select_best_solution` | `candidates, policies` | get_llm_client | None |
| `llm_interface\llm_client.py` | `__init__` | `self, model, max_retries, timeout, enable_streaming, enable_caching, cache_size, enable_optimized_prompts` | None | None |
| `llm_interface\llm_client.py` | `create_chat_messages` | `self, system_prompt, user_prompt, context` | ChatMessage | None |
| `llm_interface\llm_client.py` | `ask` | `self, prompt, expect_json, enable_function_calls, system_prompt, priority` | None | None |
| `llm_interface\llm_client.py` | `ask_optimized` | `self, user_prompt, system_prompt, expect_json, context, priority` | ChatMessage, LLMResponse | None |
| `llm_interface\llm_client.py` | `_execute_chat_request` | `self, messages, timeout, expect_json` | None | None |
| `llm_interface\llm_client.py` | `_enhance_prompt_for_function_calling` | `self, original_prompt` | None | None |
| `llm_interface\llm_client.py` | `_process_function_calls` | `self, content` | None | None |
| `llm_interface\llm_client.py` | `_execute_function_call` | `self, function_name, arguments` | None | None |
| `llm_interface\llm_client.py` | `chat_with_function_calling` | `self, messages, max_function_calls` | None | None |
| `llm_interface\llm_client.py` | `_format_conversation_for_prompt` | `self, messages` | None | None |
| `llm_interface\llm_client.py` | `get_average_inference_time` | `self` | None | None |
| `llm_interface\llm_client.py` | `get_total_inference_time` | `self` | None | None |
| `llm_interface\llm_client.py` | `get_inference_count` | `self` | None | None |
| `llm_interface\llm_client.py` | `validate_response` | `self, response, expected_keys` | None | None |
| `llm_interface\llm_client.py` | `_parse_json_response_fast` | `self, content` | None | None |
| `llm_interface\llm_client.py` | `_fix_common_json_issues` | `self, json_str` | None | None |
| `llm_interface\llm_client.py` | `_validate_atc_json_structure` | `self, parsed_json` | None | None |
| `llm_interface\llm_client.py` | `get_safe_default_resolution` | `self, scenario_type` | None | None |
| `llm_interface\llm_client.py` | `_create_cache_key` | `self, user_prompt, system_prompt` | None | None |
| `llm_interface\llm_client.py` | `_cache_response` | `self, cache_key, response` | None | None |
| `llm_interface\llm_client.py` | `_get_priority_timeout` | `self, priority` | None | None |
| `llm_interface\llm_client.py` | `get_conflict_resolution_system_prompt` | `self` | None | None |
| `llm_interface\llm_client.py` | `get_conflict_detection_system_prompt` | `self` | None | None |
| `llm_interface\llm_client.py` | `get_performance_stats` | `self` | None | None |
| `llm_interface\llm_client.py` | `reset_stats` | `self` | None | None |
| `llm_interface\llm_client.py` | `quick_conflict_resolution` | `aircraft_1, aircraft_2, time_to_conflict, client` | LLMClient, LLMResponse | None |
| `llm_interface\llm_client.py` | `quick_conflict_detection` | `aircraft_states, client` | LLMClient, LLMResponse | None |
| `analysis\enhanced_hallucination_detection.py` | `__init__` | `self, prompt_engine` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `_init_detection_patterns` | `self` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `detect_hallucinations` | `self, llm_response, baseline_response, context` | HallucinationResult, HallucinationType | None |
| `analysis\enhanced_hallucination_detection.py` | `_check_aircraft_existence` | `self, response_text, context` | aircraft_list | None |
| `analysis\enhanced_hallucination_detection.py` | `_check_altitude_validity` | `self, response_text` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `_check_heading_validity` | `self, response_text` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `_check_protocol_violations` | `self, response_text` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `_check_impossible_maneuvers` | `self, response_text, context` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `_check_nonsensical_response` | `self, response_text` | None | None |
| `analysis\enhanced_hallucination_detection.py` | `_determine_severity` | `self, detected_types` | HallucinationType | None |
| `analysis\enhanced_hallucination_detection.py` | `create_enhanced_detector` | `` | EnhancedHallucinationDetector | None |
| `analysis\visualisation.py` | `__init__` | `self, output_dir, style, dpi` | None | None |
| `analysis\visualisation.py` | `generate_comprehensive_report` | `self, data, title` | None | None |
| `analysis\visualisation.py` | `_generate_distribution_analysis` | `self, data` | None | None |
| `analysis\visualisation.py` | `_generate_trend_analysis` | `self, data` | None | None |
| `analysis\visualisation.py` | `_generate_sensitivity_analysis` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_metric_distributions` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_shift_comparisons` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_violin_comparisons` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_ridge_plots` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_cumulative_error_curves` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_time_series_analysis` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_performance_evolution` | `self, data` | None | None |
| `analysis\visualisation.py` | `_plot_tornado_sensitivity` | `self, data` | None | None |
| `analysis\visualisation.py` | `plot_cd_timeline` | `df, sim_id, output_dir` | MonteCarloVisualizer | None |
| `analysis\visualisation.py` | `plot_cr_flowchart` | `sim_id, tier, output_dir` | MonteCarloVisualizer | None |
| `analysis\visualisation.py` | `plot_tier_comparison` | `df, output_dir` | MonteCarloVisualizer | None |
| `analysis\visualisation.py` | `create_visualization_summary` | `output_dir` | None | None |

## Generated Patches

No patches generated (no removed code detected)
