# analysis/extract_and_analyze.py
import re
import json
import logging
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analysis.metrics import compute_metrics, print_metrics_summary, create_empty_metrics

def extract_json_from_log(log_file):
    """Extract JSON data from log file containing mixed content."""
    json_entries = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Use regex to find JSON objects in the log
        json_pattern = r'INFO:root:(\{.*?\})\s*$'
        matches = re.findall(json_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            try:
                # Clean up the JSON string
                json_str = match.strip()
                if json_str.startswith('{') and json_str.endswith('}'):
                    data = json.loads(json_str)
                    # Only include entries with the expected structure
                    if 'conflict' in data and 'candidates' in data:
                        json_entries.append(data)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {e}")
                continue
    
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        return []
    
    return json_entries

def compute_metrics_from_extracted_data(json_entries):
    """Compute metrics from extracted JSON entries."""
    if not json_entries:
        return create_empty_metrics()
    
    metrics = {
        'total_conflicts': len(json_entries),
        'hallucination_events': 0,
        'policy_violations': 0,
        'llm_errors': 0,
        'safety_margin_differences': [],
        'response_validity': {'valid': 0, 'invalid': 0},
        'maneuver_type_distribution': {},
        'avg_safety_score_llm': 0.0,
        'avg_safety_score_baseline': 0.0
    }
    
    safety_scores_llm = []
    safety_scores_baseline = []
    
    for entry in json_entries:
        best_by_llm = entry.get('best_by_llm')
        baseline_best = entry.get('baseline_best')
        
        # Check for hallucination indicators
        if best_by_llm != baseline_best:
            metrics['hallucination_events'] += 1
        
        # Analyze LLM choice
        if isinstance(best_by_llm, dict):
            llm_safety = best_by_llm.get('safety_score', 0.5)
            safety_scores_llm.append(llm_safety)
            
            # Track maneuver types
            maneuver_type = best_by_llm.get('type', 'unknown')
            metrics['maneuver_type_distribution'][maneuver_type] = \
                metrics['maneuver_type_distribution'].get(maneuver_type, 0) + 1
            
            metrics['response_validity']['valid'] += 1
        else:
            metrics['response_validity']['invalid'] += 1
            
        # Analyze baseline choice
        if isinstance(baseline_best, dict):
            baseline_safety = baseline_best.get('safety_score', 0.5)
            safety_scores_baseline.append(baseline_safety)
            
            # Calculate safety margin difference
            if isinstance(best_by_llm, dict):
                margin_diff = llm_safety - baseline_safety
                metrics['safety_margin_differences'].append(margin_diff)
    
    # Calculate averages
    if safety_scores_llm:
        metrics['avg_safety_score_llm'] = sum(safety_scores_llm) / len(safety_scores_llm)
    if safety_scores_baseline:
        metrics['avg_safety_score_baseline'] = sum(safety_scores_baseline) / len(safety_scores_baseline)
    
    # Calculate hallucination rate
    metrics['hallucination_rate'] = (
        metrics['hallucination_events'] / metrics['total_conflicts'] 
        if metrics['total_conflicts'] > 0 else 0.0
    )
    
    # Calculate average safety margin difference
    if metrics['safety_margin_differences']:
        metrics['avg_safety_margin_diff'] = sum(metrics['safety_margin_differences']) / len(metrics['safety_margin_differences'])
    else:
        metrics['avg_safety_margin_diff'] = 0.0
    
    return metrics

def main():
    print("Extracting and analyzing simulation results...")
    
    # Extract JSON entries from log
    json_entries = extract_json_from_log('simulation.log')
    print(f"Extracted {len(json_entries)} valid conflict resolution entries")
    
    # Compute metrics
    metrics = compute_metrics_from_extracted_data(json_entries)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Create final JSON report
    final_report = {
        'scenarios_run': 2,  # We know we ran 2 scenarios
        'conflicts_detected': metrics['total_conflicts'],
        'resolutions_attempted': metrics['total_conflicts'],
        'hallucination_rate': metrics['hallucination_rate'],
        'avg_safety_margin_diff': metrics['avg_safety_margin_diff'],
        'valid_llm_responses': metrics['response_validity']['valid'],
        'invalid_llm_responses': metrics['response_validity']['invalid'],
        'maneuver_preferences': metrics['maneuver_type_distribution']
    }
    
    print("\n" + "="*50)
    print("FINAL JSON SUMMARY REPORT")
    print("="*50)
    print(json.dumps(final_report, indent=2))
    
    return final_report

if __name__ == "__main__":
    main()
