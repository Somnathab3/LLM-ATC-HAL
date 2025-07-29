# llm_interface/filter_sort.py
from .llm_client import LLMClient
import logging
import json

logging.basicConfig(level=logging.INFO)

# Global LLM client instance to track stats across calls
_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

def get_llm_stats():
    """Get LLM timing statistics."""
    client = get_llm_client()
    return {
        'total_calls': client.get_inference_count(),
        'total_time': client.get_total_inference_time(),
        'avg_time_per_call': client.get_average_inference_time()
    }

def select_best_solution(candidates, policies):
    """Filter and select the best solution using LLM based on policies."""
    if not candidates:
        logging.warning("No candidates provided")
        return None
        
    llm = get_llm_client()
    filtered = []
    hallucination_events = []
    
    for candidate in candidates:
        violation = False
        candidate_violations = []
        
        for policy in policies:
            prompt = f"""Analyze this aviation conflict resolution solution against the policy.
Policy: '{policy}'
Solution: {json.dumps(candidate)}

Respond with ONLY 'yes' if it violates the policy, or 'no' if it complies.
Consider safety margins, operational efficiency, and standard ATC procedures."""
            
            try:
                answer = llm.ask(prompt).strip().lower()
                logging.info(f"LLM response for candidate {candidate.get('action', 'Unknown')} and policy '{policy}': {answer}")
                
                # Check for hallucination indicators
                if not any(keyword in answer for keyword in ['yes', 'no']):
                    hallucination_events.append({
                        'type': 'invalid_response',
                        'candidate': candidate,
                        'policy': policy,
                        'response': answer
                    })
                    logging.warning(f"Potential hallucination - unexpected response: {answer}")
                    violation = True  # Err on side of caution
                elif 'yes' in answer:
                    violation = True
                    candidate_violations.append(policy)
                elif 'no' not in answer:
                    hallucination_events.append({
                        'type': 'ambiguous_response',
                        'candidate': candidate,
                        'policy': policy,
                        'response': answer
                    })
                    logging.warning(f"Ambiguous LLM response: {answer}")
                    
            except Exception as e:
                logging.error(f"Error querying LLM: {e}")
                violation = True  # Err on side of caution
                hallucination_events.append({
                    'type': 'llm_error',
                    'candidate': candidate,
                    'policy': policy,
                    'error': str(e)
                })
                
        if not violation:
            filtered.append(candidate)
        else:
            candidate['policy_violations'] = candidate_violations
    
    # Log hallucination events for analysis
    if hallucination_events:
        logging.warning(f"Detected {len(hallucination_events)} potential hallucination events")
        for event in hallucination_events:
            logging.warning(f"Hallucination event: {json.dumps(event)}")
    
    # Select best from filtered candidates
    if filtered:
        # Prefer higher safety scores
        best = max(filtered, key=lambda x: x.get('safety_score', 0.5))
        logging.info(f"Selected best solution: {best.get('action', 'Unknown')}")
        return best
    else:
        logging.warning("No candidates passed policy filters")
        return candidates[0] if candidates else None  # Fallback to first candidate
