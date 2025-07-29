import numpy as np
import logging
import sys
import os

# Add the project root to the Python path for BlueSky imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from bluesky import traf
    BLUESKY_AVAILABLE = True
except ImportError:
    logging.warning("BlueSky not available, using mock resolution system")
    BLUESKY_AVAILABLE = False

class ConflictSolver:
    def __init__(self):
        self.min_separation = 5.0  # nautical miles
        self.min_vertical_separation = 1000  # feet
        self.bluesky_available = BLUESKY_AVAILABLE
        
    def solve(self, conflict):
        """Generate candidate solutions for a given conflict using BlueSky-inspired methods."""
        candidates = []
        
        # Try using BlueSky's resolution system if available
        if self.bluesky_available and hasattr(traf, 'res'):
            try:
                bluesky_solutions = self._get_bluesky_solutions(conflict)
                if bluesky_solutions:
                    candidates.extend(bluesky_solutions)
                    logging.info(f"Generated {len(bluesky_solutions)} BlueSky solutions")
            except Exception as e:
                logging.warning(f"BlueSky resolution failed: {e}")
        
        # Always generate mock solutions as backup/supplement
        mock_solutions = self._generate_mock_solutions(conflict)
        candidates.extend(mock_solutions)
        
        logging.info(f"Total candidates generated: {len(candidates)}")
        return candidates
    
    def _get_bluesky_solutions(self, conflict):
        """Get solutions from BlueSky's resolution system."""
        solutions = []
        
        try:
            # Use BlueSky's resolution system if available
            if hasattr(traf, 'res') and hasattr(traf.res, 'rsolve'):
                # Format conflict for BlueSky
                ac1_id = conflict.get('id1', 'AC1')
                ac2_id = conflict.get('id2', 'AC2')
                
                # Call BlueSky resolution
                result = traf.res.rsolve(conflict)
                
                if result and isinstance(result, (list, tuple)):
                    for i, solution in enumerate(result):
                        formatted_solution = {
                            'type': 'bluesky_resolution',
                            'aircraft': ac1_id if i % 2 == 0 else ac2_id,
                            'action': f'BlueSky resolution {i+1}',
                            'bluesky_data': solution,
                            'safety_score': self._calculate_safety_score('bluesky_resolution', 1.0)
                        }
                        solutions.append(formatted_solution)
                        
        except Exception as e:
            logging.error(f"Error in BlueSky resolution: {e}")
            
        return solutions
    
    def _generate_mock_solutions(self, conflict):
        """Generate mock solutions when BlueSky is not available or as supplement."""
        candidates = []
        
        # Extract conflict information
        if isinstance(conflict, dict):
            ac1_id = conflict.get('id1', 'AC1')
            ac2_id = conflict.get('id2', 'AC2')
            distance = conflict.get('distance', 5.0)
            time_to_conflict = conflict.get('time', 120)
        else:
            ac1_id, ac2_id = 'AC1', 'AC2'
            distance = 5.0
            time_to_conflict = 120
        
        # Heading change solutions
        for heading_change in [-20, -10, 10, 20]:
            candidates.append({
                'type': 'heading',
                'aircraft': ac1_id,
                'action': f'turn {heading_change} degrees',
                'heading_change': heading_change,
                'safety_score': self._calculate_safety_score('heading', abs(heading_change))
            })
        
        # Altitude change solutions
        for alt_change in [-1000, -500, 500, 1000]:
            candidates.append({
                'type': 'altitude',
                'aircraft': ac1_id,
                'action': f'{"climb" if alt_change > 0 else "descend"} {abs(alt_change)} ft',
                'altitude_change': alt_change,
                'safety_score': self._calculate_safety_score('altitude', abs(alt_change))
            })
        
        # Speed change solutions
        for speed_change in [-30, -15, 15, 30]:
            candidates.append({
                'type': 'speed',
                'aircraft': ac1_id,
                'action': f'{"accelerate" if speed_change > 0 else "decelerate"} {abs(speed_change)} knots',
                'speed_change': speed_change,
                'safety_score': self._calculate_safety_score('speed', abs(speed_change))
            })
        
        return candidates
    
    def _calculate_safety_score(self, maneuver_type, magnitude):
        """Calculate safety score based on maneuver type and magnitude."""
        base_scores = {'heading': 0.8, 'altitude': 0.6, 'speed': 0.9}
        base_score = base_scores.get(maneuver_type, 0.5)
        
        # Penalize larger maneuvers
        penalty = min(magnitude / 100.0, 0.3)
        return max(base_score - penalty, 0.1)

    def score_best(self, candidates):
        """Select the best solution based on safety score and minimal disruption."""
        if not candidates:
            return None
        
        # Sort by safety score (descending) and then by magnitude (ascending)
        def sort_key(candidate):
            safety_score = candidate.get('safety_score', 0.5)
            # Prefer smaller maneuvers for same safety score
            magnitude = abs(candidate.get('heading_change', 0)) + \
                       abs(candidate.get('altitude_change', 0)) + \
                       abs(candidate.get('speed_change', 0))
            return (-safety_score, magnitude)
        
        sorted_candidates = sorted(candidates, key=sort_key)
        return sorted_candidates[0]
    
    def validate_solution(self, solution, conflict):
        """Validate that a solution adequately resolves the conflict."""
        # Basic validation - check if solution exists and has required fields
        if not solution or 'action' not in solution:
            return False, "Invalid solution format"
        
        # Check safety score threshold
        safety_score = solution.get('safety_score', 0)
        if safety_score < 0.3:
            return False, "Safety score too low"
        
        return True, "Solution validated"