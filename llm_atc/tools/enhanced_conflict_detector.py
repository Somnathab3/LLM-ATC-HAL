# llm_atc/tools/enhanced_conflict_detector.py
"""
Enhanced Conflict Detection Module for LLM-ATC-HAL
=================================================
Uses multiple BlueSky conflict detection methods (SWARM, STATEBASED) and CPA analysis
to provide comprehensive conflict detection and validation for false positive/negative analysis.

Key Features:
- Multiple conflict detection methods
- ICAO-compliant separation standards (5.0nm horizontal, 1000ft vertical)  
- 300s time horizon check
- CPA (Closest Point of Approach) calculations
- Enhanced validation for LLM responses
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

try:
    import bluesky as bs
    from bluesky import sim, stack, traf
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False
    logging.warning("BlueSky not available - using mock detection")


class ConflictDetectionMethod(Enum):
    """Available conflict detection methods in BlueSky"""
    SWARM = "SWARM"
    STATEBASED = "STATEBASED"
    ENHANCED = "ENHANCED"  # Our custom implementation


@dataclass
class ConflictData:
    """Comprehensive conflict information"""
    aircraft_1: str
    aircraft_2: str
    time_to_cpa: float  # seconds
    distance_at_cpa: float  # nautical miles
    current_horizontal_separation: float  # nautical miles
    current_vertical_separation: float  # feet
    min_horizontal_separation: float  # nautical miles (during entire trajectory)
    min_vertical_separation: float  # feet (during entire trajectory)
    conflict_detected: bool
    violates_icao_separation: bool
    severity: str  # 'critical', 'high', 'medium', 'low'
    detection_method: str
    confidence: float  # 0.0 to 1.0
    

class EnhancedConflictDetector:
    """
    Enhanced conflict detector using multiple BlueSky methods and ICAO standards
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ICAO separation standards - consistent with scenario generator
        self.CONFLICT_THRESHOLD_NM = 5.0  # Both detection and ground truth
        self.CONFLICT_THRESHOLD_FT = 1000.0  # Both detection and ground truth
        self.TIME_HORIZON_SEC = 300.0  # 5 minutes lookahead
        
        # Detection method priorities (for validation)
        self.detection_methods = [
            ConflictDetectionMethod.SWARM,
            ConflictDetectionMethod.STATEBASED,
            ConflictDetectionMethod.ENHANCED
        ]
        
    def detect_conflicts_comprehensive(self) -> List[ConflictData]:
        """
        Detect conflicts using all available methods and cross-validate results
        
        Returns:
            List of validated conflict data
        """
        if not BLUESKY_AVAILABLE:
            return self._mock_conflict_detection()
            
        all_conflicts = {}  # Key: (ac1, ac2), Value: list of detections
        
        # Method 1: SWARM detection
        swarm_conflicts = self._detect_with_swarm()
        for conflict in swarm_conflicts:
            key = self._get_aircraft_pair_key(conflict.aircraft_1, conflict.aircraft_2)
            if key not in all_conflicts:
                all_conflicts[key] = []
            all_conflicts[key].append(conflict)
            
        # Method 2: STATEBASED detection  
        statebased_conflicts = self._detect_with_statebased()
        for conflict in statebased_conflicts:
            key = self._get_aircraft_pair_key(conflict.aircraft_1, conflict.aircraft_2)
            if key not in all_conflicts:
                all_conflicts[key] = []
            all_conflicts[key].append(conflict)
            
        # Method 3: Enhanced geometric analysis
        enhanced_conflicts = self._detect_with_enhanced_analysis()
        for conflict in enhanced_conflicts:
            key = self._get_aircraft_pair_key(conflict.aircraft_1, conflict.aircraft_2)
            if key not in all_conflicts:
                all_conflicts[key] = []
            all_conflicts[key].append(conflict)
            
        # Cross-validate and merge results
        validated_conflicts = self._cross_validate_conflicts(all_conflicts)
        
        return validated_conflicts
        
    def _detect_with_swarm(self) -> List[ConflictData]:
        """Detect conflicts using BlueSky SWARM method"""
        conflicts = []
        
        try:
            # Set SWARM conflict detection
            stack.stack("CDMETHOD SWARM")
            stack.stack(f"CDSEP {self.CONFLICT_THRESHOLD_NM} {self.CONFLICT_THRESHOLD_FT}")
            
            # Access SWARM conflict detection results
            if hasattr(traf, 'cd') and hasattr(traf.cd, 'confpairs_all'):
                confpairs = traf.cd.confpairs_all
                if confpairs is not None:
                    for i, (ac1_id, ac2_id) in enumerate(confpairs):
                        # Convert aircraft IDs to indices in traf.id list
                        try:
                            if isinstance(ac1_id, str):
                                ac1_idx = list(traf.id).index(ac1_id)
                            else:
                                ac1_idx = int(ac1_id)
                                
                            if isinstance(ac2_id, str):
                                ac2_idx = list(traf.id).index(ac2_id)
                            else:
                                ac2_idx = int(ac2_id)
                                
                            if ac1_idx < len(traf.id) and ac2_idx < len(traf.id):
                                conflict_data = self._analyze_aircraft_pair(
                                    ac1_idx, ac2_idx, "SWARM"
                                )
                                if conflict_data:
                                    conflicts.append(conflict_data)
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Invalid aircraft ID in SWARM confpairs: {ac1_id}, {ac2_id} - {e}")
                            continue
                                
        except Exception as e:
            self.logger.exception(f"SWARM detection failed: {e}")
            
        return conflicts
        
    def _detect_with_statebased(self) -> List[ConflictData]:
        """Detect conflicts using BlueSky STATEBASED method"""
        conflicts = []
        
        try:
            # Set STATEBASED conflict detection
            stack.stack("CDMETHOD STATEBASED")
            stack.stack(f"CDSEP {self.CONFLICT_THRESHOLD_NM} {self.CONFLICT_THRESHOLD_FT}")
            
            # Access STATEBASED conflict detection results
            if hasattr(traf, 'cd') and hasattr(traf.cd, 'confpairs_all'):
                confpairs = traf.cd.confpairs_all
                if confpairs is not None:
                    for i, (ac1_id, ac2_id) in enumerate(confpairs):
                        # Convert aircraft IDs to indices in traf.id list
                        try:
                            if isinstance(ac1_id, str):
                                ac1_idx = list(traf.id).index(ac1_id)
                            else:
                                ac1_idx = int(ac1_id)
                                
                            if isinstance(ac2_id, str):
                                ac2_idx = list(traf.id).index(ac2_id)
                            else:
                                ac2_idx = int(ac2_id)
                                
                            if ac1_idx < len(traf.id) and ac2_idx < len(traf.id):
                                conflict_data = self._analyze_aircraft_pair(
                                    ac1_idx, ac2_idx, "STATEBASED"
                                )
                                if conflict_data:
                                    conflicts.append(conflict_data)
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Invalid aircraft ID in STATEBASED confpairs: {ac1_id}, {ac2_id} - {e}")
                            continue
                                
        except Exception as e:
            self.logger.exception(f"STATEBASED detection failed: {e}")
            
        return conflicts
        
    def _detect_with_enhanced_analysis(self) -> List[ConflictData]:
        """Detect conflicts using enhanced geometric analysis"""
        conflicts = []
        
        try:
            # Analyze all aircraft pairs manually
            for i in range(len(traf.id)):
                for j in range(i + 1, len(traf.id)):
                    conflict_data = self._analyze_aircraft_pair(i, j, "ENHANCED")
                    if conflict_data and conflict_data.conflict_detected:
                        conflicts.append(conflict_data)
                        
        except Exception as e:
            self.logger.exception(f"Enhanced analysis failed: {e}")
            
        return conflicts
        
    def _analyze_aircraft_pair(self, ac1_idx: int, ac2_idx: int, method: str) -> Optional[ConflictData]:
        """
        Analyze specific aircraft pair for conflicts with CPA calculation
        Implements the 300s time horizon check as requested
        """
        try:
            if ac1_idx >= len(traf.id) or ac2_idx >= len(traf.id):
                return None
                
            ac1_id = traf.id[ac1_idx]
            ac2_id = traf.id[ac2_idx]
            
            # Current positions
            lat1, lon1, alt1 = traf.lat[ac1_idx], traf.lon[ac1_idx], traf.alt[ac1_idx]
            lat2, lon2, alt2 = traf.lat[ac2_idx], traf.lon[ac2_idx], traf.alt[ac2_idx]
            
            # Current velocities  
            hdg1, hdg2 = traf.hdg[ac1_idx], traf.hdg[ac2_idx]
            spd1, spd2 = traf.tas[ac1_idx], traf.tas[ac2_idx]  # True airspeed
            vs1, vs2 = traf.vs[ac1_idx], traf.vs[ac2_idx]  # Vertical speed
            
            # Calculate current separation
            current_h_sep = self._calculate_horizontal_distance(lat1, lon1, lat2, lon2)
            current_v_sep = abs(alt1 - alt2)
            
            # Calculate CPA (Closest Point of Approach)
            time_to_cpa, distance_at_cpa, min_h_sep, min_v_sep = self._calculate_cpa(
                lat1, lon1, alt1, hdg1, spd1, vs1,
                lat2, lon2, alt2, hdg2, spd2, vs2
            )
            
            # Key improvement: Check if conflict will happen within 300s
            # Even if initial separation at 300s is 10nm but CPA is 2nm after 250s, that's a collision
            conflict_detected = (
                time_to_cpa <= self.TIME_HORIZON_SEC and  # Within 300 seconds
                time_to_cpa > 0 and  # CPA is in the future
                (min_h_sep < self.CONFLICT_THRESHOLD_NM or min_v_sep < self.CONFLICT_THRESHOLD_FT)
            )
            
            # Check if ICAO separation standards are violated
            violates_icao = (
                min_h_sep < self.CONFLICT_THRESHOLD_NM and 
                min_v_sep < self.CONFLICT_THRESHOLD_FT
            )
            
            # Assess severity
            severity = self._assess_conflict_severity(min_h_sep, min_v_sep, time_to_cpa, violates_icao)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(method, min_h_sep, min_v_sep, time_to_cpa)
            
            return ConflictData(
                aircraft_1=ac1_id,
                aircraft_2=ac2_id,
                time_to_cpa=time_to_cpa,
                distance_at_cpa=distance_at_cpa,
                current_horizontal_separation=current_h_sep,
                current_vertical_separation=current_v_sep,
                min_horizontal_separation=min_h_sep,
                min_vertical_separation=min_v_sep,
                conflict_detected=conflict_detected,
                violates_icao_separation=violates_icao,
                severity=severity,
                detection_method=method,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.exception(f"Failed to analyze aircraft pair {ac1_idx}, {ac2_idx}: {e}")
            return None
            
    def _calculate_cpa(self, lat1: float, lon1: float, alt1: float, hdg1: float, spd1: float, vs1: float,
                      lat2: float, lon2: float, alt2: float, hdg2: float, spd2: float, vs2: float) -> Tuple[float, float, float, float]:
        """
        Calculate Closest Point of Approach (CPA) for two aircraft
        
        Returns:
            Tuple of (time_to_cpa, distance_at_cpa, min_horizontal_sep, min_vertical_sep)
        """
        try:
            # Convert to Cartesian coordinates (simplified for local area)
            # Using small angle approximation
            earth_radius_nm = 3440.065  # Earth radius in nautical miles
            
            # Convert lat/lon to local Cartesian (NM)
            x1 = lon1 * math.cos(math.radians(lat1)) * 60
            y1 = lat1 * 60
            z1 = alt1
            
            x2 = lon2 * math.cos(math.radians(lat2)) * 60  
            y2 = lat2 * 60
            z2 = alt2
            
            # Convert velocity to Cartesian components (NM/s)
            spd1_nm_per_sec = spd1 / 3600  # knots to NM/s
            spd2_nm_per_sec = spd2 / 3600
            
            vx1 = spd1_nm_per_sec * math.sin(math.radians(hdg1))
            vy1 = spd1_nm_per_sec * math.cos(math.radians(hdg1))
            vz1 = vs1 / 60  # fpm to ft/s
            
            vx2 = spd2_nm_per_sec * math.sin(math.radians(hdg2))
            vy2 = spd2_nm_per_sec * math.cos(math.radians(hdg2))
            vz2 = vs2 / 60  # fpm to ft/s
            
            # Relative position and velocity
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            
            dvx = vx2 - vx1
            dvy = vy2 - vy1  
            dvz = vz2 - vz1
            
            # Calculate time to CPA
            rel_speed_horizontal = math.sqrt(dvx**2 + dvy**2)
            rel_speed_3d = math.sqrt(dvx**2 + dvy**2 + (dvz/6076)**2)  # Convert ft to NM for 3D speed
            
            if rel_speed_3d < 1e-6:  # Aircraft moving in parallel
                time_to_cpa = 0
                distance_at_cpa = math.sqrt(dx**2 + dy**2)
                min_h_sep = distance_at_cpa
                min_v_sep = abs(dz)
            else:
                # Dot product method for CPA
                dot_product = dx * dvx + dy * dvy + (dz/6076) * (dvz/6076)
                time_to_cpa = -dot_product / (rel_speed_3d**2)
                
                if time_to_cpa < 0:
                    # CPA in the past, aircraft diverging
                    time_to_cpa = 0
                    distance_at_cpa = math.sqrt(dx**2 + dy**2)
                    min_h_sep = distance_at_cpa
                    min_v_sep = abs(dz)
                else:
                    # Project to CPA time
                    future_dx = dx + dvx * time_to_cpa
                    future_dy = dy + dvy * time_to_cpa
                    future_dz = dz + dvz * time_to_cpa
                    
                    distance_at_cpa = math.sqrt(future_dx**2 + future_dy**2)
                    min_h_sep = distance_at_cpa
                    min_v_sep = abs(future_dz)
            
            return time_to_cpa, distance_at_cpa, min_h_sep, min_v_sep
            
        except Exception as e:
            self.logger.exception(f"CPA calculation failed: {e}")
            # Return safe defaults
            return 999999, 999999, 999999, 999999
            
    def _calculate_horizontal_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate horizontal distance between two points in nautical miles"""
        # Haversine formula
        R = 3440.065  # Earth radius in nautical miles
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
        
    def _assess_conflict_severity(self, h_sep: float, v_sep: float, time_to_cpa: float, violates_icao: bool) -> str:
        """Assess conflict severity based on ICAO standards and time to conflict"""
        if violates_icao:
            if time_to_cpa <= 60:  # Critical: conflict within 1 minute
                return "critical"
            elif time_to_cpa <= 120:  # High: conflict within 2 minutes
                return "high" 
            else:
                return "medium"
        elif h_sep < self.CONFLICT_THRESHOLD_NM * 0.6 or v_sep < self.CONFLICT_THRESHOLD_FT * 0.6:
            return "medium"  # Near-miss scenario
        else:
            return "low"
            
    def _calculate_confidence(self, method: str, h_sep: float, v_sep: float, time_to_cpa: float) -> float:
        """Calculate confidence score for conflict detection"""
        base_confidence = {
            "SWARM": 0.9,
            "STATEBASED": 0.85,
            "ENHANCED": 0.8
        }.get(method, 0.7)
        
        # Adjust based on separation margins
        if h_sep < self.CONFLICT_THRESHOLD_NM * 0.5:
            base_confidence += 0.1  # Very close - high confidence
        elif h_sep > self.CONFLICT_THRESHOLD_NM * 1.5:
            base_confidence -= 0.2  # Far apart - lower confidence
            
        # Adjust based on time to conflict
        if time_to_cpa < 60:
            base_confidence += 0.1  # Imminent - high confidence
        elif time_to_cpa > 240:
            base_confidence -= 0.1  # Far future - lower confidence
            
        return max(0.0, min(1.0, base_confidence))
        
    def _cross_validate_conflicts(self, all_conflicts: dict) -> List[ConflictData]:
        """Cross-validate conflicts detected by multiple methods"""
        validated_conflicts = []
        
        for aircraft_pair, detections in all_conflicts.items():
            if len(detections) == 1:
                # Only one method detected - use with reduced confidence
                conflict = detections[0]
                conflict.confidence *= 0.8
                validated_conflicts.append(conflict)
            else:
                # Multiple methods detected - merge and increase confidence
                merged_conflict = self._merge_conflict_detections(detections)
                validated_conflicts.append(merged_conflict)
                
        return validated_conflicts
        
    def _merge_conflict_detections(self, detections: List[ConflictData]) -> ConflictData:
        """Merge multiple conflict detections for the same aircraft pair"""
        # Use the detection with highest confidence as base
        base_detection = max(detections, key=lambda x: x.confidence)
        
        # Average key metrics from all detections
        avg_time_to_cpa = sum(d.time_to_cpa for d in detections) / len(detections)
        avg_distance_at_cpa = sum(d.distance_at_cpa for d in detections) / len(detections)
        
        # Determine consensus on conflict detection
        conflict_votes = sum(1 for d in detections if d.conflict_detected)
        conflict_detected = conflict_votes > len(detections) / 2
        
        # Boost confidence for consensus
        boosted_confidence = min(1.0, base_detection.confidence + 0.1 * (len(detections) - 1))
        
        # Create merged detection
        merged = ConflictData(
            aircraft_1=base_detection.aircraft_1,
            aircraft_2=base_detection.aircraft_2,
            time_to_cpa=avg_time_to_cpa,
            distance_at_cpa=avg_distance_at_cpa,
            current_horizontal_separation=base_detection.current_horizontal_separation,
            current_vertical_separation=base_detection.current_vertical_separation,
            min_horizontal_separation=base_detection.min_horizontal_separation,
            min_vertical_separation=base_detection.min_vertical_separation,
            conflict_detected=conflict_detected,
            violates_icao_separation=base_detection.violates_icao_separation,
            severity=base_detection.severity,
            detection_method=f"MULTI({len(detections)})",
            confidence=boosted_confidence
        )
        
        return merged
        
    def _get_aircraft_pair_key(self, ac1: str, ac2: str) -> Tuple[str, str]:
        """Get consistent key for aircraft pair (sorted order)"""
        return tuple(sorted([ac1, ac2]))
        
    def validate_llm_conflicts(self, llm_conflicts: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """
        Validate LLM-detected conflicts against BlueSky ground truth
        Returns list of (aircraft1, aircraft2, confidence) for validated conflicts
        """
        validated = []
        ground_truth_conflicts = self.detect_conflicts_comprehensive()
        
        # Create lookup set for ground truth
        gt_pairs = set()
        gt_confidence = {}
        for gt in ground_truth_conflicts:
            key = self._get_aircraft_pair_key(gt.aircraft_1, gt.aircraft_2)
            gt_pairs.add(key)
            gt_confidence[key] = gt.confidence
            
        # Validate each LLM conflict
        for llm_pair in llm_conflicts:
            if len(llm_pair) >= 2:
                key = self._get_aircraft_pair_key(llm_pair[0], llm_pair[1])
                if key in gt_pairs:
                    # True positive - LLM correctly detected conflict
                    confidence = gt_confidence[key]
                    validated.append((llm_pair[0], llm_pair[1], confidence))
                    
        return validated
        
    def _mock_conflict_detection(self) -> List[ConflictData]:
        """Mock conflict detection when BlueSky is not available"""
        return [
            ConflictData(
                aircraft_1="AC001",
                aircraft_2="AC002", 
                time_to_cpa=180.0,
                distance_at_cpa=3.2,
                current_horizontal_separation=8.5,
                current_vertical_separation=500,
                min_horizontal_separation=3.2,
                min_vertical_separation=500,
                conflict_detected=True,
                violates_icao_separation=True,
                severity="high",
                detection_method="MOCK",
                confidence=0.9
            )
        ]
