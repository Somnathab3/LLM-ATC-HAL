# baseline_models/conflict_detector.py
"""
Baseline Conflict Detector using RandomForest and XGBoost
Traditional ML-based binary classifier for conflict detection
"""

import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


@dataclass
class ConflictPrediction:
    """Conflict detection prediction result"""

    has_conflict: bool
    confidence: float
    time_to_conflict: float
    conflict_pairs: list[tuple[str, str]]
    risk_factors: dict[str, float]


class BaselineConflictDetector:
    """
    Traditional ML-based conflict detector using RandomForest/XGBoost.
    Serves as baseline comparison for LLM-based detection.
    """

    def __init__(self, model_type: str = "random_forest") -> None:
        """
        Initialize baseline conflict detector.

        Args:
            model_type: Either "random_forest" or "xgboost"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.logger = logging.getLogger(__name__)

        # Check dependencies
        if model_type == "random_forest" and not SKLEARN_AVAILABLE:
            msg = "scikit-learn required for RandomForest model"
            raise ImportError(msg)
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            msg = "xgboost required for XGBoost model"
            raise ImportError(msg)

        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)

    def extract_features(self, scenario: dict[str, Any]) -> np.ndarray:
        """
        Extract features from scenario for ML model.

        Args:
            scenario: Scenario dictionary with aircraft states

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Get aircraft data
        aircraft_data = scenario.get("aircraft", [])

        if len(aircraft_data) < 2:
            # Not enough aircraft for conflict
            return np.array([0] * 20)  # Return zero vector

        # Basic scenario features
        features.extend(
            [
                len(aircraft_data),  # Number of aircraft
                scenario.get("time_horizon", 600),  # Time horizon
                scenario.get("traffic_density", 0.5),  # Traffic density
                scenario.get("weather_severity", 0.0),  # Weather severity
            ],
        )

        # Pairwise aircraft features (take first two for simplicity)
        ac1, ac2 = aircraft_data[0], aircraft_data[1]

        # Position differences
        lat_diff = abs(ac1.get("lat", 0) - ac2.get("lat", 0))
        lon_diff = abs(ac1.get("lon", 0) - ac2.get("lon", 0))
        alt_diff = abs(ac1.get("alt", 0) - ac2.get("alt", 0))

        features.extend([lat_diff, lon_diff, alt_diff])

        # Velocity differences
        speed_diff = abs(ac1.get("speed", 0) - ac2.get("speed", 0))
        heading_diff = abs(ac1.get("heading", 0) - ac2.get("heading", 0))
        vspeed_diff = abs(ac1.get("vertical_speed", 0) - ac2.get("vertical_speed", 0))

        features.extend([speed_diff, heading_diff, vspeed_diff])

        # Derived features
        horizontal_distance = np.sqrt(lat_diff**2 + lon_diff**2)
        relative_speed = np.sqrt(speed_diff**2 + vspeed_diff**2)
        approach_rate = horizontal_distance / max(relative_speed, 1.0)  # Avoid division by zero

        features.extend([horizontal_distance, relative_speed, approach_rate])

        # Aircraft types and capabilities
        ac1_type = self._encode_aircraft_type(ac1.get("type", "unknown"))
        ac2_type = self._encode_aircraft_type(ac2.get("type", "unknown"))

        features.extend([ac1_type, ac2_type])

        # Flight phases
        ac1_phase = self._encode_flight_phase(ac1.get("flight_phase", "cruise"))
        ac2_phase = self._encode_flight_phase(ac2.get("flight_phase", "cruise"))

        features.extend([ac1_phase, ac2_phase])

        # Pad or truncate to exactly 20 features
        while len(features) < 20:
            features.append(0.0)
        features = features[:20]

        return np.array(features, dtype=np.float32)

    def train(self, training_data: list[dict[str, Any]], labels: list[bool]) -> dict[str, float]:
        """
        Train the baseline model.

        Args:
            training_data: List of scenario dictionaries
            labels: List of boolean conflict labels

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            msg = "scikit-learn not available for training"
            raise RuntimeError(msg)

        # Extract features
        X = np.array([self.extract_features(scenario) for scenario in training_data])
        y = np.array(labels)

        # Store feature names for reference
        self.feature_names = [
            "num_aircraft",
            "time_horizon",
            "traffic_density",
            "weather_severity",
            "lat_diff",
            "lon_diff",
            "alt_diff",
            "speed_diff",
            "heading_diff",
            "vspeed_diff",
            "horizontal_distance",
            "relative_speed",
            "approach_rate",
            "ac1_type",
            "ac2_type",
            "ac1_phase",
            "ac2_phase",
            "feature_18",
            "feature_19",
            "feature_20",
        ]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }

        self.logger.info("Model trained with metrics: %s", metrics)
        return metrics

    def predict(self, scenario: dict[str, Any]) -> ConflictPrediction:
        """
        Predict conflicts in scenario.

        Args:
            scenario: Scenario dictionary

        Returns:
            ConflictPrediction object
        """
        if not self.is_trained:
            msg = "Model must be trained before prediction"
            raise RuntimeError(msg)

        # Extract features
        features = self.extract_features(scenario).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Make prediction
        has_conflict = bool(self.model.predict(features_scaled)[0])
        confidence = float(self.model.predict_proba(features_scaled)[0, 1])

        # Estimate time to conflict (simplified)
        time_to_conflict = self._estimate_time_to_conflict(scenario, features[0])

        # Identify conflict pairs (simplified - assume first two aircraft)
        aircraft_data = scenario.get("aircraft", [])
        conflict_pairs = []
        if len(aircraft_data) >= 2 and has_conflict:
            conflict_pairs = [
                (aircraft_data[0].get("id", "AC1"), aircraft_data[1].get("id", "AC2")),
            ]

        # Analyze risk factors
        risk_factors = self._analyze_risk_factors(features[0])

        return ConflictPrediction(
            has_conflict=has_conflict,
            confidence=confidence,
            time_to_conflict=time_to_conflict,
            conflict_pairs=conflict_pairs,
            risk_factors=risk_factors,
        )

    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if not self.is_trained:
            msg = "No trained model to save"
            raise RuntimeError(msg)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info("Model saved to %s", filepath)

    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_type = model_data["model_type"]
        self.feature_names = model_data["feature_names"]
        self.is_trained = model_data["is_trained"]

        self.logger.info("Model loaded from %s", filepath)

    def _encode_aircraft_type(self, aircraft_type: str) -> float:
        """Encode aircraft type as numeric value"""
        type_mapping = {
            "commercial": 1.0,
            "cargo": 2.0,
            "private": 3.0,
            "military": 4.0,
            "unknown": 0.0,
        }
        return type_mapping.get(aircraft_type.lower(), 0.0)

    def _encode_flight_phase(self, flight_phase: str) -> float:
        """Encode flight phase as numeric value"""
        phase_mapping = {
            "takeoff": 1.0,
            "climb": 2.0,
            "cruise": 3.0,
            "descent": 4.0,
            "approach": 5.0,
            "landing": 6.0,
            "unknown": 0.0,
        }
        return phase_mapping.get(flight_phase.lower(), 3.0)  # Default to cruise

    def _estimate_time_to_conflict(self, scenario: dict[str, Any], features: np.ndarray) -> float:
        """Estimate time to conflict based on features"""
        # Simplified calculation based on approach rate
        approach_rate = features[12] if len(features) > 12 else 1.0
        horizontal_distance = features[10] if len(features) > 10 else 10.0

        if approach_rate > 0:
            return max(0.0, horizontal_distance / approach_rate * 60)  # Convert to seconds
        return 600.0  # Default 10 minutes

    def _analyze_risk_factors(self, features: np.ndarray) -> dict[str, float]:
        """Analyze risk factors from features"""
        risk_factors = {}

        if len(features) >= 20:
            risk_factors["proximity_risk"] = 1.0 - min(
                features[10] / 10.0, 1.0,
            )  # Horizontal distance
            risk_factors["altitude_risk"] = min(features[6] / 1000.0, 1.0)  # Altitude difference
            risk_factors["speed_risk"] = min(features[7] / 100.0, 1.0)  # Speed difference
            risk_factors["approach_risk"] = 1.0 - min(features[12] / 10.0, 1.0)  # Approach rate
            risk_factors["traffic_density"] = features[2] if features[2] <= 1.0 else 1.0

        return risk_factors


# Example usage and testing
if __name__ == "__main__":
    # Create sample training data
    training_scenarios = []
    labels = []

    # Generate synthetic training data
    for i in range(100):
        scenario = {
            "aircraft": [
                {
                    "id": f"AC{i*2}",
                    "lat": 52.0 + np.random.normal(0, 0.1),
                    "lon": 4.0 + np.random.normal(0, 0.1),
                    "alt": 35000 + np.random.normal(0, 2000),
                    "speed": 250 + np.random.normal(0, 50),
                    "heading": np.random.uniform(0, 360),
                    "vertical_speed": np.random.normal(0, 500),
                    "type": "commercial",
                    "flight_phase": "cruise",
                },
                {
                    "id": f"AC{i*2+1}",
                    "lat": 52.0 + np.random.normal(0, 0.1),
                    "lon": 4.0 + np.random.normal(0, 0.1),
                    "alt": 35000 + np.random.normal(0, 2000),
                    "speed": 250 + np.random.normal(0, 50),
                    "heading": np.random.uniform(0, 360),
                    "vertical_speed": np.random.normal(0, 500),
                    "type": "commercial",
                    "flight_phase": "cruise",
                },
            ],
            "traffic_density": np.random.uniform(0, 1),
            "weather_severity": np.random.uniform(0, 0.5),
        }

        training_scenarios.append(scenario)
        # Random labels for demo (in real use, these would be ground truth)
        labels.append(np.random.random() > 0.7)

    # Test the detector
    if SKLEARN_AVAILABLE:
        detector = BaselineConflictDetector("random_forest")
        metrics = detector.train(training_scenarios, labels)

        # Test prediction
        test_scenario = training_scenarios[0]
        prediction = detector.predict(test_scenario)
    else:
        pass
