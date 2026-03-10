"""
insight_extraction.py

Insight Extraction Engine for Smart AI Data Intelligence System.

This module:
- Extracts feature importance
- Detects driver direction
- Identifies nonlinear behavior
- Performs residual diagnostics
- Detects anomalies
- Computes signal strength and risk score
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict


# ============================================================
# Insight Object
# ============================================================

@dataclass
class InsightObject:
    top_positive_drivers: List[Dict]
    top_negative_drivers: List[Dict]
    nonlinear_features: List[str]
    residual_bias_detected: bool
    anomalies_detected: List[Dict]
    overall_signal_strength: float
    risk_score: float

    def to_dict(self):
        return self.__dict__


# ============================================================
# Insight Extraction Engine
# ============================================================

class InsightExtractionEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, model_obj, feature_obj, df: pd.DataFrame) -> InsightObject:

        model = model_obj.trained_model

        X_train = feature_obj.X_train
        X_test = feature_obj.X_test
        y_train = feature_obj.y_train
        y_test = feature_obj.y_test
        feature_names = feature_obj.feature_names

        # ----------------------------------------------------
        # 1️⃣ Feature Importance Extraction
        # ----------------------------------------------------
        importance = self._extract_importance(model, feature_names)

        # Normalize importance
        total_importance = sum(importance.values()) + 1e-8
        normalized_importance = {
            k: v / total_importance
            for k, v in importance.items()
        }

        # ----------------------------------------------------
        # 2️⃣ Direction Detection
        # ----------------------------------------------------
        direction_map = {}
        predictions = model.predict(X_test)

        for feature in feature_names:
            corr = np.corrcoef(X_test[feature], predictions)[0, 1]
            direction_map[feature] = np.sign(corr)

        # Separate positive & negative drivers
        sorted_features = sorted(
            normalized_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_positive = []
        top_negative = []

        for feature, imp in sorted_features[:5]:
            if direction_map.get(feature, 0) >= 0:
                top_positive.append({
                    "feature": feature,
                    "impact": round(float(imp), 3)
                })
            else:
                top_negative.append({
                    "feature": feature,
                    "impact": round(float(-imp), 3)
                })

        # ----------------------------------------------------
        # 3️⃣ Nonlinear Detection
        # Simple: Tree models → nonlinear behavior likely
        # ----------------------------------------------------
        nonlinear_features = []

        if hasattr(model, "feature_importances_"):
            nonlinear_features = list(normalized_importance.keys())[:2]

        # ----------------------------------------------------
        # 4️⃣ Residual Diagnostics
        # ----------------------------------------------------
        residuals = y_test - predictions

        mean_residual = np.mean(residuals)
        residual_bias_detected = abs(mean_residual) > 0.1 * np.std(y_test)

        # ----------------------------------------------------
        # 5️⃣ Anomaly Detection (Z-score)
        # ----------------------------------------------------
        anomalies = []

        if np.std(residuals) > 0:
            z_scores = residuals / np.std(residuals)

            for idx, z in enumerate(z_scores):
                if abs(z) > 3:
                    anomalies.append({
                        "index": int(idx),
                        "severity": round(float(abs(z)), 3)
                    })

        anomaly_ratio = len(anomalies) / len(y_test)

        # ----------------------------------------------------
        # 6️⃣ Signal Strength
        # ----------------------------------------------------
        cv_stability = model_obj.stability
        overfit_gap = 0
        model_conf = model_obj.confidence

        signal_strength = (
            0.5 * model_conf +
            0.3 * cv_stability +
            0.2 * (1 - overfit_gap)
        )

        signal_strength = max(0, min(1, signal_strength))

        # ----------------------------------------------------
        # 7️⃣ Risk Score
        # ----------------------------------------------------
        risk_score = (
            0.4 * anomaly_ratio +
            0.3 * overfit_gap +
            0.3 * (1 - cv_stability)
        )

        risk_score = max(0, min(1, risk_score))

        return InsightObject(
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            nonlinear_features=nonlinear_features,
            residual_bias_detected=residual_bias_detected,
            anomalies_detected=anomalies,
            overall_signal_strength=round(float(signal_strength), 3),
            risk_score=round(float(risk_score), 3)
        )

    # ========================================================
    # Extract Importance
    # ========================================================

    def _extract_importance(self, model, feature_names):

        importance = {}

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            importance = dict(zip(feature_names, values))

        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_)
            importance = dict(zip(feature_names, values))

        else:
            importance = {f: 1 for f in feature_names}

        return importance
