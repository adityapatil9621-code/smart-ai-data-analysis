"""
scoring_engine.py

Unified Intelligence Scoring Engine for Smart AI Data Intelligence System.

This module:
- Computes final intelligence score
- Assigns grade
- Computes overall confidence level
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================
# Score Object
# ============================================================

@dataclass
class IntelligenceScoreObject:
    score: float
    grade: str
    confidence_level: str

    def to_dict(self):
        return self.__dict__


# ============================================================
# Scoring Engine
# ============================================================

class IntelligenceScoringEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(
        self,
        cleaned_obj,
        model_obj,
        insight_obj,
        forecast_obj: Optional[object]
    ) -> IntelligenceScoreObject:

        data_quality = cleaned_obj.quality_score
        model_conf = model_obj.confidence
        signal_strength = insight_obj.overall_signal_strength
        risk_score = insight_obj.risk_score

        forecast_conf = 0

        if forecast_obj:
            forecast_conf = forecast_obj.forecast_confidence

            score = (
                0.25 * data_quality +
                0.30 * model_conf +
                0.20 * signal_strength +
                0.15 * forecast_conf -
                0.10 * risk_score
            )

        else:
            # Redistribute forecast weight
            score = (
                0.30 * data_quality +
                0.35 * model_conf +
                0.25 * signal_strength -
                0.10 * risk_score
            )

        score = max(0, min(1, score))
        score = round(float(score), 3)

        # ----------------------------------------------------
        # Grade Assignment
        # ----------------------------------------------------
        if score >= 0.85:
            grade = "A"
        elif score >= 0.7:
            grade = "B"
        elif score >= 0.55:
            grade = "C"
        else:
            grade = "D"

        # ----------------------------------------------------
        # Confidence Level
        # ----------------------------------------------------
        if score >= 0.8:
            confidence_level = "High"
        elif score >= 0.6:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        return IntelligenceScoreObject(
            score=score,
            grade=grade,
            confidence_level=confidence_level
        )
