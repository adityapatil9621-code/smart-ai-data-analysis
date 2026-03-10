"""
suggestion_engine.py

Strategic Recommendation Engine for Smart AI Data Intelligence System.

This module:
- Converts analytical intelligence into structured strategic suggestions
- Uses deterministic rule-based logic
- Maintains human-in-the-loop philosophy
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


# ============================================================
# Strategic Object
# ============================================================

@dataclass
class StrategicObject:
    growth_opportunities: List[str]
    risk_mitigation_actions: List[str]
    stability_recommendations: List[str]
    confidence_advisory: str
    priority_level: str
    human_oversight_note: str

    def to_dict(self):
        return self.__dict__


# ============================================================
# Suggestion Engine
# ============================================================

class SuggestionEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(
        self,
        insight_obj,
        forecast_obj: Optional[object],
        score_obj
    ) -> StrategicObject:

        growth = []
        risk_actions = []
        stability = []

        signal_strength = insight_obj["overall_signal_strength"]
        risk_score = insight_obj["risk_score"]

        # ----------------------------------------------------
        # 1️⃣ Driver-Based Strategy
        # ----------------------------------------------------
        for driver in insight_obj["top_positive_drivers"]:
            if driver["impact"] > 0.2:
                growth.append(
                    f"Consider leveraging '{driver['feature']}' as it shows strong positive influence."
                )

        for driver in insight_obj["top_negative_drivers"]:
            if driver["impact"] > 0.15:
                risk_actions.append(
                    f"Review and stabilize '{driver['feature']}' to reduce negative impact."
                )

        # ----------------------------------------------------
        # 2️⃣ Forecast-Based Strategy
        # ----------------------------------------------------
        if forecast_obj:

            if forecast_obj["trend_direction"] == "Upward":
                growth.append(
                    "Forecast indicates upward trend; expansion strategies may be evaluated."
                )

            elif forecast_obj["trend_direction"] == "Downward":
                risk_actions.append(
                    "Forecast shows potential downward movement; proactive risk containment is advisable."
                )

            if forecast_obj["volatility_score"] > 0.5:
                risk_actions.append(
                    "High forecast volatility detected; implement stability monitoring mechanisms."
                )

            elif forecast_obj["volatility_score"] < 0.2:
                stability.append(
                    "Forecast stability appears strong; optimization strategies may be considered."
                )

        # ----------------------------------------------------
        # 3️⃣ Signal Strength Advisory
        # ----------------------------------------------------
        if signal_strength < 0.5:
            stability.append(
                "Model signal strength is moderate; interpret findings with analytical caution."
            )
        elif signal_strength > 0.8:
            stability.append(
                "Strong analytical signal detected; insights appear robust."
            )

        # ----------------------------------------------------
        # 4️⃣ Risk Advisory
        # ----------------------------------------------------
        if risk_score > 0.6:
            risk_actions.append(
                "Elevated analytical risk detected; further data validation is recommended."
            )

        # ----------------------------------------------------
        # 5️⃣ Priority Level
        # ----------------------------------------------------
        if risk_score > 0.6:
            priority = "High"
        elif signal_strength > 0.75:
            priority = "Medium"
        else:
            priority = "Moderate"

        # ----------------------------------------------------
        # 6️⃣ Confidence Advisory
        # ----------------------------------------------------
        if score_obj["confidence_level"] == "High":

            confidence_note = "Overall analytical confidence is high."
        elif score_obj["confidence_level"] == "Moderate":
            confidence_note = "Analytical confidence is moderate."
        else:
            confidence_note = "Analytical confidence is limited; cautious interpretation advised."

        # ----------------------------------------------------
        # 7️⃣ Human Oversight Clause
        # ----------------------------------------------------
        human_note = (
            "These recommendations are derived from historical data patterns "
            "and model-based analysis. Final strategic decisions should incorporate "
            "external factors and human expertise."
        )

        return StrategicObject(
            growth_opportunities=growth,
            risk_mitigation_actions=risk_actions,
            stability_recommendations=stability,
            confidence_advisory=confidence_note,
            priority_level=priority,
            human_oversight_note=human_note
        )
