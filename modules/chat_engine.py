"""
chat_engine.py

Controlled Chat Engine for Smart AI Data Intelligence System.

This module:
- Detects user intent
- Extracts structured intelligence from SystemMemory
- Constructs safe analytical response
- Sends structured response to LLM for rewriting
"""

from typing import Dict
import requests  # Use your OpenAI API or LLM provider


# ============================================================
# Chat Engine
# ============================================================

class ChatEngine:

    def __init__(self):
        pass

    # ========================================================
    # MAIN CHAT METHOD
    # ========================================================

    def respond(self, user_query: str, system_memory: Dict) -> str:

        intent = self._detect_intent(user_query)

        structured_response = self._build_structured_response(
            intent,
            system_memory
        )

        final_response = self._rewrite_with_llm(structured_response)

        return final_response

    # ========================================================
    # Intent Detection (Deterministic)
    # ========================================================

    def _detect_intent(self, query: str) -> str:

        query = query.lower()

        if "forecast" in query or "future" in query or "trend" in query:
            return "forecast"

        if "driver" in query or "cause" in query or "influence" in query:
            return "drivers"

        if "risk" in query or "problem" in query:
            return "risk"

        if "anomaly" in query or "outlier" in query:
            return "anomalies"

        if "confidence" in query:
            return "confidence"

        if "recommend" in query or "suggest" in query:
            return "recommendation"

        return "summary"

    # ========================================================
    # Structured Response Builder
    # ========================================================

    def _build_structured_response(self, intent: str, memory: Dict) -> str:

        metadata = memory["metadata"]
        model_info = memory["model_intelligence"]
        insight = memory["insight_intelligence"]
        forecast = memory.get("forecast_intelligence")
        score = memory["intelligence_score"]

        response = ""

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------
        if intent == "summary":

            response = (
                f"The dataset contains {metadata['rows']} records "
                f"with a quality score of {metadata['quality_score']}. "
                f"The selected model is {model_info['selected_model']} "
                f"with confidence level {score['confidence_level']}. "
                f"Overall intelligence grade: {score['grade']}."
            )

        # ----------------------------------------------------
        # DRIVERS
        # ----------------------------------------------------
        elif intent == "drivers":

            response = (
                f"Top positive drivers: {insight['top_positive_drivers']}. "
                f"Top negative drivers: {insight['top_negative_drivers']}."
            )

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------
        elif intent == "risk":

            response = (
                f"Risk score is {insight['risk_score']}. "
                f"Residual bias detected: {insight['residual_bias_detected']}."
            )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------
        elif intent == "forecast" and forecast:

            response = (
                f"Forecast trend direction: {forecast['trend_direction']}. "
                f"Volatility score: {forecast['volatility_score']}. "
                f"Forecast confidence: {forecast['forecast_confidence']}."
            )

        # ----------------------------------------------------
        # ANOMALIES
        # ----------------------------------------------------
        elif intent == "anomalies":

            response = (
                f"Anomalies detected: {len(insight['anomalies_detected'])} cases."
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------
        elif intent == "recommendation":

            response = (
                f"Based on analytical insights, "
                f"priority level is {score['confidence_level']}. "
                f"Consider reviewing key drivers and managing identified risks."
            )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------
        elif intent == "confidence":

            response = (
                f"Overall intelligence score is {score['score']} "
                f"with grade {score['grade']} "
                f"and confidence level {score['confidence_level']}."
            )

        else:
            response = "Insufficient data to provide requested insight."

        return response

    # ========================================================
    # LLM Rewrite Layer (Controlled)
    # ========================================================

    def _rewrite_with_llm(self, structured_text: str) -> str:

        prompt = (
            "Rewrite the following structured analytical summary into "
            "clear, professional, natural language. "
            "Do not add new data. Do not invent statistics. "
            "Keep a balanced decision-support tone.\n\n"
            f"{structured_text}"
        )

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                }
            )

            response.raise_for_status()

            return response.json()["response"]

        except Exception as e:
            return f"LLM Error: {str(e)}"
