"""
core_engine.py

Central Orchestrator for Smart AI Data Intelligence System.

This module:
- Enforces cleaning-first rule
- Controls execution flow
- Builds structured SystemMemory
- Ensures deterministic intelligence generation
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime

from data_cleaning import DataCleaningEngine


# These will be implemented next modules
# (temporary placeholders until implemented)
from data_understanding import DataUnderstandingEngine
from visual_intelligence import VisualIntelligenceEngine
from feature_engineering import FeatureEngineeringEngine
from model_training import ModelTrainingEngine
from insight_extraction import InsightExtractionEngine
from forecasting_engine import ForecastEngine
from scoring_engine import IntelligenceScoringEngine


# ============================================================
# Core Engine
# ============================================================

class SmartAIEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.system_memory: Dict[str, Any] = {}

        # Initialize modules
        self.cleaning_engine = DataCleaningEngine(config)
        self.understanding_engine = DataUnderstandingEngine(config)
        self.visual_engine = VisualIntelligenceEngine(config)
        self.feature_engine = FeatureEngineeringEngine(config)
        self.model_engine = ModelTrainingEngine(config)
        self.insight_engine = InsightExtractionEngine(config)
        self.forecast_engine = ForecastEngine(config)
        self.scoring_engine = IntelligenceScoringEngine(config)

    # ========================================================
    # MASTER PIPELINE EXECUTION
    # ========================================================

    def run_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # ----------------------------------------------------
        # 1️⃣ CLEANING GATE (MANDATORY)
        # ----------------------------------------------------
        cleaned_obj = self.cleaning_engine.run(df)

        if cleaned_obj.quality_score < 0.4:
            raise ValueError(
                f"Data quality too low ({cleaned_obj.quality_score}). "
                "Please improve dataset before analysis."
            )

        cleaned_df = cleaned_obj.cleaned_df

        # ----------------------------------------------------
        # 2️⃣ DATA UNDERSTANDING
        # ----------------------------------------------------
        understanding_obj = self.understanding_engine.run(cleaned_df)

        # ----------------------------------------------------
        # 3️⃣ VISUAL INTELLIGENCE
        # ----------------------------------------------------
        visual_obj = self.visual_engine.run(cleaned_df, understanding_obj)

        # ----------------------------------------------------
        # 4️⃣ FEATURE ENGINEERING
        # ----------------------------------------------------
        feature_obj = self.feature_engine.run(cleaned_df, understanding_obj)

        # ----------------------------------------------------
        # 5️⃣ MODEL TRAINING
        # ----------------------------------------------------
        model_obj = self.model_engine.run(feature_obj)
        self.model_obj = model_obj

        # ----------------------------------------------------
        # 6️⃣ INSIGHT EXTRACTION
        # ----------------------------------------------------
        insight_obj = self.insight_engine.run(
            model_obj,
            feature_obj,
            cleaned_df
        )

        # ----------------------------------------------------
        # 7️⃣ FORECAST (ONLY IF TIME SERIES)
        # ----------------------------------------------------
        if understanding_obj.is_time_series:
            forecast_obj = self.forecast_engine.run(
                cleaned_df,
                understanding_obj
            )
        else:
            forecast_obj = None

        # ----------------------------------------------------
        # 8️⃣ INTELLIGENCE SCORING
        # ----------------------------------------------------
        score_obj = self.scoring_engine.run(
            cleaned_obj,
            model_obj,
            insight_obj,
            forecast_obj
        )

        # ----------------------------------------------------
        # 9️⃣ BUILD SYSTEM MEMORY
        # ----------------------------------------------------
        self.system_memory = self.build_system_memory(
            cleaned_obj,
            understanding_obj,
            visual_obj,
            model_obj,
            insight_obj,
            forecast_obj,
            score_obj
        )


        return self.system_memory

    # ========================================================
    # SYSTEM MEMORY BUILDER
    # ========================================================

    def build_system_memory(
        self,
        cleaned_obj,
        understanding_obj,
        visual_obj,
        model_obj,
        insight_obj,
        forecast_obj,
        score_obj
    ) -> Dict[str, Any]:

        return {
            "metadata": {
               "rows": cleaned_obj.cleaned_df.shape[0],
                "columns": cleaned_obj.cleaned_df.shape[1],
                "quality_score": cleaned_obj.quality_score
            },


            "data_profile": understanding_obj.to_dict(),

            "visual_intelligence": visual_obj.to_dict(),

            "model_intelligence": model_obj.to_dict(),

            "insight_intelligence": insight_obj.to_dict(),

            "forecast_intelligence": (
                forecast_obj.to_dict() if forecast_obj else None
            ),

            "intelligence_score": score_obj.to_dict(),

            "audit_log": [
                f"[CLEANING] Quality Score = {cleaned_obj.quality_score}",
                f"[MODEL] Selected = {model_obj.selected_model}",
                f"[FORECAST] Enabled = {understanding_obj.is_time_series}",
                f"[SCORE] Final Intelligence Score = {score_obj.score}"
            ]
        }

    # ========================================================
    # GET SYSTEM MEMORY (FOR CHAT ENGINE)
    # ========================================================

    def get_memory(self) -> Dict[str, Any]:
        return self.system_memory
