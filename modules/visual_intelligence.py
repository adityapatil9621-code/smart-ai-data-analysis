"""
visual_intelligence.py

Upgraded Visual Intelligence Engine
- Removes ID columns
- Aggregates time-series monthly
- Adds rolling smoothing
- Produces clean professional visuals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict


# ============================================================
# Visual Object
# ============================================================

@dataclass
class VisualObject:
    primary_chart: str
    selection_reason: str
    visual_confidence: float
    figures: Dict

    def to_dict(self):
        return {
            "primary_chart": self.primary_chart,
            "selection_reason": self.selection_reason,
            "visual_confidence": self.visual_confidence
        }


# ============================================================
# Visual Intelligence Engine
# ============================================================

class VisualIntelligenceEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    def run(self, df: pd.DataFrame, understanding_obj) -> VisualObject:

        df = df.copy()

        figures = {}

        target = understanding_obj.target_column
        time_column = understanding_obj.time_column

        # ----------------------------------------------------
        # 1️⃣ Remove ID-like columns
        # ----------------------------------------------------
        id_cols = [col for col in df.columns if "id" in col.lower()]
        df = df.drop(columns=id_cols, errors="ignore")

        # ----------------------------------------------------
        # 2️⃣ Time-Series Plot (Monthly Aggregated)
        # ----------------------------------------------------
        if understanding_obj.is_time_series and time_column in df.columns:

            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df = df.dropna(subset=[time_column])

            df = df.set_index(time_column)

            monthly = df[target].resample("ME").mean()

            # Rolling smoothing
            smooth = monthly.rolling(window=3, min_periods=1).mean()

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(monthly.index, monthly.values, alpha=0.4, label="Raw Monthly")
            ax1.plot(smooth.index, smooth.values, linewidth=2, label="Smoothed Trend")

            ax1.set_title(f"{target} Trend (Monthly Aggregated)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel(target)
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

            figures["time_series"] = fig1

            primary_chart = "Time-Series Trend"
            selection_reason = "Detected datetime structure in dataset."
            visual_confidence = 0.9

        # ----------------------------------------------------
        # 3️⃣ Correlation Heatmap
        # ----------------------------------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) > 1:

            corr = df[numeric_cols].corr()

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                ax=ax2
            )

            ax2.set_title("Correlation Heatmap (Filtered Numeric Features)")
            figures["heatmap"] = fig2

            if not understanding_obj.is_time_series:
                primary_chart = "Correlation Heatmap"
                selection_reason = "Multiple numeric features detected."
                visual_confidence = 0.85

        # ----------------------------------------------------
        # Fallback if no numeric correlation
        # ----------------------------------------------------
        if not figures:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient numeric data for visualization",
                    ha="center", va="center")
            figures["fallback"] = fig
            primary_chart = "Basic Overview"
            selection_reason = "Limited numeric structure detected."
            visual_confidence = 0.6

        return VisualObject(
            primary_chart=primary_chart,
            selection_reason=selection_reason,
            visual_confidence=visual_confidence,
            figures=figures
        )
