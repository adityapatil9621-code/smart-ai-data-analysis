"""
data_understanding.py

Data Understanding Engine for Smart AI Data Intelligence System.

This module:
- Detects task type (regression / classification)
- Detects time-series structure
- Identifies target column
- Identifies numeric & categorical columns
- Detects class imbalance
- Detects skewness
- Computes correlation strength
- Ensures datetime columns are handled properly
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# Understanding Object
# ============================================================

@dataclass
class UnderstandingObject:
    task_type: str
    target_column: str
    time_column: Optional[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    is_time_series: bool
    class_imbalance_ratio: Optional[float]
    skewed_features: List[str]
    correlation_strength: float

    def to_dict(self):
        return self.__dict__


# ============================================================
# Data Understanding Engine
# ============================================================

class DataUnderstandingEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, df: pd.DataFrame) -> UnderstandingObject:

        df = df.copy()

        # ----------------------------------------------------
        # 1️⃣ Identify Column Types
        # ----------------------------------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

        # ----------------------------------------------------
        # 2️⃣ Detect Time-Series
        # ----------------------------------------------------
        time_column = None
        is_time_series = False

        if len(datetime_cols) > 0:
            time_column = datetime_cols[0]  # Choose first datetime column
            is_time_series = True

        # ----------------------------------------------------
        # 3️⃣ Detect Target Column
        # Strategy:
        # If time-series → last numeric column
        # Else → numeric column with highest variance
        # ----------------------------------------------------
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found. Cannot determine target.")

        if is_time_series:
            target_column = numeric_cols[-1]
        else:
            variances = df[numeric_cols].var()
            target_column = variances.idxmax()

        # Remove target from numeric feature list
        numeric_cols = [col for col in numeric_cols if col != target_column]

        # ----------------------------------------------------
        # 4️⃣ Detect Task Type
        # ----------------------------------------------------
        unique_values = df[target_column].nunique()

        if unique_values <= 10 and not pd.api.types.is_float_dtype(df[target_column]):
            task_type = "classification"
        else:
            task_type = "regression"

        # ----------------------------------------------------
        # 5️⃣ Detect Class Imbalance (if classification)
        # ----------------------------------------------------
        class_imbalance_ratio = None

        if task_type == "classification":
            value_counts = df[target_column].value_counts(normalize=True)
            class_imbalance_ratio = round(float(value_counts.max()), 3)

        # ----------------------------------------------------
        # 6️⃣ Detect Skewness
        # ----------------------------------------------------
        skewed_features = []

        for col in numeric_cols:
            if col in df.columns:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    skewed_features.append(col)

        # ----------------------------------------------------
        # 7️⃣ Compute Correlation Strength
        # ----------------------------------------------------
        correlation_strength = 0.0

        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()

            # Remove self-correlation
            np.fill_diagonal(corr_matrix.values, 0)

            correlation_strength = corr_matrix.max().max()

        correlation_strength = round(float(correlation_strength), 3)

        # ----------------------------------------------------
        # RETURN STRUCTURED OBJECT
        # ----------------------------------------------------
        return UnderstandingObject(
            task_type=task_type,
            target_column=target_column,
            time_column=time_column,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            is_time_series=is_time_series,
            class_imbalance_ratio=class_imbalance_ratio,
            skewed_features=skewed_features,
            correlation_strength=correlation_strength
        )
