"""
data_cleaning.py

Data Cleaning Layer for Smart AI Data Intelligence System.

Features:
- Intelligent missing handling
- Safe datetime detection
- Duplicate removal
- Identifier detection
- Outlier capping
- Data quality scoring
"""

import pandas as pd
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, List


# ============================================================
# Cleaned Data Object
# ============================================================

@dataclass
class CleanedDataObject:
    cleaned_df: pd.DataFrame
    quality_score: float
    identifiers: List[str]

    def to_dict(self):
        return {
            "quality_score": self.quality_score,
            "identifiers": self.identifiers
        }


# ============================================================
# Data Cleaning Engine
# ============================================================

class DataCleaningEngine:

    def __init__(self, config: Dict = None):
        self.config = config or {}

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(self, df: pd.DataFrame) -> CleanedDataObject:

        df = df.copy()

        initial_rows = len(df)

        # ----------------------------------------------------
        # 1️⃣ Remove Duplicates
        # ----------------------------------------------------
        df = df.drop_duplicates()

        # ----------------------------------------------------
        # 2️⃣ Safe Type Correction
        # ----------------------------------------------------
        df = self._correct_types(df)

        # ----------------------------------------------------
        # 3️⃣ Missing Value Handling
        # ----------------------------------------------------
        df = self._handle_missing(df)

        # ----------------------------------------------------
        # 4️⃣ Outlier Capping
        # ----------------------------------------------------
        df = self._handle_outliers(df)

        # ----------------------------------------------------
        # 5️⃣ Identifier Detection
        # ----------------------------------------------------
        identifiers = [
            col for col in df.columns
            if "id" in col.lower()
        ]

        # ----------------------------------------------------
        # 6️⃣ Data Quality Score
        # ----------------------------------------------------
        missing_ratio = df.isna().mean().mean()
        duplicate_ratio = (initial_rows - len(df)) / max(initial_rows, 1)

        quality_score = 1 - (0.5 * missing_ratio + 0.5 * duplicate_ratio)
        quality_score = round(float(max(0, min(1, quality_score))), 3)

        return CleanedDataObject(
            cleaned_df=df,
            quality_score=quality_score,
            identifiers=identifiers
        )

    # ========================================================
    # Type Correction
    # ========================================================

    def _correct_types(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.columns:

            # Try numeric conversion
            if df[col].dtype == "object":

                numeric_conversion = pd.to_numeric(df[col], errors="coerce")

                if numeric_conversion.notna().sum() > 0.8 * len(df):
                    df[col] = numeric_conversion
                    continue

                # Try datetime detection safely
                sample_values = df[col].dropna().astype(str).head(5)

                if any(('-' in val or '/' in val) for val in sample_values):

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        converted = pd.to_datetime(
                            df[col],
                            errors="coerce",
                            infer_datetime_format=True
                        )

                    if converted.notna().sum() > 0.8 * len(df):
                        df[col] = converted

        return df

    # ========================================================
    # Missing Handling
    # ========================================================

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")

        return df

    # ========================================================
    # Outlier Handling
    # ========================================================

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = df.select_dtypes(include=np.number).columns

        # Ensure float for safe replacement
        df[numeric_cols] = df[numeric_cols].astype(float)

        for col in numeric_cols:

            mean = df[col].mean()
            std = df[col].std()

            if std == 0:
                continue

            z_scores = (df[col] - mean) / std

            df.loc[z_scores > 3, col] = mean + 3 * std
            df.loc[z_scores < -3, col] = mean - 3 * std

        return df
