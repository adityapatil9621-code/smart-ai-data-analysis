"""
forecasting_engine.py

Tree-Based Lag Forecasting Engine for Smart AI Data Intelligence System.

This module:
- Creates lag features
- Trains Gradient Boosting model
- Performs recursive forecasting
- Generates confidence intervals (bootstrap)
- Detects trend direction
- Computes volatility & forecast confidence
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# ============================================================
# Forecast Object
# ============================================================

@dataclass
class ForecastObject:
    forecast_horizon: int
    forecast_values: List[float]
    confidence_band: Dict[str, List[float]]
    trend_direction: str
    volatility_score: float
    forecast_confidence: float

    def to_dict(self):
        return self.__dict__


# ============================================================
# Forecast Engine
# ============================================================

class ForecastEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.forecast_horizon = self.config.get("forecast_horizon", 6)
        self.bootstrap_iterations = self.config.get("bootstrap_iterations", 20)
        self.random_state = self.config.get("random_state", 42)
        self.lags = [1, 2, 3, 6]

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, df: pd.DataFrame, understanding_obj) -> ForecastObject:

        target = understanding_obj.target_column
        time_col = understanding_obj.time_column

        df = df.sort_values(by=time_col).reset_index(drop=True)

        series = df[target].values

        if len(series) < 20:
            raise ValueError("Not enough data for forecasting.")

        # ----------------------------------------------------
        # 1️⃣ Create Lag Features
        # ----------------------------------------------------
        lag_df = self._create_lag_features(series)

        X = lag_df.drop(columns=["target"])
        y = lag_df["target"]

        split_index = int(len(X) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # ----------------------------------------------------
        # 2️⃣ Train Model
        # ----------------------------------------------------
        model = GradientBoostingRegressor(
            random_state=self.random_state
        )

        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        model_r2 = r2_score(y_test, y_pred_test)

        # ----------------------------------------------------
        # 3️⃣ Recursive Forecast
        # ----------------------------------------------------
        forecast_values = self._recursive_forecast(
            model,
            series,
            self.forecast_horizon
        )

        # ----------------------------------------------------
        # 4️⃣ Bootstrap Confidence Band
        # ----------------------------------------------------
        lower_band, upper_band = self._bootstrap_confidence(
            series,
            self.forecast_horizon
        )

        # ----------------------------------------------------
        # 5️⃣ Trend Detection
        # ----------------------------------------------------
        slope = np.polyfit(
            range(len(forecast_values)),
            forecast_values,
            1
        )[0]

        if slope > 0:
            trend = "Upward"
        elif slope < 0:
            trend = "Downward"
        else:
            trend = "Stable"

        # ----------------------------------------------------
        # 6️⃣ Volatility Score
        # ----------------------------------------------------
        volatility = np.std(forecast_values) / (np.mean(forecast_values) + 1e-8)
        volatility = max(0, min(1, volatility))

        # ----------------------------------------------------
        # 7️⃣ Forecast Confidence
        # ----------------------------------------------------
        forecast_confidence = (
            0.6 * model_r2 +
            0.2 * (1 - volatility) +
            0.2 * (1 - abs(slope) / (abs(np.mean(series)) + 1e-8))
        )

        forecast_confidence = max(0, min(1, forecast_confidence))

        return ForecastObject(
            forecast_horizon=self.forecast_horizon,
            forecast_values=[round(float(v), 3) for v in forecast_values],
            confidence_band={
                "lower": [round(float(v), 3) for v in lower_band],
                "upper": [round(float(v), 3) for v in upper_band]
            },
            trend_direction=trend,
            volatility_score=round(float(volatility), 3),
            forecast_confidence=round(float(forecast_confidence), 3)
        )

    # ========================================================
    # Lag Feature Creation
    # ========================================================

    def _create_lag_features(self, series):

        df = pd.DataFrame({"target": series})

        for lag in self.lags:

            df[f"lag_{lag}"] = df["target"].shift(lag)

        df.dropna(inplace=True)

        return df

    # ========================================================
    # Recursive Forecast
    # ========================================================

    def _recursive_forecast(self, model, series, horizon):

        history = list(series[-max(self.lags):])

        forecast = []

        for _ in range(horizon):
            features = pd.DataFrame(
                [[history[-lag] for lag in self.lags]],
                columns=[f"lag_{lag}" for lag in self.lags]
            )

            prediction = model.predict(features)[0]

            forecast.append(prediction)
            history.append(prediction)

        return forecast

    # ========================================================
    # Bootstrap Confidence
    # ========================================================

    def _bootstrap_confidence(self, series, horizon):

        forecasts = []

        for _ in range(self.bootstrap_iterations):

            sample = np.random.choice(series, size=len(series), replace=True)
            lag_df = self._create_lag_features(sample)

            X = lag_df.drop(columns=["target"])
            y = lag_df["target"]

            model = GradientBoostingRegressor()
            model.fit(X, y)

            forecast = self._recursive_forecast(model, sample, horizon)
            forecasts.append(forecast)

        forecasts = np.array(forecasts)

        lower = np.percentile(forecasts, 5, axis=0)
        upper = np.percentile(forecasts, 95, axis=0)

        return lower, upper
