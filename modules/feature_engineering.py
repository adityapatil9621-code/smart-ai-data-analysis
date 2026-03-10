"""
feature_engineering.py

Feature Engineering Engine for Smart AI Data Intelligence System.

This module:
- Encodes categorical features
- Scales numeric features
- Handles skewness
- Removes multicollinearity
- Creates lag features for time-series
- Splits train/test sets
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Feature Object
# ============================================================

@dataclass
class FeatureObject:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list

    def to_dict(self):
        return {
            "feature_count": len(self.feature_names),
            "train_size": len(self.X_train),
            "test_size": len(self.X_test)
        }


# ============================================================
# Feature Engineering Engine
# ============================================================

class FeatureEngineeringEngine:

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(self, df: pd.DataFrame, understanding_obj) -> FeatureObject:

        target = understanding_obj.target_column
        time_column = understanding_obj.time_column
        is_time_series = understanding_obj.is_time_series

        df_model = df.copy()

        # ----------------------------------------------------
        # 1️⃣ Drop identifier columns
        # ----------------------------------------------------
        # Already excluded in understanding stage

        # ----------------------------------------------------
        # 2️⃣ Handle skewed features (log transform)
        # ----------------------------------------------------
        for col in understanding_obj.skewed_features:
            if col in df_model.columns:
                df_model[col] = np.log1p(df_model[col])

        # ----------------------------------------------------
        # 3️⃣ One-hot encode categorical features
        # ----------------------------------------------------
        categorical_cols = understanding_obj.categorical_columns

        if categorical_cols:
            df_model = pd.get_dummies(
                df_model,
                columns=categorical_cols,
                drop_first=True
            )

        # ----------------------------------------------------
        # 4️⃣ Handle time & datetime columns
        # ----------------------------------------------------
        if time_column and time_column in df_model.columns:
            df_model = df_model.sort_values(by=time_column)

        # Remove ALL datetime columns (sklearn cannot handle datetime)
        datetime_cols = df_model.select_dtypes(include=["datetime64[ns]"]).columns
        if len(datetime_cols) > 0:
            df_model.drop(columns=datetime_cols, inplace=True)

        # ----------------------------------------------------
        # 5️⃣ Separate Features and Target
        # ----------------------------------------------------
        X = df_model.drop(columns=[target])
        y = df_model[target]

        # ----------------------------------------------------
        # 6️⃣ Remove multicollinearity (correlation > 0.9)
        # ----------------------------------------------------
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > 0.9)
        ]

        X.drop(columns=to_drop, inplace=True)

        # ----------------------------------------------------
        # 7️⃣ Scaling numeric features
        # ----------------------------------------------------
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=np.number).columns

        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # ----------------------------------------------------
        # 8️⃣ Train/Test Split
        # Time-series uses chronological split
        # ----------------------------------------------------
        if is_time_series:

            split_index = int(len(X) * (1 - self.test_size))

            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]

        else:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True
            )

        return FeatureObject(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=X.columns.tolist()
        )
