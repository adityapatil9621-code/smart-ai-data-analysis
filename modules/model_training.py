"""
model_training.py

Model Training & Selection Engine
- Uses engineered features
- Performs cross-validation
- Selects best model (mean - std)
- Computes confidence & stability
- Stores trained model
- Extracts linear regression diagnostics
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


# ============================================================
# Model Object
# ============================================================

@dataclass
class ModelObject:
    selected_model: str
    confidence: float
    stability: float
    trained_model: object
    feature_names: list
    regression_details: Optional[Dict] = None

    def to_dict(self):
        return {
            "selected_model": self.selected_model,
            "confidence": self.confidence,
            "stability": self.stability
        }


# ============================================================
# Model Training Engine
# ============================================================

class ModelTrainingEngine:

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def run(self, feature_obj):

        # Use pre-split data from FeatureObject
        X_train = feature_obj.X_train
        X_test = feature_obj.X_test
        y_train = feature_obj.y_train
        y_test = feature_obj.y_test
        feature_names = feature_obj.feature_names

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            )
        }

        results = {}

        # Cross-validation on training data only
        for name, model in models.items():
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="r2"
            )

            results[name] = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores)
            }

        # Select best model
        selected_model_name = max(
            results,
            key=lambda x: results[x]["mean_score"] - results[x]["std_score"]
        )

        best_model = models[selected_model_name]
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        confidence = round(results[selected_model_name]["mean_score"], 4)
        stability = round(1 - results[selected_model_name]["std_score"], 4)

        regression_details = None

        if selected_model_name == "Linear Regression":
            coefficients = dict(
                zip(feature_names, best_model.coef_)
            )

            regression_details = {
                "r2_score": round(r2, 4),
                "coefficients": coefficients,
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist()
            }

        return ModelObject(
            selected_model=selected_model_name,
            confidence=confidence,
            stability=stability,
            trained_model=best_model,
            feature_names=feature_names,
            regression_details=regression_details
        )

