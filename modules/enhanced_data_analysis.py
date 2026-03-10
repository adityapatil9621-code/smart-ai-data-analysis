import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats
from scipy.stats import pearsonr
import warnings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================
# ENHANCED SEASONALITY DETECTION
# ======================================================
def detect_seasonality(series, max_lag=24):
    """
    Advanced seasonality detection using multiple lags and statistical tests
    """
    if len(series) < max_lag:
        return {"score": 0, "significant_lags": [], "confidence": "low"}

    significant_lags = []
    autocorr_values = []

    # Check multiple lags for seasonality
    for lag in [7, 12, 24, 30, 365]:  # weekly, monthly, quarterly, yearly
        if lag < len(series):
            autocorr = series.autocorr(lag=lag)
            if abs(autocorr) > 0.3:  # Threshold for significance
                significant_lags.append(lag)
                autocorr_values.append(abs(autocorr))

    if autocorr_values:
        avg_score = np.mean(autocorr_values) * 100
        confidence = "high" if avg_score > 70 else "medium" if avg_score > 40 else "low"
    else:
        avg_score = 0
        confidence = "low"

    return {
        "score": round(avg_score, 2),
        "significant_lags": significant_lags,
        "confidence": confidence
    }


# ======================================================
# ADVANCED FORECASTING WITH MULTIPLE METHODS
# ======================================================
def advanced_forecast(series, periods=20):
    """
    Enhanced forecasting with confidence intervals and multiple methods
    """
    if len(series) < 10:
        logger.warning("Series too short for reliable forecasting")
        return None

    try:
        # Method 1: Linear Regression with confidence intervals
        X_time = np.arange(len(series)).reshape(-1, 1)
        model_lr = LinearRegression()
        model_lr.fit(X_time, series)

        # Predictions
        X_future = np.arange(len(series) + periods).reshape(-1, 1)
        predictions_lr = model_lr.predict(X_future)

        # Calculate confidence intervals (95%)
        residuals = series - model_lr.predict(X_time)
        residual_std = np.std(residuals)
        confidence_interval = 1.96 * residual_std

        # Method 2: Exponential Smoothing
        alpha = 0.3  # Smoothing factor
        smoothed = [series.iloc[0]]
        for i in range(1, len(series)):
            smoothed.append(alpha * series.iloc[i] + (1 - alpha) * smoothed[-1])

        # Simple extrapolation for forecast
        last_value = smoothed[-1]
        trend = (smoothed[-1] - smoothed[-10]) / 10 if len(smoothed) >= 10 else 0

        forecast_es = [last_value + trend * i for i in range(1, periods + 1)]

        # Combine methods (average)
        forecast_combined = []
        forecast_lower = []
        forecast_upper = []

        for i in range(len(series), len(series) + periods):
            pred_lr = predictions_lr[i]
            pred_es = forecast_es[i - len(series)]

            combined = (pred_lr + pred_es) / 2
            forecast_combined.append(combined)
            forecast_lower.append(combined - confidence_interval)
            forecast_upper.append(combined + confidence_interval)

        # Calculate forecast quality metrics
        mape = np.mean(np.abs(residuals / (series + 1e-10))) * 100
        rmse = np.sqrt(mean_squared_error(series, model_lr.predict(X_time)))

        return {
            "historical": series.tolist(),
            "forecast": forecast_combined,
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "confidence_level": 95
        }

    except Exception as e:
        logger.error(f"Forecasting failed: {str(e)}")
        return None


# ======================================================
# ENHANCED FEATURE IMPORTANCE WITH STATISTICAL TESTS
# ======================================================
def analyze_drivers(X, y, feature_names):
    """
    Advanced driver analysis with multiple importance measures
    """
    if X.shape[0] < 30:
        logger.warning("Sample size too small for reliable driver analysis")

    # Train/test split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Random Forest for feature importance
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predictions and metrics
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation for robustness
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

    # Feature importance
    importance_rf = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    # Calculate correlation-based importance
    importance_corr = {}
    for col in feature_names:
        corr, p_value = pearsonr(X[col], y)
        importance_corr[col] = abs(corr)

    importance_corr = pd.Series(importance_corr).sort_values(ascending=False)

    # Combine both measures (weighted average)
    combined_importance = (0.7 * importance_rf + 0.3 * importance_corr).sort_values(ascending=False)

    return {
        "importance": combined_importance,
        "top_driver": combined_importance.index[0],
        "r2_train": round(r2_train, 3),
        "r2_test": round(r2_test, 3),
        "r2_cv_mean": round(cv_scores.mean(), 3),
        "r2_cv_std": round(cv_scores.std(), 3),
        "mae_test": round(mae_test, 3),
        "overfitting_gap": round(r2_train - r2_test, 3)
    }


# ======================================================
# ENHANCED CORRELATION ANALYSIS WITH SIGNIFICANCE
# ======================================================
def analyze_relationships(numeric_df):
    """
    Correlation analysis with statistical significance testing
    """
    corr_matrix = numeric_df.corr()

    # Find strongest correlation (excluding diagonal)
    corr_abs = corr_matrix.abs()
    np.fill_diagonal(corr_abs.values, 0)

    if corr_abs.empty or corr_abs.max().max() == 0:
        return None

    max_corr = corr_abs.max().max()
    idx = np.where(corr_abs == max_corr)

    col1 = corr_matrix.columns[idx[0][0]]
    col2 = corr_matrix.columns[idx[1][0]]

    # Calculate p-value for significance
    corr_value, p_value = pearsonr(numeric_df[col1], numeric_df[col2])

    # Determine strength and significance
    strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.4 else "weak"
    direction = "positive" if corr_value > 0 else "negative"
    significant = p_value < 0.05

    return {
        "col1": col1,
        "col2": col2,
        "correlation": round(corr_value, 3),
        "p_value": round(p_value, 4),
        "significant": significant,
        "strength": strength,
        "direction": direction,
        "interpretation": f"{strength.capitalize()} {direction} correlation"
    }


# ======================================================
# MAIN ANALYSIS ENGINE (ENHANCED)
# ======================================================
def run_analysis(df):
    """
    Enhanced analysis with proper validation and error handling
    """
    try:
        # Validation
        if df.empty:
            raise ValueError("Empty dataframe provided")

        if len(df) < 10:
            raise ValueError("Insufficient data: Need at least 10 rows for analysis")

        numeric = df.select_dtypes(include=np.number)

        if numeric.empty:
            raise ValueError("No numeric columns found for analysis")

        if len(df) < 30:
            warnings.warn("Small sample size (<30 rows). Results may be unreliable.")

        visual_blocks = []

        # =====================================
        # 1. RELATIONSHIP ANALYSIS
        # =====================================
        relationship_result = analyze_relationships(numeric)

        if relationship_result:
            fig_rel = px.scatter(
                numeric,
                x=relationship_result["col1"],
                y=relationship_result["col2"],
                title=f"Strongest Relationship: {relationship_result['col1']} vs {relationship_result['col2']}<br>" +
                      f"<sub>Correlation: {relationship_result['correlation']}, " +
                      f"p-value: {relationship_result['p_value']}, " +
                      f"{relationship_result['interpretation']}</sub>",
                trendline="ols"
            )

            visual_blocks.append({
                "type": "relationship",
                "chart": fig_rel,
                "data": relationship_result,
                "score": min(abs(relationship_result["correlation"]) * 100, 100)
            })

        # =====================================
        # 2. DRIVER ANALYSIS
        # =====================================
        if len(numeric.columns) >= 2:
            target = numeric.columns[-1]
            X = numeric.drop(columns=[target])
            y = numeric[target]

            driver_result = analyze_drivers(X, y, X.columns)

            fig_imp = go.Figure()
            top_10_features = driver_result["importance"].head(10)

            fig_imp.add_trace(go.Bar(
                x=top_10_features.values,
                y=top_10_features.index,
                orientation='h',
                marker=dict(
                    color=top_10_features.values,
                    colorscale='Viridis'
                )
            ))

            fig_imp.update_layout(
                title=f"Top Drivers of {target}<br>" +
                      f"<sub>Test R²: {driver_result['r2_test']} | " +
                      f"CV R²: {driver_result['r2_cv_mean']} ± {driver_result['r2_cv_std']}</sub>",
                xaxis_title="Importance Score",
                yaxis_title="Features"
            )

            visual_blocks.append({
                "type": "drivers",
                "chart": fig_imp,
                "data": driver_result,
                "score": max(driver_result["r2_test"] * 100, 0)
            })

        # =====================================
        # 3. TIME SERIES ANALYSIS & FORECAST
        # =====================================
        series = numeric[numeric.columns[-1]]

        # Detect seasonality
        seasonality_result = detect_seasonality(series)

        # Generate forecast
        forecast_result = advanced_forecast(series, periods=20)

        if forecast_result:
            fig_forecast = make_subplots(
                rows=1, cols=1,
                subplot_titles=["Historical Data & Forecast with Confidence Intervals"]
            )

            # Historical data
            fig_forecast.add_trace(go.Scatter(
                y=forecast_result["historical"],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))

            # Forecast
            forecast_x = list(range(len(forecast_result["historical"]),
                                    len(forecast_result["historical"]) + len(forecast_result["forecast"])))

            fig_forecast.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_result["forecast"],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_result["forecast_upper"],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig_forecast.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_result["forecast_lower"],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))

            fig_forecast.update_layout(
                title=f"Forecast Analysis<br>" +
                      f"<sub>MAPE: {forecast_result['mape']}% | " +
                      f"RMSE: {forecast_result['rmse']}</sub>",
                xaxis_title="Time Period",
                yaxis_title="Value",
                hovermode='x unified'
            )

            # Calculate forecast quality score
            forecast_score = max(100 - forecast_result["mape"], 0)

            visual_blocks.append({
                "type": "forecast",
                "chart": fig_forecast,
                "data": forecast_result,
                "seasonality": seasonality_result,
                "score": forecast_score
            })

        # =====================================
        # 4. DISTRIBUTION ANALYSIS
        # =====================================
        fig_dist = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Distribution", "Box Plot"]
        )

        target_col = numeric.columns[-1]

        # Histogram
        fig_dist.add_trace(
            go.Histogram(x=numeric[target_col], name='Distribution', nbinsx=30),
            row=1, col=1
        )

        # Box plot
        fig_dist.add_trace(
            go.Box(y=numeric[target_col], name='Box Plot'),
            row=1, col=2
        )

        # Calculate skewness and kurtosis
        skewness = stats.skew(numeric[target_col].dropna())
        kurtosis = stats.kurtosis(numeric[target_col].dropna())

        fig_dist.update_layout(
            title=f"Distribution Analysis of {target_col}<br>" +
                  f"<sub>Skewness: {skewness:.2f} | Kurtosis: {kurtosis:.2f}</sub>",
            showlegend=False
        )

        visual_blocks.append({
            "type": "distribution",
            "chart": fig_dist,
            "data": {
                "skewness": round(skewness, 3),
                "kurtosis": round(kurtosis, 3),
                "mean": round(numeric[target_col].mean(), 3),
                "median": round(numeric[target_col].median(), 3),
                "std": round(numeric[target_col].std(), 3)
            },
            "score": 100  # Distribution always gets full score as it's descriptive
        })

        # =====================================
        # CALCULATE OVERALL STABILITY
        # =====================================
        scored_blocks = [b for b in visual_blocks if "score" in b]
        if scored_blocks:
            stability_index = round(np.mean([b["score"] for b in scored_blocks]), 2)
        else:
            stability_index = 0

        return {
            "visual_blocks": visual_blocks,
            "stability_index": stability_index,
            "sample_size": len(df),
            "num_features": len(numeric.columns),
            "warnings": []
        }

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {
            "error": str(e),
            "visual_blocks": [],
            "stability_index": 0
        }


# ======================================================
# ENHANCED NARRATIVE GENERATION
# ======================================================
def generate_insights(analysis_output):
    """
    Generate natural language insights from analysis
    """
    insights = []

    if "error" in analysis_output:
        return [f"Analysis failed: {analysis_output['error']}"]

    # Sample size warning
    if analysis_output["sample_size"] < 30:
        insights.append(
            f"⚠️ Note: Analysis based on {analysis_output['sample_size']} samples. "
            f"Results may be more reliable with larger datasets (30+ samples recommended)."
        )

    # Analyze each block
    for block in analysis_output["visual_blocks"]:
        if block["type"] == "relationship":
            data = block["data"]
            if data["significant"]:
                insights.append(
                    f"📊 Found a {data['strength']} {data['direction']} relationship between "
                    f"{data['col1']} and {data['col2']} (r={data['correlation']}, p<0.05). "
                    f"This relationship is statistically significant."
                )
            else:
                insights.append(
                    f"📊 The relationship between {data['col1']} and {data['col2']} "
                    f"is {data['strength']} but not statistically significant (p={data['p_value']})."
                )

        elif block["type"] == "drivers":
            data = block["data"]
            insights.append(
                f"🎯 Top driver: {data['top_driver']} is the most influential variable. "
                f"Model performance: R² = {data['r2_test']} on test data "
                f"(CV: {data['r2_cv_mean']} ± {data['r2_cv_std']})."
            )

            if data["overfitting_gap"] > 0.2:
                insights.append(
                    f"⚠️ Warning: Model may be overfitting (train-test gap: {data['overfitting_gap']}). "
                    f"Consider regularization or simpler models."
                )

        elif block["type"] == "forecast":
            data = block["data"]
            seasonality = block["seasonality"]

            insights.append(
                f"🔮 Forecast generated with {data['confidence_level']}% confidence intervals. "
                f"Expected accuracy: MAPE = {data['mape']}%."
            )

            if seasonality["score"] > 50:
                insights.append(
                    f"📅 Strong seasonality detected (score: {seasonality['score']}) "
                    f"at lags: {seasonality['significant_lags']}. Consider seasonal models."
                )

        elif block["type"] == "distribution":
            data = block["data"]

            if abs(data["skewness"]) > 1:
                skew_dir = "right" if data["skewness"] > 0 else "left"
                insights.append(
                    f"📈 Data is highly skewed to the {skew_dir} (skewness={data['skewness']}). "
                    f"Consider log transformation for modeling."
                )

            if data["mean"] != data["median"]:
                diff_pct = abs(data["mean"] - data["median"]) / data["mean"] * 100
                if diff_pct > 20:
                    insights.append(
                        f"📊 Mean ({data['mean']}) differs significantly from median ({data['median']}), "
                        f"suggesting outliers or skewness."
                    )

    # Overall stability
    stability = analysis_output["stability_index"]
    if stability > 70:
        insights.append(
            f"✅ Overall analysis stability is strong ({stability}/100). "
            f"Results are reliable for decision-making."
        )
    elif stability > 40:
        insights.append(
            f"⚠️ Overall analysis stability is moderate ({stability}/100). "
            f"Use results with caution and validate with domain knowledge."
        )
    else:
        insights.append(
            f"❌ Overall analysis stability is low ({stability}/100). "
            f"Consider improving data quality or collecting more samples."
        )

    return insights