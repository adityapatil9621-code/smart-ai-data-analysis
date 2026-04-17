import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================
# IDENTIFIER DETECTION (ENHANCED)
# ======================================================
def detect_identifiers(df: pd.DataFrame) -> List[str]:
    """
    Enhanced identifier detection with multiple criteria

    Args:
        df: Input dataframe

    Returns:
        List of column names that are identifiers
    """
    identifiers = []

    for col in df.columns:
        # Perfect uniqueness
        if df[col].nunique() == len(df):
            identifiers.append(col)
            continue

        # Name-based detection
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'number']):
            # Check if high uniqueness
            if df[col].nunique() / len(df) > 0.95:
                identifiers.append(col)

    return identifiers


# ======================================================
# ENHANCED DOMAIN DETECTION WITH ML
# ======================================================
def detect_domain(df: pd.DataFrame) -> Tuple[str, float, Dict]:
    """
    Enhanced domain detection with confidence and detailed scoring

    Args:
        df: Input dataframe

    Returns:
        Tuple of (domain, confidence, detailed_scores)
    """
    cols = " ".join(df.columns).lower()

    # Expanded keyword dictionary
    keywords = {
        "retail": ["sales", "price", "product", "quantity", "customer", "order", "discount", "revenue", "sku"],
        "weather": ["temperature", "rain", "humidity", "wind", "pressure", "cloud", "precipitation", "forecast"],
        "finance": ["stock", "profit", "revenue", "market", "investment", "portfolio", "balance", "equity", "asset"],
        "healthcare": ["patient", "diagnosis", "treatment", "hospital", "medical", "doctor", "prescription", "symptom"],
        "marketing": ["campaign", "conversion", "click", "impression", "ctr", "roi", "engagement", "lead"],
        "hr": ["employee", "salary", "department", "hire", "performance", "bonus", "attendance", "leave"],
        "logistics": ["shipment", "delivery", "warehouse", "inventory", "tracking", "carrier", "freight"],
        "education": ["student", "grade", "course", "teacher", "exam", "enrollment", "semester", "gpa"],
        "manufacturing": ["production", "quality", "defect", "machine", "factory", "assembly", "output", "yield"],
        "ecommerce": ["cart", "checkout", "payment", "shipping", "browse", "view", "add", "purchase"]
    }

    # Score each domain
    scores = {}
    for domain, keywords_list in keywords.items():
        score = 0
        for keyword in keywords_list:
            if keyword in cols:
                score += 1
        scores[domain] = score

    # Get top domain
    if max(scores.values()) == 0:
        return "general", 0.0, scores

    detected = max(scores, key=scores.get)

    # Calculate confidence
    total = sum(scores.values())
    confidence = (scores[detected] / total) * 100 if total > 0 else 0

    # Get runner-up for uncertainty estimation
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[1][1] > 0:
        runner_up_ratio = sorted_scores[1][1] / sorted_scores[0][1]
        # If runner-up is close, reduce confidence
        if runner_up_ratio > 0.7:
            confidence *= 0.7

    logger.info(f"Detected domain: {detected} with confidence {confidence:.2f}%")

    return detected, round(confidence, 2), scores


# ======================================================
# ENSEMBLE ANOMALY DETECTION
# ======================================================
def detect_anomalies(
        df: pd.DataFrame,
        contamination: float = 0.05,
        methods: List[str] = ['isolation_forest', 'lof', 'elliptic']
) -> Tuple[float, pd.Series, Dict]:
    """
    Enhanced ensemble anomaly detection with multiple methods

    Args:
        df: Input dataframe
        contamination: Expected proportion of outliers
        methods: List of methods to use

    Returns:
        Tuple of (anomaly_percentage, anomaly_scores, method_details)
    """
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        logger.warning("No numeric columns for anomaly detection")
        return 0.0, pd.Series([0] * len(df)), {}

    # Handle missing values
    numeric_filled = numeric.fillna(numeric.mean())

    predictions = []
    scores_dict = {}
    method_details = {}

    # Isolation Forest
    if 'isolation_forest' in methods:
        try:
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            iso_pred = iso_forest.fit_predict(numeric_filled)
            iso_scores = iso_forest.score_samples(numeric_filled)

            predictions.append(iso_pred)
            scores_dict['isolation_forest'] = iso_scores

            method_details['isolation_forest'] = {
                "anomalies_detected": int((iso_pred == -1).sum()),
                "percentage": round((iso_pred == -1).sum() / len(df) * 100, 2)
            }
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")

    # Local Outlier Factor
    if 'lof' in methods:
        try:
            lof = LocalOutlierFactor(contamination=contamination, novelty=False)
            lof_pred = lof.fit_predict(numeric_filled)
            lof_scores = lof.negative_outlier_factor_

            predictions.append(lof_pred)
            scores_dict['lof'] = lof_scores

            method_details['lof'] = {
                "anomalies_detected": int((lof_pred == -1).sum()),
                "percentage": round((lof_pred == -1).sum() / len(df) * 100, 2)
            }
        except Exception as e:
            logger.warning(f"LOF failed: {e}")

    # Elliptic Envelope
    if 'elliptic' in methods and len(numeric_filled) > numeric_filled.shape[1]:
        try:
            elliptic = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
            elliptic_pred = elliptic.fit_predict(numeric_filled)

            predictions.append(elliptic_pred)

            method_details['elliptic'] = {
                "anomalies_detected": int((elliptic_pred == -1).sum()),
                "percentage": round((elliptic_pred == -1).sum() / len(df) * 100, 2)
            }
        except Exception as e:
            logger.warning(f"Elliptic Envelope failed: {e}")

    if not predictions:
        logger.error("All anomaly detection methods failed")
        return 0.0, pd.Series([0] * len(df)), {}

    # Ensemble voting: majority vote
    predictions_array = np.array(predictions)
    ensemble_votes = predictions_array.sum(axis=0)

    # Classify as anomaly if majority of methods agree (more than half vote -1)
    threshold = -len(predictions) / 2
    final_anomalies = ensemble_votes < threshold

    anomaly_percentage = round(final_anomalies.sum() / len(df) * 100, 2)

    # Create anomaly scores (average of normalized scores)
    if scores_dict:
        # Normalize all scores to 0-1 range
        normalized_scores = []
        for method_name, scores in scores_dict.items():
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                norm_scores = (scores - min_score) / (max_score - min_score)
            else:
                norm_scores = np.zeros_like(scores)
            normalized_scores.append(norm_scores)

        avg_scores = np.mean(normalized_scores, axis=0)
        anomaly_scores = pd.Series(avg_scores, index=df.index)
    else:
        anomaly_scores = pd.Series([0] * len(df), index=df.index)

    method_details['ensemble'] = {
        "anomalies_detected": int(final_anomalies.sum()),
        "percentage": anomaly_percentage,
        "methods_used": len(predictions)
    }

    logger.info(f"Detected {anomaly_percentage}% anomalies using ensemble of {len(predictions)} methods")

    return anomaly_percentage, anomaly_scores, method_details


# ======================================================
# ENHANCED DECISION CONFIDENCE
# ======================================================
def decision_confidence(
        stability: float,
        anomaly: float,
        domain_conf: float,
        sample_size: int,
        missing_ratio: float = 0.0
) -> Tuple[float, str, Dict]:
    """
    Enhanced decision confidence with detailed breakdown

    Args:
        stability: Model stability score (0-100)
        anomaly: Anomaly pressure percentage (0-100)
        domain_conf: Domain detection confidence (0-100)
        sample_size: Number of samples
        missing_ratio: Ratio of missing data (0-1)

    Returns:
        Tuple of (confidence_score, confidence_level, breakdown)
    """
    # Base confidence calculation
    confidence = (
            stability * 0.40 +  # Most important: model quality
            (100 - anomaly) * 0.25 +  # Data quality
            domain_conf * 0.15  # Domain understanding
    )

    # Sample size adjustment
    if sample_size < 30:
        sample_penalty = 20
    elif sample_size < 100:
        sample_penalty = 10
    elif sample_size < 300:
        sample_penalty = 5
    else:
        sample_penalty = 0

    confidence -= sample_penalty

    # Missing data penalty
    if missing_ratio > 0.3:
        missing_penalty = 15
    elif missing_ratio > 0.15:
        missing_penalty = 10
    elif missing_ratio > 0.05:
        missing_penalty = 5
    else:
        missing_penalty = 0

    confidence -= missing_penalty

    # Clip to valid range
    confidence = np.clip(confidence, 0, 100)

    # Determine confidence level
    if confidence >= 80:
        level = "Very High"
    elif confidence >= 65:
        level = "High"
    elif confidence >= 50:
        level = "Medium"
    elif confidence >= 35:
        level = "Low"
    else:
        level = "Very Low"

    # Create breakdown
    breakdown = {
        "base_score": round(confidence + sample_penalty + missing_penalty, 2),
        "stability_contribution": round(stability * 0.40, 2),
        "anomaly_contribution": round((100 - anomaly) * 0.25, 2),
        "domain_contribution": round(domain_conf * 0.15, 2),
        "sample_penalty": sample_penalty,
        "missing_penalty": missing_penalty,
        "final_score": round(confidence, 2)
    }

    return round(confidence, 2), level, breakdown


# ======================================================
# DATA PROFILING
# ======================================================
def profile_data(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data profiling

    Args:
        df: Input dataframe

    Returns:
        Detailed profile dictionary
    """
    profile = {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "column_types": {
            "numeric": len(df.select_dtypes(include=np.number).columns),
            "categorical": len(df.select_dtypes(include='object').columns),
            "datetime": len(df.select_dtypes(include='datetime').columns),
            "boolean": len(df.select_dtypes(include='bool').columns)
        },
        "missing_data": {
            "total_missing": int(df.isna().sum().sum()),
            "percentage": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
            "columns_with_missing": df.columns[df.isna().any()].tolist()
        },
        "duplicates": {
            "count": int(df.duplicated().sum()),
            "percentage": round(df.duplicated().sum() / len(df) * 100, 2)
        },
        "memory_usage": {
            "total_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    }

    # Cardinality analysis
    cardinality = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality[col] = {
            "unique_values": int(unique_count),
            "cardinality_ratio": round(unique_count / len(df), 4)
        }

    profile["cardinality"] = cardinality

    return profile


# ======================================================
# CONTEXTUAL STRATEGY ENGINE (ENHANCED)
# ======================================================
def generate_suggestions(
        domain: str,
        analysis_output: Dict,
        data_profile: Dict = None
) -> List[Dict]:
    """
    Enhanced contextual suggestions with priority and rationale

    Args:
        domain: Detected domain
        analysis_output: Analysis results
        data_profile: Optional data profile

    Returns:
        List of suggestion dictionaries
    """
    suggestions = []
    blocks = analysis_output.get("visual_blocks", [])

    # Extract key metrics
    trend = None
    driver = None
    seasonality = 0
    forecast_accuracy = None

    for b in blocks:
        if b["type"] == "forecast":
            trend = b.get("data", {}).get("trend_direction") or b.get("trend_direction")
            forecast_data = b.get("data", {})
            if "mape" in forecast_data:
                forecast_accuracy = forecast_data["mape"]
        if b["type"] == "drivers":
            driver = b.get("data", {}).get("top_driver") or b.get("top_driver")

    seasonality = analysis_output.get("seasonality_score", 0)

    # Domain-specific suggestions
    if domain == "retail":
        if trend == "upward":
            suggestions.append({
                "priority": "High",
                "category": "Operations",
                "suggestion": "Demand expansion detected. Increase supply chain readiness and inventory levels.",
                "rationale": "Upward trend indicates growing demand that could lead to stockouts if not prepared.",
                "action_items": [
                    "Review inventory levels for top products",
                    "Negotiate with suppliers for bulk orders",
                    "Consider expanding warehouse capacity"
                ]
            })
        elif trend == "downward":
            suggestions.append({
                "priority": "High",
                "category": "Strategy",
                "suggestion": "Demand contraction detected. Optimize pricing strategy and consider promotions.",
                "rationale": "Downward trend suggests decreasing demand requiring intervention.",
                "action_items": [
                    "Analyze competitor pricing",
                    "Plan promotional campaigns",
                    "Review product mix and customer feedback"
                ]
            })

        if seasonality > 50:
            suggestions.append({
                "priority": "Medium",
                "category": "Planning",
                "suggestion": "Strong seasonal patterns detected. Implement seasonal inventory management.",
                "rationale": f"Seasonality score of {seasonality} indicates predictable periodic fluctuations.",
                "action_items": [
                    "Build seasonal forecast models",
                    "Adjust staffing for peak seasons",
                    "Plan seasonal marketing campaigns"
                ]
            })

    elif domain == "finance":
        if trend == "upward":
            suggestions.append({
                "priority": "Medium",
                "category": "Investment",
                "suggestion": "Positive financial trajectory. Consider strategic investments for growth.",
                "rationale": "Upward trend indicates financial health and capacity for expansion.",
                "action_items": [
                    "Identify growth opportunities",
                    "Review capital allocation strategy",
                    "Consider market expansion"
                ]
            })

        if driver:
            suggestions.append({
                "priority": "High",
                "category": "Risk Management",
                "suggestion": f"Monitor '{driver}' closely as it significantly affects financial performance.",
                "rationale": "Top driver identification helps focus risk management efforts.",
                "action_items": [
                    f"Set up alerts for {driver} changes",
                    "Perform sensitivity analysis",
                    "Develop contingency plans"
                ]
            })

    elif domain == "weather":
        suggestions.append({
            "priority": "Medium",
            "category": "Analysis",
            "suggestion": "Environmental data detected. Consider climate adaptation strategies.",
            "rationale": "Weather patterns can inform long-term planning and risk assessment.",
            "action_items": [
                "Analyze historical patterns",
                "Assess climate change impacts",
                "Plan for extreme weather events"
            ]
        })

    elif domain == "marketing":
        if driver:
            suggestions.append({
                "priority": "High",
                "category": "Campaign Optimization",
                "suggestion": f"Focus optimization efforts on '{driver}' to improve campaign performance.",
                "rationale": "Top driver represents the most influential factor in marketing success.",
                "action_items": [
                    f"A/B test variations of {driver}",
                    "Allocate more budget to top-performing channels",
                    "Analyze customer segments"
                ]
            })

    # Generic suggestions based on driver
    if driver and not any(s["category"] == "Optimization" for s in suggestions):
        suggestions.append({
            "priority": "High",
            "category": "Optimization",
            "suggestion": f"Strategic control over '{driver}' may improve target outcome.",
            "rationale": "Feature importance analysis identified this as the primary driver.",
            "action_items": [
                f"Conduct detailed analysis of {driver}",
                "Identify levers to control this variable",
                "Monitor changes and measure impact"
            ]
        })

    # Data quality suggestions
    if data_profile:
        missing_pct = data_profile.get("missing_data", {}).get("percentage", 0)
        if missing_pct > 10:
            suggestions.append({
                "priority": "Medium",
                "category": "Data Quality",
                "suggestion": f"High missing data rate ({missing_pct}%). Improve data collection processes.",
                "rationale": "Missing data reduces analysis reliability and model accuracy.",
                "action_items": [
                    "Review data collection procedures",
                    "Implement data validation at source",
                    "Consider data imputation strategies"
                ]
            })

    # Forecast quality suggestion
    if forecast_accuracy and forecast_accuracy > 15:
        suggestions.append({
            "priority": "Medium",
            "category": "Modeling",
            "suggestion": f"Forecast accuracy is moderate (MAPE: {forecast_accuracy}%). Consider advanced models.",
            "rationale": "Higher accuracy forecasts enable better planning and decision-making.",
            "action_items": [
                "Try Prophet or SARIMA for time series",
                "Include more relevant features",
                "Collect more historical data"
            ]
        })

    return suggestions


# ======================================================
# COMPREHENSIVE AWARENESS REPORT
# ======================================================
def generate_awareness_report(
        df: pd.DataFrame,
        analysis_output: Dict = None
) -> Dict:
    """
    Generate comprehensive data awareness report

    Args:
        df: Input dataframe
        analysis_output: Optional analysis results

    Returns:
        Complete awareness report
    """
    # Profile data
    profile = profile_data(df)

    # Detect identifiers
    identifiers = detect_identifiers(df)

    # Detect domain
    domain, domain_confidence, domain_scores = detect_domain(df)

    # Detect anomalies
    anomaly_pct, anomaly_scores, anomaly_details = detect_anomalies(df)

    # Calculate decision confidence
    stability = analysis_output.get("stability_index", 50) if analysis_output else 50
    missing_ratio = profile["missing_data"]["percentage"] / 100

    conf_score, conf_level, conf_breakdown = decision_confidence(
        stability=stability,
        anomaly=anomaly_pct,
        domain_conf=domain_confidence,
        sample_size=len(df),
        missing_ratio=missing_ratio
    )

    # Generate suggestions
    suggestions = generate_suggestions(domain, analysis_output or {}, profile)

    report = {
        "data_profile": profile,
        "identifiers": identifiers,
        "domain_detection": {
            "detected_domain": domain,
            "confidence": domain_confidence,
            "all_scores": domain_scores
        },
        "anomaly_detection": {
            "percentage": anomaly_pct,
            "details": anomaly_details,
            "anomaly_indices": anomaly_scores.nlargest(10).index.tolist()
        },
        "decision_confidence": {
            "score": conf_score,
            "level": conf_level,
            "breakdown": conf_breakdown
        },
        "suggestions": suggestions
    }

    return report