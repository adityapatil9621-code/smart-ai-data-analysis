import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================
# ENHANCED NARRATIVE TEMPLATES
# ======================================================

NARRATIVE_TEMPLATES = {
    "forecast": {
        "upward": {
            "strong": "The analysis reveals a robust upward trajectory with {growth}% projected growth. "
                      "This strong positive trend suggests {context} and presents significant opportunities for strategic expansion.",
            "moderate": "A moderate upward trend of {growth}% is detected. "
                        "While positive, this suggests {context} and warrants careful monitoring.",
            "weak": "A slight upward movement of {growth}% is observed. "
                    "This weak trend indicates {context} and may require intervention to strengthen."
        },
        "downward": {
            "strong": "The analysis indicates a significant downward trend of {growth}% decline. "
                      "This sharp decrease suggests {context} and requires immediate strategic attention.",
            "moderate": "A moderate downward trend of {growth}% is identified. "
                        "This decline suggests {context} and should be addressed proactively.",
            "weak": "A slight downward movement of {growth}% is noted. "
                    "While concerning, this minor decline suggests {context} and may self-correct."
        },
        "stable": "The forecast shows relative stability with minimal change ({growth}%). "
                  "This plateau suggests {context} and may indicate market maturity."
    },

    "relationship": {
        "strong_positive": "A strong positive correlation (r={corr}) exists between {var1} and {var2}. "
                           "This statistically significant relationship (p<0.05) indicates that increases in {var1} "
                           "are consistently associated with increases in {var2}. {implication}",
        "strong_negative": "A strong negative correlation (r={corr}) is identified between {var1} and {var2}. "
                           "This inverse relationship (p<0.05) reveals that as {var1} increases, "
                           "{var2} tends to decrease proportionally. {implication}",
        "moderate_positive": "A moderate positive relationship (r={corr}) connects {var1} and {var2}. "
                             "While statistically significant, other factors also influence this relationship. {implication}",
        "moderate_negative": "A moderate negative relationship (r={corr}) exists between {var1} and {var2}. "
                             "The inverse correlation is notable but not dominant. {implication}",
        "weak": "The relationship between {var1} and {var2} is weak (r={corr}), "
                "suggesting that other variables may be more influential in explaining variance."
    },

    "driver": {
        "high_performance": "The variable '{driver}' emerges as the dominant influence with exceptional "
                            "predictive power (R²={r2}). This high-performance indicator suggests {context}.",
        "good_performance": "'{driver}' is identified as the primary driver with good explanatory power (R²={r2}). "
                            "This indicates {context}.",
        "moderate_performance": "'{driver}' shows moderate influence (R²={r2}). "
                                "While significant, model performance suggests additional factors should be considered.",
        "low_performance": "'{driver}' is the top driver, but overall model performance is limited (R²={r2}). "
                           "This indicates either missing important variables or high inherent variability."
    },

    "seasonality": {
        "strong": "Pronounced seasonal patterns are detected (score: {score}) at cycles of {lags}. "
                  "This strong periodicity enables reliable seasonal forecasting and planning.",
        "moderate": "Moderate seasonal behavior is observed (score: {score}) with cycles at {lags}. "
                    "These patterns should be incorporated into forecasting models.",
        "weak": "Weak or no significant seasonality detected (score: {score}). "
                "The data appears primarily driven by trends and irregular components."
    },

    "anomaly": {
        "high": "Elevated anomaly pressure ({percentage}%) is detected in the dataset. "
                "This high level of outliers suggests potential data quality issues or rare events requiring investigation.",
        "medium": "Moderate anomaly presence ({percentage}%) is identified. "
                  "These outliers should be examined to determine if they represent errors or genuine exceptional cases.",
        "low": "Low anomaly incidence ({percentage}%) indicates good data consistency. "
               "The few outliers present appear to be natural variation rather than systematic issues."
    }
}

CONTEXTUAL_IMPLICATIONS = {
    "retail": {
        "upward": "increasing consumer demand and market expansion",
        "downward": "declining consumer interest or increased competition",
        "stable": "market equilibrium or saturation",
        "price_sales": "Price elasticity is significant - pricing strategy directly impacts sales volume",
        "marketing_sales": "Marketing effectiveness is proven - continued investment yields returns"
    },
    "finance": {
        "upward": "improving financial health and growth momentum",
        "downward": "deteriorating financial performance requiring corrective action",
        "stable": "financial equilibrium with balanced risk-reward profile",
        "general": "Strategic financial management can optimize outcomes"
    },
    "weather": {
        "upward": "climate pattern shifts or seasonal intensification",
        "downward": "moderating conditions or inverse seasonal effects",
        "stable": "consistent environmental conditions",
        "general": "Environmental monitoring and adaptation strategies are recommended"
    },
    "marketing": {
        "upward": "campaign effectiveness and audience engagement growth",
        "downward": "declining campaign performance or market saturation",
        "stable": "consistent market response and mature campaign lifecycle",
        "general": "Data-driven optimization can enhance campaign ROI"
    },
    "general": {
        "upward": "positive momentum in the measured system",
        "downward": "declining performance requiring strategic intervention",
        "stable": "equilibrium state in the system dynamics",
        "general": "Strategic decisions should be data-informed"
    }
}


# ======================================================
# NARRATIVE GENERATION ENGINE
# ======================================================
def build_semantic_narrative(
        analysis_output: Dict,
        awareness_output: Dict
) -> Tuple[List[str], List[str]]:
    """
    Enhanced narrative generation with context-aware templates

    Args:
        analysis_output: Analysis results
        awareness_output: Awareness/profiling results

    Returns:
        Tuple of (narratives, suggestions)
    """
    narratives = []
    suggestions = []

    # Extract context
    domain = awareness_output.get("domain_detection", {}).get("detected_domain", "general")
    anomaly = awareness_output.get("anomaly_detection", {}).get("percentage", 0)
    stability = analysis_output.get("stability_index", 0)
    sample_size = analysis_output.get("sample_size", 0)

    blocks = analysis_output.get("visual_blocks", [])

    # Get domain-specific context
    domain_context = CONTEXTUAL_IMPLICATIONS.get(domain, CONTEXTUAL_IMPLICATIONS["general"])

    # -----------------------------------
    # SAMPLE SIZE WARNING
    # -----------------------------------
    if sample_size > 0 and sample_size < 30:
        narratives.append(
            f"⚠️ Note: This analysis is based on {sample_size} samples. "
            f"For more reliable insights, consider collecting additional data (recommended: 100+ samples)."
        )

    # -----------------------------------
    # FORECAST INTERPRETATION
    # -----------------------------------
    forecast_block = next((b for b in blocks if b["type"] == "forecast"), None)

    if forecast_block:
        data = forecast_block.get("data", {})
        trend = data.get("trend_direction", "stable")
        growth = abs(data.get("growth_rate", 0))
        mape = data.get("mape", 0)
        seasonality_info = forecast_block.get("seasonality", {})

        # Determine trend strength
        if growth > 20:
            strength = "strong"
        elif growth > 10:
            strength = "moderate"
        elif growth > 2:
            strength = "weak"
        else:
            trend = "stable"
            strength = None

        # Get appropriate template
        if trend == "stable":
            template = NARRATIVE_TEMPLATES["forecast"]["stable"]
        else:
            template = NARRATIVE_TEMPLATES["forecast"][trend][strength]

        # Get context
        context = domain_context.get(trend, "system dynamics shifting")

        # Format narrative
        narrative = template.format(
            growth=round(growth, 1),
            context=context
        )

        # Add forecast quality assessment
        if mape <= 5:
            narrative += " The forecast demonstrates excellent accuracy (MAPE: {:.1f}%), providing high confidence for planning.".format(
                mape)
        elif mape <= 15:
            narrative += " The forecast shows good accuracy (MAPE: {:.1f}%), suitable for strategic planning.".format(
                mape)
        else:
            narrative += " Note: Forecast accuracy is moderate (MAPE: {:.1f}%). Consider this uncertainty in planning decisions.".format(
                mape)

        narratives.append(narrative)

        # Seasonality narrative
        seasonality_score = seasonality_info.get("score", 0)
        significant_lags = seasonality_info.get("significant_lags", [])

        if seasonality_score > 60:
            strength_key = "strong"
        elif seasonality_score > 30:
            strength_key = "moderate"
        else:
            strength_key = "weak"

        if significant_lags:
            lag_str = ", ".join(map(str, significant_lags))
        else:
            lag_str = "none identified"

        seasonality_narrative = NARRATIVE_TEMPLATES["seasonality"][strength_key].format(
            score=seasonality_score,
            lags=lag_str
        )
        narratives.append(seasonality_narrative)

    # -----------------------------------
    # RELATIONSHIP INTERPRETATION
    # -----------------------------------
    relationship_block = next((b for b in blocks if b["type"] == "relationship"), None)

    if relationship_block:
        data = relationship_block.get("data", {})
        if data:
            corr = data.get("correlation", 0)
            var1 = data.get("col1", "Variable 1")
            var2 = data.get("col2", "Variable 2")
            significant = data.get("significant", False)

            # Determine strength and direction
            abs_corr = abs(corr)
            if abs_corr > 0.7:
                strength = "strong"
            elif abs_corr > 0.4:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if corr > 0 else "negative"

            # Get template
            if strength == "weak":
                template = NARRATIVE_TEMPLATES["relationship"]["weak"]
            else:
                template_key = f"{strength}_{direction}"
                template = NARRATIVE_TEMPLATES["relationship"][template_key]

            # Get domain-specific implication
            var_lower = f"{var1}_{var2}".lower()
            if "price" in var_lower and "sales" in var_lower:
                implication = domain_context.get("price_sales", "This relationship has strategic implications.")
            elif "marketing" in var_lower and "sales" in var_lower:
                implication = domain_context.get("marketing_sales",
                                                 "This relationship validates investment effectiveness.")
            else:
                implication = "Understanding this relationship enables targeted optimization strategies."

            if not significant:
                implication = "However, this relationship is not statistically significant (p>0.05), suggesting it may be due to chance."

            narrative = template.format(
                corr=round(corr, 3),
                var1=var1,
                var2=var2,
                implication=implication
            )
            narratives.append(narrative)

    # -----------------------------------
    # DRIVER INTERPRETATION
    # -----------------------------------
    driver_block = next((b for b in blocks if b["type"] == "drivers"), None)

    if driver_block:
        data = driver_block.get("data", {})
        driver = data.get("top_driver", "Unknown")
        r2_test = data.get("r2_test", 0)
        r2_cv = data.get("r2_cv_mean", 0)
        overfitting_gap = data.get("overfitting_gap", 0)

        # Check if driver is an identifier (should be excluded)
        if "id" in driver.lower():
            narratives.append(
                f"⚠️ Warning: The detected top driver '{driver}' appears to be an identifier or index. "
                f"This suggests potential data leakage or improper feature selection. "
                f"Identifiers should be excluded from predictive modeling."
            )
        else:
            # Determine performance level
            if r2_test > 0.8:
                perf_key = "high_performance"
            elif r2_test > 0.6:
                perf_key = "good_performance"
            elif r2_test > 0.4:
                perf_key = "moderate_performance"
            else:
                perf_key = "low_performance"

            template = NARRATIVE_TEMPLATES["driver"][perf_key]

            # Get context
            context = domain_context.get("general", "this variable has measurable impact")

            narrative = template.format(
                driver=driver,
                r2=round(r2_test, 3),
                context=context
            )

            # Add cross-validation note
            narrative += f" Cross-validation confirms robustness (CV R²: {round(r2_cv, 3)})."

            # Overfitting warning
            if overfitting_gap > 0.15:
                narrative += f" ⚠️ However, a notable train-test gap ({round(overfitting_gap, 3)}) suggests some overfitting. Consider model regularization."

            narratives.append(narrative)

    # -----------------------------------
    # ANOMALY INTERPRETATION
    # -----------------------------------
    if anomaly > 0:
        if anomaly > 10:
            strength_key = "high"
        elif anomaly > 5:
            strength_key = "medium"
        else:
            strength_key = "low"

        anomaly_narrative = NARRATIVE_TEMPLATES["anomaly"][strength_key].format(
            percentage=round(anomaly, 1)
        )
        narratives.append(anomaly_narrative)

        if anomaly > 10 and forecast_block:
            narratives.append(
                "The elevated anomaly levels may reduce forecast reliability. "
                "Consider investigating and addressing outliers before making critical decisions."
            )

    # -----------------------------------
    # STABILITY ASSESSMENT
    # -----------------------------------
    if stability >= 75:
        narratives.append(
            f"✅ Overall analysis stability is strong ({round(stability, 1)}/100). "
            f"The models demonstrate consistent performance and the results are reliable for strategic decision-making."
        )
    elif stability >= 50:
        narratives.append(
            f"⚠️ Analysis stability is moderate ({round(stability, 1)}/100). "
            f"Results provide useful directional insights but should be validated with domain expertise and additional data."
        )
    else:
        narratives.append(
            f"❌ Analysis stability is low ({round(stability, 1)}/100). "
            f"Results should be interpreted with significant caution. Consider improving data quality, "
            f"collecting more samples, or revisiting feature selection."
        )

    # -----------------------------------
    # GENERATE SUGGESTIONS
    # -----------------------------------
    # Use suggestions from awareness module if available
    if "suggestions" in awareness_output:
        awareness_suggestions = awareness_output["suggestions"]
        # Convert to simple list format
        for sug in awareness_suggestions[:5]:  # Top 5 suggestions
            suggestion_text = f"[{sug.get('priority', 'Medium')}] {sug.get('suggestion', '')}"
            suggestions.append(suggestion_text)

    return narratives, suggestions


# ======================================================
# EXECUTIVE SUMMARY GENERATION
# ======================================================
def generate_executive_summary(
        analysis_output: Dict,
        awareness_output: Dict
) -> Dict:
    """
    Generate concise executive summary

    Args:
        analysis_output: Analysis results
        awareness_output: Awareness results

    Returns:
        Executive summary dictionary
    """
    summary = {
        "headline": "",
        "key_findings": [],
        "risk_factors": [],
        "recommendations": [],
        "confidence_assessment": ""
    }

    # Extract key metrics
    domain = awareness_output.get("domain_detection", {}).get("detected_domain", "general")
    stability = analysis_output.get("stability_index", 0)
    sample_size = analysis_output.get("sample_size", 0)

    decision_conf = awareness_output.get("decision_confidence", {})
    conf_score = decision_conf.get("score", 0)
    conf_level = decision_conf.get("level", "Unknown")

    blocks = analysis_output.get("visual_blocks", [])

    # Generate headline
    forecast_block = next((b for b in blocks if b["type"] == "forecast"), None)
    if forecast_block:
        data = forecast_block.get("data", {})
        trend = data.get("trend_direction", "stable")
        growth = abs(data.get("growth_rate", 0))

        if trend == "upward":
            summary[
                "headline"] = f"{domain.capitalize()} Analysis: Positive Growth Trajectory Detected ({growth:.1f}% projected)"
        elif trend == "downward":
            summary[
                "headline"] = f"{domain.capitalize()} Analysis: Declining Trend Requires Attention ({growth:.1f}% decline)"
        else:
            summary[
                "headline"] = f"{domain.capitalize()} Analysis: Stable Performance with Strategic Optimization Opportunities"
    else:
        summary["headline"] = f"{domain.capitalize()} Data Analysis: Comprehensive Insights Generated"

    # Key findings
    driver_block = next((b for b in blocks if b["type"] == "drivers"), None)
    if driver_block:
        data = driver_block.get("data", {})
        driver = data.get("top_driver", "")
        r2 = data.get("r2_test", 0)

        summary["key_findings"].append(
            f"Primary driver identified: '{driver}' (R²={r2:.2f})"
        )

    if forecast_block:
        data = forecast_block.get("data", {})
        mape = data.get("mape", 0)
        summary["key_findings"].append(
            f"Forecast accuracy: MAPE={mape:.1f}%"
        )

    # Risk factors
    if sample_size < 100:
        summary["risk_factors"].append(
            f"Limited sample size ({sample_size} records) may affect reliability"
        )

    anomaly_pct = awareness_output.get("anomaly_detection", {}).get("percentage", 0)
    if anomaly_pct > 10:
        summary["risk_factors"].append(
            f"Elevated anomaly rate ({anomaly_pct:.1f}%) requires investigation"
        )

    if stability < 50:
        summary["risk_factors"].append(
            "Low model stability suggests high variability or missing important features"
        )

    # Recommendations (top 3)
    if "suggestions" in awareness_output:
        for sug in awareness_output["suggestions"][:3]:
            summary["recommendations"].append(sug.get("suggestion", ""))

    # Confidence assessment
    summary["confidence_assessment"] = (
        f"Decision Confidence: {conf_level} ({conf_score}/100). "
        f"Analysis Stability: {stability:.1f}/100."
    )

    return summary


# ======================================================
# INSIGHT RANKING
# ======================================================
def rank_insights(
        narratives: List[str],
        analysis_output: Dict
) -> List[Dict]:
    """
    Rank insights by importance and actionability

    Args:
        narratives: List of narrative strings
        analysis_output: Analysis results

    Returns:
        Ranked list of insight dictionaries
    """
    ranked = []

    for i, narrative in enumerate(narratives):
        # Determine importance based on keywords
        importance = "Medium"
        actionability = "Medium"

        narrative_lower = narrative.lower()

        # High importance indicators
        if any(word in narrative_lower for word in ["strong", "significant", "critical", "warning", "high"]):
            importance = "High"
        elif any(word in narrative_lower for word in ["weak", "low", "minor", "slight"]):
            importance = "Low"

        # High actionability indicators
        if any(word in narrative_lower for word in ["optimize", "improve", "consider", "strategic", "intervention"]):
            actionability = "High"
        elif any(word in narrative_lower for word in ["observe", "note", "suggests"]):
            actionability = "Low"

        ranked.append({
            "insight": narrative,
            "importance": importance,
            "actionability": actionability,
            "order": i
        })

    # Sort by importance then actionability
    priority_order = {"High": 3, "Medium": 2, "Low": 1}
    ranked.sort(
        key=lambda x: (priority_order[x["importance"]], priority_order[x["actionability"]]),
        reverse=True
    )

    return ranked