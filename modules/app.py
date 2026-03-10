"""
app.py

Streamlit Interface for Smart AI Data Intelligence System.
"""

import streamlit as st
import pandas as pd

from core_engine import SmartAIEngine
from auth_db import create_tables
from suggestion_engine import SuggestionEngine
from chat_engine import ChatEngine


# NEW IMPORTS
from auth_db import create_tables
from auth_ui import authentication_ui
create_tables()

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Smart AI Data Intelligence",
    layout="wide"
)


# ============================================================
# Upload Section
# ============================================================
user_id = authentication_ui()

if user_id is None:
    st.stop()
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV format)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    if st.button("Run Analysis"):

        try:
            engine = SmartAIEngine()
            memory = engine.run_pipeline(df)

            suggestion_engine = SuggestionEngine()
            strategic_obj = suggestion_engine.run(
                insight_obj=memory["insight_intelligence"],
                forecast_obj=memory["forecast_intelligence"],
                score_obj=memory["intelligence_score"]
            )

            st.success("Analysis Completed Successfully")

            # =================================================
            # Cleaning Summary
            # =================================================
            st.header("🔹 Data Cleaning Summary")

            st.write(f"Rows: {memory['metadata']['rows']}")
            st.write(f"Columns: {memory['metadata']['columns']}")
            st.write(f"Quality Score: {memory['metadata']['quality_score']}")

            # =================================================
            # Visual Intelligence
            # =================================================
            st.header("🔹 Visual Intelligence")

            visual_info = memory["visual_intelligence"]

            st.write(f"Primary Chart: {visual_info['primary_chart']}")
            st.write(f"Reason: {visual_info['selection_reason']}")
            st.write(f"Confidence: {visual_info['visual_confidence']}")

            # Access figure objects from engine
            visual_engine = engine.visual_engine
            visual_obj = visual_engine.run(df, engine.understanding_engine.run(df))

            for fig in visual_obj.figures.values():
                st.pyplot(fig)

            # =================================================
            # Key Analysis
            # =================================================
            st.header("🔹 Key Analysis")

            model_info = memory["model_intelligence"]
            insight = memory["insight_intelligence"]

            st.write("Selected Model:", model_info["selected_model"])
            st.write("Model Confidence:", model_info["confidence"])
            st.write("Stability Score:", model_info["stability"])

            st.subheader("Top Positive Drivers")
            st.write(insight["top_positive_drivers"])

            st.subheader("Top Negative Drivers")
            st.write(insight["top_negative_drivers"])

            st.subheader("Risk Score")
            st.write(insight["risk_score"])
            model_obj = engine.model_obj

            if model_obj.selected_model == "Linear Regression" and model_obj.regression_details:
                details = model_obj.regression_details

                st.subheader("📈 Linear Regression Fit")

                # R2 Score
                st.metric("R² Score", details["r2_score"])

                # -----------------------
                # Actual vs Predicted
                # -----------------------
                import matplotlib.pyplot as plt

                fig1, ax1 = plt.subplots()

                ax1.scatter(details["y_test"], details["y_pred"], alpha=0.5)
                ax1.set_xlabel("Actual")
                ax1.set_ylabel("Predicted")
                ax1.set_title("Actual vs Predicted")

                st.pyplot(fig1)

                # -----------------------
                # Residual Plot
                # -----------------------
                residuals = [
                    actual - pred
                    for actual, pred in zip(details["y_test"], details["y_pred"])
                ]

                fig2, ax2 = plt.subplots()
                ax2.scatter(details["y_pred"], residuals, alpha=0.5)
                ax2.axhline(0, linestyle="--")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Residual")
                ax2.set_title("Residual Analysis")

                st.pyplot(fig2)

                # -----------------------
                # Top Coefficients
                # -----------------------
                st.subheader("📊 Top Feature Coefficients")

                coef_df = (
                    pd.DataFrame(details["coefficients"].items(),
                                 columns=["Feature", "Coefficient"])
                    .sort_values(by="Coefficient", key=abs, ascending=False)
                    .head(5)
                )

                st.dataframe(coef_df)

            # =================================================
            # Forecast Section
            # =================================================
            if memory["forecast_intelligence"]:

                st.header("🔹 Forecast")

                forecast = memory["forecast_intelligence"]

                st.write("Trend Direction:", forecast["trend_direction"])
                st.write("Volatility Score:", forecast["volatility_score"])
                st.write("Forecast Confidence:", forecast["forecast_confidence"])

                st.line_chart(forecast["forecast_values"])

            # =================================================
            # Strategic Recommendations
            # =================================================
            st.header("🔹 Strategic Recommendations")

            st.write("Growth Opportunities:")
            for g in strategic_obj.growth_opportunities:
                st.write("-", g)

            st.write("Risk Mitigation:")
            for r in strategic_obj.risk_mitigation_actions:
                st.write("-", r)

            st.write("Stability Recommendations:")
            for s in strategic_obj.stability_recommendations:
                st.write("-", s)

            st.write("Confidence Advisory:")
            st.write(strategic_obj.confidence_advisory)

            st.write("Human Oversight:")
            st.write(strategic_obj.human_oversight_note)

            # =================================================
            # Chat Assistant
            # =================================================
            st.header("💬 Ask Questions About Your Data")

            chat_engine = ChatEngine()

            user_query = st.text_input("Enter your question")

            if user_query:
                response = chat_engine.respond(user_query, memory)
                st.write("AI Response:")
                st.write(response)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
