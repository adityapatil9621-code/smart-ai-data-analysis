import streamlit as st
import pandas as pd
from modules.data_cleaning import clean_data
from modules.data_cleaning import (
    clean_data,
    compute_data_quality_score,
    compute_drift_metrics
)


st.set_page_config(page_title="Smart AI Data Analyst", layout="wide")

st.title("📊 Smart AI-Based Data Analysis System")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    if st.button("🧹 Clean Data Intelligently"):
        cleaned_df, explanation, warnings, impact_report = clean_data(df)

        # Quality Score
        column_scores, dataset_score = compute_data_quality_score(
            impact_report, len(df)
        )

        # Drift Metrics
        drift_report = compute_drift_metrics(df, cleaned_df)

        st.subheader("✅ Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head())

        # -------------------------------
        # 🔽 DOWNLOAD FULL CLEANED DATA
        # -------------------------------
        csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Full Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        st.subheader("🧠 Cleaning Explanation")
        for step in explanation:
            st.write("•", step)

        if warnings:
            st.subheader("⚠️ Warnings (Analysis Awareness)")
            for warn in warnings:
                st.warning(warn)

        st.subheader("📊 Data Quality Score")
        st.write(f"**Overall Dataset Quality Score:** {dataset_score} / 100")

        quality_df = pd.DataFrame.from_dict(
            column_scores, orient="index", columns=["Quality Score"]
        )
        st.dataframe(quality_df)

        st.subheader("📉 Before vs After Drift Metrics")
        drift_df = pd.DataFrame.from_dict(drift_report, orient="index")
        st.dataframe(drift_df)


